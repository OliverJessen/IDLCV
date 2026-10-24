"""Late-fusion CNN models for video classification.

Inputs use the dataset convention (batch, channels, time, height, width).
Each temporal segment predicts class logits independently; segment logits
are averaged to produce the final prediction.
"""

import torch
import torch.nn as nn


class _Segment2DCNN(nn.Module):
    """2D CNN that classifies a short temporal segment as one input."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, segment):
        return self.classifier(self.convolutional(segment))


class LateFusion2DCNN(nn.Module):
    """Classify multi-frame segments with 2D CNNs, then fuse their scores.

    Frames within each segment are stacked as channels for that segment's
    2D CNN. Each segment has its own CNN branch; branch logits are averaged.
    This is segment-level late fusion, not per-frame prediction aggregation.
    """

    def __init__(self, num_classes=10, num_frames=10, num_segments=2):
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be at least 1.")
        if num_frames < num_segments or num_frames % num_segments != 0:
            raise ValueError("num_frames must be divisible by num_segments.")
        self.num_frames = num_frames
        self.num_segments = num_segments
        frames_per_segment = num_frames // num_segments
        self.segment_models = nn.ModuleList(
            [
                _Segment2DCNN(3 * frames_per_segment, num_classes)
                for _ in range(num_segments)
            ]
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Expected input shaped (B, C, T, H, W).")
        batch_size, channels, num_frames, height, width = x.shape
        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels, got {channels}.")
        if num_frames != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, got {num_frames}."
            )

        segments = torch.chunk(x, self.num_segments, dim=2)
        segment_logits = []
        for segment, model in zip(segments, self.segment_models):
            # Fuse frames within this segment at the input channels.
            segment = segment.reshape(
                batch_size, channels * segment.shape[2], height, width
            )
            segment_logits.append(model(segment))
        return torch.stack(segment_logits, dim=1).mean(dim=1)


class _Clip3DCNN(nn.Module):
    """3D CNN stream for one temporal segment of a video."""

    def __init__(self, num_classes):
        super().__init__()
        self.convolutional = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.1),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout3d(0.1),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, clip):
        return self.classifier(self.convolutional(clip))


class LateFusion3DCNN(nn.Module):
    """Classify temporal segments with 3D CNN streams, then fuse their logits.

    Streams share CNN weights. With the default ten-frame input and two
    segments, each stream processes five consecutive frames.
    """

    def __init__(self, num_classes=10, num_segments=2):
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be at least 1.")
        self.num_segments = num_segments
        self.segment_model = _Clip3DCNN(num_classes)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Expected input shaped (B, C, T, H, W).")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {x.shape[1]}.")
        if x.shape[2] < 2 * self.num_segments:
            raise ValueError(
                "Each 3D stream needs at least two frames; provide at least "
                "2 * num_segments frames."
            )

        # Split into consecutive, nearly equal temporal chunks; fuse after
        # every 3D stream has produced its own class scores.
        segments = torch.tensor_split(x, self.num_segments, dim=2)
        segment_logits = torch.stack(
            [self.segment_model(segment) for segment in segments], dim=1
        )
        return segment_logits.mean(dim=1)
