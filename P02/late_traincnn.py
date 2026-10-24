"""Train and evaluate both late-fusion models defined in modelonj.py."""

import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets import DEFAULT_ROOT_DIR, FrameVideoDataset
from modelonj import LateFusion2DCNN, LateFusion3DCNN


NUM_CLASSES = 10
NUM_FRAMES = 10
NUM_SEGMENTS = 2
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ".")


def build_model(model_type):
    if model_type == "late_2d":
        return LateFusion2DCNN(
            num_classes=NUM_CLASSES,
            num_frames=NUM_FRAMES,
            num_segments=NUM_SEGMENTS,
        )
    if model_type == "late_3d":
        return LateFusion3DCNN(
            num_classes=NUM_CLASSES,
            num_segments=NUM_SEGMENTS,
        )
    raise ValueError(
        f"Unknown MODEL_TYPE {model_type!r}. Choose 'late_2d' or 'late_3d'."
    )


def augment_videos(videos):
    """Flip whole videos consistently so frame-to-frame motion is preserved."""
    batch_size = videos.shape[0]
    flip = torch.rand(batch_size, device=videos.device) < 0.5
    return torch.where(flip.view(batch_size, 1, 1, 1, 1), videos.flip(-1), videos)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device).long()
            if training:
                videos = augment_videos(videos)
                optimizer.zero_grad()

            logits = model(videos)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    if total == 0:
        raise ValueError("The data loader is empty; check the dataset root and split.")
    return loss_sum / total, correct / total


def train_and_test(model_type, train_loader, val_loader, test_loader, device):
    model = build_model(model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())

    print(f"\n=== Training {model_type} on {device} ===", flush=True)

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
            f"Train loss={train_loss:.4f}, Train acc={train_accuracy:.2%} | "
            f"Val loss={val_loss:.4f}, Val acc={val_accuracy:.2%}",
            flush=True,
        )

    # Restore the checkpoint selected using validation accuracy before the
    # single final evaluation on the held-out test split.
    model.load_state_dict(best_state)
    test_loss, test_accuracy = run_epoch(
        model, test_loader, criterion, device
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_best.pt")
    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": best_state,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy,
            "num_classes": NUM_CLASSES,
            "num_frames": NUM_FRAMES,
            "num_segments": NUM_SEGMENTS,
            "dataset_root": DEFAULT_ROOT_DIR,
        },
        checkpoint_path,
    )
    print(f"Best validation accuracy: {best_val_accuracy:.2%}", flush=True)
    print(f"Final test loss: {test_loss:.4f}", flush=True)
    print(f"Final test accuracy: {test_accuracy:.2%}", flush=True)
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    return best_val_accuracy, test_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    train_dataset = FrameVideoDataset(
        root_dir=DEFAULT_ROOT_DIR,
        split="train",
        transform=transform,
        stack_frames=True,
    )
    val_dataset = FrameVideoDataset(
        root_dir=DEFAULT_ROOT_DIR,
        split="val",
        transform=transform,
        stack_frames=True,
    )
    test_dataset = FrameVideoDataset(
        root_dir=DEFAULT_ROOT_DIR,
        split="test",
        transform=transform,
        stack_frames=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset root: {DEFAULT_ROOT_DIR}", flush=True)
    results = {}
    for model_type in ("late_2d", "late_3d"):
        results[model_type] = train_and_test(
            model_type, train_loader, val_loader, test_loader, device
        )

    print("\nFinal results", flush=True)
    for model_type, (val_accuracy, test_accuracy) in results.items():
        print(
            f"{model_type}: best val acc={val_accuracy:.2%}, "
            f"test acc={test_accuracy:.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
