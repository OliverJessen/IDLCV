from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T


DEFAULT_ROOT_DIR = os.environ.get(
    "UCF101_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ufc10"),
)


def _resolve_root_dir(root_dir):
    """Resolve the dataset root from an argument, environment, or local folder."""
    return os.path.abspath(os.path.expanduser(os.fspath(root_dir or DEFAULT_ROOT_DIR)))


def _load_metadata(root_dir, split):
    metadata_path = os.path.join(root_dir, "metadata", f"{split}.csv")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. Set UCF101_ROOT to the "
            "dataset directory containing videos/, frames/, and metadata/."
        )
    return pd.read_csv(metadata_path)


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir=None,
    split='train', 
    transform=None
):
        root_dir = _resolve_root_dir(root_dir)
        self.frame_paths = sorted(glob(os.path.join(root_dir, 'frames', split, '*', '*', '*.jpg')))
        self.df = _load_metadata(root_dir, split)
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir=None, 
    split = 'train', 
    transform = None,
    stack_frames = True
):
        root_dir = _resolve_root_dir(root_dir)
        self.video_paths = sorted(glob(os.path.join(root_dir, 'videos', split, '*', '*.avi')))
        self.df = _load_metadata(root_dir, split)
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = os.path.join(
            os.path.dirname(video_path).replace(
                f"{os.sep}videos{os.sep}", f"{os.sep}frames{os.sep}"
            ),
            os.path.splitext(os.path.basename(video_path))[0],
        )
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = DEFAULT_ROOT_DIR

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
            
