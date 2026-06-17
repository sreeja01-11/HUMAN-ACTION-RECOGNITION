import torch
from torch.utils.data import Dataset
from preprocess import sample_frames, resize_frames, normalize_and_tensor, format_for_r3d

class UCF101Dataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        frames = sample_frames(path)
        resized = resize_frames(frames)
        tensor = normalize_and_tensor(resized)
        final_tensor = format_for_r3d(tensor)
        final_tensor = final_tensor.squeeze(0)
        return final_tensor, label