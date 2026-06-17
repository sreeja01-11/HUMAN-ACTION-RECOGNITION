import cv2
import torch
import numpy as np

def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def resize_frames(frames, size=(112, 112)):
    resized_frames = [cv2.resize(f, size) for f in frames]
    return np.array(resized_frames)

def normalize_and_tensor(frames_array):
    frames_array = frames_array.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frames_array)
    return tensor

def format_for_r3d(tensor):
    tensor = tensor.permute(3, 0, 1, 2)
    tensor = tensor.unsqueeze(0) 
    return tensor