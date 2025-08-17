import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def extract_frames(video_path, max_frames=40, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()

    if len(frames) < max_frames:
        # Pad with last frame
        frames += [frames[-1]] * (max_frames - len(frames))
    
    return np.array(frames) / 255.0  # Normalize to [0, 1]


import os
from glob import glob

def load_ucf50_dataset(root_dir, max_frames=40):
    class_names = sorted(os.listdir(root_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    videos = []
    labels = []

    for class_name in class_names:
        video_files = glob(os.path.join(root_dir, class_name, '*.avi'))
        for vf in video_files:
            frames = extract_frames(vf, max_frames)
            videos.append(frames)
            labels.append(label_map[class_name])
    
    X = np.array(videos)
    y = to_categorical(labels, num_classes=len(class_names))
    return X, y, class_names

X, y, class_names = load_ucf50_dataset('/media/disk01/medias/UCF50_limited')
print("Data shape:", X.shape)
print("Labels shape:", y.shape)
