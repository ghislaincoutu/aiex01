# TensorFlow -- Définition du jeu de données de la banque de vidéos UCF50

import os
import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image
os.system("clear")

def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")

print("TensorFlow -- Définition du jeu de données de la banque de vidéos UCF50")
pause()

def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")

def load_video_tensor(frame_dir, num_frames=30, img_size=(224, 224)):
    frame_paths = sorted(glob(os.path.join(frame_dir, '*.jpg')))[:num_frames]
    frames = []
    for frame_path in frame_paths:
        img = Image.open(frame_path).resize(img_size)
        frames.append(np.array(img))
    frames = np.stack(frames, axis=0)  # shape: (T, H, W, C)
    return frames.astype(np.float32) / 255.0


def build_ucf50_dataset(frames_root, num_frames=30):
    class_names = sorted(os.listdir(frames_root))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    data = []
    labels = []

    for class_name in class_names:
        class_path = os.path.join(frames_root, class_name)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            video_tensor = load_video_tensor(video_path, num_frames)
            if video_tensor.shape[0] == num_frames:
                data.append(video_tensor)
                labels.append(class_to_idx[class_name])

    data = np.stack(data)
    labels = np.array(labels)

    return tf.data.Dataset.from_tensor_slices((data, labels))

dataset = build_ucf50_dataset("/media/disk01/medias/UCF50_frames", num_frames=30)
dataset = dataset.shuffle(100).batch(8).prefetch(tf.data.AUTOTUNE)
