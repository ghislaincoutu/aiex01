# TensorFlow -- Conversion des vidéos UCF50 en séries d'images-clés de format JPEG

import os
import cv2
os.system("clear")

def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")

print("TensorFlow -- Conversion des vidéos UCF50 en images-clés de format JPEG")
pause()

ucf50_dir = "/media/disk01/medias/UCF50_limited"
output_dir = "/media/disk01/medias/UCF50_frames"
frame_interval = 5

def extract_frames(video_path, output_folder, every_n=5):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    os.makedirs(output_folder, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
        count += 1
    cap.release()

# Loop through each class folder and extract frames from each video
for class_name in os.listdir(ucf50_dir):
    class_path = os.path.join(ucf50_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for video_file in os.listdir(class_path):
        if not video_file.endswith('.avi'):
            continue
        video_path = os.path.join(class_path, video_file)
        # Output path: /output_dir/class_name/video_name_without_extension/
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(output_dir, class_name, video_name)
        print(f"Extracting frames from {video_path} to {output_folder}")
        extract_frames(video_path, output_folder, every_n=frame_interval)
