print("hello")
import cv2
import os

input_folder = "videos"
output_folder = "dataset/train"

frame_skip = 5 

# check videos folder
if not os.path.exists(input_folder):
    print("videos' folder nahi mila")
    exit()

print("Videos folder found:", os.listdir(input_folder))

for pose in os.listdir(input_folder):
    pose_path = os.path.join(input_folder, pose)

    if not os.path.isdir(pose_path):
        continue

    print("\n➡️ Processing pose:", pose)

    for quality in os.listdir(pose_path):
        quality_path = os.path.join(pose_path, quality)

        if not os.path.isdir(quality_path):
            continue

        print("   ➤ Quality:", quality)

        save_path = os.path.join(output_folder, pose, quality)
        os.makedirs(save_path, exist_ok=True)

        videos = os.listdir(quality_path)

        if len(videos) == 0:
            print("   ⚠️ No videos found in", quality_path)
            continue

        for video in videos:
            video_path = os.path.join(quality_path, video)

            print("      🎥 Processing video:", video)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("      ❌ Cannot open video:", video)
                continue

            count = 0
            saved = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % frame_skip == 0:
                    filename = f"{pose}_{quality}_{saved}.jpg"
                    cv2.imwrite(os.path.join(save_path, filename), frame)
                    saved += 1

                count += 1

            cap.release()

            print(f"  Saved {saved} frames")

print("\n Frame extraction COMPLETED!")