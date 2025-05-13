import cv2
import os

class VideoGenerator:
    def __init__(self, video_path, output_dir, image_size=224):
        self.video_path = video_path
        self.output_dir = output_dir
        self.image_size = image_size
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def convert_to_images(self):
        cap = cv2.VideoCapture(self.video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            image_path = os.path.join(self.output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(image_path, frame)
            count += 1
        cap.release()
        print(f"Frames extracted: {count}")
