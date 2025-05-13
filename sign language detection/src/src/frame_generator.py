import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import yaml

class VideoFrameGenerator(Sequence):
    def __init__(self, video_dir, batch_size, img_size):
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.img_size = tuple(img_size)
        self.classes = ['violence', 'non_violence']
        self.video_paths, self.labels = self._load_videos()

    def _load_videos(self):
        video_paths, labels = [], []
        for idx, category in enumerate(self.classes):
            folder = os.path.join(self.video_dir, category)
            if not os.path.exists(folder):
                continue
            for file in os.listdir(folder):
                if file.endswith((".mp4", ".avi", ".mov")):
                    video_paths.append(os.path.join(folder, file))
                    labels.append(idx)
        return video_paths, labels

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, y = [], []
        for i, path in enumerate(batch_paths):
            frame = self._get_middle_frame(path)
            if frame is not None:
                X.append(frame)
                y.append(batch_labels[i])
        return np.array(X), np.array(y)

    def _get_middle_frame(self, path):
        cap = cv2.VideoCapture(path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle = count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
        success, frame = cap.read()
        cap.release()
        if not success:
            return None
        frame = cv2.resize(frame, self.img_size)
        return frame / 255.0

def get_generator_from_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    video_cfg = config["video_data"]
    return VideoFrameGenerator(video_cfg["path"], video_cfg["batch_size"], video_cfg["image_size"])
