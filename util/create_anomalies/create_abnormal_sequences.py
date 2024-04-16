import glob
import os.path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class AbnormalSequences:

    def __init__(self, parser):
        self.frames_path = os.path.join(parser.ubnormal_path, "train", "frames")
        self.annotations = os.path.join(parser.ubnormal_path, "train", "annotations")
        # create path to store abnormal sequences
        os.makedirs(os.path.join(parser.ubnormal_path, "abnormal_sequences"), exist_ok=True)
        self.store_dir_sequences = os.path.join(parser.ubnormal_path, "abnormal_sequences")
        self.args = parser

    def create_abnormal_sequences(self):
        for video in tqdm(os.listdir(self.frames_path)):
            if "abnormal" in video:
                frames = os.listdir(os.path.join(self.frames_path, video))
                annotations = os.listdir(os.path.join(self.annotations, video+"_annotations"))
                annotations = [annotation for annotation in annotations if 'png' in annotation]
                annotations.sort(key = lambda x: int(x.split(".")[0].split("_")[-2]))
                frames.sort(key = lambda x: int(x.split(".")[0]))
                frames = np.array(frames)
                annotations = np.array(annotations)
                tracks = np.loadtxt(os.path.join(self.annotations, video+"_annotations", video+"_tracks.txt"), delimiter=",", dtype=float)
                tracks = tracks.astype(dtype=np.int32)
                if len(tracks.shape)==1:
                   tracks = np.array([tracks])
                for t, track in enumerate(tracks):
                    selected_annotations = annotations[track[1]:track[2]]
                    selected_frames = frames[track[1]:track[2]]
                    object_no = track[0]
                    for i, frame in enumerate(selected_frames):
                        image = cv2.imread(os.path.join(self.frames_path, video, frame))
                        annotation = cv2.imread(os.path.join(self.annotations, video+"_annotations", selected_annotations[i]))
                        mask = (annotation==object_no)*1
                        mask = mask[:image.shape[0],:image.shape[1], :]
                        if mask.sum() < 800:
                            break
                        anomaly = image*mask
                        os.makedirs(os.path.join(self.store_dir_sequences, video, f"sequence_{t:04d}", "masks"), exist_ok=True)
                        os.makedirs(os.path.join(self.store_dir_sequences, video, f"sequence_{t:04d}", "anomalies"), exist_ok=True)
                        anomaly = cv2.resize(anomaly.astype(np.uint8), self.args.target_size)
                        mask*=255
                        mask = cv2.resize(mask.astype(np.uint8), self.args.target_size)
                        cv2.imwrite(os.path.join(self.store_dir_sequences, video, f"sequence_{t:04d}", "anomalies", f"{i}.png"), anomaly)
                        cv2.imwrite(os.path.join(self.store_dir_sequences, video, f"sequence_{t:04d}", "masks", f"{i}.png"), mask)