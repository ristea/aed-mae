import copy
import os.path

import cv2
import numpy as np


class DatasetAbnormalAug:

    def __init__(self, parser):
        self.input_dataset_loc = os.path.join(parser.input_dataset, "train", "frames")
        self.output_dir = os.path.join(parser.output_dataset,"train","frames")
        self.output_dir_masks = os.path.join(parser.output_dataset,"train","masks")
        self.abnormal_source = os.path.join(parser.ubnormal_path, "abnormal_sequences")
        self.abnormal_sources = os.listdir(self.abnormal_source)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir_masks, exist_ok=True)
        self.args = parser

    def do_aug(self):
        for video in os.listdir(self.input_dataset_loc):
           print(f"Processing video:{video}")
           frames = os.listdir(os.path.join(self.input_dataset_loc, video))
           frames.sort(key=lambda x: int(x.split(".")[0]))
           source, anomalies, masks = self.get_abnormal_source_seq()
           index = 0
           index_anomalies = 0
           while index < len(frames):
             image = cv2.imread(os.path.join(self.input_dataset_loc, video, frames[index]))
             mask = cv2.imread(os.path.join(source, "masks", masks[index_anomalies]))
             target_mask = copy.deepcopy(mask)
             mask = (mask == 0) * 1
             mask = mask.astype(np.uint8)
             while np.sum((mask == 0))>5000:
                 index_anomalies = 0
                 source, anomalies, masks = self.get_abnormal_source_seq()
                 mask = cv2.imread(os.path.join(source, "masks", masks[index_anomalies]))
                 target_mask = copy.deepcopy(mask)
                 mask = (mask == 0) * 1
                 mask = mask.astype(np.uint8)
             anomaly = cv2.imread(os.path.join(source, "anomalies", anomalies[index_anomalies]), cv2.IMREAD_GRAYSCALE)
             anomaly = np.expand_dims(anomaly, axis=-1).repeat(3, axis=-1)
             image = cv2.resize(image, self.args.target_size)
             anomaly = cv2.GaussianBlur(anomaly, (3, 3), 0)

             # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
             # mask = cv2.erode(mask, kernel, iterations=1)
             masked_image = image * mask
             final_image = anomaly+masked_image
             os.makedirs(os.path.join(self.output_dir, video), exist_ok=True)
             os.makedirs(os.path.join(self.output_dir_masks, video), exist_ok=True)
             cv2.imwrite(os.path.join(self.output_dir, video, frames[index]), final_image)
             cv2.imwrite(os.path.join(self.output_dir_masks, video, frames[index]), target_mask)
             index+=1
             index_anomalies+=1
             if index_anomalies==len(anomalies):
                 index_anomalies=0
                 source, anomalies, masks = self.get_abnormal_source_seq()


    def get_abnormal_source_seq(self):
        random_source = self.abnormal_sources[np.random.randint(0, len(self.abnormal_sources))]
        sequences = os.listdir(os.path.join(self.abnormal_source, random_source))
        random_sequence = sequences[np.random.randint(0, len(sequences))]
        source = os.path.join(self.abnormal_source, random_source, random_sequence)
        anomalies = os.listdir(os.path.join(source, "anomalies"))
        masks = os.listdir(os.path.join(source, "masks"))
        anomalies.sort(key=lambda x: int(x.split(".")[0]))
        masks.sort(key=lambda x: int(x.split(".")[0]))
        return source, anomalies, masks
