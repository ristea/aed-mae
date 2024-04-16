import glob
import os
import random
import time

import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]


class AbnormalDatasetGradientsTrain(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
        else:
            raise Exception("Unknown dataset!")
        self.percent_abnormal = args.percent_abnormal
        self.input_3d = args.input_3d
        self.abnormal_data, self.data, self.gradients, self.masks_abnormal = self._read_data(data_path)

    def _read_data(self, data_path):
        data = []
        gradients = []
        abnormal_data = []
        masks_abnormal = []
        extension = None
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "train/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break

        dirs = list(glob.glob(os.path.join(data_path, "train", "frames", "*")))
        for dir in dirs:
            imgs_path = list(glob.glob(os.path.join(dir, f"*{extension}")))
            data += imgs_path
            video_name = os.path.basename(dir)
            gradients_path = []
            for img_path in imgs_path:
                gradients_path.append(os.path.join(data_path, "train", "gradients2", video_name,
                                              f"{int(os.path.basename(img_path).split('.')[0])}.png"))
                abnormal_data.append(os.path.join(data_path, "train", "frames_abnormal", video_name,
                                                  f"{int(os.path.basename(img_path).split('.')[0])}.png"))
                masks_abnormal.append(os.path.join(data_path, "train", "masks_abnormal", video_name,
                                                  f"{int(os.path.basename(img_path).split('.')[0])}.png"))
            gradients += gradients_path
        return abnormal_data, data, gradients, masks_abnormal

    def __getitem__(self, index):
        random_uniform = random.uniform(0, 1)
        if random_uniform <= self.percent_abnormal:
            img = cv2.imread(self.abnormal_data[index])
            # img = cv2.resize(img, self.args.usual_size[::-1])
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.abnormal_data, index)
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            # previous_img = cv2.resize(previous_img, self.args.usual_size[::-1])
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
            # next_img = cv2.resize(next_img, self.args.usual_size[::-1])
            if self.input_3d:
                img = np.concatenate([previous_img, img, next_img], axis=-1)
            mask = cv2.imread(self.masks_abnormal[index])[:,:,:1]
            # mask = cv2.resize(mask, self.args.usual_size[::-1])
            # mask = np.expand_dims(mask, axis=-1)
        else:
            img = cv2.imread(self.data[index])
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
            if self.input_3d:
                img = np.concatenate([previous_img, img, next_img], axis=-1)
            mask = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8)
        gradient = cv2.imread(self.gradients[index])
        target = cv2.imread(self.data[index])

        if img.shape[:2] != self.args.input_size or gradient.shape[:2] != self.args.input_size:
            img = cv2.resize(img, self.args.input_size[::-1])
            gradient = cv2.resize(gradient, self.args.input_size[::-1])
            mask = cv2.resize(mask, self.args.input_size[::-1])
            mask = np.expand_dims(mask, axis=-1)
        if target.shape[:2] != self.args.input_size:
            target = cv2.resize(target, self.args.input_size[::-1])

        target = np.concatenate((target, mask), axis=-1)
        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)
        target = target.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        target = (target - 127.5) / 127.5
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        return img, gradient, target

    def extract_meta_info(self, data, index):
        frame_no = int(data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(data[index].split("/")[:-1])
        len_frame_no = len(data[index].split("/")[-1].split('.')[0])
        return dir_path, frame_no, len_frame_no

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        frame_path = dir_path + "/" + str(frame_no + direction).zfill(length) + ".png"
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            return cv2.imread(dir_path + "/" + str(frame_no).zfill(length) + ".png")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__
