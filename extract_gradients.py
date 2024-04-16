import glob
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

def compute_gradients(data_root_folder, step, folder):
    extension = None
    for ext in IMG_EXTENSIONS:
        if len(list(glob.glob(os.path.join(data_root_folder, f"{folder}/frames", f"*/*{ext}")))) > 0:
            extension = ext
            break

    dirs = list(glob.glob(os.path.join(data_root_folder, folder, "frames", "*")))
    for video in tqdm(dirs):
        img_paths = list(glob.glob(os.path.join(video, f"*{extension}")))
        img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
        for i, img_path in enumerate(img_paths):
            previous = i-step
            if i-step <0:
                previous = i
            next = i+step
            if i + step >= len(img_paths):
                next = i
            previous_img = cv2.imread(img_paths[previous])
            previous_img = previous_img.astype(np.int32)
            next_img = cv2.imread(img_paths[next])
            next_img = next_img.astype(np.int32)
            gradient = np.abs(previous_img-next_img)
            gradient = gradient.astype(np.uint8)
            # Image.fromarray(gradient.astype(np.uint8)).show()
            # Image.fromarray(previous_img.astype(np.uint8)).show()
            # Image.fromarray(next_img.astype(np.uint8)).show()
            os.makedirs(os.path.join(data_root_folder, f"{folder}/gradients2/{os.path.basename(video)}"), exist_ok=True)
            gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)
            Image.fromarray(gradient).save(os.path.join(data_root_folder, f"{folder}/gradients2/{os.path.basename(video)}",
                                                        os.path.basename(img_path)))

if __name__=="__main__":
    root_folder_avenue = "/home/alin/datasets/Avenue_Dataset/Avenue Dataset"
    root_folder_shanghai = "/home/alin/datasets/SanhaiTech"
    compute_gradients(root_folder_avenue, 1, "train")
    compute_gradients(root_folder_avenue, 1, "test")
