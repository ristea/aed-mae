import datetime
import gc

import numpy as np
import torch
from model.model_factory import mae_cvt_patch16




def get_MAE(img_size, device):
    model = mae_cvt_patch16(img_size=img_size)
    model.to(device)
    model.eval()
    return model


def banchmark(batch_size, model, device, img_size, num_iter=100):
    img = torch.randn(batch_size, 9, img_size[0], img_size[1]).to(device)
    targets = torch.randn(batch_size, 4, img_size[0], img_size[1]).to(device)
    time = []
    for i in range(0, num_iter):
        start = datetime.datetime.now()
        _ = model(img, targets, mask_ratio=0.50)
        end = datetime.datetime.now()
        time.append((end - start).microseconds / 1e3)
        if i % 5 ==0:
            gc.collect()

    return np.mean(time)


if __name__ == '__main__':
    device = 'cuda'
    img_size = (320, 640)
    batch_size = 32

    model = get_MAE(img_size=img_size, device=device)
    mean_time = banchmark(batch_size, model, device, img_size=img_size, num_iter=1000)

    print(f"Mean time: {mean_time}")
    print(f"FPS: {batch_size * 1000 / mean_time}")
