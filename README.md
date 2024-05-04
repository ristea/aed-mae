# Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors (CVPR 2024) - Official Repository

### by Nicolae-Catalin Ristea*, Florinel-Alin Croitoru*, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah

\* Authors have contributed equally.

This is the official repository of "Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors" accepted at CVPR 2024.

ArXiv preprint: https://arxiv.org/abs/2306.12041

## License

The source code and models are released under the Creative Common Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) license.

## Description

We propose an efficient abnormal event detection model based on a lightweight masked auto-encoder (AE) applied at the video frame level. The novelty of the proposed model is threefold. First, we introduce an approach to weight tokens based on motion gradients, thus shifting the focus from the static background scene to the foreground objects. Second, we integrate a teacher decoder and a student decoder into our architecture, leveraging the discrepancy between the outputs given by the two decoders to improve anomaly detection. Third, we generate synthetic abnormal events to augment the training videos, and task the masked AE model to jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps. Our design leads to an efficient and effective model, as demonstrated by the extensive experiments carried out on four benchmarks: Avenue, ShanghaiTech, UBnormal and UCSD Ped2. The empirical results show that our model achieves an excellent trade-off between speed and accuracy, obtaining competitive AUC scores, while processing 1655 FPS. Our model is between 8 and 70 times faster than competing methods.

## Citation 
Please cite our work if you use any material released in this repository.
```
@InProceedings{Ristea-CVPR-2024,
  author    = {Ristea, Nicolae-Catalin and Croitoru, Florinel-Alin and Ionescu, Radu Tudor and Popescu, Marius and Khan, Fahad Shahbaz and Shah, Mubarak},
  title     = "{Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors}",
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  }
```

## Preprocessing steps

1. Compute the temporal gradients
```bash
python extract_gradients.py 
```
Before running the above command, you have to change the root folders used in the script to reflect the location where your dataset is stored.

2. Include pseudo anomalies from UBNormal
```bash
cd util/create_anomalies
python main.py
```
Same as before, you have to change the arguments to reflect the location where the data is stored.

## Training/Inference
0. Preliminaries

Set the dataset location in "configs/configs.py".

1. Train.
```bash
python main.py --dataset <avenue or shanghai>
```
The "dataset" parameter will choose between the two config options.

2. Inference.

If you want to check the Micro-AUC and Macro-AUC scores you have to change in configs/configs.py the run_type variable to "inference"
and then rerun "main.py".

## Checkpoints:

https://drive.google.com/drive/folders/1Qpx1ZohOPgdeR0uMZkLqFaaNCOpcZ_aF?usp=sharing

