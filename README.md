# Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors

Nicolae-Catalin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, & Mubarak Shah. (2024). Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors on CVPR, 2024

ArXiv URL: https://arxiv.org/abs/2306.12041

This is the official implementation of "Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors"



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

