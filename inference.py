from collections.abc import Iterable

import numpy as np
import torch
from sklearn import metrics

from util import misc
from util.abnormal_utils import filt


def inference(model: torch.nn.Module, data_loader: Iterable,
                   device: torch.device,
                   log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing '

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions_teacher = []
    predictions_student_teacher = []
    labels = []
    videos = []
    frames = []
    for data_iter_step, (samples, grads, targets, label, vid, frame_name) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        frames += list(frame_name)
        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        model.train_TS = True # student-teacher reconstruction error
        if args.dataset == 'avenue':
            model.abnormal_score_func_TS = "L2"
        else:
            model.abnormal_score_func_TS = 'L1'
        _, _, _, recon_error_st_tc = model(samples, targets=targets, grad_mask=grads, mask_ratio=args.mask_ratio)
        recon_error_st_tc[0] = recon_error_st_tc[0].detach().cpu().numpy()
        recon_error_st_tc[1] = recon_error_st_tc[1].detach().cpu().numpy()
        predictions_student_teacher += list(recon_error_st_tc[0])
        predictions_teacher += list(recon_error_st_tc[1])

    # Compute statistics
    predictions_teacher = np.array(predictions_teacher)
    predictions_student_teacher = np.array(predictions_student_teacher)
    predictions = predictions_teacher+predictions_student_teacher
    labels = np.array(labels)
    videos = np.array(videos)

    if args.dataset =='avenue':
        evaluate_model(predictions, labels, videos,
                                           normalize_scores=False,
                                           range=38, mu=11)
    else:
        evaluate_model(predictions_teacher, labels, videos,
                       normalize_scores=True,
                       range=900, mu=282)


def evaluate_model(predictions, labels, videos,
                   range=302, mu=21, normalize_scores=False):

    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = filt(pred, range=range, mu=mu)
        if normalize_scores:
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        pred = np.nan_to_num(pred, nan=0.)

        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)
        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)

    # Micro-AUC
    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    # gather the stats from all processes
    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}, range:{range}, mu:{mu}, normalize scores:{normalize_scores}")
    return micro_auc, macro_auc
