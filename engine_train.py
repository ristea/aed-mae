import math
import sys
import torch
import util.misc as misc
from typing import Iterable
import numpy as np
from util.abnormal_utils import filt
import sklearn.metrics as metrics


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None, args=None):
    model.train(True)
    model = model.float()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    if epoch >= args.start_TS_epoch:
        model.train_TS = True
        model.freeze_backbone()

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, grad_mask, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        grad_mask = grad_mask.to(device, non_blocking=True)

        loss, _, _ = model(samples, grad_mask=grad_mask, targets=targets, mask_ratio=args.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                   device: torch.device, epoch: int,
                   log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    if epoch >= args.start_TS_epoch:
        model.train_TS = True
        model.freeze_backbone()

    predictions = []
    labels = []
    videos = []
    for data_iter_step, (samples, grads, targets, label, vid, _) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())

        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        _, _, _, recon_error = model(samples, grad_mask=grads,targets=targets, mask_ratio=args.mask_ratio)
        if isinstance(recon_error, list) or isinstance(recon_error, tuple):
            if len(recon_error)==2:
                recon_error = recon_error[0] + recon_error[1]
            else:
                recon_error = 2.2*recon_error[0] + 1.1*recon_error[1] +recon_error[2]
        recon_error = recon_error.detach().cpu().numpy()
        predictions += list(recon_error)

    # Compute statistics
    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = np.nan_to_num(pred, nan=0.)
        if args.dataset=='avenue':
            pred = filt(pred, range=102, mu=12)
        else:
            pred = filt(pred, range=102, mu=12)
            # raise ValueError('Unknown parameters for predictions postprocessing')
        # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

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
    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")
    return {"micro": micro_auc, "macro": macro_auc}
