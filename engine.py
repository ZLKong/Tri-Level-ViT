

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


WARM_UP_EPOCH = 10
TOTAL_DECAY_EPOCH = 100


def adjust_keep_rate(iters, epoch, warmup_epochs, total_epochs,
                    iter_per_epoch, base_keep_rate=0.7, max_keep_rate=1):
    # reference: https://github.com/youweiliang/evit/blob/master/helpers.py#L7
    # Token-Level Function
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate

    total_decay_iters = iter_per_epoch * (total_epochs - warmup_epochs)
    iters = iters - iter_per_epoch * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_decay_iters * math.pi) + 1) * 0.5

    return keep_rate



def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    example_stats_train=None,  train_example_idx=None,     # additional
                    set_training_mode=True, args=None,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    #i=0

    # Example-level variables 
    correct = 0.
    total = 0.

    # Token-Level variables 
    cur_iter = epoch * len(data_loader)
    iter_per_epoch = len(data_loader)
    ite_step = 0


    for samples, targets, idx in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        targets_org=targets

        # >>>>>>>> Dynamic Keep Ratio (Token Sparse)  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    
        token_ratio = adjust_keep_rate(cur_iter + ite_step, epoch, warmup_epochs=WARM_UP_EPOCH,
                                total_epochs=WARM_UP_EPOCH + TOTAL_DECAY_EPOCH,
                                iter_per_epoch=iter_per_epoch, base_keep_rate=args.keep_ratio)  # get current keep ratio, gradually decrease from 1
        
        attn_ratio = adjust_keep_rate(cur_iter + ite_step, epoch, warmup_epochs=WARM_UP_EPOCH,
                                total_epochs=WARM_UP_EPOCH + TOTAL_DECAY_EPOCH,
                                iter_per_epoch=iter_per_epoch, base_keep_rate=args.attn_ratio)  # get current keep ratio, gradually decrease from 1

        ite_step += 1
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():#(enabled=False):
            #outputs = model(samples)
            outputs = model(samples, ratio=token_ratio, attn_ratio=attn_ratio)
            loss = criterion(samples, outputs, targets)

        # >>>>>>>>  data sparse  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        outputs_pred = outputs
        _, predicted = torch.max(outputs_pred.data, 1)
        acc = predicted == targets_org
        correct += predicted.eq(targets_org.data).cpu().sum()
        total += targets_org.size(0)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        optimizer.zero_grad()

        # >>>>>>>>  data sparse  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for j, index in enumerate(idx):

            # Get index in original dataset (not sorted by forgetting)
            #index_in_original_dataset = index
            index_in_original_dataset = train_example_idx[index]


            # Compute missclassification margin
            output_correct_class = outputs_pred.data[j, targets_org[j].item()]
            sorted_output, _ = torch.sort(outputs_pred.data[j, :])
            if acc[j]:
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_output[-2]
            else:
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(
            ) - output_highest_incorrect_class.item()

            # Add the statistics of the current training example to dictionary
            index_stats = example_stats_train.get(index_in_original_dataset, [[], [], []])
            index_stats[0].append(loss.item())
            index_stats[1].append(acc[j].sum().item())
            index_stats[2].append(margin)
            example_stats_train[index_in_original_dataset] = index_stats

        # Add training accuracy to dict
        index_stats = example_stats_train.get('train', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats_train['train'] = index_stats
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        #i+=1
        #if i > 20: break

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    # fix keep ratio in inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for images, target, idx in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            #output = model(images)
            output = model(images, ratio=args.keep_ratio, attn_ratio=args.attn_ratio)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
