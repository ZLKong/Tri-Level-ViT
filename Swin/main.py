# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from data.build import MyTrainDataset
from torch.utils.data import Dataset
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

# class MyTrainDataset(Dataset):
#     def __init__(self, is_train, config, train_example_idx):
#         transform = build_transform(is_train, config)
#         if config.DATA.DATASET == 'imagenet':
#             root = os.path.join(config.DATA.DATA_PATH, 'train' if is_train else 'val')
#             self.dataset = datasets.ImageFolder(root, transform=transform)
#             #self.nb_classes = 1000
#         if train_example_idx is not None:
#             self.dataset.imgs = [self.dataset.imgs[i] for i in train_example_idx]
#             self.dataset.targets = [self.dataset.targets[i] for i in train_example_idx]
#             self.dataset.samples = [self.dataset.samples[i] for i in train_example_idx]
#     def __getitem__(self, index):
#         data, target = self.dataset[index]
#         # Your transformations here (or set it in CIFAR10)
#         return data, target, index#, nb_classes
#     def __len__(self):
#         return len(self.dataset)

#     def forward(self):
#         dataset=self.dataset
#         print('len(dataset)',len(dataset))
#         return dataset

def compute_forgetting_statistics(diag_stats, npresentations):

    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}
    print('len(diag_stats.items())',len(diag_stats.items()))

    for example_id, example_stats in diag_stats.items():

        # Skip 'train' and 'test' keys of diag_stats
        if not isinstance(example_id, str):

            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[1][:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find number of presentations needed to learn example,
            # e.g. last presentation when acc is 0
            if len(np.where(presentation_acc == 0)[0]) > 0:
                presentations_needed_to_learn[example_id] = np.where(
                    presentation_acc == 0)[0][-1] + 1
            else:
                presentations_needed_to_learn[example_id] = 0

            # Find the misclassication margin for each presentation of the example
            margins_per_presentation = np.array(
                example_stats[2][:npresentations])

            # Find the presentation at which the example was first learned,
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

    return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned


# Sorts examples by number of forgetting counts during training, in ascending order
# If an example was never learned, it is assigned the maximum number of forgetting counts
# If multiple training runs used, sort examples by the sum of their forgetting counts over all runs
#
# unlearned_per_presentation_all: list of dictionaries, one per training run
# first_learned_all: list of dictionaries, one per training run
# npresentations: number of training epochs
#
# Returns 2 numpy arrays containing the sorted example ids and corresponding forgetting counts
#
def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                first_learned_all, npresentations):

    # Initialize lists
    example_original_order = []
    example_stats = []

    for example_id in unlearned_per_presentation_all[0].keys():

        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Iterate over all training runs to calculate the total forgetting count for current example
        for i in range(len(unlearned_per_presentation_all)):

            # Get all presentations when current example was forgotten during current training run
            stats = unlearned_per_presentation_all[i][example_id]

            # If example was never learned during current training run, add max forgetting counts
            if np.isnan(first_learned_all[i][example_id]):
                example_stats[-1] += npresentations
            else:
                example_stats[-1] += len(stats)

    print('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    return np.array(example_original_order)[np.argsort(
        example_stats)], np.sort(example_stats)



def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true', help='Fused window shift & window partition, similar for reversed part.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn, dataset_remove, train_example_idx, removed_example_idx = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    example_stats_train = {}
    putback_batch = 0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)


    ################################################# Sparse data #################################################
            #data_loader_remove.sampler.set_epoch(epoch)
            
        print('len(example_stats_train)', len(example_stats_train))
            #if (epoch+1) % 30==0: #> 0:
        if epoch == 30 * (putback_batch + 1):
        #if epoch >0:
            if putback_batch <= 4:
                print('epoch',epoch)

                unlearned_per_presentation_all, first_learned_all = [], []

                _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
                    example_stats_train, 30)
                print('unlearned_per_presentation',len(unlearned_per_presentation))
                print('first_learned',len(first_learned))

                unlearned_per_presentation_all.append(
                    unlearned_per_presentation)
                first_learned_all.append(first_learned)

                print('unlearned_per_presentation_all',len(unlearned_per_presentation_all))
                print('first_learned_all',len(first_learned_all))


                # Sort examples by forgetting counts in ascending order, over one or more training runs
                ordered_examples, ordered_values = sort_examples_by_forgetting(
                        unlearned_per_presentation_all, first_learned_all, 30)

                print('epoch before ordered_examples len',len(ordered_examples))
                #print('epoch before len(dataset_train.targets)',len(dataset_train.targets))
                elements_to_remove = np.array(
                    ordered_examples)[0:64058]
                # Remove the corresponding elements
                print('elements_to_remove',len(elements_to_remove))

                targets = [i.tolist() for i in elements_to_remove] #class
                #print('targets', targets)
                elements_to_remove = np.array(targets)
                #print('len(elements_to_remove)',len(elements_to_remove))
                #print('elements_to_remove',elements_to_remove)
                print('before prune len(train_example_idx)', len(train_example_idx))
                #print('train_example_idx',train_example_idx)
                #print('elements_to_remove',elements_to_remove.tolist())

                train_example_idx  = np.setdiff1d(
                    #range(len(train_example_idx)), elements_to_remove)
                    train_example_idx , elements_to_remove)
                print('prune len(train_example_idx)',len(train_example_idx))

                #random shuffle removed_example_idx before grow

                np.random.shuffle(removed_example_idx)
                train_example_idx = np.concatenate((train_example_idx, removed_example_idx[0:64058]), axis=0)
                removed_example_idx=removed_example_idx[64058:]

                putback_batch = putback_batch + 1
                #add 5000 back to the training dataset from the originally removed 5000 data
                #train_example_idx =np.concatenate((train_example_idx ,removed_example_idx[len(elements_to_remove)*putback_batch:len(elements_to_remove)*(putback_batch+1)]),axis=0)
                #putback_batch=putback_batch+1
                print('grow len(train_example_idx)',len(train_example_idx))

                #add the removed 5000 to the removed dataset.
                removed_example_idx=np.concatenate((removed_example_idx,elements_to_remove), axis=0)

                print('new len(removed_example_idx)', len(removed_example_idx))

                dataset_train = MyTrainDataset(is_train=True, config=config,train_example_idx=train_example_idx)

                #config.MODEL.NUM_CLASSES = 1000
                num_tasks = dist.get_world_size()
                global_rank = dist.get_rank()
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                    )
                data_loader_train = torch.utils.data.DataLoader(
                    dataset_train, sampler=sampler_train,
                    batch_size=config.DATA.BATCH_SIZE,
                    num_workers=config.DATA.NUM_WORKERS,
                    pin_memory=config.DATA.PIN_MEMORY,
                    drop_last=True,
                )
                example_stats_train = {}

#    ################################################# Training #################################################


        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, example_stats_train, train_example_idx)
        if dist.get_rank() == 0 and (epoch % 30 == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, example_stats_train, train_example_idx):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    # Example-level variables 
    correct = 0.
    total = 0.

    for idx, (samples, targets, idx_data) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        targets_org=targets
        idx_data = idx_data.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # >>>>>>>>  data sparse  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        outputs_pred = outputs
        _, predicted = torch.max(outputs_pred.data, 1)
        acc = predicted == targets_org
        correct += predicted.eq(targets_org.data).cpu().sum()
        total += targets_org.size(0)

        for j, index in enumerate(idx_data):

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
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target, idx_data) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
