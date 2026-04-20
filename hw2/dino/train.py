"""
DINO training and evaluation script (single-GPU).
"""
import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_dataset, get_coco_api_from_dataset
from model import build_model
import util.misc as utils
from util.misc import collate_fn

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import copy
import contextlib

try:
    import wandb
except Exception:
    wandb = None


# ---------------------------------------------------------------------------
# COCO Evaluator (simplified, single-GPU)
# ---------------------------------------------------------------------------

class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids_eval, eval_imgs = _coco_evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            _create_common_coco_eval(self.coco_eval[iou_type],
                                     self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction["boxes"]
            boxes = _convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            coco_results.extend([
                {"image_id": original_id, "category_id": labels[k],
                 "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            ])
        return coco_results


def _convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def _coco_evaluate(coco_eval):
    p = coco_eval.params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    coco_eval.params = p
    coco_eval._prepare()
    catIds = p.catIds if p.useCats else [-1]
    if p.iouType in ('segm', 'bbox'):
        computeIoU = coco_eval.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = coco_eval.computeOks
    coco_eval.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds for catId in catIds}
    maxDet = p.maxDets[-1]
    evalImgs = [
        coco_eval.evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds for areaRng in p.areaRng for imgId in p.imgIds]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    return p.imgIds, evalImgs


def _create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids = list(np.unique(img_ids))
    eval_imgs_flat = list(eval_imgs.flatten())
    coco_eval.evalImgs = eval_imgs_flat
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def load_env_file(env_path):
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def setup_wandb(config, config_path):
    env_path = Path(config_path).resolve().parent / '.env'
    load_env_file(env_path)
    api_key = os.getenv('WANDB_API_KEY') or os.getenv('wandb_api_key')
    if not api_key:
        print('WANDB API key not found in .env, skipping wandb logging.')
        return None
    if wandb is None:
        print('wandb package is not installed, skipping wandb logging.')
        return None
    os.environ['WANDB_API_KEY'] = api_key
    run = wandb.init(
        project=config.get('wandb_project', 'dino-model'),
        config=config,
        name=config.get('wandb_run_name'),
        reinit=True)
    return run


def compute_map95(coco_evaluator):
    if coco_evaluator is None or 'bbox' not in coco_evaluator.coco_eval:
        return float('nan')
    bbox_eval = coco_evaluator.coco_eval['bbox']
    if 'precision' not in bbox_eval.eval:
        return float('nan')
    iou_thrs = bbox_eval.params.iouThrs
    idx = np.where(np.isclose(iou_thrs, 0.95))[0]
    if len(idx) == 0:
        return float('nan')
    precision = bbox_eval.eval['precision']
    p = precision[idx[0], :, :, 0, -1]
    p = p[p > -1]
    if p.size == 0:
        return float('nan')
    return float(np.mean(p))


# ---------------------------------------------------------------------------
# Optimizer with lr_linear_proj_mult for deformable params
# ---------------------------------------------------------------------------

def build_optimizer(model, config):
    lr = config['lr']
    lr_backbone = config['lr_backbone']
    lr_linear_proj_mult = config.get('lr_linear_proj_mult', 0.1)
    lr_linear_proj_names = config.get('lr_linear_proj_names',
                                      ['reference_points', 'sampling_offsets'])
    weight_decay = config['weight_decay']

    backbone_params = []
    linear_proj_params = []
    other_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params.append(p)
        elif any(lpn in n for lpn in lr_linear_proj_names):
            linear_proj_params.append(p)
        else:
            other_params.append(p)

    param_dicts = [
        {'params': other_params, 'lr': lr},
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': linear_proj_params, 'lr': lr * lr_linear_proj_mult},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    return optimizer


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    max_norm=0, use_dn=True):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    progress = tqdm(data_loader, desc=f'Train Epoch {epoch}', leave=False)
    for samples, targets in progress:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if use_dn:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items()
                                    if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        progress.set_postfix(loss=f'{loss_value:.4f}',
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    metric_logger.synchronize_between_processes()
    print('Averaged train stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device,
             epoch, use_dn=True):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('class_error',
                            utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    iou_types = tuple(k for k in ('bbox',) if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    progress = tqdm(data_loader, desc=f'Valid Epoch {epoch}', leave=False)
    for samples, targets in progress:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if use_dn:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items()
                                    if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        valid_loss = sum(loss_dict_reduced_scaled.values()).item()
        metric_logger.update(loss=valid_loss,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output
               for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        progress.set_postfix(loss=f'{valid_loss:.4f}')

    metric_logger.synchronize_between_processes()
    print('Averaged valid stats:', metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None and 'bbox' in postprocessors.keys():
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device(config['device'])

    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    wandb_run = setup_wandb(config, config_path)

    model, criterion, postprocessors = build_model(config)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = build_optimizer(model, config)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config.get('lr_eta_min', 1e-6))

    dataset_train = build_dataset(image_set='train', config=config)
    dataset_val = build_dataset(image_set='val', config=config)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config['batch_size'], drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn,
                                   num_workers=config['num_workers'])
    data_loader_val = DataLoader(dataset_val, config['batch_size'], sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn,
                                 num_workers=config['num_workers'])

    base_ds = get_coco_api_from_dataset(dataset_val)

    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output_dir']) / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir}')

    start_epoch = 0
    use_dn = config.get('use_dn', True)

    if config.get('resume') and os.path.isfile(config['resume']):
        print(f"Resuming from checkpoint: {config['resume']}")
        checkpoint = torch.load(config['resume'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if ('optimizer' in checkpoint and 'lr_scheduler' in checkpoint
                and 'epoch' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

    print('Start training')
    start_time = time.time()
    best_map = 0.0
    best_val_loss = float('inf')

    epoch_progress = tqdm(range(start_epoch, config['epochs']), desc='Epochs')
    for epoch in epoch_progress:
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            config['clip_max_norm'], use_dn=use_dn)
        lr_scheduler.step()

        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if (epoch + 1) % config['lr_drop'] == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

        for checkpoint_path in checkpoint_paths:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config': config,
            }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device,
            epoch, use_dn=use_dn)

        coco_stats = test_stats.get('coco_eval_bbox', [float('nan')] * 12)
        coco_map_50_95 = float(coco_stats[0]) if len(coco_stats) > 0 else float('nan')
        coco_map_50 = float(coco_stats[1]) if len(coco_stats) > 1 else float('nan')
        coco_map_75 = float(coco_stats[2]) if len(coco_stats) > 2 else float('nan')
        coco_map_95 = compute_map95(coco_evaluator)

        val_loss = float(test_stats.get('loss', float('inf')))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config': config,
            }, output_dir / 'best.pth')
            print(f'New best val_loss: {best_val_loss:.4f} at epoch {epoch}')

        if coco_map_50_95 > best_map:
            best_map = coco_map_50_95
            print(f'New best mAP: {best_map:.4f} at epoch {epoch}')

        epoch_progress.set_postfix(
            train_loss=f"{train_stats.get('loss', float('nan')):.4f}",
            valid_loss=f"{test_stats.get('loss', float('nan')):.4f}",
            map5095=f'{coco_map_50_95:.4f}')

        if wandb_run is not None:
            wandb.log({
                'epoch': epoch,
                'train/loss': float(train_stats.get('loss', float('nan'))),
                'train/loss_ce': float(train_stats.get('loss_ce', float('nan'))),
                'train/loss_bbox': float(train_stats.get('loss_bbox', float('nan'))),
                'train/loss_giou': float(train_stats.get('loss_giou', float('nan'))),
                'train/class_error': float(train_stats.get('class_error', float('nan'))),
                'train/lr': float(train_stats.get('lr', optimizer.param_groups[0]['lr'])),
                'valid/loss': float(test_stats.get('loss', float('nan'))),
                'valid/loss_ce': float(test_stats.get('loss_ce', float('nan'))),
                'valid/loss_bbox': float(test_stats.get('loss_bbox', float('nan'))),
                'valid/loss_giou': float(test_stats.get('loss_giou', float('nan'))),
                'valid/class_error': float(test_stats.get('class_error', float('nan'))),
                'valid/mAP@50': coco_map_50,
                'valid/mAP@75': coco_map_75,
                'valid/mAP@95': coco_map_95,
                'valid/mAP@50:95': coco_map_50_95,
            }, step=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'coco_mAP@50': coco_map_50,
                     'coco_mAP@75': coco_map_75,
                     'coco_mAP@95': coco_map_95,
                     'coco_mAP@50:95': coco_map_50_95}

        with (output_dir / 'log.txt').open('a') as f:
            f.write(json.dumps(log_stats) + '\n')

    if wandb_run is not None:
        wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO Training')
    parser.add_argument('--config', default='config.json', type=str,
                        help='path to config json file')
    args = parser.parse_args()
    main(args.config)
