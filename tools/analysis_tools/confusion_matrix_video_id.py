# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import tempfile

import torch
from mmengine import dump, list_from_file, load
from mmengine.config import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

import numpy as np
from mmaction.evaluation import ConfusionMatrix
from mmaction.registry import DATASETS
from mmaction.utils import register_all_modules


def calculate_metrics(confusion_matrix):
    """Calculate precision, recall, and F1-score from the confusion matrix."""
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (
            precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    return precision, recall, f1_score


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval a checkpoint and draw the confusion matrix.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'ckpt_or_result',
        type=str,
        help='The checkpoint file (.pth) or '
        'dumpped predictions pickle file (.pkl).')
    parser.add_argument('--out', help='the file to save the confusion matrix.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the metric result by matplotlib if supports.')
    parser.add_argument(
        '--show-path', type=str, help='Path to save the visualization image.')
    parser.add_argument(
        '--include-values',
        action='store_true',
        help='To draw the values in the figure.')
    parser.add_argument('--label-file', default=None, help='Labelmap file')
    parser.add_argument(
        '--target-classes',
        type=int,
        nargs='+',
        default=[],
        help='Selected classes to evaluate, and remains will be neglected')
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='The color map to use. Defaults to "viridis".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def calculate_video_predictions(data_samples):
    """Calculate video-level predictions and return a mapping to add to data samples."""
    # Group predictions by source
    source_predictions = {}
    
    for sample in data_samples:
        source = sample['gt_source_id']
        pred_label = sample['pred_label'].item()
        
        if source not in source_predictions:
            source_predictions[source] = []
        
        source_predictions[source].append(pred_label)
    
    # Determine video-level prediction for each source using majority voting
    source_to_pred_video_label = {}
    
    for source, predictions in source_predictions.items():
        unique_labels, counts = np.unique(predictions, return_counts=True)
        max_count = np.max(counts)
        max_indices = np.where(counts == max_count)[0]
        
        # If tie, choose the higher value
        if len(max_indices) > 1:
            pred_video_label = unique_labels[max_indices[-1]]
        else:
            pred_video_label = unique_labels[max_indices[0]]
        
        # Store mapping from source to video-level prediction
        source_to_pred_video_label[source] = pred_video_label
    
    return source_to_pred_video_label

def add_pred_video_label_to_samples(data_samples):
    """Add pred_video_label field to each data sample."""
    # Calculate video-level predictions
    source_to_pred_video_label = calculate_video_predictions(data_samples)
    
    # Add pred_video_label to each sample
    for sample in data_samples:
        source = sample['gt_source_id']
        pred_video_label = source_to_pred_video_label[source]
        sample['pred_video_label'] = torch.tensor([pred_video_label])
    
    return data_samples

def main():
    args = parse_args()

    # Register all modules in mmaction into the registries
    register_all_modules(init_default_scope=False)

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.ckpt_or_result.endswith('.pth'):
        # Set confusion matrix as the metric.
        cfg.test_evaluator = dict(type='ConfusionMatrix')
        cfg.load_from = str(args.ckpt_or_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.work_dir = tmpdir
            runner = Runner.from_cfg(cfg)
            classes = runner.test_loop.dataloader.dataset.metainfo.get('classes')
            
            # Get predictions and frame-level confusion matrix
            # results = runner.test()
            # cm_frame = results['confusion_matrix/result']
            
            # Get data samples for video-level aggregation
            data_samples = []
            for outputs in runner.test_loop.predictions:
                data_samples.extend(outputs)  # Collect all data samples
            
            # Add pred_video_label to data samples
            data_samples = add_pred_video_label_to_samples(data_samples)
            
            # Save updated data samples
            dump(data_samples, args.ckpt_or_result + '.updated')
            
            logging.shutdown()
    else:
        # Loading from prediction file
        data_samples = load(args.ckpt_or_result)
        evaluator = Evaluator(ConfusionMatrix())
        metrics = evaluator.offline_evaluate(data_samples, None)
        cm_frame = metrics['confusion_matrix/result']
        
        # Add pred_video_label to data samples
        data_samples = add_pred_video_label_to_samples(data_samples)
        
        # Save updated data samples back to original location
        dump(data_samples, args.ckpt_or_result)
        
        try:
            # Try to build the dataset.
            dataset = DATASETS.build({**cfg.test_dataloader.dataset, 'pipeline': []})
            classes = dataset.metainfo.get('classes')
        except Exception:
            classes = None

    # Create video-level confusion matrix using the newly added pred_video_label field
    if classes is None:
        num_classes = cm_frame.shape[0]
        classes = list(range(num_classes))

    if args.label_file is not None:
        classes = list_from_file(args.label_file)

    # Calculate video-level confusion matrix
    source_seen = set()
    seg_gt_labels = []
    seg_pred_labels = []
    video_gt_labels = []
    video_pred_labels = []
    
    for sample in data_samples:
        source = sample['gt_source_id']
        seg_gt_label = sample['gt_label'].item()
        seg_pred_label = sample['pred_label'].item()
        seg_gt_labels.append(seg_gt_label)
        seg_pred_labels.append(seg_pred_label)
        if source not in source_seen:
            source_seen.add(source)
            gt_label = sample['gt_label'].item()
            pred_video_label = sample['pred_video_label'].item()
            video_gt_labels.append(gt_label)
            video_pred_labels.append(pred_video_label)
    
    num_classes = len(classes)
    cm_frame = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for gt, pred in zip(seg_gt_labels, seg_pred_labels):
        cm_frame[gt, pred] += 1

    cm_video = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for gt, pred in zip(video_gt_labels, video_pred_labels):
        cm_video[gt, pred] += 1

    if args.target_classes:
        assert len(args.target_classes) > 1, \
            'please ensure select more than one class'
        target_idx = torch.tensor(args.target_classes)
        cm_frame = cm_frame[target_idx][:, target_idx]
        cm_video = cm_video[target_idx][:, target_idx]
        classes = [classes[idx] for idx in target_idx]

    if args.out is not None:
        # Save both confusion matrices
        frame_out = args.out.replace('.', '_frame.')
        video_out = args.out.replace('.', '_video.')
        dump(cm_frame, frame_out)
        dump(cm_video, video_out)
        print(f'Frame-level confusion matrix saved to {frame_out}')
        print(f'Video-level confusion matrix saved to {video_out}')

    if args.show or args.show_path is not None:
        # Plot frame-level confusion matrix
        fig_frame = ConfusionMatrix.plot(
            cm_frame,
            show=False,
            classes=classes,
            include_values=args.include_values,
            cmap=args.cmap)
        
        # Plot video-level confusion matrix
        fig_video = ConfusionMatrix.plot(
            cm_video,
            show=False,
            classes=classes,
            include_values=args.include_values,
            cmap=args.cmap)
        
        if args.show_path is not None:
            frame_path = args.show_path.replace('.', '_frame.')
            video_path = args.show_path.replace('.', '_video.')
            fig_frame.savefig(frame_path)
            fig_video.savefig(video_path)
            print(f'The frame-level confusion matrix is saved at {frame_path}.')
            print(f'The video-level confusion matrix is saved at {video_path}.')
        
        if args.show:
            import matplotlib.pyplot as plt
            plt.figure(fig_frame.number)
            plt.title("Frame-level Confusion Matrix")
            plt.figure(fig_video.number)
            plt.title("Video-level Confusion Matrix")
            plt.show()

    # Calculate frame-level metrics
    frame_precision, frame_recall, frame_f1_score = calculate_metrics(cm_frame.numpy())
    print('Frame-level Precision:', frame_precision)
    print('Frame-level Recall:', frame_recall)
    print('Frame-level F1-score:', frame_f1_score)
    
    # Calculate video-level metrics
    video_precision, video_recall, video_f1_score = calculate_metrics(cm_video.numpy())
    print('Video-level Precision:', video_precision)
    print('Video-level Recall:', video_recall)
    print('Video-level F1-score:', video_f1_score)
    
    print(f'Updated data samples with pred_video_label saved to {args.ckpt_or_result}')

if __name__ == '__main__':
    main()
