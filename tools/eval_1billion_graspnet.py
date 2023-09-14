import os
import cv2
import glob
import numpy as np
import imageio
import torch
import json

from tqdm import tqdm
from termcolor import colored

from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import annotations_to_instances
from detectron2.data.datasets.coco import load_coco_json

import compute_PRF
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    
    predictor = DefaultPredictor(cfg)
    data_root_path = 'datasets/1billion_graspnet'
    # W, H = cfg.INPUT.IMG_SIZE
    
    # load dataset
    json_path = 'datasets/1billion_graspnet/annotations/realsense/test_seen.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = data['images']
    
    coco_json = load_coco_json(json_path, image_root=data_root_path)
    # annos = data['annotations']
    # targets = annotations_to_instances(annos, (720, 1280))
    # print(targets)

    rgb_paths = []
    metrics_all = []
    num_inst_mat = 0
    iou_masks = 0
    for i in range(len(images)):
        rgb_paths.append(images[i]['file_name'])
    rgb_paths = sorted(rgb_paths)
    
    for i, path in enumerate(tqdm(rgb_paths)):
        rgb_path = os.path.join(data_root_path, path)
        lable_path = rgb_path.replace('rgb', 'label')
        
        inputs = cv2.imread(rgb_path)
        
        # anno = coco_json[i]['annotations']
        # anno = annotations_to_instances(anno, (720, 1280))
        # print(anno)
        # exit()
        
        outputs = predictor(inputs)
        pred_masks = outputs['instances'].pred_masks.cpu().numpy()
        # gt_masks = anno.gt_masks_bit.tensor.cpu().numpy()
        
        gt_mask = imageio.imread(lable_path)
        pred = np.zeros_like(gt_mask)
        for i, mask in enumerate(pred_masks):
            pred[mask > False] = i + 1
            
        metrics, assignments = compute_PRF.multilabel_metrics(pred, gt_mask, return_assign=True)
        metrics_all.append(metrics)
        
        #compute IoU for all instances
        num_inst_mat += len(assignments)
        assign_pred, assign_gt = 0, 0
        assign_overlap = 0
        for gt_id, pred_id in assignments:
            gt_mask = gt_mask == gt_id
            pred_mask = pred == pred_id
            
            assign_gt += np.count_nonzero(gt_mask)
            assign_pred += np.count_nonzero(pred_mask)
            
            mask_overlap = np.logical_and(gt_mask, pred_mask)
            assign_overlap += np.count_nonzero(mask_overlap)
        
        if assign_pred + assign_gt - assign_overlap > 0:
            iou = assign_overlap / (assign_pred+assign_gt-assign_overlap)
        else:
            iou = 0
        iou_masks += iou
        
    miou = iou_masks / len(metrics_all)
    
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num
    
    print('\n')
    print(colored("Visible Metrics for OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
        
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)