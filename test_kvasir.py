# Test script for Kvasir dataset
import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_image_kvasir_mean
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_Kvasir import Kvasir_dataset
import pandas as pd
import cv2

def inference(args, multimask_output, model, test_save_path=None):
    # Create test dataset
    db_test = Kvasir_dataset(base_dir=args.root_path, split="val")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Testing on {len(testloader)} images")
    
    metric_list = []
    case_names = []
    
    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        image = sampled_batch["image"]  # Keep as tensor (1, C, H, W)
        label = sampled_batch["label"].squeeze(0).cpu().numpy()  # (H, W)
        case_name = sampled_batch["case_name"][0]
        
        # Test single image
        metric_i = test_single_image_kvasir_mean(
            image, label, model, 
            classes=args.num_classes, 
            patch_size=[args.img_size, args.img_size], 
            input_size=[args.input_size, args.input_size]
        )
        
        metric_list.append(metric_i[0])  # Only one class (polyp)
        case_names.append(case_name)
    
    # Convert to numpy array
    metric_array = np.array(metric_list)
    
    # Calculate mean metrics (only dice and hd95 are returned)
    avg_dice = np.nanmean(metric_array[:, 0])
    avg_hd95 = np.nanmean(metric_array[:, 1])
    
    print(f"\n{'='*60}")
    print(f"Kvasir Test Results")
    print(f"{'='*60}")
    print(f"Average Dice:    {avg_dice:.4f}")
    print(f"Average HD95:    {avg_hd95:.4f}")
    print(f"{'='*60}\n")
    
    # Save results to CSV
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    write_csv = os.path.join(args.output_dir, f"{args.exp}_test_kvasir.csv")
    
    results_df = pd.DataFrame({
        'case': case_names,
        'dice': metric_array[:, 0],
        'hd95': metric_array[:, 1]
    })
    results_df.to_csv(write_csv, index=False, sep=',')
    print(f"Saved results to {write_csv}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f"{args.exp}_test_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Kvasir Test Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Average Dice:    {avg_dice:.4f} ± {np.nanstd(metric_array[:, 0]):.4f}\n")
        f.write(f"Average HD95:    {avg_hd95:.4f} ± {np.nanstd(metric_array[:, 1]):.4f}\n")
        f.write(f"Average ASD:     {avg_asd:.4f} ± {np.nanstd(metric_array[:, 2]):.4f}\n")
        f.write(f"Average Jaccard: {avg_jc:.4f} ± {np.nanstd(metric_array[:, 3]):.4f}\n")
    print(f"Saved summary to {summary_path}")
    
    logging.info("Testing Finished!")
    return avg_dice, avg_hd95
def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='C:/ai-agent/data/Kvasir', help='Root directory of dataset')
    parser.add_argument('--config', type=str, default=None, help='Config file from trained model')
    parser.add_argument('--dataset', type=str, default='Kvasir', help='Dataset name')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--output_dir', type=str, default='C:/ai-agent/CPC-SAM-main/output/test_results/')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--input_size', type=int, default=224, help='SAM input size')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--is_savenii', action='store_true', default=False, help='Save results')
    parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
    parser.add_argument('--ckpt', type=str, 
                        default='C:/ai-agent/CPC-SAM-main/checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained SAM checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default=None, 
                        help='Finetuned LoRA checkpoint (provide trained model path)')
    parser.add_argument('--vit_name', type=str, 
                        default='vit_b_dualmask_same_prompt_class_random_large', 
                        help='ViT model name')
    parser.add_argument('--rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_prompt')
    parser.add_argument('--exp', type=str, default='kvasir_test', help='Experiment name')
    parser.add_argument('--promptmode', type=str, default='point', help='Prompt mode')
    
    args = parser.parse_args()
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Register SAM model
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1]
    )
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()
    
    # Load trained model
    if args.lora_ckpt is not None:
        print(f"Loading trained model from {args.lora_ckpt}")
        net.load_lora_parameters(args.lora_ckpt)
    else:
        print("Warning: No trained model provided! Testing with pretrained SAM only.")
    
    multimask_output = True
    
    # Create save directory
    test_save_path = None
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    
    # Run inference
    inference(args, multimask_output, net, test_save_path)
