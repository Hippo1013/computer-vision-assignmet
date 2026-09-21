import argparse
import cv2
import glob
import json
import matplotlib
import numpy as np
import os
import time
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def synchronize_device(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='also save raw depth as *_raw_depth_meter.npy; <image_stem>.npy is always saved in meters')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    inference_timings = []
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        # Time the complete infer_image call, excluding file reads and writes.
        synchronize_device(DEVICE)
        start_time = time.perf_counter()
        depth = depth_anything.infer_image(raw_image, args.input_size)
        synchronize_device(DEVICE)
        elapsed_seconds = time.perf_counter() - start_time
        inference_timings.append({'filename': filename, 'seconds': elapsed_seconds})
        print(f'Inference time: {elapsed_seconds:.4f} s')

        # Save metric depth in meters before normalization for visualization.
        np.save(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.npy'), depth)
        
        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(output_path, combined_result)

    total_seconds = sum(item['seconds'] for item in inference_timings)
    average_seconds = total_seconds / len(inference_timings) if inference_timings else None
    timing_path = os.path.join(args.outdir, 'inference_timing.json')
    with open(timing_path, 'w', encoding='utf-8') as f:
        json.dump({
            'device': DEVICE,
            'encoder': args.encoder,
            'input_size': args.input_size,
            'load_from': args.load_from,
            'max_depth': args.max_depth,
            'depth_unit': 'meters',
            'scope': 'infer_image including preprocessing, inference, and conversion to numpy; excluding file I/O',
            'warmup_runs': 0,
            'image_count': len(inference_timings),
            'total_seconds': total_seconds,
            'average_seconds': average_seconds,
            'images': inference_timings,
        }, f, ensure_ascii=False, indent=2)
        f.write('\n')
    if inference_timings:
        print(f'Total inference time: {total_seconds:.4f} s; average: {average_seconds:.4f} s/image (includes first run, no warmup)')
    print(f'Inference timing saved to: {timing_path}')
