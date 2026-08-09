import os
import glob
import time
import argparse
import numpy as np
import torch
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.models.archs.NAFNet_SR_arch import NAFNetSR

def parse_args():
    parser = argparse.ArgumentParser(description='NAFNet Metrics Benchmarking Script for KLA AI Hackathon')
    parser.add_argument('--gt_dir', type=str, default='../dataset/train/train/GT', help='Path to Ground Truth directory containing .npy files')
    parser.add_argument('--lq_dir', type=str, default='../dataset/train/train/NoisyLR', help='Path to Low-Quality / Noisy directory containing .npy files')
    parser.add_argument('--weights', type=str, default='models/net_g_swa.pth', help='Path to trained model weights (.pth)')
    parser.add_argument('--num_samples', type=int, default=3200, help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=16, help='Inference batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    return parser.parse_args()

def evaluate_metrics(gt_dir, lq_dir, weights_path, num_samples=3200, batch_size=16, device='cuda'):
    if not os.path.exists(weights_path):
        print(f"[!] Error: Weights path '{weights_path}' not found.")
        return

    # Load model
    model = NAFNetSR(img_channel=1, width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=1, dec_blk_nums=[2, 2, 2, 2], up_scale=2)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    elif isinstance(checkpoint, dict) and 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.to(device)
    model.eval()

    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))[:num_samples]
    lq_files = sorted(glob.glob(os.path.join(lq_dir, '*.npy')))[:num_samples]

    if len(gt_files) == 0 or len(lq_files) == 0:
        print(f"[!] Error: Ground truth or Low Quality directory does not contain .npy files.")
        return

    num_files = min(len(gt_files), len(lq_files))
    psnr_scores = []
    ssim_scores = []

    print(f"[+] Computing PSNR & SSIM metrics across {num_files} sample pairs...")
    start_time = time.perf_counter()

    for i in range(0, num_files, batch_size):
        batch_lq_files = lq_files[i:i + batch_size]
        batch_gt_files = gt_files[i:i + batch_size]

        batch_lq_tensors = []
        batch_gt_arrays = []

        for lq_path, gt_path in zip(batch_lq_files, batch_gt_files):
            lq_arr = np.load(lq_path).astype(np.float32)
            gt_arr = np.load(gt_path).astype(np.float32)

            if lq_arr.ndim == 2:
                lq_tensor = torch.from_numpy(lq_arr).unsqueeze(0)
            else:
                lq_tensor = torch.from_numpy(lq_arr)

            batch_lq_tensors.append(lq_tensor)
            batch_gt_arrays.append(gt_arr)

        batch_input = torch.stack(batch_lq_tensors, dim=0).to(device)

        with torch.no_grad():
            outputs = model(batch_input)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        outputs_np = outputs.cpu().numpy()

        for j in range(len(batch_lq_files)):
            gt = batch_gt_arrays[j]
            restored = outputs_np[j, 0]
            restored = np.clip(restored, 0.0, 1.0)

            gt_3d = gt[:, :, np.newaxis]
            restored_3d = restored[:, :, np.newaxis]

            psnr_val = calculate_psnr(gt_3d, restored_3d, crop_border=0, input_order='HWC')
            ssim_val = calculate_ssim(gt_3d, restored_3d, crop_border=0, input_order='HWC')

            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

    if device == 'cuda':
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / num_files) * 1000.0 if num_files > 0 else 0
    fps = num_files / total_time if total_time > 0 else 0

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print("\n" + "="*55)
    print("         ACCURACY & LATENCY BENCHMARK REPORT         ")
    print("="*55)
    print(f" Target Checkpoint Weights                   : {weights_path}")
    print(f" Average PSNR (Peak Signal-to-Noise Ratio) : {avg_psnr:.4f} dB")
    print(f" Average SSIM (Structural Similarity Index)  : {avg_ssim:.4f}")
    print(f" Total Processed Images                     : {num_files}")
    print(f" Total Execution Time                       : {total_time:.3f} seconds")
    print(f" Average Latency Per Image                  : {avg_latency_ms:.2f} ms / image")
    print(f" Inference Throughput (FPS)                 : {fps:.1f} frames / sec")
    print("="*55)

if __name__ == '__main__':
    args = parse_args()
    evaluate_metrics(
        gt_dir=args.gt_dir,
        lq_dir=args.lq_dir,
        weights_path=args.weights,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
