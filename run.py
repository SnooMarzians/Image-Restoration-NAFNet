import os
import sys
import glob
import time
import argparse
import numpy as np
import torch

# Ensure local submission directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.NAFNet_SR_arch import NAFNetSR

def parse_args():
    parser = argparse.ArgumentParser(description='KLA AI Hackathon - Entry Script (run.py)')
    # Support positional arguments (python run.py <input_dir> <output_dir>)
    parser.add_argument('input_dir_pos', type=str, nargs='?', default=None, help='Input directory containing .npy files')
    parser.add_argument('output_dir_pos', type=str, nargs='?', default=None, help='Output directory to write restored .npy files')
    # Support flag arguments (--input_path and --output_path)
    parser.add_argument('--input_path', type=str, default=None, help='Path to test images directory containing .npy files')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output directory to write restored .npy files')
    parser.add_argument('--weights', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'net_g_swa.pth'), help='Path to trained model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=16, help='Inference batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    
    args = parser.parse_args()

    # Resolve input_dir
    if args.input_dir_pos is not None:
        args.input_dir = args.input_dir_pos
    elif args.input_path is not None:
        args.input_dir = args.input_path
    else:
        parser.error("Input directory is required. Usage: python run.py <input-dir> <output-dir>")

    # Resolve output_dir
    if args.output_dir_pos is not None:
        args.output_dir = args.output_dir_pos
    elif args.output_path is not None:
        args.output_dir = args.output_path
    else:
        parser.error("Output directory is required. Usage: python run.py <input-dir> <output-dir>")

    return args

def load_model(weights_path, device):
    model = NAFNetSR(img_channel=1, width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=1, dec_blk_nums=[2, 2, 2, 2], up_scale=2)
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=True)
        elif isinstance(checkpoint, dict) and 'params_ema' in checkpoint:
            model.load_state_dict(checkpoint['params_ema'], strict=True)
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print(f"[+] Loaded trained weights from: {weights_path}")
    else:
        raise FileNotFoundError(f"[!] Critical Error: Weights file not found at '{weights_path}'")

    model.to(device)
    model.eval()
    return model

def run_evaluation():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    weights_path = args.weights
    batch_size = args.batch_size
    device = args.device

    print(f"[+] Input Directory:  {input_dir}")
    print(f"[+] Output Directory: {output_dir}")
    print(f"[+] Weights Path:     {weights_path}")
    print(f"[+] Using Device:     {device}")

    os.makedirs(output_dir, exist_ok=True)
    model = load_model(weights_path, device)

    # Search for test .npy files
    npy_files = sorted(glob.glob(os.path.join(input_dir, '*.npy')))
    if len(npy_files) == 0:
        # Check subdirectories
        npy_files = sorted(glob.glob(os.path.join(input_dir, '**', '*.npy'), recursive=True))

    if len(npy_files) == 0:
        print(f"[!] Error: No .npy files found in input directory '{input_dir}'")
        return

    print(f"[+] Found {len(npy_files)} .npy files to process.")

    # Warmup GPU
    if device == 'cuda':
        dummy_input = torch.zeros((1, 1, 128, 128), device=device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # Process in mini-batches
    for i in range(0, len(npy_files), batch_size):
        batch_files = npy_files[i:i + batch_size]
        batch_tensors = []

        for filepath in batch_files:
            img_np = np.load(filepath).astype(np.float32)
            # Ensure 4D tensor shape: [1, H, W]
            if img_np.ndim == 2:
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            elif img_np.ndim == 3:
                img_tensor = torch.from_numpy(img_np)
            batch_tensors.append(img_tensor)

        batch_input = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            outputs = model(batch_input)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        outputs_np = outputs.cpu().numpy()

        for j, filepath in enumerate(batch_files):
            out_filename = os.path.basename(filepath)
            out_filepath = os.path.join(output_dir, out_filename)
            
            # Extract 2D array: [1, H_hr, W_hr] -> [H_hr, W_hr]
            out_img = outputs_np[j, 0]
            # Clip and sanitize (ensure no NaN or Inf)
            out_img = np.nan_to_num(np.clip(out_img, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            np.save(out_filepath, out_img)

    if device == 'cuda':
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    num_files = len(npy_files)
    avg_latency_ms = (total_time / num_files) * 1000.0 if num_files > 0 else 0
    fps = num_files / total_time if total_time > 0 else 0

    print("\n" + "="*55)
    print("        END-TO-END INFERENCE LATENCY REPORT        ")
    print("="*55)
    print(f" Total Processed Images        : {num_files}")
    print(f" Total Execution Time          : {total_time:.3f} seconds")
    print(f" Average Latency Per Image     : {avg_latency_ms:.2f} ms / image")
    print(f" Inference Throughput (FPS)    : {fps:.1f} frames / sec")
    print(f" Restored Outputs Saved To     : {output_dir}")
    print("="*55)

if __name__ == '__main__':
    run_evaluation()
