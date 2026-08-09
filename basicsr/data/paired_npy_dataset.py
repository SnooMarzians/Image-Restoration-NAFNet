import os
import numpy as np
import torch
import random
from torch.utils import data as data
from basicsr.utils import get_root_logger

class PairedNpyDataset(data.Dataset):
    """Paired .npy dataset for single-channel (grayscale) image restoration & super-resolution.

    Reads LQ (Low Quality / NoisyLR) and GT (Ground Truth) .npy files.

    Args:
        opt (dict): Config for dataset.
            dataroot_gt (str): Data root path for gt (.npy files).
            dataroot_lq (str): Data root path for lq (.npy files).
            use_flip (bool): Use horizontal/vertical flips.
            use_rot (bool): Use 90-degree rotations.
            scale (int): Upscaling factor (e.g. 2).
    """

    def __init__(self, opt):
        super(PairedNpyDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.use_flip = opt.get('use_flip', False)
        self.use_rot = opt.get('use_rot', False)
        self.scale = opt.get('scale', 2)

        # Get sorted list of .npy files
        self.gt_filenames = sorted([f for f in os.listdir(self.gt_folder) if f.endswith('.npy')])
        self.lq_filenames = sorted([f for f in os.listdir(self.lq_folder) if f.endswith('.npy')])

        assert len(self.gt_filenames) == len(self.lq_filenames), (
            f'GT count ({len(self.gt_filenames)}) and LQ count ({len(self.lq_filenames)}) do not match!'
        )

        logger = get_root_logger()
        logger.info(f'PairedNpyDataset initialized with {len(self.gt_filenames)} pairs.')

    def __len__(self):
        return len(self.gt_filenames)

    def __getitem__(self, index):
        gt_filename = self.gt_filenames[index]
        lq_filename = self.lq_filenames[index]

        gt_path = os.path.join(self.gt_folder, gt_filename)
        lq_path = os.path.join(self.lq_folder, lq_filename)

        # Load numpy arrays
        gt_img = np.load(gt_path).astype(np.float32)  # Shape (H_gt, W_gt)
        lq_img = np.load(lq_path).astype(np.float32)  # Shape (H_lq, W_lq)

        # Data augmentation (flip & rotation)
        if self.opt.get('phase') == 'train':
            gt_img, lq_img = self.augment_pair(gt_img, lq_img, self.use_flip, self.use_rot)

        # Add channel dimension: (H, W) -> (1, H, W)
        if gt_img.ndim == 2:
            gt_img = gt_img[np.newaxis, :, :]
        if lq_img.ndim == 2:
            lq_img = lq_img[np.newaxis, :, :]

        # Convert to PyTorch tensors
        tensor_gt = torch.from_numpy(gt_img)
        tensor_lq = torch.from_numpy(lq_img)

        return {
            'lq': tensor_lq,
            'gt': tensor_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    @staticmethod
    def augment_pair(gt, lq, use_flip=True, use_rot=True):
        """Augment GT and LQ image pairs synchronously."""
        hflip = use_flip and random.random() < 0.5
        vflip = use_flip and random.random() < 0.5
        rot90 = use_rot and random.random() < 0.5

        if hflip:
            gt = np.ascontiguousarray(np.fliplr(gt))
            lq = np.ascontiguousarray(np.fliplr(lq))
        if vflip:
            gt = np.ascontiguousarray(np.flipud(gt))
            lq = np.ascontiguousarray(np.flipud(lq))
        if rot90:
            gt = np.ascontiguousarray(np.rot90(gt))
            lq = np.ascontiguousarray(np.rot90(lq))

        # Synthetic speckle noise augmentation for out-of-distribution generalization
        if random.random() < 0.3:
            noise_std = random.uniform(0.01, 0.03)
            speckle = np.random.normal(0, noise_std, size=lq.shape).astype(np.float32)
            lq = np.ascontiguousarray(lq * (1.0 + speckle))

        return gt, lq
