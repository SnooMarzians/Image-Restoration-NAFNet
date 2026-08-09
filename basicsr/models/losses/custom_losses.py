import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basicsr.models.losses.losses import PSNRLoss

def gaussian_window(window_size, sigma):
    gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)]))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    """Differentiable SSIM Loss (1 - SSIM)."""
    def __init__(self, window_size=11, size_average=True, loss_weight=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.loss_weight = loss_weight
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel

        ssim_val = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return self.loss_weight * (1.0 - ssim_val)

class CombinedPSNRSSIMLoss(nn.Module):
    """Combined PSNR Loss + SSIM Loss for optimizing pixel accuracy AND structural sharpness."""
    def __init__(self, loss_weight=1.0, ssim_weight=0.5, reduction='mean'):
        super(CombinedPSNRSSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ssim_weight = ssim_weight
        self.psnr_loss = PSNRLoss(loss_weight=1.0, reduction=reduction)
        self.ssim_loss = SSIMLoss(loss_weight=1.0)

    def forward(self, pred, target, **kwargs):
        l_psnr = self.psnr_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        total_loss = l_psnr + self.ssim_weight * l_ssim
        return self.loss_weight * total_loss

class SobelL1Loss(nn.Module):
    """Sobel Gradient + L1 Loss for sharp edge enhancement during fine-tuning."""
    def __init__(self, loss_weight=1.0, sobel_weight=0.2, reduction='mean'):
        super(SobelL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.sobel_weight = sobel_weight

        # 2D Sobel Filters for horizontal and vertical spatial derivatives
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target, **kwargs):
        l1_loss = F.l1_loss(pred, target)

        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        total_loss = l1_loss + self.sobel_weight * grad_loss
        return self.loss_weight * total_loss
