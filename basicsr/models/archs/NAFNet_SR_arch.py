import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.NAFNet_arch import NAFBlock, LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

class NAFNetSR(nn.Module):
    """NAFNet model adapted for single-channel (grayscale) restoration and 2x Super-Resolution.

    Args:
        img_channel (int): Number of input/output image channels. Default: 1.
        width (int): Base feature width. Default: 32.
        middle_blk_num (int): Number of NAFBlocks in middle section. Default: 1.
        enc_blk_nums (list): Block numbers for encoder levels. Default: [2, 2, 4, 8].
        dec_blk_nums (list): Block numbers for decoder levels. Default: [2, 2, 2, 2].
        up_scale (int): Upscaling factor. Default: 2.
    """

    def __init__(self, img_channel=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2], up_scale=2):
        super(NAFNetSR, self).__init__()

        self.up_scale = up_scale
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        # Upsampling tail for Super-Resolution
        self.up_head = nn.Sequential(
            nn.Conv2d(width, img_channel * (up_scale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(up_scale)
        )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)

        # Base upsampled residual connection
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear', align_corners=False)

        x = self.intro(inp_padded)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # Upsample feature maps to target resolution
        out = self.up_head(x)

        # Crop back to exact target HR shape if padded
        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        
        # Add skip connection from bilinear upsampled input
        out = out + inp_hr
        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x

class NAFNetSROfficial(Local_Base, NAFNetSR):
    def __init__(self, *args, train_size=(1, 1, 128, 128), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
