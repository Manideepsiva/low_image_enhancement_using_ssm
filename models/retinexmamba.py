import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion=2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        self.net  = nn.Conv2d(dim, hidden * 2, 1)
        self.act  = nn.GELU()
        self.proj = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        x1, x2 = self.net(x).chunk(2, dim=1)
        return self.proj(self.act(x1) * x2)


class IlluminationEstimator(nn.Module):
    """Per-pixel illumination estimation (Retinex: I = R * L)."""
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1      = nn.Conv2d(n_fea_in, n_fea_middle, 1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, 5, padding=2,
                                    groups=n_fea_middle, bias=True)
        self.conv2      = nn.Conv2d(n_fea_middle, n_fea_out, 1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1, keepdim=True)
        inp = torch.cat([mean_c, img], dim=1)
        return self.conv2(self.depth_conv(self.conv1(inp)))


class MambaBlock(nn.Module):
    """Selective SSM block: flatten → Mamba → reshape → gated FFN."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.ffn   = FeedForward(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_seq = self.norm1(x).flatten(2).transpose(1, 2)
        x_seq = self.mamba(x_seq)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W) + shortcut
        return x + self.ffn(self.norm2(x))


class RetinexMambaBlock(nn.Module):
    """Illumination-guided stack of MambaBlocks."""
    def __init__(self, dim, num_mamba=2):
        super().__init__()
        self.illum_proj   = nn.Sequential(nn.Conv2d(3, dim, 1), nn.GELU())
        self.fusion       = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU())
        self.mamba_blocks = nn.ModuleList([MambaBlock(dim) for _ in range(num_mamba)])
        self.norm         = LayerNorm(dim)

    def forward(self, x, illum):
        illum_feat = self.illum_proj(illum)
        out = self.fusion(torch.cat([x, illum_feat], dim=1))
        for blk in self.mamba_blocks:
            out = blk(out)
        return self.norm(out) + x


class HVIColorCorrection(nn.Module):
    """Decoupled intensity / hue correction to prevent color distortion."""
    def __init__(self, dim):
        super().__init__()
        self.intensity = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.hue       = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.alpha     = nn.Parameter(torch.tensor(0.5))
        self.beta      = nn.Parameter(torch.tensor(0.5))
        self.out       = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        out = torch.sigmoid(self.alpha) * self.intensity(x) + torch.sigmoid(self.beta) * self.hue(x)
        return self.out(out) + x


class RetinexMamba(nn.Module):
    """
    Full RetinexMamba network for low-light image enhancement.
    U-Net encoder-decoder with illumination-guided Mamba blocks.
    """
    def __init__(self, in_channels=3, dim=48, num_blocks=[2,2,2], num_refinement=2):
        super().__init__()

        self.shallow_extract  = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        )
        self.illum_estimator  = IlluminationEstimator(n_fea_middle=dim, n_fea_in=4, n_fea_out=3)

        # Encoder
        self.encoder_l1 = nn.ModuleList([RetinexMambaBlock(dim,     num_mamba=num_blocks[0])])
        self.down1      = nn.Conv2d(dim,     dim*2, 2, stride=2)
        self.encoder_l2 = nn.ModuleList([RetinexMambaBlock(dim*2,   num_mamba=num_blocks[1])])
        self.down2      = nn.Conv2d(dim*2,   dim*4, 2, stride=2)

        # Bottleneck
        self.bottleneck = nn.ModuleList([RetinexMambaBlock(dim*4,   num_mamba=num_blocks[2])])

        # Decoder
        self.up2        = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.skip_attn2 = nn.Conv2d(dim*4, dim*2, 1)
        self.decoder_l2 = nn.ModuleList([RetinexMambaBlock(dim*2,   num_mamba=num_blocks[1])])
        self.up1        = nn.ConvTranspose2d(dim*2, dim,   2, stride=2)
        self.skip_attn1 = nn.Conv2d(dim*2, dim, 1)
        self.decoder_l1 = nn.ModuleList([RetinexMambaBlock(dim,     num_mamba=num_blocks[0])])

        # Refinement + output
        self.refinement       = nn.ModuleList([RetinexMambaBlock(dim, num_mamba=num_refinement)])
        self.color_correction = HVIColorCorrection(dim)
        self.output           = nn.Conv2d(dim, in_channels, 3, padding=1, bias=False)

    def forward(self, inp):
        illum_map = self.illum_estimator(inp)
        x = self.shallow_extract(inp)

        enc1 = x
        for blk in self.encoder_l1: enc1 = blk(enc1, illum_map)

        illum2 = F.interpolate(illum_map, scale_factor=0.5,  mode='bilinear', align_corners=False)
        enc2   = self.down1(enc1)
        for blk in self.encoder_l2: enc2 = blk(enc2, illum2)

        illum4 = F.interpolate(illum_map, scale_factor=0.25, mode='bilinear', align_corners=False)
        btn    = self.down2(enc2)
        for blk in self.bottleneck: btn = blk(btn, illum4)

        d2 = self.skip_attn2(torch.cat([self.up2(btn), enc2], dim=1))
        for blk in self.decoder_l2: d2 = blk(d2, illum2)

        d1 = self.skip_attn1(torch.cat([self.up1(d2), enc1], dim=1))
        for blk in self.decoder_l1: d1 = blk(d1, illum_map)

        for blk in self.refinement: d1 = blk(d1, illum_map)

        d1  = self.color_correction(d1)
        out = self.output(d1)
        return torch.clamp(out + inp, 0, 1), illum_map
