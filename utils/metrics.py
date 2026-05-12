import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


def psnr_tensor(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)

def ssim_tensor(pred, target):
    mu_p=F.avg_pool2d(pred,3,1,1); mu_t=F.avg_pool2d(target,3,1,1)
    sp=F.avg_pool2d(pred*pred,3,1,1)-mu_p**2; st=F.avg_pool2d(target*target,3,1,1)-mu_t**2
    spt=F.avg_pool2d(pred*target,3,1,1)-mu_p*mu_t; C1,C2=0.01**2,0.03**2
    return ((2*mu_p*mu_t+C1)*(2*spt+C2)/((mu_p**2+mu_t**2+C1)*(sp+st+C2))).mean().item()

def psnr_numpy(pred, target):
    return sk_psnr(target, pred, data_range=1.0)

def ssim_numpy(pred, target):
    return sk_ssim(target, pred, channel_axis=-1, data_range=1.0)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnr_vals, ssim_vals = [], []
    for low, high in loader:
        pred, _ = model(low.to(device))
        pred = pred.clamp(0,1)
        p = pred[0].cpu().permute(1,2,0).numpy()
        t = high[0].cpu().permute(1,2,0).numpy()
        psnr_vals.append(psnr_numpy(p, t))
        ssim_vals.append(ssim_numpy(p, t))
    return {'psnr': float(np.mean(psnr_vals)), 'ssim': float(np.mean(ssim_vals))}
