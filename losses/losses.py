import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexMambaLoss(nn.Module):
    """
    Total = L1 + 0.3*SSIM + 0.05*FFT + 0.01*IllumTV
    """
    def __init__(self, w_l1=1.0, w_ssim=0.3, w_fft=0.05, w_illum=0.01):
        super().__init__()
        self.w_l1=w_l1; self.w_ssim=w_ssim; self.w_fft=w_fft; self.w_illum=w_illum

    def l1_loss(self, p, t):
        return F.l1_loss(p, t)

    def ssim_loss(self, p, t):
        mu_p=F.avg_pool2d(p,3,1,1); mu_t=F.avg_pool2d(t,3,1,1)
        sp=F.avg_pool2d(p*p,3,1,1)-mu_p**2; st=F.avg_pool2d(t*t,3,1,1)-mu_t**2
        spt=F.avg_pool2d(p*t,3,1,1)-mu_p*mu_t
        C1,C2=0.01**2,0.03**2
        num=(2*mu_p*mu_t+C1)*(2*spt+C2); den=(mu_p**2+mu_t**2+C1)*(sp+st+C2)
        return 1-(num/den).mean()

    def fft_loss(self, p, t):
        return F.l1_loss(torch.abs(torch.fft.fft2(p)), torch.abs(torch.fft.fft2(t)))

    def illum_smooth_loss(self, L):
        dx=L[:,:,:,1:]-L[:,:,:,:-1]; dy=L[:,:,1:,:]-L[:,:,:-1,:]
        return dx.abs().mean()+dy.abs().mean()

    def forward(self, pred, target, illum):
        l1=self.l1_loss(pred,target); ssim=self.ssim_loss(pred,target)
        fft=self.fft_loss(pred,target); ill=self.illum_smooth_loss(illum)
        total=self.w_l1*l1+self.w_ssim*ssim+self.w_fft*fft+self.w_illum*ill
        return total, l1, ssim, fft
