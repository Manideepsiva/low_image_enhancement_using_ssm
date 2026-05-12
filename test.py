import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models.retinexmamba import RetinexMamba
from utils.metrics import psnr_numpy, ssim_numpy
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='RetinexMamba — Inference')
    parser.add_argument('--input',   required=True,  help='Path to input image or folder')
    parser.add_argument('--output',  default='./output', help='Output folder')
    parser.add_argument('--weights', required=True,  help='Path to .pth checkpoint')
    parser.add_argument('--gt',      default=None,   help='Ground truth folder (optional, for metrics)')
    return parser.parse_args()


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(weights_path, device):
    model = RetinexMamba(in_channels=3, dim=48,
                         num_blocks=[2, 2, 2], num_refinement=2)
    state = torch.load(weights_path, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── Process single image ──────────────────────────────────────────────────────

@torch.no_grad()
def enhance_image(model, img_path, device):
    img = Image.open(img_path).convert('RGB')
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    enhanced, illum = model(tensor)
    return enhanced.squeeze(0).cpu(), illum.squeeze(0).cpu()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.weights, device)
    os.makedirs(args.output, exist_ok=True)

    # Collect input files
    if os.path.isdir(args.input):
        exts  = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = [f for f in sorted(os.listdir(args.input)) if os.path.splitext(f)[1].lower() in exts]
        paths = [os.path.join(args.input, f) for f in files]
    else:
        paths = [args.input]
        files = [os.path.basename(args.input)]

    psnr_list, ssim_list = [], []

    for fname, fpath in zip(files, paths):
        enhanced, _ = enhance_image(model, fpath, device)
        out_path    = os.path.join(args.output, fname)
        save_image(enhanced, out_path)
        print(f"Saved → {out_path}")

        # Optional metrics against GT
        if args.gt:
            gt_path = os.path.join(args.gt, fname)
            if os.path.exists(gt_path):
                gt  = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0
                enh = enhanced.permute(1, 2, 0).numpy()
                p   = psnr_numpy(enh, gt)
                s   = ssim_numpy(enh, gt)
                psnr_list.append(p)
                ssim_list.append(s)
                print(f"  PSNR: {p:.2f} dB   SSIM: {s:.4f}")

    if psnr_list:
        print(f"\nAverage PSNR : {np.mean(psnr_list):.2f} dB")
        print(f"Average SSIM : {np.mean(ssim_list):.4f}")


if __name__ == '__main__':
    main()
