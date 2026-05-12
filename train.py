import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.retinexmamba import RetinexMamba
from losses.losses import RetinexMambaLoss
from data.dataset import LOLDataset
from utils.metrics import evaluate

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_LOW    = './data/lol/our485/low'
TRAIN_HIGH   = './data/lol/our485/high'
TEST_LOW     = './data/lol/eval15/low'
TEST_HIGH    = './data/lol/eval15/high'

SAVE_DIR     = './checkpoints'
LOG_EVERY    = 10     # print loss every N epochs
SAVE_EVERY   = 50     # save checkpoint every N epochs
EPOCHS       = 200
BATCH_SIZE   = 8
PATCH_SIZE   = 128
LR           = 2e-4
NUM_WORKERS  = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Build model, loss, optimizer ─────────────────────────────────────────────

def build_model():
    return RetinexMamba(
        in_channels=3,
        dim=48,
        num_blocks=[2, 2, 2],
        num_refinement=2
    ).to(DEVICE)


def build_optimizer(model):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    return optimizer, scheduler


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Data
    train_set    = LOLDataset(TRAIN_LOW, TRAIN_HIGH, patch=PATCH_SIZE, is_train=True)
    test_set     = LOLDataset(TEST_LOW,  TEST_HIGH,  patch=PATCH_SIZE, is_train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model / loss / optimizer
    model     = build_model()
    criterion = RetinexMambaLoss()
    optimizer, scheduler = build_optimizer(model)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'='*55}")
    print(f"  RetinexMamba  —  Training")
    print(f"{'='*55}")
    print(f"  Parameters   : {params:.2f}M")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Device       : {DEVICE}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"{'='*55}\n")

    best_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for low, high in train_loader:
            low  = low.to(DEVICE)
            high = high.to(DEVICE)

            optimizer.zero_grad()
            pred, illum = model(low)
            loss, l1, ssim, fft = criterion(pred, high, illum)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch % LOG_EVERY == 0 or epoch == 1:
            metrics = evaluate(model, test_loader, DEVICE)
            print(
                f"Epoch [{epoch:3d}/{EPOCHS}]  "
                f"Loss: {avg_loss:.4f}  "
                f"PSNR: {metrics['psnr']:.2f} dB  "
                f"SSIM: {metrics['ssim']:.4f}  "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            # Save best checkpoint
            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))
                print(f"  ↳ New best saved  (PSNR={best_psnr:.2f} dB)")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f'epoch_{epoch:03d}.pth')
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'loss':       avg_loss,
            }, ckpt_path)
            print(f"  ↳ Checkpoint saved → {ckpt_path}")

    print(f"\nTraining complete ✅   Best PSNR: {best_psnr:.2f} dB")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train()
