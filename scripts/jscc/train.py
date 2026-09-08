#!/usr/bin/env python3
"""Train a small DeepJSCC model on CIFAR-10 (CPU-friendly).

Usage:
    python train.py [--epochs N] [--channel AWGN|Rayleigh] [--snr-db X] [--inner-channels C]
"""
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from cifar_dataset import HFCifar10
from model import DeepJSCC

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
CKPT_DIR = SCRIPT_DIR / "checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh"])
    ap.add_argument("--snr-db", type=float, default=10.0)
    ap.add_argument("--inner-channels", type=int, default=8, help="c: gives ratio=c*64/3072")
    ap.add_argument("--out", default=None, help="checkpoint output path")
    ap.add_argument("--init-checkpoint", default=None, help="load these weights as the starting point "
                     "instead of training from scratch (e.g. continue training an existing checkpoint "
                     "with more epochs)")
    ap.add_argument("--seed", type=int, default=42, help="for reproducible runs; also avoids two runs "
                     "with identical hyperparameters (-> identical default --out path) silently "
                     "diverging from different random initializations")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else CKPT_DIR / f"cifar10_c{args.inner_channels}_{args.channel.lower()}{int(args.snr_db)}.pth"

    transform = transforms.ToTensor()
    train_set = HFCifar10(root=DATA_DIR, split="train", transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepJSCC(c=args.inner_channels, channel_type=args.channel, snr_db=args.snr_db).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    epochs_done = 0
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        if ckpt.get("c") != args.inner_channels or ckpt.get("channel") != args.channel:
            print(f"WARNING: --init-checkpoint was trained with c={ckpt.get('c')} channel={ckpt.get('channel')}, "
                  f"but this run uses c={args.inner_channels} channel={args.channel} - loading anyway, "
                  f"but shapes may not match", file=__import__("sys").stderr)
        model.load_state_dict(ckpt["state_dict"])
        epochs_done = ckpt.get("epochs_done", 0)
        print(f"loaded initial weights from {args.init_checkpoint} ({epochs_done} epoch(s) already trained)")

    print(f"training DeepJSCC c={args.inner_channels} channel={args.channel} snr_db={args.snr_db} "
          f"for {args.epochs} more epochs (epoch {epochs_done+1}..{epochs_done+args.epochs} overall) "
          f"on {len(train_set)} images, batch_size={args.batch_size}, device={device}")

    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        n = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            opt.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        mse = total_loss / n
        psnr = 10 * torch.log10(torch.tensor(1.0 / mse)) if mse > 0 else float("inf")
        total_epochs = epochs_done + epoch + 1
        print(f"epoch {epoch+1}/{args.epochs} (overall {total_epochs}): mse={mse:.5f} psnr={psnr:.2f}dB "
              f"elapsed={time.time()-t0:.1f}s")
        torch.save({"state_dict": model.state_dict(), "c": args.inner_channels,
                    "channel": args.channel, "snr_db": args.snr_db, "epochs_done": total_epochs}, out_path)

    print(f"saved checkpoint to {out_path} ({epochs_done + args.epochs} epochs total)")


if __name__ == "__main__":
    main()
