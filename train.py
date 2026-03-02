"""
Training script for Super Resolution models
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import SRCNN, ESRGAN, EDSR, RRDBNet
from data import create_dataloader
from losses import PerceptualLoss, GANLoss, RelativisticGANLoss
from evaluate import calculate_psnr, calculate_ssim
from utils import get_device, count_parameters, AverageMeter, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Train Super Resolution Model")
    
    # Data arguments
    parser.add_argument("--train-hr", type=str, required=True,
                        help="Path to training HR images")
    parser.add_argument("--train-lr", type=str, default=None,
                        help="Path to training LR images (optional)")
    parser.add_argument("--val-hr", type=str, default=None,
                        help="Path to validation HR images")
    parser.add_argument("--val-lr", type=str, default=None,
                        help="Path to validation LR images")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="esrgan",
                        choices=["srcnn", "esrgan", "edsr"],
                        help="Model architecture")
    parser.add_argument("--scale", type=int, default=4,
                        choices=[2, 4, 8],
                        help="Upscaling factor")
    parser.add_argument("--num-blocks", type=int, default=16,
                        help="Number of residual blocks")
    parser.add_argument("--num-features", type=int, default=64,
                        help="Number of feature channels")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--patch-size", type=int, default=96,
                        help="HR patch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lr-decay-step", type=int, default=50,
                        help="LR decay every N epochs")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5,
                        help="LR decay factor")
    
    # Loss arguments
    parser.add_argument("--pixel-weight", type=float, default=1.0,
                        help="Weight for pixel loss")
    parser.add_argument("--perceptual-weight", type=float, default=1.0,
                        help="Weight for perceptual loss")
    parser.add_argument("--gan-weight", type=float, default=0.005,
                        help="Weight for GAN loss (ESRGAN only)")
    parser.add_argument("--use-gan", action="store_true",
                        help="Use adversarial training (ESRGAN only)")
    
    # Other arguments
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N iterations")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    return parser.parse_args()


def build_model(args, device):
    """Build model based on arguments"""
    if args.model == "srcnn":
        model = SRCNN(scale_factor=args.scale)
    elif args.model == "esrgan":
        model = ESRGAN(
            scale_factor=args.scale,
            num_features=args.num_features,
            num_blocks=args.num_blocks,
            use_discriminator=args.use_gan
        )
    elif args.model == "edsr":
        model = EDSR(
            scale_factor=args.scale,
            num_features=args.num_features,
            num_blocks=args.num_blocks
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Model: {args.model.upper()}")
    print(f"Parameters: {count_parameters(model):,}")
    
    return model


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    args,
    writer,
    discriminator=None,
    d_optimizer=None,
    gan_criterion=None
):
    """Train for one epoch"""
    model.train()
    if discriminator:
        discriminator.train()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for i, (lr_imgs, hr_imgs) in enumerate(pbar):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        # Forward pass
        if hasattr(model, 'generator'):
            sr_imgs = model.generator(lr_imgs)
        else:
            sr_imgs = model(lr_imgs)
        
        # Generator/Model loss
        total_loss, pixel_loss, perceptual_loss = criterion(sr_imgs, hr_imgs)
        
        # Add GAN loss if using adversarial training
        if discriminator and gan_criterion:
            # Train discriminator
            d_optimizer.zero_grad()
            
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs.detach())
            
            d_loss = gan_criterion(real_pred, fake_pred, for_discriminator=True)
            d_loss.backward()
            d_optimizer.step()
            
            # Generator adversarial loss
            fake_pred = discriminator(sr_imgs)
            real_pred = discriminator(hr_imgs).detach()
            g_gan_loss = gan_criterion(real_pred, fake_pred, for_discriminator=False)
            total_loss = total_loss + args.gan_weight * g_gan_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Calculate PSNR
        with torch.no_grad():
            psnr = calculate_psnr(sr_imgs, hr_imgs)
        
        # Update meters
        loss_meter.update(total_loss.item(), lr_imgs.size(0))
        psnr_meter.update(psnr, lr_imgs.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'psnr': f'{psnr_meter.avg:.2f}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + i
        if i % args.log_interval == 0:
            writer.add_scalar('train/loss', loss_meter.avg, global_step)
            writer.add_scalar('train/psnr', psnr_meter.avg, global_step)
    
    return loss_meter.avg, psnr_meter.avg


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(val_loader, desc="Validating"):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            if hasattr(model, 'generator'):
                sr_imgs = model.generator(lr_imgs)
            else:
                sr_imgs = model(lr_imgs)
            
            total_loss, _, _ = criterion(sr_imgs, hr_imgs)
            psnr = calculate_psnr(sr_imgs, hr_imgs)
            ssim = calculate_ssim(sr_imgs, hr_imgs)
            
            loss_meter.update(total_loss.item(), lr_imgs.size(0))
            psnr_meter.update(psnr, lr_imgs.size(0))
            ssim_meter.update(ssim, lr_imgs.size(0))
    
    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


def main():
    args = parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Build model
    model = build_model(args, device)
    
    # Setup discriminator for ESRGAN
    discriminator = None
    d_optimizer = None
    gan_criterion = None
    
    if args.model == "esrgan" and args.use_gan:
        if hasattr(model, 'discriminator') and model.discriminator is not None:
            discriminator = model.discriminator.to(device)
            d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
            gan_criterion = RelativisticGANLoss()
            print("Using adversarial training with discriminator")
    
    # Build dataloaders
    train_loader = create_dataloader(
        hr_dir=args.train_hr,
        lr_dir=args.train_lr,
        batch_size=args.batch_size,
        scale_factor=args.scale,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        mode="train"
    )
    
    val_loader = None
    if args.val_hr:
        val_loader = create_dataloader(
            hr_dir=args.val_hr,
            lr_dir=args.val_lr,
            batch_size=args.batch_size,
            scale_factor=args.scale,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
            mode="val"
        )
    
    # Loss function
    criterion = PerceptualLoss(
        pixel_weight=args.pixel_weight,
        perceptual_weight=args.perceptual_weight
    ).to(device)
    
    # Optimizer and scheduler
    if hasattr(model, 'generator'):
        params = model.generator.parameters()
    else:
        params = model.parameters()
    
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_decay_step, 
        gamma=args.lr_decay_gamma
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_psnr = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args, writer,
            discriminator, d_optimizer, gan_criterion
        )
        
        scheduler.step()
        
        # Validate
        if val_loader:
            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/psnr', val_psnr, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val PSNR={val_psnr:.2f}, Val SSIM={val_ssim:.4f}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"New best model saved with PSNR={best_psnr:.2f}")
        else:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train PSNR={train_psnr:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print(f"Training complete. Best PSNR: {best_psnr:.2f}")
    writer.close()


if __name__ == "__main__":
    main()
