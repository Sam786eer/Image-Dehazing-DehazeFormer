import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='dehazeformer-b', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--no_autocast', action='store_false', default=True)
parser.add_argument('--save_dir', default='./saved_models/', type=str)
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--log_dir', default='./logs/', type=str)
parser.add_argument('--dataset', default='RESIDE-IN', type=str)
parser.add_argument('--exp', default='reside6k', type=str)
parser.add_argument('--gpu', default='0', type=str)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# ---------------- TRAIN ----------------
def train(train_loader, network, criterion, optimizer, scaler):

    losses = AverageMeter()
    network.train()

    for batch in train_loader:

        source_img = batch['source'].cuda(non_blocking=True)
        target_img = batch['target'].cuda(non_blocking=True)

        with autocast(enabled=args.no_autocast):
            output = network(source_img)
            loss = criterion(output, target_img)

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), source_img.size(0))

    return losses.avg


# ---------------- VALID ----------------
def valid(val_loader, network):

    PSNR = AverageMeter()
    network.eval()

    with torch.no_grad():

        for batch in val_loader:

            source_img = batch['source'].cuda(non_blocking=True)
            target_img = batch['target'].cuda(non_blocking=True)

            output = network(source_img).clamp_(-1, 1)

            mse_loss = F.mse_loss(
                output * 0.5 + 0.5,
                target_img * 0.5 + 0.5,
                reduction='none'
            ).mean((1, 2, 3))

            psnr = 10 * torch.log10(1 / mse_loss).mean()

            PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


# ---------------- MAIN ----------------
if __name__ == '__main__':

    # Load config
    setting_file = os.path.join('configs', args.exp, args.model + '.json')

    if not os.path.exists(setting_file):
        setting_file = os.path.join('configs', args.exp, 'default.json')

    with open(setting_file, 'r') as f:
        setting = json.load(f)

    print("Creating model:", args.model)

    network = eval(args.model.replace('-', '_'))()

    # ---------------- LOAD PRETRAINED ----------------
    pretrained_path = os.path.join('pretrained_models', args.model + '.pth')

    if os.path.exists(pretrained_path):

        print("==> Loading pretrained weights from:", pretrained_path)

        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        new_state_dict = {}

        for k, v in checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        network.load_state_dict(new_state_dict, strict=False)

        print("==> Pretrained weights loaded successfully!")

    else:
        print("WARNING: pretrained weights not found")

    network = nn.DataParallel(network).cuda()

    # ---------------- LOSS ----------------
    criterion = nn.L1Loss()

    # ---------------- OPTIMIZER ----------------
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])

    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])

    else:
        raise Exception("Unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=setting['epochs'],
        eta_min=setting['lr'] * 1e-2
    )

    scaler = GradScaler()

    # ---------------- DATASET ----------------
    dataset_dir = os.path.join(args.data_dir, args.dataset)

    train_dataset = PairLoader(
        dataset_dir,
        'train',
        'train',
        setting['patch_size'],
        setting['edge_decay'],
        setting['only_h_flip']
    )

    val_dataset = PairLoader(
        dataset_dir,
        'test',
        setting['valid_mode'],
        setting['patch_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=setting['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=setting['batch_size'],
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ---------------- SAVE DIR ----------------
    save_dir = os.path.join(args.save_dir, args.exp)

    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, args.model + '.pth')

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

    best_psnr = 0
    start_epoch = 0

    # ---------------- RESUME TRAINING ----------------
    if os.path.exists(checkpoint_path):

        print("==> Found checkpoint:", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)

        if 'state_dict' in checkpoint:
            network.load_state_dict(checkpoint['state_dict'])
        else:
            network.load_state_dict(checkpoint)

        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint['best_psnr']
        else:
            start_epoch = 5

        print("Resuming from epoch:", start_epoch)

    print("\n===== STARTING TRAINING =====\n")

    # ---------------- TRAIN LOOP ----------------
    for epoch in tqdm(range(start_epoch, setting['epochs'])):

        loss = train(train_loader, network, criterion, optimizer, scaler)

        writer.add_scalar('train_loss', loss, epoch)

        scheduler.step()

        avg_psnr = valid(val_loader, network)

        writer.add_scalar('valid_psnr', avg_psnr, epoch)

        print(f"Epoch {epoch+1}/{setting['epochs']} | Loss {loss:.4f} | PSNR {avg_psnr:.2f}")

        if avg_psnr > best_psnr:

            best_psnr = avg_psnr

            torch.save({
                'epoch': epoch,
                'best_psnr': best_psnr,
                'state_dict': network.state_dict()
            }, checkpoint_path)

            print("✔ Model improved — checkpoint saved")

    print("\nBest PSNR:", best_psnr)