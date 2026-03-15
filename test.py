import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str)
parser.add_argument('--num_workers', default=4, type=int)  # reduce for CPU
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--save_dir', default='./saved_models/', type=str)
parser.add_argument('--result_dir', default='./results/', type=str)
parser.add_argument('--dataset', default='RESIDE-IN', type=str)
parser.add_argument('--exp', default='indoor', type=str)
args = parser.parse_args()

# 🔥 FORCE CPU
device = torch.device("cpu")


def single(save_dir):
    state_dict = torch.load(save_dir, map_location=device)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):

        # 🔥 Move tensors to CPU
        input = batch['source'].to(device)
        target = batch['target'].to(device)

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))

            ssim_val = ssim(
                F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1,
                size_average=False
            ).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print(f"Test: [{idx}]  PSNR: {PSNR.val:.02f} ({PSNR.avg:.02f})  "
              f"SSIM: {SSIM.val:.03f} ({SSIM.avg:.03f})")

        f_result.write(f"{filename},{psnr_val:.02f},{ssim_val:.03f}\n")

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()

    os.rename(
        os.path.join(result_dir, 'results.csv'),
        os.path.join(result_dir, f'{PSNR.avg:.02f} | {SSIM.avg:.04f}.csv')
    )


if __name__ == '__main__':

    network = eval(args.model.replace('-', '_'))()
    network.to(device)   # 🔥 CPU instead of CUDA

    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)

    test_dataset = PairLoader(dataset_dir, 'test', 'test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,     # 🔥 IMPORTANT for CPU
        pin_memory=False   # 🔥 IMPORTANT for CPU
    )

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)

    test(test_loader, network, result_dir)