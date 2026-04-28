import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.mwirnet import MWIRNet

import lightning.pytorch as pl
import torch.nn.functional as F

class MWIRLitModel(pl.LightningModule):
    def __init__(self, ablation_mode="full"):
        super().__init__()
        self.net = MWIRNet(decoder=True, ablation_mode=ablation_mode)
    
    def forward(self,x):
        return self.net(x)


def restore_with_tta(net, degraded):
    restored_sum = torch.zeros_like(degraded)
    dims = (-2, -1)

    for k in range(4):
        aug = torch.rot90(degraded, k=k, dims=dims)
        restored = net(aug)
        restored_sum += torch.rot90(restored, k=-k, dims=dims)

        aug = torch.flip(torch.rot90(degraded, k=k, dims=dims), dims=(-1,))
        restored = net(aug)
        restored = torch.rot90(torch.flip(restored, dims=(-1,)), k=-k, dims=dims)
        restored_sum += restored

    return restored_sum / 8.0


def restore_image(net, degraded):
    if getattr(testopt, "tta", False):
        return restore_with_tta(net, degraded)
    return net(degraded)


def output_name_for_task(task, split):
    if task == "derain":
        return "derain" if split == "Rain100L" else "derain_" + split
    return "dehaze_" + split



def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = restore_image(net, degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))



def test_Derain_Dehaze(net, dataset, task="derain", output_name=None):
    output_name = output_name or task
    output_path = testopt.output_path + output_name + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = restore_image(net, degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint file name under ckpt/')
    parser.add_argument('--ckpt_path', type=str, default=None, help='direct checkpoint path; overrides --ckpt_name')
    parser.add_argument('--derain_splits', nargs='+', default=["Rain100L"],
                        help='derain test split names under --derain_path')
    parser.add_argument('--dehaze_splits', nargs='+', default=["outdoor"],
                        help='dehaze test split names under --dehaze_path')
    parser.add_argument('--tta', action='store_true',
                        help='use 8-way flip/rotation self-ensemble during inference')
    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'zero_prompt', 'no_channel_attention'],
                        help='MWIR-Net module ablation mode.')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = testopt.ckpt_path if testopt.ckpt_path is not None else "ckpt/" + testopt.ckpt_name


    
    denoise_splits = ["bsd68"]
    derain_splits = testopt.derain_splits
    dehaze_splits = testopt.dehaze_splits

    denoise_tests = []
    derain_tests = []

    if testopt.mode in [0, 3]:
        base_path = testopt.denoise_path
        for i in denoise_splits:
            testopt.denoise_path = os.path.join(base_path, i) + '/'
            if os.path.isdir(testopt.denoise_path):
                denoise_testset = DenoiseTestDataset(testopt)
                denoise_tests.append((denoise_testset, i))


    print("CKPT name : {}".format(ckpt_path))

    net = MWIRLitModel.load_from_checkpoint(
        ckpt_path,
        strict=False,
        ablation_mode=testopt.ablation_mode,
    ).cuda()
    net.eval()

    
    if testopt.mode == 0:
        for testset,name in denoise_tests:
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name) + '/'
            derain_set = DerainDehazeDataset(testopt, task="derain", addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain", output_name=output_name_for_task("derain", name))
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        dehaze_base_path = testopt.dehaze_path
        for name in dehaze_splits:
            print('Start testing {} dehazing...'.format(name))
            testopt.dehaze_path = os.path.join(dehaze_base_path, name) + '/'
            dehaze_set = DerainDehazeDataset(testopt, task="dehaze", addnoise=False, sigma=15)
            test_Derain_Dehaze(net, dehaze_set, task="dehaze", output_name=output_name_for_task("dehaze", name))
    elif testopt.mode == 3:
        for testset,name in denoise_tests:
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)



        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name) + '/'
            derain_set = DerainDehazeDataset(testopt, task="derain", addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain", output_name=output_name_for_task("derain", name))

        print('Start testing SOTS...')
        dehaze_base_path = testopt.dehaze_path
        for name in dehaze_splits:
            print('Start testing {} dehazing...'.format(name))
            testopt.dehaze_path = os.path.join(dehaze_base_path, name) + '/'
            dehaze_set = DerainDehazeDataset(testopt, task="dehaze", addnoise=False, sigma=15)
            test_Derain_Dehaze(net, dehaze_set, task="dehaze", output_name=output_name_for_task("dehaze", name))
