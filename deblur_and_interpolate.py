import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import math

import argparse
from tqdm import tqdm
import numpy as np
from plot import imsave, tsave

from model import UNet
from utils import PerceptualLoss, psnr, dssim
from blur_dataset_old import BlurDataset

### Create transform to display image from tensor
mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


### Utils
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(args):
    writer = SummaryWriter(os.path.join('./logs'))
    # torch.backends.cudnn.benchmark = False
    # if not os.path.isdir(args.checkpoint_dir):
    #     os.mkdir(args.checkpoint_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('[MODEL] CUDA DEVICE : {}'.format(device))

    # TODO DEFINE TRAIN AND TEST TRANSFORMS
    train_tf = None
    test_tf = None

    # Channel wise mean calculated on adobe240-fps training dataset
    mean = [0.429, 0.431, 0.397]
    std = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_valid = 'validation' if args.valid else 'test'
    # train_data = BlurDataset(os.path.join(args.dataset_root, 'train'),
    #                         seq_len=args.sequence_length, tau=args.num_frame_blur, delta=5, transform=train_tf)
    test_data = BlurDataset(os.path.join(args.dataset_root, test_valid),
                            seq_len=args.sequence_length, tau=args.num_frame_blur, delta=5, transform=train_tf, return_path=True)

    # train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    # TODO IMPORT YOUR CUSTOM MODEL
    model = UNet(3, 3, device, decode_mode=args.decode_mode)

    if args.checkpoint:
        store_dict = torch.load(args.checkpoint)
        try:
            print('Loading checkpoint...')
            model.load_state_dict(store_dict['state_dict'])
            print('Done.')
        except KeyError:
            print('Loading checkpoint...')
            model.load_state_dict(store_dict)
            print('Done.')

    model.to(device)
    model.train(False)

    # model = nn.DataParallel(model)

    # TODO DEFINE MORE CRITERIA
    # input(True if device == torch.device('cuda:0') else False)
    criterion = {
                  'MSE': nn.MSELoss(),
                  'L1' : nn.L1Loss(),
                  # 'Perceptual': PerceptualLoss(model='net-lin', net='vgg', dataparallel=False,
                  #                             use_gpu=True if device == torch.device('cuda:0') else False)
                  }


    # Validation
    running_loss_test = 0.0
    psnr_test = 0.0
    dssim_test = 0.0

    tqdm_loader_test = tqdm(range(len(test_loader)), ncols=150)

    loss_tracker_test = {}
    for loss_fn in criterion.keys():
        loss_tracker_test[loss_fn] = 0.0

    with torch.no_grad():
        model.eval()
        total_steps_test = 0.0
        interp_idx = int(math.ceil((args.num_frame_blur / 2) - 0.49))
        for test_idx, data in enumerate(test_loader, 1):
            loss = 0.0
            blur_data, sharpe_data, sharp_names = data
            import pdb; pdb.set_trace()
            interp_idx = int(math.ceil((args.num_frame_blur / 2) - 0.49))
            # input(interp_idx)
            if args.decode_mode == 'interp':
                sharpe_data = sharpe_data[:, :, 1::2, :, :]
            elif args.decode_mode == 'deblur':
                sharpe_data = sharpe_data[:, :, 0::2, :, :]
            else:
                # print('\nBoth\n')
                sharpe_data = sharpe_data

            # print(sharpe_data.shape)
            # input(blur_data.shape)
            blur_data = blur_data.to(device)[:, :, :, :352, :].permute(0, 1, 2, 4, 3)
            try:
                sharpe_data = sharpe_data.squeeze().to(device)[:, :, :, :352, :].permute(0, 1, 2, 4, 3)
            except:
                sharpe_data = sharpe_data.squeeze(3).to(device)[:, :, :, :352, :].permute(0, 1, 2, 4, 3)

            # forward pass
            sharpe_out = model(blur_data).float()

            # compute losses
            sharpe_out = sharpe_out.permute(0, 2, 1, 3, 4)
            B, C, S, Fx, Fy = sharpe_out.shape
            for loss_fn in criterion.keys():
                loss_tmp = 0.0
                if loss_fn == 'Perceptual':
                    for bidx in range(B):
                        loss_tmp += criterion_w[loss_fn] * \
                                    criterion[loss_fn](sharpe_out[bidx].permute(1, 0, 2, 3),
                                                       sharpe_data[bidx].permute(1, 0, 2, 3)).sum()
                    # loss_tmp /= B
                else:
                    loss_tmp = criterion_w[loss_fn] * \
                               criterion[loss_fn](sharpe_out, sharpe_data)
                loss += loss_tmp
                try:
                    loss_tracker_test[loss_fn] += loss_tmp.item()
                except KeyError:
                    loss_tracker_test[loss_fn] = loss_tmp.item()

            # statistics
            #sharpe_out = sharpe_out.detach().cpu().numpy()
            #sharpe_data = sharpe_data.cpu().numpy()
            #  import pdb; pdb.set_trace()
            # t_grid = torchvision.utils.make_grid(torch.stack([blur_data[0], sharpe_out[0], sharpe_data[0]], dim=0),
            #                                    nrow=3)
            # tsave(t_grid, './imgs/{}/combined.jpg'.format(test_idx))
            for sidx in range(S):
                for bidx in range(B):
                    if not os.path.exists('./imgs/{}'.format(sharp_names[1])):
                        os.makedirs('./imgs/{}'.format(test_idx))
                    blur_path = './imgs/{}/blur_input_{}.jpg'.format(test_idx, sidx)

                    # import pdb; pdb.set_trace()
                    # torchvision.utils.save_image(sharpe_out[bidx, :, sidx, :, :],blur_path, normalize=True, range=(0,255));

                    imsave(blur_data, blur_path, bidx, sidx)

                    sharp_path = './imgs/{}/sharpe_gt_{}{}.jpg'.format(test_idx, sidx, sidx)
                    imsave(sharpe_data, sharp_path, bidx, sidx)

                    deblur_path = './imgs/{}/out_{}{}.jpg'.format(test_idx, sidx, sidx)
                    imsave(sharpe_out, deblur_path, bidx, sidx)

                    if sidx > 0 and sidx < S:
                        interp_path = './imgs/{}/out_{}{}.jpg'.format(test_idx, sidx-1, sidx)
                        imsave(sharpe_out, interp_path, bidx, sidx)
                        sharp_path = './imgs/{}/sharpe_gt_{}{}.jpg'.format(test_idx, sidx-1, sidx)
                        imsave(sharpe_data, sharp_path, bidx, sidx)

                    psnr_local = psnr(im_nm * sharpe_out[bidx, :, sidx, :, :].detach().cpu().numpy(),
                                      im_nm * sharpe_data[bidx, :, sidx, :, :].cpu().numpy())
                    dssim_local = dssim(np.moveaxis(im_nm * sharpe_out[bidx, :, sidx, :, :].cpu().numpy(), 0, 2),
                                        np.moveaxis(im_nm * sharpe_data[bidx, :, sidx, :, :].cpu().numpy(), 0, 2)
                                        )
                    psnr_test += psnr_local
                    dssim_test += dssim_local
            f = open('./imgs/{0}/psnr-{1:.4f}-dssim-{2:.4f}.txt'.format(test_idx, psnr_local/(B), dssim_local/(B)),'w')
            f.close()
            running_loss_test += loss.item()
            total_steps_test += B*S
            loss_str = ''
            for key in loss_tracker_test.keys():
                loss_str += ' {0} : {1:6.4f} '.format(key, 1.0 * loss_tracker_test[key] / total_steps_test)

            # set display info

            tqdm_loader_test.set_description(
                        ('\r[Test    ] loss: {0:6.4f} PSNR: {1:6.4f} SSIM: {2:6.4f} '.format
                         ( running_loss_test / total_steps_test,
                          psnr_test / total_steps_test,
                          dssim_test / total_steps_test
                          ) + loss_str
                         )
                    )
            tqdm_loader_test.update(1)
        tqdm_loader_test.close()
    return None


if __name__ == "__main__":
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help='path to dataset folder containing train-test-validation folders')
    parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
    parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
    parser.add_argument("--test_batch_size", type=int, default=10, help='batch size for validation. Default: 10.')
    parser.add_argument("--progress_iter", type=int, default=100,
                        help='frequency of reporting progress and validation. N: after every N iterations. '
                             'Default: 100.')
    parser.add_argument("--sequence_length", type=int, default=3,
                        help='length of video sequence to train.')
    parser.add_argument("--num_frame_blur", type=int, default=2,
                        help='number of frames to blur.')
    parser.add_argument("--decode_mode", type=bool, default=False,
                        help='length of video sequence to train.')
    parser.add_argument("--valid", type=bool, default=False,
                        help='to run on validation set or test set.')

    args = parser.parse_args()

    # Training
    main(args)