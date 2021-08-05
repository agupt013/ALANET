import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import math

import argparse
from tqdm import tqdm
import numpy as np
# from plot import imsave, tsave

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
    # torch.backends.cudnn.benchmark = True
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

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
    train_data = BlurDataset(os.path.join(args.dataset_root, 'train'),
                             seq_len=args.sequence_length, tau=args.num_frame_blur, delta=5, transform=train_tf)
    test_data = BlurDataset(os.path.join(args.dataset_root, test_valid),
                            seq_len=args.sequence_length, tau=args.num_frame_blur, delta=5, transform=train_tf)

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    # TODO IMPORT YOUR CUSTOM MODEL
    model = UNet(3, 3, device, decode_mode=args.decode_mode)

    if args.checkpoint:
        store_dict = torch.load(args.checkpoint)
        try:
            model.load_state_dict(store_dict['state_dict'])
        except KeyError:
            model.load_state_dict(store_dict)

    if args.train_continue:
        store_dict = torch.load(args.checkpoint)
        model.load_state_dict(store_dict['state_dict'])

    else:
        store_dict = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    model.to(device)
    model.train(True)

    # model = nn.DataParallel(model)

    # TODO DEFINE MORE CRITERIA
    # input(True if device == torch.device('cuda:0') else False)
    criterion = {
                  'MSE': nn.MSELoss(),
                  'L1' : nn.L1Loss(),
                  # 'Perceptual': PerceptualLoss(model='net-lin', net='vgg', dataparallel=True,
                  #                              use_gpu=True if device == torch.device('cuda:0') else False)
                  }

    criterion_w = {
                  'MSE': 1.0,
                  'L1': 10.0,
                  'Perceptual': 10.0
            }

    # Define optimizers
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate)

    # Define lr scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

   # best_acc = 0.0
    # start = time.time()
    cLoss = store_dict['loss']
    valLoss = store_dict['valLoss']
    valPSNR = store_dict['valPSNR']
    checkpoint_counter = 0

    loss_tracker = {}
    loss_tracker_test = {}

    psnr_old = 0.0
    dssim_old = 0.0

    for epoch in range(1, 10*args.epochs):  # loop over the dataset multiple times

        # Append and reset
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        running_loss = 0

        # Increment scheduler count
        scheduler.step()

        tqdm_loader = tqdm(range(len(train_loader)), ncols=150)

        loss = 0.0
        psnr_ = 0.0
        dssim_ = 0.0

        loss_tracker = {}
        for loss_fn in criterion.keys():
            loss_tracker[loss_fn] = 0.0

        # Train
        model.train(True)
        total_steps = 0.01
        total_steps_test = 0.01
        '''for train_idx, data in enumerate(train_loader, 1):
            loss = 0.0
            blur_data, sharpe_data = data
            #import pdb; pdb.set_trace()
            # input(sharpe_data.shape)
            #import pdb; pdb.set_trace()
            interp_idx = int(math.ceil((args.num_frame_blur/2) - 0.49))
            #input(interp_idx)
            if args.decode_mode == 'interp':
                sharpe_data = sharpe_data[:, :, 1::2, :, :]
            elif args.decode_mode == 'deblur':
                sharpe_data = sharpe_data[:, :, 0::2, :, :]
            else:
                #print('\nBoth\n')
                sharpe_data = sharpe_data

            #print(sharpe_data.shape)
            #input(blur_data.shape)
            blur_data = blur_data.to(device)[:, :, :, :352, :].permute(0, 1, 2, 4, 3)
            try:
                sharpe_data = sharpe_data.squeeze().to(device)[:, :, :, :352, :].permute(0, 1, 2, 4, 3)
            except:
                sharpe_data = sharpe_data.squeeze(3).to(device)[:, :, :, :352, :].permute(0, 1, 2, 4, 3)

            # clear gradient
            optimizer.zero_grad()

            # forward pass
            sharpe_out = model(blur_data)
            # import pdb; pdb.set_trace()
            # input(sharpe_out.shape)

            # compute losses
            # import pdb;
            # pdb.set_trace()
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


                # try:
                # import pdb; pdb.set_trace()
                loss += loss_tmp # if
                # except :
                try:
                    loss_tracker[loss_fn] += loss_tmp.item()
                except KeyError:
                    loss_tracker[loss_fn] = loss_tmp.item()

            # Backpropagate
            loss.backward()
            optimizer.step()

            # statistics
            # import pdb; pdb.set_trace()
            sharpe_out = sharpe_out.detach().cpu().numpy()
            sharpe_data = sharpe_data.cpu().numpy()
            for sidx in range(S):
                for bidx in range(B):
                    psnr_ += psnr(sharpe_out[bidx, :, sidx, :, :], sharpe_data[bidx, :, sidx, :, :]) #, peak=1.0)
                    """dssim_ += dssim(np.moveaxis(sharpe_out[bidx, :, sidx, :, :], 0, 2),
                                    np.moveaxis(sharpe_data[bidx, :, sidx, :, :], 0, 2)
                                    )"""

            """sharpe_out = sharpe_out.reshape(-1,3, sx, sy).detach().cpu().numpy()
            sharpe_data = sharpe_data.reshape(-1, 3, sx, sy).cpu().numpy()
            for idx in range(sharpe_out.shape[0]):
                # import pdb; pdb.set_trace()
                psnr_ += psnr(sharpe_data[idx], sharpe_out[idx])
                dssim_ += dssim(np.swapaxes(sharpe_data[idx], 2, 0), np.swapaxes(sharpe_out[idx], 2, 0))"""

            # psnr_ /= sharpe_out.shape[0]
            # dssim_ /= sharpe_out.shape[0]
            running_loss += loss.item()
            loss_str = ''
            total_steps += B*S
            for key in loss_tracker.keys():
               loss_str += ' {0} : {1:6.4f} '.format(key, 1.0*loss_tracker[key] / total_steps)

            # set display info
            if train_idx % 5 == 0:
                tqdm_loader.set_description(('\r[Training] [Ep {0:6d}] loss: {1:6.4f} PSNR: {2:6.4f} SSIM: {3:6.4f} '.format
                                    (epoch, running_loss / total_steps,
                                     psnr_ / total_steps,
                                     dssim_ / total_steps) + loss_str
                                    ))

                tqdm_loader.update(5)
        tqdm_loader.close()'''


        # Validation
        running_loss_test = 0.0
        psnr_test = 0.0
        dssim_test = 0.0
        # print('len', len(test_loader))
        tqdm_loader_test = tqdm(range(len(test_loader)), ncols=150)
        # import pdb; pdb.set_trace()

        loss_tracker_test = {}
        for loss_fn in criterion.keys():
            loss_tracker_test[loss_fn] = 0.0

        with torch.no_grad():
            model.eval()
            total_steps_test = 0.0

            for test_idx, data in enumerate(test_loader, 1):
                loss = 0.0
                blur_data, sharpe_data = data
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

                # clear gradient
                optimizer.zero_grad()

                # forward pass
                sharpe_out = model(blur_data)
                # import pdb; pdb.set_trace()
                # input(sharpe_out.shape)

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

                if ((test_idx % args.progress_iter) == args.progress_iter - 1):
                    itr = test_idx + epoch*len(test_loader)
                    # itr_train
                    writer.add_scalars('Loss', {'trainLoss': running_loss / total_steps,
                                            'validationLoss': running_loss_test / total_steps_test}, itr)
                    writer.add_scalar('Train PSNR', psnr_ / total_steps, itr)
                    writer.add_scalar('Test PSNR', psnr_test / total_steps_test, itr)
                    # import pdb; pdb.set_trace()
                    # writer.add_image('Validation', sharpe_out.permute(0, 2, 3, 1), itr)

                # statistics
                sharpe_out = sharpe_out.detach().cpu().numpy()
                sharpe_data = sharpe_data.cpu().numpy()
                for sidx in range(S):
                    for bidx in range(B):
                        psnr_test += psnr(sharpe_out[bidx, :, sidx, :, :], sharpe_data[bidx, :, sidx, :, :]) #, peak=1.0)
                        dssim_test += dssim(np.moveaxis(sharpe_out[bidx, :, sidx, :, :], 0, 2),
                                            np.moveaxis(sharpe_data[bidx, :, sidx, :, :], 0, 2)) #,range=1.0  )

                running_loss_test += loss.item()
                total_steps_test += B*S
                loss_str = ''
                for key in loss_tracker.keys():
                    loss_str += ' {0} : {1:6.4f} '.format(key, 1.0 * loss_tracker_test[key] / total_steps_test)

                # set display info

                tqdm_loader_test.set_description(
                            ('\r[Test    ] [Ep {0:6d}] loss: {1:6.4f} PSNR: {2:6.4f} SSIM: {3:6.4f} '.format
                             (epoch, running_loss_test / total_steps_test,
                              psnr_test / total_steps_test,
                              dssim_test / total_steps_test
                              ) + loss_str
                             )
                        )
                tqdm_loader_test.update(1)
            tqdm_loader_test.close()

        # save model
        if psnr_old < (psnr_test / total_steps_test):
            if epoch != 1:
                os.remove(os.path.join(args.checkpoint_dir,
                           'epoch-{}-test-psnr-{}-ssim-{}.ckpt'.format(epoch_old, str(round(psnr_old, 4)).replace('.', 'pt'),
                                                                  str(round(dssim_old, 4)).replace('.', 'pt')
                                                                  )
                                       )
                          )
            epoch_old = epoch
            psnr_old = psnr_test / total_steps_test
            dssim_old = dssim_test / total_steps_test

            checkpoint_dict = {
                'epoch': epoch_old,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_psnr': psnr_ / total_steps,
                'train_dssim': dssim_ / total_steps,
                'train_mse': loss_tracker['MSE'] / total_steps,
                'train_l1': loss_tracker['L1'] / total_steps,
                # 'train_percp': loss_tracker['Perceptual'] / total_steps,
                'test_psnr': psnr_old,
                'test_dssim': dssim_old,
                'test_mse': loss_tracker_test['MSE'] / total_steps_test,
                'test_l1': loss_tracker_test['L1'] / total_steps_test,
                # 'test_percp': loss_tracker_test['Perceptual'] / total_steps_test,
                               }

            torch.save(checkpoint_dict, os.path.join(args.checkpoint_dir,
                       'epoch-{}-test-psnr-{}-ssim-{}.ckpt'.format(epoch_old, str(round(psnr_old,4)).replace('.', 'pt'),
                                                              str(round(dssim_old,4)).replace('.', 'pt')
                                                              )
                                                        )
                       )


        # if epoch % args.checkpoint_epoch == 0:
        #    torch.save(model.state_dict(),args.checkpoint_dir + str(int(epoch/100))+".ckpt")



    return None


if __name__ == "__main__":

    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help='path to dataset folder containing train-test-validation folders')
    parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
    parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
    parser.add_argument("--train_continue", type=bool, default=False,
                        help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs to train. Default: 200.')
    parser.add_argument("--train_batch_size", type=int, default=6, help='batch size for training. Default: 6.')
    parser.add_argument("--test_batch_size", type=int, default=10, help='batch size for validation. Default: 10.')
    parser.add_argument("--init_learning_rate", type=float, default=0.00004,
                        help='set initial learning rate. Default: 0.00002.')
    parser.add_argument("--milestones", type=list, default=[30, 60],
                        help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. '
                             'Default: [100, 150]')
    parser.add_argument("--progress_iter", type=int, default=100,
                        help='frequency of reporting progress and validation. N: after every N iterations. '
                             'Default: 100.')
    parser.add_argument("--checkpoint_epoch", type=int, default=5,
                        help='checkpoint saving frequency. N: after every N epochs.\
                        Each checkpoint is roughly of size 151 MB.Default: 5.')
    parser.add_argument("--sequence_length", type=int, default=3,
                        help='length of video sequence to train.')
    parser.add_argument("--num_frame_blur", type=int, default=2,
                        help='length of video sequence to train.')
    parser.add_argument("--decode_mode", type=str, default='both',
                        help='length of video sequence to train.')
    parser.add_argument("--valid", type=bool, default=False,
                        help='length of video sequence to train.')
    

    args = parser.parse_args()

    # Training
    main(args)
