#test
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import matplotlib.pyplot as plt

dir_checkpoint = 'checkpoints/'

def apply_disparity(img, disp):
    """
    img = [b,18,h,w]
    disp = [b,h,w]
    """
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                           padding_mode='zeros')

    return output

def train_net(net,
              device,
              epochs=100,
              batch_size=20,
              lr=0.001,
              val_percent=0.0,
              save_cp=True,
              img_scale=1.0):

    torch.cuda.empty_cache() 



    dataset = BasicDataset('/media/patterson_sl_data/patterson_dataset/train', img_scale, mode='show_room')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # num workers should be 4 for both
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr) # they used rms w/ momentum 0.9, weight_decay=1e-8
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    criterion = nn.BCELoss(reduction='none')
    warped_criterion = nn.BCELoss(reduction='none')
    gd = nn.MSELoss()

    full_steps = int(n_train/batch_size) # total number of steps
    print_step = int(int(full_steps*0.8))
    vali_step = int(int(full_steps*0.8))

    www = (torch.tensor([[[0., 0., 0.],
                    [-1.,0.,1.],
                    [0., 0., 0.]]], dtype=torch.float32, requires_grad=False) + 1e-9).repeat(9, 1, 1)

    grad_k = www.view(1, 9, 3, 3).to(device=device, dtype=torch.float32)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            dataset.data_mode = 'train'
            for batch in train_loader:
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                disparity = batch['disparity'].to(device=device, dtype=torch.float32)
                depth_mask = batch['depth_mask'].to(device=device, dtype=torch.float32)
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'


                # graph run
                masks_pred = net(imgs)

                sig_loss = torch.mean(criterion(masks_pred, true_masks))

		# Creating the derivative loss
                binaryPredR = masks_pred[:,:9]
                binaryPredL = masks_pred[:,9:]
                truePredR = true_masks[:,:9]
                truePredL = true_masks[:,9:]
                truePredL_grads = torch.nn.functional.conv2d(truePredL, grad_k, padding=(1,1))
                truePredR_grads = torch.nn.functional.conv2d(truePredR, grad_k, padding=(1,1))
                truePredFull_grads = torch.cat((truePredR_grads,truePredL_grads),1)
                binaryPredL_grads = torch.nn.functional.conv2d(binaryPredL, grad_k, padding=(1,1))
                binaryPredR_grads = torch.nn.functional.conv2d(binaryPredR, grad_k, padding=(1,1))
                binaryPredFull_grads = torch.cat((binaryPredR_grads,binaryPredL_grads),1)


                warped_left_sl = apply_disparity(binaryPredR, disparity[:,0])
                warped_right_sl = apply_disparity(binaryPredL, disparity[:,1])
                warped_sl = torch.cat((warped_right_sl, warped_left_sl),1)
                warped_sig_loss = torch.mean(warped_criterion(warped_sl*depth_mask, true_masks*depth_mask))/40.

                grad_loss = gd(truePredFull_grads,binaryPredFull_grads)/80.


                #loss = sig_loss + grad_loss + warped_sig_loss
                loss = sig_loss + grad_loss
                #loss = warped_sig_loss


                epoch_loss += loss.item()
                writer.add_scalar('Loss/train_gd', grad_loss, global_step)
                writer.add_scalar('Loss/train_sig', sig_loss, global_step)
                writer.add_scalar('Loss/train_warped_sig', warped_sig_loss, global_step)
                writer.add_scalar('Loss/train', loss, global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

                full_steps = int(n_train/batch_size) # total number of steps
                global_step += 1
                if global_step % print_step == 0: # print loss every 10%
                    print('Sample Loss: ', loss.item())
                if global_step % vali_step == 0: # vali every half epoch
                    print('Validation')
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    dataset.data_mode = 'valid'
                    val_score = eval_net(net, val_loader, device, writer, global_step)
                    dataset.data_mode = 'train'
                    #scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)


                    logging.info('Validation Dice Coeff: {}'.format(val_score))



        if save_cp and epoch % 20 == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=50.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-ic', '--inputchannels', type=int, default=6,
                        help='input channel stuff')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=args.inputchannels)    # THIS VARIES THE INPUT CHANNEL LEN, MUST ALWAYS REMEMBER THIS, SHOULD ADD AS ARGPARSE
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n')
                 #f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
