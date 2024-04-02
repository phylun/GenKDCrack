"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2
from termcolor import cprint

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def GetPallete(choice, cls_img, w, h):
    if choice == 1:
        b = np.float32(np.where(cls_img > 127, 255, 0))
        g = np.float32(np.zeros([w, h, 1]))
        r = np.float32(np.zeros([w, h, 1]))
    elif choice == 2:
        b = np.float32(np.zeros([w, h, 1]))
        g = np.float32(np.where(cls_img > 127, 255, 0))
        r = np.float32(np.zeros([w, h, 1]))
    elif choice == 3:
        b = np.float32(np.zeros([w, h, 1]))
        g = np.float32(np.zeros([w, h, 1]))
        r = np.float32(np.where(cls_img > 127, 255, 0))
    elif choice == 4:
        b = np.float32(np.where(cls_img > 127, 255, 0))
        g = np.float32(np.where(cls_img > 127, 255, 0))
        r = np.float32(np.where(cls_img > 127, 255, 0))
    else:
        b = np.zeros((h, w, 1), dtype=np.uint8)
        g = np.zeros((h, w, 1), dtype=np.uint8)
        r = np.zeros((h, w, 1), dtype=np.uint8)

    pallete = cv2.merge((b, g, r))

    return pallete


def print_train_multi_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        # cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss0:{:.4f}||Loss1:{:.4f}||Loss2:{:.4f}||Loss3:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')

def print_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||EpochIter:{}/{}||Loss(Seg):{:.4f}||Loss(S_Const):{:.6f}||Loss(U_Const):{:.6f}||Batch_Time:{:.4f}'.format(*info_list), 'green')

def print_super_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')

        # print_train_log(i, 5, [time.ctime(), i_iter % 500, loss_sup.item(), loss_sup_r.item(), cps_loss.item(), load_t1 - load_t0])
        
def print_train_dual_cps_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||Model:{}||EpochIter:{}/{}||Loss(Up-Left):{:.4f}||Loss(Up-Right):{:.4f}||Loss(S-Left):{:.4f}||Loss(S-Right):{:.4f}||Loss(CPS):{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        
def print_train_KD_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        # cprint('Time:{}||EpochIter:{}/{}||Loss(Left):{:.4f}||Loss(Right):{:.4f}||Loss(CPS):{:.4f}||Loss(Self):{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        cprint('Time:{}||EpochIter:{}/{}||Loss(Student):{:.4f}||Loss(Const):{:.7f}||Batch_Time:{:.2f}'.format(*info_list), 'green')