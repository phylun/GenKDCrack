# This file is for producing Teacher Network

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import os.path as osp
import pickle

from models.network_selector import network_selection
from con_dataset import CONDataSet

from utils.util import print_super_train_log
import time

from options.Super_train_options import SuperTrainOptions

# load train options
opt = SuperTrainOptions().parse()

expr_dir = os.path.join(opt.save_dir, opt.name)

input_size =  (opt.crop_size, opt.crop_size) if opt.crop else (opt.image_size, opt.image_size)

super_train_folder = opt.dataroot
super_train_list = os.path.join(super_train_folder, opt.listfile)
super_train_dataset = CONDataSet(super_train_folder, super_train_list, crop_size=input_size,
                                 mirror=False, mean=np.array((0.0, 0.0, 0.0)))
super_train_dataset_size = len(super_train_dataset)
train_ids = np.arange(super_train_dataset_size)
# print(train_ids)
np.random.seed(1234)
np.random.shuffle(train_ids)
pickle.dump(train_ids, open(osp.join(expr_dir, 'train_id.pkl'), 'wb'))

super_train_load = data.DataLoader(super_train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)

# create network
model = network_selection(opt, sel_net=0)
model.train()
model.cuda(opt.gpu_ids[0])


# train
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
expr_dir = os.path.join(opt.save_dir, opt.name)
losses_history = np.zeros((0, 2))
for i_iter in range(opt.n_epoch):

    avg_loss = list()
    for i, data in enumerate(super_train_load):
        iteration = i + (i_iter * super_train_load.__len__())

        load_t0 = time.time()
        images, labels, _, names = data
        images = images.float().cuda(opt.gpu_ids[0])
        labels = labels.long().cuda(opt.gpu_ids[0])

        pred = model(images)

        optimizer.zero_grad()
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        load_t1 = time.time()
        print_super_train_log(i, 5, [time.ctime(), i_iter+1, i_iter+1 % opt.n_epoch, opt.n_epoch, iteration, loss.item(), load_t1 - load_t0])

        avg_loss.append(loss.item())

    avg_loss = np.array(avg_loss)
    avg_loss = avg_loss.mean()
    tmp = np.array([i_iter, avg_loss])
    tmp = np.expand_dims(tmp, axis=0)
    losses_history = np.vstack((losses_history, tmp))
    # print(losses_history)

    # if (i_iter+1) % 100 == 0:
    if (i_iter+1) % opt.save_freq == 0:
        torch.save(model.state_dict(), expr_dir + '/Student_%s_epoch%d.pth'%(opt.Snet, i_iter+1))

np.save((expr_dir + '/%s_losses_history.npy'%(opt.name)), losses_history)

plt.figure('super_Losses History')
plt.plot(losses_history[:, 0], losses_history[:, 1], label='seg')
plt.title('losses history along with iteration')
plt.legend(loc='best')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.grid(True)
# plt.show()
plt.draw()
plt.savefig(expr_dir + '/%s_losses_history.png'%(opt.name))
plt.close()
