import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import torch.distributed as dist
import numpy as np
import pickle

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import sys
import os
import os.path as osp
import pickle

from models.network_selector import network_selection

from con_dataset import CONDataSet, CONUNLABELDataSet

from utils.util import print_train_KD_log
import time

from options.KDGen_train_options import KDGenTrainOptions

def consistency_loss(student_logits, teacher_logits, loss_func):    
    teacher_probs = torch.softmax(teacher_logits, dim=1)
    teacher_probs = torch.argmax(teacher_probs, dim=1).long()
    return loss_func(student_logits, teacher_probs)


# load train options
opt = KDGenTrainOptions().parse()

expr_dir = os.path.join(opt.save_dir, opt.name)

input_size =  (opt.crop_size, opt.crop_size) if opt.crop else (opt.image_size, opt.image_size)


super_train_folder = opt.dataroot
super_train_list = os.path.join(super_train_folder, opt.listfile)
# super_train_list = os.path.join(super_train_folder, 'valConc.txt')
super_train_dataset = CONDataSet(super_train_folder, super_train_list, crop_size=input_size,
                                 mirror=False, mean=np.array((0.0, 0.0, 0.0)))
super_train_dataset_size = len(super_train_dataset)

unsup_train_folder = opt.semidataroot
unsup_train_list = os.path.join(unsup_train_folder, opt.semilistfile)
unsup_train_dataset = CONUNLABELDataSet(unsup_train_folder, unsup_train_list, crop_size=input_size,
                                        mirror=False, mean=np.array((0.0, 0.0, 0.0)))


train_ids = np.arange(super_train_dataset_size)
# print(train_ids)
np.random.seed(1234)
np.random.shuffle(train_ids)
pickle.dump(train_ids, open(osp.join(expr_dir, 'train_id.pkl'), 'wb'))

super_train_load = data.DataLoader(super_train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)
unsup_train_load = data.DataLoader(unsup_train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)


# create network
T_model_A = network_selection(opt, sel_net=1)
T_model_B = network_selection(opt, sel_net=2)
S_model = network_selection(opt, sel_net=0)

# student for train
S_model.train()
S_model.cuda(opt.gpu_ids[0])

###############################################################
# load pre-trained weight model
pretrain_model_A_name = [f for f in os.listdir('./pretrained') if f.find('%s'%opt.TnetA)> -1]
pretrain_model_A_name = pretrain_model_A_name[0]
saved_state_dict_teacher_A = torch.load(os.path.join('./pretrained/%s'%(pretrain_model_A_name)))
T_model_A.load_state_dict(saved_state_dict_teacher_A)

pretrain_model_B_name = [f for f in os.listdir('./pretrained') if f.find('%s'%opt.TnetB)> -1]
pretrain_model_B_name = pretrain_model_B_name[0]
saved_state_dict_teacher_B = torch.load(os.path.join('./pretrained/%s'%(pretrain_model_B_name)))
T_model_B.load_state_dict(saved_state_dict_teacher_B)
############################################################### 

# teacher for evaluation
T_model_A.eval()
T_model_A.cuda(opt.gpu_ids[0]) 

T_model_B.eval()
T_model_B.cuda(opt.gpu_ids[0])

S_optimizer = torch.optim.Adam(S_model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

losses_history = np.zeros((0, 3))
for i_iter in range(opt.n_epoch):

    avg_loss = list()

    super_train_dataloader = iter(super_train_load)
    unsup_train_dataloader = iter(unsup_train_load)

    avg_loss_S_seg = list()
    avg_loss_u_const = list()    
    
    for i in range(super_train_load.__len__()):

        load_t0 = time.time()        
        S_optimizer.zero_grad()        
        s_images, s_labels, _, s_names = next(super_train_dataloader)
        s_images = s_images.float().cuda(opt.gpu_ids[0])
        s_labels = s_labels.long().cuda(opt.gpu_ids[0])

        u_images, _, u_names = next(unsup_train_dataloader)
        u_images = u_images.float().cuda(opt.gpu_ids[0])
                        
        S_s_preds = S_model(s_images)
                
        S_u_preds = S_model(u_images)
        with torch.no_grad():
            T_u_preds_A = T_model_A(u_images)
            T_u_preds_B = T_model_B(u_images)
            T_u_preds = 0.5 * T_u_preds_A + 0.5 * T_u_preds_B
        
            
        S_s_loss = criterion(S_s_preds, s_labels)        
        u_consist_loss = opt.consist_weight * consistency_loss(S_u_preds, T_u_preds.detach(), criterion)                                    
        loss = S_s_loss + u_consist_loss
        
                
        if (i % opt.lp_freq == 0) and (i_iter > opt.lp_epoch):
            
            # save unlabeled origin image
            sample_image = u_images[0, ].cpu().detach().numpy().copy()            
            sample_image = sample_image.transpose(1, 2, 0)
            sample_image = sample_image[:, :, ::-1]
            sample_image = sample_image * 255.0
            sample_image = sample_image.astype(np.uint8)
            cv2.imwrite(os.path.join('learning_progress', opt.name, 'Epoch%d_'%(i_iter) + u_names[0] + '_unlabel_orign.jpg'), sample_image)
            
            # save pseudo label of teacher            
            sample_T_u_pred = torch.softmax(T_u_preds[0, ], dim=0)            
            sample_T_u_pred = sample_T_u_pred.cpu().detach().numpy().copy()            
            sample_T_u_pred = np.asarray(np.argmax(sample_T_u_pred, axis=0), dtype=np.int32)            
            sample_T_u_pred = sample_T_u_pred * 255.0
            sample_T_u_pred = sample_T_u_pred.astype(np.uint8)
            cv2.imwrite(os.path.join('learning_progress', opt.name, 'Epoch%d_'%(i_iter) + u_names[0] + '_pseudo_label.jpg'), sample_T_u_pred)

            # save prediction of student
            sample_S_u_pred = torch.softmax(S_u_preds[0, ], dim=0)
            sample_S_u_pred = sample_S_u_pred.cpu().detach().numpy().copy()
            sample_S_u_pred = np.asarray(np.argmax(sample_S_u_pred, axis=0), dtype=np.int32)
            sample_S_u_pred = sample_S_u_pred * 255.0
            sample_S_u_pred = sample_S_u_pred.astype(np.uint8)
            cv2.imwrite(os.path.join('learning_progress', opt.name, 'Epoch%d_'%(i_iter) + u_names[0] + '_unlabel_pred.jpg'), sample_S_u_pred)
                    
        loss.backward()        
        S_optimizer.step()                
                        
        avg_loss_S_seg.append(S_s_loss.item())
        avg_loss_u_const.append(u_consist_loss.item())


        load_t1 = time.time()
        print_train_KD_log(i, 5, [time.ctime(), opt.name, i_iter+1, opt.n_epoch, S_s_loss.item(), u_consist_loss.item(), load_t1 - load_t0])

    
    avg_loss_S_seg = np.array(avg_loss_S_seg)
    avg_loss_u_const = np.array(avg_loss_u_const)
    
    avg_loss_S_seg = avg_loss_S_seg.mean()
    avg_loss_u_const = avg_loss_u_const.mean()

    tmp = np.array([i_iter, avg_loss_S_seg, avg_loss_u_const])
    tmp = np.expand_dims(tmp, axis=0)
    losses_history = np.vstack((losses_history, tmp))    

    if (i_iter+1) % opt.save_freq == 0:
        torch.save(S_model.state_dict(), expr_dir + '/Student_%s_epoch%d.pth'%(opt.Snet, i_iter+1))    

np.save((expr_dir + '/%s_losses_history.npy'%(opt.name)), losses_history)

plt.figure('Losses History')
plt.plot(losses_history[:, 0], losses_history[:, 1], label='S_seg')
plt.plot(losses_history[:, 0], losses_history[:, 2], label='u_consist')
plt.title('losses history along with iteration')
plt.legend(loc='best')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.grid(True)
# plt.show()
plt.draw()
plt.savefig(expr_dir + '/%s_losses_history.png'%opt.name)
plt.close()





