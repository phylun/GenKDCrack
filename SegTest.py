
import cv2
import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from eval_pre_rec_f1 import evaluate_segmentation, compute_mean_iou

from utils.util import GetPallete
from models.network_selector import network_selection

from con_dataset import CONDataSet
from options.test_options import TestOptions


Testclass = TestOptions()
opt = Testclass.parse()

input_size =  (opt.crop_size, opt.crop_size) if opt.crop else (opt.image_size, opt.image_size)

if opt.phase == 'test':
    opt.testdataroot = opt.testdataroot.replace('valConc', 'testConc')
    opt.testlistfile = opt.testlistfile.replace('valConc', 'testConc')
    Testclass.print_options(opt)

result_dir = os.path.join(opt.results_dir, opt.name)    

super_test_folder = opt.testdataroot
super_test_list = os.path.join(super_test_folder, opt.testlistfile)


super_test_dataset = CONDataSet(super_test_folder, super_test_list, crop_size=input_size,
                                mirror=False, mean=np.array((0.0, 0.0, 0.0)))
super_test_dataset_size = len(super_test_dataset)
super_test_load = data.DataLoader(super_test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

model = network_selection(opt, sel_net=0)

model_folder = os.path.join(opt.save_dir, opt.name)
model_names = [f for f in os.listdir(model_folder) if f.find('Student') > -1]

sel_flag = False
if not opt.epoch == -1:        
    idx = [i for i, item in enumerate(model_names) if item.find('Student_%s_epoch%d.pth' %(opt.Snet, opt.epoch)) == 0]
    model_names = [model_names[idx[0]]]        
    sel_flag = True

score_list = list()
num_models = len(model_names)

cnt = 0
for model_name in model_names:
    saved_state_dict = torch.load(os.path.join(model_folder, model_name))
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(opt.gpu_ids[0])

    output_mat = list()
    gt_label_mat = list()

    for index, batch in enumerate(super_test_load):
        image, label, size, name = batch
        size = size[0].numpy()
        with torch.no_grad():
            x = image.cuda(opt.gpu_ids[0])
            output = model(x)
            output = F.softmax(output, dim=1).cpu().detach()[0].numpy()

            # print(output)
        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int32)

        
        output = output.transpose(1, 2, 0)        
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
        filename = os.path.join('Results', '{}.png'.format(name[0]))
        

        cimg = image.cpu().detach()[0].numpy()
        cimg = cimg.transpose(1, 2, 0)
        cimg = cimg[:, :, ::-1]
        cimg *= 255.0
        cimg = cimg.astype(np.uint8)

        gimg = cimg.copy()
        gt_label = label.cpu().detach()[0].numpy()
        g_pal = GetPallete(3, gt_label * 255, size[0], size[1])
        pal = g_pal.astype(np.uint8)
        gimg = np.add(gimg, (0.5 * pal))
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        gimg = gimg.astype(np.uint8)

        # Evaluation
        output_mat.append(output.flatten())
        gt_label_mat.append(gt_label.flatten())

        rimg = cimg.copy()        
        o_pal = GetPallete(1, output * 255, size[0], size[1])
        pal = o_pal.astype(np.uint8)
        rimg = np.add(rimg, (0.5 * pal))
        rimg = np.clip(rimg, 0, 255).astype(np.uint8)
        
        showimg = np.hstack((cimg, np.stack((g_pal[:, :, -1],) * 3, -1), np.stack((o_pal[:, :, 0],) * 3, -1))).astype(
            np.uint8)
        # showimg = np.hstack((cimg, gimg, rimg))
        # cv2.imshow('Result', showimg)
        # k = cv2.waitKey(0)
        # if k==27:
        #     quit()
        # cv2.imwrite(filename, showimg)
        # show_all(gt, output)
        
        
        if sel_flag:
            cv2.imwrite(os.path.join(result_dir, '%s_origin.jpg' % name[0]), cimg)
            cv2.imwrite(os.path.join(result_dir, '%s_gt.jpg' % name[0]), np.stack((g_pal[:, :, -1],) * 3, -1))
            cv2.imwrite(os.path.join(result_dir, '%s_pred.jpg' % name[0]), np.stack((o_pal[:, :, 0],) * 3, -1))

    filename = os.path.join('Results', 'result.txt')
    
    output_mat = np.array(output_mat)
    gt_label_mat = np.array(gt_label_mat)
    
    cnt += 1
    
    if sel_flag:
        print(model_name)    
        re_val = np.array(evaluate_segmentation(output_mat, gt_label_mat, num_classes=2))
        print('pixel_accuracy: {:2.2f}%'.format(re_val[0] * 100))
        print('precision: {:2.2f}%'.format(re_val[1] * 100))
        print('recall: {:2.2f}%'.format(re_val[2] * 100))
        print('f1: {:2.2f}%'.format(re_val[3] * 100))
        print('m-IoU: {:2.2f}%'.format(re_val[4] * 100))
        score_list.append([str(re_val[4]), model_name])    
    else:                                
        m_IOU = compute_mean_iou(output_mat, gt_label_mat)
        print('[%d / %d] model name: %s, m-IoU: %2.3f %%' % (cnt, num_models, model_name[:-4], m_IOU * 100))
        score_list.append([str(m_IOU), model_name])

if not sel_flag:
    with open(os.path.join(model_folder, '%s_mIoU_history.txt'%(opt.name)), 'w') as f:
        for score in score_list:
            str_score = 'model name: {:s}, m-IoU: {:s}% \n'.format(score[1], score[0])
            f.write(str_score) 

arr_score = np.array(score_list)
arr_score = list(map(float, arr_score[:, 0]))
max_idx = np.argmax(arr_score)
print(score_list[int(max_idx)])
print('\n\n\n')

