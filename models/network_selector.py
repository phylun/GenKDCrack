from models.frrn import frrn
from models.LedNet import LEDNet
from models.CGNet import CGNet
from models.FDDWNet import FddwNet
from models.ERFNet import ERFNet
from models.DDRNet_39 import DualResNet_imagenet
from models.RegSeg import RegSeg
from models.pidnet import PIDNet
from models.deeplabv3p import DeepLabv3_plus


def network_selection(opt, sel_net=0):
    
    
    if sel_net == 1:
        NetName = opt.TnetA 
    elif sel_net == 2:
        NetName = opt.TnetB
    else:
        NetName = opt.Snet
    
        
    if NetName == 'FRRNA':
        model = frrn(n_classes=opt.output_nc, model_type="A")
    elif NetName == 'FRRNB':
        model = frrn(n_classes=opt.output_nc, model_type="B")    
    elif NetName == 'CGNet':
        model = CGNet(n_classes=opt.output_nc)
    elif NetName == 'LEDNet':
        if opt.phase == 'train':
            model = LEDNet(input_size=opt.crop_size, num_classes=opt.output_nc)
        else:
            model = LEDNet(input_size=opt.image_size, num_classes=opt.output_nc)
    elif NetName == 'FDDWNet':
        model = FddwNet(classes=opt.output_nc, in_ch=opt.input_nc)
    elif NetName == 'ERFNet':
        model = ERFNet(num_classes=opt.output_nc)
    elif NetName == 'DDRNet':
        model = DualResNet_imagenet()
    elif NetName == 'RegSeg':
        model = RegSeg("exp48_decoder26", opt.output_nc)
    elif NetName == 'PIDNet':
        model = PIDNet(m=3, n=4, num_classes=opt.output_nc, planes=64, ppm_planes=112, head_planes=256, augment=False)                
    elif NetName == 'Deeplabv3p':
        model = DeepLabv3_plus(nInputChannels=opt.input_nc, n_classes=opt.output_nc, os=16, pretrained=False, _print=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % NetName)
    
    return model
    
    