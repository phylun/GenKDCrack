# from blocks import *
# from competitor_blocks import BiseNetDecoder,SFNetDecoder,FaPNDecoder

from torchvision.ops import DeformConv2d
# from benchmark import benchmark_eval,benchmark_train,benchmark_memory
from torch import nn
import torch
from torch.nn import functional as F

def activation():
    return nn.ReLU(inplace=True)

def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2,2,ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self,w,dilations,group_width,stride,bias):
        super().__init__()
        num_splits=len(dilations)
        assert(w%num_splits==0)
        temp=w//num_splits
        assert(temp%group_width==0)
        groups=temp//group_width
        convs=[]
        for d in dilations:
            convs.append(nn.Conv2d(temp,temp,3,padding=d,dilation=d,stride=stride,bias=bias,groups=groups))
        self.convs=nn.ModuleList(convs)
        self.num_splits=num_splits
    def forward(self,x):
        # x=torch.tensor_split(x,self.num_splits,dim=1)
        x=torch.chunk(x,self.num_splits,dim=1)
        res=[]
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res,dim=1)

class ConvBnActConv(nn.Module):
    def __init__(self,w,stride,dilation,groups,bias):
        super().__init__()
        self.conv=ConvBnAct(w,w,3,stride,dilation,dilation,groups)
        self.project=nn.Conv2d(w,w,1,bias=bias)
    def forward(self,x):
        x=self.conv(x)
        x=self.project(x)
        return x


class YBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,group_width, stride):
        super(YBlock, self).__init__()
        groups = out_channels // group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        self.se=SEModule(out_channels,in_channels//4)
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"):
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        if len(dilations)==1:
            dilation=dilations[0]
            self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        else:
            self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        if attention=="se":
            self.se=SEModule(out_channels,in_channels//4)
        elif attention=="se2":
            self.se=SEModule(out_channels,out_channels//4)
        else:
            self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        if self.se is not None:
            x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class Exp2_LRASPP(nn.Module):
    # LRASPP
    def __init__(self, num_classes,channels,inter_channels=128):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.cbr=ConvBnAct(channels16,inter_channels,1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels16, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(channels8, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x = self.cbr(x16)
        s = self.scale(x16)
        x = x * s
        x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x= self.low_classifier(x8) + self.high_classifier(x)
        return x
class Exp2_Decoder4(nn.Module):
    def __init__(self, num_classes,channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head8=ConvBnAct(channels8,32,1)
        self.head16=ConvBnAct(channels16,128,1)
        self.conv=ConvBnAct(128+32,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x=torch.cat((x8, x16), dim=1)
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder10(nn.Module):
    def __init__(self, num_classes,channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head8=ConvBnAct(channels8,32,1)
        self.head16=ConvBnAct(channels16,128,1)
        self.conv=DBlock(128+32,128,[1],16,1,"se")
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x=torch.cat((x8, x16), dim=1)
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder12(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16" ]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.conv=ConvBnAct(128,128,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x= x8 + x16
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder14(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.conv=ConvBnAct(128,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x= x8 + x16
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder26(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class Exp2_Decoder29(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 256, 1)
        self.head8=ConvBnAct(channels8, 256, 1)
        self.head4=ConvBnAct(channels4, 16, 1)
        self.conv8=ConvBnAct(256,128,3,1,1)
        self.conv4=ConvBnAct(128+16,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

def generate_stage(num,block_fun):
    blocks=[]
    for _ in range(num):
        blocks.append(block_fun())
    return blocks
class RegNetY600MF(nn.Module):
    def __init__(self):
        super().__init__()
        group_width=16
        self.stage4=YBlock(32,48,1,group_width,2)
        self.stage8=nn.Sequential(
            YBlock(48, 112, 1, group_width, 2),
            YBlock(112, 112, 1, group_width, 1),
            YBlock(112, 112, 1, group_width, 1)
        )
        self.stage16=nn.Sequential(
            YBlock(112, 256, 1, group_width, 2),
            *generate_stage(6, lambda : YBlock(256,256, 1, group_width, 1))
        )
        self.stage32=nn.Sequential(
            YBlock(256, 608, 1, group_width, 1),
            *generate_stage(3, lambda : YBlock(608,608, 2, group_width, 1))
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x16=self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":112,"16":608}

def generate_stage2(ds,block_fun):
    blocks=[]
    for d in ds:
        blocks.append(block_fun(d))
    return blocks
class RegSegBody(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=16
        attention="se"
        self.stage4=DBlock(32, 48, [1], gw, 2, attention)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2, attention),
            DBlock(128, 128, [1], gw, 1, attention),
            DBlock(128, 128, [1], gw, 1, attention)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention)),
            DBlock(256, 320, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":320}

class RegSegBody2(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=24
        attention="se"
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2, attention),
            DBlock(48, 48, [1], gw, 1, attention),
        )
        self.stage8=nn.Sequential(
            DBlock(48, 120, [1], gw, 2, attention),
            *generate_stage(5,lambda: DBlock(120, 120, [1], gw, 1, attention)),
        )
        self.stage16=nn.Sequential(
            DBlock(120, 336, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(336, 336, d, gw, 1, attention)),
            DBlock(336, 384, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":120,"16":384}

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, out_channels=128, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        convs = []
        for size in sizes:
            convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(size, size)),
                    ConvBnAct(in_channels,out_channels,apply_act=False)
                )
            )
        self.stages=nn.ModuleList(convs)
        self.bottleneck=ConvBnAct(in_channels+len(sizes)*out_channels,out_channels)
        self.dropout=nn.Dropout2d(0.1)

    def forward(self, x):
        y=[x]
        for stage in self.stages:
            z=stage(x)
            z=F.interpolate(z,size=x.shape[-2:],align_corners=False,mode="bilinear")
            y.append(z)
        x=torch.cat(y,1)
        x = self.bottleneck(x)
        return x
class AlignedModule(nn.Module):
    #SFNet-DFNet
    def __init__(self, inplane, outplane):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size,mode="bilinear",align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid,align_corners=False)
        return output

class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2=nn.Sigmoid()

    def forward(self, x):
        y=self.avg_pool(x)
        y=self.act1(self.conv1(y))
        y=self.act2(self.conv2(y))
        return x * y

class FeatureFusionModule(nn.Module):
    # BiseNet
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.conv=ConvBnAct(in_chan, out_chan)
        self.se=SEModule(out_chan,out_chan//4)

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = self.conv(x)
        x= self.se(x)+x
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv=ConvBnAct(in_chan,out_chan,3,1,1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        atten=self.scale(x)
        x=x*atten
        return x

class FeatureSelectionModule(nn.Module):
    # FaPN paper
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)

    def forward(self, x):
        x = x*self.conv_atten(x) + x
        x = self.conv(x)
        return x

class DeformConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        offset_out_channels=2*deformable_groups*kernel_size[0]*kernel_size[1]
        mask_out_channels=deformable_groups*kernel_size[0]*kernel_size[1]
        self.deform_conv=DeformConv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=True)
        self.offset_conv=nn.Conv2d(in_channels,offset_out_channels,kernel_size,stride,padding,bias=True)
        self.mask_conv=nn.Conv2d(in_channels,mask_out_channels,kernel_size,stride,padding,bias=True)
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()
        self.mask_conv.weight.data.zero_()
        self.mask_conv.bias.data.zero_()
    def forward(self,x):
        x,x2=x
        offset=self.offset_conv(x2)
        mask=self.mask_conv(x2)
        mask = torch.sigmoid(mask)
        x=self.deform_conv(x,offset,mask)
        return x

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc)
        self.project=ConvBnAct(out_nc * 2, out_nc,apply_act=False)
        self.dcpack_L2 = DeformConv(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_l, feat_s):
        feat_up = F.interpolate(feat_s, feat_l.shape[-2:], mode='bilinear', align_corners=False)
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.project(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))
        return feat_align, feat_arm

class BiseNetDecoder(nn.Module):
    def __init__(self, num_classes, channels):
        super(BiseNetDecoder, self).__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.arm16 = AttentionRefinementModule(channels16, 128)
        self.conv_head16 = ConvBnAct(128,128,3,1,1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv_avg = ConvBnAct(channels16,128)
        self.ffm=FeatureFusionModule(128+channels8,128)
        self.conv=ConvBnAct(128,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x8,x16= x["8"], x["16"]

        avg=self.avg_pool(x16)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=x16.shape[-2:], mode='nearest')

        x16 = self.arm16(x16)
        x16 = x16 + avg_up
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='nearest')
        x16 = self.conv_head16(x16)

        x=self.ffm(x8,x16)
        x=self.conv(x)
        x=self.classifier(x)
        return x

class SFNetDecoder(nn.Module):
    def __init__(self, num_classes, channels, fpn_dim=64, fpn_dsn=False):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head16 = PSPModule(channels16,fpn_dim)
        self.head8=ConvBnAct(channels8,fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_align=AlignedModule(inplane=fpn_dim, outplane=fpn_dim//2)
        self.conv=ConvBnAct(fpn_dim, fpn_dim, 3, 1, 1)
        self.conv_last = nn.Sequential(
            ConvBnAct(2*fpn_dim,fpn_dim,3,1,1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x8,x16= x["8"], x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x16_up = self.fpn_align([x8, x16])
        x8 = x8 + x16_up
        x8=self.conv(x8)
        x16_up=F.interpolate(x16, x8.shape[-2:], mode="bilinear", align_corners=True)
        x8=torch.cat([x8,x16_up],dim=1)
        x = self.conv_last(x8)
        return x

class FaPNDecoder(nn.Module):
    # FaPN paper
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.align=FeatureAlign_V2(channels8,128)
        self.conv8=ConvBnAct(256,128,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16= x["8"],x["16"]
        x16=self.head16(x16)
        x8, x16_up=self.align(x8,x16)
        x8=torch.cat([x8,x16_up],dim=1)
        x8=self.conv8(x8)
        return self.classifier(x8)

class RegSeg(nn.Module):
    # exp48_decoder26 is what we call RegSeg in our paper
    # exp53_decoder29 is a larger version of exp48_decoder26
    # all the other models are for ablation studies
    def __init__(self, name, num_classes, pretrained="", ablate_decoder=False,change_num_classes=False):
        super().__init__()
        self.stem=ConvBnAct(3,32,3,2,1)
        body_name, decoder_name=name.split("_")
        if "exp30" == body_name:
            self.body=RegSegBody(5*[[1,4]]+8*[[1,10]])
        elif "exp43"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,12]])
        elif "exp46"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8]]+8*[[1,10]])
        elif "exp47"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10],[1,12]]+6*[[1,14]])
        elif "exp48"==body_name:
            self.body=RegSegBody([[1],[1,2]]+4*[[1,4]]+7*[[1,14]])
        elif "exp49"==body_name:
            self.body=RegSegBody([[1],[1,2]]+6*[[1,4]]+5*[[1,6,12,18]])
        elif "exp50"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,3,6,12]])
        elif "exp51"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,4,8,12]])
        elif "exp52"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4]]+10*[[1,6]])
        elif "exp53"==body_name:
            self.body=RegSegBody2([[1],[1,2]]+4*[[1,4]]+7*[[1,14]])
        elif "regnety600mf"==body_name:
            self.body=RegNetY600MF()
        else:
            raise NotImplementedError()
        if "decoder4" ==decoder_name:
            self.decoder=Exp2_Decoder4(num_classes,self.body.channels())
        elif "decoder10" ==decoder_name:
            self.decoder=Exp2_Decoder10(num_classes,self.body.channels())
        elif "decoder12" ==decoder_name:
            self.decoder=Exp2_Decoder12(num_classes,self.body.channels())
        elif "decoder14"==decoder_name:
            self.decoder=Exp2_Decoder14(num_classes,self.body.channels())
        elif "decoder26"==decoder_name:
            self.decoder=Exp2_Decoder26(num_classes,self.body.channels())
        elif "decoder29"==decoder_name:
            self.decoder=Exp2_Decoder29(num_classes,self.body.channels())
        elif "BisenetDecoder"==decoder_name:
            self.decoder=BiseNetDecoder(num_classes,self.body.channels())
        elif "SFNetDecoder"==decoder_name:
            self.decoder=SFNetDecoder(num_classes,self.body.channels())
        elif "FaPNDecoder"==decoder_name:
            self.decoder=FaPNDecoder(num_classes,self.body.channels())
        else:
            raise NotImplementedError()
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)
    def forward(self,x):
        input_shape=x.shape[-2:]
        x=self.stem(x)
        x=self.body(x)
        x=self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x




if __name__ == "__main__":
    # cityscapes_speed_test()
    print('RegSeg test')
    regseg = RegSeg("exp48_decoder26", 2)
    regseg.eval()
    regseg.cuda(0)
    img = torch.rand(1, 3, 448, 448).cuda(0)
    out = regseg(img)
    print(out.size())

