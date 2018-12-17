import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os


class FEDet(nn.Module):
    """
    Feature Enhanced single shot detector base on SSD
    The network is conposed of a base VGG network followed by the
    added multiboc conv layers.
    Three added modules:
        semantic supervision module
        feature fusion module
        receptive field module

    Args:
        phase: (string) can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox  head" consists of loc and cong conv layers
    """

    def __init__(self, cfg, phase, size, num_classes, base, extras, head):
        super(FEDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.L2Norm = L2Norm(512, 20)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg, 0, 200, 0.01, 0.45)

        self.c1_lateral = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.c2_lateral = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.c3_lateral = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.p1_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.p2_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.p3_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.p4_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        if self.size == 512:
            self.p5_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.p6_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.p1_RFM = RFM(256, channels[str(size)][0], 0.25)
        self.p2_RFM = RFM(256, channels[str(size)][1], 0.25)
        if self.size == 300:
            self.p3_RFM = RFM(256, channels[str(size)][2], 0.5) # 300->0.5, 512->0.25
        else:
            self.p3_RFM = RFM(256, channels[str(size)][2], 0.25)
        self.p4_RFM = RFM(256, channels[str(size)][3], 0.5)
        self.p5_RFM = RFM(256, channels[str(size)][4], 0.5)
        if self.size == 512:
            self.p6_RFM = RFM(256, channels[str(size)][5], 0.5)
            self.p7_RFM = RFM(256, channels[str(size)][6], 0.5)

        self.p1_SSM = SSM(512, self.num_classes)
        self.p2_SSM = SSM(1024, self.num_classes)

        for layer in self.vgg:
            init_weights_msra(layer)

        for layer in self.extras:
            init_weights_msra(layer)

        for layer in self.loc:
            init_weights_msra(layer)

        for layer in self.conf:
            init_weights_msra(layer)

    def _upsample_add(self, x, y):
        # the size of upsample torch may not be same as the lateral conv
        b, c, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def feature_fusion(self, f1, f2, f3, f4, f5, f6=None, f7=None):
        if self.size == 512:
            out_f7 = f7

            f7_upsampled = self._upsample_add(f7, f6)
            out_f6 = self.p6_conv(f7_upsampled)

            f6_upsampled = self._upsample_add(out_f6, f5)
            out_f5 = self.p5_conv(f6_upsampled)
            f5_upsampled = self._upsample_add(out_f5, f4)
        else:
            f5_upsampled = self._upsample_add(f5, f4)

        if self.size == 300:
            out_f5 = f5

        out_f4 = self.p4_conv(f5_upsampled)

        p3 = self.c3_lateral(f3)
        f4_upsampled = self._upsample_add(out_f4, p3)
        out_f3 = self.p3_conv(f4_upsampled)

        p2 = self.c2_lateral(f2)
        f3_upsampled = self._upsample_add(out_f3, p2)
        out_f2 = self.p2_conv(f3_upsampled)

        p1 = self.c1_lateral(f1)
        f2_upsampled = self._upsample_add(out_f2, p1)
        out_f1 = self.p1_conv(f2_upsampled)
        if self.size == 300:
            return out_f1, out_f2, out_f3, out_f4, out_f5
        else:
            return out_f1, out_f2, out_f3, out_f4, out_f5, out_f6, out_f7

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train_FEDet:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)    ## f1
        output_ssm1 = self.p1_SSM(x)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)    ## f2
        output_ssm2 = self.p2_SSM(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)  ## f3, f4, f5, (f6, f7)
        if self.size == 300:
            assert len(sources) == 5
            f1, f2, f3, f4, f5 = self.feature_fusion(sources[0], sources[1],
                                                 sources[2], sources[3], sources[4])
        else:
            assert len(sources) == 7
            f1, f2, f3, f4, f5, f6, f7 = self.feature_fusion(sources[0], sources[1],
                                                 sources[2], sources[3], sources[4], sources[5], sources[6])
        sources.clear()
        assert len(sources) == 0
        sources.append(self.p1_RFM(f1))
        sources.append(self.p2_RFM(f2))
        sources.append(self.p3_RFM(f3))
        sources.append(self.p4_RFM(f4))
        sources.append(self.p5_RFM(f5))
        if self.size == 512:
            sources.append(self.p6_RFM(f6))
            sources.append(self.p7_RFM(f7))

        ## apply multibox head to source layers
        for (idx, x, l, c) in zip(range(len(sources)), sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),
                self.priors.type(type(x.data))
            )
        else:

            output = (
                output_ssm1,
                output_ssm2,
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weight into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished')
        else:
            print('Sorry only .pth or .pkl files supported.')


def init_weights(module, std=0.01):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, std)


def init_weights_xavier(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight.data)
            m.bias.data.zero_()


def init_weights_msra(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()


def backbone():
    layers = [
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),  # stride=2
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),  # stride=4
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),  # stride=8
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # feature map=38
        nn.ReLU(inplace=True), ## conv4_3 relu
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),  # stride=16
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1)),
        nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6)),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),  # feature map=19
        nn.ReLU(inplace=True)]
    return layers


def extras(size=300):
    layers = [
        nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
        # stride=32, feature map=10
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
        # stride=64, feature map=5
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
        # stride=100, feature map=3
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
        # nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
        # # stride=300, feature map=1
        # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
    ]

    if size == 512:
        layers += [nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                   nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))]

    return layers


def multibox(feature_maps_channel, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(feature_maps_channel):
        loc_layers += [nn.Conv2d(v, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers


class SSM(nn.Module):
    """
    high-level semantic supervision module attached on feature map 38 and 19
    """

    def __init__(self, in_planes, num_classes):
        super(SSM, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_planes, in_planes * 2, kernel_size=3, padding=1, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_planes * 2, in_planes)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(in_planes, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFM(nn.Module):
    def __init__(self, in_planes, out_planes, scale=0.25, stride=1):
        super(RFM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        mid_planes = int(in_planes * scale)
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, mid_planes, kernel_size=1, stride=stride),
            BasicConv(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, mid_planes, kernel_size=1, stride=1),
            BasicConv(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, mid_planes, kernel_size=1, stride=1),
            BasicConv(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=5, dilation=5, relu=False)
        )
        self.conv1x1 = BasicConv(int(4 * scale * in_planes), out_planes, kernel_size=1, stride=1, relu=False)
        self.shotcut = BasicConv(in_planes, mid_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        shotcut = self.shotcut(x)

        x = torch.cat((x0, x1, x2, shotcut), 1)
        x = self.conv1x1(x)
        x = self.relu(x)

        return x


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4, padding=1)]
    return layers


channels = {
    '300': [128, 128, 128, 128, 128],
    '512': [128, 128, 128, 128, 128, 128, 128]
}
# channels = [256, 256, 256, 256, 256]
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
}
mbox = {
    '300': [4, 6, 6, 6, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4]
}


def build_fedet(cfg, phase, size=300, num_classes=21):
    if phase != 'test' and phase != 'train':
        print("ERROR: Phase " + phase + "not recongized")
        return
    if size != 300 and size != 512:
        print("ERROR: Sorry only FEDet300 or FEDet512 is supported")
        return
    return FEDet(cfg, phase, size, num_classes,
                 base=vgg(base[str(size)], 3),
                 extras=add_extras(extras[str(size)], size, 1024),
                 head=multibox(channels[str(size)], mbox[str(size)], num_classes))
