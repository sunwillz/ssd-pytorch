from data import *
from utils.augmentations import SSDAugmentation
from utils.log_helper import init_log
from data.coco import COCOAnnotationTransform
from layers.modules import MultiBoxLoss
from fedet import build_fedet
import os
import logging
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Feature enhanced Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--img_dim', default=300, type=int,
                    help='Size of the input image, only support 300 or 512')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--pretrained_model', default='weights/',
                    help='Directory for saving pretrained model')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')

parser.add_argument('--arch', default='FEDet', choices=['SSD', 'FEDet'],
                    help='Architecture: SSD or FEDet')
parser.add_argument('--use_aux', default=False, type=str2bool,
                    help='Use high-level semantic supervisied')
parser.add_argument('--use_rfm', default=False, type=str2bool,
                    help='Use Receptive field module')
parser.add_argument('--use_feature_fusion', default=False, type=str2bool,
                    help='Use FPN like feature fusion module')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    init_log('global', logging.INFO)
    logger = logging.getLogger("global")
    if args.img_dim == 300:
        cfg = ((SSD_VOC_300, FEDet_VOC_300), (SSD_COCO_300, FEDet_COCO_300))[args.dataset == 'COCO'][args.arch == 'FEDet']
    else:
        cfg = ((SSD_VOC_512, FEDet_VOC_512), (SSD_COCO_512, FEDet_COCO_512))[args.dataset == 'COCO'][args.arch == 'FEDet']
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            logger.warning("WARNING: Using default COCO dataset_root because " +
                           "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        dataset = COCODetection(root=args.dataset_root, image_sets=[("2017", "train")],
                                transform=SSDAugmentation(cfg['min_dim'], MEANS),
                                target_transform=COCOAnnotationTransform(),
                                aux=args.use_aux)
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')

        args.dataset_root = VOC_ROOT
        dataset = VOCDetection(root=args.dataset_root,image_sets=[('2007', 'trainval'), ('2007', 'test'), ('2012', 'trainval')],
                               transform=SSDAugmentation(cfg['min_dim'], MEANS),
                               aux=args.use_aux)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    if args.arch == 'FEDet':
        build_net = build_fedet(cfg, 'train', cfg['min_dim'], cfg['num_classes'])
    else:
        logger.error('architenture error!!!')
        return
    net = build_net
    logger.info(net)
    logger.info('---------config-----------')
    logger.info(cfg)
    if args.cuda:
        net = torch.nn.DataParallel(build_net)
        cudnn.benchmark = True

    if args.resume:
        logger.info('Resuming training, loading {}...'.format(args.resume))
        build_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.pretrained_model + args.basenet)
        logger.info('Loading base network...')
        build_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        logger.info('Initializing weights...')

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        # initialize newly added layers' weights with xavier method
        build_net.extras.apply(weights_init)
        build_net.loc.apply(weights_init)
        build_net.conf.apply(weights_init)

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion1 = MultiBoxLoss(cfg, 0.5, True, 0, True, 3, 0.5,
                              False, args.cuda)
    criterion2 = nn.BCELoss(size_average=True).cuda()

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    ssm_loss = 0  ## SSM loss counter
    epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info('Training FEDet on: %s' % dataset.name)
    logger.info('Trainging images size: %d' % len(dataset))
    logger.info('Using the specified args:')
    logger.info(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(viz, 'Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate_fedet,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(viz, epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            ssm_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            if args.use_aux:
                images, targets, aux_targets = next(batch_iterator)
            else:
                images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            if args.use_aux:
                images, targets, aux_targets = next(batch_iterator)
            else:
                images, targets = next(batch_iterator)
        if images.size(0) < args.batch_size:
            continue
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
            if args.use_aux:
                aux_targets = Variable(aux_targets.cuda())
        else:
            images = Variable(images)
            targets = [Variable(ann) for ann in targets]
            if args.use_aux:
                aux_targets = Variable(aux_targets)
        # forward
        t0 = time.time()
        assert images.size(2) == args.img_dim and images.size(3) == args.img_dim
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_loc, loss_cls = criterion1(out[2:], targets)
        loss_ssm1 = criterion2(out[0], aux_targets)
        loss_ssm2 = criterion2(out[1], aux_targets)
        loss = loss_loc + loss_cls + loss_ssm1.double() + loss_ssm2.double()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss = loss_loc.item()
        conf_loss = loss_cls.item()
        ssm_loss = loss_ssm1.item() + loss_ssm2.item()

        if iteration % 10 == 0:
            logger.info(
                'iter ' + repr(iteration) + '/' + str(cfg['max_iter']) + ' || epoch: ' + str(epoch + 1) + ' || LR: ' +
                repr(optimizer.param_groups[0]['lr']) +
                ' || total loss: %.4f || loc Loss: %.4f || conf Loss: %.4f || SSM loss: %.4f || ' %
                (loss.item(), loc_loss, conf_loss, ssm_loss) +
                'timer: %.4f sec.' % (t1 - t0))

        if args.visdom:
            update_vis_plot(viz, iteration, loss_loc.item(), loss_cls.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 50000 == 0:
            logger.info('Saving state, iter: %d' % iteration)
            ckpt_path = os.path.join(args.save_folder,
                                     args.arch + str(args.img_dim) + '_' + str(args.dataset) + '_' + str(
                                         iteration) + '.pth')
            torch.save(build_net.state_dict(), ckpt_path)
    torch.save(build_net.state_dict(), os.path.join(args.save_folder, 'model.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         # init.xavier_normal_(m.weight.data)
#         init.kaiming_normal_(m.weight.data)
#         m.bias.data.zero_()


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
