from data import *
from utils.augmentations import SSDAugmentation, Augmentation
from data.coco import COCOAnnotationTransform
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from utils.log_helper import init_log
import logging
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
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
parser.add_argument('--use_dataAug', default=True, type=str2bool,
                    help='use data augmentation trick or not')

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

init_log('global', logging.INFO)
logger = logging.getLogger("global")


def train():

    if args.img_dim == 300:
        cfg = (SSD_VOC_300, SSD_COCO_300)[args.dataset == 'COCO']
    else:
        cfg = (SSD_VOC_512, SSD_COCO_512)[args.dataset == 'COCO']

    if args.use_dataAug:
        train_transform = SSDAugmentation(cfg['min_dim'], MEANS)
    else:
        train_transform = Augmentation(cfg['min_dim'], MEANS)
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            logger.error("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        dataset = COCODetection(root=args.dataset_root, image_sets=[("2017", "train"), ],
                                transform=train_transform,
                                target_transform=COCOAnnotationTransform())
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')

        dataset = VOCDetection(root=args.dataset_root,
                               image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                               transform=train_transform)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd(cfg, 'train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    logger.info(net)
    logger.info('---------config---------')
    logger.info(cfg)
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        logger.info('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.pretrained_model+ args.basenet)
        logger.info('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        logger.info('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info('Training SSD on: %s ' % dataset.name)
    logger.info('Trainging images size: %d ' % len(dataset))
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
                                  shuffle=True, collate_fn=detection_collate,
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

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann) for ann in targets]
        # forward
        t0 = time.time()
        assert images.size(2) == args.img_dim and images.size(3) == args.img_dim
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_loc, loss_cls = criterion(out, targets)
        loss = loss_loc + loss_cls
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_loc.item()
        conf_loss += loss_cls.item()

        if iteration % 10 == 0:
            logger.info('iter ' + repr(iteration) + '/' + str(cfg['max_iter']) + ' || epoch: ' + str(epoch+1) + ' || LR: ' +
                  repr(optimizer.param_groups[0]['lr']) + ' || Loss: %.4f || ' % (loss.item()) +
                  'timer: %.4f sec.' % (t1 - t0))

        if args.visdom:
            update_vis_plot(viz, iteration, loss_loc.item(), loss_cls.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            logger.info('Saving state, iter: %d' % iteration)
            ckpt_path = os.path.join(args.save_folder,
                                     'ssd' + str(args.img_dim) + '_' + str(args.dataset) + '_' + str(iteration) + 'iter.pth')
            torch.save(ssd_net.state_dict(), ckpt_path)
    torch.save(ssd_net.state_dict(),
               os.path.join(args.save_folder, 'model.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_normal_(param)


def kaiming(param):
    init.kaiming_normal_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        kaiming(m.weight.data)
        m.bias.data.zero_()


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