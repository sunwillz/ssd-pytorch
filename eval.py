"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function

import argparse
import logging
import os
import pickle

import numpy as np
import torch.backends.cudnn as cudnn
from models.FEDet import build_fedet
# from models.SSD_FFM import build_ssd
# from models.ssd import build_ssd
from models.SSD_RFM import build_ssd
from torch.autograd import Variable

from data import *

from utils.log_helper import init_log


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('-m', '--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('-a', '--arch', default='SSD', choices=['SSD', 'FEDet'])
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-s', '--size', default=300, type=int,
                    help='300 or 512 input size')
parser.add_argument('--top_k', default=100, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train models')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--coco_root', default=COCO_ROOT,
                    help='Location of COCO root directtory')
parser.add_argument('--reuse', default=False, type=str2bool,
                    help='evaluate the detection results file')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

init_log('global', logging.INFO)
logger = logging.getLogger("global")


def test_net(save_folder, net, cuda, dataset, top_k, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    _t = {'im_detect': Timer(), 'misc': Timer(), 'load_data': Timer()}
    det_file = os.path.join(save_folder, 'detections_bbox.pkl')

    for i in range(num_images):
        _t['load_data'].tic()
        img, h, w = dataset.pull_image(i)
        load_data_time = _t['load_data'].toc(average=False)
        x = Variable(img.unsqueeze(0))
        scale = torch.Tensor([w, h, w, h])
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=True)

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        assert detections.size(1) == num_classes, 'evaluate error!!!'
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            inds = np.where(dets[:, 0].cpu().numpy() > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes *= scale
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32, copy=False)
            # keep = nms2(boxes, scores)
            # all_boxes[j][i] = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets
        if top_k > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > top_k:
                image_thresh = np.sort(image_scores)[-top_k]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc(average=False)

        logger.info('im_detect: {:d}/{:d} || detect_time: {:.3f}s || nms_time: {:.3f}s '
                    '|| load data time:{:.3f}s'.format(i + 1,
                    num_images, detect_time, nms_time, load_data_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Evaluating detections')

    dataset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = (300, 512)[args.size == 512]
    num_classes = (21, 81)[args.dataset == 'COCO']
    dataset_mean = (104, 117, 123) # BGR layout
    if img_dim == 300:
        cfg = ((SSD_VOC_300, FEDet_VOC_300), (SSD_COCO_300, FEDet_COCO_300))[args.dataset == 'COCO'][args.arch == 'FEDet']
    else:
        cfg = ((SSD_VOC_512, FEDet_VOC_512), (SSD_COCO_512, FEDet_COCO_512))[args.dataset == 'COCO'][args.arch == 'FEDet']
    # load data
    if args.dataset == 'VOC':
        dataset = VOCDetection(VOC_ROOT, [('2007', 'test')],
                               BaseTransform(img_dim, dataset_mean),
                               VOCAnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCO_ROOT, [('2017', 'val')],
                                BaseTransform(img_dim, dataset_mean),
                                COCOAnnotationTransform())
    else:
        logger.error('Only VOC and COCO dataset are supported now!')
    if args.arch == 'SSD':
        net = build_ssd(cfg, 'test', img_dim, num_classes)            # initialize SSD
    elif args.arch == 'FEDet':
        net = build_fedet(cfg, 'test', img_dim, num_classes)
    else:
        logger.error('Architecture error!!!')
        SystemExit
    logger.info(net)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    logger.info('Finished loading models!')
    logger.info('Evaluating dataset size: %d' % len(dataset))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    if args.reuse:
        dataset.evaluate(args.save_folder)
    else:
        test_net(args.save_folder, net, args.cuda, dataset, args.top_k,
             thresh=args.confidence_threshold)
