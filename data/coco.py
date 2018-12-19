from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import pickle
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_ROOT = osp.join(HOME, 'data/coco/')
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1]) - 1
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, targets, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in targets:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']]
                final_box = list(np.array(bbox) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]

    def aux_label_trans(self, targets):
        res = [0] * (len(COCO_CLASSES) + 1)
        for obj in targets:
            if 'bbox' in obj:
                cls = self.label_map[obj['category_id']]
                res[cls+1] = 1
        return np.array(res)


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_sets=[("2017", "train"),], transform=None,
                 target_transform=None, aux=False, dataset_name='MS COCO'):
        sys.path.append(osp.join(root, COCO_API))
        self.root = root
        self.cache_path = osp.join(root, 'cache')
        self.name = dataset_name
        self.aux = aux
        # self.img_paths = list()
        # self.annotations = list()
        # self.anno_sets = list()
        self.transform = transform
        self.target_transform = target_transform
        self.coco_name = image_sets[0][1]+image_sets[0][0]
        self.coco = COCO(self.get_anno_filename(self.coco_name, image_sets[0][1]))
        cats = self.coco.loadCats(self.coco.getCatIds())
        if 'test' in image_sets[0][1]:
            self.img_ids = self.coco.getImgIds()
        else:
            self.img_ids = [i for i in self.coco.getImgIds() if len(self.coco.imgToAnns[i]) > 0]
        self.classes_name = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self.classes_name)
        self.class_to_ind = dict(zip(self.classes_name, range(self.num_classes)))
        self.class_to_cat_id = dict(zip([c['name'] for c in cats], self.coco.getCatIds()))

        # for (year, image_set) in image_sets:
        #     coco_name = image_set + year
        #     annofile = self.get_anno_filename(coco_name)
        #     _COCO = COCO(annofile)
        #     self._COCO = _COCO
        #     self.coco_name = coco_name
        #     cats = _COCO.loadCats(_COCO.getCatIds())
        #     self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        #     self.num_classes = len(self._classes)
        #     self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        #     self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
        #                                           _COCO.getCatIds()))
        #     indexes = _COCO.getImgIds()
        #     n_indexes = [i for i in indexes if len(_COCO.imgToAnns[i]) > 0]
        #     self.anno_sets += [_COCO.imgToAnns[i] for i in n_indexes]
        #     self.img_ids += n_indexes
        #     self.img_paths += [self.image_path_from_index(coco_name, index) for index in n_indexes]
        #     if image_set.find('test') != -1:
        #         print('test set will not load annotations')
        #     else:
        #         self.annotations += self._load_coco_annotations(coco_name, n_indexes, _COCO)

    def get_anno_filename(self, name, split):
        if split == 'val' or split == 'train':
            prefix = 'instances'
            return osp.join(self.root, 'annotations', prefix + '_' + name + '.json')
        elif split == 'test-dev':
            return osp.join(self.root, 'annotations', 'image_info_' + name + '.json')

    def image_path_from_index(self, name, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/000000119993.jpg
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = os.path.join(self.root, 'images',
                                 'test2017' if 'test' in name else name, file_name)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)

        return image_path

    # def _load_coco_annotations(self, coco_name, indexes, _COCO):
    #     cache_file = os.path.join(self.cache_path, coco_name + '_gt_roidb.pkl')
    #     if os.path.exists(cache_file):
    #         with open(cache_file, 'rb') as fid:
    #             roidb = pickle.load(fid)
    #         print('{} gt roidb loaded from {}'.format(coco_name, cache_file))
    #         return roidb
    #
    #     gt_roidb = [self._annotation_from_index(index, _COCO)
    #                 for index in indexes]
    #     with open(cache_file, 'wb') as fid:
    #         pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    #     print('wrote gt roidb to {}'.format(cache_file))
    #
    #     return gt_roidb
    #
    # def _annotation_from_index(self, index, _COCO):
    #     """
    #     Loads COCO bounding-box instance annotations. Crowd instances are
    #     handled by marking their overlaps (with all categories) to -1. This
    #     overlap value means that crowd "instances" are excluded from training.
    #     """
    #     im_ann = _COCO.loadImgs(index)[0]
    #     width = im_ann['width']
    #     height = im_ann['height']
    #     scale = [width, height, width, height]
    #     annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
    #     objs = _COCO.loadAnns(annIds)
    #     # Sanitize bboxes -- some are invalid
    #     valid_objs = []
    #     for obj in objs:
    #         x1 = np.max((0, obj['bbox'][0]))
    #         y1 = np.max((0, obj['bbox'][1]))
    #         x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
    #         y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
    #         if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
    #             obj['clean_bbox'] = [x1, y1, x2, y2]
    #             valid_objs.append(obj)
    #     objs = valid_objs
    #     num_objs = len(objs)
    #
    #     res = np.zeros((num_objs, 5))
    #
    #     # Lookup table to map from COCO category ids to our internal class
    #     # indices
    #     coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
    #                                       self._class_to_ind[cls])
    #                                      for cls in self._classes[1:]])
    #
    #     for ix, obj in enumerate(objs):
    #         cls = coco_cat_id_to_class_ind[obj['category_id']]
    #         res[ix, 0:4] = list(np.array(obj['clean_bbox']) / scale)
    #         res[ix, 4] = cls
    #
    #     return res

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        ## method 1
        # img_id = self.img_paths[index]
        # target = self.annotations[index] # shape:(N, 5)
        # assert len(self.img_paths) == len(self.img_ids)
        # target = self.anno_sets[index]
        ## method 2
        # img_id = self.img_ids[index]
        # target = self.coco.imgToAnns[img_id]
        # path = self.image_path_from_index(self.coco_name, img_id)
        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        # height, width, _ = img.shape
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target, width, height)
        # if self.transform is not None:
        #     target = np.array(target)
        #     img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
        #     target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        #
        # return img, torch.from_numpy(target).float()
        if self.aux:
            img, gt, h, w, aux_label = self.pull_item(index)
            return img, gt, aux_label
        else:
            img, gt, h, w = self.pull_item(index)
            return img, gt

    def pull_item(self, index):
        img_id = self.img_ids[index]
        target = self.coco.imgToAnns[img_id]
        path = self.image_path_from_index(self.coco_name, img_id)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            if self.aux:
                aux_label = self.target_transform.aux_label_trans(target)
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        if self.aux:
            return img, torch.from_numpy(target).float(), \
                   height, width, aux_label
        else:
            return img, torch.from_numpy(target).float(),\
                   height, width

    def __len__(self):
        return len(self.img_ids)

    def pull_image(self, index):
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img, height, width
        """
        img_id = self.img_ids[index]
        path = self.image_path_from_index(self.coco_name, img_id)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if self.transform:
            img, _, _ = self.transform(img)
            img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1).float(), height, width

    def pull_anno(self, index):
        """Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def pull_tensor(self, index):
        """Returns the original image at an index in tensor form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
              '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes_name):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.img_ids):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes_name):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes-1))
            coco_cat_id = self.class_to_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir,
                            ('detections_' + self.coco_name + '_results'))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)

    def evaluate(self, output_dir):
        res_file = os.path.join(output_dir, 'detections_' + self.coco_name + '_results.json')
        self._do_detection_eval(res_file, output_dir)