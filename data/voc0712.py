"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import torch
import torch.utils.data as data
import cv2
from data.util import *

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

    def aux_label_trans(self, targets):
        res = [0] * (len(VOC_CLASSES) + 1)
        objs = targets.findall('object')
        for obj in objs:
            cls = self.class_to_ind[obj.find('name').text.lower().strip()]
            res[cls+1] = 1
        return np.array(res)


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(), aux=False,
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.db_name = image_sets[0][0] + image_sets[0][1]
        self.aux = aux
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.imgsetpath = osp.join(VOC_ROOT, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
        self.year = image_sets[0][0]
        self.devkit_path = osp.join(VOC_ROOT, 'VOC_det_results', self.year)
        self.set_type = image_sets[0][1]
        self.ids = list()
        for (year, name) in image_sets:
            if 'test' in name:
                rootpath = osp.join(self.root, 'VOC' + year+ 'test')
            else:
                rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        if self.aux:
            im, gt, h, w, aux_label = self.pull_item(index)
            return im, gt, aux_label
        else:
            im, gt, h, w = self.pull_item(index)
            return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            if self.aux:
                aux_label = self.target_transform.aux_label_trans(target)
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        if self.aux:
            return img, target, height, width, aux_label
        else:
            return img, target, height, width

    def aux_target(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            aux = self.target_transform.aux_label_trans(target)

        return aux

    def pull_image(self, index):
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_id = self.ids[index]

        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        if self.transform:
            img, _, _ = self.transform(img)
            img = img[:, :, (2, 1, 0)] # to RGB layout
        img = torch.from_numpy(img).permute(2, 0, 1).float() # to torch tensor, CxHxW
        return img, height, width

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
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        """Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def get_output_dir(self, name, phase):
        """Return the directory where experimental artifacts are placed.
        If the directory does not exist, it is created.
        A canonical path is built using the name from an imdb and a network
        (if not None).
        """
        filedir = osp.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir

    def get_voc_results_file_template(self, image_set, cls, output_dir):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'comp3_det_' + image_set + '_%s.txt' % (cls)
        # filedir = os.path.join(self.devkit_path, 'results')
        if 'test' == image_set:
            output_dir = os.path.join(output_dir, 'results/VOC2012/Main')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, filename)
        return path

    def write_voc_results_file(self, all_boxes, output_dir):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(self.set_type, cls, output_dir)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    dets = all_boxes[cls_ind+1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def do_python_eval(self, output_dir='output'):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        rootpath = osp.join(VOC_ROOT, 'VOC'+self.year)
        annopath = osp.join(rootpath, 'Annotations', '%s.xml')

        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(VOC_CLASSES):
            filename = self.get_voc_results_file_template(self.set_type, cls, output_dir)
            rec, prec, ap = voc_eval(
               filename, annopath, self.imgsetpath.format(self.set_type), cls, cachedir,
               ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir=None):

        self.write_voc_results_file(all_boxes, output_dir)
        if 'test' not in self.db_name:
            self.do_python_eval(output_dir)

    def evaluate(self, output_dir=None):
        self.do_python_eval(output_dir)