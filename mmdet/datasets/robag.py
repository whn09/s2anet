from .registry import DATASETS
from .xml_style import XMLDataset
import os

import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from mmdet.core import norm_angle
from DOTA_devkit.ResultMerge_multi_process import mergebypoly
from DOTA_devkit.dota_evaluation_task1 import voc_eval
from mmdet.core import rotated_box_to_poly_single

@DATASETS.register_module
class RobagDataset(XMLDataset):

    CLASSES = ('bag',)

    def __init__(self, **kwargs):
        super(RobagDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
#             raise ValueError('Cannot infer dataset year from img_prefix')
            self.year = 2021
    
    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        self.img_ids = img_ids
        self.cat_ids = self.CLASSES
        for img_id in img_ids:
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos
    
    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('robndbox')
            bbox = [
                float(bnd_box.find('cx').text),
                float(bnd_box.find('cy').text),
                float(bnd_box.find('w').text),
                float(bnd_box.find('h').text),
                float(bnd_box.find('angle').text)+1  # important!
            ]
            
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            cx, cy, w, h, a = list(map(float, bbox))
            # set w the long side and h the short side
            new_w, new_h = max(w, h), min(w, h)
            # adjust angle
            a = a if w > h else a + np.pi / 2
            # normalize angle to [-np.pi/4, pi/4*3]
            a = norm_angle(a)
            bbox = [cx, cy, new_w, new_h, a]
            
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 5))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 5))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self, results, work_dir=None, gt_dir=None, imagesetfile=None):
        results_path = osp.join(work_dir, 'results_txt')
        mmcv.mkdir_or_exist(results_path)

        print('Saving results to {}'.format(results_path))
        self.result_to_txt(results, results_path)

        detpath = osp.join(results_path, '{:s}.txt')
        annopath = osp.join(gt_dir, '{:s}.xml')  # data/HRSC2016/Test/Annotations/{:s}.xml

        classaps = []
        map = 0
        for classname in self.CLASSES:
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,  # default: 0.5
                                     use_07_metric=True)
            map = map + ap
            print(classname, ': ', rec, prec, ap)
            classaps.append(ap)

        map = map / len(self.CLASSES)
        print('map:', map)
        classaps = 100 * np.array(classaps)
        print('classaps: ', classaps)
        # Saving results to disk
        with open(osp.join(work_dir, 'eval_results.txt'), 'w') as f:
            res_str = 'mAP:' + str(map) + '\n'
            res_str += 'classaps: ' + ' '.join([str(x) for x in classaps])
            f.write(res_str)
        return map

    def result_to_txt(self, results, results_path):
        img_names = [img_info['id'] for img_info in self.img_infos]

        assert len(results) == len(img_names), 'len(results) != len(img_names)'

        for classname in self.CLASSES:
            f_out = open(osp.join(results_path, classname + '.txt'), 'w')
            print(classname + '.txt')
            # per result represent one image
            for img_id, result in enumerate(results):
                for class_id, bboxes in enumerate(result):
                    if self.CLASSES[class_id] != classname:
                        continue
                    if bboxes.size != 0:
                        for bbox in bboxes:
                            score = bbox[5]
                            bbox = rotated_box_to_poly_single(bbox[:5])
                            temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                                osp.splitext(img_names[img_id])[0], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                                bbox[5], bbox[6], bbox[7])
                            f_out.write(temp_txt)
            f_out.close()
