import argparse
import os
import os.path as osp
import pdb
import random

import cv2
import mmcv
from mmcv import Config

from mmdet.core import rotated_box_to_poly_single
from mmdet.datasets import build_dataset

import time

def show_result_rbox(img,
                     detections,
                     class_names,
                     scale=1.0,
                     threshold=0.2,
                     colormap=None,
                     show_label=False):
    assert isinstance(class_names, (tuple, list))
    if colormap:
        assert len(class_names) == len(colormap)
    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for dets, name in zip(detections['bboxes'], detections['labels']):
        if colormap:
            color = colormap[name-1]
        else:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

        for det in dets.reshape(1, -1):
            cv2.circle(img, (det[0], det[1]), radius=2, color=color, thickness=2)
            cv2.putText(img, '%.3f' % (det[4]), (det[0], det[1]),
                            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            
            score = 1.0
            det = rotated_box_to_poly_single(det)
            
            bbox = det[:8] * scale
            if score < threshold:
                continue
            bbox = list(map(int, bbox))

            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=color,
                         thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
            if show_label:
                cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img


def save_det_result(config_file, out_dir, img_dir=None, colormap=None):
    cfg = Config.fromfile(config_file)
#     data_train = cfg.data.train
    data_test = cfg.data.test
#     dataset = build_dataset(data_train)
    dataset = build_dataset(data_test)
    classnames = dataset.CLASSES
#     classnames = [i for i in range(15)]  # TODO
    # use testset in cfg
    if not img_dir:
        img_dir = data_test.img_prefix

    img_list = dataset.img_ids
    for idx, img_name in enumerate(img_list):
        img_path = osp.join(img_dir, img_name+'.jpg')
        img_out_path = osp.join(out_dir, img_name+'.jpg')
        start = time.time()
        result = dataset.get_ann_info(idx)
        img = show_result_rbox(img_path,
                               result,
                               classnames,
                               scale=1.0,
                               threshold=0.5,
                               colormap=colormap)
        end = time.time()
        print(img_out_path, end-start)
        cv2.imwrite(img_out_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('config_file', help='input config file')
    parser.add_argument('img_dir', help='img dir')
    parser.add_argument('out_dir', help='output dir')
    parser.add_argument(
        'data',
        choices=['coco', 'dota', 'dota_large', 'dota_hbb', 'hrsc2016', 'voc', 'robag'],
        default='dota',
        type=str,
        help='dataset type')
    args = parser.parse_args()

    dota_colormap = [
        (54, 67, 244),
        (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121)]
    hrsc2016_colormap = [(212, 188, 0)]
    robag_colormap = [(212, 188, 0)]
    
    data_name = args.data
    if data_name == 'dota':
        colormap = dota_colormap
    elif data_name == 'hrsc2016':
        colormap = hrsc2016_colormap
    elif data_name == 'robag':
        colormap = robag_colormap
    else:
        print('ERROR:', data_name, 'not supported')
    
    save_det_result(args.config_file, args.out_dir, img_dir=args.img_dir,
                    colormap=colormap)
