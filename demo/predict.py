import argparse
import os
import os.path as osp
import pdb
import random

import cv2
import mmcv

from mmdet.apis import init_detector, inference_detector
from mmdet.core import rotated_box_to_poly_single

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

    for j, name in enumerate(class_names):
        if colormap:
            color = colormap[j]
        else:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        try:
            dets = detections[j]
        except:
#             pdb.set_trace()
            print('no result')
            return img  # TODO don't debug
#         import ipdb;ipdb.set_trace()
        for det in dets:
            score = det[-1]
            det = rotated_box_to_poly_single(det[:-1])
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


def save_det_result(out_dir, img_dir=None, colormap=None, img_name=None):
    img_path = osp.join(img_dir, img_name)
    img_out_path = osp.join(out_dir, img_name)
    start = time.time()
    result = inference_detector(model, img_path)
    img = show_result_rbox(img_path,
                            result,
                            classnames,
                            scale=1.0,
                            threshold=0.1,
                            colormap=colormap)
    end = time.time()
    print('result:', result)
    print('time:', img_out_path, end-start)
    cv2.imwrite(img_out_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('--config_file', help='input config file', default='configs/robag/s2anet_r50_fpn_1x_robag.py')
    parser.add_argument('--model', help='pretrain model', default='work_dirs/s2anet_r50_fpn_1x_robag/latest.pth')
    parser.add_argument('--img_dir', help='img dir', default='data/ROBAG/JPEGImages/')
    parser.add_argument('--out_dir', help='output dir', default='data/ROBAG/output/')
    parser.add_argument(
        '--data',
        choices=['coco', 'dota', 'dota_large', 'dota_hbb', 'hrsc2016', 'voc', 'robag'],
        default='robag',
        type=str,
        help='dataset type')
    parser.add_argument('--img_name', help='img name', default='00001.jpg')
    args = parser.parse_args()

    robag_colormap = [(212, 188, 0)]
    
    data_name = args.data
    if data_name == 'robag':
        colormap = robag_colormap
    else:
        print('ERROR:', data_name, 'not supported')

    classnames = ['bag']

    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model = init_detector(args.config_file, args.model, device='cpu')
    
    save_det_result(args.out_dir, img_dir=args.img_dir, colormap=colormap, img_name=args.img_name)
