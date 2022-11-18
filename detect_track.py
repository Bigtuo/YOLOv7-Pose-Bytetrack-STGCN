import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from bytetrack_utils.visualize import plot_tracking
from bytetrack_utils.byte_tracker import BYTETracker
import numpy as np


def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label, track_thresh, track_buffer, match_thresh =\
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label,opt.track_thresh,opt.track_buffer,opt.match_thresh
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    # 跟踪器
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh)  # ct+++
    frame_id = 0  # ct+++

    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        if frame_idx % 1 != 0:
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        print(pred[...,4].max())
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                Results = []  # ct+++

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    Results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)])  # ct+++

                online_tlwhs, online_ids = track_main(tracker, np.array(Results), frame_id, 1080, 1920,
                                                      (1080, 1920))  # ct+++
                online_tlwhs = np.array(online_tlwhs).reshape(-1, 4)
                online_xyxys = tlwh2xyxy(online_tlwhs)
                outputs = np.concatenate((online_xyxys, np.array(online_ids).reshape(-1, 1)), axis=1)

                # Write results

                for det_index, (output, conf) in enumerate(zip(outputs, det[:, 4])):

                    xyxy = output[0:4]
                    id = output[4]

                    if save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                    if view_img:  # Add bbox to image
                        label = f'{int(id)} {conf:.2f}'
                        kpts = det[det_index, 6:]
                        plot_one_box(xyxy, im0, label=label, color=None,
                                     line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3,
                                     orig_shape=im0.shape[:2])

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


def track_main(tracker, detection_results, frame_id, image_height, image_width, test_size):
    '''
    main function for tracking
    :param args: the input arguments, mainly about track_thresh, track_buffer, match_thresh
    :param detection_results: the detection bounds results, a list of [x1, y1, x2, y2, score]
    :param frame_id: the current frame id
    :param image_height: the height of the image
    :param image_width: the width of the image
    :param test_size: the size of the inference model
    '''
    online_targets = tracker.update(detection_results, [image_height, image_width], test_size)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    results = []
    aspect_ratio_thresh = 1.6  # +++++
    min_box_area = 10  # ++++

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area or vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(target.score)
            # save results
            results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
                    )

    return online_tlwhs, online_ids


def tlwh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='pose_test1.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs= '+', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', default=True, action='store_true', help='use keypoint labels')

    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")  # ct+++
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks, usually as same with FPS")  # ct+++
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")  # ct+++

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
