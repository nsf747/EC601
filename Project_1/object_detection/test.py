
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import pyrealsense2 as rs
import numpy as np
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import torch
from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,scale_coords)
from utils.plots import Annotator, colors, save_one_box, plot_one_box, get_orientation
from utils.torch_utils import select_device, smart_inference_mode, load_classifier
import torch
import torch.backends.cudnn as cudnn
import math

def find_plane(points):

    c = np.mean(points, axis=0)
    r0 = points - c
    u, s, v = np.linalg.svd(r0)
    nv = v[-1, :]
    ds = np.dot(points, nv)
    param = np.r_[nv, -np.mean(ds)]
    return param

@torch.no_grad()
def run():
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=True  # existing project/name ok, do not increment
    weights='yolov5m.pt'  # model.pt path(s)
    imgsz=640  # inference size (pixels)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    stride = 32
    device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt= False

    # Directory setups
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    #set_logging()
    device = select_device(device_num)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, device = device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location = device)['model']).to(device).eval()

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pc = rs.pointcloud()
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    while(True):
        t0 = time.time()

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not depth_frame or not color_frame:
            continue

        #image numpy array
        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # check for common shapes
        s1 = np.stack([letterbox(x, imgsz, stride=stride)[0].shape for x in img], 0)  # shapes
        rect = np.unique(s1, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        #t1 = time_synchronized()
        pred = model(img, augment=augment,
                     visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        #t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred =0 # apply_classifier(pred, modelc, img, img0)

        imc=img0.copy()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #angle=get_orientation(xyxy, imc, file=save_dir / 'crops' / names[c] / "f.jpg", BGR=True)
                    plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    offset_x = int((xyxy[2] - xyxy[0])/50)
                    offset_y = int((xyxy[3] - xyxy[1])/50)
                    interval_x = int((xyxy[2] - xyxy[0] -2 * offset_x)/2)
                    interval_y = int((xyxy[3] - xyxy[1] -2 * offset_y)/2)
                    points = np.zeros([9,3])
                    for i1 in range(3):
                        for j in range(3):
                            x = int(xyxy[0]) + offset_x + interval_x*i1
                            y = int(xyxy[1]) + offset_y + interval_y*j
                            if x>=640:
                                x=639
                            if x<0:
                                x=0
                            if y>=480:
                                y=479
                            if y<0:
                                y=0
                            dist = depth_frame.get_distance(x, y)*1000
                            Xtemp = dist*(x - intr.ppx)/intr.fx
                            Ytemp = dist*(y - intr.ppy)/intr.fy
                            Ztemp = dist
                            points[j+i1*3][0] = Xtemp
                            points[j+i1*3][1] = Ytemp
                            points[j+i1*3][2] = Ztemp

                    param = find_plane(points)
                    centre_coordinates= points[5]
                    alpha = math.atan(param[2]/param[0])*180/math.pi
                    if(alpha < 0):
                        alpha = alpha + 90
                    else:
                        alpha = alpha - 90

                    gamma = math.atan(param[2]/param[1])*180/math.pi
                    if(gamma < 0):
                        gamma = gamma + 90
                    else:
                        gamma = gamma - 90
                    
                    text1 = "x : " + str(round(centre_coordinates[0])) + "mm"
                    text2 = "y : " + str(round(centre_coordinates[1])) + "mm"
                    
                    #print(points)
                    
                    Dz = depth_frame.get_distance(int((xyxy[0] + xyxy[2])/2),int((xyxy[1] + xyxy[3])/2))*1000 # get Dz
                    
                    s+= "alpha: " + f"{alpha:.2f}" + ", gamma: " + f"{gamma:.2f}" + "x:y:z=" + f"{centre_coordinates[0]:.2f}" + ":" + f"{centre_coordinates[1]:.2f}" + ":" + f"{centre_coordinates[2]:.2f}" 
                    text3 = "z : " + str(round(Dz)) + "mm"
                    s+="\n"
                    cv2.putText(img0, text1, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2) - 40), cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(img0, text2, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2) ), cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(img0, text3, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2) + 40), cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    #vtx = np.asanyarray(points.get_vertices())
                    #tex = np.asanyarray(points.get_texture_coordinates())
                    #print(type(points), points)
                    #print(type(vtx), vtx.shape, vtx)
                    #print(type(tex), tex.shape, tex)
        
        print(s)

        cv2.imshow("IMAGE", img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

if __name__ == '__main__':
    try:
        run()

    finally:
        cv2.destroyAllWindows()