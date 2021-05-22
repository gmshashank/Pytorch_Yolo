from __future__ import division
from darknet import Darknet
from preprocess import prep_image
from torch.autograd import Variable
from util import *

import argparse
import cv2
import itertools
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import time
import torch
import torch.nn as nn


def arg_parse():
    # Parse arguments to detec module
    parser = argparse.ArgumentParser(description="Yolo v3 Detection Module")
    parser.add_argument(
        "--images",
        dest="images",
        help="Image directory conatining images to perform detection upon",
        default="imgs",
        type=str,
    )
    parser.add_argument(
        "--det",
        dest="det",
        help="Image directory to store detections",
        default="det",
        type=str,
    )
    parser.add_argument("--bs", dest="bs", help="Batch Size", default=1)
    parser.add_argument(
        "--confidence",
        dest="confidence",
        help="Object Confidence to filter predictions",
        default=0.5,
    )
    parser.add_argument(
        "--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4
    )
    parser.add_argument(
        "--cfg", dest="cfgfile", help="Config file", default="cfg/yolov3.cfg", type=str
    )
    parser.add_argument(
        "--weights",
        dest="weightsfile",
        help="Weights file",
        default="yolov3.weights",
        type=str,
    )
    parser.add_argument(
        "--reso",
        dest="reso",
        help="INput resolution of network",
        default="416",
        type=str,
    )
    parser.add_argument(
        "--scales",
        dest="scales",
        help="Scales to use for detection",
        default="1,2,3",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    scales = args.scales
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    num_classes = 80
    classes = load_classes("data/coco.names")

    # Set up the network
    print("Loading Network....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    # Set model to evaluation mode
    model.eval()

    read_dir = time.time()
    imglist = []

    # Detection Phase
    try:
        imglist = [
            os.path.join(os.path.realpath("."), images, img)
            for img in os.listdir(images)
            if os.path.splitext(img)[1] == ".png"
            or os.path.splitext(img)[1] == ".jpeg"
            or os.path.splitext(img)[1] == ".jpg"
        ]
    except NotADirectoryError:
        imglist.append(os.path.join(os.path.realpath("."), images))
    except FileNotFoundError:
        print(f"No file or directory with name {images}")
        exit()

        # file_path = os.path.join(os.path.realpath("."), images)
        # if "\\" in file_path:
        #     print(file_path.replace('\\', '/'))
        #     file_path = file_path.replace("\\", "/")
        # imglist.append(file_path)

    imglist = [x.replace("\\", "/") if "\\" in x else x for x in imglist]
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()
    batches = list(map(prep_image, imglist, [inp_dim for x in range(len(imglist))]))
    img_batches = [x[0] for x in batches]
    orig_imgs = [x[1] for x in batches]
    img_dim_list = [x[2] for x in batches]
    img_dim_list = torch.FloatTensor(img_dim_list).repeat(1, 2)

    if CUDA:
        img_dim_list = img_dim_list.cuda()

    leftover = 0
    if len(img_dim_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = len(imglist) // batch_size + leftover
        img_batches = [
            torch.cat(
                (
                    img_batches[
                        i * batch_size : min((i + 1) * batch_size, len(img_batches))
                    ]
                )
            )
            for i in range(num_batches)
        ]

    i = 0
    write = False
    model(get_test_input_cv(imglist, inp_dim, CUDA), CUDA)
    start_det_loop = time.time()
    objs = {}

    for batch in img_batches:
        # Load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        # Tranform predictions
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(
            prediction, confidence, num_classes, nms=True, nms_conf=nms_thresh
        )
        if type(prediction) == int:
            i += 1
            continue

        end = time.time()
        prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for img_num, image in enumerate(
            imglist[i * batch_size : min((i + 1) * batch_size, len(imglist))]
        ):
            img_id = i * batch_size + img_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == img_id]
            print(
                "{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start) / batch_size
                )
            )
            print("{0:20s}{1:s}".format("Objects Detected:", " ".join(objs)))
            print("-------------------------------------------------------")
        i += 1

        if CUDA:
            torch.cuda.synchronize()

        # Drawing bounding boxes on images
        try:
            output
        except NameError:
            print("No detection were made")
            exit()

        img_dim_list = torch.index_select(img_dim_list, 0, output[:, 0].long())
        scaling_factor = torch.min(inp_dim / img_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (
            inp_dim - scaling_factor * img_dim_list[:, 0].view(-1, 1)
        ) / 2
        output[:, [2, 4]] -= (
            inp_dim - scaling_factor * img_dim_list[:, 1].view(-1, 1)
        ) / 2
        output[:, 1:5] /= scaling_factor

        for i_val in range(output.shape[0]):
            output[i_val, [1, 3]] = torch.clamp(
                output[i_val, [1, 3]], 0.0, img_dim_list[i_val, 0]
            )
            output[i_val, [2, 4]] = torch.clamp(
                output[i_val, [2, 4]], 0.0, img_dim_list[i_val, 1]
            )

        output_recast = time.time()
        class_load = time.time()
        colors = pkl.load(open("pallete", "rb"))

        draw = time.time()

        def write(x, batches, results):
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            img = results[int(x[0])]
            cls = int(x[-1])
            label = f"{classes[cls]}"
            color = random.choice(colors)
            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                [225, 255, 255],
                1,
            )
            return img

        list(map(lambda x: write(x, batch, orig_imgs), output))

        det_names = pd.Series(imglist).apply(
            lambda x: "{}/det_{}".format(args.det, x.split("/")[-1])
        )
        list(map(cv2.imwrite, det_names, orig_imgs))
        end = time.time()

        print("Summary")
        print("---------------------------------------------------------------")
        print(f"Image: {imglist[img_id]}")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print("{:25s}: {}".format("Reading addresses", load_batch - read_dir))
        print("{:25s}: {}".format("Loading batch", start_det_loop - load_batch))
        print(
            "{:25s}: {:2.3f}".format(
                "Detection (" + str(len(imglist)) + " images",
                output_recast - start_det_loop,
            )
        )
        print("{:25s}: {}".format("Output processing", class_load - output_recast))
        print("{:25s}: {}".format("Drawing Boxes", end - draw))
        print(
            "{:25s}: {}".format(
                "Average time per image", (end - load_batch) / len(imglist)
            )
        )

        torch.cuda.empty_cache()
