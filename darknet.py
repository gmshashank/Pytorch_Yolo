from __future__ import division
from util import *

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """

    file = open(cfgfile, "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(
            prediction, inp_dim, self.anchors, num_classes, confidence, CUDA
        )
        return prediction


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # check type of block
        # create a new module for block
        # append to module_list

        if x["type"] == "convolutional":  # convolutional layer
            # get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add ing the convolutional layer
            conv = nn.Conv2d(
                in_channels=prev_filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias,
            )
            module.add_module(f"conv_{index}", conv)

            # Adding the BatchNorm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)

            # Checking the activation - Linear / Leaky ReLU
            if activation == "leaky":
                layer_activation = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{index}", layer_activation)

        elif x["type"] == "upsample":  # Bilinear2d Upsampling
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        elif x["type"] == "route":  # route layer
            x["layers"] = x["layers"].split(",")
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module(f"route_{index}", route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x["type"] == "shortcut":  # shortcut / skip connections
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.Tensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, use_gpu):
        modules = self.blocks[1:]
        outputs = {}  # cache output of route layers
        write = 0
        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(layer) for layer in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_layer = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_layer]
                # x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.net_info["height"])  # get input dimensions
                num_classes = int(module["classes"])  # get number of classes
                x = x.data
                x = predict_transform(x, input_dim, anchors, num_classes, use_gpu)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        # major version, minor version, subversion, images seen by network f=during training
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    num_bn_bias = bn.bias.numel()
                    bn_bias = torch.from_numpy(weights[ptr : ptr + num_bn_bias])
                    ptr += num_bn_bias
                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_bias])
                    ptr += num_bn_bias
                    bn_running_mean = torch.from_numpy(weights[ptr : ptr + num_bn_bias])
                    ptr += num_bn_bias
                    bn_running_var = torch.from_numpy(weights[ptr : ptr + num_bn_bias])
                    ptr += num_bn_bias

                    # reshape the loaded weights according to the dims of the model weights
                    bn_bias = bn_bias.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy data to model
                    bn.bias.data.copy_(bn_bias)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_bias = conv.bias.numel()
                    conv_bias = torch.from_numpy(weights[ptr : ptr + num_bias])
                    ptr += num_bias

                    # reshape the loaded weights according to the dims of the model weights
                    conv_bias = conv_bias.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_bias)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
                ptr += num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = (
        img_[np.newaxis, :, :, :] / 255.0
    )  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


# blocks = parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))

# model = Darknet("cfg/yolov3.cfg")
# model.load_weights("yolov3.weights")
# inp = get_test_input()
# print(f"CUDA available {torch.cuda.is_available()}")
# pred = model(inp, torch.cuda.is_available())
# print(pred)
# print(pred.shape)
