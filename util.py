from __future__ import division
import numpy as np
import torch


def predict_transform(prediction, input_dim, anchors, num_classes, use_gpu=True):

    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attrs * num_anchors, grid_size * grid_size
    )
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size * grid_size * num_anchors, bbox_attrs
    )

    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]

    # Sigmoid the centerX,centerY and objectness score
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if use_gpu:
        prediction = prediction.cuda()
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = (
        torch.cat((x_offset, y_offset), 1)
        .repeat(1, num_anchors)
        .view(-1, 2)
        .unsqueeze(0)
    )
    prediction[:, :, :2] += x_y_offset

    # Log Space transform of height and width
    anchors = torch.FloatTensor(anchors)

    if use_gpu:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # sigmoid activation to the the class scores
    prediction[:, :, 5 : 5 + num_classes] = torch.sigmoid(
        (prediction[:, :, 5 : 5 + num_classes])
    )
    prediction[
        :, :, :4
    ] *= stride  # resize the detections map to the size of the input image

    return prediction


# def predict_transform1(prediction, inp_dim, anchors, num_classes, CUDA=True):
#     batch_size = prediction.size(0)
#     stride = inp_dim // prediction.size(2)
#     grid_size = inp_dim // stride
#     bbox_attrs = 5 + num_classes
#     num_anchors = len(anchors)
#
#     anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
#
#     prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
#     prediction = prediction.transpose(1, 2).contiguous()
#     prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
#
#     # Sigmoid the  centre_X, centre_Y. and object confidencce
#     prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
#     prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
#     prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
#
#     # Add the center offsets
#     grid_len = np.arange(grid_size)
#     a, b = np.meshgrid(grid_len, grid_len)
#
#     x_offset = torch.FloatTensor(a).view(-1, 1)
#     y_offset = torch.FloatTensor(b).view(-1, 1)
#
#     if CUDA:
#         x_offset = x_offset.cuda()
#         y_offset = y_offset.cuda()
#
#     x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
#
#     prediction[:, :, :2] += x_y_offset
#
#     # log space transform height and the width
#     anchors = torch.FloatTensor(anchors)
#
#     if CUDA:
#         anchors = anchors.cuda()
#
#     anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
#     prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
#
#     # Softmax the class scores
#     prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
#
#     prediction[:, :, :4] *= stride
#
#     return prediction