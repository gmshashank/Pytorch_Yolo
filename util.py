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


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = tensor.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    # returns IoU of two bounding boxes
    b1_x1, b1_y1, b2_x2, b2_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    if torch.cuda.is_available():
        inter_area = torch.max(
            inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()
        ) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda()
        )
    else:
        inter_area = torch.max(
            inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)
        ) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_retc_x2.shape)
        )

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y1 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # Object Confidence Thresholding
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # NMS
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 1] / 2
    box_corner[:, :, 2] = prediction[:, :, 2] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 3] + prediction[:, :, 3] / 2
    prediction[:, :, 4] = box_corner[:, :, 4]

    batch_size = prediction.size(0)
    write = false

    for idx in range(batch_size):
        # select the image from the batch
        image_pred = prediction[idx]  # Image Tensor
        max_conf, max_conf_score = torch.max(image_pred[:, 5 : 5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of the zero entries
        non_zero_ind = torch.nonzero((image_pred[:, 4]))
        image_pred_ = image_pred[non_zero_idx.squeeze(), :].view(-1, 7)
        try:
            img_classes = unique(image_pred_[:, -1])
        except:
            continue

        for cls in img_classes:
            # get detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).unsqueeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections for maximum objectness confidence
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_Class[conf_sort_index]
            idx = image_pred_class.size(0)

            if nms:
                # for each detection
                for i in range(idx):
                    try:
                        ious=bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                    except ValueError:
                        break
                    except IndexError:
                        break

                    iou_mask=(ious<nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:]*=iou_mask

                    non_zero_ind=torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class=image_pred_class[non_zero_ind].view(-1,7)

            batch_ind=image_pred_class.new(image_pred_class.size(0),1).fill_(ind)
            seq=batch_ind,image_pred_class
            if not write:
                output=torch.cat(seq,1)
                write=True
            else:
                out=torch.cat(seq,1)
                output=torch.cat((output,out))

    return output
