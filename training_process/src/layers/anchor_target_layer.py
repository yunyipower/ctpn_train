#encoding: utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from config import cfg
import numpy as np
import numpy.random as npr
#from generate_anchors import generate_anchors
from anchor import AnchorText
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import ctpn_bbox_transform, ctpn_horizontal_bbox_transform


class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self.anchor_generator = AnchorText()
        #self._anchors = generate_anchors(scales=np.array(anchor_scales))
        #self._num_anchors = self._anchors.shape[0]
        self._anchors = self.anchor_generator.basic_anchors()
        self._num_anchors = self.anchor_generator.anchor_num
        self._feat_stride = layer_params['feat_stride']

        if cfg.TRAIN.DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            print "rpn_cls_score shapes:"
            print bottom[0].data.shape
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 2))
            self._squared_sums = np.zeros((1, 2))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if cfg.TRAIN.DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 2, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 2, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 2, height, width)
        # rpn_xside_targets 
        top[4].reshape(1, A, height, width)
        # rpn_xside_inside_weight 
        top[5].reshape(1, A, height, width)
        # rpn_xside_outside_weight 
        top[6].reshape(1, A, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]
        # xsides added by youlie
        xside = bottom[4].data
        if cfg.TRAIN.DEBUG:
            print 'AnchorTargetLayer.xside ctn {}'.format(xside)
            print 'AnchorTargetLayer.gt_boxes ctn {}'.format(gt_boxes)

        if cfg.TRAIN.DEBUG:
            print 'bottom 0 shape {}'.format(bottom[0].data.shape)
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'shape.xside.shape {}, gt_boxes.shape {}'.format(xside.shape, gt_boxes.shape)
            print 'value.xside.value {}, gt_boxes.value {}'.format(xside, gt_boxes)
            print 'rpn: gt_boxes.ctn', gt_boxes
            print 'xside ctn {}'.format(xside)

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0] # (height * width)
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        
        if cfg.TRAIN.DEBUG:
            print "shift shape: {}".format(shifts.shape)
            print "shift re shape: {}".format(shifts.reshape((1, K, 4)).shape)
            print "shift re transpose shape: {}".format(shifts.reshape((1, K, 4)).transpose(1, 0, 2).shape)
            print "shift re transpose shape type: {}".format(type(shifts.reshape((1, K, 4)).transpose(1, 0, 2)))
            print "anchors shape: {}".format(self._anchors.shape)
            print "anchors re shape: {}".format(self._anchors.reshape((1, A, 4)).shape)
            print "anchors re shape type: {}".format(type(self._anchors.reshape((1, A, 4))))
            print "all_anchors shape: {}".format(all_anchors.shape)

        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) & # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)   # height
        )[0]

        if cfg.TRAIN.DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if cfg.TRAIN.DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

        if cfg.TRAIN.DEBUG:
            print 'anchors.shape', anchors.shape
            print 'gt_boxes.shape', gt_boxes.shape

        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        if cfg.TRAIN.DEBUG:
            print 'overlaps shape: {}'.format(overlaps.shape) 
            print 'overlaps ctn: {}'.format(overlaps) 

        # base on anchors, which gt_boxes matches most
        argmax_overlaps = overlaps.argmax(axis=1) 
        #max_overlaps: anchor->gt_box max value
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] 

        if cfg.TRAIN.DEBUG:
            print 'argmax_overlaps shape: {}'.format(argmax_overlaps.shape) 
            print 'max_overlaps shape: {}'.format(max_overlaps.shape)

        # base on gt_boxes, which anchors matches most
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        #gt_max_overlaps: gt_box max -> anchor value
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]

        if cfg.TRAIN.DEBUG:
            print 'gt_argmax_overlaps shape: {}'.format(gt_argmax_overlaps.shape) 
            print 'gt_max_overlaps shape: {}'.format(gt_max_overlaps.shape)

        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]


        left_side_xside_idx = np.where(gt_boxes[:, 0]==xside[:, 0])[0]
        left_side_less_1_xside_idx = np.where(gt_boxes[:, 0]==xside[:, 0]+1)[0]
        right_side_xside_idx = np.where(gt_boxes[:, 2]==xside[:, 0])[0]
        if cfg.TRAIN.DEBUG:
            print "AnchorTargetLayer left_side_xside_idx:{}, left_side_less_1_xside_idx:{}, right_side_xside_idx:{}".\
                format(left_side_xside_idx, left_side_less_1_xside_idx, right_side_xside_idx)

            print "AnchorTargetLayer shape left_side_xside_idx:{}, left_side_less_1_xside_idx:{}, right_side_xside_idx:{}".\
                format(left_side_xside_idx.shape, left_side_less_1_xside_idx.shape, right_side_xside_idx.shape)

        left_side_xside_idx = np.append(left_side_xside_idx, left_side_less_1_xside_idx)
        if len(left_side_xside_idx)!=len(right_side_xside_idx):
            print "AnchorTargetLayer debug shape left_side_xside_idx:{}, left_side_less_1_xside_idx:{}, \
                right_side_xside_idx:{}".\
                format(left_side_xside_idx.shape, left_side_less_1_xside_idx.shape, right_side_xside_idx.shape)

        _side_xside_idx = np.append(left_side_xside_idx, right_side_xside_idx)
        xside_labels = np.copy(labels)
        anchor_contain_idx = np.in1d(argmax_overlaps, _side_xside_idx)
        positive_xside_idx = np.where(anchor_contain_idx!=0)[0]
        negative_xside_idx = np.where(anchor_contain_idx==0)[0]
        xside_labels[positive_xside_idx] = 1
         
        anchor_xside_targets = np.zeros((len(inds_inside), 1), dtype=np.float32)
        anchor_xside_targets = _compute_xside_targets(anchors, xside[argmax_overlaps, :])
        
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        xside_labels[max_overlaps <= cfg.TRAIN.RPN_XSIDE_NEGATIVE_OVERLAP] = -1

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
        
        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        xside_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
            print "cut positive was %s inds, disabling %s, now %s inds" % (
                len(fg_inds), len(disable_inds), np.sum(labels == 1))

        #try to subsample positive xside_labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(xside_labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            xside_labels[disable_inds] = -1
            print "xside labels cut positive was %s inds, disabling %s, now %s inds" % (
                len(fg_inds), len(disable_inds), np.sum(labels == 1))
            xside_labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            print "cut negative was %s inds, disabling %s, now %s inds" % (
                len(bg_inds), len(disable_inds), np.sum(labels == 0))
        
        xside_labels[negative_xside_idx] = -1

        _valid_xside_idx = np.where(xside_labels ==1)
        anchor_xside_targets_valid = anchor_xside_targets[_valid_xside_idx,:]
        bbox_targets = np.zeros((len(inds_inside), 2), dtype=np.float32)

        bbox_targets = _compute_v_targets(anchors, gt_boxes[argmax_overlaps, :])
        bbox_inside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels > 0)
            positive_weights = np.ones((1, 2)) * 1.0 / (num_examples+1)
            negative_weights = np.ones((1, 2)) * 1.0 / (num_examples+1)
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if cfg.TRAIN.DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means of (target_v_c, target_v_h):'
            print means
            print 'stdevs of (target_v_c, target_v_h):'
            print stds

        bbox_xside_inside_weights = np.zeros((len(inds_inside), 1), dtype=np.float32)
        bbox_xside_inside_weights[xside_labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_XSIDE_INSIDE_WEIGHTS)

        bbox_xside_outside_weights = np.zeros((len(inds_inside), 1), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(xside_labels > 0)
            num_examples_v1 = np.sum(xside_labels >= 0)
            print "bbox_xside_outside_weights num_examples_1:{}, num_examples_01:{}".format(num_examples, num_examples_v1)
            print "bbox_xside_inside_weights label eq 1: {}".format((xside_labels == 1).shape)
            positive_weights = np.ones((1, 1)) * 1.0 / (num_examples + 1)
            negative_weights = np.ones((1, 1)) * 1.0 / (num_examples + 1)
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(xside_labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(xside_labels == 0))
        bbox_xside_outside_weights[xside_labels == 1, :] = positive_weights
        bbox_xside_outside_weights[xside_labels == 0, :] = negative_weights

        ### map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
        anchor_xside_targets = _unmap(anchor_xside_targets, total_anchors, inds_inside, fill=0)
         
        if cfg.TRAIN.DEBUG:
            print "anchor_xside_targets 1 shape:", anchor_xside_targets.shape
        
        bbox_xside_inside_weights = _unmap(bbox_xside_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_xside_outside_weights = _unmap(bbox_xside_outside_weights, total_anchors, inds_inside, fill=0)

        if cfg.TRAIN.DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))

        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 2)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 2)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 2)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

        # bbox_xside_targets
        anchor_xside_targets = anchor_xside_targets \
            .reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        top[4].reshape(*anchor_xside_targets.shape)
        top[4].data[...] = anchor_xside_targets
        if cfg.TRAIN.DEBUG:
            print "anchor_xside_targets shape_2:", anchor_xside_targets.shape

        bbox_xside_inside_weights = bbox_xside_inside_weights \
            .reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        assert bbox_xside_inside_weights.shape[2] == height
        assert bbox_xside_inside_weights.shape[3] == width
        top[5].reshape(*bbox_xside_inside_weights.shape)
        top[5].data[...] = bbox_xside_inside_weights
        if cfg.TRAIN.DEBUG:
            print "bbox_xside_inside_weights shape:", bbox_xside_inside_weights.shape
        
        bbox_xside_outside_weights = bbox_xside_outside_weights \
            .reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        assert bbox_xside_outside_weights.shape[2] == height
        assert bbox_xside_outside_weights.shape[3] == width
        top[6].reshape(*bbox_xside_outside_weights.shape)
        top[6].data[...] = bbox_xside_outside_weights
        if cfg.TRAIN.DEBUG:
            print "bbox_xside_outside_weights shape:", bbox_xside_outside_weights.shape

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _compute_v_targets(anchors, gt_rois):
    """Compute vertical regression targets for each proposal in an image."""    

    if cfg.TRAIN.DEBUG:
        print "_compute_v_targets anchors shape,", anchors.shape
        print "_compute_v_targets gt_rois shape,", gt_rois.shape

    assert anchors.shape[0] == gt_rois.shape[0]
    assert anchors.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return ctpn_bbox_transform(anchors, gt_rois[:, :4]).astype(np.float32, copy=False)

def _compute_xside_targets(anchors, gt_xside):
    """Compute horizontal regression targets for each proposal in an image."""

    if cfg.TRAIN.DEBUG:
        print "_compute_xside_targets anchors shape,", anchors.shape
        print "_compute_xside_targets gt_xside shape,", gt_xside.shape
        print "_compute_xside_targets gt_xside dtype,", gt_xside.dtype

    assert anchors.shape[0] == gt_xside.shape[0]
    assert anchors.shape[1] == 4
    assert gt_xside.shape[1] == 1

    return ctpn_horizontal_bbox_transform(anchors, gt_xside).astype(np.float32, copy=False)
