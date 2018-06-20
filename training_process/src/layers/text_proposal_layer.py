#encoding: utf-8
import numpy as np
import yaml, caffe
from other import clip_boxes
from anchor import AnchorText

DEBUG = True

class ProposalLayer(caffe.Layer):
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        self.anchor_generator=AnchorText()
        self._num_anchors = self.anchor_generator.anchor_num

        top[0].reshape(1, 4)
        top[1].reshape(1, 1, 1, 1)
        top[2].reshape(1, 1)

    def forward(self, bottom, top):
        assert bottom[0].data.shape[0]==1, \
            'Only single item batches are supported'

        # the latter 10 denotes cls:1 
        scores = bottom[0].data[:, self._num_anchors:, :, :] 
        height, width = scores.shape[-2:] 

        bbox_deltas = bottom[1].data 
        print "ProposalLayer.forward: bbox_deltas shape {}".format(bbox_deltas.shape)
        im_info = bottom[2].data[0, :] 
        print "ProposalLayer.forward: im_info shape {}".format(bottom[2].data.shape)
        rpn_bbox_xside_pred = bottom[3].data[0, :].reshape((height, width, self._num_anchors, -1)) 
        print "ProposalLayer.forward: rpn_bbox_xside_pred shape {}".format(rpn_bbox_xside_pred.shape)

        anchors=self.anchor_generator.locate_anchors((height, width), self._feat_stride)
        print "ProposalLayer.forward: anchors shape {}".format(anchors.shape)

        scores=scores.transpose((0, 2, 3, 1)).reshape(-1, 1) 
        bbox_deltas=bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 2))
        print "ProposalLayer.forward: bbox_deltas shape {}".format(bbox_deltas.shape)
        rpn_bbox_xside_pred=rpn_bbox_xside_pred.transpose((0, 2, 3, 1)).reshape(-1, 1)

        proposals=self.anchor_generator.apply_deltas_to_anchors(bbox_deltas, anchors)
        print "after delta vertical proposals shape: {}".format(proposals.shape)

        # clip the proposals in excess of the boundaries of the image
        proposals=clip_boxes(proposals, im_info[:2])

        blob=proposals.astype(np.float32, copy=False)
        top[0].reshape(*(blob.shape))
        top[0].data[...]=blob

        top[1].reshape(*(scores.shape))
        top[1].data[...]=scores
   
        top[2].reshape(*(rpn_bbox_xside_pred.shape))
        top[2].data[...]=rpn_bbox_xside_pred

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
