# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate, BatchNorm2d
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

_TOTAL_SKIPPED = 0

__all__ = [
    "ROI_KEYPOINT_HEAD_REGISTRY",
    "build_keypoint_head",
    "BaseKeypointRCNNHead",
    "KRCNNConvDeconvUpsampleHead",
]

ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss


def keypoint_rcnn_inference(pred_keypoint_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

    Returns:
        None. Each element in pred_instances will contain extra "pred_keypoints" and
            "pred_keypoint_heatmaps" fields. "pred_keypoints" is a tensor of shape
            (#instance, K, 3) where the last dimension corresponds to (x, y, score).
            The scores are larger than 0. "pred_keypoint_heatmaps" contains the raw
            keypoint logits as passed to this function.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    pred_keypoint_logits = pred_keypoint_logits.detach()
    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits, bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
    heatmap_results = pred_keypoint_logits.split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, heatmap_results_per_image, instances_per_image in zip(
        keypoint_results, heatmap_results, pred_instances
    ):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        # heatmap_results_per_image is (num instances)x(num keypoints)x(side)x(side)
        instances_per_image.pred_keypoints = keypoint_results_per_image
        instances_per_image.pred_keypoint_heatmaps = heatmap_results_per_image


class BaseKeypointRCNNHead(nn.Module):
    """
    Implement the basic Keypoint R-CNN losses and inference logic described in
    Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self, *, num_keypoints, loss_weight=1.0, loss_normalizer=1.0):
        """
        NOTE: this interface is experimental.

        Args:
            num_keypoints (int): number of keypoints to predict
            loss_weight (float): weight to multiple on the keypoint loss
            loss_normalizer (float or str):
                If float, divide the loss by `loss_normalizer * #images`.
                If 'visible', the loss is normalized by the total number of
                visible keypoints across images.
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.loss_weight = loss_weight
        assert loss_normalizer == "visible" or isinstance(loss_normalizer, float), loss_normalizer
        self.loss_normalizer = loss_normalizer

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "loss_weight": cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT,
            "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        }
        normalize_by_visible = (
            cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS
        )  # noqa
        if not normalize_by_visible:
            batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
            positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
            ret["loss_normalizer"] = (
                ret["num_keypoints"] * batch_size_per_image * positive_sample_fraction
            )
        else:
            ret["loss_normalizer"] = "visible"
        return ret

    def forward(self, x, instances: List[Instances], maxout):
        """
        Args:
            x: input 4D region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
            )
            return {
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer)
                * self.loss_weight
            }
        else:
            keypoint_rcnn_inference(x, instances)

            # because the boxes is extend location, need process keypoints
            maxOut = maxout * 32
            for instances_per_image in instances:
                keypoints = instances_per_image.pred_keypoints
                keypoints[:, :, :2] -= maxOut[:2]
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from regional input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(BaseKeypointRCNNHead, nn.Sequential):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    It is described in Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self, input_shape, *, num_keypoints, conv_dims, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
        """
        super().__init__(num_keypoints=num_keypoints, **kwargs)

        # default up_scale to 2.0 (this can be made an option)
        up_scale = 2.0
        in_channels = input_shape.channels

        for idx, layer_channels in enumerate(conv_dims, 1):
            if in_channels != layer_channels:
                self.add_module("conv_channel", Conv2d(in_channels, layer_channels, 1, stride=1))
                self.add_module("conv_relu{}".format(idx), nn.ReLU())
                in_channels = layer_channels

            # MHSA.
            if idx % 4 == 1:
                self.add_module("MHSA_{}".format(idx),
                                SABlock(in_channels, in_channels, input_shape))

            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.add_module("conv_fcn_relu{}".format(idx), nn.ReLU())
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "rel" in name:
                continue
            if "norm" in name and "bias" not in name:
                nn.init.uniform_(param, a=0, b=1)
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["conv_dims"] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x


class SABlock(nn.Module):
    def __init__(self, in_planes, planes, input_shape):
        super(SABlock, self).__init__()

        fmap_size = (input_shape.height, input_shape.width)
        self.MHSA = MHSA(in_planes, fmap_size)

        self.norm1 = BatchNorm2d(planes)
        self.norm2 = BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv_fc = Conv2d(in_planes, planes, 3, stride=1, padding=1)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.MHSA(out)
        x = out + x
        out = self.norm2(x)
        out = self.relu(out)
        out = self.conv_fc(out)
        out = out + x
        return out


class MHSA(nn.Module):
    def __init__(self, dim, fmap_size, heads=4, dim_qk=128, dim_v=128):
        super(MHSA, self).__init__()
        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = Conv2d(dim, out_channels_qk * 2, 1, bias=False)
        self.to_v = Conv2d(dim, out_channels_v, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        height, width = fmap_size
        self.pos_emb = RelPosEmb(height, width, dim_qk)

    def forward(self, featuremap):
        B, C, H, W = featuremap.shape
        if B == 0:
            return featuremap
        q, k = self.to_qk(featuremap).chunk(2, axis=1)
        v = self.to_v(featuremap)
        q, k, v = map(self.transpose_multihead, [q, k, v])
        q *= self.scale

        logits = torch.matmul(q, k.permute(0, 1, 3, 2))  # q_k_T
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = torch.matmul(weights, v)
        a_B, a_N, a_, a_D = attn_out.shape
        attn_out = attn_out.reshape([a_B, a_N, H, -1, a_D])  # "B N (H W) D -> B N H W D"
        attn_out = attn_out.permute(0, 1, 4, 2, 3)  # "B N H W D -> B N D H W"
        attn_out = attn_out.reshape([a_B, a_N * a_D, H, -1])  # "B N D H W -> B (N D) H W"
        return attn_out

    def transpose_multihead(self, x):
        B, N, H, W = x.shape
        x = x.reshape([B, self.heads, -1, H, W])  # "B (h D) H W -> B h D H W"
        x = x.permute(0, 1, 3, 4, 2)  # "B h D H W -> B h H W D"
        x = x.reshape([B, self.heads, H * W, -1])  # "B h H W D -> B h (H W) D"
        return x


def expand_dim(t, dim, k):
    t = t.unsqueeze(axis=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(expand_shape)


def rel_to_abs(x):
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    col_pad = torch.zeros([B, Nh, L, 1]).to(x.device)
    x = torch.cat([x, col_pad], axis=3)
    flat_x = x.reshape([B, Nh, L * 2 * L])
    flat_pad = torch.zeros([B, Nh, L - 1]).to(flat_x.device)
    flat_x = torch.cat([flat_x, flat_pad], axis=2)
    # Reshape and slice out the padded elements
    final_x = flat_x.reshape([B, Nh, L + 1, 2 * L - 1])
    return final_x[:, :, :L, L - 1:]


def relative_logits_1d(q, rel_k):
    B, Nh, H, W, _ = q.shape
    rel_logits = torch.matmul(q, rel_k.T)
    # Collapse height and heads
    rel_logits = rel_logits.reshape([-1, Nh * H, W, 2 * W - 1])
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = rel_logits.reshape([-1, Nh, H, W, W])
    rel_logits = expand_dim(rel_logits, dim=3, k=H)
    return rel_logits


class RelPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()

        scale = dim_head ** -0.5

        self.height = height
        self.width = width
        h_array = torch.randn(self.height * 2 - 1, dim_head) * scale
        w_array = torch.randn(self.width * 2 - 1, dim_head) * scale

        self.rel_height = nn.Parameter(h_array)
        self.rel_width = nn.Parameter(w_array)

    def forward(self, q):
        H = self.height
        W = self.width
        B, N, _, D = q.shape
        q = q.reshape([B, N, H, W, D])  # "B N (H W) D -> B N H W D"
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rel_logits_w.permute(0, 1, 2, 4, 3, 5)
        B, N, X, I, Y, J = rel_logits_w.shape
        rel_logits_w = rel_logits_w.reshape([B, N, X * Y, I * J])  # "B N X I Y J-> B N (X Y) (I J)"

        q = q.permute(0, 1, 3, 2, 4)  # "B N H W D -> B N W H D"
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rel_logits_h.permute(0, 1, 4, 2, 5, 3)
        B, N, X, I, Y, J = rel_logits_h.shape
        rel_logits_h = rel_logits_h.reshape([B, N, Y * X, J * I])  # "B N X I Y J -> B N (Y X) (J I)"

        return rel_logits_w + rel_logits_h
