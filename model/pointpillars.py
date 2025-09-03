import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import config
from package import Voxelization, nms_cuda
from utils import project_point_to_camera, Anchors, anchor_target, anchors2bboxes, limit_period


class ImageFeature(nn.Module):
    def __init__(self, new_shape, mean, std, out_channels):
        super(ImageFeature, self).__init__()
        self.new_shape = new_shape
        self.mean = torch.as_tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.as_tensor(std, dtype=torch.float32)[:, None, None]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Stage1
        self.stage1_down = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage1_bn = nn.BatchNorm2d(32)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_dw = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32)
        )

        # Stage2
        self.stage2_down = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2_bn = nn.BatchNorm2d(64)
        self.stage2_relu = nn.ReLU(inplace=True)
        self.stage2_dw = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64)
        )

        # Stage3
        self.stage3_down = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2, dilation=2, bias=False)
        self.stage3_bn = nn.BatchNorm2d(128)
        self.stage3_relu = nn.ReLU(inplace=True)
        self.stage3_dw = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128)
        )

        # 1x1 conv (pointwise) cho multi-scale FPN
        self.mid1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.mid2 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.mid3 = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.last = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
    def transform_tensor(self, batch_image_paths):
        batch_tensor = []
        for image_path in batch_image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"{image_path} not found!")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image_tensor = torch.from_numpy(image).permute(2,0,1).to(dtype=torch.float32).div_(255.0)
            image_tensor = image_tensor.sub_(self.mean).div_(self.std).unsqueeze(0) 
            image_tensor = F.interpolate(image_tensor, size=self.new_shape, mode='bilinear', align_corners=False)
            
            batch_tensor.append(image_tensor.squeeze(0))
        
        batch_tensor = torch.stack(batch_tensor, dim=0) # (B, C, H, W)
        return batch_tensor
        
    def forward(self, batch_image_paths, device):
        batch_tensor = self.transform_tensor(batch_image_paths).to(device)
        
        # Stem, stride = 1 (B, 16, H, W)
        c0 = self.stem(batch_tensor)

        # Stage1, stride = 2 (B, 32, H/2, W/2)
        c1 = self.stage1_down(c0)
        c1 = self.stage1_bn(c1)
        c1 = self.stage1_relu(c1)
        c1 = self.stage1_dw(c1) + c1

        # Stage2, stride = 2 (B, 64, H/4, W/4)
        c2 = self.stage2_down(c1)
        c2 = self.stage2_bn(c2)
        c2 = self.stage2_relu(c2)
        c2 = self.stage2_dw(c2) + c2

        # Stage3, stride = 2 (B, 128, H/8, W/8)
        c3 = self.stage3_down(c2)
        c3 = self.stage3_bn(c3)
        c3 = self.stage3_relu(c3)
        c3 = self.stage3_dw(c3) + c3

        # Multi-scale FPN
        m1 = self.mid1(c1) # (B, 64, H/2, W/2)
        m2 = self.mid2(c2) # (B, 64, H/4, W/4)
        m3 = self.mid3(c3) # (B, 64, H/8, W/8)

        up1 = F.interpolate(m2, size=m1.shape[-2:], mode='bilinear', align_corners=False) # (B, 64, H/2, W/2)
        up2 = F.interpolate(m3, size=m1.shape[-2:], mode='bilinear', align_corners=False) # (B, 64, H/2, W/2)

        concat = torch.cat([m1, up1, up2], dim=1) # (B, 192, H/2, W/2)
        out = self.last(concat) # (B, out_channels, H/2, W/2)
        
        return out
        
        
class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super(PillarLayer, self).__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        """ generate pillar from points
        Args:
            batched_pts [list torch.tensor float32, (N, 4)]: list of batch points, each batch have shape (N, 4)
            
        Returns:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar
    
class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channels, out_channels):
        super(PillarEncoder, self).__init__()
        self.out_channel = out_channels
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.gate_fusion = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.LayerNorm(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()                
        )
        
    def forward(self, pillars, coors_batch, npoints_per_pillar, batch_image_map, batched_calibs, batch_size):
    
        device = pillars.device
        
        mean_center = torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, 1, 3)
        
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - mean_center # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.conv(features))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)
        
        image_features = []        
        for i in range(batch_size):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_mean_center = mean_center[cur_coors_idx] # (pi, 1, 3)
            cur_mean_center = cur_mean_center.squeeze(1) # (pi, 3) x, y, z of the mean_center in current batch
            
            calib = batched_calibs[i]
            image_map = batch_image_map[i] # (out_channels, H/2, W/2)
            h, w = image_map.shape[1:]
            
            u, v = project_point_to_camera(point=cur_mean_center, calib=calib)
            u = torch.clamp(u/2, 0, w - 1).long()
            v = torch.clamp(v/2, 0, h - 1).long()    
            
            img_feat = image_map.permute(1, 2, 0)[v, u] # (pi, out_channels)
            
            image_features.append(img_feat)
            
        image_features = torch.cat(image_features, dim=0) # (p1 + p2 + ... + pb, out_channels)
        
        concat_features = torch.cat([pooling_features, image_features], dim=1) # (p1 + p2 + ... + pb, out_channels * 2)
        
        gate_weight = self.gate_fusion(concat_features) # (p1 + p2 + ... + pb, out_channels)
        
        gate_features = pooling_features * gate_weight + image_features * (1 - gate_weight) # (p1 + p2 + ... + pb, out_channels)
        
        # 6. pillar scatter
        batched_canvas = []
        for i in range(batch_size):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = gate_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, out_channels, self.y_l, self.x_l)
        return batched_canvas  
    
class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super(Backbone, self).__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs
    
class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super(Neck, self).__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out
    
class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super(Head, self).__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
    
class Pointpillars(nn.Module):
    def __init__(
        self,
        nclasses=config['num_classes'], 
        voxel_size=config['voxel_size'],
        point_cloud_range=config['pc_range'],
        max_num_points=config['max_points'],
        max_voxels=config['max_voxels'],
        new_shape = config['new_shape'], 
        mean = config['mean'],
        std = config['std'], 
        device = None
    ):
        super(Pointpillars, self).__init__()
        
        self.device = device
        
        self.nclasses = nclasses
        
        self.image_backbone = ImageFeature(
            new_shape=new_shape,
            mean=mean,
            std=std,
            out_channels=64
        )
        
        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size, 
            point_cloud_range=point_cloud_range, 
            max_num_points=max_num_points, 
            max_voxels=max_voxels
        )
        
        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size, 
            point_cloud_range=point_cloud_range, 
            in_channels=9, 
            out_channels=64
        )
        
        self.backbone = Backbone(
            in_channel=64, 
            out_channels=[64, 128, 256], 
            layer_nums=[3, 5, 5]
        )
        
        self.neck = Neck(
            in_channels=[64, 128, 256], 
            upsample_strides=[1, 2, 4], 
            out_channels=[128, 128, 128]
        )
        
        self.head = Head(
            in_channel=384, 
            n_anchors=2 * nclasses, 
            n_classes=nclasses
        )
        
        self.anchors_generator = Anchors(
            ranges=config['ANCHOR']['ranges'], 
            sizes=config['ANCHOR']['sizes'], 
            rotations=config['ANCHOR']['rotations']
        )
        
        # train
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 300
        self.nms_thr = 0.1
        self.score_thr = 0.1
        self.max_num = 100
        
    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results
    
    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None, batched_image_paths = None, batched_calibs = None):
        batch_size = len(batched_pts)
        
        image_map = self.image_backbone(batched_image_paths, self.device)

        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar, image_map, batched_calibs, batch_size)

        xs = self.backbone(pillar_features)

        x = self.neck(xs)

        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors, 
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_gt_labels, 
                assigners=self.assigners,
                nclasses=self.nclasses
            )
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        
        elif mode == 'val':
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred, 
                bbox_pred=bbox_pred, 
                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                batched_anchors=batched_anchors
            )
            
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred, 
                bbox_pred=bbox_pred, 
                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                batched_anchors=batched_anchors
            )
            
            return results
        
        else:
            raise ValueError