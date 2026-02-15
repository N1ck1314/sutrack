"""
Actor for training sutrack_activev1 with RGBD dynamic fusion
支持自适应RGBD深度融合的训练
"""
from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
import torch.nn.functional as F
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap


class SUTrack_activev1_Actor(BaseActor):
    """ Actor for training the sutrack_activev1 with RGBD dynamic fusion"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        
        # 动态激活损失
        self.activeness_loss_weight = float(getattr(cfg.TRAIN, "ACTIVENESS_LOSS_WEIGHT", 0.01))
        self.target_active_ratio = float(getattr(cfg.TRAIN, "TARGET_ACTIVE_RATIO", 0.5))
        
        # 深度效率损失
        self.depth_efficiency_weight = float(getattr(cfg.TRAIN, "DEPTH_EFFICIENCY_WEIGHT", 0.01))
        self.target_depth_ratio = float(getattr(cfg.TRAIN, "TARGET_DEPTH_RATIO", 0.5))

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 6, H, W)  # 6通道：RGB+Depth
            search_images: (N_s, batch, 6, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)

        if self.multi_modal_language:
            text = data['nlp_ids'].permute (1,0)
            text_src = self.net(text_data=text, mode='text')
        else:
            text_src = None

        # task_class
        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        task_index_batch = torch.tensor(task_index_batch).cuda()

        enc_opt = self.net(template_list=template_list,
                           search_list=search_list,
                           template_anno_list=template_anno_list,
                           text_src=text_src,
                           task_index=task_index_batch,
                           mode='encoder')
        
        # 处理返回值
        if isinstance(enc_opt, tuple):
            enc_opt, probs_active = enc_opt
        else:
            probs_active = None
        
        outputs, task_class_output = self.net(feature=enc_opt, mode="decoder")
        task_class_output = task_class_output.view(-1, task_class_output.size(-1))
        outputs['task_class'] = task_class_output
        outputs['task_class_label'] = task_index_batch
        
        # 将激活概率传递给损失函数
        outputs['probs_active'] = probs_active

        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # task classification loss
        task_cls_loss = self.objective['task_cls'](pred_dict['task_class'], pred_dict['task_class_label'])

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # === 动态激活效率损失 ===
        activeness_loss = torch.zeros((), device=l1_loss.device)
        actual_activation_rate = 0.0
        probs_active = pred_dict.get('probs_active')
        if probs_active is not None and len(probs_active) > 0:
            prob_active_m = torch.cat(probs_active, dim=1).mean(dim=1)
            expected_active_ratio = torch.full_like(prob_active_m, self.target_active_ratio)
            activeness_loss = F.l1_loss(prob_active_m, expected_active_ratio)
            actual_activation_rate = prob_active_m.detach().mean().item()
        
        # === 新增：深度效率损失 ===
        depth_efficiency_loss = torch.zeros((), device=l1_loss.device)
        actual_depth_usage = 0.0
        
        # 从 probs_active 中提取深度使用概率（如果存在）
        if probs_active is not None and len(probs_active) > 0:
            # 假设最后一个维度是深度使用概率
            depth_probs = []
            for prob in probs_active:
                if prob.size(-1) > 1:  # 如果包含深度概率
                    depth_prob = prob[:, -1]  # 最后一个维度是深度使用概率
                    depth_probs.append(depth_prob)
            
            if len(depth_probs) > 0:
                depth_usage = torch.stack(depth_probs).mean()
                target_depth = torch.tensor(self.target_depth_ratio, device=depth_usage.device)
                depth_efficiency_loss = F.mse_loss(depth_usage, target_depth)
                actual_depth_usage = depth_usage.detach().item()
        
        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss +
                self.loss_weight['task_cls'] * task_cls_loss +
                self.activeness_loss_weight * activeness_loss +
                self.depth_efficiency_weight * depth_efficiency_loss)

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/task_class": task_cls_loss.item(),
                      "Loss/activeness": activeness_loss.item(),
                      "Loss/depth_efficiency": depth_efficiency_loss.item(),
                      "ActRate": actual_activation_rate,
                      "DepthUsage": actual_depth_usage,
                      "TargetDepthRate": self.target_depth_ratio,
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
