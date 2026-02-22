from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
import torch
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap


class SUTrack_SGLA_RGBD_Actor(BaseActor):
    """Actor for training SUTrack with SGLA-RGBD support"""
    
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        
        # SGLA-RGBD损失权重
        self.sgla_rgbd_comp_weight = cfg.MODEL.ENCODER.SGLA_RGBD.get('COMPLEMENTARITY_LOSS_WEIGHT', 0.1)
        self.modal_balance_weight = cfg.MODEL.ENCODER.SGLA_RGBD.get('MODAL_BALANCE_WEIGHT', 0.05)
        
        # 损失记录
        self.sgla_rgbd_comp_loss = 0.0
        self.modal_balance_loss = 0.0

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b, dim=0)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b, dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b, dim=0)

        if self.multi_modal_language:
            text = data['nlp_ids'].permute(1, 0)
            text_src = self.net(text_data=text, mode='text')
        else:
            text_src = None

        # Task classification
        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        task_index_batch = torch.tensor(task_index_batch).cuda()

        enc_opt = self.net(
            template_list=template_list,
            search_list=search_list,
            template_anno_list=template_anno_list,
            text_src=text_src,
            task_index=task_index_batch,
            mode='encoder'
        )
        
        # 检查encoder输出是否有NaN
        if isinstance(enc_opt, (list, tuple)):
            enc_tensor = enc_opt[0] if len(enc_opt) > 0 else None
        else:
            enc_tensor = enc_opt
        
        if enc_tensor is not None and torch.is_tensor(enc_tensor) and torch.isnan(enc_tensor).any():
            print("\n" + "="*60)
            print("⚠️  Encoder输出检测到NaN!")
            print(f"NaN count: {torch.isnan(enc_tensor).sum().item()}")
            print("="*60 + "\n")
            raise ValueError("Encoder outputs is NAN! Stop Training")
        
        outputs, task_class_output = self.net(feature=enc_opt, mode="decoder")
        
        # 获取SGLA-RGBD损失
        encoder = self.net.module.encoder if hasattr(self.net, 'module') else self.net.encoder
        
        if hasattr(encoder, 'get_sgla_rgbd_loss'):
            self.sgla_rgbd_comp_loss = encoder.get_sgla_rgbd_loss()
        else:
            self.sgla_rgbd_comp_loss = 0.0
        
        # 获取模态权重并计算平衡损失
        if hasattr(encoder, 'sgla_rgbd_aux_info'):
            aux_info = encoder.sgla_rgbd_aux_info
            if 'modal_weights' in aux_info and aux_info['modal_weights'] is not None:
                modal_weights = aux_info['modal_weights']
                # 模态平衡损失:鼓励两个模态都被使用
                target_weights = torch.tensor([0.5, 0.5], device=modal_weights.device)
                self.modal_balance_loss = F.mse_loss(
                    modal_weights.mean(dim=0), 
                    target_weights
                )
            else:
                self.modal_balance_loss = 0.0
        else:
            self.modal_balance_loss = 0.0
        
        task_class_output = task_class_output.view(-1, task_class_output.size(-1))
        outputs['task_class'] = task_class_output
        outputs['task_class_label'] = task_index_batch

        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # Task classification loss
        task_cls_loss = self.objective['task_cls'](pred_dict['task_class'], pred_dict['task_class_label'])

        # GT gaussian map
        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'], 
            self.cfg.DATA.SEARCH.SIZE, 
            self.cfg.MODEL.ENCODER.STRIDE
        )
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Predicted boxes
        pred_boxes = pred_dict['pred_boxes']
        
        # NaN检测
        if torch.isnan(pred_boxes).any():
            print("\n" + "="*60)
            print("⚠️  预测框检测到NaN!")
            print(f"pred_boxes NaN count: {torch.isnan(pred_boxes).sum().item()}")
            print("="*60 + "\n")
            raise ValueError("Network outputs is NAN! Stop Training")
        
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        
        # Compute GIOU and IOU
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        
        # Compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # Weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss +
                self.loss_weight['task_cls'] * task_cls_loss)
        
        # Add SGLA-RGBD losses
        if isinstance(self.sgla_rgbd_comp_loss, torch.Tensor) and self.sgla_rgbd_comp_loss != 0.0:
            loss += self.sgla_rgbd_comp_weight * self.sgla_rgbd_comp_loss
        
        if isinstance(self.modal_balance_loss, torch.Tensor) and self.modal_balance_loss != 0.0:
            loss += self.modal_balance_weight * self.modal_balance_loss

        if return_status:
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "Loss/task_class": task_cls_loss.item(),
                "IoU": mean_iou.item()
            }
            
            # 添加SGLA-RGBD损失到状态
            if isinstance(self.sgla_rgbd_comp_loss, torch.Tensor) and self.sgla_rgbd_comp_loss != 0.0:
                status["Loss/sgla_rgbd_comp"] = self.sgla_rgbd_comp_loss.item()
            
            if isinstance(self.modal_balance_loss, torch.Tensor) and self.modal_balance_loss != 0.0:
                status["Loss/modal_balance"] = self.modal_balance_loss.item()
                
            return loss, status
        else:
            return loss
