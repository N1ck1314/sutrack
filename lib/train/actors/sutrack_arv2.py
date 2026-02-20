from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
import torch.nn.functional as F
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap

class SUTrack_ARV2_Actor(BaseActor):
    """ 
    Actor for training SUTrack with ARTrackV2 integration
    Handles additional ARTrackV2 losses: IoU prediction and appearance reconstruction
    """
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        
        # ARTrackV2损失权重
        self.arv2_iou_weight = float(getattr(cfg.TRAIN.ARTRACKV2, "IOU_LOSS_WEIGHT", 0.5))
        self.arv2_recon_weight = float(getattr(cfg.TRAIN.ARTRACKV2, "APPEARANCE_RECON_LOSS_WEIGHT", 0.3))

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
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
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)

        if self.multi_modal_language:
            text = data['nlp_ids'].permute (1,0)
            text_src = self.net(text_data=text, mode='text')
        else:
            text_src = None

        # task_class
        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        task_index_batch = torch.tensor(task_index_batch).cuda() #torch.Size([bs])

        # Forward encoder
        enc_opt, aux_dict = self.net(template_list=template_list,
                                      search_list=search_list,
                                      template_anno_list=template_anno_list,
                                      text_src=text_src,
                                      task_index=task_index_batch,
                                      mode='encoder')
        
        # Forward decoder with gt_bbox for ARTrackV2 IoU loss
        gt_bbox = data['search_anno'][-1]  # (batch, 4) (x1,y1,w,h)
        outputs, task_class_output = self.net(feature=(enc_opt, aux_dict), 
                                               gt_bbox=gt_bbox, 
                                               mode="decoder")
        
        task_class_output = task_class_output.view(-1, task_class_output.size(-1))
        outputs['task_class'] = task_class_output
        outputs['task_class_label'] = task_index_batch

        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # task classification loss
        task_cls_loss = self.objective['task_cls'](pred_dict['task_class'], pred_dict['task_class_label'])

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1) # torch.Size([b, 1, H, W])

        # Get boxes
        pred_boxes = pred_dict['pred_boxes'] # torch.Size([b, 1, 4])
        if torch.isnan(pred_boxes).any():
            print("="*60)
            print("⚠️  警告：检测到NaN！")
            print("="*60)
            print(f"pred_boxes NaN检测: {torch.isnan(pred_boxes).any()}")
            print(f"pred_boxes min/max: {pred_boxes.min().item():.6f} / {pred_boxes.max().item():.6f}")
            
            # 检查ARTrackV2相关输出
            if 'arv2_aux' in pred_dict:
                arv2_aux = pred_dict['arv2_aux']
                print(f"arv2_aux keys: {arv2_aux.keys()}")
                if 'iou_loss' in arv2_aux:
                    print(f"iou_loss NaN: {torch.isnan(arv2_aux['iou_loss']).any()}")
                if 'appearance_recon_loss' in arv2_aux:
                    print(f"recon_loss NaN: {torch.isnan(arv2_aux['appearance_recon_loss']).any()}")
            
            print("="*60 + "\n")
            raise ValueError("Network outputs is NAN! Stop Training")
        
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        
        # compute location loss (ARTrackV2不使用score_map)
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # === ARTrackV2额外损失项 ===
        arv2_iou_loss = torch.tensor(0.0, device=l1_loss.device)
        arv2_recon_loss = torch.tensor(0.0, device=l1_loss.device)
        arv2_confidence = 0.0
        
        if 'arv2_aux' in pred_dict:
            arv2_aux = pred_dict['arv2_aux']
            
            # IoU预测损失
            if 'iou_loss' in arv2_aux:
                arv2_iou_loss = arv2_aux['iou_loss']
            
            # 外观重建损失
            if 'appearance_recon_loss' in arv2_aux:
                arv2_recon_loss = arv2_aux['appearance_recon_loss']
        
        # 获取置信度（用于日志）
        if 'confidence' in pred_dict:
            arv2_confidence = pred_dict['confidence'].detach().mean().item()
        # =======================================
        
        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss +
                self.loss_weight['task_cls'] * task_cls_loss +
                self.arv2_iou_weight * arv2_iou_loss +
                self.arv2_recon_weight * arv2_recon_loss)

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "Loss/task_class": task_cls_loss.item(),
                "Loss/arv2_iou": arv2_iou_loss.item(),
                "Loss/arv2_recon": arv2_recon_loss.item(),
                "ARV2_confidence": arv2_confidence,
                "IoU": mean_iou.item()
            }
            return loss, status
        else:
            return loss
