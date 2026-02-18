from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.sutrack_ss import build_sutrack_ss
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
import clip
import numpy as np


class SUTRACK_SS(BaseTracker):
    """
    SUTrack-SS Tracker
    Based on SSTrack: Decoupled Spatio-Temporal Consistency Learning for Self-Supervised Tracking
    """
    def __init__(self, params, dataset_name):
        super(SUTRACK_SS, self).__init__(params)
        network = build_sutrack_ss(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        # mapping similar datasets
        if 'GOT10K' in DATASET_NAME:
            DATASET_NAME = 'GOT10K'
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        # multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT
        print("MULTI_MODAL_VISION is: ", self.multi_modal_vision)

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT
        print("MULTI_MODAL_LANGUAGE is: ", self.multi_modal_language)

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT
        print("USE_NLP is: ", self.use_nlp)

        self.task_index_batch = None


    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        template_list = []
        template_list.append(z_patch_arr)
        self.template_list = template_list

        # get the initial templates for multi-modal vision
        if self.multi_modal_vision:
            z_patch_arr_event, resize_factor_event = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
            template_list_event = []
            template_list_event.append(z_patch_arr_event)
            self.template_list_event = template_list_event

        # get the text
        if self.use_nlp:
            language = info['language']
            self.text = clip.tokenize([language]).cuda()
            self.text_features = self.network.forward(text_data=self.text, mode="text")
        else:
            self.text_features = None

        self.state = info['init_bbox']

        # get the task index
        if 'dataset_name' in info:
            self.task_index = self.get_task_index(info['dataset_name'])
        else:
            self.task_index = 0


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # get the search region
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                    output_sz=self.params.search_size)
        search_list = []
        search_list.append(x_patch_arr)

        # get the search region for multi-modal vision
        if self.multi_modal_vision:
            x_patch_arr_event, resize_factor_event = sample_target(image, self.state, self.params.search_factor,
                                                    output_sz=self.params.search_size)
            search_list_event = []
            search_list_event.append(x_patch_arr_event)

        # run the network
        with torch.no_grad():
            if self.multi_modal_vision:
                xz = self.network.forward(template_list=self.template_list, search_list=search_list,
                                          template_anno_list=None, text_src=self.text_features,
                                          task_index=self.task_index, mode="encoder")
            else:
                xz = self.network.forward(template_list=self.template_list, search_list=search_list,
                                          template_anno_list=None, text_src=self.text_features,
                                          task_index=self.task_index, mode="encoder")

            out_dict, task_decoded = self.network.forward(feature=xz, mode="decoder")

        pred_boxes = out_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            print("nan in pred_boxes")
            pred_boxes = torch.tensor([0.5, 0.5, 0.5, 0.5]).cuda()

        pred_box = (pred_boxes.view(-1, 4) * torch.tensor([self.params.search_size, self.params.search_size, self.params.search_size, self.params.search_size]).cuda()).view(-1, 4)

        # baseline: take the mean of all pred boxes as the final result
        pred_box = pred_box.mean(dim=0)

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            x_patch_arr_vis = x_patch_arr.copy()
            x_patch_arr_vis = cv2.cvtColor(x_patch_arr_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite("debug/{:04d}_search.jpg".format(self.frame_id), x_patch_arr_vis)

        # update the template
        if self.frame_id % self.update_intervals == 0:
            conf_score = out_dict['score_map'].view(-1).sigmoid().max().item()
            if conf_score > self.update_threshold:
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)
                self.template_list.append(z_patch_arr)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(0)

        return {"target_bbox": self.state, "best_score": out_dict['score_map'].view(-1).sigmoid().max().item()}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def get_task_index(self, dataset_name):
        """Get the task index for the dataset"""
        if hasattr(self.cfg.MODEL, 'TASK_INDEX'):
            if dataset_name.upper() in self.cfg.MODEL.TASK_INDEX:
                return self.cfg.MODEL.TASK_INDEX[dataset_name.upper()]
        return 0


def get_tracker_class():
    return SUTRACK_SS
