from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pdb
import cv2
import torch
# import vot
import sys
import time
import os
from lib.test.evaluation import Tracker
import lib.test.vot.vot as vot
from lib.test.vot.vot_utils import *
from lib.train.dataset.depth_utils import get_rgbd_frame


class SUTrack(object):
    def __init__(self, tracker_name='sutrack', para_name='sutrack_b224'):
        # create tracker
        tracker_info = Tracker(tracker_name, para_name, "depthtrack", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def write(self, str):
        txt_path = ""
        file = open(txt_path, 'a')
        file.write(str)

    def initialize(self, img_rgb, selection):
        # init on the 1st frame
        # region = rect_from_mask(mask)
        x, y, w, h = selection
        bbox = [x,y,w,h]
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': bbox}
        _ = self.tracker.initialize(img_rgb, init_info)

    def track(self, img_rgb):
        # track
        outputs = self.tracker.track(img_rgb)
        pred_bbox = outputs['target_bbox']
        max_score = outputs['best_score'].max().cpu().numpy()
        return pred_bbox, max_score


def run_vot_exp(tracker_name, para_name, vis=False, out_conf=False, channel_type='color'):

    torch.set_num_threads(1)
    save_root = os.path.join('', para_name)
    if vis and (not os.path.exists(save_root)):
        os.mkdir(save_root)
    tracker = SUTrack(tracker_name=tracker_name, para_name=para_name)

    if channel_type=='rgb':
        channel_type=None
    handle = vot.VOT("rectangle", channels=channel_type)

    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root,seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # read rgbd data
    if isinstance(imagefile, list) and len(imagefile)==2:
        image = get_rgbd_frame(imagefile[0], imagefile[1], dtype='rgbcolormap', depth_clip=True)
    else:
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB) # Right

    tracker.initialize(image, selection)
    
    # FPS计算变量
    frame_count = 0
    total_time = 0
    fps_list = []
    
    # 创建FPS日志文件 - 保存到results目录下对应的tracker文件夹
    results_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 获取tracker标识符（从.ini文件中的label）
    tracker_id = f"{tracker_name}_{para_name}" if tracker_name != para_name else tracker_name
    tracker_result_dir = os.path.join(results_dir, tracker_id)
    if not os.path.exists(tracker_result_dir):
        os.makedirs(tracker_result_dir)
    
    fps_log_path = os.path.join(tracker_result_dir, f'fps_log.txt')
    fps_file = open(fps_log_path, 'w')
    fps_file.write(f"Tracker: {tracker_name}, Config: {para_name}\n")
    fps_file.write(f"Results Directory: {tracker_result_dir}\n")
    fps_file.write("="*60 + "\n")
    fps_file.flush()

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        # read rgbd data
        if isinstance(imagefile, list) and len(imagefile) == 2:
            image = get_rgbd_frame(imagefile[0], imagefile[1], dtype='rgbcolormap', depth_clip=True)
        else:
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right

        # 计时开始
        start_time = time.time()
        b1, max_score = tracker.track(image)
        # 计时结束
        end_time = time.time()
        
        # 计算FPS
        frame_time = end_time - start_time
        total_time += frame_time
        frame_count += 1
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_list.append(current_fps)
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        # 写入FPS信息到文件和stderr（stderr不会被VOT捕获）
        fps_info = f"Frame {frame_count}: FPS={current_fps:.2f}, Avg FPS={avg_fps:.2f}, Time={frame_time*1000:.2f}ms\n"
        fps_file.write(fps_info)
        fps_file.flush()
        sys.stderr.write(fps_info)
        sys.stderr.flush()

        if out_conf:
            handle.report(vot.Rectangle(*b1), max_score)
        else:
            handle.report(vot.Rectangle(*b1))
        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:,:,::-1].copy() # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg','_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
    
    # 写入最终统计信息
    if frame_count > 0:
        summary = "\n" + "="*60 + "\n"
        summary += f"FPS 统计信息:\n"
        summary += f"  总帧数: {frame_count}\n"
        summary += f"  总时间: {total_time:.2f}s\n"
        summary += f"  平均FPS: {avg_fps:.2f}\n"
        summary += f"  最大FPS: {max(fps_list):.2f}\n"
        summary += f"  最小FPS: {min(fps_list):.2f}\n"
        summary += "="*60 + "\n"
        
        fps_file.write(summary)
        fps_file.close()
        
        sys.stderr.write(summary)
        sys.stderr.flush()
        
        # 同时创建一个简单的FPS摘要文件，便于快速查看
        fps_summary_path = os.path.join(tracker_result_dir, 'fps_summary.txt')
        with open(fps_summary_path, 'w') as f:
            f.write(f"Average FPS: {avg_fps:.2f}\n")
            f.write(f"Max FPS: {max(fps_list):.2f}\n")
            f.write(f"Min FPS: {min(fps_list):.2f}\n")
            f.write(f"Total Frames: {frame_count}\n")
            f.write(f"Total Time: {total_time:.2f}s\n")
        
        # 打印日志文件位置
        sys.stderr.write(f"\nFPS日志已保存到: {fps_log_path}\n")
        sys.stderr.write(f"FPS摘要已保存到: {fps_summary_path}\n")
        sys.stderr.flush()

