import os
import torch
import torch.nn as nn

# --- 关键配置与构建函数 ---
from lib.test.parameter.sutrack import parameters as sutrack_params
from lib.models.sutrack import build_sutrack


class SUTrackTinyRGBDWrapper(nn.Module):
    """
    封装 Tiny@224 的 SUTrack 模型：
      输入:
        - template:      (B, 6, Ht, Wt)  RGBD 模板 patch（已是 float32，是否归一化由你决定，只要和训练一致即可）
        - search:        (B, 6, Hs, Ws)  RGBD 搜索 patch
        - template_anno: (B, 4)          模板 bbox，格式 (cx, cy, w, h)，均为相对坐标 (0~1)
      输出:
        - pred_boxes: (B, 1, 4)  (cx, cy, w, h)，相对坐标
        - score_map:  (B, 1, fx, fx)  中心响应图
    """
    def __init__(self, sutrack_model):
        super().__init__()
        self.model = sutrack_model

    def forward(self, template, search, template_anno):
        # 封装成列表以符合原始接口
        template_list = [template]      # len = 1
        search_list = [search]          # len = 1
        template_anno_list = [template_anno]  # (B,4)

        # 不用文本模态，所以 text_src=None；task_index 也不用，给 None 即可
        enc_opt = self.model.forward_encoder(
            template_list=template_list,
            search_list=search_list,
            template_anno_list=template_anno_list,
            text_src=None,
            task_index=None
        )
        out = self.model.forward_decoder(feature=enc_opt)

        # 返回关心的两个输出：bbox 和 score_map
        return out["pred_boxes"], out["score_map"]


def build_tiny_model(yaml_name: str = "sutrack_t224"):
    """
    使用测试参数逻辑构建 Tiny@224 模型并加载权重。
    yaml_name 一般设为 'sutrack_t224'，如果你训练用的是别的 YAML 名，可以改掉这个。
    """
    # 复用测试时的参数加载逻辑（自动加载 experiments/sutrack/{yaml_name}.yaml 和 checkpoint）
    params = sutrack_params(yaml_name)
    cfg = params.cfg

    print("Using yaml:", yaml_name)
    print("Search size:", cfg.TEST.SEARCH_SIZE, "Template size:", cfg.TEST.TEMPLATE_SIZE)
    print("Encoder type:", cfg.MODEL.ENCODER.TYPE)

    # 构建 SUTrack 模型
    model = build_sutrack(cfg)

    # 加载权重
    ckpt_path = params.checkpoint
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["net"], strict=False)

    model.eval()
    model.cpu()   # 导出 ONNX 建议先放到 CPU，ONNX 与设备无关
    return model, cfg


def main():
    # 1) 构建 Tiny@224 模型并加载权重
    base_model, cfg = build_tiny_model("sutrack_t224")

    # 2) 封装成 RGBD Tiny Wrapper（只做一次编码+解码）
    wrapper = SUTrackTinyRGBDWrapper(base_model)

    # 3) 准备导出用的 dummy 输入
    B = 1  # 可以动态 batch，导出时先用 B=1 即可

    template_size = cfg.TEST.TEMPLATE_SIZE
    search_size = cfg.TEST.SEARCH_SIZE

    # ONNX 输入为 float32，形状 (B, 6, H, W)，你实际推理时喂 RGBD patch 即可
    dummy_template = torch.randn(B, 6, template_size, template_size, dtype=torch.float32)
    dummy_search = torch.randn(B, 6, search_size, search_size, dtype=torch.float32)

    # 模板 bbox：这里随便给一个中间位置的框作 dummy
    # (cx, cy, w, h) 全是相对坐标 [0,1]，和 debug_eucb_encoder.py 用法一致
    dummy_template_anno = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32)  # shape (1,4)

    onnx_path = "sutrack_tiny_rgbd.onnx"

    # 4) 导出 ONNX
    print("Exporting ONNX to:", onnx_path)
    torch.onnx.export(
        wrapper,
        (dummy_template, dummy_search, dummy_template_anno),
        onnx_path,
        input_names=["template", "search", "template_anno"],
        output_names=["pred_boxes", "score_map"],
        opset_version=13,
        dynamic_axes={
            "template": {0: "batch"},
            "search": {0: "batch"},
            "template_anno": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "score_map": {0: "batch"},
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()