"""
STAtten (Spiking Transformer Attention) Module
基于论文: https://arxiv.org/pdf/2409.19764
代码参考: https://github.com/Intelligent-Computing-Lab-Panda/STAtten

核心功能：融合脉冲神经网络（SNN）与时空注意力机制，实现低功耗高效跟踪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    SPIKINGJELLY_AVAILABLE = True
except ImportError:
    SPIKINGJELLY_AVAILABLE = False
    print("Warning: spikingjelly not installed. STAtten will use standard operations.")
    print("Install with: pip install spikingjelly")


class DVSPooling(nn.Module):
    """
    DVS数据专用池化模块：针对事件相机的3D脉冲序列设计
    输入：[T, B, C, H, W] (T=时间步, B=批次, C=通道, H/W=空间尺寸)
    输出：池化后序列 [T, B, C, H, W] (空间尺寸不变，时间维度无压缩)
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class STAttenAttention(nn.Module):
    """
    STAtten注意力模块：用于替代ViT的标准自注意力
    适配SUTrack的token序列格式 [B, N, C]
    
    参数：
        dim: 输入特征维度
        num_heads: 注意力头数
        qkv_bias: 是否使用QKV偏置
        attn_drop: 注意力dropout率
        proj_drop: 投影dropout率
        attention_mode: 注意力模式 ("STAtten"或"SDT")
        chunk_size: 时间分块大小
        use_snn: 是否使用脉冲神经网络
    """
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            attention_mode="STAtten",
            chunk_size=2,
            use_snn=False
    ):
        super().__init__()
        
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_mode = attention_mode
        self.chunk_size = chunk_size
        self.use_snn = use_snn and SPIKINGJELLY_AVAILABLE
        
        # Q/K/V投影层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 如果使用SNN，添加LIF神经元
        if self.use_snn:
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        """
        前向传播
        输入：x [B, N, C]
        输出：x [B, N, C]
        """
        B, N, C = x.shape
        
        # 生成Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 如果使用SNN，通过LIF神经元
        if self.use_snn:
            # 需要添加时间维度 [B, num_heads, N, head_dim] -> [T=1, B, num_heads, N, head_dim]
            q = self.q_lif(q.unsqueeze(0)).squeeze(0)
            k = self.k_lif(k.unsqueeze(0)).squeeze(0)
            v = self.v_lif(v.unsqueeze(0)).squeeze(0)
        
        if self.attention_mode == "STAtten":
            # 时空注意力模式
            # 这里简化处理，因为输入是2D token序列，我们在token维度上应用注意力
            # K^T @ V
            kv = torch.matmul(k.transpose(-2, -1), v) * self.scale  # B, num_heads, head_dim, head_dim
            # Q @ KV
            out = torch.matmul(q, kv)  # B, num_heads, N, head_dim
            
            if self.use_snn:
                # 通过注意力LIF神经元
                out = self.attn_lif(out.unsqueeze(0)).squeeze(0)
                
        elif self.attention_mode == "SDT":
            # 脉冲驱动Transformer模式
            kv = k * v  # 逐元素乘法
            kv = kv.sum(dim=-2, keepdim=True)  # 在token维度求和
            out = q * kv
            
            if self.use_snn:
                out = self.attn_lif(out.unsqueeze(0)).squeeze(0)
        else:
            # 标准注意力（默认）
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
                
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            out = attn @ v
        
        # 重组并投影
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MS_SSA_Conv(nn.Module):
    """
    多步长时空自注意力卷积模块（原始STAtten实现）
    用于处理3D序列 [T, B, C, H, W]
    """
    def __init__(
            self,
            dim,
            num_heads=8,
            mode="direct_xor",
            dvs=False,
            layer=0,
            attention_mode="T_STAtten",
            chunk_size=2,
            spike_mode="lif",
            use_snn=True
    ):
        super().__init__()
        
        assert dim % num_heads == 0, f"通道数 {dim} 需能被注意力头数 {num_heads} 整除"

        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.attention_mode = attention_mode
        self.chunk_size = chunk_size
        self.use_snn = use_snn and SPIKINGJELLY_AVAILABLE
        
        if use_snn and not SPIKINGJELLY_AVAILABLE:
            print(f"Warning: Layer {layer} - spikingjelly not available, using standard operations")
            self.use_snn = False

        if dvs:
            self.pool = DVSPooling()

        self.scale = 0.125

        # Q/K/V生成模块
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if self.use_snn:
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        else:
            self.q_lif = nn.ReLU(inplace=True)

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if self.use_snn:
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        else:
            self.k_lif = nn.ReLU(inplace=True)

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        if self.use_snn:
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        else:
            self.v_lif = nn.ReLU(inplace=True)

        if self.use_snn:
            self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")
        else:
            self.attn_lif = nn.ReLU(inplace=True)
            
        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        if self.use_snn:
            self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")
        else:
            self.talking_heads_lif = nn.ReLU(inplace=True)

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if self.use_snn:
            self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        else:
            self.shortcut_lif = nn.ReLU(inplace=True)

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        """
        前向传播
        输入：x [T, B, C, H, W] 或 [B, C, H, W]
        输出：x, v, hook
        """
        # 处理输入维度
        if x.dim() == 4:
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(0)  # [1, B, C, H, W]
        else:
            T, B, C, H, W = x.shape
            
        head_dim = C // self.num_heads
        identity = x
        N = H * W

        x = self.shortcut_lif(x)
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        if self.dvs:
            x_pool = self.pool(x)

        x_for_qkv = x.flatten(0, 1)  # [T*B, C, H, W]

        # 生成Q
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        if self.dvs:
            q_conv_out = self.pool(q_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()

        q = (q_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous())

        # 生成K
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()

        k = (k_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous())

        # 生成V
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()

        v = (v_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous())

        # 注意力计算
        if self.attention_mode == "STAtten" or self.attention_mode == "T_STAtten":
            if self.dvs:
                scaling_factor = 1 / (H * H * self.chunk_size)
            else:
                scaling_factor = 1 / H

            num_chunks = T // self.chunk_size
            q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)

            q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)

            attn = torch.matmul(k_chunks.transpose(-2, -1), v_chunks) * scaling_factor
            out = torch.matmul(q_chunks, attn)

            out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
            output = out.reshape(T, B, self.num_heads, N, head_dim)

            x = output.transpose(4, 3).reshape(T, B, C, N).contiguous()
            x = self.attn_lif(x).reshape(T, B, C, H, W)

            if self.dvs:
                x = x.mul(x_pool)
                x = x + x_pool

            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_after_qkv"] = x

            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )

        elif self.attention_mode == "SDT":
            kv = k.mul(v)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv_before"] = kv

            if self.dvs:
                kv = self.pool(kv)

            kv = kv.sum(dim=-2, keepdim=True)
            kv = self.talking_heads_lif(kv)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()

            x = q.mul(kv)
            if self.dvs:
                x = self.pool(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

            x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )

        else:
            raise ValueError(f"不支持的注意力模式：{self.attention_mode}")

        x = x + identity
        
        # 如果输入是4D，返回4D
        if T == 1:
            x = x.squeeze(0)
            
        return x, v, hook
