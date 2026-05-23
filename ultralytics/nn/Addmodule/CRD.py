import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import DFL, C2f
from ultralytics.nn.modules.conv import Conv, DWConv, RepConv
from ultralytics.utils.tal import dist2bbox, make_anchors


class PFE(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        self.conv1 = Conv(inc, inc, k=3)
        self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
        self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
        self.conv4 = Conv(inc, inc, 1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        conv2_out = self.conv2(conv1_out_1)
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
        conv3_out = self.conv3(conv2_out_1)
        out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
        out = self.conv4(out) + x
        return out


class MS_PFE(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(PFE(self.c) for _ in range(n))


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):
        super().__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPP_Block(nn.Module):
    def __init__(self, ch_in, ch_out, k, pool_size):
        super().__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module(f"pool{i}", pool)
            self.pool.append(pool)
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, block_fn="BasicBlock_3x3_Reverse", ch_hidden_ratio=1.0, act="silu", spp=False):
        super().__init__()
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == "BasicBlock_3x3_Reverse":
                self.convs.add_module(
                    str(i), BasicBlock_3x3_Reverse(next_ch_in, ch_hidden_ratio, ch_mid, shortcut=True)
                )
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module("spp", SPP_Block(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0
        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.0)
        self.register_buffer("init_pos", self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="xy"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords, mode="bilinear", align_corners=False, padding_mode="border"
        ).reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


# 在 Detect_DyHead 类前面加上这段代码
class DyHeadBlock(nn.Module):
    """DyHead 基本模块，包含通道注意力和空间注意力."""

    def __init__(self, ch_in, reduction=4):
        super().__init__()
        # 通道注意力 (Channel Attention)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, ch_in, 1, bias=False),
        )

        # 空间注意力 (Spatial Attention)
        self.spatial = Conv(2, 1, k=7, act=False)

    def forward(self, x):
        # 输入 x 是一个列表 [x1, x2, x3]，对应不同尺度
        # 这里做一个简化版：先融合，再过注意力
        # 注意：这里假设输入是单尺度特征，如果你是多尺度输入，需要调整
        if isinstance(x, list):
            # 如果是列表，简单相加融合 (你可以根据需要改成更复杂的融合)
            x = x[0]  # 这里先简单取第一个，你可以根据实际网络结构修改

        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca

        # 空间注意力
        sa = torch.sigmoid(
            self.spatial(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        )
        x = x * sa

        return x


class Detect_DyHead(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, block_num=2, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.dyhead = nn.Sequential(*[DyHeadBlock(hidc) for i in range(block_num)])
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(hidc, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
        x = self.dyhead(x)
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0

            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
