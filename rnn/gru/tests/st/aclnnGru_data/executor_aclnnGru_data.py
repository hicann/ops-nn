#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import torch_npu
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.configs.results_config import OutputData


def get_output_data_from_tensor(t):
    data = OutputData()
    data.dtype = str(t.dtype)
    data.shape = list(t.shape)
    data.stride = list(t.stride())
    return data


def peek(tensor, n=20):
    print(tensor.cpu().ravel()[:n])


def _flatten_tensors(struct, out=None):
    """把基类 after_call 返回的（可能嵌套的）结构归一化为扁平 torch.Tensor 列表，保持顺序"""
    if out is None:
        out = []
    if isinstance(struct, (list, tuple)):
        for item in struct:
            _flatten_tensors(item, out)
    elif isinstance(struct, torch.Tensor):
        out.append(struct)
    return out


def gru_reference_data(
    input_tensor, hx_tensor, params, batchSizes, hasBias, T, H, D, L
):
    """
    手搓GRU公式（PackedSequence 数据模式）：按公式逐步计算可选门输出 r/z/n/hn/h。
    每个为按(层,方向)排列的紧凑 (totalValidSteps, H) 张量列表：
      r  = sigmoid(W_ir*x + b_ir + W_hr*h + b_hr)
      z  = sigmoid(W_iz*x + b_iz + W_hz*h + b_hz)
      hn = W_hn*h + b_hn                       （kernel 在乘 r 之前写出的原始 hidden 新门）
      n  = tanh(W_in*x + b_in + r*hn)
      h  = (1‑z)*n + z*h_prev
    紧凑布局：行索引 = 原始 packed 行位置（与输入一致），即 (time t, sample s) 位置
    = sum(batchSizes[:t]) + s；正反向均如此（kernel 按 GetInputMMRowOffset 写回，反向
    只是处理顺序倒置，写回位置不变）。
    out/hy 不作为对比标杆（由 torch.ops.aten.gru.data 原生产出），本函数只产出门输出。
    """
    # fp16：matmul 用 fp32 计算（累加精度），但 h 逐时刻舍入回 fp16，
    # 模拟 kernel/torch 原生 fp16 的步间/层间 fp16 存储（与竞品同族）
    use_fp16 = input_tensor.dtype == torch.float16
    calc_dtype = torch.float32 if use_fp16 else input_tensor.dtype
    x = input_tensor.to(calc_dtype)
    hx = hx_tensor.to(calc_dtype)
    p = [w.to(calc_dtype) for w in params]
    bs_list = [int(b) for b in batchSizes.cpu().tolist()]
    offsets = [0]
    for b in bs_list:
        offsets.append(offsets[-1] + b)
    totalValid = offsets[-1]
    n_param = 2 + (2 if hasBias else 0)
    r_list, z_list, n_list, hn_list, h_list = [], [], [], [], []
    layer_input = x
    for layer_idx in range(L):
        dir_hseq = []  # 该层各方向的紧凑隐藏序列 (totalValidSteps, H)
        for d in range(D):
            base = (layer_idx * D + d) * n_param
            w_ih, w_hh = p[base], p[base + 1]
            b_ih = p[base + 2] if hasBias else None
            b_hh = p[base + 3] if hasBias else None
            # 反向方向按时间逆序处理（与 kernel seqIdx = T‑1‑tIdx 一致），
            # 输出仍写回原始 packed 行位置（与 kernel GetInputMMRowOffset 一致）
            seq_order = range(T) if d == 0 else range(T - 1, -1, -1)
            h_prev = hx[layer_idx * D + d].clone()  # (B, H)
            r_out = torch.zeros([totalValid, H], dtype=calc_dtype)
            z_out = torch.zeros([totalValid, H], dtype=calc_dtype)
            n_out = torch.zeros([totalValid, H], dtype=calc_dtype)
            hn_out = torch.zeros([totalValid, H], dtype=calc_dtype)
            h_out = torch.zeros([totalValid, H], dtype=calc_dtype)
            for t in seq_order:
                bs = bs_list[t]
                s0 = offsets[t]
                x_t = layer_input[s0 : s0 + bs]  # (bs, input_dim)
                h_cur = h_prev[:bs]  # (bs, H) 活跃样本 0..bs‑1
                i_g = torch.matmul(x_t, w_ih.t())  # (bs, 3H)
                h_g = torch.matmul(h_cur, w_hh.t())
                if b_ih is not None:
                    i_g = i_g + b_ih
                if b_hh is not None:
                    h_g = h_g + b_hh
                g = i_g + h_g
                r = torch.sigmoid(g[:, 0:H])
                z = torch.sigmoid(g[:, H : 2 * H])
                hn = h_g[:, 2 * H : 3 * H]  # 原始 hidden 新门部分（未乘 r）
                n = torch.tanh(i_g[:, 2 * H : 3 * H] + r * hn)
                h_new = (1 - z) * n + z * h_cur
                r_out[s0 : s0 + bs] = r
                z_out[s0 : s0 + bs] = z
                n_out[s0 : s0 + bs] = n
                hn_out[s0 : s0 + bs] = hn
                h_out[s0 : s0 + bs] = h_new
                # 逐时刻舍入回 fp16：模拟 kernel/torch 步间与层间 h 的 fp16 存储（同族）
                if use_fp16:
                    h_prev[:bs] = h_new.half().float()
                else:
                    h_prev[:bs] = h_new
            r_list.append(r_out)
            z_list.append(z_out)
            n_list.append(n_out)
            hn_list.append(hn_out)
            h_list.append(h_out)
            dir_hseq.append(h_out)
        # 下一层输入：双向时拼接两个方向的紧凑隐藏序列 (totalValidSteps, D*H)
        layer_input = torch.cat(dir_hseq, dim=-1) if D == 2 else dir_hseq[0]
    if use_fp16:
        r_list = [r.to(torch.float16) for r in r_list]
        z_list = [z.to(torch.float16) for z in z_list]
        n_list = [n.to(torch.float16) for n in n_list]
        hn_list = [hn.to(torch.float16) for hn in hn_list]
        h_list = [h.to(torch.float16) for h in h_list]
    return r_list, z_list, n_list, hn_list, h_list


def gru_benchmark_data(
    input_tensor, hx_tensor, params, batchSizes, hasBias, T, H, D, L, need_gates=True
):
    """
    合并标杆（数据模式，ATK 基准与 aclnn 侧调试统一使用）：
      out / hy -> torch.ops.aten.gru.data(CPU) 原生实现（保持原 out/hy 对比口径）
      可选门输出 -> 手搓公式 gru_reference_data()（CPU 无门输出，只能手搓）
    返回 (out, hy, r, z, n, hn, h)，顺序与 aclnnGRU 训练模式输出一致。
    out/hy 参考固定用 train=False, dropout=0.0（确定性前向，与定长标杆 gru_torch_reference 一致；
    训练/推理前向在 dropout=0 时数值一致，避免 dropout>0 时非确定干扰对比）。
    """
    # 输入可能来自 NPU（aclnn 侧调试/基准侧），统一搬到 CPU 计算，
    # 避免与 torch.ops.aten.gru.data 的 CPU 参数混用设备触发 device mismatch。
    input_tensor = input_tensor.detach().cpu()
    hx_tensor = hx_tensor.detach().cpu()
    params = [w.detach().cpu() for w in params]
    batchSizes = batchSizes.detach().cpu()
    with torch.no_grad():
        out_hy = torch.ops.aten.gru.data(
            input_tensor,
            batchSizes,
            hx_tensor,
            params,
            hasBias,
            L,
            0.0,
            False,
            (D == 2),
        )
    out, hy = out_hy[0], out_hy[1]
    if not need_gates:
        return out, hy
    gates = gru_reference_data(
        input_tensor, hx_tensor, params, batchSizes, hasBias, T, H, D, L
    )
    return (out, hy) + tuple(gates)


@register("gru_data_pyaclnn")
class GruDataPyaclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        """
        作用：将InputDataset处理为input_args（convert_input_data）、构造output_packages（convert_output_data）
            input_args包含除了workspace和executor以外的aclnn参数，其中对象为封装的acl结构，其中输出的部分slice即为output_packages
        时机：调用标杆，并获取参考输出后
        """
        # 信息提取
        self.T = input_data.kwargs["batchSizes"].shape[0]
        self.H = int(input_data.kwargs["params"][0].shape[0] / 3)  # GRU gate_num = 3
        self.D = 2 if input_data.kwargs["bidirection"] else 1
        self.L = input_data.kwargs["numLayers"]
        self.dtype = input_data.kwargs["input"].dtype
        # batchSizes 合法化 (与 GruDataApi 保持一致)
        batchSizes = input_data.kwargs["batchSizes"]
        batchSizes, _ = batchSizes.sort(descending=True)
        self.B = int(batchSizes[0].item())
        batchSizes[0] = self.B
        input_data.kwargs["batchSizes"] = batchSizes
        self.batchSizes = batchSizes
        self.totalValidSteps = int(self.batchSizes.sum().item())
        # 确保 input 为紧凑 2D (sum(batchSizes), input_dim)
        # kernel 通过 GetInputMMRowOffset 按紧凑偏移读取数据，必须传入紧凑格式
        if input_data.kwargs["input"].dim() == 3:
            inp = input_data.kwargs["input"]
            input_data.kwargs["input"] = torch.cat(
                [inp[t, : batchSizes[t]] for t in range(len(batchSizes))], dim=0
            )
        input_args, output_packages = super().init_by_input_data(input_data)
        # C++ 侧 output 为紧凑 2D: (totalValidSteps, D*H)
        # 训练模式可选输出 r/z/n/hn/h：各为 L*D 个紧凑 (totalValidSteps, H) 张量（按层、层内先正向再反向排列）
        # 推理模式 host 不写这些可选输出，保持原占位（不参与对比）
        self.is_train = bool(input_data.kwargs["train"])

        def gate_list():
            if not self.is_train:
                return [get_output_data_from_tensor(torch.empty([], dtype=self.dtype))]
            return [
                get_output_data_from_tensor(
                    torch.empty([self.totalValidSteps, self.H], dtype=self.dtype)
                )
                for _ in range(self.L * self.D)
            ]

        output_data = [
            get_output_data_from_tensor(
                torch.empty([self.totalValidSteps, self.D * self.H], dtype=self.dtype)
            ),
            get_output_data_from_tensor(
                torch.empty([self.L * self.D, self.B, self.H], dtype=self.dtype)
            ),
            gate_list(),  # rOut
            gate_list(),  # zOut
            gate_list(),  # nOut
            gate_list(),  # hnOut
            gate_list(),  # hOut
        ]
        # 每个 gate 是 TensorList，convert_output_data 会递归展开成 L*D 个张量，
        # 必须全部展平进 output_packages（不能用 [0] 截断，否则 LD>1 时门输出被丢弃）
        output_packages = []
        for _, data in enumerate(output_data):
            output_packages.extend(self.backend.convert_output_data(data, _))
        input_args = input_args[:10] + output_packages
        return input_args, output_packages

    def after_call(self, *args, **kwargs):
        """
        作用：将acl类型转成torch类型
        时机：调用pyaclnn后
        兼容不同 atk 版本对 after_call 的调用约定与 TensorList 输出结构：
            双参版本: after_call(input_args, output_packages)   （本地 atk，gate 已展开为单个 AclTensorStruct）
            单参版本: after_call(output_packages)               （服务器 atk，gate 为 AclTensorlistStruct）
        统一走基类 after_call 完成 acl‑>torch（由基类处理 TensorList），
        再用 _flatten_tensors 归一化为扁平张量列表，按 LD 切片。
        """
        if len(args) >= 2:
            raw = super().after_call(args[0], args[-1])
        else:
            output_packages = args[-1] if args else kwargs.get("output_packages")
            raw = super().after_call(output_packages)
        # outputs 为扁平列表: [output, hy, r0..r_{LD‑1}, z0.., n0.., hn0.., h0..]
        outputs = _flatten_tensors(raw)
        # 推理模式只返回 output/hy（可选输出不参与对比）
        if not self.is_train:
            return (outputs[0], outputs[1])
        # 训练模式返回 (output, hy, r, z, n, hn, h)，与标杆 GruDataApi 顺序一一对应
        LD = self.L * self.D
        out, hy = outputs[0], outputs[1]
        idx = 2
        r = outputs[idx : idx + LD]
        idx += LD
        z = outputs[idx : idx + LD]
        idx += LD
        n = outputs[idx : idx + LD]
        idx += LD
        hn = outputs[idx : idx + LD]
        idx += LD
        h = outputs[idx : idx + LD]
        idx += LD
        return (out, hy, r, z, n, hn, h)


@register("gru_data")
class GruDataApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        OpsDataset.seed_everything()

    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        # 信息提取
        self.T = input_data.kwargs["input"].shape[0]
        self.B = input_data.kwargs["input"].shape[1]
        self.input_dim = input_data.kwargs["input"].shape[2]
        self.H = int(input_data.kwargs["params"][0].shape[0] / 3)  # GRU gate_num = 3
        self.D = 2 if input_data.kwargs["bidirection"] else 1
        self.L = input_data.kwargs["numLayers"]
        self.dtype = input_data.kwargs["input"].dtype
        # batchSizes元素值合法化
        batchSizes = input_data.kwargs["batchSizes"]
        batchSizes, _ = batchSizes.sort(descending=True)
        # 修复F631：移除元组，原生assert
        assert batchSizes[0] <= self.B, "generate data error"
        batchSizes[0] = self.B
        input_data.kwargs["batchSizes"] = batchSizes
        # input进行压缩 (pack) — NPU 和 CPU 路径统一使用紧凑 2D 格式
        # kernel 在不定长模式下通过 GetInputMMRowOffset 按紧凑偏移读取数据，
        # 必须传入紧凑 (sum(batchSizes), input_dim) 而非 padded (T*B, input_dim)
        input_val = input_data.kwargs["input"]
        input_val = torch.cat(
            [input_val[t, : batchSizes[t]] for t in range(len(batchSizes))], dim=0
        )
        input_data.kwargs["input"] = input_val

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            _device = f"cuda:{self.device_id}"
            input_data.kwargs["batchSizes"] = input_data.kwargs["batchSizes"].to("cpu")
        elif self.device == "npu":
            _device = f"{self.device}:{self.device_id}"
            torch_npu.npu.set_compile_mode(jit_compile=True)  # aclop模式。
        else:
            _device = "cpu"
        # 合并标杆：out/hy 来自 torch.ops.aten.gru.data(CPU) 独立实现，可选门输出来自手搓公式
        is_train = bool(input_data.kwargs["train"])
        res = gru_benchmark_data(
            input_data.kwargs["input"],
            input_data.kwargs["hx"],
            input_data.kwargs["params"],
            input_data.kwargs["batchSizes"],
            input_data.kwargs["hasBias"],
            self.T,
            self.H,
            self.D,
            self.L,
            need_gates=is_train,
        )
        if is_train:
            return res  # (out, hy, r, z, n, hn, h)
        return (res[0], res[1])  # 推理模式只比 out/hy
