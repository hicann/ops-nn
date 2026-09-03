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
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

GRU_DEBUG_DUMP = False  # 临时调试：train=true 时打印实际输出 vs 手搓标杆 的逐张量 diff，定位完请置 False


def get_output_data_from_tensor(t):
    data = OutputData()
    data.dtype = str(t.dtype)
    data.shape = list(t.shape)
    data.stride = list(t.stride())
    return data


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


@register("gru_pyaclnn")
class GruPyaclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        """
        作用：将InputDataset处理为input_args（convert_input_data）、构造output_packages（convert_output_data）
            input_args包含除了workspace和executor以外的aclnn参数，其中对象为封装的acl结构，其中输出的部分slice即为output_packages
        时机：调用标杆，并获取参考输出后
        """
        # 信息提取（定长模式: input为3D，batchFirst时 shape 为 (B, T, input_dim)）
        batchFirst = input_data.kwargs.get("batchFirst", False)
        if batchFirst:
            self.B, self.T, self.input_dim = input_data.kwargs["input"].shape
        else:
            self.T, self.B, self.input_dim = input_data.kwargs["input"].shape
        self.H = int(input_data.kwargs["params"][0].shape[0] / 3)  # GRU gate_num = 3
        self.D = 2 if input_data.kwargs["bidirection"] else 1
        self.L = input_data.kwargs["numLayers"]
        self.dtype = input_data.kwargs["input"].dtype
        self._input_data = input_data  # 临时调试：供 after_call 复算手搓标杆
        input_args, output_packages = super().init_by_input_data(input_data)
        # 定长模式: batchSizes 传 nullptr
        input_args[3] = TensorPtr()
        # 定长模式: output为3D，batchFirst时 (B, T, D*H)，否则 (T, B, D*H)
        if batchFirst:
            output_shape = [self.B, self.T, self.D * self.H]
        else:
            output_shape = [self.T, self.B, self.D * self.H]
        # 训练模式可选输出 r/z/n/hn/h：各为 L*D 个 (T,B,H) 张量（按层、层内先正向再反向排列）
        # 推理模式 host 不写这些可选输出，保持原占位（不参与对比）
        self.is_train = bool(input_data.kwargs["train"])

        def gate_list():
            if not self.is_train:
                return [get_output_data_from_tensor(torch.empty([], dtype=self.dtype))]
            return [
                get_output_data_from_tensor(
                    torch.empty([self.T, self.B, self.H], dtype=self.dtype)
                )
                for _ in range(self.L * self.D)
            ]

        output_data = [
            get_output_data_from_tensor(torch.empty(output_shape, dtype=self.dtype)),
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
        for i, data in enumerate(output_data):
            output_packages.extend(self.backend.convert_output_data(data, i))
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
        # 训练模式返回 (output, hy, r, z, n, hn, h)，与标杆 GruApi 顺序一一对应
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
        result = (out, hy, r, z, n, hn, h)
        if GRU_DEBUG_DUMP:
            _dump_gru_diff(self, result)
        return result


def gru_reference(
    input_tensor, hx_tensor, params, hasBias, batchFirst, T, B, input_dim, H, D, L
):
    """
    手搓GRU公式：按公式逐步计算可选门输出 r/z/n/hn/h（每个为按(层,方向)排列的 (T,B,H) 张量列表）：
      r  = sigmoid(W_ir*x + b_ir + W_hr*h + b_hr)
      z  = sigmoid(W_iz*x + b_iz + W_hz*h + b_hz)
      hn = W_hn*h + b_hn                       （kernel 在乘 r 之前写出的原始 hidden 新门）
      n  = tanh(W_in*x + b_in + r*hn)
      h  = (1‑z)*n + z*h_prev
    out/hy 不再由本函数作为对比标杆（改由 torch.ops.aten.gru 原生产出），此处返回的 out/hy
    仅随递归顺带产生，供 gru_benchmark 切片丢弃；门输出是 CPU 无法提供的，只能手搓。
    """
    # fp16：matmul 用 fp32 计算（累加精度），但 h 逐时刻舍入回 fp16，
    # 模拟 kernel/torch 原生 fp16 的步间/层间 fp16 存储（与竞品同族）
    use_fp16 = input_tensor.dtype == torch.float16
    calc_dtype = torch.float32 if use_fp16 else input_tensor.dtype
    x = input_tensor.transpose(0, 1) if batchFirst else input_tensor  # (T,B,input_dim)
    x = x.to(calc_dtype)
    hx = hx_tensor.to(calc_dtype)
    p = [w.to(calc_dtype) for w in params]
    n_param = 2 + (2 if hasBias else 0)  # W_ih, W_hh, [b_ih, b_hh]
    r_list, z_list, n_list, hn_list, h_list, hy_list = [], [], [], [], [], []
    layer_input = x
    for layer_idx in range(L):
        dir_out = []  # 该层各方向的隐藏序列 (T,B,H)
        for d in range(D):
            # 反向方向处理时对时间维翻转
            seq = torch.flip(layer_input, dims=[0]) if d == 1 else layer_input
            base = (layer_idx * D + d) * n_param
            w_ih, w_hh = p[base], p[base + 1]
            b_ih = p[base + 2] if hasBias else None
            b_hh = p[base + 3] if hasBias else None
            h_prev = hx[layer_idx * D + d]  # (B,H)
            rs, zs, ns, hns, hs = [], [], [], [], []
            for t in range(T):
                i_g = torch.matmul(seq[t], w_ih.t())  # (B,3H)
                h_g = torch.matmul(h_prev, w_hh.t())  # (B,3H)
                if b_ih is not None:
                    i_g = i_g + b_ih
                if b_hh is not None:
                    h_g = h_g + b_hh
                g = i_g + h_g
                r = torch.sigmoid(g[:, 0:H])
                z = torch.sigmoid(g[:, H : 2 * H])
                hn = h_g[:, 2 * H : 3 * H]  # 原始 hidden 新门部分（未乘 r）
                n = torch.tanh(
                    i_g[:, 2 * H : 3 * H] + r * hn
                )  # 与 kernel 一致: n = tanh(i_n + r*h_n)
                h = (1 - z) * n + z * h_prev
                rs.append(r)
                zs.append(z)
                ns.append(n)
                hns.append(hn)
                if use_fp16:
                    # 逐时刻舍入回 fp16：模拟 kernel/torch 步间与层间 h 的 fp16 存储（同族），
                    # 避免理想 fp32 参考比竞品(torch 原生 fp16)本身更严格
                    h_prev = h.half().float()
                else:
                    h_prev = h
                hs.append(h_prev)
            # 反向方向输入已翻转，输出需翻回正序（与 kernel 按时间正序写 outG 一致）
            if d == 1:
                rs = torch.flip(torch.stack(rs, 0), dims=[0])
                zs = torch.flip(torch.stack(zs, 0), dims=[0])
                ns = torch.flip(torch.stack(ns, 0), dims=[0])
                hns = torch.flip(torch.stack(hns, 0), dims=[0])
                hs = torch.flip(torch.stack(hs, 0), dims=[0])
            else:
                rs = torch.stack(rs, 0)
                zs = torch.stack(zs, 0)
                ns = torch.stack(ns, 0)
                hns = torch.stack(hns, 0)
                hs = torch.stack(hs, 0)
            r_list.append(rs)  # (T,B,H)
            z_list.append(zs)
            n_list.append(ns)
            hn_list.append(hns)
            h_list.append(hs)
            hy_list.append(h_prev)  # 反向方向最终 h 即时间最早一步，无需翻转
            dir_out.append(hs)
        # 下一层输入：双向时拼接两个方向的隐藏序列
        layer_input = torch.cat(dir_out, dim=-1) if D == 2 else dir_out[0]
    output = layer_input  # (T,B,D*H)
    if batchFirst:
        output = output.transpose(0, 1)
    hy = torch.stack(hy_list, 0)  # (L*D,B,H)
    if use_fp16:
        output = output.to(torch.float16)
        hy = hy.to(torch.float16)
        r_list = [r.to(torch.float16) for r in r_list]
        z_list = [z.to(torch.float16) for z in z_list]
        n_list = [n.to(torch.float16) for n in n_list]
        hn_list = [hn.to(torch.float16) for hn in hn_list]
        h_list = [h.to(torch.float16) for h in h_list]
    return output, hy, r_list, z_list, n_list, hn_list, h_list


def gru_torch_reference(
    input_tensor, hx_tensor, params, hasBias, batchFirst, T, B, input_dim, H, D, L
):
    """
    torch.ops.aten.gru(CPU) 原生参考：直接以输入 dtype 跑，产出 out/hy 作为 out/hy 的对照标杆。
    - 原生 fp16 路径即逐时刻 fp16 存 h（与 kernel 步间/层间 fp16 存储同族），不做 fp32 升档，
      避免参考比竞品(torch 原生 fp16)本身更严格
    - 推理参考固定用 train=False, dropout=0.0（确定性前向，与 torch.nn.GRU 模块推理语义一致）
    - params 布局按(层,方向)展开 [W_ih, W_hh, (b_ih, b_hh)]，与生成器一致，aten.gru 直接消费
    """
    with torch.no_grad():
        out, hy = torch.ops.aten.gru(
            input_tensor,
            hx_tensor,
            params,
            hasBias,
            L,
            0.0,
            False,
            (D == 2),
            batchFirst,
        )
    return out, hy


def gru_benchmark(
    input_tensor,
    hx_tensor,
    params,
    hasBias,
    batchFirst,
    T,
    B,
    input_dim,
    H,
    D,
    L,
    need_gates=True,
):
    """
    合并标杆（ATK 基准与 aclnn 侧调试统一使用）：
      out / hy -> torch.ops.aten.gru(CPU) 原生实现（以输入 dtype 原生计算，与竞品 torch 数值一致）
      可选门输出 -> 手搓公式 gru_reference()（CPU 无门输出，只能手搓；fp16 时逐时刻 fp16 存 h 与 kernel 同族）
    返回 (out, hy, r, z, n, hn, h)，顺序与 aclnnGRU 训练模式输出一致。
    """
    # 输入可能来自 NPU（aclnn 侧调试/基准侧），统一搬到 CPU 计算，
    # 避免与 torch.ops.aten.gru 的 CPU 参数混用设备触发 device mismatch。
    input_tensor = input_tensor.detach().cpu()
    hx_tensor = hx_tensor.detach().cpu()
    params = [w.detach().cpu() for w in params]
    out, hy = gru_torch_reference(
        input_tensor, hx_tensor, params, hasBias, batchFirst, T, B, input_dim, H, D, L
    )
    if not need_gates:
        return out, hy
    gates = gru_reference(
        input_tensor, hx_tensor, params, hasBias, batchFirst, T, B, input_dim, H, D, L
    )
    return (out, hy) + tuple(gates[2:])


def _dump_gru_diff(api, actual):
    """临时调试：对比 aclnn 实际输出与合并标杆，打印每个输出张量的 shape 与最大绝对误差"""
    idata = api._input_data
    ref = gru_benchmark(
        idata.kwargs["input"],
        idata.kwargs["hx"],
        idata.kwargs["params"],
        idata.kwargs["hasBias"],
        idata.kwargs["batchFirst"],
        api.T,
        api.B,
        api.input_dim,
        api.H,
        api.D,
        api.L,
    )
    LD = api.L * api.D
    names = ["out", "hy"] + [
        f"{g}{i}" for g in ["r", "z", "n", "hn", "h"] for i in range(LD)
    ]
    flat_act = [actual[0], actual[1]] + [t for lst in actual[2:7] for t in lst]
    flat_ref = [ref[0], ref[1]] + [t for lst in ref[2:7] for t in lst]
    print("==== GRU DEBUG DUMP (actual vs benchmark) ====")
    for name, a, b in zip(names, flat_act, flat_ref):
        a = a.detach().cpu().float()
        b = b.detach().cpu().float()
        if a.shape != b.shape:
            print(
                f"  {name:6s}: SHAPE MISMATCH actual={tuple(a.shape)} ref={tuple(b.shape)}"
            )
            continue
        diff = (a - b).abs()
        md = diff.max().item()
        flat_idx = torch.argmax(diff).item()
        rem = flat_idx
        coord = []
        for s in reversed(diff.shape):
            coord.append(rem % s)
            rem //= s
        coord = tuple(reversed(coord))
        print(
            f"  {name:6s}: shape={tuple(a.shape)} max_abs_diff={md:.6e} "
            f"actual@max={a[coord].item():.6f} ref@max={b[coord].item():.6f} coord={coord}"
        )
        # hn 是唯一未饱和的信息张量：逐时刻最大误差，定位分歧起点
        if name.startswith("hn") and a.dim() == 3:
            t_diff = diff.max(dim=2).values.max(dim=1).values  # (T,)
            first_bad = (t_diff > 1e-3).nonzero().flatten()
            if len(first_bad) > 0:
                t0 = first_bad[0].item()
                print(
                    f"         first diverging t={t0}  t0_max_abs_diff={t_diff[t0].item():.6e}  "
                    f"actual[..,{t0},...]@max={a[t0].max().item():.6f} ref@max={b[t0].max().item():.6f}"
                )
                # t=0 单独报告：若 t=0 就错说明首个 hiddenMM(matmul W_hh·h0+b_hh) 有问题
                print(
                    f"         t=0    max_abs_diff={t_diff[0].item():.6e}  "
                    f"actual[0]@max={a[0].max().item():.6f} ref[0]@max={b[0].max().item():.6f}"
                )
                print(
                    f"         bad_t_count={len(first_bad)}  first_bad_t={first_bad[:5].tolist()}"
                )
    print("==============================================")


@register("gru")
class GruApi(BaseApi):
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
        # 信息提取（定长模式: input为3D，batchFirst时 shape 为 (B, T, input_dim)）
        batchFirst = input_data.kwargs.get("batchFirst", False)
        if batchFirst:
            self.B, self.T, self.input_dim = input_data.kwargs["input"].shape
        else:
            self.T, self.B, self.input_dim = input_data.kwargs["input"].shape
        self.H = int(input_data.kwargs["params"][0].shape[0] / 3)  # GRU gate_num = 3
        self.D = 2 if input_data.kwargs["bidirection"] else 1
        self.L = input_data.kwargs["numLayers"]
        self.dtype = input_data.kwargs["input"].dtype

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            _device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            _device = f"{self.device}:{self.device_id}"
            torch_npu.npu.set_compile_mode(jit_compile=True)  # aclop模式。
        else:
            _device = "cpu"
        # 合并标杆：out/hy 来自 torch.nn.GRU(CPU) 独立实现，可选门输出来自手搓公式
        is_train = bool(input_data.kwargs["train"])
        res = gru_benchmark(
            input_data.kwargs["input"],
            input_data.kwargs["hx"],
            input_data.kwargs["params"],
            input_data.kwargs["hasBias"],
            input_data.kwargs["batchFirst"],
            self.T,
            self.B,
            self.input_dim,
            self.H,
            self.D,
            self.L,
            need_gates=is_train,
        )
        if is_train:
            return res  # (out, hy, r, z, n, hn, h)
        return (res[0], res[1])  # 推理模式只比 out/hy
