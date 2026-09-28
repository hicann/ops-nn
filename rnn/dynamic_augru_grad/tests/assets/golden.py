#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""DynamicAUGRUGrad TestSpec —— TTK（ops-test-kit）kernel/GEIR 模式 golden 与输入定制插件。

用法（配合 tests/st/arch35/ttk_kernel_dynamic_augru_grad_st.csv，50个核心用例）：
    source ${自定义算子包}/vendors/custom_nn/bin/set_env.bash
    python3 -m ttk geir -i rnn/dynamic_augru_grad/tests/st/arch35/ttk_kernel_dynamic_augru_grad_st.csv \
        -b release --plugin rnn/dynamic_augru_grad/tests/assets/golden.py

注：本算子为AscendC预编译二进制交付，GEIR执行须带 -b release（binary复用模式）；
JIT路径受框架tiling parse缺陷影响（ParseAutoTilingRun: compile info not contain
[_pattern]），与算子实现无关。

XPU 三方校验（GPU 机器需先部署 xpu-server 并配置 ttk.conf.yaml endpoints，见
ops-test-kit/docs/XPU_Cross_Check.md）：
    # 精度：torch 参考（DynamicAugruGradTorch）与 NPU 输出/golden 三方误差比对
    python3 -m ttk geir -i rnn/dynamic_augru_grad/tests/st/arch35/ttk_kernel_dynamic_augru_grad_st.csv \
        -b release --compare cross_check \
        --plugin rnn/dynamic_augru_grad/tests/assets/golden.py --config ttk.conf.yaml
    # 性能：仅采集 GPU 侧 device_ms（参考实现为下述 torch 移植，非原生融合算子）
    python3 -m ttk geir -i rnn/dynamic_augru_grad/tests/st/arch35/ttk_kernel_dynamic_augru_grad_st.csv \
        -b release --xpu-perf --config ttk.conf.yaml

注册名与 CSV 的 op_name 一致：dynamic_augru_grad（算子信息库 opInterface 值）。
"""

__spec__ = {"dynamic_augru_grad": "DynamicAugruGradSpec"}

from functools import wraps

import numpy


class DynamicAugruGradTorch:
    """AUGRU 反向 BPTT —— torch 参考实现（由 xpu-server 在 GPU/CPU 上执行）。

    与 FP64 CPU golden 使用相同的接口输入；GPU 使用 FP32 计算、按输入 dtype 落盘。
    经 third_party["torch"] 声明后用于 cross_check 三方比对 / --xpu-perf 性能采集。
    torch 延迟到 __call__ 内导入：spec 文件在本机（无 torch）也要可导入。
    """

    def _compute(
        self,
        x,
        weight_input,
        weight_hidden,
        weight_att,
        y,
        init_h,
        h,
        dy,
        dh,
        update,
        update_att,
        reset,
        new,
        hidden_new,
        seq_length=None,
        mask=None,
        *,
        gate_order="zrh",
        **kwargs,
    ):
        import torch

        with torch.no_grad():
            f32 = torch.float32
            dev = x.device
            out_dtype = x.dtype
            w_in = weight_input.to(f32)
            w_h = weight_hidden.to(f32)
            init_h = init_h.to(f32)
            h = h.to(f32)
            dy = dy.to(f32)
            dh = dh.to(f32)
            z = update.to(f32)
            u = update_att.to(f32)
            a = weight_att.to(f32)
            r = reset.to(f32)
            n = new.to(f32)
            hn = hidden_new.to(f32)

            T, B, inSize = x.shape
            H = int(h.shape[2])
            threeH = 3 * H
            tb = T * B
            zrh = str(gate_order) == "zrh"
            z_slot = 0 if zrh else 1
            r_slot = 1 - z_slot
            n_slot = 2

            xr = x.reshape(tb, inSize).to(f32)
            dGi = torch.zeros((tb, threeH), dtype=f32, device=dev)
            dGh = torch.zeros((tb, threeH), dtype=f32, device=dev)
            hPrev = torch.zeros((tb, H), dtype=f32, device=dev)
            dh_prev = dh.clone()
            dw_att = torch.zeros((tb,), dtype=f32, device=dev)
            seq = None if seq_length is None else seq_length.reshape(-1)

            for t in range(T - 1, -1, -1):
                base = t * B
                hp = init_h if t == 0 else h[t - 1]
                hPrev[base : base + B] = hp
                if seq is not None:
                    m = (seq > t).to(f32).unsqueeze(1)
                else:
                    m = torch.ones((B, 1), dtype=f32, device=dev)
                grad_h = m * dh_prev + dy[t]
                zt, ut, at, rt, nt, hnt = z[t], u[t], a[t], r[t], n[t], hn[t]
                ghn = grad_h * (hp - nt)
                dnt = grad_h * (1.0 - ut) * (1.0 - nt * nt)
                dr = dnt * rt * (1.0 - rt) * hnt
                dz = ghn * (1.0 - at) * zt * (1.0 - zt)
                dGi[base : base + B, n_slot * H : (n_slot + 1) * H] = dnt
                dGh[base : base + B, n_slot * H : (n_slot + 1) * H] = dnt * rt
                dGi[base : base + B, r_slot * H : (r_slot + 1) * H] = dr
                dGh[base : base + B, r_slot * H : (r_slot + 1) * H] = dr
                dGi[base : base + B, z_slot * H : (z_slot + 1) * H] = dz
                dGh[base : base + B, z_slot * H : (z_slot + 1) * H] = dz
                dw_att[base : base + B] = -(ghn * zt).sum(dim=1)
                dh_prev = grad_h * ut + dGh[base : base + B] @ w_h.T

            dw_input = xr.T @ dGi
            dw_hidden = hPrev.T @ dGh
            db_input = dGi.sum(dim=0)
            db_hidden = dGh.sum(dim=0)
            dx = (dGi @ w_in.T).reshape(T, B, inSize)

            outs = [
                dw_input,
                dw_hidden,
                db_input,
                db_hidden,
                dx,
                dh_prev,
                dw_att.reshape(T, B),
            ]
            return [o.to(out_dtype) for o in outs]

    @wraps(_compute)
    def __call__(self, *args, **kwargs):
        import torch

        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            return self._compute(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


class DynamicAugruGradSpec:
    """DynamicAUGRUGrad（AUGRU 反向 BPTT）CPU golden。

    语义对齐 canndev AUGRUHiddenGradCell 并修正其两处掩码缺陷（cell 读 mask
    行索引错、最终 cell 用 mask[-1]）；掩码语义与 forward DynamicAUGRU 一致：
        grad_h_t = mask[t,b] * dh_prev + dy[t]，mask[t,b] = (t < seq_length[b])

    内部计算使用 FP64。普通调用按接口 dtype 输出；TTK Promote 保留 FP64
    输出，避免比较前再次量化参考结果。
    """

    third_party = {"torch": DynamicAugruGradTorch}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
    }

    def golden(
        x,
        weight_input,
        weight_hidden,
        weight_att,
        y,
        init_h,
        h,
        dy,
        dh,
        update,
        update_att,
        reset,
        new,
        hidden_new,
        seq_length,
        mask,
        *,
        gate_order="zrh",
        **kwargs,
    ):
        try:
            return DynamicAugruGradSpec._golden_impl(
                x,
                weight_input,
                weight_hidden,
                weight_att,
                y,
                init_h,
                h,
                dy,
                dh,
                update,
                update_att,
                reset,
                new,
                hidden_new,
                seq_length,
                mask,
                gate_order=gate_order,
                promoted=kwargs.get("golden_mode") == "Promote",
            )
        except Exception:
            # 异常probe降级：非法shape/dtype组合下golden数学必然失败，但转测契约要求
            # golden不得先行GOLDEN_FAILURE中断流程——返回确定性的零参考输出，让用例
            # 继续进入CANN/GE路径，由算子侧校验产生拒绝证据
            return DynamicAugruGradSpec._fallback_outputs(x, h, dh)

    @staticmethod
    def _fallback_outputs(x, h, dh):
        f32 = numpy.float32
        out_dtype = x.dtype
        try:
            T, B, inSize = x.shape
            H = int(h.shape[2])
            shapes = [
                (inSize, 3 * H),
                (H, 3 * H),
                (3 * H,),
                (3 * H,),
                (T, B, inSize),
                (dh.shape[0], dh.shape[1]) if dh.ndim == 2 else (1,),
                (T, B),
            ]
        except Exception:
            shapes = [(1,)] * 7
        return [numpy.zeros(s, dtype=f32).astype(out_dtype) for s in shapes]

    @staticmethod
    def _golden_impl(
        x,
        weight_input,
        weight_hidden,
        weight_att,
        y,
        init_h,
        h,
        dy,
        dh,
        update,
        update_att,
        reset,
        new,
        hidden_new,
        seq_length,
        mask,
        *,
        gate_order="zrh",
        promoted=False,
    ):
        compute_dtype = numpy.float64
        out_dtype = compute_dtype if promoted else x.dtype
        x = x.astype(compute_dtype)
        w_in = weight_input.astype(compute_dtype)
        w_h = weight_hidden.astype(compute_dtype)
        init_h = init_h.astype(compute_dtype)
        h = h.astype(compute_dtype)
        dy = dy.astype(compute_dtype)
        dh = dh.astype(compute_dtype)
        z = update.astype(compute_dtype)
        u = update_att.astype(compute_dtype)
        a = weight_att.astype(compute_dtype)
        r = reset.astype(compute_dtype)
        n = new.astype(compute_dtype)
        hn = hidden_new.astype(compute_dtype)

        T, B, inSize = x.shape
        H = int(h.shape[2])
        threeH = 3 * H
        tb = T * B
        zrh = str(gate_order) == "zrh"
        z_slot = 0 if zrh else 1
        r_slot = 1 - z_slot
        n_slot = 2

        xr = x.reshape(tb, inSize)
        dGi = numpy.zeros((tb, threeH), compute_dtype)
        dGh = numpy.zeros((tb, threeH), compute_dtype)
        hPrev = numpy.zeros((tb, H), compute_dtype)
        dh_prev = dh.copy()
        dw_att = numpy.zeros((tb,), compute_dtype)
        seq = None if seq_length is None else numpy.asarray(seq_length).reshape(-1)

        for t in range(T - 1, -1, -1):
            base = t * B
            hp = init_h if t == 0 else h[t - 1]
            hPrev[base : base + B] = hp
            if seq is not None:
                m = (seq > t).astype(compute_dtype)[:, None]
            else:
                m = numpy.ones((B, 1), compute_dtype)
            grad_h = m * dh_prev + dy[t]
            zt, ut, at, rt, nt, hnt = z[t], u[t], a[t], r[t], n[t], hn[t]
            ghn = grad_h * (hp - nt)
            dnt = grad_h * (1.0 - ut) * (1.0 - nt * nt)
            dr = dnt * rt * (1.0 - rt) * hnt
            dz = ghn * (1.0 - at) * zt * (1.0 - zt)
            dGi[base : base + B, n_slot * H : (n_slot + 1) * H] = dnt
            dGh[base : base + B, n_slot * H : (n_slot + 1) * H] = dnt * rt
            dGi[base : base + B, r_slot * H : (r_slot + 1) * H] = dr
            dGh[base : base + B, r_slot * H : (r_slot + 1) * H] = dr
            dGi[base : base + B, z_slot * H : (z_slot + 1) * H] = dz
            dGh[base : base + B, z_slot * H : (z_slot + 1) * H] = dz
            dw_att[base : base + B] = -(ghn * zt).sum(axis=1)
            dh_prev = grad_h * ut + dGh[base : base + B] @ w_h.T

        dw_input = xr.T @ dGi
        dw_hidden = hPrev.T @ dGh
        db_input = dGi.sum(axis=0)
        db_hidden = dGh.sum(axis=0)
        dx = (dGi @ w_in.T).reshape(T, B, inSize)

        outs = [
            dw_input,
            dw_hidden,
            db_input,
            db_hidden,
            dx,
            dh_prev,
            dw_att.reshape(T, B),
        ]
        return [o.astype(out_dtype) for o in outs]

    # 输入索引：x(0)..hidden_new(13), seq_length(14), mask(15)
    _INPUT_NAMES = (
        "x",
        "weight_input",
        "weight_hidden",
        "weight_att",
        "y",
        "init_h",
        "h",
        "dy",
        "dh",
        "update",
        "update_att",
        "reset",
        "new",
        "hidden_new",
    )

    @staticmethod
    def _inject_inf_nan(inputs, testcase_name):
        """inf/nan边界用例注入（naninf_前缀）：确定性稀疏模式，与golden的IEEE传播对齐。

        命名约定：naninf_<input>_f32/f16（单输入注入）、naninf_multi_*（x/weight_att/dy多输入）。
        模式固定：+inf 每11个位置、-inf 每13个（错开2）、nan每7个（再错开），保证
        失败可复现且有限值/nan/±inf混合分布。
        """
        suffix = testcase_name.removeprefix("naninf_")
        if suffix.startswith("multi_"):
            targets = [0, 3, 7]  # x / weight_att / dy
        else:
            targets = [
                i
                for i, name in enumerate(DynamicAugruGradSpec._INPUT_NAMES)
                if suffix.startswith(name + "_")
            ]
        for i in targets:
            arr = inputs[i]
            if arr is None:
                continue
            flat = arr.reshape(-1)
            flat[0::11] = numpy.inf
            flat[2::13] = -numpy.inf
            flat[4::7] = numpy.nan

    def customize_inputs(*args, **kwargs):
        """输入定制：
        1. seq_length 覆写为边界值模式，保证每个带 seq 的用例都覆盖掩码边界语义
           {0, 1, T-1, T, 中间值}（随机生成无法保证）。
        2. naninf_前缀用例注入确定性inf/nan模式（IEEE边界语义验证）。

        输入按算子定义序位置传入（共 16 个，缺省可选输入为 None 占位）：
        x(0)..hidden_new(13), seq_length(14), mask(15)。返回结构须与原输入一致。

        seq_length覆写保持原dtype与长度：dtype/长度变化会使TTK输入bin字节数与
        CSV声明不一致（Input file size mismatch），异常probe将无法到达算子校验。
        """
        inputs = list(args)
        seq_idx = 14
        if len(inputs) > seq_idx and inputs[seq_idx] is not None:
            seq_arr = inputs[seq_idx]
            try:
                T = int(inputs[0].shape[0])
                B = int(seq_arr.shape[0])
                pattern = [0, T, 1, T - 1 if T > 1 else 0, T // 2]
                inputs[seq_idx] = numpy.array(
                    [min(pattern[b % len(pattern)], T) for b in range(B)],
                    dtype=seq_arr.dtype,
                )
            except Exception:
                pass  # 异常probe（x非法等）：保留TTK原生成值
        name = kwargs.get("testcase_name", "")
        if name.startswith("naninf_"):
            DynamicAugruGradSpec._inject_inf_nan(inputs, name)
        return tuple(inputs)
