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

import sys
import os
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return tuple(shape_list)


def apply_adam_w_golden(
    var,
    m,
    v,
    beta1_power,
    beta2_power,
    lr,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    grad,
    max_grad_norm=None,
    amsgrad: bool = False,
    maximize: bool = False,
):
    """
    Golden function for apply_adam_w.
    """
    input_dtype = var.dtype
    if input_dtype.name in ("bfloat16", "float16"):
        var = var.astype("float32")
        m = m.astype("float32")
        v = v.astype("float32")
        grad = grad.astype("float32")
        if max_grad_norm is not None:
            max_grad_norm = max_grad_norm.astype("float32")
        beta1_power = beta1_power.astype("float32")
        beta2_power = beta2_power.astype("float32")
        lr = lr.astype("float32")
        weight_decay = weight_decay.astype("float32")
        beta1 = beta1.astype("float32")
        beta2 = beta2.astype("float32")
        epsilon = epsilon.astype("float32")

    gt = -grad if maximize else grad
    m_out = m * beta1 - (beta1 + (-1)) * gt
    v_out = v * beta2 - (beta2 + (-1)) * gt * gt

    var_t = var * (1 + (-lr * weight_decay))

    beta1_power_out = beta1_power * beta1
    beta2_power_out = beta2_power * beta2

    if amsgrad and max_grad_norm is not None:
        max_grad_norm_out = np.maximum(max_grad_norm, v_out)
        sqrt_v_t = np.sqrt(max_grad_norm_out / (1 - beta2_power_out))
        denom = sqrt_v_t + epsilon
    else:
        v_t = v_out / (1 - beta2_power_out)
        sqrt_v_t = np.sqrt(v_t)
        denom = sqrt_v_t + epsilon

    m_t = m_out / (beta1_power_out - 1)
    m_t_mul_lr = lr * m_t

    m_t_mul_lr_div_denom = m_t_mul_lr / denom
    var_out = var_t + m_t_mul_lr_div_denom

    var_out = var_out.astype(input_dtype, copy=False)
    m_out = m_out.astype(input_dtype, copy=False)
    v_out = v_out.astype(input_dtype, copy=False)

    # 按照amsgrad状态返回max_grad_norm_out
    if amsgrad and max_grad_norm is not None:
        return var_out, m_out, v_out, max_grad_norm_out.astype(input_dtype, copy=False)
    return var_out, m_out, v_out, None


def gen_data_and_golden(shape_str, d_type="float32", amsgrad=True, maximize=True):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_type = d_type_dict.get(d_type, np.float32)
    shape = parse_str_to_shape_list(shape_str)

    # 1. 主干 Tensor: 随机生成，注意 v 和 max_grad_norm 需要是正数
    var = np.random.uniform(-1.0, 1.0, size=shape).astype(np_type)
    m = np.random.uniform(-1.0, 1.0, size=shape).astype(np_type)
    v = np.random.uniform(0.01, 1.0, size=shape).astype(np_type)
    grad = np.random.uniform(-1.0, 1.0, size=shape).astype(np_type)
    max_grad_norm = np.random.uniform(0.01, 1.0, size=shape).astype(np_type)

    # 2. 超参数 Tensor: shape 固定为 [1]，使用 AdamW 的常规建议值
    beta1_power = np.array([0.431]).astype(np_type)
    beta2_power = np.array([0.992]).astype(np_type)
    lr = np.array([0.001]).astype(np_type)
    weight_decay = np.array([0.01]).astype(np_type)
    beta1 = np.array([0.9]).astype(np_type)
    beta2 = np.array([0.999]).astype(np_type)
    epsilon = np.array([1e-8]).astype(np_type)

    # 3. 运行 Golden 逻辑
    var_out, m_out, v_out, max_grad_norm_out = apply_adam_w_golden(
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm,
        amsgrad=amsgrad,
        maximize=maximize,
    )

    # 4. 保存为 bin 文件供 C++ 侧读取
    suffix = f"{d_type}_apply_adam_w.bin"

    # 保存 Input
    var.tofile(f"input_var_{suffix}")
    m.tofile(f"input_m_{suffix}")
    v.tofile(f"input_v_{suffix}")
    grad.tofile(f"input_grad_{suffix}")
    if amsgrad:
        max_grad_norm.tofile(f"input_max_grad_norm_{suffix}")

    beta1_power.tofile(f"input_beta1_power_{suffix}")
    beta2_power.tofile(f"input_beta2_power_{suffix}")
    lr.tofile(f"input_lr_{suffix}")
    weight_decay.tofile(f"input_weight_decay_{suffix}")
    beta1.tofile(f"input_beta1_{suffix}")
    beta2.tofile(f"input_beta2_{suffix}")
    epsilon.tofile(f"input_eps_{suffix}")

    # 保存 Output Golden
    var_out.tofile(f"golden_var_out_{suffix}")
    m_out.tofile(f"golden_m_out_{suffix}")
    v_out.tofile(f"golden_v_out_{suffix}")
    if amsgrad and max_grad_norm_out is not None:
        max_grad_norm_out.tofile(f"golden_max_grad_norm_out_{suffix}")

    print(f"Data generated successfully for shape {shape_str}, dtype {d_type}")
    print(f"Mode: amsgrad={amsgrad}, maximize={maximize}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_data.py <shape> [dtype]")
        print("Example: python3 gen_data.py 2,2 float32")
        exit(1)

    shape_input = sys.argv[1]
    dtype_input = sys.argv[2] if len(sys.argv) > 2 else "float32"

    # 清理遗留 bin 文件
    os.system("rm -rf *.bin")

    # 默认生成开启 amsgrad 和 maximize 模式的数据
    gen_data_and_golden(shape_input, dtype_input, amsgrad=True, maximize=True)
