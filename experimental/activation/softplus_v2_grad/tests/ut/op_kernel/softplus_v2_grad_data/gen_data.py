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
import torch
import numpy as np
from ml_dtypes import bfloat16

# 数据类型映射：命令行字符串 → numpy类型 / torch类型
DTYPE_MAP = {
    "float32": (np.float32, torch.float32),
    "float16": (np.float16, torch.float16),
    "bfloat16": (bfloat16, torch.bfloat16)
}

def parse_str_to_shape_list(shape_str):
    """解析形状字符串（如"(2,3,4)"）为整数列表"""
    shape_str = shape_str.strip('(').strip(')').strip()
    if not shape_str:
        raise ValueError("形状字符串不能为空，示例：(2,3)")
    shape_list = [int(x.strip()) for x in shape_str.split(",")]
    return shape_list

def SoftplusBackward(gradOutput, input_x, beta, threshold):
    """计算Softplus反向梯度，并处理bfloat16兼容"""
    BFLOAT16_FLAG = False
    np_type, torch_type = DTYPE_MAP[input_x.dtype.name]  # 统一类型映射
    
    # 转换为torch张量，处理bfloat16精度兼容
    if input_x.dtype == bfloat16:
        BFLOAT16_FLAG = True
        # bfloat16先转float32计算，避免精度丢失
        input_x_torch = torch.from_numpy(input_x.astype(np.float32)).to(torch.bfloat16)
        gradOutput_torch = torch.from_numpy(gradOutput.astype(np.float32)).to(torch.bfloat16)
    else:
        input_x_torch = torch.from_numpy(input_x).to(torch_type)
        gradOutput_torch = torch.from_numpy(gradOutput).to(torch_type)
    
    input_x_torch.requires_grad = True  # 开启梯度追踪
    
    # 构建Softplus算子（指定beta和threshold）
    softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
    y_softplus = softplus(input_x_torch)
    
    # 反向传播计算梯度
    y_softplus.backward(gradOutput_torch, retain_graph=True)
    
    # 提取梯度并还原数据类型
    if BFLOAT16_FLAG:
        x_grad = input_x_torch.grad.to(torch.float32).numpy().astype(bfloat16)
    else:
        x_grad = input_x_torch.grad.numpy().astype(np_type)
    
    return x_grad

def gen_softplus_backward_data(shape_str, dtype_str, beta, threshold):
    """生成输入数据、计算golden并保存为bin文件"""
    # 1. 校验参数
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"不支持的数据类型：{dtype_str}，仅支持：{list(DTYPE_MAP.keys())}")
    np_type, _ = DTYPE_MAP[dtype_str]
    shape = parse_str_to_shape_list(shape_str)
    
    # 2. 生成随机输入数据
    # gradOutput：-5~5的随机数
    gradOutput = np.random.uniform(-5, 5, shape).astype(np_type)
    # input_x：-10~10的随机数（覆盖Softplus阈值常见范围）
    input_x = np.random.uniform(-10, 10, shape).astype(np_type)
    
    # 3. 计算反向梯度（golden）
    x_grad = SoftplusBackward(gradOutput, input_x, beta, threshold)
    
    # 4. 保存二进制文件（含输入、参数、golden）
    # 输入数据
    gradOutput.tofile(f"{dtype_str}_gradOutput_softplus_backward.bin")
    input_x.tofile(f"{dtype_str}_input_x_softplus_backward.bin")
    # 标量参数（beta/threshold）：保存为float32（通用精度）
    np.array([beta], dtype=np.float32).tofile(f"{dtype_str}_beta_softplus_backward.bin")
    np.array([threshold], dtype=np.float32).tofile(f"{dtype_str}_threshold_softplus_backward.bin")
    # 梯度golden
    x_grad.tofile(f"{dtype_str}_golden_x_grad_softplus_backward.bin")
    
    print(f"数据生成完成！")
    print(f"形状：{shape} | 类型：{dtype_str} | beta：{beta} | threshold：{threshold}")
    print(f"生成文件：")
    print(f"  - {dtype_str}_gradOutput_softplus_backward.bin (反向输入梯度)")
    print(f"  - {dtype_str}_input_x_softplus_backward.bin (Softplus输入x)")
    print(f"  - {dtype_str}_beta_softplus_backward.bin (beta参数)")
    print(f"  - {dtype_str}_threshold_softplus_backward.bin (阈值参数)")
    print(f"  - {dtype_str}_golden_x_grad_softplus_backward.bin (预期梯度输出)")

if __name__ == "__main__":
    # 命令行参数校验
    if len(sys.argv) != 5:
        print("【用法错误】参数数量不符！")
        print("正确格式：python script.py <形状字符串> <数据类型> <beta> <threshold>")
        print("示例：")
        print("  python script.py '(2,3)' float32 1.0 20.0")
        print("  python script.py '(1,1024)' bfloat16 0.5 15.0")
        sys.exit(1)
    
    # 解析命令行参数
    shape_str = sys.argv[1]
    dtype_str = sys.argv[2].lower()  # 统一小写
    try:
        beta = float(sys.argv[3])
        threshold = float(sys.argv[4])
    except ValueError:
        print("【错误】beta和threshold必须是浮点数！")
        sys.exit(1)
    
    # 清理旧bin文件
    os.system("rm -rf *_softplus_backward.bin")
    
    # 生成数据
    try:
        gen_softplus_backward_data(shape_str, dtype_str, beta, threshold)
    except Exception as e:
        print(f"【生成失败】{e}")
        sys.exit(1)