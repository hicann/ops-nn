#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import math
import torch

# Ascend Cube单元的N维度固定值
CUBE_N = 16

# 数据类型到K0值的映射
# K0是Ascend优化格式中通道块的大小，由数据类型决定：
# - FP32: K0=8 (便于矩阵乘法对齐）
# - FP16/BF16: K0=16 (利用16位数据的优势）
K0_MAP = {
    torch.float32: 8,
    torch.float16: 16,
    torch.bfloat16: 16,
}

"""
    计算最小公倍数

    该函数计算两个整数的最小公倍数（Least Common Multiple）。

    @param m 第一个整数
    @param n 第二个整数
    @return m和n的最小公倍数
"""
def lcm(m, n):
    return (m * n) // math.gcd(m, n)

"""
    向上取整除法

    该函数计算(m / n)的向上取整结果。
    公式：ceil(m / n) = (m + n - 1) // n

    @param m 被除数
    @param n 除数
    @return m除以n的向上取整结果
"""
def ceil(m, n):
    return (m + n - 1) // n


"""
    将3D张量从NCDHW格式转换为NDC1HWC0格式

    该函数将标准3D张量格式转换为Ascend优化的NDC1HWC0格式，
    用于提高Ascend NPU上的计算效率。

    转换过程：
    1. 获取数据类型对应的K0值（FP32:8, FP16/BF16:16）
    2. 计算需要在C维度上padding的数量，使C能被K0整除
    3. 在C维度上进行padding（在尾部添加0）
    4. 将张量reshape为6D: [N, C1, K0, D, H, W]
    5. 将张量permute为NDC1HWC0格式: [N, D, C1, H, W, K0]

    格式转换示例：
    - 输入: [N=2, C=3, D=4, H=5, W=6], K0=16
    - padding后的C: 16 (pad_value=13)
    - reshape: [2, 1, 16, 4, 5, 6]
    - 输出: [2, 4, 1, 5, 6, 16]

    @param input_tensor 输入张量，格式为NCDHW [N, C, D, H, W]
    @return 转换后的张量，格式为NDC1HWC0 [N, D, C1, H, W, K0]
"""
def ncdhw_to_ndc1hwc0(input_tensor):
    # 获取数据类型对应的K0值
    cube_k = K0_MAP[input_tensor.dtype]
    
    input_shape = input_tensor.shape
    dim_c = input_shape[1]  # 获取通道数C
    
    # 计算padding数量，使C能被K0整除
    # pad_value = (K0 - C % K0) % K0
    pad_value = (cube_k - dim_c % cube_k) % cube_k
    
    # 在C维度上进行padding（在尾部添加0）
    # padding格式: (W_left, W_right, H_left, H_right, D_left, D_right, C_left, C_right)
    res_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, 0, 0, 0, 0, pad_value), mode="constant", value=0)
    
    # 计算C1 = ceil(padded_C / K0)
    dim_c1 = (dim_c + pad_value) // cube_k
    
    # reshape为6D格式: [N, C1, K0, D, H, W]
    res_tensor = res_tensor.reshape((input_shape[0], dim_c1, cube_k, input_shape[2], input_shape[3], input_shape[4]))
    
    # permute为NDC1HWC0格式: [N, D, C1, H, W, K0]
    # 原始索引: [0, 1, 2, 3, 4, 5] -> [N, C1, K0, D, H, W]
    # 目标索引: [0, 3, 1, 4, 5, 2] -> [N, D,  C1, H, W, K0]
    res_tensor = res_tensor.permute(0, 3, 1, 4, 5, 2)
    
    return res_tensor.contiguous()


"""
    将3D张量从NDC1HWC0格式转换回NCDHW格式

    该函数将Ascend优化的NDC1HWC0格式转换回标准的NCDHW格式。
    转换过程需要两个步骤：
    1. 内部函数ndc1hwc0_to_ndchw：将NDC1HWC0转换为NCDHW，处理填充的通道
    2. 将结果permute回NCDHW格式

    转换过程：
    1. 创建NCDHW格式的输出张量，C维度使用真实的通道数real_cout
    2. 遍历NDC1HWC0张量，将有效通道数据复制到输出张量
    3. 跳过padding填充的通道（k * K0 + m >= real_cout）
    4. 将NCDHW格式转换为标准NCDHW格式

    @param output_tensor 输出张量，格式为NDC1HWC0 [N, D, C1, H, W, K0]
    @param real_cout 真实的输出通道数（不含padding）
    @return 转换后的张量，格式为NCDHW [N, C, D, H, W]
"""
def ndc1hwc0_to_ncdhw(output_tensor, real_cout):
    # 内部函数：将NDC1HWC0转换为NCDHW
    def ndc1hwc0_to_ndchw():
        output_shape = output_tensor.shape
        # 创建输出张量，形状为[N, D, real_cout, H, W]
        res_shape = [output_shape[0], output_shape[1], real_cout, output_shape[3], output_shape[4]]
        res_tensor = torch.zeros(res_shape, dtype=output_tensor.dtype)
        
        # 遍历所有维度，将有效通道数据复制到输出张量
        for i in range(output_shape[0]):  # N维度
            for j in range(output_shape[1]):  # D维度
                for k in range(output_shape[2]):  # C1维度
                    for m in range(output_shape[5]):  # K0维度
                        # 只复制有效通道（跳过padding填充的通道）
                        if k * output_shape[5] + m < real_cout:
                            res_tensor[i, j, k * output_shape[5] + m, :, :] = output_tensor[i, j, k, :, :, m]
        return res_tensor
    
    # 第一步：将NDC1HWC0转换为NCDHW
    res_tensor = ndc1hwc0_to_ndchw()
    
    # 第二步：将NCDHW格式转换为标准NCDHW格式
    # 原始: [N, D, C, H, W]
    # 目标: [N, C, D, H, W]
    res_tensor = res_tensor.permute(0, 2, 1, 3, 4)
    
    return res_tensor.contiguous()


"""
    将3D卷积核从NCDHW格式转换为FRACTAL_Z_3D格式

    该函数将标准的3D卷积核格式转换为Ascend优化的FRACTAL_Z_3D格式。
    FRACTAL_Z格式是Ascend硬件优化的权重布局，专门用于矩阵乘法加速。

    转换过程：
    1. 根据数据类型获取K0值（FP32:8, FP16/BF16:16）
    2. 计算扩展因子enlarge：考虑输入输出通道数和分组数
    3. 计算优化后的通道数：使能被K0或CUBE_N整除
    4. 创建FRACTAL_Z格式的输出张量
    5. 填充FRACTAL_Z格式张量，处理分组卷积
    6. reshape为最终的FRACTAL_Z_3D格式

    FRACTAL_Z_3D格式结构：
    - 第一维度：G_opt * Kd * Cin_opt/K0 * Kh * Kw (合并了多个维度）
    - 第二维度：N1 = Cout_opt / CUBE_N
    - 第三维度：N0 = CUBE_N = 16
    - 第四维度：K0 (8 or 16)

    @param filter_tensor 输入卷积核张量，格式为NCDHW [Co, Ci, Kd, Kh, Kw]
    @param groups 分组数，默认为1
    @return 转换后的卷积核张量，格式为FRACTAL_Z_3D
"""
def ncdhw_to_fz3d(filter_tensor, groups=1):
    # 获取数据类型对应的K0值
    cube_k = K0_MAP[filter_tensor.dtype]

    # 提取原始卷积核形状
    cout, cin_ori, dk, kh, kw = filter_tensor.shape  # [Co, Ci, Kd, Kh, Kw]
    cout_ori = cout // groups  # 每组的输出通道数
    # 计算扩展因子enlarge
    # 考虑以下因素：
    # - 输入通道数cin_ori和K0的关系
    # - 输出通道数cout_ori和CUBE_N的关系
    # - 分组数groups
    enlarge = min(lcm(lcm(cin_ori, cube_k) // cin_ori, lcm(cout_ori, CUBE_N) // cout_ori), groups)
    
    # 计算优化后的通道数
    # Cin_opt: 使(enlarge * Cin_ori)能被K0整除后，再乘以K0
    # Cout_opt: 使(enlarge * Cout_ori)能被CUBE_N整除后，再乘以CUBE_N
    cin_opt = ceil(enlarge * cin_ori, cube_k) * cube_k
    cout_opt = ceil(enlarge * cout_ori, CUBE_N) * CUBE_N
    group_opt = ceil(groups, enlarge)  # 优化后的分组数
 
    # 创建FRACTAL_Z格式的输出张量
    # 形状：[G_opt, Kd, Cin_opt/K0, Kh, Kw, Cout_opt, K0]
    filter_shape_gdc1hwnc0 = [group_opt, dk, cin_opt // cube_k, kh, kw, cout_opt, cube_k]
    res_tensor = torch.zeros(filter_shape_gdc1hwnc0, dtype=filter_tensor.dtype)
    
    # 填充FRACTAL_Z格式张量
    for g in range(groups):  # 遍历所有分组
        for ci in range(cin_ori):  # 遍历输入通道
            for co in range(cout_ori):  # 遍历输出通道
                for kernel_d in range(dk):  # 遍历卷积核深度
                    # 计算目标索引
                    e = g % enlarge
                    dst_ci = e * cin_ori + ci
                    dst_co = e * cout_ori + co
                    src_co = g * cout_ori + co
                    
                    # 复制卷积核数据到目标位置
                    res_tensor[g // enlarge, kernel_d, dst_ci // cube_k, :, :, dst_co,
                    dst_ci % cube_k] = filter_tensor[
                                       src_co, ci,
                                       kernel_d, :,
                                       :]
    # reshape为最终的FRACTAL_Z_3D格式
    # 第一维度合并了多个维度：G_opt * Kd * (Cin_opt/K0) * Kh * Kw
    res_tensor = res_tensor.reshape(group_opt * dk * (cin_opt // cube_k) * kh * kw, cout_opt // CUBE_N, CUBE_N,
                                    cube_k)
    
    return res_tensor.contiguous()
