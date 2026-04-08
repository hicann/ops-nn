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

import torch
from torch import Tensor
import torch_npu
from ascend_ops.trans_conv3d_format import ncdhw_to_ndc1hwc0, ncdhw_to_fz3d, ndc1hwc0_to_ncdhw

# 导出公共接口
__all__ = ["conv3d_custom",]

"""
    3D卷积自定义算子接口函数

    该函数是3D卷积操作的高级Python接口，负责：
    1. 根据NPU设备类型选择对应的算子实现
    2. 对于Ascend910B设备，进行数据格式转换
    3. 调用底层C++算子实现
    4. 返回计算结果

    设备支持情况：
    - Ascend950PR_9589: 使用conv3d_v2_custom算子，输入输出为NCDHW格式
    - Ascend910B (其他设备): 使用conv3d_custom算子，需要转换为NDC1HWC0和FRACTAL_Z_3D格式

    数据格式说明：
    - NCDHW: 标准格式 [N, C, D, H, W]
    - NDC1HWC0: Ascend优化格式 [N, D, C1, H, W, C0]，其中C0=16
    - FRACTAL_Z_3D: 权重格式 [Kd*C1*Kh*Kw, N1, N0, C0]

    @param input 输入张量 [N, C, D, H, W] (NCDHW格式）
    @param weight 权重张量 [Co, Ci, Kd, Kh, Kw] (NCDHW格式）
    @param strides 步长 [strideD, strideH, strideW]
    @param pads padding [padD, padH, padW] (每个方向2面，共6面）
    @param dilations 膨胀 [dilationD, dilationH, dilationW]
    @param bias 偏置张量 [Co] (可选）
    @param enable_hf32 是否启用HF32混合精度模式（仅Ascend910B支持）
    @return 输出张量 [N, Co, Do, Ho, Wo] (NCDHW格式）
"""
def conv3d_custom(input: Tensor, weight: Tensor, strides: list, pads: list, dilations: list,
                  bias: Tensor = None, enable_hf32: bool = False) -> Tensor:
    # 判断当前NPU设备类型
    if torch_npu.npu.get_device_name() == "Ascend950PR_9589":
        # Ascend950设备路径：使用conv3d_v2_custom算子
        
        # 检查算子是否已注册
        print(torch.ops.ascend_ops.conv3d_v2_custom)
        assert hasattr(torch.ops.ascend_ops, "conv3d_v2_custom"), "The 'conv3d_v2_custom' operator is not registered in 'torch.ops.ascend_ops' namespace."
        # 保存原始输入输出形状
        origin_input_shape = list(input.shape)
        origin_weight_shape = list(weight.shape)
        # 将张量移动到NPU设备
        input = input.npu()
        weight = weight.npu()
        bias = bias.npu() if bias is not None else None
        # 调用底层C++算子conv3d_v2_custom执行3D卷积
        # Ascend950设备不需要格式转换，直接使用NCDHW格式
        output = torch.ops.ascend_ops.conv3d_v2_custom(input, weight, strides, pads, dilations, origin_input_shape,
                                                    origin_weight_shape, bias).cpu()
        return output
    
    # Ascend910B设备路径：使用conv3d_custom算子
    
    # 检查算子是否已注册
    print(torch.ops.ascend_ops.conv3d_custom)
    assert hasattr(torch.ops.ascend_ops, "conv3d_custom"), "The 'conv3d_custom' operator is not registered in 'torch.ops.ascend_ops' namespace."
    
    # 保存原始输入输出形状
    origin_input_shape = list(input.shape)
    origin_weight_shape = list(weight.shape)
    origin_cout = origin_weight_shape[0]  # 输出通道数
    
    # 数据格式转换：NCDHW -> NDC1HWC0 (输入/输出）和 FRACTAL_Z_3D (权重）
    # Ascend910B设备需要使用优化的数据格式以提高计算效率
    input = ncdhw_to_ndc1hwc0(input).npu()  # 输入: [N, C, D, H, W] -> [N, D, C1, H, W, C0]
    weight = ncdhw_to_fz3d(weight).npu()  # 权重: [Co, Ci, Kd, Kh, Kw] -> [Kd*C1*Kh*Kw, N1, N0, C0]
    bias = bias.npu() if bias is not None else None
    
    # 调用底层C++算子conv3d_custom执行3D卷积
    output = torch.ops.ascend_ops.conv3d_custom(input, weight, strides, pads, dilations, origin_input_shape,
                                                origin_weight_shape, bias, enable_hf32).cpu()
    
    # 数据格式转换：NDC1HWC0 -> NCDHW (输出）
    # 将Ascend优化格式转换回标准格式
    return ndc1hwc0_to_ncdhw(output, origin_cout)