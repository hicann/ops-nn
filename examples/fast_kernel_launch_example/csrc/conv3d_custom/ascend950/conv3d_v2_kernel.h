/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
  
/*!
 * \file conv3dv2_kernel.h
 * \brief Conv3D V2 kernel函数头文件(Ascend950版本)
 * 
 * 本文件定义了Conv3D V2操作的kernel调用接口和相关的常量定义。
 * 该kernel支持多种tiling策略和ping-pong配置，可以根据硬件特性和数据特征选择最优的实现。
 */
  
#ifndef CONV3D_V2_KERNEL_H
#define CONV3D_V2_KERNEL_H
  
#include <stdint.h>
#include <string>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "conv3d_v2_tiling_data.h"

// 特征图Tiling策略常量定义
#define CONV_FMAP_TILING_FULLLOAD_AL1 0           // 特征图全加载到L1缓冲区
#define CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 1 // 只有M方向全加载到L1和L0
#define CONV_FMAP_TILING_OTHER 2                  // 其他tiling策略

// 权重Tiling策略常量定义
#define CONV_WEIGHT_TILING_FULLLOAD_BL1 0           // 权重全加载到L1缓冲区
#define CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 1 // 只有N方向全加载到L1和L0
#define CONV_WEIGHT_TILING_OTHER 2                  // 其他tiling策略

// L1级ping-pong缓冲区策略常量定义
#define CONV_L1_PINGPONG_ALL_CLOSE 0    // L1级所有缓冲区都关闭ping-pong
#define CONV_L1_PINGPONG_AL1_OPEN 1     // 只在AL1缓冲区开启ping-pong
#define CONV_L1_PINGPONG_BL1_OPEN 2     // 只在BL1缓冲区开启ping-pong
#define CONV_L1_PINGPONG_ALL_OPEN 3     // L1级所有缓冲区都开启ping-pong

// L0级ping-pong缓冲区策略常量定义
#define CONV_L0_PINGPONG_ALL_CLOSE 0    // L0级所有缓冲区都关闭ping-pong
#define CONV_L0_PINGPONG_AL0_OPEN 1     // 只在AL0缓冲区开启ping-pong
#define CONV_L0_PINGPONG_BL0_OPEN 2     // 只在BL0缓冲区开启ping-pong
#define CONV_L0_PINGPONG_ALL_OPEN 3     // L0级所有缓冲区都开启ping-pong

// 输出数据顺序常量定义
#define CONV_OUTPUT_ORDER_HW_MODE 0  // 输出按HW(H,W)顺序排列
#define CONV_OUTPUT_ORDER_M_MODE 1    // 输出按M模式排列

// 迭代顺序常量定义
#define CONV_ITER_ORDER_MITER_FIRST 0  // M方向迭代优先
#define CONV_ITER_ORDER_NITER_FIRST 1  // N方向迭代优先

// 分组卷积类型常量定义
#define CONV_GROUP_TYPE_NORMAL_CONV 0       // 普通卷积(无分组)
#define CONV_GROUP_TYPE_ORI_GROUP_CONV 1    // 原始分组卷积
#define CONV_GROUP_TYPE_OPT_GROUP_CONV 2     // 优化的分组卷积

// 小通道优化标志常量定义
#define CONV_ENABLE_SMALL_CHANNEL_CLOSE 0  // 关闭小通道优化
#define CONV_ENABLE_SMALL_CHANNEL_OPEN 1   // 开启小通道优化

// 权重UB转置标志常量定义
#define CONV_WEIGHT_UB_TRANS_CLOSE 0  // 关闭权重UB转置
#define CONV_WEIGHT_UB_TRANS_OPEN 1   // 开启权重UB转置

// 特征图加载模式常量定义
#define CONV_FMAP_LOAD3D_MODE 0  // 3D加载模式
#define CONV_FMAP_DMA_MODE 1     // DMA加载模式

// 内部batch处理模式常量定义
#define CONV_INNER_BATCH_SINGLE 0             // 单batch处理
#define CONV_INNER_BATCH_KERNEL_1X1_MULTI 1   // 1x1卷核多batch处理
#define CONV_INNER_BATCH_MULTI 2               // 多batch处理

/**
 * Conv3D V2模板kernel调用函数
 * 
 * 该函数根据不同的tiling策略和数据类型，调用相应的conv3dv2_template kernel实现。
 * 通过模板特化和条件编译，支持多种性能优化策略的组合。
 * 
 * @param x: 输入特征图的GM地址
 * @param filter: 卷积核(权重)的GM地址
 * @param bias: 偏置项的GM地址(可为NULL)
 * @param scale: 量化缩放因子的GM地址(可为NULL)
 * @param offset: 量化偏移的GM地址(可为NULL)
 * @param offset_w: 权重量化偏移的GM地址(可为NULL)
 * @param y: 输出特征图的GM地址
 * @param workspace: 工作空间的GM地址(可为NULL)
 * @param tiling: Conv3D tiling数据结构，包含所有tiling参数
 * @param FmapTiling: 特征图tiling策略
 * @param WeightTiling: 权重tiling策略
 * @param L1PingPong: L1级ping-pong策略
 * @param L0PingPong: L0级ping-pong策略
 * @param OutputOrder: 输出数据顺序
 * @param IterOrder: 迭代顺序
 * @param dtype: 数据类型字符串("float"或"float16"等)
 * @param numBlocks: 使用的AI Core数量
 * @param stream: ACL流句柄
 */
void Conv3dv2Template(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace, Ops::NN::Conv3dV2::Conv3DV2TilingData& tiling,
    int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong,
    int8_t OutputOrder, int8_t IterOrder,
    const std::string& dtype,
    int32_t numBlocks, aclrtStream stream);
  
#endif // CONV3D_V2_KERNEL_H
