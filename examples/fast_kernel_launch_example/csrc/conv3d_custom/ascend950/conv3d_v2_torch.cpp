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
 * \file conv3d_v2_torch.cpp
 * \brief 基于Ascend 950处理器的3D卷积操作实现
 * 
 * 本文件实现了一个自定义的PyTorch扩展，用于在Ascend NPU上执行3D卷积操作。
 * 该实现使用了AscendC编程模型，通过tiling策略、双缓冲和流水线技术优化数据传输和计算性能。
 * 支持3D卷积操作，数据格式为NCDHW（批次、通道、深度、高度、宽度）。
 * 
 * 主要功能：
 * 1. 支持多种数据类型：FP16、BF16、FP32
 * 2. 支持stride、padding、dilation等卷积参数
 * 3. 使用tiling策略进行数据分块优化
 * 4. 支持bias偏置项
 * 5. 自动计算输出张量形状
 */

#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include "tiling/platform/platform_ascendc.h"
#include "acl/acl.h"
#include "graph/types.h"
#include "conv3d_v2_tiling_data.h"
#include "conv_template_utils.h"
#include "conv_base_numblocks_decision.h"
#include "kernel_operator.h"
#include "conv3d_v2_api_tiling.h"
#include "conv3d_v2_base_tiling_template_tilingkey.h"
#include "conv3d_v2_kernel.h"
#include "conv_api_tiling_util.h"
#include <torch/all.h>
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#define DEMO_RET_FAIL       (-1)  // 操作失败的返回值
#define DEMO_RET_SUC        (0)   // 操作成功的返回值

// NCDHW格式中各维度的索引
constexpr uint32_t FORMAT_NCDHW_N_INDEX = 0;  // NCDHW中N（批次）维度的索引
constexpr uint32_t FORMAT_NCDHW_C_INDEX = 1;  // NCDHW中C（通道）维度的索引
constexpr uint32_t FORMAT_NCDHW_D_INDEX = 2;  // NCDHW中D（深度）维度的索引
constexpr uint32_t FORMAT_NCDHW_H_INDEX = 3;  // NCDHW中H（高度）维度的索引
constexpr uint32_t FORMAT_NCDHW_W_INDEX = 4;  // NCDHW中W（宽度）维度的索引

// 卷积属性（stride、padding、dilation）在3D空间中的索引
constexpr uint8_t ATTRS_D_DIM_IDX_NCDHW = 0;  // 3D空间中D（深度）维度的索引
constexpr uint8_t ATTRS_H_DIM_IDX_NCDHW = 1;  // 3D空间中H（高度）维度的索引
constexpr uint8_t ATTRS_W_DIM_IDX_NCDHW = 2;  // 3D空间中W（宽度）维度的索引

// 6面padding的索引
constexpr uint32_t PAD_HEAD_INDEX = 0;   // 头部padding索引
constexpr uint32_t PAD_TAIL_INDEX = 1;   // 尾部padding索引
constexpr uint32_t PAD_UP_INDEX = 2;     // 上部padding索引
constexpr uint32_t PAD_DOWN_INDEX = 3;   // 下部padding索引
constexpr uint32_t PAD_LEFT_INDEX = 4;    // 左侧padding索引
constexpr uint32_t PAD_RIGHT_INDEX = 5;  // 右侧padding索引

constexpr static uint8_t NCDHW_DIM = 5;        // NCDHW格式的维度数（5维）
constexpr static uint8_t CONV_ATTRS_DIM = 3;   // 卷积属性（stride、padding、dilation）的维度数（3D）

namespace ascend_ops {
namespace Conv3dCustom {

/*!
 * \brief 初始化3D卷积运行信息
 * 
 * 该函数将tiling信息转换为kernel运行时需要的Conv3DRunInfo结构体。
 * 包含输入输出形状、kernel大小、步长、膨胀、padding等参数。
 * 
 * @param conv3dRunInfo 输出参数，3D卷积运行信息结构体
 * @param tilingInfo 输入参数，tiling信息，包含形状、属性、块维度等信息
 */
static void InitConv3dRunInfo(Ops::NN::Conv3dV2::Conv3DRunInfo& conv3dRunInfo,
                               optiling::conv_ops_tiling::ConvAscendcTilingInfo& tilingInfo)
{
    // 输入输出形状信息
    conv3dRunInfo.batch     = static_cast<uint32_t>(tilingInfo.shapeInfo.batch);  // 批次大小
    conv3dRunInfo.cin       = static_cast<uint32_t>(tilingInfo.shapeInfo.ci);      // 输入通道数
    conv3dRunInfo.din       = static_cast<uint32_t>(tilingInfo.shapeInfo.di);      // 输入深度
    conv3dRunInfo.hin       = static_cast<uint32_t>(tilingInfo.shapeInfo.hi);      // 输入高度
    conv3dRunInfo.win       = static_cast<uint32_t>(tilingInfo.shapeInfo.wi);      // 输入宽度
    conv3dRunInfo.cout      = static_cast<uint32_t>(tilingInfo.shapeInfo.co);      // 输出通道数
    
    // 卷积核大小
    conv3dRunInfo.kd        = static_cast<uint32_t>(tilingInfo.shapeInfo.kd);      // 卷积核深度
    conv3dRunInfo.kh        = static_cast<uint32_t>(tilingInfo.shapeInfo.kh);      // 卷积核高度
    conv3dRunInfo.kw        = static_cast<uint32_t>(tilingInfo.shapeInfo.kw);      // 卷积核宽度
    
    // 输出形状信息
    conv3dRunInfo.dout      = static_cast<uint32_t>(tilingInfo.shapeInfo.dout);    // 输出深度
    conv3dRunInfo.hout      = static_cast<uint32_t>(tilingInfo.shapeInfo.ho);      // 输出高度
    conv3dRunInfo.wout      = static_cast<uint32_t>(tilingInfo.shapeInfo.wo);      // 输出宽度
    
    // 块维度信息（用于并行计算）
    conv3dRunInfo.batchDim  = tilingInfo.numBlocksRes.batchDim;  // 批次维度分块数
    conv3dRunInfo.doDim     = tilingInfo.numBlocksRes.doDim;     // 输出深度维度分块数
    conv3dRunInfo.mDim      = tilingInfo.numBlocksRes.mDim;      // M维度（输出通道）分块数
    conv3dRunInfo.wDim      = tilingInfo.numBlocksRes.woDim;     // 输出宽度维度分块数
    conv3dRunInfo.nDim      = tilingInfo.numBlocksRes.nDim;      // N维度（批次）分块数
    conv3dRunInfo.groupDim  = tilingInfo.numBlocksRes.groupDim;  // 组维度分块数
    conv3dRunInfo.hoDim     = tilingInfo.numBlocksRes.hoDim;     // 输出高度维度分块数
    
    // 卷积参数：步长
    conv3dRunInfo.strideD   = static_cast<uint32_t>(tilingInfo.attrInfo.strideD);  // 深度方向步长
    conv3dRunInfo.strideH   = static_cast<uint32_t>(tilingInfo.attrInfo.strideH);  // 高度方向步长
    conv3dRunInfo.strideW   = static_cast<uint32_t>(tilingInfo.attrInfo.strideW);  // 宽度方向步长
    
    // 卷积参数：膨胀（dilation）
    conv3dRunInfo.dilationD = static_cast<uint32_t>(tilingInfo.attrInfo.dilationD);  // 深度方向膨胀
    conv3dRunInfo.dilationH = static_cast<uint32_t>(tilingInfo.attrInfo.dilationH);  // 高度方向膨胀
    conv3dRunInfo.dilationW = static_cast<uint32_t>(tilingInfo.attrInfo.dilationW);  // 宽度方向膨胀
    
    // 卷积参数：padding
    conv3dRunInfo.padHead   = static_cast<uint32_t>(tilingInfo.attrInfo.padHead);    // 头部padding
    conv3dRunInfo.padTail   = static_cast<uint32_t>(tilingInfo.attrInfo.padTail);    // 尾部padding
    conv3dRunInfo.padTop    = static_cast<uint32_t>(tilingInfo.attrInfo.padTop);     // 顶部padding
    conv3dRunInfo.padBottom = static_cast<uint32_t>(tilingInfo.attrInfo.padBottom);  // 底部padding
    conv3dRunInfo.padLeft   = static_cast<uint32_t>(tilingInfo.attrInfo.padLeft);    // 左侧padding
    conv3dRunInfo.padRight  = static_cast<uint32_t>(tilingInfo.attrInfo.padRight);  // 右侧padding
    
    // 其他参数
    conv3dRunInfo.groups    = 1;                                                    // 分组数（当前为1，表示标准卷积）
    conv3dRunInfo.enlarge   = 1;                                                    // 扩展标志
    conv3dRunInfo.cinOpt    = static_cast<uint32_t>(tilingInfo.shapeInfo.ci);      // 优化后的输入通道数
    conv3dRunInfo.coutOpt   = static_cast<uint32_t>(tilingInfo.shapeInfo.co);      // 优化后的输出通道数
    conv3dRunInfo.groupOpt  = 1;                                                    // 优化后的分组数
    conv3dRunInfo.hasBias   = static_cast<uint8_t>(tilingInfo.flagInfo.hasBias);   // 是否有偏置项
    
    // 根据分割模式确定hoDim
    if (tilingInfo.flagInfo.mSplitModeFlag) {
        // M分割模式：使用mDim作为hoDim
        conv3dRunInfo.hoDim = static_cast<uint32_t>(tilingInfo.numBlocksRes.mDim);
    } else {
        // 标准模式：使用hoDim作为hoDim
        conv3dRunInfo.hoDim = static_cast<uint32_t>(tilingInfo.numBlocksRes.hoDim);
    }
}

/*!
 * \brief 初始化tiling数据结构
 * 
 * 该函数将tiling信息转换为kernel运行时需要的Conv3DV2TilingData结构体。
 * 该结构体包含所有kernel执行所需的参数和配置信息。
 * 
 * @param tilingData 输出参数，tiling数据结构体
 * @param tilingInfo 输入参数，tiling信息
 */
static void InitTilingData(Ops::NN::Conv3dV2::Conv3DV2TilingData& tilingData,
    optiling::conv_ops_tiling::ConvAscendcTilingInfo& tilingInfo)
{
    // 将tiling信息转换为Conv3DRunInfo结构体
    InitConv3dRunInfo(tilingData.conv3dRunInfo, tilingInfo);
}


/*!
 * \brief 初始化卷积基础配置信息
 * 
 * 该函数初始化卷积操作的形状信息、数据类型、数据格式和功能标志。
 * 这些信息是tiling策略计算的基础。
 * 
 * @param oriInputShapeList 输入张量的原始形状 [N, C, D, H, W]
 * @param oriWeightShapeList 权重张量的原始形状 [Co, Ci, Kd, Kh, Kw]
 * @param oriOutputShapeList 输出张量的原始形状 [N, Co, Do, Ho, Wo]
 * @param tilingInfo 输出参数，tiling信息结构体
 * @param dataType 输入输出的数据类型（FP16、BF16、FP32等）
 * @return DEMO_RET_SUC表示成功，DEMO_RET_FAIL表示失败
 */
static int32_t InitConvBase(const vector<int64_t>& oriInputShapeList,
    const vector<int64_t>& oriWeightShapeList, const vector<int64_t>& oriOutputShapeList,
    optiling::conv_ops_tiling::ConvAscendcTilingInfo& tilingInfo, ge::DataType dataType)
{
    // 设置输入形状信息
    tilingInfo.shapeInfo.batch  = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_N_INDEX]);  // 批次大小
    tilingInfo.shapeInfo.ci     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_C_INDEX]);  // 输入通道数
    tilingInfo.shapeInfo.di     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_D_INDEX]);  // 输入深度
    tilingInfo.shapeInfo.hi     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_H_INDEX]);  // 输入高度
    tilingInfo.shapeInfo.wi     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_W_INDEX]);  // 输入宽度
    
    // 设置权重形状信息
    tilingInfo.shapeInfo.kd     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_D_INDEX]); // 卷积核深度
    tilingInfo.shapeInfo.kh     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_H_INDEX]); // 卷积核高度
    tilingInfo.shapeInfo.kw     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_W_INDEX]); // 卷积核宽度
    tilingInfo.shapeInfo.co     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_N_INDEX]); // 输出通道数
    
    // 设置输出形状信息
    tilingInfo.shapeInfo.dout   = static_cast<uint64_t>(oriOutputShapeList[FORMAT_NCDHW_D_INDEX]); // 输出深度
    tilingInfo.shapeInfo.ho     = static_cast<uint64_t>(oriOutputShapeList[FORMAT_NCDHW_H_INDEX]); // 输出高度
    tilingInfo.shapeInfo.wo     = static_cast<uint64_t>(oriOutputShapeList[FORMAT_NCDHW_W_INDEX]); // 输出宽度

    // 设置各张量的数据类型
    tilingInfo.descInfo.weightDtype    = dataType;  // 权重数据类型
    tilingInfo.descInfo.fMapDtype      = dataType;  // 特征图数据类型
    tilingInfo.descInfo.biasDtype      = dataType;  // 偏置数据类型
    tilingInfo.descInfo.outDtype       = dataType;  // 输出数据类型
    tilingInfo.descInfo.out1Dtype      = dataType;  // 辅助输出数据类型

    // 设置各张量的数据格式
    tilingInfo.descInfo.weightFormat   = ge::FORMAT_NCDHW;  // 权重格式：NCDHW
    tilingInfo.descInfo.fMapFormat     = ge::FORMAT_NCDHW;  // 特征图格式：NCDHW
    tilingInfo.descInfo.biasFormat     = ge::FORMAT_NCDHW;  // 偏置格式：NCDHW
    tilingInfo.descInfo.outFormat      = ge::FORMAT_NCDHW;  // 输出格式：NCDHW
    tilingInfo.descInfo.out1Format     = ge::FORMAT_NCDHW;  // 辅助输出格式：NCDHW
    tilingInfo.descInfo.scaleFormat    = ge::FORMAT_NCDHW;  // 缩放因子格式：NCDHW

    // 设置缩放因子的数据类型
    tilingInfo.descInfo.scaleDtype = ge::DataType::DT_INT64;
    
    // 设置功能标志（全部禁用，使用标准卷积）
    tilingInfo.flagInfo.quantFlag      = false;  // 量化标志
    tilingInfo.flagInfo.extendConvFlag = false;  // 扩展卷积标志
    tilingInfo.flagInfo.enableC04Flag  = false;  // C04优化标志
    tilingInfo.flagInfo.mSplitModeFlag = false;  // M分割模式标志
    tilingInfo.flagInfo.convGroupType  = optiling::conv_ops_tiling::ConvGroupType::NORMAL_CONV;  // 卷积类型：标准卷积
    tilingInfo.flagInfo.mBasicBlockFlag = false; // M基本块标志
    tilingInfo.flagInfo.useTilingRepo  = false; // 使用tiling仓库标志
    tilingInfo.flagInfo.useTilingCache = false; // 使用tiling缓存标志

    return DEMO_RET_SUC;
}

/*!
 * \brief 初始化卷积属性信息
 * 
 * 该函数初始化卷积操作的属性参数，包括步长、膨胀、padding等。
 * 
 * @param strideList 步长列表 [strideD, strideH, strideW]
 * @param paddingList padding列表 [padHead, padTail, padTop, padBottom, padLeft, padRight]
 * @param dilationList 膨胀列表 [dilationD, dilationH, dilationW]
 * @param attrInfo 输出参数，卷积属性信息结构体
 * @return DEMO_RET_SUC表示成功，DEMO_RET_FAIL表示失败
 */
static int32_t InitAttrInfo(const vector<int64_t>& strideList, const vector<int64_t>& paddingList,
    const vector<int64_t>& dilationList, optiling::conv_ops_tiling::ConvAscendcAttrInfo& attrInfo)
{
    // 设置膨胀参数（控制卷积核元素之间的间隔）
    attrInfo.dilationD  = dilationList[ATTRS_D_DIM_IDX_NCDHW];  // 深度方向膨胀
    attrInfo.dilationH  = dilationList[ATTRS_H_DIM_IDX_NCDHW];  // 高度方向膨胀
    attrInfo.dilationW  = dilationList[ATTRS_W_DIM_IDX_NCDHW];  // 宽度方向膨胀
    
    // 设置步长参数（控制卷积核滑动的步幅）
    attrInfo.strideD    = strideList[ATTRS_D_DIM_IDX_NCDHW];    // 深度方向步长
    attrInfo.strideH    = strideList[ATTRS_H_DIM_IDX_NCDHW];    // 高度方向步长
    attrInfo.strideW    = strideList[ATTRS_W_DIM_IDX_NCDHW];    // 宽度方向步长
    
    // 设置padding参数（在输入张量周围填充的元素数量）
    attrInfo.padHead    = paddingList[PAD_HEAD_INDEX];   // 头部padding（深度方向负方向）
    attrInfo.padTail    = paddingList[PAD_TAIL_INDEX];   // 尾部padding（深度方向正方向）
    attrInfo.padTop     = paddingList[PAD_UP_INDEX];     // 顶部padding（高度方向负方向）
    attrInfo.padBottom  = paddingList[PAD_DOWN_INDEX];   // 底部padding（高度方向正方向）
    attrInfo.padLeft    = paddingList[PAD_LEFT_INDEX];   // 左侧padding（宽度方向负方向）
    attrInfo.padRight   = paddingList[PAD_RIGHT_INDEX];  // 右侧padding（宽度方向正方向）
    
    // 设置其他属性参数
    attrInfo.hf32Mode   = 0;    // HF32模式标志
    attrInfo.offsetx    = 0;    // X方向偏移
    attrInfo.groups     = 1;    // 分组数（1表示标准卷积）
    attrInfo.roundMode  = 0;    // 舍入模式
    attrInfo.dualOutput = 0;    // 双输出标志
    
    return DEMO_RET_SUC;
}

/*!
 * \brief 初始化平台信息
 * 
 * 该函数查询Ascend NPU平台的各种硬件资源信息，包括：
 * - AI Core数量
 * - 各级缓存大小（L1、L0_A、L0_B、L0_C、UB、BT）
 * - L2缓存带宽
 * - NPU架构类型
 * 这些信息用于tiling策略的计算和资源分配决策。
 * 
 * @param platformInfo 输出参数，平台信息结构体
 * @return DEMO_RET_SUC表示成功，DEMO_RET_FAIL表示失败
 */
static int32_t InitPlatformInfo(optiling::conv_ops_tiling::ConvAscendcPlatformInfo& platformInfo)
{
    // 获取AscendC平台实例
    platform_ascendc::PlatformAscendC* ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return DEMO_RET_FAIL;
    }

    // 获取AI Core数量
    platformInfo.aicariNum = ascendcPlatform->GetCoreNumAic();
    
    // 获取各级缓存的内存大小
    uint64_t size {};
    
    // L1缓存：AI Core的一级缓存
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, size);
    platformInfo.l1Size = size;
    
    // L0_A缓存：矩阵乘法A输入的零级缓存
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, size);
    platformInfo.l0aSize = size;
    
    // L0_B缓存：矩阵乘法B输入的零级缓存
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, size);
    platformInfo.l0bSize = size;
    
    // L0_C缓存：矩阵乘法C输出的零级缓存
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, size);
    platformInfo.l0cSize = size;
    
    // UB缓存：Unified Buffer，AI Core的通用缓冲区
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, size);
    platformInfo.ubSize = size;
    
    // BT缓存：Buffer Tensor，用于张量存储
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, size);
    platformInfo.btSize = size;
    
    // 获取L2缓存的带宽
    uint64_t bwSize = 0;
    ascendcPlatform->GetCoreMemBw(platform_ascendc::CoreMemType::L2, bwSize);
    platformInfo.l2Rate = bwSize;
    
    // 设置NPU架构类型（DAV_3510表示Ascend 950）
    platformInfo.npuArch = NpuArch::DAV_3510;

    return DEMO_RET_SUC;
}

/*!
 * \brief 初始化节点信息
 * 
 * 该函数设置计算图节点的名称和类型，用于调试和日志记录。
 * 
 * @param nodeInfo 输出参数，节点信息结构体
 * @return DEMO_RET_SUC表示成功，DEMO_RET_FAIL表示失败
 */
static int32_t InitNodeInfo(optiling::conv_ops_tiling::ConvAscendcNodeInfo& nodeInfo)
{
    nodeInfo.nodeName = "conv3d_v2";  // 节点名称
    nodeInfo.nodeType = "conv3d_v2";  // 节点类型
    return DEMO_RET_SUC;
}

/*!
 * \brief 3D卷积自定义API函数模板
 * 
 * 该函数是3D卷积操作的核心实现，负责：
 * 1. 初始化tiling信息（属性、平台、节点、形状等）
 * 2. 根据数据类型初始化卷积基础配置
 * 3. 计算分块信息（numBlocks）
 * 4. 获取tiling数据
 * 5. 获取tiling键（用于选择最优的kernel实现）
 * 6. 调用Conv3dv2Template函数执行实际的卷积计算
 * 
 * @tparam T 数据类型（FP16、BF16、FP32）
 * @param stream ACL流，用于异步执行
 * @param input 输入张量 [N, C, D, H, W]
 * @param weight 权重张量 [Co, Ci, Kd, Kh, Kw]
 * @param output 输出张量 [N, Co, Do, Ho, Wo]
 * @param strideList 步长列表 [strideD, strideH, strideW]
 * @param paddingList padding列表 [padHead, padTail, padTop, padBottom, padLeft, padRight]
 * @param dilationList 膨胀列表 [dilationD, dilationH, dilationW]
 * @param oriInputShapeList 输入张量的原始形状
 * @param oriWeightShapeList 权重张量的原始形状
 * @param bias 偏置张量（可选）
 */
template <typename T>
void Conv3dV2CustomApi(
    aclrtStream stream, const at::Tensor& input, const at::Tensor& weight, const torch::Tensor& output,
    const vector<int64_t> strideList, const vector<int64_t> paddingList, const vector<int64_t> dilationList,
    const vector<int64_t> oriInputShapeList, const vector<int64_t> oriWeightShapeList,
    const c10::optional<torch::Tensor>& bias)
{
    // 获取输出张量的形状
    vector<int64_t> oriOutputShapeList = {
        output.size(FORMAT_NCDHW_N_INDEX), output.size(FORMAT_NCDHW_C_INDEX), output.size(FORMAT_NCDHW_D_INDEX),
        output.size(FORMAT_NCDHW_H_INDEX), output.size(FORMAT_NCDHW_W_INDEX)};

    // 获取输入、权重、输出张量的设备地址指针
    __gm__ uint8_t* input_ptr = (__gm__ uint8_t*)input.data_ptr<T>();
    __gm__ uint8_t* weight_ptr = (__gm__ uint8_t*)weight.data_ptr<T>();
    __gm__ uint8_t* bias_ptr = nullptr;
    __gm__ uint8_t* output_ptr = (__gm__ uint8_t*)output.data_ptr<T>();

    // 初始化tiling相关结构体
    optiling::conv_ops_tiling::ConvBaseDeci convBaseDeci {};
    optiling::conv_ops_tiling::ConvAscendcTilingInfo tilingInfo {};
    
    // 初始化属性信息、平台信息、节点信息
    TORCH_CHECK(InitAttrInfo(strideList, paddingList, dilationList, tilingInfo.attrInfo) == 0,
                "Failed to initialize attribute information");
    TORCH_CHECK(InitPlatformInfo(tilingInfo.platformInfo) == 0,
                "Failed to initialize platform information");
    TORCH_CHECK(InitNodeInfo(tilingInfo.nodeInfo) == 0,
                "Failed to initialize node information");

    // 根据数据类型初始化卷积基础配置
    string inDtype;
    if constexpr (std::is_same_v<T, c10::Half>) {
        // FP16类型
        TORCH_CHECK(InitConvBase(oriInputShapeList, oriWeightShapeList, oriOutputShapeList, tilingInfo,
                    ge::DataType::DT_FLOAT16) == 0,
                    "Failed to initialize convolution base configuration");
        inDtype = "float16";
    } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
        // BF16类型
        TORCH_CHECK(InitConvBase(oriInputShapeList, oriWeightShapeList, oriOutputShapeList, tilingInfo,
                    ge::DataType::DT_BF16) == 0,
                    "Failed to initialize convolution base configuration with BF16 data type");
        inDtype = "bfloat16";
    } else if constexpr (std::is_same_v<T, float>) {
        // FP32类型
        TORCH_CHECK(InitConvBase(oriInputShapeList, oriWeightShapeList, oriOutputShapeList, tilingInfo,
                    ge::DataType::DT_FLOAT) == 0,
                    "Failed to initialize convolution base configuration with FP32 data type");
        inDtype = "float";
    } else {
        // 不支持的数据类型
        TORCH_CHECK(false, "Unsupported data type: only FP16, BF16, and FP32 are supported");
    }
    
    // 处理bias偏置
    if (bias.has_value()) {
        bias_ptr = (__gm__ uint8_t*)bias.value().data_ptr<T>();
        tilingInfo.flagInfo.hasBias = static_cast<bool>(bias.has_value());
    }
    
    // 获取分块信息（numBlocks）
    TORCH_CHECK(convBaseDeci.GetNumBlocksInfo(tilingInfo) == 0,
        "Failed to get block dimension information from convolution base decision");

    // 初始化tiling数据结构
    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    InitTilingData(tilingData, tilingInfo);

    // 设置平台信息并获取tiling数据
    conv_tiling::PlatformInfo platform;
    platform.npuArch = NpuArch::DAV_3510;
    platform.l1Size = tilingInfo.platformInfo.l1Size;
    platform.l0ASize = tilingInfo.platformInfo.l0aSize;
    platform.l0BSize = tilingInfo.platformInfo.l0bSize;
    platform.l0CSize = tilingInfo.platformInfo.l0cSize;
    platform.ubSize = tilingInfo.platformInfo.ubSize;
    platform.btSize = tilingInfo.platformInfo.btSize;
    conv_tiling::Conv3dTiling conv3dApiTiling(platform);
    conv3dApiTiling.GetTilingData(tilingInfo.attrInfo, tilingInfo.descInfo, tilingInfo.flagInfo, tilingInfo.shapeInfo,
        tilingInfo.convOpsConstParams, tilingInfo.numBlocksRes, tilingData);

    // 计算需要的AI Core数量
    uint32_t g_numBlocks = tilingData.conv3dRunInfo.batchDim * tilingData.conv3dRunInfo.doDim *
                        tilingData.conv3dRunInfo.hoDim * tilingData.conv3dRunInfo.nDim;
    
    // 获取tiling键（用于选择最优的kernel实现）
    optiling::conv_ops_tiling::ConvTilingKeyPara tilingKeyPara {};
    optiling::conv_ops_tiling::Conv3dV2BaseTilingKey tilingKey {tilingData, tilingInfo.flagInfo,
        tilingInfo.descInfo, tilingInfo.shapeInfo, tilingInfo.numBlocksRes, tilingInfo.convOpsConstParams};
    tilingKey.GetTemplateTilingKey(tilingKeyPara);

    // 调用Conv3dv2Template函数执行实际的3D卷积计算
    Conv3dv2Template(input_ptr, weight_ptr, bias_ptr, nullptr, nullptr, nullptr,
        output_ptr, nullptr, tilingData,
        static_cast<int8_t>(tilingKeyPara.fmpTiling), static_cast<int8_t>(tilingKeyPara.weightTiling),
        static_cast<int8_t>(tilingKeyPara.l1PingPong), static_cast<int8_t>(tilingKeyPara.l0PingPong),
        static_cast<int8_t>(tilingKeyPara.outputOrder), static_cast<int8_t>(tilingKeyPara.iterOrder),
        inDtype, g_numBlocks, stream);
}


/*!
 * \brief 根据输入参数计算并创建输出张量
 * 
 * 该函数根据输入张量形状、权重形状、步长、padding和膨胀参数，
 * 计算输出张量的形状并创建一个全零的输出张量。
 * 
 * 输出形状计算公式（以高度H为例）：
 * Ho = (H + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1
 * 
 * @param input 输入张量
 * @param oriInputShape 输入张量的原始形状 [N, C, D, H, W]
 * @param oriWeightShape 权重张量的原始形状 [Co, Ci, Kd, Kh, Kw]
 * @param strides 步长 [strideD, strideH, strideW]
 * @param pads padding [padD, padH, padW] (每个方向2面，共6面)
 * @param dilations 膨胀 [dilationD, dilationH, dilationW]
 * @return 输出张量
 */
torch::Tensor CreateOutputTensor(
    const torch::Tensor& input, const c10::IntArrayRef& oriInputShape, const c10::IntArrayRef& oriWeightShape,
    const c10::IntArrayRef& strides, const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations)
{
    // 提取输入张量的各个维度
    int64_t N = oriInputShape[FORMAT_NCDHW_N_INDEX];  // 批次大小
    int64_t D = oriInputShape[FORMAT_NCDHW_D_INDEX];  // 输入深度
    int64_t H = oriInputShape[FORMAT_NCDHW_H_INDEX];  // 输入高度
    int64_t W = oriInputShape[FORMAT_NCDHW_W_INDEX];  // 输入宽度
    int64_t Co = oriWeightShape[FORMAT_NCDHW_N_INDEX]; // 输出通道数
    vector<int64_t> kernelSize = {
        oriWeightShape[FORMAT_NCDHW_D_INDEX], oriWeightShape[FORMAT_NCDHW_H_INDEX], oriWeightShape[FORMAT_NCDHW_W_INDEX]};

    // 计算输出张量的各个维度
    // 公式：(in_size + 2*pad - dilation*(kernel_size-1) - 1) / stride + 1
    int64_t Do = (D + 2 * pads[ATTRS_D_DIM_IDX_NCDHW] -
                  dilations[ATTRS_D_DIM_IDX_NCDHW] * (kernelSize[ATTRS_D_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_D_DIM_IDX_NCDHW] +
                  1;
    int64_t Ho = (H + 2 * pads[ATTRS_H_DIM_IDX_NCDHW] -
                  dilations[ATTRS_H_DIM_IDX_NCDHW] * (kernelSize[ATTRS_H_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_H_DIM_IDX_NCDHW] +
                  1;
    int64_t Wo = (W + 2 * pads[ATTRS_W_DIM_IDX_NCDHW] -
                  dilations[ATTRS_W_DIM_IDX_NCDHW] * (kernelSize[ATTRS_W_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_W_DIM_IDX_NCDHW] +
                  1;

    // 验证输出形状的合法性
    TORCH_CHECK(N > 0, "Output of N has to be positive, but got ", N);
    TORCH_CHECK(Co > 0, "Output of C has to be positive, but got ", Co);
    TORCH_CHECK(Do > 0, "Output of D has to be positive, but got ", Do);
    TORCH_CHECK(Ho > 0, "Output of H has to be positive, but got ", Ho);
    TORCH_CHECK(Wo > 0, "Output of W has to be positive, but got ", Wo);
    
    // 打印输出形状信息
    printf("Output shape: N=%d, Co=%d, Do=%d, Ho=%d, Wo=%d\n",
       static_cast<int>(N),
       static_cast<int>(Co),
       static_cast<int>(Do),
       static_cast<int>(Ho),
       static_cast<int>(Wo));
    
    // 创建并返回全零输出张量
    return torch::zeros({N, Co, Do, Ho, Wo}, input.options());
}

/*!
 * \brief 3D卷卷积NPU实现函数
 * 
 * 该函数是3D卷积操作在Ascend NPU上的主入口点，负责：
 * 1. 参数验证：检查输入、权重、bias是否在NPU设备上
 * 2. 形状验证：检查输入输出的维度是否正确
 * 3. 创建输出张量：根据输入参数计算输出形状
 * 4. 转换参数：将PyTorch参数转换为内部格式
 * 5. 数据类型分发：根据输入数据类型选择对应的模板实现
 * 6. 执行kernel：通过OpCommand调用Conv3dV2CustomApi
 * 
 * @param input 输入张量 [N, C, D, H, W]
 * @param weight 权重张量 [Co, Ci, Kd, Kh, Kw]
 * @param strides 步长 [strideD, strideH, strideW]
 * @param pads padding [padD, padH, padW]
 * @param dilations 膨胀 [dilationD, dilationH, dilationW]
 * @param oriInputShape 输入张量的原始形状
 * @param oriWeightShape 权重张量的原始形状
 * @param bias 偏置张量（可选）
 * @return 输出张量 [N, Co, Do, Ho, Wo]
 */
torch::Tensor Conv3dV2CustomNpu(
    const torch::Tensor& input, const torch::Tensor& weight, const c10::IntArrayRef& strides,
    const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations, const c10::IntArrayRef& oriInputShape,
    const c10::IntArrayRef& oriWeightShape, const c10::optional<torch::Tensor>& bias)
{
    // OptionalDeviceGuard确保后续操作在正确的设备上下文执行
    // 它会记录当前设备状态，执行完作用域代码后自动恢复
    const c10::OptionalDeviceGuard guard(input.device());
    
    // 验证输入张量在NPU设备上
    TORCH_CHECK(torch_npu::utils::is_npu(input), "Input tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(weight), "Weight tensor must be on NPU device");
    if (bias.has_value()) {
        TORCH_CHECK(torch_npu::utils::is_npu(bias.value()), "Bias tensor must be on NPU device");
    }
    
    // 验证张量维度和参数大小
    TORCH_CHECK(input.dim() == NCDHW_DIM, "Input must be 5D (N, C, D, H, W)");
    TORCH_CHECK(weight.dim() == NCDHW_DIM, "Weight must be 5D (N, C, D, H, W)");
    TORCH_CHECK(strides.size() == CONV_ATTRS_DIM, "Stride must have 3 elements");
    TORCH_CHECK(pads.size() == CONV_ATTRS_DIM, "Padding must have 3 elements");
    TORCH_CHECK(dilations.size() == CONV_ATTRS_DIM, "Dilation must have 3 elements");
    TORCH_CHECK(oriInputShape.size() == NCDHW_DIM, "Origin Input must have 5 elements (N, C, D, H, W)");
    TORCH_CHECK(oriWeightShape.size() == NCDHW_DIM, "Origin Weight must have 5 elements (N, C, D, H, W)");

    // 创建输出张量
    auto output = CreateOutputTensor(input, oriInputShape, oriWeightShape, strides, pads, dilations);
    
    // 获取当前NPU流
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // 转换参数为内部格式
    vector<int64_t> oriInputShapeList(oriInputShape.begin(), oriInputShape.end());
    vector<int64_t> oriWeightShapeList(oriWeightShape.begin(), oriWeightShape.end());
    vector<int64_t> strideList(strides.begin(), strides.end());
    vector<int64_t> paddingList = {pads[ATTRS_D_DIM_IDX_NCDHW], pads[ATTRS_D_DIM_IDX_NCDHW],
                                   pads[ATTRS_H_DIM_IDX_NCDHW], pads[ATTRS_H_DIM_IDX_NCDHW],
                                   pads[ATTRS_W_DIM_IDX_NCDHW], pads[ATTRS_W_DIM_IDX_NCDHW]};
    vector<int64_t> dilationList(dilations.begin(), dilations.end());
    
    // 定义ACL调用的lambda函数，在NPU上执行
    // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏根据数据，型分发到对应的模板实现
    auto acl_call = [=]() -> int {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "Conv3dV2CustomNpu", [&] {
            Conv3dV2CustomApi<scalar_t>(
                stream, input, weight, output, strideList, paddingList, dilationList, oriInputShapeList,
                oriWeightShapeList, bias);
        });
        return 0;
    };
    
    // 通过OpCommand执行API调用
    at_npu::native::OpCommand::RunOpApiV2("Conv3dv2", acl_call);
    
    return output;
}

// 注册算子接口定义，声明函数签名
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "conv3d_v2_custom(Tensor input, Tensor weight, int[3] stride, int[3] padding, int[3] dilation, int[5] "
        "oriInputShape, int[5] oriWeightShape, Tensor? bias) -> Tensor");
}

// 注册NPU实现，将conv3d_v2_custom算子映射到Conv3dV2CustomNpu函数
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("conv3d_v2_custom", Conv3dV2CustomNpu);
}

}
}