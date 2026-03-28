/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/max_pool_with_argmax_v3_tiling_data.h"
#include "../op_kernel/max_pool_with_argmax_v3_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;

// 获取平台信息
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static inline int64_t AlignUp(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}

// Tiling 分发入口
static ge::graphStatus MaxPoolWithArgmaxV3TilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取 workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 3. 获取输入 shape (TilingContext 返回 StorageShape*)
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    auto inputShape = inputShapePtr->GetStorageShape();
    OP_CHECK_IF(inputShape.GetDimNum() != 4,
        OP_LOGE(context, "input must be 4D"),
        return ge::GRAPH_FAILED);

    int64_t N = inputShape.GetDim(0);
    int64_t C = inputShape.GetDim(1);
    int64_t H = inputShape.GetDim(2);
    int64_t W = inputShape.GetDim(3);

    // 4. 获取输出 shape (已由 InferShape 计算)
    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    int64_t Hout = yShape.GetDim(2);
    int64_t Wout = yShape.GetDim(3);

    // 5. 获取属性参数 (通过 GetAttrs() 获取 RuntimeAttrs)
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 属性顺序: 0=kernel_size, 1=strides, 2=pads, 3=dilations, 4=ceil_mode
    const auto* kernelSizeList = attrs->GetListInt(0);
    const auto* stridesList = attrs->GetListInt(1);
    const auto* padsList = attrs->GetListInt(2);
    const auto* dilationsList = attrs->GetListInt(3);

    OP_CHECK_NULL_WITH_CONTEXT(context, kernelSizeList);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesList);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsList);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilationsList);

    const int64_t* ksData = kernelSizeList->GetData();
    const int64_t* stData = stridesList->GetData();
    const int64_t* pdData = padsList->GetData();
    const int64_t* dlData = dilationsList->GetData();

    int32_t kH = static_cast<int32_t>((kernelSizeList->GetSize() >= 1) ? ksData[0] : 1);
    int32_t kW = static_cast<int32_t>((kernelSizeList->GetSize() >= 2) ? ksData[1] : kH);
    int32_t sH = static_cast<int32_t>((stridesList->GetSize() >= 1) ? stData[0] : 1);
    int32_t sW = static_cast<int32_t>((stridesList->GetSize() >= 2) ? stData[1] : sH);
    int32_t padH = static_cast<int32_t>((padsList->GetSize() >= 1) ? pdData[0] : 0);
    int32_t padW = static_cast<int32_t>((padsList->GetSize() >= 2) ? pdData[1] : padH);
    int32_t dH = static_cast<int32_t>((dilationsList->GetSize() >= 1) ? dlData[0] : 1);
    int32_t dW = static_cast<int32_t>((dilationsList->GetSize() >= 2) ? dlData[1] : dH);

    // 6. 获取 dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    int64_t dtypeSize = 2; // default float16
    if (dataType == ge::DT_FLOAT) {
        dtypeSize = 4;
    } else if (dataType == ge::DT_BF16) {
        dtypeSize = 2;
    }

    // 7. 计算对齐参数
    int64_t inputWidthAligned = AlignUp(W * dtypeSize, 32) / dtypeSize;
    int64_t outputWidthAligned = AlignUp(Wout * dtypeSize, 32) / dtypeSize;

    // 8. 多核切分
    int64_t totalSlices = N * C;

    // 空 tensor 快速返回（N=0 或 C=0 或 Hout=0 或 Wout=0）
    if (totalSlices == 0 || Hout <= 0 || Wout <= 0) {
        MaxPoolWithArgmaxV3TilingData* tiling = context->GetTilingData<MaxPoolWithArgmaxV3TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(
            memset_s(tiling, sizeof(MaxPoolWithArgmaxV3TilingData), 0, sizeof(MaxPoolWithArgmaxV3TilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"),
            return ge::GRAPH_FAILED);
        tiling->batchSize = N;
        tiling->channels = C;
        tiling->inputHeight = H;
        tiling->inputWidth = W;
        tiling->outputHeight = Hout;
        tiling->outputWidth = Wout;
        tiling->kernelH = kH;
        tiling->kernelW = kW;
        tiling->strideH = sH;
        tiling->strideW = sW;
        tiling->padH = padH;
        tiling->padW = padW;
        tiling->dilationH = dH;
        tiling->dilationW = dW;
        tiling->totalSlices = 0;
        tiling->slicesPerCore = 0;
        // Set safe minimum values for buffer allocation in kernel Init
        tiling->outputRowsPerTile = 1;
        tiling->inputWidthAligned = (inputWidthAligned > 0) ? inputWidthAligned : (32 / dtypeSize);
        tiling->outputWidthAligned = (outputWidthAligned > 0) ? outputWidthAligned : (32 / dtypeSize);
        context->SetBlockDim(1);
        if (dataType == ge::DT_FLOAT16) {
            context->SetTilingKey(GET_TPL_TILING_KEY(MAXPOOL_TPL_SCH_MODE_FP16));
        } else if (dataType == ge::DT_FLOAT) {
            context->SetTilingKey(GET_TPL_TILING_KEY(MAXPOOL_TPL_SCH_MODE_FP32));
        } else if (dataType == ge::DT_BF16) {
            context->SetTilingKey(GET_TPL_TILING_KEY(MAXPOOL_TPL_SCH_MODE_BF16));
        } else {
            OP_LOGE(context, "unsupported dtype on arch32");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    int64_t slicesPerCore = CeilDiv(totalSlices, coreNum);
    int64_t usedCoreNum = CeilDiv(totalSlices, slicesPerCore);

    // 9. UB 切分（迭代二：动态计算 outputRowsPerTile，double buffer）
    constexpr int64_t BUFFER_NUM = 2;
    constexpr int64_t UB_RESERVE = 4096; // 安全余量

    int64_t sizePerInputRow = AlignUp(W * dtypeSize, 32);
    int64_t sizePerYRow = AlignUp(Wout * dtypeSize, 32);
    int64_t argmaxWidthAligned = AlignUp(Wout * 8, 32) / 8;
    int64_t sizePerArgmaxRow = argmaxWidthAligned * 8;

    // 第一行输出所需的输入行数: dilationH * (kH - 1) + 1
    int64_t baseInputRows = static_cast<int64_t>(dH) * (kH - 1) + 1;

    // 第一行输出的总 UB 开销 (单 buffer)
    int64_t ubFirstRow = baseInputRows * sizePerInputRow + sizePerYRow + sizePerArgmaxRow;

    // 每增加一行输出的额外 UB 开销 (单 buffer): strideH 行新输入 + 一行 y + 一行 argmax
    int64_t ubPerAdditionalRow = static_cast<int64_t>(sH) * sizePerInputRow + sizePerYRow + sizePerArgmaxRow;

    // double buffer 需要 BUFFER_NUM 倍空间
    int64_t ubAvailable = static_cast<int64_t>(ubSize) - UB_RESERVE;

    int64_t outputRowsPerTile;
    if (ubAvailable < BUFFER_NUM * ubFirstRow) {
        // UB 不足以容纳 double buffer + 1行，降级为单 buffer
        // 此处仍使用 outputRowsPerTile=1，BUFFER_NUM 在 kernel 侧固定为 2
        // 但 tiling 需要保证 2 * singleRowCost <= ubAvailable
        // 若连单 buffer 单行都放不下，则 outputRowsPerTile = 1 作为兜底
        outputRowsPerTile = 1;
    } else {
        outputRowsPerTile = 1 + (ubAvailable - BUFFER_NUM * ubFirstRow) / (BUFFER_NUM * ubPerAdditionalRow);
        if (outputRowsPerTile < 1) {
            outputRowsPerTile = 1;
        }
        if (outputRowsPerTile > Hout) {
            outputRowsPerTile = Hout;
        }
    }

    // 10. 设置 TilingData
    MaxPoolWithArgmaxV3TilingData* tiling = context->GetTilingData<MaxPoolWithArgmaxV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(MaxPoolWithArgmaxV3TilingData), 0, sizeof(MaxPoolWithArgmaxV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->batchSize = N;
    tiling->channels = C;
    tiling->inputHeight = H;
    tiling->inputWidth = W;
    tiling->outputHeight = Hout;
    tiling->outputWidth = Wout;

    tiling->kernelH = kH;
    tiling->kernelW = kW;
    tiling->strideH = sH;
    tiling->strideW = sW;
    tiling->padH = padH;
    tiling->padW = padW;
    tiling->dilationH = dH;
    tiling->dilationW = dW;

    tiling->totalSlices = totalSlices;
    tiling->slicesPerCore = slicesPerCore;

    tiling->outputRowsPerTile = outputRowsPerTile;
    tiling->inputWidthAligned = inputWidthAligned;
    tiling->outputWidthAligned = outputWidthAligned;

    // 11. 设置 BlockDim
    context->SetBlockDim(usedCoreNum);

    // 12. 设置 TilingKey
    if (dataType == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(MAXPOOL_TPL_SCH_MODE_FP16));
    } else if (dataType == ge::DT_FLOAT) {
        context->SetTilingKey(GET_TPL_TILING_KEY(MAXPOOL_TPL_SCH_MODE_FP32));
    } else if (dataType == ge::DT_BF16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(MAXPOOL_TPL_SCH_MODE_BF16));
    } else {
        OP_LOGE(context, "unsupported dtype on arch32");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaxPoolWithArgmaxV3(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MaxPoolWithArgmaxV3CompileInfo {};

IMPL_OP_OPTILING(MaxPoolWithArgmaxV3)
    .Tiling(MaxPoolWithArgmaxV3TilingFunc)
    .TilingParse<MaxPoolWithArgmaxV3CompileInfo>(TilingParseForMaxPoolWithArgmaxV3);

} // namespace optiling
