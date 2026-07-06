/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_grad_tiling.cpp
 * \brief SoftmaxGrad算子的tiling(分块)策略实现，按行多核切分
 *
 * Softmax反向传播公式（per sample i）:
 *   dot_i   = sum_j(dy[i,j] * y[i,j])      -- 行内点积标量
 *   dx[i,j] = y[i,j] * (dy[i,j] - dot_i)   -- 逐元素计算
 *
 * 输入:  y  [N, C], dy [N, C]
 * 输出:  dx [N, C]
 *
 * 多核策略: 按行切分，usedCoreNum = min(N, maxCoreNum)，尾核多一行
 * BUFFER_NUM = 2（流水）
 * 使用 DataCopyPad 处理 C 非32B对齐场景（CPadded = align_up(C, 32/sizeof(T))）
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/softmax_grad_tiling_data.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t BUFFER_NUM = 2;

/*
 * UB_NUM_FLOAT = 3 TQue + 2 TBuf = 5
 * UB_NUM_OTHER = 3 TQue + 5 float TBuf = 8
 * Reserve one extra block in both paths for DataCopyPad alignment.
 */
#define UB_NUM_FLOAT 6U
#define UB_NUM_OTHER 9U

struct SoftmaxGradCompileInfo {
};

static int64_t GetTypeSize(ge::DataType dtype) {
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

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

ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t& N, uint64_t& C, ge::DataType& dataType)
{
    auto inputY = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = Ops::NN::OpTiling::EnsureNotScalar(inputY->GetStorageShape());

    auto inputDy = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDy);
    (void)Ops::NN::OpTiling::EnsureNotScalar(inputDy->GetStorageShape());

    auto outDx = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDx);
    (void)Ops::NN::OpTiling::EnsureNotScalar(outDx->GetStorageShape());

    const auto& yShape = inputShapeY;
    if (yShape.GetDimNum() < 2) {
        OP_LOGE(context, "y shape dim num < 2");
        return ge::GRAPH_FAILED;
    }
    N = static_cast<uint64_t>(yShape.GetDim(0));
    C = static_cast<uint64_t>(yShape.GetDim(1));

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftmaxGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    uint64_t N;
    uint64_t C;
    ge::DataType dataType;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, N, C, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SoftmaxGradTilingData* tiling = context->GetTilingData<SoftmaxGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftmaxGradTilingData), 0, sizeof(SoftmaxGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    if (N == 0 || C == 0) {
        OP_LOGE(context, "N or C is 0");
        return ge::GRAPH_FAILED;
    }

    // 数据类型大小
    int64_t typeSize = GetTypeSize(dataType);

    // 使用 GetUbBlockSize 获取对齐单位，避免硬编码
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    if (ubBlockSize == 0) {
        return ge::GRAPH_FAILED;
    }

    // CPadded: C 向上对齐到 ubBlockSize / typeSize 的整数倍（32/sizeof(T)）
    int64_t alignElems = static_cast<int64_t>(static_cast<uint64_t>(ubBlockSize) / static_cast<uint64_t>(typeSize));
    uint64_t CPadded = static_cast<uint64_t>(((C + alignElems - 1) / alignElems) * alignElems);

    // 验证 CPadded 行数据能放入 UB
    uint64_t ubDataNumber = (dataType == ge::DT_FLOAT) ? UB_NUM_FLOAT : UB_NUM_OTHER;
    uint64_t tileBlockNum = (ubSize / static_cast<uint64_t>(ubBlockSize)) / ubDataNumber;
    int64_t ubPartDataNum = static_cast<int64_t>(
        (tileBlockNum * static_cast<uint64_t>(ubBlockSize)) / static_cast<uint64_t>(typeSize));
    if (ubPartDataNum <= 0) {
        OP_LOGE(context, "ubPartDataNum <= 0");
        return ge::GRAPH_FAILED;
    }
    if (static_cast<int64_t>(CPadded) > ubPartDataNum) {
        OP_LOGE(context, "CPadded(%lu) > ubPartDataNum(%ld), row does not fit in UB", CPadded, ubPartDataNum);
        return ge::GRAPH_FAILED;
    }

    // 多核按行切分：前 tailCoreRows 个核各多处理 1 行
    uint64_t usedCoreNum = std::min(N, static_cast<uint64_t>(coreNum));
    if (usedCoreNum == 0) {
        OP_LOGE(context, "usedCoreNum is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t rowsPerCore = N / usedCoreNum;
    uint64_t tailCoreRows = N % usedCoreNum;

    tiling->N = N;
    tiling->C = C;
    tiling->CPadded = CPadded;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailCoreRows = tailCoreRows;
    tiling->usedCoreNum = usedCoreNum;

    context->SetBlockDim(static_cast<int32_t>(usedCoreNum));
    context->SetTilingKey(0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftmaxGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SoftmaxGrad).Tiling(SoftmaxGradTilingFunc).TilingParse<SoftmaxGradCompileInfo>(TilingParseForSoftmaxGrad);
} // namespace optiling
