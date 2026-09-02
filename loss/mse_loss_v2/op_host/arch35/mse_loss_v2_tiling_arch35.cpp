/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mse_loss_v2_tiling_arch35.cpp
 * \brief MSELossV2 tiling implementation for arch35 (Ascend950)
 *
 * Elementwise multi-core (blockFactor) + UB (ubFactor) split; single/double buffer by data
 * volume threshold. reduction (none/sum/mean) is carried as a runtime tiling field; for sum/mean
 * a per-core fp32 partial workspace is reserved and batch schedule mode is set for SyncAll.
 */

#include "mse_loss_v2_tiling_arch35.h"
#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/mse_loss_v2_tiling_data.h"
#include "../../op_kernel/arch35/mse_loss_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

namespace {
constexpr int64_t COMPUTE_TYPE_SIZE = 4; // fp32 compute/reduction
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t COMPARE_ALIGN_ELEMENTS = 256 / COMPUTE_TYPE_SIZE;
// UB split counts (fp32-element units). Counted from the kernel's resident buffers with margin
// for partialBuf + alignment (worst case fp32 DB=8 / fp16 DB=7; SB max 6). See 02 design
// §4. Over-estimating shrinks ubFactor (safe); under-estimating overflows UB (VEC_ERROR).
constexpr int64_t BUFFER_NUM_DB = 10;
constexpr int64_t BUFFER_NUM_SB = 7;
constexpr int64_t WS_CORE_STRIDE = 8; // 32B(=8 fp32) per-core partial slot
constexpr int64_t MERGE_VL_FP32 = 64; // 跨核合并的矢量车道数(fp32)
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t ATTR_REDUCTION_IDX = 0;
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_SUM = 1;
constexpr uint32_t REDUCTION_MEAN = 2;

const gert::Shape g_vec_1_shape = {1};

inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
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

// input/target/output shapes must all be identical, dim num <= 8. Returns the input shape.
ge::graphStatus CheckShapeInfo(gert::TilingContext* context, gert::Shape& shapeInput)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    shapeInput = EnsureNotScalar(inputShape->GetStorageShape());

    auto targetShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    auto shapeTarget = EnsureNotScalar(targetShape->GetStorageShape());
    OP_CHECK_IF(shapeInput != shapeTarget,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "input and target",
                    (Ops::Base::ToString(shapeInput) + " and " + Ops::Base::ToString(shapeTarget)).c_str(),
                    "The shape of target must be the same as the shape of input"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(shapeInput.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input",
                                                         std::to_string(shapeInput.GetDimNum()).c_str(),
                                                         "The dim num of input must be less than or equal to 8"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// input dtype must be float32/float16/bfloat16, and target must match input.
ge::graphStatus CheckDtypeInfo(gert::TilingContext* context, ge::DataType& dataType)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "input", Ops::Base::ToString(dataType).c_str(),
                                  "float32, float16, bfloat16");
        return ge::GRAPH_FAILED;
    }

    auto targetDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    OP_CHECK_IF(targetDesc->GetDataType() != dataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "input, target",
                    (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(targetDesc->GetDataType())).c_str(),
                    "The dtype of target must be the same as the dtype of input"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetReduction(gert::TilingContext* context, uint32_t& reduction)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* reductionStr = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionStr);
    if (strcmp(reductionStr, "none") == 0) {
        reduction = REDUCTION_NONE;
    } else if (strcmp(reductionStr, "sum") == 0) {
        reduction = REDUCTION_SUM;
    } else if (strcmp(reductionStr, "mean") == 0) {
        reduction = REDUCTION_MEAN;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "reduction", reductionStr,
                                              "The value of reduction must be none, sum or mean");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
// 填 tiling 数据 + workspace/blockDim/调度模式。入参(ubSize/coreNum/totalIdx/reduction)均已在入口校验。
static ge::graphStatus FillTilingData4MSELossV2Arch35(gert::TilingContext* context, uint64_t ubSize, int64_t coreNum,
                                                      int64_t totalIdx, uint32_t reduction, bool& useDoubleBuffer)
{
    MSELossV2Arch35TilingData* tiling = context->GetTilingData<MSELossV2Arch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MSELossV2Arch35TilingData), 0, sizeof(MSELossV2Arch35TilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    int64_t ubBlockSize = GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD);
    int64_t bufferNum = useDoubleBuffer ? BUFFER_NUM_DB : BUFFER_NUM_SB;
    int64_t alignUnit = std::max(ubBlockSize, COMPARE_ALIGN_ELEMENTS);
    tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / COMPUTE_TYPE_SIZE, bufferNum), alignUnit);
    tiling->meanCof = 1.0f / static_cast<float>(totalIdx);
    tiling->reduction = reduction;
    // 跨核合并的 UB 用量由 host 按真实核数算(kernel 不再假设平台最大核数):
    // 矢量合并整轮读 MERGE_VL_FP32 车道, 故按整轮向上取整。
    tiling->partialUbElems = (reduction == REDUCTION_NONE) ?
                                 0U :
                                 static_cast<uint32_t>(CeilAlign(usedCoreNum * WS_CORE_STRIDE, MERGE_VL_FP32));

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (reduction != REDUCTION_NONE) {
        // one fp32 partial per core, each in its own 32B(=8 fp32) block to avoid sub-block
        // multi-core GM write races; + system reserved workspace queried from the platform.
        currentWorkspace[0] = static_cast<size_t>(usedCoreNum) * WS_CORE_STRIDE * sizeof(float) + sysWorkspaceSize;
    } else {
        currentWorkspace[0] = sysWorkspaceSize;
    }

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    // sum/mean cross-core reduction uses SyncAll -> all launched cores must be co-resident -> batch mode.
    if (reduction != REDUCTION_NONE) {
        context->SetScheduleMode(1);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus Tiling4MSELossV2Arch35(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    gert::Shape shapeInput;
    OP_CHECK_IF(CheckShapeInfo(context, shapeInput) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckShapeInfo error"),
                return ge::GRAPH_FAILED);
    int64_t totalIdx = shapeInput.GetShapeSize();
    // Empty tensor is not supported (aligns with ascend910b mse_loss_v2 tiling, which rejects totalLength==0,
    // and with sibling losses). Empty is short-circuited earlier by aclnn L2.
    OP_CHECK_IF(totalIdx == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "input", "0",
                                                          "empty tensor is not supported"),
                return ge::GRAPH_FAILED);

    ge::DataType dataType;
    OP_CHECK_IF(CheckDtypeInfo(context, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckDtypeInfo error"),
                return ge::GRAPH_FAILED);

    uint32_t reduction = REDUCTION_MEAN;
    OP_CHECK_IF(GetReduction(context, reduction) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetReduction error"),
                return ge::GRAPH_FAILED);

    bool useDoubleBuffer = false;
    OP_CHECK_IF(FillTilingData4MSELossV2Arch35(context, ubSize, coreNum, totalIdx, reduction, useDoubleBuffer) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "FillTilingData4MSELossV2Arch35 error"), return ge::GRAPH_FAILED);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer ? 1ULL : 0ULL);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4MSELossV2Arch35(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<MSELossV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MSELossV2)
    .Tiling(Tiling4MSELossV2Arch35)
    .TilingParse<MSELossV2CompileInfo>(TilingParse4MSELossV2Arch35);

} // namespace optiling
