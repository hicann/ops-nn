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
/*!
 * \file apply_adam_d_tiling.cpp
 * \brief ApplyAdamD tiling implementation (ascend910b / DAV_2201)
 *
 * 单一 tiling 策略：元素总数多核切分 + UB 切分 + 双缓冲。
 * TilingKey 选择 (D_T x USE_NESTEROV)，6 组合 = 3 dtype x 2 useNesterov：
 *   D_T     ∈ {C_DT_FLOAT16, C_DT_BF16, C_DT_FLOAT}
 *   USE_NESTEROV ∈ {0, 1}  (读 use_nesterov attr，attr 顺序 use_locking=0/use_nesterov=1)
 * 6 个标量参数为运行期 GM 张量输入，不进 TilingData，kernel 侧 GM 读取。
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_adam_d_tiling_data.h"
#include "../op_kernel/apply_adam_d_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

// System-reserved workspace (this op uses no extra workspace, set 0).
constexpr uint32_t WS_SYS_SIZE = 0U;

// Min-work-per-core lower bound (elements). Small shapes must NOT be split across all AIV
// cores: per-core launch/scalar-load/sync overhead then dominates over the trivial compute,
// making the op slower than a few-core dispatch. The deployed builtin ApplyAdamD uses ~2-4
// cores for n=4096 (block_dim observed 2/4); we cap usedCoreNum so each core processes at
// least this many elements. Large/medium shapes are unaffected (they clamp to full coreNum).
// Value 2048 -> n=4096 dispatches to 2 cores (matches builtin's low end); on a 40-AIV part
// only shapes < 40*2048 ≈ 82K elements are re-tiled, so measured medium(262144)/large
// (4194304) keep full-core parallelism. Does NOT change the 6 TilingKey semantics/precision.
constexpr int64_t MIN_ELEM_PER_CORE = 2048;

// Input indices (consistent with op_def registration order):
//   0:var 1:m 2:v | 3:beta1_power 4:beta2_power 5:lr 6:beta1 7:beta2 8:epsilon (scalars) | 9:grad
static constexpr size_t IDX_VAR = 0;
static constexpr size_t IDX_M = 1;
static constexpr size_t IDX_V = 2;
static constexpr size_t IDX_BETA1_POWER = 3; // first scalar input
static constexpr size_t IDX_EPSILON = 8;     // last scalar input
static constexpr size_t IDX_GRAD = 9;
static constexpr size_t INPUT_NUM = 10;

// Attr indices (matching op_def Attr registration order)
static constexpr size_t ATTR_USE_LOCKING = 0;
static constexpr size_t ATTR_USE_NESTEROV = 1;

// Exact same-shape comparison (rank + each dim). Used to enforce var/m/v/grad identical shape.
static inline bool ShapeEqual(const gert::Shape& a, const gert::Shape& b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

// Lightweight input contract validation (spec.yaml error_codes / DESIGN §3.3 §8).
// Enforces, before tiling computes GM offsets, the contract that kernel Init assumes
// (m/v/grad use var's totalNum for SetGlobalBuffer); a violated contract would otherwise
// cause an out-of-bounds GM read in the kernel. Read-only checks: legal inputs are
// untouched (zero behavior change), only contract violations fail tiling.
//   1) m/v/grad shape == var shape exactly                       -> shape_mismatch
//   2) 6 scalars (beta1_power..epsilon) GetShapeSize() == 1      -> shape_mismatch
//   3) all 10 inputs share var's dtype (var dtype set is checked
//      separately in GetShapeAttrsInfo against {fp32,fp16,bf16}) -> dtype_not_supported
static ge::graphStatus CheckInputContract(gert::TilingContext* context)
{
    auto varDesc = context->GetInputDesc(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    const ge::DataType varDtype = varDesc->GetDataType();

    auto varShapePtr = context->GetInputShape(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    const gert::Shape& varShape = varShapePtr->GetStorageShape();

    // (1) tensor inputs m/v/grad must be exactly the same shape as var.
    const size_t tensorIdx[] = {IDX_M, IDX_V, IDX_GRAD};
    for (size_t idx : tensorIdx) {
        auto shapePtr = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        OP_CHECK_IF(!ShapeEqual(shapePtr->GetStorageShape(), varShape),
                    OP_LOGE(context, "ApplyAdamD: input[%zu] shape must match var (shape_mismatch)", idx),
                    return ge::GRAPH_FAILED);
    }

    // (2) scalar inputs beta1_power..epsilon must have GetShapeSize()==1 (rank-0 [] or [1]).
    for (size_t idx = IDX_BETA1_POWER; idx <= IDX_EPSILON; ++idx) {
        auto shapePtr = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        OP_CHECK_IF(shapePtr->GetStorageShape().GetShapeSize() != 1,
                    OP_LOGE(context, "ApplyAdamD: scalar input[%zu] must have shape size 1 (shape_mismatch)", idx),
                    return ge::GRAPH_FAILED);
    }

    // (3) all 10 inputs must share var's dtype.
    for (size_t idx = 0; idx < INPUT_NUM; ++idx) {
        auto desc = context->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != varDtype,
                    OP_LOGE(context, "ApplyAdamD: input[%zu] dtype must match var (dtype_not_supported)", idx),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static const gert::Shape K_VEC_1_SHAPE = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return K_VEC_1_SHAPE;
    }
    return inShape;
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

// Get var element count + dtype, validate dtype in {FP32, FP16, BF16}.
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto varDesc = context->GetInputDesc(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    dataType = varDesc->GetDataType();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(dataType) == 0,
                OP_LOGE(context, "ApplyAdamD: unsupported dtype %d", static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);

    auto varShapePtr = context->GetInputShape(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    auto varShape = EnsureNotScalar(varShapePtr->GetStorageShape());
    totalNum = varShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitTilingData(gert::TilingContext* context, ApplyAdamDTilingData*& tiling)
{
    tiling = context->GetTilingData<ApplyAdamDTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ApplyAdamDTilingData), 0, sizeof(ApplyAdamDTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Multi-core split: blockFactor (32B-aligned) + usedCoreNum. Empty tensor -> usedCoreNum=1, blockFactor=0.
static ge::graphStatus ComputeMultiCoreSplit(gert::TilingContext* context, ge::DataType dataType, int64_t totalNum,
                                             int64_t coreNum, ApplyAdamDTilingData* tiling, int64_t& alignElem,
                                             int64_t& usedCoreNum)
{
    constexpr int64_t TYPE_SIZE_FP32 = 4;
    constexpr int64_t TYPE_SIZE_FP16 = 2;
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context); // 32B
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? TYPE_SIZE_FP32 : TYPE_SIZE_FP16;
    alignElem = ubBlockSize / typeSize; // fp32: 8, fp16/bf16: 16
    tiling->totalNum = totalNum;

    if (totalNum <= 0) {
        // Empty tensor: single core, zero block, kernel Process early-returns.
        tiling->blockFactor = 0;
        usedCoreNum = 1;
        return ge::GRAPH_SUCCESS;
    }
    if (totalNum < alignElem) {
        tiling->blockFactor = totalNum;
        usedCoreNum = 1;
    } else {
        // Cap core count by the min-work-per-core lower bound (avoids small-shape over-split),
        // then clamp to the physical AIV core count. blockFactor is 32B/dtype aligned and the
        // final usedCoreNum is recomputed from it so the split fully covers totalNum with no
        // empty trailing core.
        int64_t wantCore = Ops::Base::CeilDiv(totalNum, MIN_ELEM_PER_CORE);
        if (wantCore < 1) {
            wantCore = 1;
        }
        if (wantCore > coreNum) {
            wantCore = coreNum;
        }
        int64_t perCoreRaw = Ops::Base::CeilDiv(totalNum, wantCore);
        tiling->blockFactor = Ops::Base::CeilAlign(perCoreRaw, alignElem);
        usedCoreNum = Ops::Base::CeilDiv(totalNum, tiling->blockFactor);
    }
    OP_CHECK_IF(usedCoreNum == 0, OP_LOGE(context, "usedCoreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// UB split: ubFactor. bytePerElem = 64 (fp32) / 52 (fp16/bf16) (DESIGN §3.7).
static ge::graphStatus ComputeUbSplit(gert::TilingContext* context, ge::DataType dataType, uint64_t ubSize,
                                      int64_t alignElem, ApplyAdamDTilingData* tiling)
{
    constexpr int64_t RESERVED_UB = 48 * 1024; // 48 KB reserved (sys + scalar buf + sync)
    constexpr int64_t BYTE_PER_ELEM_FP32 = 64;
    constexpr int64_t BYTE_PER_ELEM_FP16 = 52;
    int64_t availableUb = static_cast<int64_t>(ubSize) - RESERVED_UB;
    OP_CHECK_IF(availableUb <= 0, OP_LOGE(context, "availableUb<=0"), return ge::GRAPH_FAILED);
    int64_t bytePerElem = (dataType == ge::DT_FLOAT) ? BYTE_PER_ELEM_FP32 : BYTE_PER_ELEM_FP16;
    int64_t tileElem = availableUb / bytePerElem;
    tileElem = Ops::Base::FloorAlign(tileElem, static_cast<int64_t>(256));
    if (tileElem < alignElem) {
        tileElem = alignElem;
    }
    tiling->ubFactor = tileElem;
    return ge::GRAPH_SUCCESS;
}

// Select TilingKey based on dtype x useNesterov.
static ge::graphStatus SelectTilingKey(gert::TilingContext* context, ge::DataType dataType)
{
    uint32_t dtypeKey;
    if (dataType == ge::DT_FLOAT) {
        dtypeKey = static_cast<uint32_t>(C_DT_FLOAT);
    } else if (dataType == ge::DT_BF16) {
        dtypeKey = static_cast<uint32_t>(C_DT_BF16);
    } else {
        dtypeKey = static_cast<uint32_t>(C_DT_FLOAT16);
    }

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* useNesterovPtr = attrs->GetAttrPointer<bool>(ATTR_USE_NESTEROV);
    uint32_t useNesterov = (useNesterovPtr != nullptr && *useNesterovPtr) ? 1U : 0U;

    ASCENDC_TPL_SEL_PARAM(context, dtypeKey, useNesterov);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyAdamDTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // Lightweight input contract check (same-shape tensors / scalar size==1 / same dtype).
    // Placed before any GM-offset computation so a violated contract fails tiling (no kernel
    // launch -> no OOB GM read). No-op for legal inputs (preserves tiling output byte-for-byte).
    OP_CHECK_IF(CheckInputContract(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckInputContract error"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    ApplyAdamDTilingData* tiling = nullptr;
    OP_CHECK_IF(InitTilingData(context, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "InitTilingData error"),
                return ge::GRAPH_FAILED);

    int64_t alignElem = 0;
    int64_t usedCoreNum = 0;
    OP_CHECK_IF(ComputeMultiCoreSplit(context, dataType, totalNum, coreNum, tiling, alignElem, usedCoreNum) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeMultiCoreSplit error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeUbSplit(context, dataType, ubSize, alignElem, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeUbSplit error"), return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);

    OP_CHECK_IF(SelectTilingKey(context, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SelectTilingKey error"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyAdamD([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyAdamDCompileInfo {};

IMPL_OP_OPTILING(ApplyAdamD).Tiling(ApplyAdamDTilingFunc).TilingParse<ApplyAdamDCompileInfo>(TilingParseForApplyAdamD);

} // namespace optiling
