/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_with_min_max_vars_gradient_tiling_arch35.cpp
 * \brief Host-side tiling for fake_quant_with_min_max_vars_gradient (arch35)
 *
 * Key differences from args version:
 *  - min/max are tensor inputs, NOT read in Tiling
 *  - No CalcNudge() — nudgedMin/nudgedMax computed on Device side
 *  - Workspace = 0 (multi-core reduce via SetAtomicAdd, no workspace needed)
 *  - TilingData includes numBits and narrowRange (NOT nudgedMin/nudgedMax)
 */

#include "fake_quant_with_min_max_vars_gradient_tiling_arch35.h"

#include <cmath>
#include <cstring>

#include "../../op_kernel/arch35/fake_quant_with_min_max_vars_gradient_struct.h"

using namespace FakeQuantWithMinMaxVarsGradientOp;

namespace optiling {

constexpr size_t INPUT_GRAD_INDEX = 0;
constexpr size_t INPUT_X_INDEX = 1;
constexpr size_t ATTR_NUM_BITS_INDEX = 0;
constexpr size_t ATTR_NARROW_RANGE_INDEX = 1;
constexpr int64_t DEFAULT_BASE_LEN = 8192;
constexpr int64_t BLOCK_ALIGN_FP32 = 8;
constexpr int64_t BUFF_NUM = 2;
constexpr int64_t MIN_BLOCK_FACTOR = 64;

ge::graphStatus FakeQuantWithMinMaxVarsGradientTiling::GetCompileInfo()
{
    auto compileInfo = context_->GetCompileInfo<FakeQuantWithMinMaxVarsGradientCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->vectorCoreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(context_->GetNodeName(),
                "FakeQuantWithMinMaxVarsGradient GetCompileInfo Failed, coreNum:%ld, ubSize:%lu.", coreNum_, ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxVarsGradientTiling::GetOpParam()
{
    // Read x shape to compute totalLen
    auto xShape = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    const gert::Shape& shape = xShape->GetStorageShape();
    totalLen_ = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalLen_ *= shape.GetDim(i);
    }
    if (totalLen_ < 0) {
        OP_LOGE(context_->GetNodeName(), "Invalid input shape, total len %ld.", totalLen_);
        return ge::GRAPH_FAILED;
    }

    // Validate dtypes
    auto gDesc = context_->GetInputDesc(INPUT_GRAD_INDEX);
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF((gDesc->GetDataType() != ge::DT_FLOAT || xDesc->GetDataType() != ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxVarsGradient only supports float32 inputs."),
                return ge::GRAPH_FAILED);

    // Read attributes (NOT tensor values)
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* attrNumBits = attrs->GetAttrPointer<int64_t>(ATTR_NUM_BITS_INDEX);
    const bool* attrNarrowRange = attrs->GetAttrPointer<bool>(ATTR_NARROW_RANGE_INDEX);

    numBits_ = (attrNumBits != nullptr) ? *attrNumBits : 8;
    narrowRange_ = (attrNarrowRange != nullptr) ? *attrNarrowRange : false;

    // Validate num_bits range [2, 16]
    OP_CHECK_IF((numBits_ < 2 || numBits_ > 16),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxVarsGradient num_bits must be in [2,16], got %ld.",
                        numBits_),
                return ge::GRAPH_FAILED);

    // NOTE: min/max are tensor inputs, NOT read here.
    // nudgedMin/nudgedMax are computed on Device side in the Kernel.

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxVarsGradientTiling::CalcTiling()
{
    if (totalLen_ == 0) {
        numCore_ = 0;
        blockFactor_ = 0;
        blockTailFactor_ = 0;
        baseLen_ = DEFAULT_BASE_LEN;
        return ge::GRAPH_SUCCESS;
    }

    // UB budget: 3 queues (g, x, y) × 2 buffers × baseLen + 2 queues (min, max) × 1 buffer × 32B
    // Approximate as 7 slots × baseLen × sizeof(float) (conservative)
    int64_t ubAvail = static_cast<int64_t>(ubSize_) - 2048;
    int64_t maxBaseLen = ubAvail / (7 * static_cast<int64_t>(sizeof(float)));
    maxBaseLen = (maxBaseLen / BLOCK_ALIGN_FP32) * BLOCK_ALIGN_FP32;
    if (maxBaseLen <= 0) {
        OP_LOGE(context_->GetNodeName(), "UB size %lu too small for fake_quant_with_min_max_vars_gradient.", ubSize_);
        return ge::GRAPH_FAILED;
    }
    baseLen_ = (DEFAULT_BASE_LEN < maxBaseLen) ? DEFAULT_BASE_LEN : maxBaseLen;
    if (baseLen_ > totalLen_) {
        baseLen_ = ((totalLen_ + BLOCK_ALIGN_FP32 - 1) / BLOCK_ALIGN_FP32) * BLOCK_ALIGN_FP32;
        if (baseLen_ == 0) {
            baseLen_ = BLOCK_ALIGN_FP32;
        }
    }

    int64_t coreUpper = (totalLen_ + MIN_BLOCK_FACTOR - 1) / MIN_BLOCK_FACTOR;
    numCore_ = (coreNum_ < coreUpper) ? coreNum_ : coreUpper;
    if (numCore_ < 1) {
        numCore_ = 1;
    }

    int64_t bf = (totalLen_ + numCore_ - 1) / numCore_;
    bf = ((bf + BLOCK_ALIGN_FP32 - 1) / BLOCK_ALIGN_FP32) * BLOCK_ALIGN_FP32;
    if (bf == 0) {
        bf = BLOCK_ALIGN_FP32;
    }
    numCore_ = (totalLen_ + bf - 1) / bf;
    if (numCore_ > coreNum_) {
        numCore_ = coreNum_;
    }
    blockFactor_ = bf;
    blockTailFactor_ = totalLen_ - blockFactor_ * (numCore_ - 1);

    OP_CHECK_IF((blockTailFactor_ <= 0 || blockTailFactor_ > blockFactor_),
                OP_LOGE(context_->GetNodeName(), "tiling not conservative: totalLen=%ld numCore=%ld bf=%ld bt=%ld",
                        totalLen_, numCore_, blockFactor_, blockTailFactor_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxVarsGradientTiling::CalcWorkspace()
{
    // workspace = 2 * max(numCore, 1) * sizeof(float)
    // Used for reduce: [0, numCore) for backprop_wrt_min local sums,
    //                  [numCore, 2*numCore) for backprop_wrt_max local sums
    size_t wsSize = 2 * static_cast<size_t>(numCore_ > 0 ? numCore_ : 1) * sizeof(float);

    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaceSizes);
    workspaceSizes[0] = wsSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxVarsGradientTiling::WriteTilingData()
{
    tilingData_.totalLen = totalLen_;
    tilingData_.numCore = numCore_;
    tilingData_.blockFactor = blockFactor_;
    tilingData_.blockTailFactor = blockTailFactor_;
    tilingData_.baseLen = baseLen_;
    tilingData_.numBits = numBits_;
    tilingData_.narrowRange = narrowRange_;

    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint64_t>(FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_TPL_DEFAULT));

    context_->SetBlockDim(static_cast<uint32_t>(numCore_ > 0 ? numCore_ : 1));
    context_->SetTilingKey(tilingKey_);

    const size_t dataSize = sizeof(tilingData_);
    auto rawTiling = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTiling);
    errno_t ret = memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), reinterpret_cast<void*>(&tilingData_),
                           dataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s tiling failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(dataSize);

    OP_LOGD(context_->GetNodeName(),
            "FakeQuantWithMinMaxVarsGradient tiling: totalLen=%ld numCore=%ld blockFactor=%ld blockTail=%ld "
            "baseLen=%ld numBits=%ld narrowRange=%d tilingKey=%lu",
            totalLen_, numCore_, blockFactor_, blockTailFactor_, baseLen_, numBits_, narrowRange_ ? 1 : 0, tilingKey_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxVarsGradientTiling::DoTiling()
{
    OP_CHECK_IF((GetCompileInfo() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "GetCompileInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetOpParam() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "GetOpParam failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CalcTiling() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalcTiling failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CalcWorkspace() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalcWorkspace failed."),
                return ge::GRAPH_FAILED);
    return WriteTilingData();
}

static ge::graphStatus TilingForFakeQuantWithMinMaxVarsGradient(gert::TilingContext* context)
{
    OP_LOGD("FakeQuantWithMinMaxVarsGradientTiling", "Enter TilingForFakeQuantWithMinMaxVarsGradient");
    OP_CHECK_IF(context == nullptr, OP_LOGE("FakeQuantWithMinMaxVarsGradientTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    FakeQuantWithMinMaxVarsGradientTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForFakeQuantWithMinMaxVarsGradient(gert::TilingParseContext* context)
{
    OP_LOGD("FakeQuantWithMinMaxVarsGradientTiling", "Enter TilingPrepareForFakeQuantWithMinMaxVarsGradient");
    OP_CHECK_IF(context == nullptr, OP_LOGE("FakeQuantWithMinMaxVarsGradientTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompiledInfo<FakeQuantWithMinMaxVarsGradientCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF((compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "GetHardwareInfo failed, vectorCoreNum:%d, ubSize:%lu.",
                        compileInfo->vectorCoreNum, compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FakeQuantWithMinMaxVarsGradient)
    .Tiling(TilingForFakeQuantWithMinMaxVarsGradient)
    .TilingParse<FakeQuantWithMinMaxVarsGradientCompileInfo>(TilingPrepareForFakeQuantWithMinMaxVarsGradient);

} // namespace optiling
