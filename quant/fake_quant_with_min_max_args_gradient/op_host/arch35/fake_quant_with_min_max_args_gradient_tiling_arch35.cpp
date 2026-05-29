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
 * \file fake_quant_with_min_max_args_gradient_tiling_arch35.cpp
 * \brief
 *
 * Host-side Nudge (per design §1.1 & §5):
 *   H2) zeroPointFromMin = qMin - fMin/scale  [division, NOT * scaleInv]
 *   H3) nudgedZeroPoint = std::round(zeroPointFromMin)  [round-half-away]
 *   H4) clip: <= qMin / >= qMax  [closed intervals]
 *   (gradient only needs nudgedMin/nudgedMax; H1 scaleInv and H5 quantZero not needed)
 */

#include "fake_quant_with_min_max_args_gradient_tiling_arch35.h"

#include <cmath>
#include <cstring>

#include "../../op_kernel/arch35/fake_quant_with_min_max_args_gradient_struct.h"

using namespace FakeQuantWithMinMaxArgsGradientOp;

namespace optiling {

constexpr size_t INPUT_GRAD_INDEX = 0;
constexpr size_t INPUT_X_INDEX = 1;
constexpr size_t ATTR_MIN_INDEX = 0;
constexpr size_t ATTR_MAX_INDEX = 1;
constexpr size_t ATTR_NUM_BITS_INDEX = 2;
constexpr size_t ATTR_NARROW_RANGE_INDEX = 3;
constexpr size_t SYNC_WORKSPACE_SIZE = 0;
constexpr int64_t DEFAULT_BASE_LEN = 8192;
constexpr int64_t BLOCK_ALIGN_FP32 = 8;
constexpr int64_t BUFF_NUM = 2;
constexpr int64_t MIN_BLOCK_FACTOR = 64;

ge::graphStatus FakeQuantWithMinMaxArgsGradientTiling::GetCompileInfo()
{
    auto compileInfo = context_->GetCompileInfo<FakeQuantWithMinMaxArgsGradientCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->vectorCoreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((coreNum_ <= 0 || ubSize_ <= 0),
                OP_LOGE(context_->GetNodeName(),
                        "FakeQuantWithMinMaxArgsGradient GetCompileInfo Failed, coreNum:%ld, ubSize:%lu.",
                        coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsGradientTiling::GetOpParam()
{
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

    auto gDesc = context_->GetInputDesc(INPUT_GRAD_INDEX);
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF((gDesc->GetDataType() != ge::DT_FLOAT || xDesc->GetDataType() != ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(),
                        "FakeQuantWithMinMaxArgsGradient only supports float32 inputs."),
                return ge::GRAPH_FAILED);

    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const float* attrMin = attrs->GetAttrPointer<float>(ATTR_MIN_INDEX);
    const float* attrMax = attrs->GetAttrPointer<float>(ATTR_MAX_INDEX);
    const int64_t* attrNumBits = attrs->GetAttrPointer<int64_t>(ATTR_NUM_BITS_INDEX);
    const bool* attrNarrowRange = attrs->GetAttrPointer<bool>(ATTR_NARROW_RANGE_INDEX);

    fMin_ = (attrMin != nullptr) ? *attrMin : -6.0f;
    fMax_ = (attrMax != nullptr) ? *attrMax : 6.0f;
    numBits_ = (attrNumBits != nullptr) ? *attrNumBits : 8;
    narrowRange_ = (attrNarrowRange != nullptr) ? *attrNarrowRange : false;

    OP_CHECK_IF((std::isnan(fMin_) || std::isnan(fMax_) || std::isinf(fMin_) || std::isinf(fMax_)),
                OP_LOGE(context_->GetNodeName(),
                        "FakeQuantWithMinMaxArgsGradient requires finite min/max, got min=%f max=%f.", fMin_, fMax_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fMin_ >= fMax_),
                OP_LOGE(context_->GetNodeName(),
                        "FakeQuantWithMinMaxArgsGradient requires min < max, got min=%f max=%f.", fMin_, fMax_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numBits_ < 2 || numBits_ > 16),
                OP_LOGE(context_->GetNodeName(),
                        "FakeQuantWithMinMaxArgsGradient num_bits must be in [2,16], got %ld.", numBits_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsGradientTiling::CalcNudge()
{
    const float qMinF = narrowRange_ ? 1.0f : 0.0f;
    const float qMaxF = static_cast<float>((1ULL << static_cast<uint32_t>(numBits_)) - 1ULL);

    const float scale = (fMax_ - fMin_) / (qMaxF - qMinF);
    // H2: zeroPointFromMin uses division.
    const float zeroPointFromMin = qMinF - fMin_ / scale;

    float nudgedZeroPoint = 0.0f;
    if (zeroPointFromMin <= qMinF) {
        nudgedZeroPoint = qMinF;
    } else if (zeroPointFromMin >= qMaxF) {
        nudgedZeroPoint = qMaxF;
    } else {
        nudgedZeroPoint = std::round(zeroPointFromMin);
    }

    const float nudgedMin = (qMinF - nudgedZeroPoint) * scale;
    const float nudgedMax = (qMaxF - nudgedZeroPoint) * scale;

    tilingData_.nudgedMin = nudgedMin;
    tilingData_.nudgedMax = nudgedMax;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsGradientTiling::CalcTiling()
{
    if (totalLen_ == 0) {
        numCore_ = 0;
        blockFactor_ = 0;
        blockTailFactor_ = 0;
        baseLen_ = DEFAULT_BASE_LEN;
        return ge::GRAPH_SUCCESS;
    }

    // UB budget: gradient holds 2 inputs (g, x) + 1 output (y) buffers; sign-bit handling is
    // done in registers, no UB scratch. 3 queues * BUFF_NUM = 6 fp32 slots; divisor 7 keeps a
    // 1-slot safety margin.
    int64_t ubAvail = static_cast<int64_t>(ubSize_) - 2048;
    int64_t maxBaseLen = ubAvail / (7 * static_cast<int64_t>(sizeof(float)));
    maxBaseLen = (maxBaseLen / BLOCK_ALIGN_FP32) * BLOCK_ALIGN_FP32;
    if (maxBaseLen <= 0) {
        OP_LOGE(context_->GetNodeName(),
                "UB size %lu too small for fake_quant_with_min_max_args_gradient.", ubSize_);
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
                OP_LOGE(context_->GetNodeName(),
                        "tiling not conservative: totalLen=%ld numCore=%ld bf=%ld bt=%ld",
                        totalLen_, numCore_, blockFactor_, blockTailFactor_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsGradientTiling::WriteTilingData()
{
    tilingData_.totalLen = totalLen_;
    tilingData_.numCore = numCore_;
    tilingData_.blockFactor = blockFactor_;
    tilingData_.blockTailFactor = blockTailFactor_;
    tilingData_.baseLen = baseLen_;

    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint64_t>(FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_TPL_DEFAULT));

    context_->SetBlockDim(static_cast<uint32_t>(numCore_ > 0 ? numCore_ : 1));
    context_->SetTilingKey(tilingKey_);

    const size_t dataSize = sizeof(tilingData_);
    auto rawTiling = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTiling);
    errno_t ret = memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), dataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s tiling failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(dataSize);

    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaceSizes);
    workspaceSizes[0] = SYNC_WORKSPACE_SIZE;

    OP_LOGD(context_->GetNodeName(),
            "FakeQuantWithMinMaxArgsGradient tiling: totalLen=%ld numCore=%ld blockFactor=%ld blockTail=%ld "
            "baseLen=%ld nudgedMin=%f nudgedMax=%f tilingKey=%lu",
            totalLen_, numCore_, blockFactor_, blockTailFactor_, baseLen_,
            tilingData_.nudgedMin, tilingData_.nudgedMax, tilingKey_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsGradientTiling::DoTiling()
{
    OP_CHECK_IF((GetCompileInfo() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "GetCompileInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetOpParam() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "GetOpParam failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CalcNudge() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "CalcNudge failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CalcTiling() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "CalcTiling failed."),
                return ge::GRAPH_FAILED);
    return WriteTilingData();
}

static ge::graphStatus TilingForFakeQuantWithMinMaxArgsGradient(gert::TilingContext* context)
{
    OP_LOGD("FakeQuantWithMinMaxArgsGradientTiling", "Enter TilingForFakeQuantWithMinMaxArgsGradient");
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FakeQuantWithMinMaxArgsGradientTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    FakeQuantWithMinMaxArgsGradientTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForFakeQuantWithMinMaxArgsGradient(gert::TilingParseContext* context)
{
    OP_LOGD("FakeQuantWithMinMaxArgsGradientTiling", "Enter TilingPrepareForFakeQuantWithMinMaxArgsGradient");
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FakeQuantWithMinMaxArgsGradientTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompiledInfo<FakeQuantWithMinMaxArgsGradientCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF((compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize <= 0),
                OP_LOGE(context->GetNodeName(),
                        "GetHardwareInfo failed, vectorCoreNum:%d, ubSize:%lu.",
                        compileInfo->vectorCoreNum, compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FakeQuantWithMinMaxArgsGradient)
    .Tiling(TilingForFakeQuantWithMinMaxArgsGradient)
    .TilingParse<FakeQuantWithMinMaxArgsGradientCompileInfo>(TilingPrepareForFakeQuantWithMinMaxArgsGradient);

} // namespace optiling
