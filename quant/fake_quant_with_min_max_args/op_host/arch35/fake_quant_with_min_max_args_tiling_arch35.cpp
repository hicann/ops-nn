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
 * \file fake_quant_with_min_max_args_tiling_arch35.cpp
 * \brief
 *
 * Host-side Nudge (per design §1.1 & §5):
 *   H1) scaleInv = (qMax-qMin)/(fMax-fMin)  [re-divide, NOT 1/scale]
 *   H2) zeroPointFromMin = qMin - fMin/scale  [division, NOT * scaleInv]
 *   H3) nudgedZeroPoint = std::round(zeroPointFromMin)  [round-half-away]
 *   H4) clip: <= qMin / >= qMax  [closed intervals]
 *   H5) quantZero = floor(-nudgedMin * scaleInv + 0.5f)  [based on nudgedMin, NOT reusing nudgedZeroPoint]
 */

#include "fake_quant_with_min_max_args_tiling_arch35.h"

#include <cmath>
#include <cstring>

#include "../../op_kernel/arch35/fake_quant_with_min_max_args_struct.h"

using namespace FakeQuantWithMinMaxArgsOp;

namespace optiling {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t ATTR_MIN_INDEX = 0;
constexpr size_t ATTR_MAX_INDEX = 1;
constexpr size_t ATTR_NUM_BITS_INDEX = 2;
constexpr size_t ATTR_NARROW_RANGE_INDEX = 3;
constexpr size_t SYNC_WORKSPACE_SIZE = 0;
constexpr int64_t DEFAULT_BASE_LEN = 8192; // 32KB per buffer (fp32)
constexpr int64_t BLOCK_ALIGN_FP32 = 8;    // 32B / 4B
constexpr int64_t BUFF_NUM = 2;            // in + out double-buffering => 2 * 2 buffers logically
constexpr int64_t MIN_BLOCK_FACTOR = 64;   // 至少给单核 256B 工作量

ge::graphStatus FakeQuantWithMinMaxArgsTiling::GetCompileInfo()
{
    auto compileInfo = context_->GetCompileInfo<FakeQuantWithMinMaxArgsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->vectorCoreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((coreNum_ <= 0 || ubSize_ <= 0),
                OP_LOGE(context_->GetNodeName(),
                        "FakeQuantWithMinMaxArgs GetCompileInfo Failed, coreNum:%ld, ubSize:%lu.", coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsTiling::GetOpParam()
{
    // Shape -> totalLen.
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

    // Dtype check (kernel only supports fp32).
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF((xDesc->GetDataType() != ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs only supports float32 input."),
                return ge::GRAPH_FAILED);

    // Attrs.
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
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs requires finite min/max, got min=%f max=%f.",
                        fMin_, fMax_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fMin_ >= fMax_),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs requires min < max, got min=%f max=%f.",
                        fMin_, fMax_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (numBits_ < 2 || numBits_ > 16),
        OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs num_bits must be in [2,16], got %ld.", numBits_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsTiling::CalcNudge()
{
    // TF Eigen Nudge() algorithm with all H1-H5 precision-sensitive cases handled.
    const float qMinF = narrowRange_ ? 1.0f : 0.0f;
    const float qMaxF = static_cast<float>((1ULL << static_cast<uint32_t>(numBits_)) - 1ULL);

    const float scale = (fMax_ - fMin_) / (qMaxF - qMinF); // 反量化
    // H1: scaleInv must be re-divided, NOT 1/scale.
    const float scaleInv = (qMaxF - qMinF) / (fMax_ - fMin_);

    // H2: zeroPointFromMin uses division (qMin - fMin / scale), NOT (qMin - fMin * scaleInv).
    const float zeroPointFromMin = qMinF - fMin_ / scale;

    // H3 & H4: closed-interval clip + std::round (round-half-away).
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

    // H5: quantZero must be recomputed from nudgedMin, NOT directly reuse nudgedZeroPoint.
    const float quantZero = std::floor(-nudgedMin * scaleInv + 0.5f);

    tilingData_.nudgedMin = nudgedMin;
    tilingData_.nudgedMax = nudgedMax;
    tilingData_.scale = scale;
    tilingData_.scaleInv = scaleInv;
    tilingData_.quantZero = quantZero;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsTiling::CalcTiling()
{
    if (totalLen_ == 0) {
        numCore_ = 0;
        blockFactor_ = 0;
        blockTailFactor_ = 0;
        baseLen_ = DEFAULT_BASE_LEN;
        return ge::GRAPH_SUCCESS;
    }

    // baseLen: 限制在 UB 单次能放下 2 份 fp32 buffer 的范围内（in + out），
    //          每份双缓冲（BUFF_NUM），保留少量预留。
    int64_t ubAvail = static_cast<int64_t>(ubSize_) - 2048; // reserve
    int64_t maxBaseLen = ubAvail / (2 * BUFF_NUM * static_cast<int64_t>(sizeof(float)));
    // 向下对齐到 BLOCK_ALIGN_FP32。
    maxBaseLen = (maxBaseLen / BLOCK_ALIGN_FP32) * BLOCK_ALIGN_FP32;
    if (maxBaseLen <= 0) {
        OP_LOGE(context_->GetNodeName(), "UB size %lu too small for fake_quant_with_min_max_args.", ubSize_);
        return ge::GRAPH_FAILED;
    }
    baseLen_ = (DEFAULT_BASE_LEN < maxBaseLen) ? DEFAULT_BASE_LEN : maxBaseLen;
    if (baseLen_ > totalLen_) {
        // 数据少于一份 baseLen，向上取整对齐到 BLOCK_ALIGN_FP32（防止 0）。
        baseLen_ = ((totalLen_ + BLOCK_ALIGN_FP32 - 1) / BLOCK_ALIGN_FP32) * BLOCK_ALIGN_FP32;
        if (baseLen_ == 0) {
            baseLen_ = BLOCK_ALIGN_FP32;
        }
    }

    // 选核数：每核工作量不少于 MIN_BLOCK_FACTOR，但不超过 coreNum_。
    int64_t coreUpper = (totalLen_ + MIN_BLOCK_FACTOR - 1) / MIN_BLOCK_FACTOR;
    numCore_ = (coreNum_ < coreUpper) ? coreNum_ : coreUpper;
    if (numCore_ < 1) {
        numCore_ = 1;
    }

    // blockFactor: 向 BLOCK_ALIGN_FP32 对齐，让每核都能整块对齐搬运。
    int64_t bf = (totalLen_ + numCore_ - 1) / numCore_;
    bf = ((bf + BLOCK_ALIGN_FP32 - 1) / BLOCK_ALIGN_FP32) * BLOCK_ALIGN_FP32;
    if (bf == 0) {
        bf = BLOCK_ALIGN_FP32;
    }
    // 调整 numCore_ 以保证守恒：blockFactor*(numCore-1) + blockTail == totalLen
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

ge::graphStatus FakeQuantWithMinMaxArgsTiling::WriteTilingData()
{
    tilingData_.totalLen = totalLen_;
    tilingData_.numCore = numCore_;
    tilingData_.blockFactor = blockFactor_;
    tilingData_.blockTailFactor = blockTailFactor_;
    tilingData_.baseLen = baseLen_;

    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint64_t>(FAKE_QUANT_WITH_MIN_MAX_ARGS_TPL_DEFAULT));

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

    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaceSizes);
    workspaceSizes[0] = SYNC_WORKSPACE_SIZE;

    OP_LOGD(context_->GetNodeName(),
            "FakeQuantWithMinMaxArgs tiling: totalLen=%ld numCore=%ld blockFactor=%ld blockTail=%ld baseLen=%ld "
            "nudgedMin=%f nudgedMax=%f scale=%f scaleInv=%f quantZero=%f tilingKey=%lu",
            totalLen_, numCore_, blockFactor_, blockTailFactor_, baseLen_, tilingData_.nudgedMin, tilingData_.nudgedMax,
            tilingData_.scale, tilingData_.scaleInv, tilingData_.quantZero, tilingKey_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxArgsTiling::DoTiling()
{
    OP_CHECK_IF((GetCompileInfo() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs GetCompileInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetOpParam() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs GetOpParam failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CalcNudge() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs CalcNudge failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((CalcTiling() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "FakeQuantWithMinMaxArgs CalcTiling failed."),
                return ge::GRAPH_FAILED);
    return WriteTilingData();
}

static ge::graphStatus TilingForFakeQuantWithMinMaxArgs(gert::TilingContext* context)
{
    OP_LOGD("FakeQuantWithMinMaxArgsTiling", "Enter TilingForFakeQuantWithMinMaxArgs");
    OP_CHECK_IF(context == nullptr, OP_LOGE("FakeQuantWithMinMaxArgsTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    FakeQuantWithMinMaxArgsTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForFakeQuantWithMinMaxArgs(gert::TilingParseContext* context)
{
    OP_LOGD("FakeQuantWithMinMaxArgsTiling", "Enter TilingPrepareForFakeQuantWithMinMaxArgs");
    OP_CHECK_IF(context == nullptr, OP_LOGE("FakeQuantWithMinMaxArgsTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompiledInfo<FakeQuantWithMinMaxArgsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF(
        (compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "FakeQuantWithMinMaxArgs GetHardwareInfo failed, vectorCoreNum:%d, ubSize:%lu.",
                compileInfo->vectorCoreNum, compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FakeQuantWithMinMaxArgs)
    .Tiling(TilingForFakeQuantWithMinMaxArgs)
    .TilingParse<FakeQuantWithMinMaxArgsCompileInfo>(TilingPrepareForFakeQuantWithMinMaxArgs);

} // namespace optiling
