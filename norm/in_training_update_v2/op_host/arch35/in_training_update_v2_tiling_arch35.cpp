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
 * \file in_training_update_v2_tiling_arch35.cpp
 * \brief Public-contract validation and tiling for INTrainingUpdateV2 on DAV_3510.
 */

#include "in_training_update_v2_tiling_arch35.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/in_training_update_v2_tiling_key.h"

using namespace optiling;

BEGIN_TILING_DATA_DEF(INTrainingUpdateV2TilingDataDef)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, r);
TILING_DATA_FIELD_DEF(int64_t, totalElements);
TILING_DATA_FIELD_DEF(int64_t, unitBlocks);
TILING_DATA_FIELD_DEF(int64_t, rCores);
TILING_DATA_FIELD_DEF(int64_t, formerBlockNum);
TILING_DATA_FIELD_DEF(int64_t, formerUnits);
TILING_DATA_FIELD_DEF(int64_t, latterUnits);
TILING_DATA_FIELD_DEF(int64_t, tileElems);
TILING_DATA_FIELD_DEF(int64_t, rTile);
TILING_DATA_FIELD_DEF(uint32_t, xyBufferBytes);
TILING_DATA_FIELD_DEF(uint32_t, statBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, hasAffine);
TILING_DATA_FIELD_DEF(int64_t, hasRunning);
TILING_DATA_FIELD_DEF(int64_t, gammaBatchStride);
TILING_DATA_FIELD_DEF(int64_t, betaBatchStride);
TILING_DATA_FIELD_DEF(float, invR);
TILING_DATA_FIELD_DEF(float, bessel);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, oneMinusMomentum);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, invRCorrection);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(INTrainingUpdateV2, INTrainingUpdateV2TilingDataDef)

namespace optiling {
namespace {

constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_SUM = 1;
constexpr size_t INPUT_SQUARE_SUM = 2;
constexpr size_t INPUT_GAMMA = 3;
constexpr size_t INPUT_BETA = 4;
constexpr size_t INPUT_MEAN = 5;
constexpr size_t INPUT_VARIANCE = 6;
constexpr size_t OUTPUT_Y = 0;
constexpr size_t OUTPUT_BATCH_MEAN = 1;
constexpr size_t OUTPUT_BATCH_VARIANCE = 2;
constexpr size_t ATTR_MOMENTUM = 0;
constexpr size_t ATTR_EPSILON = 1;
constexpr float DEFAULT_MOMENTUM = 0.1F;
constexpr float DEFAULT_EPSILON = 1.0E-5F;

constexpr uint64_t TILING_KEY_EMPTY = 1000;
constexpr uint64_t TILING_KEY_NCHW = 2000;
constexpr uint64_t TILING_KEY_NHWC = 3000;
constexpr int64_t PUBLIC_RANK = 4;
constexpr int64_t VECTOR_LANES = 64;
constexpr int64_t NHWC_GUARD_ELEMS = 64;
constexpr int64_t CHANNEL_TILE = 64;
constexpr int64_t STAT_CHUNK = 64;
constexpr int64_t DMA_BLOCK_BYTES = 32;
constexpr int64_t DMA_MAX_BLOCK_COUNT = 4095;
constexpr int64_t DMA_MAX_BLOCK_LENGTH_BYTES = 2097151;
constexpr int64_t DMA_MAX_GM_STRIDE_BYTES = (1LL << 40) - 1;
constexpr int64_t RESERVED_UB_BYTES = 13632;
constexpr int64_t X_Y_QUEUE_BUFFER_COUNT = 4;
constexpr int64_t STAT_BUFFER_COUNT = 8;
constexpr int64_t FLOAT16_BYTES = sizeof(uint16_t);
constexpr int64_t FLOAT_BYTES = sizeof(float);

bool IsPublicFormat(ge::Format format) { return format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC; }

bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

int64_t CeilDiv(int64_t value, int64_t divisor) { return value / divisor + ((value % divisor) != 0 ? 1 : 0); }

bool SameDims(const std::array<int64_t, 4>& lhs, const std::array<int64_t, 4>& rhs) { return lhs == rhs; }

std::string DimsToString(const std::array<int64_t, 4>& dims)
{
    return "[" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "," +
           std::to_string(dims[3]) + "]";
}

} // namespace

ge::graphStatus INTrainingUpdateV2Tiling::GetPlatformInfo()
{
    auto* platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const auto* compileInfo = context_->GetCompileInfo<INTrainingUpdateV2CompileInfo>();
        if (compileInfo == nullptr) {
            OP_LOGE(context_->GetNodeName(), "compile info is null");
            return ge::GRAPH_FAILED;
        }
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (ubSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            OP_LOGE(context_->GetNodeName(), "UB size is outside int64 range");
            return ge::GRAPH_FAILED;
        }
        ubSize_ = static_cast<int64_t>(ubSize);
    }
    if (coreNum_ <= 0 || ubSize_ <= 0) {
        OP_LOGE(context_->GetNodeName(), "invalid platform info: coreNum=%" PRId64 ", ubSize=%" PRId64, coreNum_,
                ubSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool INTrainingUpdateV2Tiling::ReadShape(const gert::StorageShape* shape, const char* name,
                                         std::array<int64_t, 4>& dims) const
{
    if (shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), name, "null shape", "shape must be present");
        return false;
    }
    const gert::Shape& originShape = shape->GetOriginShape();
    if (originShape.GetDimNum() != PUBLIC_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), name,
                                                 std::to_string(originShape.GetDimNum()).c_str(),
                                                 "the public contract requires rank 4");
        return false;
    }
    for (size_t i = 0; i < dims.size(); ++i) {
        dims[i] = originShape.GetDim(i);
        if (dims[i] < 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), name, DimsToString(dims).c_str(),
                                                  "all dimensions must be concrete and non-negative before tiling");
            return false;
        }
    }
    return true;
}

bool INTrainingUpdateV2Tiling::GetOptionalPresence(size_t index, const char* name, bool& present)
{
    const auto* desc = context_->GetOptionalInputDesc(index);
    const auto* shape = context_->GetOptionalInputShape(index);
    if ((desc == nullptr) != (shape == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), name, "descriptor/shape mismatch",
                                              "descriptor and shape must both be present or both be absent");
        return false;
    }
    present = desc != nullptr;
    return true;
}

bool INTrainingUpdateV2Tiling::ValidatePublicStatSelf(size_t index, const char* name, bool optional, ge::Format& format,
                                                      std::array<int64_t, 4>& dims)
{
    const auto* desc = optional ? context_->GetOptionalInputDesc(index) : context_->GetInputDesc(index);
    const auto* shape = optional ? context_->GetOptionalInputShape(index) : context_->GetRequiredInputShape(index);
    if (desc == nullptr || shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), name, "missing descriptor or shape",
                                              "the tensor is required by the active validation path");
        return false;
    }
    if (desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), name,
                                              std::to_string(static_cast<int64_t>(desc->GetDataType())).c_str(),
                                              "the public contract requires float32");
        return false;
    }
    format = desc->GetOriginFormat();
    if (!IsPublicFormat(format)) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), name,
                                               std::to_string(static_cast<int64_t>(format)).c_str(),
                                               "the public contract requires NCHW or NHWC");
        return false;
    }
    if (!ReadShape(shape, name, dims)) {
        return false;
    }
    const bool spatialIsOne = (format == ge::FORMAT_NCHW) ? (dims[2] == 1 && dims[3] == 1) :
                                                            (dims[1] == 1 && dims[2] == 1);
    if (!spatialIsOne) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), name, DimsToString(dims).c_str(),
                                              "the two public spatial dimensions must both equal 1");
        return false;
    }
    return true;
}

bool INTrainingUpdateV2Tiling::ValidateExactStat(size_t index, const char* name, bool optional)
{
    ge::Format format = ge::FORMAT_RESERVED;
    std::array<int64_t, 4> dims{};
    if (!ValidatePublicStatSelf(index, name, optional, format, dims)) {
        return false;
    }
    if (format != xFormat_ || !SameDims(dims, statDims_)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), name, DimsToString(dims).c_str(),
            "the tensor must use x's logical format and exact [N,C,1,1]/[N,1,1,C] shape");
        return false;
    }
    return true;
}

bool INTrainingUpdateV2Tiling::ValidateAffine(size_t index, const char* name, int64_t& batchStride)
{
    const size_t optionalSlot = index - INPUT_GAMMA;
    const auto& dims = optionalDims_[optionalSlot];
    if (optionalFormats_[optionalSlot] != xFormat_) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            context_->GetNodeName(), name, std::to_string(static_cast<int64_t>(optionalFormats_[optionalSlot])).c_str(),
            "the affine tensor must use the same logical format as x");
        return false;
    }
    const int64_t g = dims[0];
    const int64_t channel = (xFormat_ == ge::FORMAT_NCHW) ? dims[1] : dims[3];
    if ((g != 1 && g != data_.n) || channel != data_.c) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), name, DimsToString(dims).c_str(),
                                              "G must be 1 or N and C must equal x.C");
        return false;
    }
    batchStride = (g == 1) ? 0 : data_.c;
    return true;
}

ge::graphStatus INTrainingUpdateV2Tiling::ValidateRequiredInputs()
{
    const auto* xDesc = context_->GetInputDesc(INPUT_X);
    const auto* xShape = context_->GetRequiredInputShape(INPUT_X);
    if (xDesc == nullptr || xShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", "missing descriptor or shape",
                                              "x descriptor and shape must both be present");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType xDtype = xDesc->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                              std::to_string(static_cast<int64_t>(xDtype)).c_str(),
                                              "the public contract requires float16 or float32");
        return ge::GRAPH_FAILED;
    }
    xDtype_ = xDtype;
    xDtypeSize_ = (xDtype == ge::DT_FLOAT16) ? FLOAT16_BYTES : FLOAT_BYTES;
    xFormat_ = xDesc->GetOriginFormat();
    if (!IsPublicFormat(xFormat_)) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), "x",
                                               std::to_string(static_cast<int64_t>(xFormat_)).c_str(),
                                               "the public contract requires NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    if (!ReadShape(xShape, "x", xDims_)) {
        return ge::GRAPH_FAILED;
    }

    data_.n = xDims_[0];
    if (xFormat_ == ge::FORMAT_NCHW) {
        data_.c = xDims_[1];
        statDims_ = {data_.n, data_.c, 1, 1};
    } else {
        data_.c = xDims_[3];
        statDims_ = {data_.n, 1, 1, data_.c};
    }
    if (!ValidateExactStat(INPUT_SUM, "sum", false) || !ValidateExactStat(INPUT_SQUARE_SUM, "square_sum", false)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingUpdateV2Tiling::ValidateOptionalInputs()
{
    constexpr std::array<size_t, OPTIONAL_SLOT_COUNT> indexes = {INPUT_GAMMA, INPUT_BETA, INPUT_MEAN, INPUT_VARIANCE};
    constexpr std::array<const char*, OPTIONAL_SLOT_COUNT> names = {"gamma", "beta", "mean", "variance"};
    for (size_t slot = 0; slot < indexes.size(); ++slot) {
        if (!GetOptionalPresence(indexes[slot], names[slot], optionalPresent_[slot])) {
            return ge::GRAPH_FAILED;
        }
        if (!optionalPresent_[slot]) {
            continue;
        }
        if (!ValidatePublicStatSelf(indexes[slot], names[slot], true, optionalFormats_[slot], optionalDims_[slot])) {
            return ge::GRAPH_FAILED;
        }
    }

    data_.hasAffine = (optionalPresent_[OPTIONAL_GAMMA_SLOT] && optionalPresent_[OPTIONAL_BETA_SLOT]) ? 1 : 0;
    data_.hasRunning = (optionalPresent_[OPTIONAL_MEAN_SLOT] && optionalPresent_[OPTIONAL_VARIANCE_SLOT]) ? 1 : 0;
    if (data_.hasAffine != 0) {
        if (!ValidateAffine(INPUT_GAMMA, "gamma", data_.gammaBatchStride) ||
            !ValidateAffine(INPUT_BETA, "beta", data_.betaBatchStride)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (data_.hasRunning != 0) {
        if (optionalFormats_[OPTIONAL_MEAN_SLOT] != xFormat_ || optionalFormats_[OPTIONAL_VARIANCE_SLOT] != xFormat_ ||
            !SameDims(optionalDims_[OPTIONAL_MEAN_SLOT], statDims_) ||
            !SameDims(optionalDims_[OPTIONAL_VARIANCE_SLOT], statDims_)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "mean and variance", "format/shape mismatch",
                "both tensors must use x's logical format and exact statistics shape when active");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool INTrainingUpdateV2Tiling::CheckOutput(size_t index, const char* name, ge::DataType dtype, ge::Format format,
                                           const std::array<int64_t, 4>& dims) const
{
    const auto* desc = context_->GetOutputDesc(index);
    const auto* shape = context_->GetOutputShape(index);
    if (desc == nullptr || shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), name, "missing descriptor or shape",
                                              "output descriptor and shape must both be present");
        return false;
    }
    std::array<int64_t, 4> actualDims{};
    if (!ReadShape(shape, name, actualDims)) {
        return false;
    }
    if (desc->GetDataType() != dtype || desc->GetOriginFormat() != format || !SameDims(actualDims, dims)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), name, "dtype/format/shape mismatch",
                                              "output metadata must match the public inference result");
        return false;
    }
    return true;
}

ge::graphStatus INTrainingUpdateV2Tiling::ValidateOutputs()
{
    if (!CheckOutput(OUTPUT_Y, "y", xDtype_, xFormat_, xDims_) ||
        !CheckOutput(OUTPUT_BATCH_MEAN, "batch_mean", ge::DT_FLOAT, xFormat_, statDims_) ||
        !CheckOutput(OUTPUT_BATCH_VARIANCE, "batch_variance", ge::DT_FLOAT, xFormat_, statDims_)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingUpdateV2Tiling::ParseAndValidate()
{
    const auto* attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context_->GetNodeName(), "attributes are null");
        return ge::GRAPH_FAILED;
    }
    const float* momentum = attrs->GetFloat(ATTR_MOMENTUM);
    const float* epsilon = attrs->GetFloat(ATTR_EPSILON);
    data_.momentum = (momentum == nullptr) ? DEFAULT_MOMENTUM : *momentum;
    data_.oneMinusMomentum = 1.0f - data_.momentum;
    data_.epsilon = (epsilon == nullptr) ? DEFAULT_EPSILON : *epsilon;

    if (ValidateRequiredInputs() != ge::GRAPH_SUCCESS || ValidateOptionalInputs() != ge::GRAPH_SUCCESS ||
        ValidateOutputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t statElements = 0;
    if (!SafeMul(data_.n, data_.c, statElements)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "N and C", "product overflows int64",
                                               "shape products must be representable by int64");
        return ge::GRAPH_FAILED;
    }
    if (data_.n == 0 || data_.c == 0) {
        data_.r = 0;
        data_.totalElements = 0;
        unitCount_ = 0;
        data_.unitBlocks = 1;
        data_.rCores = 1;
        tilingKey_ = TILING_KEY_EMPTY;
        return ge::GRAPH_SUCCESS;
    }

    const int64_t h = (xFormat_ == ge::FORMAT_NCHW) ? xDims_[2] : xDims_[1];
    const int64_t w = (xFormat_ == ge::FORMAT_NCHW) ? xDims_[3] : xDims_[2];
    if (h <= 0 || w <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", DimsToString(xDims_).c_str(),
                                              "H and W must be positive when N and C are positive");
        return ge::GRAPH_FAILED;
    }
    if (!SafeMul(h, w, data_.r) || !SafeMul(statElements, data_.r, data_.totalElements)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", DimsToString(xDims_).c_str(),
                                              "shape products must be representable by int64");
        return ge::GRAPH_FAILED;
    }
    int64_t xByteExtent = 0;
    int64_t statByteExtent = 0;
    if (!SafeMul(data_.totalElements, xDtypeSize_, xByteExtent) ||
        !SafeMul(statElements, FLOAT_BYTES, statByteExtent)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", DimsToString(xDims_).c_str(),
                                              "tensor byte extents must be representable by int64");
        return ge::GRAPH_FAILED;
    }
    const double invRExact = 1.0 / static_cast<double>(data_.r);
    data_.invR = static_cast<float>(invRExact);
    data_.invRCorrection = static_cast<float>(invRExact - static_cast<double>(data_.invR));
    data_.bessel = (data_.r == 1) ? 0.0f : static_cast<float>(data_.r) / static_cast<float>(data_.r - 1);
    tilingKey_ = (xFormat_ == ge::FORMAT_NCHW) ? TILING_KEY_NCHW : TILING_KEY_NHWC;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingUpdateV2Tiling::CalculateNormalTiling()
{
    const int64_t ubAvailable = ubSize_ - RESERVED_UB_BYTES;
    if (ubAvailable <= 0) {
        OP_LOGE(context_->GetNodeName(), "UB size %" PRId64 " is not greater than reserved bytes %" PRId64, ubSize_,
                RESERVED_UB_BYTES);
        return ge::GRAPH_FAILED;
    }
    const int64_t perElementBytes = X_Y_QUEUE_BUFFER_COUNT * xDtypeSize_;
    const int64_t ubTileElems = (ubAvailable / perElementBytes) / VECTOR_LANES * VECTOR_LANES;
    const int64_t dmaTileElems = (DMA_MAX_BLOCK_LENGTH_BYTES / xDtypeSize_) / VECTOR_LANES * VECTOR_LANES;
    data_.tileElems = std::min(ubTileElems, dmaTileElems);
    if (data_.tileElems < VECTOR_LANES) {
        OP_LOGE(context_->GetNodeName(), "UB tile has only %" PRId64 " elements, at least 64 are required",
                data_.tileElems);
        return ge::GRAPH_FAILED;
    }
    if (data_.tileElems > std::numeric_limits<int64_t>::max() - NHWC_GUARD_ELEMS) {
        OP_LOGE(context_->GetNodeName(), "UB tile plus guard overflows int64");
        return ge::GRAPH_FAILED;
    }
    const int64_t tileElemsWithGuard = data_.tileElems + NHWC_GUARD_ELEMS;
    int64_t xyBufferBytes = 0;
    const int64_t statBufferBytes = STAT_CHUNK * FLOAT_BYTES;
    int64_t allXyBufferBytes = 0;
    int64_t allStatBufferBytes = 0;
    if (!SafeMul(tileElemsWithGuard, xDtypeSize_, xyBufferBytes) ||
        !SafeMul(X_Y_QUEUE_BUFFER_COUNT, xyBufferBytes, allXyBufferBytes) ||
        !SafeMul(STAT_BUFFER_COUNT, statBufferBytes, allStatBufferBytes) ||
        allStatBufferBytes > std::numeric_limits<int64_t>::max() - allXyBufferBytes ||
        allXyBufferBytes + allStatBufferBytes > ubSize_ ||
        xyBufferBytes > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        statBufferBytes > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context_->GetNodeName(), "calculated UB buffers exceed the available UB or API size fields");
        return ge::GRAPH_FAILED;
    }
    data_.xyBufferBytes = static_cast<uint32_t>(xyBufferBytes);
    data_.statBufferBytes = static_cast<uint32_t>(statBufferBytes);

    if (xFormat_ == ge::FORMAT_NCHW) {
        cTile_ = 0;
        data_.rTile = 0;
    } else {
        const int64_t channelUnits = CeilDiv(data_.c, CHANNEL_TILE);
        cTile_ = CeilDiv(data_.c, channelUnits);
        const int64_t rowBytes = cTile_ * xDtypeSize_;
        const int64_t rowStrideBytes = ((rowBytes + DMA_BLOCK_BYTES - 1) / DMA_BLOCK_BYTES) * DMA_BLOCK_BYTES;
        const int64_t rowStrideElems = rowStrideBytes / xDtypeSize_;
        data_.rTile = std::min(DMA_MAX_BLOCK_COUNT, data_.tileElems / rowStrideElems);
        if (data_.rTile < 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", DimsToString(xDims_).c_str(),
                                                  "the NHWC row tile must contain at least one row");
            return ge::GRAPH_FAILED;
        }
        const int64_t minTileC = data_.c / channelUnits;
        int64_t maxGmStrideBytes = 0;
        if (!SafeMul(data_.c - minTileC, xDtypeSize_, maxGmStrideBytes) || maxGmStrideBytes > DMA_MAX_GM_STRIDE_BYTES) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", DimsToString(xDims_).c_str(),
                                                  "the NHWC GM byte stride must be representable by DataCopyPad");
            return ge::GRAPH_FAILED;
        }
    }

    // Partition the logical output by complete 32-byte GM blocks.  Kernel-side
    // non-aligned DataCopyPad writes can then never share a physical block
    // across AI Vector cores.
    const int64_t elementsPerGmBlock = DMA_BLOCK_BYTES / xDtypeSize_;
    unitCount_ = CeilDiv(data_.totalElements, elementsPerGmBlock);
    data_.unitBlocks = std::min(unitCount_, coreNum_);
    if (data_.unitBlocks <= 0) {
        OP_LOGE(context_->GetNodeName(), "normal tiling produced no ownership unit");
        return ge::GRAPH_FAILED;
    }
    data_.rCores = 1;
    const int64_t baseUnits = unitCount_ / data_.unitBlocks;
    data_.formerBlockNum = unitCount_ % data_.unitBlocks;
    data_.formerUnits = baseUnits + (data_.formerBlockNum != 0 ? 1 : 0);
    data_.latterUnits = baseUnits;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingUpdateV2Tiling::FillTilingData()
{
    auto* tilingData = context_->GetTilingData<INTrainingUpdateV2TilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context_->GetNodeName(), "tiling data buffer is null or too small");
        return ge::GRAPH_FAILED;
    }
    *tilingData = data_;

    int64_t usedCores = 1;
    if (tilingKey_ != TILING_KEY_EMPTY && !SafeMul(data_.unitBlocks, data_.rCores, usedCores)) {
        OP_LOGE(context_->GetNodeName(), "used core count overflows int64");
        return ge::GRAPH_FAILED;
    }
    if (usedCores <= 0 || usedCores > coreNum_) {
        OP_LOGE(context_->GetNodeName(), "invalid used core count %" PRId64 " (available %" PRId64 ")", usedCores,
                coreNum_);
        return ge::GRAPH_FAILED;
    }
    if (context_->SetBlockDim(static_cast<uint32_t>(usedCores)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "failed to set block dim to %" PRId64, usedCores);
        return ge::GRAPH_FAILED;
    }
    const uint64_t encodedTilingKey = GET_TPL_TILING_KEY(tilingKey_);
    if (encodedTilingKey == INVALID_TILING_KEY || context_->SetTilingKey(encodedTilingKey) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "failed to encode or set tiling key %" PRIu64, tilingKey_);
        return ge::GRAPH_FAILED;
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OP_LOGE(context_->GetNodeName(), "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 0;

    OP_LOGI(context_->GetNodeName(),
            "INTrainingUpdateV2 key=%" PRIu64 " N=%" PRId64 " C=%" PRId64 " R=%" PRId64 " units=%" PRId64
            " unitBlocks=%" PRId64 " rCores=%" PRId64 " tile=%" PRId64 " cTile=%" PRId64 " rTile=%" PRId64
            " affine=%" PRId64 " running=%" PRId64,
            tilingKey_, data_.n, data_.c, data_.r, unitCount_, data_.unitBlocks, data_.rCores, data_.tileElems, cTile_,
            data_.rTile, data_.hasAffine, data_.hasRunning);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingUpdateV2Tiling::DoTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS || ParseAndValidate() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tilingKey_ != TILING_KEY_EMPTY && CalculateNormalTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return FillTilingData();
}

static ge::graphStatus TilingForINTrainingUpdateV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("INTrainingUpdateV2", "tiling context is null");
        return ge::GRAPH_FAILED;
    }
    INTrainingUpdateV2Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForINTrainingUpdateV2(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("INTrainingUpdateV2", "tiling parse context is null");
        return ge::GRAPH_FAILED;
    }
    auto* compileInfo = context->GetCompiledInfo<INTrainingUpdateV2CompileInfo>();
    if (compileInfo == nullptr || context->GetPlatformInfo() == nullptr) {
        OP_LOGE(context->GetNodeName(), "compile info or platform info is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        OP_LOGE(context->GetNodeName(), "UB size is outside int64 range");
        return ge::GRAPH_FAILED;
    }
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(INTrainingUpdateV2)
    .Tiling(TilingForINTrainingUpdateV2)
    .TilingParse<INTrainingUpdateV2CompileInfo>(TilingPrepareForINTrainingUpdateV2);

} // namespace optiling
