/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <limits>
#include "error_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "apply_came_part3_tiling_arch35.h"
#include "../../op_kernel/arch35/apply_came_part3_tiling_key.h"

namespace optiling {
namespace {
constexpr int64_t kVectorBlockBytes = 256;
constexpr int64_t kGmCacheLineBytes = 64;
constexpr int64_t kFp32Bytes = 4;
constexpr int64_t kFp32PerVectorBlock = kVectorBlockBytes / kFp32Bytes;
constexpr int64_t kWorkspaceBytes = 20 * 1024 * 1024;
constexpr int64_t kDetWorkspaceBytes = 1568;
constexpr int64_t kGlobalShapeIndex = 6;
constexpr int64_t kMaxSupportedDimension = std::numeric_limits<int32_t>::max();

int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor <= 0) {
        return value;
    }
    return value / factor + (value % factor == 0 ? 0 : 1);
}

int64_t CeilAlign(int64_t value, int64_t factor)
{
    const int64_t quotient = CeilDiv(value, factor);
    if (factor <= 0 || quotient > std::numeric_limits<int64_t>::max() / factor) {
        return std::numeric_limits<int64_t>::max();
    }
    return quotient * factor;
}

bool IsScalarShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 ||
           (shape.GetDimNum() == 1 && (shape.GetDim(0) == 1 || shape.GetDim(0) == ge::UNKNOWN_DIM));
}

ge::graphStatus SetTilingData(gert::TilingContext* context, const ApplyCamePart3TilingData& data)
{
    auto* raw = context->GetRawTilingData();
    OPS_CHECK_NULL_WITH_CONTEXT(context, raw);
    OP_TILING_CHECK(sizeof(data) > raw->GetCapacity(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "ApplyCamePart3 tiling data exceeds capacity"),
                    return ge::GRAPH_FAILED);
    auto ret = memcpy_s(raw->GetData(), raw->GetCapacity(), &data, sizeof(data));
    OP_TILING_CHECK(ret != EOK, VECTOR_INNER_ERR_REPORT_TILIING(context, "ApplyCamePart3 copy tiling data failed"),
                    return ge::GRAPH_FAILED);
    raw->SetDataSize(sizeof(data));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

bool CheckInputParams(const gert::TilingContext* context, const gert::CompileTimeTensorDesc* uDesc,
                      const gert::CompileTimeTensorDesc* mDesc, const gert::Shape& uShape, const gert::Shape& mShape)
{
    OP_TILING_CHECK(
        uShape.GetDimNum() != 2 || mShape.GetDimNum() != 2 || uShape.GetDim(0) <= 0 || uShape.GetDim(1) <= 0 ||
            uShape.GetDim(0) > kMaxSupportedDimension || uShape.GetDim(1) > kMaxSupportedDimension || mShape != uShape,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "u and m must be equal non-empty rank-2 shapes"), return false);
    for (int32_t i = 2; i <= 5; ++i) {
        const auto* shape = context->GetInputShape(i);
        OPS_CHECK_NULL_WITH_CONTEXT(context, shape);
        OP_TILING_CHECK(!IsScalarShape(shape->GetStorageShape()),
                        VECTOR_INNER_ERR_REPORT_TILIING(context, "scalar input must have one element"), return false);
    }
    OP_TILING_CHECK(uDesc->GetDataType() != ge::DT_FLOAT ||
                        (mDesc->GetDataType() != ge::DT_FLOAT && mDesc->GetDataType() != ge::DT_FLOAT16 &&
                         mDesc->GetDataType() != ge::DT_BF16),
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "unsupported ApplyCamePart3 dtype"), return false);
    for (int32_t i = 2; i <= 5; ++i) {
        OP_TILING_CHECK(context->GetInputDesc(i)->GetDataType() != ge::DT_FLOAT,
                        VECTOR_INNER_ERR_REPORT_TILIING(context, "scalar inputs must be float32"), return false);
    }
    return true;
}

bool ParseAttributes(const gert::TilingContext* context, ApplyCamePart3TilingData& data)
{
    const auto* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* useFirstMoment = attrs->GetAttrPointer<bool>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, useFirstMoment);
    data.useFirstMoment = *useFirstMoment ? 1 : 0;
    const auto* globalShape = context->GetOptionalInputShape(kGlobalShapeIndex);
    if (globalShape != nullptr) {
        const auto& globalShapeStorage = globalShape->GetStorageShape();
        OP_TILING_CHECK(globalShapeStorage.GetDimNum() != 1 || globalShapeStorage.GetDim(0) != 2,
                        VECTOR_INNER_ERR_REPORT_TILIING(context, "global_shape must be a 1D tensor with 2 elements"),
                        return false);
        OP_TILING_CHECK(context->GetInputDesc(kGlobalShapeIndex)->GetDataType() != ge::DT_INT64,
                        VECTOR_INNER_ERR_REPORT_TILIING(context, "global_shape must be int64"), return false);
        data.isGlobalShape = 1;
    }
    return true;
}

bool ComputeTileShape(const gert::TilingContext* context, ApplyCamePart3TilingData& data,
                      const gert::CompileTimeTensorDesc* mDesc)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_TILING_CHECK(coreNum <= 0 || ubSize < kVectorBlockBytes,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "invalid Ascend950 platform resources"), return false);

    data.rCoreNum = std::min<int64_t>(data.curN, coreNum);
    data.rNumCalc = CeilAlign(CeilDiv(data.curN, data.rCoreNum), kFp32PerVectorBlock);
    data.rNumCalc = std::min(data.rNumCalc, data.curN);
    data.rCoreNum = CeilDiv(data.curN, data.rNumCalc);
    data.cCoreNum = std::max<int64_t>(1, coreNum / data.rCoreNum);
    data.cNumCalc = CeilAlign(CeilDiv(data.curM, data.cCoreNum), kFp32PerVectorBlock);
    data.cNumCalc = std::min(data.cNumCalc, data.curM);
    data.cCoreNum = CeilDiv(data.curM, data.cNumCalc);
    const int64_t bytesPerElement = mDesc->GetDataType() == ge::DT_FLOAT ? 4 : 2;
    // The first-moment output can alias m. If a row is not cache-line aligned,
    // column cores would write different parts of the same GM cache line and
    // could overwrite each other. Keep each row on one column core.
    if (data.useFirstMoment != 0 && data.curM * bytesPerElement % kGmCacheLineBytes != 0) {
        data.cCoreNum = 1;
        data.cNumCalc = data.curM;
    }
    data.baseN = data.rNumCalc;
    data.baseM = data.cNumCalc;
    const int64_t maxUbElements = static_cast<int64_t>(ubSize) / (5 * kFp32Bytes);
    while (CeilAlign(data.baseM, bytesPerElement == 4 ? 8 : 16) * CeilAlign(data.baseN, bytesPerElement == 4 ? 8 : 16) >
           maxUbElements) {
        if (data.baseM >= data.baseN && data.baseM > kFp32PerVectorBlock) {
            data.baseM = CeilAlign(CeilDiv(data.baseM, 2), kFp32PerVectorBlock);
        } else if (data.baseN > kFp32PerVectorBlock) {
            data.baseN = CeilAlign(CeilDiv(data.baseN, 2), kFp32PerVectorBlock);
        } else {
            break;
        }
    }
    data.usedCoreNum = data.rCoreNum * data.cCoreNum;
    OP_TILING_CHECK(data.usedCoreNum <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "ApplyCamePart3 used core count is zero"), return false);
    return true;
}

bool ComputeWorkspaceSize(const ApplyCamePart3TilingData& data, size_t& workspaceSize)
{
    constexpr size_t kFloatBytes = sizeof(float);
    const size_t maxSize = std::numeric_limits<size_t>::max();
    const auto n = static_cast<size_t>(data.curN);
    const auto m = static_cast<size_t>(data.curM);
    if (m != 0 && n > maxSize / m) {
        return false;
    }
    const size_t elementCount = n * m;
    if (elementCount >
        (maxSize - static_cast<size_t>(kWorkspaceBytes) - static_cast<size_t>(kDetWorkspaceBytes)) / kFloatBytes) {
        return false;
    }
    workspaceSize = static_cast<size_t>(kWorkspaceBytes) + static_cast<size_t>(kDetWorkspaceBytes) +
                    elementCount * kFloatBytes;
    return true;
}
} // namespace

ge::graphStatus TilingApplyCamePart3(gert::TilingContext* context)
{
    ApplyCamePart3TilingData data;
    const auto* uDesc = context->GetInputDesc(0);
    const auto* mDesc = context->GetInputDesc(1);
    const auto* uShapePtr = context->GetInputShape(0);
    const auto* mShapePtr = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, uDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, mDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, uShapePtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, mShapePtr);
    const auto uShape = uShapePtr->GetStorageShape();
    const auto mShape = mShapePtr->GetStorageShape();
    OP_TILING_CHECK(!CheckInputParams(context, uDesc, mDesc, uShape, mShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "ApplyCamePart3 input check failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!ParseAttributes(context, data),
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "ApplyCamePart3 attribute check failed"),
                    return ge::GRAPH_FAILED);

    data.curN = uShape.GetDim(0);
    data.curM = uShape.GetDim(1);
    OP_TILING_CHECK(!ComputeTileShape(context, data, mDesc),
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "ApplyCamePart3 tile shape calculation failed"),
                    return ge::GRAPH_FAILED);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, workspace);
    size_t workspaceSize = 0;
    OP_TILING_CHECK(!ComputeWorkspaceSize(data, workspaceSize),
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "ApplyCamePart3 workspace size overflows"),
                    return ge::GRAPH_FAILED);
    workspace[0] = workspaceSize;
    context->SetBlockDim(data.usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(mDesc->GetDataType()));
    return SetTilingData(context, data);
}

static ge::graphStatus TilingPrepareForApplyCamePart3(gert::TilingParseContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return (platform.GetCoreNumAiv() > 0 && ubSize > 0) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

IMPL_OP_OPTILING(ApplyCamePart3)
    .Tiling(TilingApplyCamePart3)
    .TilingParse<ApplyCamePart3CompileInfo>(TilingPrepareForApplyCamePart3);
} // namespace optiling
