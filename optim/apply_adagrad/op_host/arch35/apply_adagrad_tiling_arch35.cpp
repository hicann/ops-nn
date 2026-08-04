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
 * \file apply_adagrad_tiling_arch35.cpp
 * \brief ApplyAdagrad arch35 tiling.
 */

#include <algorithm>
#include <cstring>
#include <graph/utils/type_utils.h>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "optim/apply_adagrad/op_kernel/arch35/apply_adagrad_tiling_key.h"
#include "apply_adagrad_tiling_arch35.h"

using namespace ge;
using namespace ApplyAdagradTilingData;

namespace optiling {
namespace {
constexpr size_t WORKSPACE_NUM = 1;
constexpr size_t WORKSPACE_SIZE = 0;
constexpr int32_t VAR_INDEX = 0;
constexpr int32_t ACCUM_INDEX = 1;
constexpr int32_t LR_INDEX = 2;
constexpr int32_t GRAD_INDEX = 3;
constexpr int32_t INPUT_NUM = 4;
constexpr int64_t MIN_BITS_PER_CORE = 32768;
constexpr int64_t BLOCK_ALIGN = 512;
constexpr int64_t UB_RESERVED = 256;
constexpr int64_t UB_ALIGN_BYTES = 256;
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t BUFFER_COUNT_FP32 = 7;
constexpr int64_t BUFFER_COUNT_NATIVE = 5;

const gert::Shape G_VEC_1_SHAPE = {1};

bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool SafeCeilDiv(int64_t value, int64_t factor, int64_t& result)
{
    if (value < 0 || factor <= 0 || value > std::numeric_limits<int64_t>::max() - factor + 1) {
        return false;
    }
    result = (value + factor - 1) / factor;
    return true;
}

bool SafeCeilAlign(int64_t value, int64_t factor, int64_t& result)
{
    int64_t ceilDiv = 0;
    if (!SafeCeilDiv(value, factor, ceilDiv)) {
        return false;
    }
    return CheckedMul(ceilDiv, factor, result);
}

inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    return inShape.IsScalar() ? G_VEC_1_SHAPE : inShape;
}

uint64_t GetDtypeTpl(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16) {
        return static_cast<uint64_t>(APPLY_ADAGRAD_TPL_FP16);
    }
    if (dtype == ge::DT_BF16) {
        return static_cast<uint64_t>(APPLY_ADAGRAD_TPL_BF16);
    }
    return static_cast<uint64_t>(APPLY_ADAGRAD_TPL_FP32);
}

int64_t GetDtypeBytes(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT ? static_cast<int64_t>(sizeof(float)) : static_cast<int64_t>(sizeof(uint16_t));
}
} // namespace

ge::graphStatus ApplyAdagradTiling::SetTilingData()
{
    tilingContext_->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0, updateSlots, dType));
    tilingContext_->SetBlockDim(static_cast<uint32_t>(tiling_->totalElements == 0 ? 1 : blockNum_));

    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradTiling::CheckDtype()
{
    auto varDesc = tilingContext_->GetInputDesc(VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varDesc);
    varDtype_ = varDesc->GetDataType();
    if (varDtype_ != ge::DT_FLOAT16 && varDtype_ != ge::DT_BF16 && varDtype_ != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "var", Ops::Base::ToString(varDtype_).c_str(),
                                  "float16, bfloat16 or float32");
        return ge::GRAPH_FAILED;
    }

    static const char* kInputNames[] = {"var", "accum", "lr", "grad"};
    for (int32_t inputIdx = ACCUM_INDEX; inputIdx < INPUT_NUM; inputIdx++) {
        auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
        auto curDtype = inputDesc->GetDataType();
        if (curDtype != varDtype_) {
            std::string paramNames = std::string(kInputNames[inputIdx]) + " and var";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                tilingContext_->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(varDtype_)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }

    auto outputDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputDesc);
    auto outputDtype = outputDesc->GetDataType();
    if (outputDtype != varDtype_) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            tilingContext_->GetNodeName(), "var(output) and var(input)",
            (Ops::Base::ToString(outputDtype) + " and " + Ops::Base::ToString(varDtype_)).c_str(),
            "Their dtypes should be the same");
        return ge::GRAPH_FAILED;
    }

    dType = GetDtypeTpl(varDtype_);
    return ge::GRAPH_SUCCESS;
}

bool ApplyAdagradTiling::CheckIsScalar(int32_t inputIdx)
{
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    return storageShape.IsScalar() || storageShape.GetShapeSize() == 1;
}

ge::graphStatus ApplyAdagradTiling::CheckShape()
{
    if (!CheckIsScalar(LR_INDEX)) {
        OP_LOGE_FOR_INVALID_SHAPE(tilingContext_->GetNodeName(), "lr", "not scalar", "lr must be a scalar");
        return ge::GRAPH_FAILED;
    }

    auto varStorageShape = tilingContext_->GetInputShape(VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varStorageShape);
    auto accumStorageShape = tilingContext_->GetInputShape(ACCUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, accumStorageShape);
    auto gradStorageShape = tilingContext_->GetInputShape(GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, gradStorageShape);

    const gert::Shape& varShape = EnsureNotScalar(varStorageShape->GetStorageShape());
    const gert::Shape& accumShape = EnsureNotScalar(accumStorageShape->GetStorageShape());
    const gert::Shape& gradShape = EnsureNotScalar(gradStorageShape->GetStorageShape());
    if (varShape != accumShape || varShape != gradShape) {
        std::string varShapeStr = Ops::Base::ToString(varShape);
        std::string accumShapeStr = Ops::Base::ToString(accumShape);
        std::string gradShapeStr = Ops::Base::ToString(gradShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext_->GetNodeName(), "var, accum and grad",
                                               (varShapeStr + ", " + accumShapeStr + " and " + gradShapeStr).c_str(),
                                               "The shapes of var, accum and grad should be the same");
        return ge::GRAPH_FAILED;
    }

    totalElements_ = varShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradTiling::ComputeTiling()
{
    if (InitTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (totalElements_ == 0) {
        tiling_->blockFactor = 0;
        tiling_->ubFactor = 1;
        blockNum_ = 1;
        return ge::GRAPH_SUCCESS;
    }

    uint64_t ubSize = 0;
    auto platform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(),
                                              "AIV core num must be greater than 0"),
        return ge::GRAPH_FAILED);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= UB_RESERVED,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                              "UB size must be greater than reserved bytes"),
        return ge::GRAPH_FAILED);

    if (ComputeBlockTiling(coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ComputeUbTiling(ubSize);
}

ge::graphStatus ApplyAdagradTiling::InitTilingData()
{
    tiling_ = tilingContext_->GetTilingData<ApplyAdagradTilingDataStruct>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, tiling_);
    if (memset_s(tiling_, sizeof(ApplyAdagradTilingDataStruct), 0, sizeof(ApplyAdagradTilingDataStruct)) != EOK) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "tilingData", "memset_s failed",
                                              "Init tiling data failed");
        return ge::GRAPH_FAILED;
    }
    tiling_->totalElements = totalElements_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradTiling::ComputeBlockTiling(int64_t coreNum)
{
    int64_t dtypeBytes = GetDtypeBytes(varDtype_);
    int64_t dtypeBits = dtypeBytes * BITS_PER_BYTE;
    int64_t totalBits = 0;
    OP_CHECK_IF(!CheckedMul(totalElements_, dtypeBits, totalBits),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "totalElements",
                                                      std::to_string(totalElements_).c_str(),
                                                      "totalElements multiply dtype bits overflow"),
                return ge::GRAPH_FAILED);
    int64_t usedCoreNum = 0;
    OP_CHECK_IF(!SafeCeilDiv(totalBits, MIN_BITS_PER_CORE, usedCoreNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "usedCoreNum", "invalid",
                                                      "Calculate used core num failed"),
                return ge::GRAPH_FAILED);
    usedCoreNum = std::max<int64_t>(1, std::min<int64_t>(usedCoreNum, coreNum));
    int64_t elementsPerCore = 0;
    OP_CHECK_IF(!SafeCeilDiv(totalElements_, usedCoreNum, elementsPerCore),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "blockFactor", "invalid",
                                                      "Calculate elements per core failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SafeCeilAlign(elementsPerCore, BLOCK_ALIGN, tiling_->blockFactor),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "blockFactor", "invalid",
                                                      "Calculate block factor failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SafeCeilDiv(totalElements_, tiling_->blockFactor, blockNum_),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "blockNum", "invalid",
                                                      "Calculate block num failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradTiling::ComputeUbTiling(uint64_t ubSize)
{
    int64_t dtypeBytes = GetDtypeBytes(varDtype_);
    int64_t bufferBytes = varDtype_ == ge::DT_FLOAT ? BUFFER_COUNT_FP32 * static_cast<int64_t>(sizeof(float)) :
                                                      BUFFER_COUNT_NATIVE * dtypeBytes +
                                                          BUFFER_COUNT_FP32 * static_cast<int64_t>(sizeof(float));
    OP_CHECK_IF(ubSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "ubSize",
                                                      std::to_string(ubSize).c_str(), "UB size overflow"),
                return ge::GRAPH_FAILED);
    int64_t maxElementNum = (static_cast<int64_t>(ubSize) - UB_RESERVED) / bufferBytes;
    int64_t alignFactor = UB_ALIGN_BYTES / dtypeBytes;
    OP_CHECK_IF(
        maxElementNum < alignFactor,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                              "UB size is insufficient for one aligned tile"),
        return ge::GRAPH_FAILED);
    tiling_->ubFactor = std::max<int64_t>(alignFactor, (maxElementNum / alignFactor) * alignFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradTiling::RunTiling()
{
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = tilingContext_->GetAttrs();
    const bool* updateSlotsAttr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(0);
    updateSlots = static_cast<uint64_t>((updateSlotsAttr == nullptr || *updateSlotsAttr) ? UPDATE_SLOTS_TPL_TRUE :
                                                                                           UPDATE_SLOTS_TPL_FALSE);

    if (ComputeTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return SetTilingData();
}

static ge::graphStatus TilingForApplyAdagrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ApplyAdagradTiling", "context", "nullptr", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    ApplyAdagradTiling tiling(context);
    return tiling.RunTiling();
}

static ge::graphStatus TilingPrepareForApplyAdagrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdagrad)
    .Tiling(TilingForApplyAdagrad)
    .TilingParse<ApplyAdagradCompileInfo>(TilingPrepareForApplyAdagrad);
} // namespace optiling
