/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file softmax_v2_base_tiling.cpp
 * \brief
 */

#include "softmax_v2_tiling.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log/log.h"
#include <iostream>
#include "op_common/op_host/util/platform_util.h"
#include "register/op_impl_registry.h"

using namespace Ops::Base;
using namespace AscendC;

namespace optiling
{

const gert::Shape g_vec_1_shape = {1};
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape,
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

std::string SoftmaxV2TilingBase::VectorToString(const std::vector<int64_t>& s)
{
    std::stringstream ss;
    for (auto iter = s.begin(); iter != s.end(); ++iter) {
        ss << *iter;
        if (iter != s.end() - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

std::string SoftmaxV2TilingBase::VectorToString(const int64_t* s, int64_t size)
{
    std::stringstream ss;
    for (int64_t i = 0; i < size; i++) {
        ss << s[i];
        if (i != size - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

ge::graphStatus SoftmaxV2TilingBase::GetAndCheckDtypes()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const bool* halfToFloat = attrs->GetBool(1);
    halfToFloat_ = (halfToFloat == nullptr) ? false : *halfToFloat;

    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    yDtype_ = yDesc->GetDataType();

    if (!halfToFloat_) {
        if (xDtype_ != yDtype_) {
            std::string dtypeMsg = ToString(xDtype_) + " and " + ToString(yDtype_);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "x and y", dtypeMsg.c_str(),
                "The dtypes of input x and output y should be the same when attribute half_to_float is false");
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_BF16,
                        OP_LOGE_FOR_INVALID_DTYPE(
                            context_->GetNodeName(), "x", ToString(xDtype_).c_str(), "FLOAT, FLOAT16 or BF16"),
                        return ge::GRAPH_FAILED);
    } else {
        if (xDtype_ != ge::DT_FLOAT16 || yDtype_ != ge::DT_FLOAT) {
            std::string dtypeMsg = ToString(xDtype_) + " and " + ToString(yDtype_);
            std::string reasonMsg = "The dtype of input x should be FLOAT16 and"
                " the dtype of output y should be FLOAT when attribute half_to_float is true";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "x and y", dtypeMsg.c_str(),
                reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (xDtype_ == ge::DT_FLOAT) {
        xDtypeSize_ = FLOAT32_BYTES;
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        xDtypeSize_ = FLOAT16_BYTES;
    }

    if (yDtype_ == ge::DT_FLOAT) {
        yDtypeSize_ = FLOAT32_BYTES;
    } else if (yDtype_ == ge::DT_FLOAT16 || yDtype_ == ge::DT_BF16) {
        yDtypeSize_ = FLOAT16_BYTES;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxV2TilingBase::GetDimsAndCheckShapeValid()
{
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = EnsureNotScalar(xShape->GetStorageShape());
    xShapeSize_ = xStorageShape.GetDimNum();

    auto yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = EnsureNotScalar(yShape->GetStorageShape());
    int64_t yShapeSize = yStorageShape.GetDimNum();

    if (xShapeSize_ != yShapeSize) {
        std::string dimsMsg = std::to_string(xShapeSize_) + " and " + std::to_string(yShapeSize);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "x and y", dimsMsg.c_str(),
            "The shape dims of input x and output y should be the same");
        return ge::GRAPH_FAILED;
    }

    if (xShapeSize_ > MAX_DIMS) {
        std::string correctMsg = "less than or equal to " + std::to_string(MAX_DIMS);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
            std::to_string(xShapeSize_).c_str(), correctMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        xShapeSize_ == 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", "0",
            "The number of dimensions of input x can not be 0, because empty tensor is not supported"),
        return ge::GRAPH_FAILED);
    xShape_.resize(xShapeSize_);
    for (int i = 0; i < xShapeSize_; i++) {
        if (xStorageShape.GetDim(i) != yStorageShape.GetDim(i)) {
            std::string shapesMsg = ToString(xStorageShape) + " and " + ToString(yStorageShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "x and y", shapesMsg.c_str(),
                "The shapes of input x and output y should be the same");
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(xStorageShape.GetDim(i) <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "x", ToString(xStorageShape).c_str(),
                "The shape of input x can not be an empty tensor or an invalid tensor with a negative dim"),
            return ge::GRAPH_FAILED);
        xShape_[i] = xStorageShape.GetDim(i);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxV2TilingBase::GetAndCheckAxes()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto axisListPtr = attrs->GetListInt(0);
    if (axisListPtr == nullptr || axisListPtr->GetSize() == 0) {
        // 默认-1轴reduce
        reduceAxes_ = xShapeSize_ - 1;
    } else {
        OP_CHECK_IF(axisListPtr->GetSize() != 1,
                    OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(),
                        "axes", std::to_string(axisListPtr->GetSize()).c_str(), "1"),
                        return ge::GRAPH_FAILED);
        reduceAxes_ = axisListPtr->GetData()[0];
        if (reduceAxes_ < -xShapeSize_ || reduceAxes_ > xShapeSize_ - 1) {
            std::string reasonMsg = "The value of attr axes should be in range [-" + std::to_string(xShapeSize_) + ", "
                + std::to_string(xShapeSize_) + ")";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "axes", std::to_string(reduceAxes_).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        reduceAxes_ = reduceAxes_ < 0 ? reduceAxes_ + xShapeSize_ : reduceAxes_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxV2TilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("SoftmaxV2TilingBase", "context is nullptr."),
                        return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAndCheckDtypes() != ge::GRAPH_SUCCESS, ,
                        return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDimsAndCheckShapeValid() != ge::GRAPH_SUCCESS, ,
                        return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAndCheckAxes() != ge::GRAPH_SUCCESS, ,
                        return ge::GRAPH_FAILED);

    // 合轴(a1_, r_, a0_)
    a1_ = DIM_NUM_ONE;
    r_ = xShape_[reduceAxes_];
    a0_ = DIM_NUM_ONE;
    for (int i = 0; i < xShapeSize_; i++) {
        if (i < reduceAxes_) {
            a1_ *= xShape_[i];
        } else if (i > reduceAxes_) {
            a0_ *= xShape_[i];
        }
    }

    if (r_ == DIM_NUM_ONE) {
        a0_ = a1_ * a0_;
        a1_ = 1;
    }

    OP_LOGD(context_->GetNodeName(), "Input original shape is:(%s), axes is:%ld, fused shape is: (%ld, %ld, %ld)\n",
            VectorToString(xShape_).c_str(), reduceAxes_, a1_, r_, a0_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxV2TilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const SoftmaxV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize_ = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32_ = static_cast<uint64_t>(compileInfo->vlFp32);
    vlFp16_ = static_cast<uint64_t>(compileInfo->vlFp16);

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info.");
        aicoreParams_.numBlocks = static_cast<int32_t>(compileInfo->coreNum);
        aicoreParams_.ubSize = static_cast<int64_t>(compileInfo->ubSize);
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from platform.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.numBlocks = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizeTemp = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
        aicoreParams_.ubSize = static_cast<int64_t>(ubSizeTemp);
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus TilingPrepareForSoftmaxV2AscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxV2AscendC enter.");

    auto compileInfoPtr = context->GetCompiledInfo<SoftmaxV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    compileInfoPtr->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfoPtr->vlFp32 = Ops::Base::GetVRegSize(context) / FLOAT32_BYTES;
    compileInfoPtr->vlFp16 = Ops::Base::GetVRegSize(context) / FLOAT16_BYTES;

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfoPtr->coreNum <= 0),
                    OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %u",
                    static_cast<uint32_t>(compileInfoPtr->coreNum)),
                    return ge::GRAPH_FAILED);
    uint64_t ubSizeTemp = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
    compileInfoPtr->ubSize = static_cast<int64_t>(ubSizeTemp);
    OP_CHECK_IF((compileInfoPtr->ubSize <= 0),
                    OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u",
                    static_cast<uint32_t>(compileInfoPtr->ubSize)),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus TilingForSoftmaxV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("SoftmaxV2TilingBase", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "TilingForSoftmaxV2 enter");
    auto compileInfo = reinterpret_cast<const SoftmaxV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "SoftmaxV2TilingBase Ascendc enter");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForSoftmaxV2(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForSoftmaxV2", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxV2 enter.");

    auto compileInfoPtr = context->GetCompiledInfo<SoftmaxV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepareForSoftmaxV2AscendC enter");
    return TilingPrepareForSoftmaxV2AscendC(context);
}

IMPL_OP_OPTILING(SoftmaxV2).Tiling(TilingForSoftmaxV2).TilingParse<SoftmaxV2CompileInfo>(TilingPrepareForSoftmaxV2);

}  // namespace optiling
