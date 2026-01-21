/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softmax_grad_tiling.h"
#include "util/platform_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace ge;
using namespace AscendC;
using Ops::NN::Optiling::TilingRegistry;
namespace optiling {

std::string SoftmaxGradTilingBase::VectorToString(const std::vector<int64_t>& s)
{
    std::stringstream ss;
    for (auto iter = s.begin(); iter != s.end(); ++iter) {
        ss << *iter;
        if (iter != s.end() - CONST_ONE) {
            ss << ", ";
        }
    }
    return ss.str();
}

std::string SoftmaxGradTilingBase::VectorToString(const int64_t* s, int64_t size)
{
    std::stringstream ss;
    for (int64_t i = 0; i < size; i++) {
        ss << s[i];
        if (i != size - CONST_ONE) {
            ss << ", ";
        }
    }
    return ss.str();
}

ge::graphStatus SoftmaxGradTilingBase::GetAndCheckDtypes()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto xDesc = context_->GetInputDesc(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();

    auto xDesc1 = context_->GetInputDesc(CONST_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc1);
    ge::DataType xDtype1 = xDesc1->GetDataType();

    auto yDesc = context_->GetOutputDesc(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    yDtype_ = yDesc->GetDataType();

    OP_CHECK_IF(
        xDtype_ != yDtype_ || xDtype_ != xDtype1,
        OP_LOGE(
            context_->GetNodeName(), "Input0 dtype [%s], Input1 dtype [%s] and Output dtype [%s] should be same.",
            ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
            ge::TypeUtils::DataTypeToSerialString(xDtype1).c_str(),
            ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_BF16,
        OP_LOGE(
            context_->GetNodeName(),
            "Input dtype is [%s], only support dtype ge::DT_FLOAT16, ge::DT_FLOAT or ge::DT_BF16.",
            ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
        return ge::GRAPH_FAILED);

    if (xDtype_ == ge::DT_FLOAT) {
        xDtypeSize_ = FLOAT32_BYTES;
    } else if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        xDtypeSize_ = FLOAT16_BYTES;
    }

    yDtypeSize_ = xDtypeSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradTilingBase::GetDimsAndCheckShapeValid()
{
    auto xShape = context_->GetInputShape(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::Base::EnsureNotScalar(xShape->GetStorageShape());
    xShapeSize_ = xStorageShape.GetDimNum();

    auto xShape1 = context_->GetInputShape(CONST_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape1);
    auto xStorageShape1 = Ops::Base::EnsureNotScalar(xShape1->GetStorageShape());
    int64_t xShapeSize1 = xStorageShape1.GetDimNum();

    auto yShape = context_->GetOutputShape(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = Ops::Base::EnsureNotScalar(yShape->GetStorageShape());
    int64_t yShapeSize = yStorageShape.GetDimNum();

    OP_CHECK_IF(
        xShapeSize_ != yShapeSize || xShapeSize_ != xShapeSize1,
        OP_LOGE(
            context_->GetNodeName(),
            "Input0 dim size [%ld], Input1 dim size [%ld] and Output dim size [%ld] should be same", xShapeSize_,
            xShapeSize1, yShapeSize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        xShapeSize_ > MAX_DIMS,
        OP_LOGE(context_->GetNodeName(), "Input dim size [%ld] is larger than 8.", xShapeSize_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xShapeSize_ == CONST_ZERO,
        OP_LOGE(context_->GetNodeName(), "Input dim size is zero, not support empty tensor."),
        return ge::GRAPH_FAILED);

    xShape_.resize(xShapeSize_);
    for (int i = 0; i < xShapeSize_; i++) {
        OP_CHECK_IF(
            xStorageShape.GetDim(i) != yStorageShape.GetDim(i) || xStorageShape.GetDim(i) != xStorageShape1.GetDim(i),
            OP_LOGE(
                context_->GetNodeName(),
                "Input0 dim[%d]: %ld, Input1 dim[%d]: %ld and Output dim[%d]: %ld should be same.", i,
                xStorageShape.GetDim(i), i, xStorageShape1.GetDim(i), i, yStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            xStorageShape.GetDim(i) <= CONST_ZERO,
            OP_LOGE(
                context_->GetNodeName(), "Not support input dim[%d]: %ld.", i, xStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
        xShape_[i] = xStorageShape.GetDim(i);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradTilingBase::GetAndCheckAxes()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto axisListPtr = attrs->GetListInt(CONST_ZERO);
    if (axisListPtr == nullptr || axisListPtr->GetSize() == CONST_ZERO) {
        // 默认-1轴reduce
        reduceAxes_ = xShapeSize_ - CONST_ONE;
    } else {
        OP_CHECK_IF(
            axisListPtr->GetSize() != CONST_ONE,
            OP_LOGE(
                context_->GetNodeName(), "Not support input axes size: %zu, which should be 1.",
                axisListPtr->GetSize()),
            return ge::GRAPH_FAILED);
        reduceAxes_ = axisListPtr->GetData()[CONST_ZERO];
        OP_CHECK_IF(
            (reduceAxes_ < -xShapeSize_ || reduceAxes_ > xShapeSize_ - CONST_ONE),
            OP_LOGE(
                context_->GetNodeName(), "Dimension is: %ld, out of range [-%ld, %ld]", reduceAxes_, xShapeSize_,
                xShapeSize_ - CONST_ONE),
            return ge::GRAPH_FAILED);

        reduceAxes_ = reduceAxes_ < CONST_ZERO ? reduceAxes_ + xShapeSize_ : reduceAxes_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("SoftmaxGradTilingBase", "context is nullptr."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAndCheckDtypes() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDimsAndCheckShapeValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAndCheckAxes() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

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
        a1_ = CONST_ONE;
    }

    OP_LOGD(
        context_->GetNodeName(), "inputs original shape is:(%s), axes is:%ld, fused shape is: (%ld, %ld, %ld)\n",
        VectorToString(xShape_).c_str(), reduceAxes_, a1_, r_, a0_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradTilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const SoftmaxGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize_ = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32_ = static_cast<uint64_t>(compileInfo->vlFp32);
    vlFp16_ = static_cast<uint64_t>(compileInfo->vlFp16);

    OP_LOGD(context_->GetNodeName(), "blockSize: %ld, vlFp32: %ld, vlFp16: %ld.", blockSize_, vlFp32_, vlFp16_);

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info.");
        aicoreParams_.blockDim = static_cast<int32_t>(compileInfo->coreNum);
        aicoreParams_.ubSize = static_cast<int64_t>(compileInfo->ubSize);
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from platform.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.blockDim = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizeTemp = CONST_ZERO;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
        aicoreParams_.ubSize = static_cast<int64_t>(ubSizeTemp);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSoftmaxGradAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxGradAscendC enter.");

    auto compileInfoPtr = context->GetCompiledInfo<SoftmaxGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    compileInfoPtr->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfoPtr->vlFp32 = Ops::Base::GetVRegSize(context) / FLOAT32_BYTES;
    compileInfoPtr->vlFp16 = Ops::Base::GetVRegSize(context) / FLOAT16_BYTES;
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (coreNum <= CONST_ZERO),
        OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %u", coreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSizeTemp = CONST_ZERO;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
    OP_CHECK_IF(
        (ubSizeTemp <= CONST_ZERO),
        OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %lu", ubSizeTemp),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSoftmaxGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("SoftmaxGradTilingBase", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "TilingForSoftmaxGrad enter");
    OP_LOGD(context->GetNodeName(), "SoftmaxGradTilingBase Ascendc enter");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForSoftmaxGrad(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForSoftmaxGrad", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxGrad enter.");
    return TilingPrepareForSoftmaxGradAscendC(context);
}

IMPL_OP_OPTILING(SoftmaxGrad)
    .Tiling(TilingForSoftmaxGrad)
    .TilingParse<SoftmaxGradCompileInfo>(TilingPrepareForSoftmaxGrad);

} // namespace optiling
