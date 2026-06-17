/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_tiling_base.cc
 * \brief
 */

#include "softmax_grad_ext_tiling.h"
#include <nlohmann/json.hpp>
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace AscendC;
using namespace Ops::NN::OpTiling;
using namespace ge;

namespace optiling {
inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context)
{
    auto json_str = context->GetCompiledJson();
    OP_CHECK_IF(json_str == nullptr, OP_LOGE(context->GetNodeName(), "json_str is nullptr!"), return nullptr);
    std::unique_ptr<nlohmann::json> parsed_object_cinfo =
        std::make_unique<nlohmann::json>(nlohmann::json::parse(json_str));
    return parsed_object_cinfo;
}

std::string SoftmaxGradExtTilingBase::VectorToString(const std::vector<int64_t>& s)
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

std::string SoftmaxGradExtTilingBase::VectorToString(const int64_t* s, int64_t size)
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

ge::graphStatus SoftmaxGradExtTilingBase::GetAndCheckDtypes()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto xDesc = context_->GetInputDesc(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();

    auto xDesc1 = context_->GetInputDesc(CONST_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc1);
    ge::DataType xDtype1 = xDesc1->GetDataType();

    auto xDesc2 = context_->GetInputDesc(CONST_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc2);
    ge::DataType xDtype2 = xDesc2->GetDataType();

    auto yDesc = context_->GetOutputDesc(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    yDtype_ = yDesc->GetDataType();

    OP_TILING_CHECK(
        xDtype_ != yDtype_ || xDtype_ != xDtype1 || xDtype_ != xDtype2,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(),
            "Input0 dtype [%s], Input1 dtype [%s], InInput2 dtype [%s] and Output dtype [%s] should be same.",
            ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
            ge::TypeUtils::DataTypeToSerialString(xDtype1).c_str(),
            ge::TypeUtils::DataTypeToSerialString(xDtype2).c_str(),
            ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_BF16,
        VECTOR_INNER_ERR_REPORT_TILIING(
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

ge::graphStatus SoftmaxGradExtTilingBase::GetDimsAndCheckShapeValid()
{
    auto xShape = context_->GetInputShape(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::Base::EnsureNotScalar(xShape->GetStorageShape());
    xShapeSize_ = xStorageShape.GetDimNum();

    auto xShape1 = context_->GetInputShape(CONST_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape1);
    auto xStorageShape1 = Ops::Base::EnsureNotScalar(xShape1->GetStorageShape());

    auto xShape2 = context_->GetInputShape(CONST_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape2);
    auto xStorageShape2 = Ops::Base::EnsureNotScalar(xShape2->GetStorageShape());
    xShapeSize2_ = xStorageShape2.GetDimNum();

    auto yShape = context_->GetOutputShape(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = Ops::Base::EnsureNotScalar(yShape->GetStorageShape());

    OP_TILING_CHECK(
        xShapeSize_ > MAX_DIMS, // 超过支持的最大维度数
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Input dim size [%ld] is larger than 6.", xShapeSize_), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xShapeSize_ == CONST_ZERO, // 检查输入的维度是否为0
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Input dim size is zero, not support empty tensor."), return ge::GRAPH_FAILED);

    xShape_.resize(xShapeSize_); // 将xshape_大小调整为xShapeSize
    if (xShapeSize2_ == 1 && xStorageShape2.GetDim(0) == 1) {
        for (int i = 0; i < xShapeSize_; i++) {
            OP_TILING_CHECK(
                xStorageShape.GetDim(i) != yStorageShape.GetDim(i) ||
                    xStorageShape.GetDim(i) != xStorageShape1.GetDim(i),
                VECTOR_INNER_ERR_REPORT_TILIING(
                    context_->GetNodeName(),
                    "Input0 dim[%d]: %ld, Input1 dim[%d]: %ld and Output dim[%d]: %ld should be "
                    "same. Input2 dim[0]: %ld. ",
                    i, xStorageShape.GetDim(i), i, xStorageShape1.GetDim(i), i, yStorageShape.GetDim(i),
                    xStorageShape2.GetDim(0)),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                xStorageShape.GetDim(i) <= CONST_ZERO,
                VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Not support input dim[%d]: %ld.", i, xStorageShape.GetDim(i)),
                return ge::GRAPH_FAILED);
            xShape_[i] = xStorageShape.GetDim(i);
        }
    } else {
        isX2Scalar_ = false;
        for (int i = 0; i < xShapeSize_; i++) {
            OP_TILING_CHECK(
                xStorageShape.GetDim(i) != yStorageShape.GetDim(i) ||
                    xStorageShape.GetDim(i) != xStorageShape1.GetDim(i) ||
                    xStorageShape.GetDim(i) != xStorageShape2.GetDim(i),
                VECTOR_INNER_ERR_REPORT_TILIING(
                    context_->GetNodeName(),
                    "Input0 dim[%d]: %ld, Input1 dim[%d]: %ld, Input2 dim[%d]: %ld and Output dim[%d]: %ld should be "
                    "same.",
                    i, xStorageShape.GetDim(i), i, xStorageShape1.GetDim(i), i, xStorageShape2.GetDim(i), i,
                    yStorageShape.GetDim(i)),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                xStorageShape.GetDim(i) <= CONST_ZERO,
                VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Not support input dim[%d]: %ld.", i, xStorageShape.GetDim(i)),
                return ge::GRAPH_FAILED);
            xShape_[i] = xStorageShape.GetDim(i);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradExtTilingBase::GetAndCheckAxes()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto attrAxis = attrs->GetAttrPointer<int64_t>(CONST_ZERO); // 默认-1轴reduce
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAxis);
    reduceAxes_ = xShapeSize_ - CONST_ONE;
    OP_TILING_CHECK(
        (*attrAxis != -1 && *attrAxis != reduceAxes_),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Dimension is: %ld, axes only support -1 or %ld", *attrAxis, reduceAxes_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradExtTilingBase::GetShapeAttrsInfo()
{
    OP_TILING_CHECK(
        context_ == nullptr, OP_LOGE("SoftmaxGradExtTilingBase", "context is nullptr."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(GetAndCheckDtypes() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetDimsAndCheckShapeValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetAndCheckAxes() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

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

    OP_LOGD(
        context_->GetNodeName(), "inputs original shape is:(%s), axes is:%ld, fused shape is: (%ld, %ld, %ld)\n",
        VectorToString(xShape_).c_str(), reduceAxes_, a1_, r_, a0_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradExtTilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const SoftmaxGradExtCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize_ = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32_ = static_cast<uint64_t>(compileInfo->vlFp32);
    vlFp16_ = static_cast<uint64_t>(compileInfo->vlFp16);

    OP_LOGD(context_->GetNodeName(), "blockSize: %ld, vlFp32: %ld, vlFp16: %ld.", blockSize_, vlFp32_, vlFp16_);

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info."); // 说明是离线编译
        aicoreParams_.blockDim = static_cast<int32_t>(compileInfo->coreNum);
        aicoreParams_.ubSize = static_cast<int64_t>(compileInfo->ubSize);
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from platform."); // 说明是在线运行环境
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.blockDim = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizeTemp = CONST_ZERO;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
        aicoreParams_.ubSize = static_cast<int64_t>(ubSizeTemp);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftmaxGradExtDSLTiling(
    gert::TilingContext* context, const SoftmaxGradExtCompileInfo* compileInfo)
{
    // get input
    uint64_t inputNums = context->GetComputeNodeInputNum(); // 获取输入数量
    OP_TILING_CHECK(
        (inputNums < CONST_ONE), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "inputNums is: %ld, less than one", inputNums),
        return ge::GRAPH_FAILED);
    std::vector<gert::Shape> inputShapes(inputNums);
    for (uint64_t i = 0; i < inputNums; i++) {
        auto inputShape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
        inputShapes[i] = Ops::Base::EnsureNotScalar(inputShape->GetStorageShape());
    }
    auto desc = context->GetInputDesc(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    DataType dataType = desc->GetDataType();

    // get attr for reduce axis
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    std::vector<int64_t> axis;
    int64_t reduceAxis = CONST_ZERO;
    auto axisListPtr = attrs->GetListInt(CONST_ZERO);
    int64_t xShapeSize = inputShapes[CONST_ZERO].GetDimNum(); // 获取输入张量的维度数
    if (axisListPtr == nullptr || axisListPtr->GetSize() == CONST_ZERO) {
        reduceAxis = xShapeSize - CONST_ONE;
        axis.emplace_back(reduceAxis);
    } else {
        for (size_t i = 0; i < axisListPtr->GetSize(); i++) {
            reduceAxis = axisListPtr->GetData()[i];
            OP_TILING_CHECK(
                (reduceAxis < -xShapeSize || reduceAxis > xShapeSize - CONST_ONE),
                VECTOR_INNER_ERR_REPORT_TILIING(
                    context->GetNodeName(), "Dimension is: %ld, out of range [-%ld, %ld]", reduceAxis, xShapeSize,
                    xShapeSize - CONST_ONE),
                return ge::GRAPH_FAILED);

            reduceAxis = reduceAxis < CONST_ZERO ? reduceAxis + xShapeSize : reduceAxis;
            if (std::find(axis.begin(), axis.end(), reduceAxis) == axis.end()) {
                axis.emplace_back(reduceAxis);
            }
        }
    }
    axis.emplace_back(reduceAxis);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSoftmaxGradExtAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxGradExtAscendC enter.");

    auto compileInfoPtr = context->GetCompiledInfo<SoftmaxGradExtCompileInfo>(); // 创建指针获取编译信息
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    compileInfoPtr->blockSize =
        Ops::Base::GetUbBlockSize(context); // 设置blocksize为平台提供的通用缓冲区对齐大小=32U(4字节)
    compileInfoPtr->vlFp32 = Ops::Base::GetVRegSize(context) /
                             FLOAT32_BYTES; // 设置vlfp32为平台提供的浮点数寄存器大小（以float32单位计算）=64
    compileInfoPtr->vlFp16 = Ops::Base::GetVRegSize(context) /
                             FLOAT16_BYTES; // 设置vlfp16为平台提供的浮点数寄存器大小（以float16单位计算）=128

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr); // 创建platformAscendC对象,获取硬件信息
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(
        (compileInfoPtr->coreNum <= CONST_ZERO),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "Get core num failed, core num: %u",
            static_cast<uint32_t>(compileInfoPtr->coreNum)),
        return ge::GRAPH_FAILED);
    uint64_t ubSizeTemp = CONST_ZERO;                                              // 初始化临时变量
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp); // 获取每个核心的通用缓冲区大小
    compileInfoPtr->ubSize = static_cast<int64_t>(ubSizeTemp);
    OP_TILING_CHECK(
        (compileInfoPtr->ubSize <= CONST_ZERO),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfoPtr->ubSize)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSoftmaxGradExtDsl(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SoftmaxGradExtCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");
    std::unique_ptr<nlohmann::json> parsedObjectCInfo = GetCompileInfoJson(context); // 获取json格式的编译信息
    OP_CHECK_NULL_WITH_CONTEXT(context, parsedObjectCInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSoftmaxGradExt(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("SoftmaxGradExtTilingBase", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "TilingForSoftmaxGradExt enter");
    auto compileInfo = reinterpret_cast<const SoftmaxGradExtCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    if (compileInfo->isAscendC) {
        OP_LOGD(context->GetNodeName(), "SoftmaxGradExtTilingBase Ascendc enter");
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    OP_LOGD(context->GetNodeName(), "SoftmaxGradExtDSLTiling enter");
    return SoftmaxGradExtDSLTiling(context, compileInfo);
}

ge::graphStatus TilingPrepareForSoftmaxGradExt(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForSoftmaxGradExt", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxGradExt enter.");

    auto compileInfoPtr = context->GetCompiledInfo<SoftmaxGradExtCompileInfo>();
    OP_CHECK_IF(
        (compileInfoPtr == nullptr), OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);
    compileInfoPtr->isAscendC = IsRegbaseSocVersion(context);
    if (compileInfoPtr->isAscendC) {
        OP_LOGD(context, "TilingPrepareForSoftmaxGradExtAscendC enter");
        return TilingPrepareForSoftmaxGradExtAscendC(context);
    }
    OP_LOGD(context, "TilingPrepareForSoftmaxGradExtDsl enter");
    return TilingPrepareForSoftmaxGradExtDsl(context);
}

IMPL_OP_OPTILING(SoftmaxGradExt)
    .Tiling(TilingForSoftmaxGradExt)
    .TilingParse<SoftmaxGradExtCompileInfo>(TilingPrepareForSoftmaxGradExt);

} // namespace optiling