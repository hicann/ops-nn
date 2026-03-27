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
 * \file deformable_offsets_grad_tiling_arch35.cpp
 * \brief deformable_offsets_grad_tiling_arch35 impl info
 */
#include "deformable_offsets_grad_tiling_arch35.h"
#include <cstdint>
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"

using namespace std;

namespace optiling {
constexpr int32_t INPUT_GRAD_INDEX = 0;
constexpr int32_t INPUT_X_INDEX = 1;
constexpr int32_t INPUT_OFFSET_INDEX = 2;
constexpr int32_t OUTPUT_GRAD_X_INDEX = 0;
constexpr int32_t OUTPUT_GRAD_OFFSETS_INDEX = 1;
constexpr int32_t DIM_NUM_4D = 4;
constexpr int32_t DIM_NUM_2D = 2;
constexpr int32_t STRIDES_INDEX = 0;
constexpr int32_t PADS_INDEX = 1;
constexpr int32_t KSIZE_INDEX = 2;
constexpr int32_t DILATIONS_INDEX = 3;
constexpr int32_t DATA_FORMAT_INDEX = 4;
constexpr int32_t DEFORMABLE_GROUPS_ATTR_INDEX = 5;
constexpr int32_t MODULATE_INDEX = 6;
constexpr int32_t LIST_INDEX_0 = 0;
constexpr int32_t LIST_INDEX_1 = 1;
constexpr int32_t LIST_INDEX_2 = 2;
constexpr int32_t LIST_INDEX_3 = 3;
constexpr int64_t POINT_WEIGHT_SIZE = 3;
constexpr int64_t POINT_NOT_WEIGHT_SIZE = 2;
constexpr int64_t MAX_INT32 = 2147483647; // INT32最大值
constexpr int64_t MAX_CORE_NUM = 64;
constexpr int64_t VF_MAX_THREAD_NUM = 256;
constexpr int64_t EXCEPTED_KERNEL_SIZE = 2;
constexpr int64_t SYS_WORK_SPACE_SIZE = static_cast<int64_t>(16 * 1024 * 1024);
constexpr int64_t SCH_KEY = 1000; // 基础tilingkey为1000

class DeformableOffsetsGradAscendCTilingImpl {
public:
    explicit DeformableOffsetsGradAscendCTilingImpl(gert::TilingContext* context) : context_(context){};

    ge::graphStatus Init(const int64_t coreNum);
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetinputInfoAndCheck();
    ge::graphStatus GetAttrInfoAndCheck();
    ge::graphStatus UpdateStrideAndDilationByFormat(
        gert::TilingContext* context, const string dataFormat, const int64_t* stridesData, const int64_t* dilatesData);
    ge::graphStatus GetOutputInfoAndCheck(DeformableOffsetsGradOffset& deformableOffsetsGradOffset);
    ge::graphStatus GetParameterAndCheck(const DeformableOffsetsGradOffset deformableOffsetsGradOffset) const;
    void FillTilingData();
    void PrintTilingData();

private:
    int64_t strideH_ = 0;
    int64_t strideW_ = 0;
    int64_t dilationH_ = 0;
    int64_t dilationW_ = 0;
    int64_t padsHeightUp_ = 0;
    int64_t padsHeightDown_ = 0;
    int64_t padsWidthLeft_ = 0;
    int64_t padsWidthRight_ = 0;
    int64_t dimKh_ = 0;
    int64_t dimKw_ = 0;
    int64_t gradSize_ = 0;
    int64_t xSize_ = 0;
    int64_t offsetSize_ = 0;
    int64_t gradBatchNum_ = 0;
    int64_t gradWidth_ = 0;
    int64_t gradHeight_ = 0;
    int64_t imgBatchNum_ = 0;
    int64_t imgChannel_ = 0;
    int64_t imgWidth_ = 0;
    int64_t imgHeight_ = 0;

    int64_t imgOutBatchNum_ = 0;
    int64_t imgOutHeight_ = 0;
    int64_t imgOutWidth_ = 0;
    int64_t offsetBatchNum_ = 0;
    int64_t offsetWidth_ = 0;
    int64_t offsetHeight_ = 0;
    int64_t offsetChannel_ = 0;
    int64_t offsetValueDim_ = 0;
    int64_t deformableGroups_ = 0;
    int64_t deformableGroupsAttr_ = 0;

    int64_t numKernels_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t realUseCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;

    int64_t outputGradXSize_ = 0;
    int64_t clearGradXCoreNum_ = 0;
    int64_t gradXFactor_ = 0;
    int64_t gradXFactorTail_ = 0;

    int64_t outputGradOffsetsSize_ = 0;
    int64_t clearGradOffsetsCoreNum_ = 0;
    int64_t gradOffsetsFactor_ = 0;
    int64_t gradOffsetsFactorTail_ = 0;
    int64_t maxCoreNum_ = 0;
    int64_t dataTypeKey_ = SCH_KEY;
    bool isModulated_ = false;

    gert::TilingContext* context_ = nullptr;
    DeformableOffsetsGradTilingData tilingData_;
};

ge::graphStatus DeformableOffsetsGradAscendCTilingImpl::Init(const int64_t coreNum)
{
    OP_LOGD(context_->GetNodeName(), "DeformableOffsetsGradTiling start");
    maxCoreNum_ = static_cast<int64_t>(coreNum);
    DeformableOffsetsGradOffset deformableOffsetsGradOffset;
    // input data Format
    auto primaryFormatPtr = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, primaryFormatPtr);
    auto inputXDataFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(primaryFormatPtr->GetStorageFormat()));
    OP_CHECK_IF(
        inputXDataFormat != ge::FORMAT_NHWC, OP_LOGE(context_->GetNodeName(), "inputX format only support [NHWC]"),
        return ge::GRAPH_FAILED);
    auto inputXDataTypePtr = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDataTypePtr);
    OP_CHECK_IF(
        GetinputInfoAndCheck() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "input info get failed or check failed."), return ge::GRAPH_FAILED);
    if (gradSize_ > MAX_INT32 || xSize_ > MAX_INT32 || offsetSize_ > MAX_INT32) {
        dataTypeKey_ = SCH_KEY + 1;
    }
    OP_CHECK_IF(
        GetOutputInfoAndCheck(deformableOffsetsGradOffset) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "output info get failed or check failed."), return ge::GRAPH_FAILED);

    ge::Format inputOffsetDataFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(INPUT_OFFSET_INDEX)->GetStorageFormat()));
    OP_CHECK_IF(
        inputOffsetDataFormat != ge::FORMAT_NHWC,
        OP_LOGE(context_->GetNodeName(), "input offsets data format is not supported."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetAttrInfoAndCheck() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "attribute value get fail or check failed."), return ge::GRAPH_FAILED);

    offsetValueDim_ = isModulated_ ? POINT_WEIGHT_SIZE : POINT_NOT_WEIGHT_SIZE;
    deformableGroups_ = offsetChannel_ / (dimKh_ * dimKw_ * offsetValueDim_);
    OP_CHECK_IF(
        strideH_ == 0 || strideW_ == 0, OP_LOGE(context_->GetNodeName(), "StrideH or StrideW must not equal to 0"),
        return ge::GRAPH_FAILED);
    // conved img
    imgOutHeight_ = static_cast<int64_t>(
        (imgHeight_ + padsHeightUp_ + padsHeightDown_ - (dilationH_ * (dimKh_ - 1) + 1)) / strideH_ + 1);
    imgOutWidth_ = static_cast<int64_t>(
        (imgWidth_ + padsWidthLeft_ + padsWidthRight_ - (dilationW_ * (dimKw_ - 1) + 1)) / strideW_ + 1);

    OP_CHECK_IF(
        GetParameterAndCheck(deformableOffsetsGradOffset) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "The parameter info get fail or check fail."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "Exit DeformableOffsetsGradAscendCTilingImpl init.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableOffsetsGradAscendCTilingImpl::GetinputInfoAndCheck()
{
    // get input grad shape info
    auto inputGrad = context_->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_IF(
        inputGrad == nullptr, OP_LOGE(context_->GetNodeName(), "Get input grad failed"), return ge::GRAPH_FAILED);
    const gert::Shape& inputGradShape = inputGrad->GetStorageShape();
    int64_t inputGradShapeDims = inputGradShape.GetDimNum();
    // get input x shape info
    auto inputX = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_IF(inputX == nullptr, OP_LOGE(context_->GetNodeName(), "Get input x failed"), return ge::GRAPH_FAILED);
    const gert::Shape& inputXShape = inputX->GetStorageShape();
    int64_t inputXShapeDims = inputXShape.GetDimNum();
    // get input offsets shape info
    auto inputDescPtr = context_->GetInputDesc(INPUT_OFFSET_INDEX);
    OP_CHECK_IF(
        inputDescPtr == nullptr, OP_LOGE(context_->GetNodeName(), "Get input offsets failed"), return ge::GRAPH_FAILED);
    auto offsetsDType = inputDescPtr->GetDataType();
    OP_CHECK_IF(
        (offsetsDType != ge::DT_FLOAT) && (offsetsDType != ge::DT_FLOAT16) && (offsetsDType != ge::DT_BF16),
        OP_LOGE(context_->GetNodeName(), "offsets not support data type"), return ge::GRAPH_FAILED);
    auto inputOffset = context_->GetInputShape(INPUT_OFFSET_INDEX);
    OP_CHECK_IF(
        inputOffset == nullptr, OP_LOGE(context_->GetNodeName(), "Get input offsets failed"), return ge::GRAPH_FAILED);
    const gert::Shape& inputOffsetShape = inputOffset->GetStorageShape();
    int64_t inputOffsetShapeDims = inputOffsetShape.GetDimNum();
    // check shape and format
    OP_CHECK_IF(
        inputXShapeDims != static_cast<int64_t>(DIM_NUM_4D) ||
            inputOffsetShapeDims != static_cast<int64_t>(DIM_NUM_4D) ||
            inputGradShapeDims != static_cast<int64_t>(DIM_NUM_4D),
        OP_LOGE(context_->GetNodeName(), "grad or inputX or offsets shape check failed"), return ge::GRAPH_FAILED);
    gradSize_ = inputGradShape.GetShapeSize();
    xSize_ = inputXShape.GetShapeSize();
    offsetSize_ = inputOffsetShape.GetShapeSize();
    OP_CHECK_IF(
        gradSize_ == 0 || xSize_ == 0 || offsetSize_ == 0,
        OP_LOGE(context_->GetNodeName(), "grad, inputX, offsets not support empty tensor."), return ge::GRAPH_FAILED);
    // input x info
    gradBatchNum_ = inputGradShape.GetDim(LIST_INDEX_0);
    gradHeight_ = inputGradShape.GetDim(LIST_INDEX_1);
    gradWidth_ = inputGradShape.GetDim(LIST_INDEX_2);
    // input x info
    imgBatchNum_ = inputXShape.GetDim(LIST_INDEX_0);
    imgChannel_ = inputXShape.GetDim(LIST_INDEX_3);
    imgWidth_ = inputXShape.GetDim(LIST_INDEX_2);
    imgHeight_ = inputXShape.GetDim(LIST_INDEX_1);
    // input offsets info
    offsetBatchNum_ = inputOffsetShape.GetDim(LIST_INDEX_0);
    offsetHeight_ = inputOffsetShape.GetDim(LIST_INDEX_1);
    offsetWidth_ = inputOffsetShape.GetDim(LIST_INDEX_2);
    offsetChannel_ = inputOffsetShape.GetDim(LIST_INDEX_3);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableOffsetsGradAscendCTilingImpl::GetOutputInfoAndCheck(
    DeformableOffsetsGradOffset& deformableOffsetsGradOffset)
{
    // get output grad_x grad_offsets shape info
    auto outputGradXDescPtr = context_->GetOutputDesc(OUTPUT_GRAD_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputGradXDescPtr);
    auto outputGradXDataType = outputGradXDescPtr->GetDataType();
    auto outputGradOffsetDescPtr = context_->GetOutputDesc(OUTPUT_GRAD_OFFSETS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputGradOffsetDescPtr);
    auto outGradOffsetsType = outputGradOffsetDescPtr->GetDataType();
    auto outputGradOffsetsShapeInfoPtr = context_->GetOutputShape(OUTPUT_GRAD_OFFSETS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputGradOffsetsShapeInfoPtr);
    auto outputGradOffsetsShapeInfo = outputGradOffsetsShapeInfoPtr->GetStorageShape();
    OP_CHECK_IF(
        (outGradOffsetsType != ge::DT_FLOAT) &&
            (outGradOffsetsType != ge::DT_FLOAT16) && (outGradOffsetsType != ge::DT_BF16),
        OP_LOGE(context_->GetNodeName(), "Not support output data type"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outGradOffsetsType != outputGradXDataType,
        OP_LOGE(context_->GetNodeName(), "outGradOffset dtype not same as outputGradX dtype"),
        return ge::GRAPH_FAILED);
    deformableOffsetsGradOffset.gradOffsetsBatchNum = outputGradOffsetsShapeInfo.GetDim(LIST_INDEX_0);
    deformableOffsetsGradOffset.gradOffsetsImgOutHeight = outputGradOffsetsShapeInfo.GetDim(LIST_INDEX_1);
    deformableOffsetsGradOffset.gradOffsetsImgOutWidth = outputGradOffsetsShapeInfo.GetDim(LIST_INDEX_2);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableOffsetsGradAscendCTilingImpl::GetAttrInfoAndCheck() {
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, strides);
    OP_CHECK_IF(
        strides->GetSize() != DIM_NUM_4D, OP_LOGE(context_->GetNodeName(), "Not stride support len"),
        return ge::GRAPH_FAILED);
    const int64_t* stridesData = static_cast<const int64_t*>(strides->GetData());
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pads);
    OP_CHECK_IF(
        pads->GetSize() != DIM_NUM_4D, OP_LOGE(context_->GetNodeName(), "Not pads support len"),
        return ge::GRAPH_FAILED);
    const int64_t* padsData = static_cast<const int64_t*>(pads->GetData());
    auto ksizes = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ksizes);
    OP_CHECK_IF(
        ksizes->GetSize() != EXCEPTED_KERNEL_SIZE, OP_LOGE(context_->GetNodeName(), "Not ksize support len"),
        return ge::GRAPH_FAILED);
    const int64_t* ksizesData = static_cast<const int64_t*>(ksizes->GetData());
    auto dilates = attrs->GetAttrPointer<gert::ContinuousVector>(DILATIONS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dilates);
    OP_CHECK_IF(
        dilates->GetSize() != DIM_NUM_4D, OP_LOGE(context_->GetNodeName(), "Not dilation support len"),
        return ge::GRAPH_FAILED);
    const int64_t* dilatesData = static_cast<const int64_t*>(dilates->GetData());
    auto modulatePtr = attrs->GetAttrPointer<bool>(MODULATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, modulatePtr);
    isModulated_ = *modulatePtr;
    auto deformableGroupsPtr = attrs->GetAttrPointer<int>(DEFORMABLE_GROUPS_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, deformableGroupsPtr);
    deformableGroupsAttr_ = *deformableGroupsPtr;
    OP_CHECK_IF(deformableGroupsAttr_ <= 0,
        OP_LOGE(context_->GetNodeName(), "Deformable groups must be greater than 0"), return ge::GRAPH_FAILED);
    strideH_ = stridesData[LIST_INDEX_1];
    strideW_ = stridesData[LIST_INDEX_2];
    dilationH_ = dilatesData[LIST_INDEX_1];
    dilationW_ = dilatesData[LIST_INDEX_2];
    padsHeightUp_ = padsData[LIST_INDEX_0];
    padsHeightDown_ = padsData[LIST_INDEX_1];
    padsWidthLeft_ = padsData[LIST_INDEX_2];
    padsWidthRight_ = padsData[LIST_INDEX_3];
    dimKh_ = ksizesData[LIST_INDEX_0];
    dimKw_ = ksizesData[LIST_INDEX_1];
    const string dataFormat = string(attrs->GetAttrPointer<char>(DATA_FORMAT_INDEX));
    OP_CHECK_IF(
        UpdateStrideAndDilationByFormat(context_, dataFormat, stridesData, dilatesData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Get stride or dilation failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableOffsetsGradAscendCTilingImpl::UpdateStrideAndDilationByFormat(
    gert::TilingContext* context, const string dataFormat, const int64_t* stridesData, const int64_t* dilatesData)
{
    if (dataFormat == "NCHW") {
        strideH_ = stridesData[LIST_INDEX_2];
        strideW_ = stridesData[LIST_INDEX_3];
        dilationH_ = dilatesData[LIST_INDEX_2];
        dilationW_ = dilatesData[LIST_INDEX_3];
    } else if (dataFormat == "NHWC") {
        strideH_ = stridesData[LIST_INDEX_1];
        strideW_ = stridesData[LIST_INDEX_2];
        dilationH_ = dilatesData[LIST_INDEX_1];
        dilationW_ = dilatesData[LIST_INDEX_2];
    } else {
        OP_LOGE(context->GetNodeName(), "Attr data format:[%s] is not support.", dataFormat.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        stridesData[LIST_INDEX_0] != 1 || dilatesData[LIST_INDEX_0] != 1,
        OP_LOGE(context_->GetNodeName(), "Stride or dilation data index 0 value must equal to 1"),
        return ge::GRAPH_FAILED);

    int64_t offsetHeightSame = offsetHeight_ * static_cast<int64_t>(dimKh_);
    int64_t offsetWidthSame = offsetWidth_ * static_cast<int64_t>(dimKw_);

    OP_CHECK_IF(
        gradHeight_ != offsetHeightSame,
        OP_LOGE(context->GetNodeName(), "Input grad height not equal to offsetHeight muti ksizesH"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        gradWidth_ != offsetWidthSame,
        OP_LOGE(context->GetNodeName(), "Input grad width not equal to offsetWidth muti ksizesW"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableOffsetsGradAscendCTilingImpl::GetParameterAndCheck(
    const DeformableOffsetsGradOffset deformableOffsetsGradOffset) const
{
    OP_CHECK_IF(
        imgOutHeight_ <= static_cast<int64_t>(0) || imgOutWidth_ <= static_cast<int64_t>(0),
        OP_LOGE(
            context_->GetNodeName(),
            "Img outputShape must greater than 0, please use appropriate parameters for inputShape, pads, dilation,\
        ksize and stride."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        imgChannel_ <= 0, OP_LOGE(context_->GetNodeName(), "Img channel must greater than 0"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        deformableGroupsAttr_ != deformableGroups_,
        OP_LOGE(
            context_->GetNodeName(), "Deformable groups attr: %u is invalid when deformableGroups is %u",
            deformableGroupsAttr_, deformableGroups_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        deformableOffsetsGradOffset.gradOffsetsImgOutHeight != offsetHeight_,
        OP_LOGE(context_->GetNodeName(), "Img out height not equal to offsetHeight"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        deformableOffsetsGradOffset.gradOffsetsImgOutWidth != offsetWidth_,
        OP_LOGE(context_->GetNodeName(), "Img out width not equal to offsetWidth"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (imgBatchNum_ != offsetBatchNum_) || (imgBatchNum_ != gradBatchNum_),
        OP_LOGE(context_->GetNodeName(), "Offsets batchNum not equal to inputX batchNum or input grad batchNum"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        imgBatchNum_ != deformableOffsetsGradOffset.gradOffsetsBatchNum,
        OP_LOGE(context_->GetNodeName(), "Output batchNum not equal to inputX batchNum"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !isModulated_, OP_LOGE(context_->GetNodeName(), "Deformable modulate attr only support true."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableOffsetsGradAscendCTilingImpl::DoTiling()
{
    numKernels_ =
        static_cast<int64_t>(imgBatchNum_ * imgOutHeight_ * imgOutWidth_ * deformableGroups_ * dimKh_ * dimKw_);
    realCoreNum_ = (numKernels_ < maxCoreNum_) ? numKernels_ : maxCoreNum_;
    blockFactor_ = numKernels_ / realCoreNum_;
    blockFactorTail_ = numKernels_ - blockFactor_ * realCoreNum_;

    outputGradXSize_ = imgBatchNum_ * imgHeight_ * imgWidth_ * imgChannel_;
    clearGradXCoreNum_ = (outputGradXSize_ < maxCoreNum_) ? outputGradXSize_ : maxCoreNum_;
    gradXFactor_ = outputGradXSize_ / clearGradXCoreNum_;
    gradXFactorTail_ = outputGradXSize_ - gradXFactor_ * clearGradXCoreNum_;

    outputGradOffsetsSize_ = numKernels_ * POINT_WEIGHT_SIZE;
    clearGradOffsetsCoreNum_ = (outputGradOffsetsSize_ < maxCoreNum_) ? outputGradOffsetsSize_ : maxCoreNum_;
    gradOffsetsFactor_ = outputGradOffsetsSize_ / clearGradOffsetsCoreNum_;
    gradOffsetsFactorTail_ = outputGradOffsetsSize_ - gradOffsetsFactor_ * clearGradOffsetsCoreNum_;

    realUseCoreNum_ = (realCoreNum_ > clearGradXCoreNum_) ? realCoreNum_ : clearGradXCoreNum_;
    FillTilingData();
    size_t sysWorkSpaceSize = SYS_WORK_SPACE_SIZE;
    size_t* userWorkSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkSpaceSize);
    userWorkSpaceSize[0] = sysWorkSpaceSize;
    PrintTilingData();
    OP_LOGD("DeformableOffsetsGradTilingForAscendC", "DeformableOffsetsGradTiling end");
    return ge::GRAPH_SUCCESS;
}

void DeformableOffsetsGradAscendCTilingImpl::FillTilingData()
{
    context_->SetTilingKey(dataTypeKey_);
    context_->SetBlockDim(realUseCoreNum_);
    tilingData_.set_tilingKey(dataTypeKey_);
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_clearGradXCoreNum(clearGradXCoreNum_);
    tilingData_.set_clearGradOffsetsCoreNum(clearGradOffsetsCoreNum_);
    tilingData_.set_strideHeight(strideH_);
    tilingData_.set_strideWidth(strideW_);
    tilingData_.set_dilationHeight(dilationH_);
    tilingData_.set_dilationWidth(dilationW_);
    tilingData_.set_padsHeight(padsHeightUp_);
    tilingData_.set_padsWidth(padsWidthLeft_);
    tilingData_.set_dimKHeight(dimKh_);
    tilingData_.set_dimKWidth(dimKw_);
    tilingData_.set_imgBatchNum(imgBatchNum_);
    tilingData_.set_imgChannel(imgChannel_);
    tilingData_.set_imgWidth(imgWidth_);
    tilingData_.set_imgHeight(imgHeight_);
    tilingData_.set_imgOutHeight(imgOutHeight_);
    tilingData_.set_imgOutWidth(imgOutWidth_);
    tilingData_.set_deformableGroups(deformableGroups_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockFactorTail(blockFactorTail_);
    tilingData_.set_gradXFactor(gradXFactor_);
    tilingData_.set_gradXFactorTail(gradXFactorTail_);
    tilingData_.set_gradOffsetsFactor(gradOffsetsFactor_);
    tilingData_.set_gradOffsetsFactorTail(gradOffsetsFactorTail_);
    tilingData_.set_outputGradXSize(outputGradXSize_);
    tilingData_.set_outputGradOffsetsSize(outputGradOffsetsSize_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void DeformableOffsetsGradAscendCTilingImpl::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is tilingKey:%d, realCoreNum_:%d, clearGradXCoreNum_:%d,clearGradOffsetsCoreNum_:%d,\
    strideHeight:%d, strideWidth:%d, dilationHeight:%d, dilationWidth:%d, padsHeight:%d, padsWidth:%d,\
    dimKHeight:%d, dimKWidth:%d, imgBatchNum_:%d, imgChannel_:%d, imgWidth_:%d, imgHeight_:%d, imgOutHeight_:%d,\
    imgOutWidth_:%d, deformableGroups_:%d, blockFacter:%d, blockFacterTail:%d, gradXFactor_:%d,\
    gradXFactorTail_:%d, gradOffsetsFactor_:%d, gradOffsetsFactorTail_:%d, outputGradXSize_:%d,\
    outputGradOffsetsSize_:%d",
        tilingData_.get_tilingKey(), tilingData_.get_realCoreNum(), tilingData_.get_clearGradXCoreNum(),
        tilingData_.get_clearGradOffsetsCoreNum(), tilingData_.get_strideHeight(), tilingData_.get_strideWidth(),
        tilingData_.get_dilationHeight(), tilingData_.get_dilationWidth(), tilingData_.get_padsHeight(),
        tilingData_.get_padsWidth(), tilingData_.get_dimKHeight(), tilingData_.get_dimKWidth(),
        tilingData_.get_imgBatchNum(), tilingData_.get_imgChannel(), tilingData_.get_imgWidth(),
        tilingData_.get_imgHeight(), tilingData_.get_imgOutHeight(), tilingData_.get_imgOutWidth(),
        tilingData_.get_deformableGroups(), tilingData_.get_blockFactor(), tilingData_.get_blockFactorTail(),
        tilingData_.get_gradXFactor(), tilingData_.get_gradXFactorTail(), tilingData_.get_gradOffsetsFactor(),
        tilingData_.get_gradOffsetsFactorTail(), tilingData_.get_outputGradXSize(),
        tilingData_.get_outputGradOffsetsSize());
}

ge::graphStatus DeformableOffsetsGradTilingForAscendC(gert::TilingContext* context_, const int64_t coreNum)
{
    OP_LOGD(context_->GetNodeName(), "Start Tiling4DeformableOffsetsGradAscendC.");
    DeformableOffsetsGradAscendCTilingImpl tilingImpl = DeformableOffsetsGradAscendCTilingImpl(context_);
    if (tilingImpl.Init(coreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Tiling4DeformableOffsetsGradAscendC init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Tiling4DeformableOffsetsGradAscendC do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4DeformableOffsetsGrad(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Tiling4DeformableOffsetsGrad running.");
    const TilingPrepareForDeformableOffsetsGradCompileInfo* compile_info =
        static_cast<const TilingPrepareForDeformableOffsetsGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    OP_LOGD(context->GetNodeName(), "AscendC deformable offsets grad simt tiling");
    int64_t maxCoreNum = compile_info->coreNum;
    return DeformableOffsetsGradTilingForAscendC(context, maxCoreNum);
}

static ge::graphStatus TilingPrepare4DeformableOffsetsGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Ascend c Tiling starting GRAPH_SUCCESS");
    auto compileInfo = context->GetCompiledInfo<TilingPrepareForDeformableOffsetsGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DeformableOffsetsGrad)
    .Tiling(Tiling4DeformableOffsetsGrad)
    .TilingParse<TilingPrepareForDeformableOffsetsGradCompileInfo>(TilingPrepare4DeformableOffsetsGrad);
} // namespace optiling
