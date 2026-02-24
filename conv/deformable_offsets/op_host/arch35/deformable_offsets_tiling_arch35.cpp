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
 * \file deformable_offsets_tiling_arch35.cpp
 * \brief deformable_offsets_tiling_arch35 impl info
 */

#include "deformable_offsets_tiling_arch35.h"
#include <cstdint>
#include "log/log.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
static constexpr int32_t OUTPUT_Y_INDEX = 0;
static constexpr int32_t INPUT_X_INDEX = 0;
static constexpr int32_t INPUT_OFFSET_INDEX = 1;
static constexpr int32_t DIM_NUM_4D = 4;
static constexpr int32_t STRIDES_INDEX = 0;
static constexpr int32_t PADS_INDEX = 1;
static constexpr int32_t KSIZE_INDEX = 2;
static constexpr int32_t DILATIONS_INDEX = 3;
static constexpr int32_t DATA_FORMAT_INDEX = 4;
static constexpr int32_t DEFORMABLE_GROUPS_ATTR_INDEX = 5;
static constexpr int32_t MODULATE_INDEX = 6;
static constexpr int32_t LIST_INDEX_0 = 0;
static constexpr int32_t LIST_INDEX_1 = 1;
static constexpr int32_t LIST_INDEX_2 = 2;
static constexpr int32_t LIST_INDEX_3 = 3;
static constexpr int64_t POINT_WEIGHT_SIZE = 3;
static constexpr int64_t POINT_NOT_WEIGHT_SIZE = 2;
static constexpr int64_t VF_MAX_THREAD_NUM = 512;
static constexpr int64_t EXCEPTED_KERNEL_SIZE = 2;
static constexpr int64_t SIMT_COMMON_TILING_KEY = 1000;
static constexpr int64_t DATA_INT32_MAX = 2147483647; // int32最大值
static const std::set<ge::DataType> supportDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

ge::graphStatus CheckDeformableOffsetParams(
    gert::TilingContext* context, gert::Shape& inputXShape, gert::Shape& inputOffsetShape, gert::Shape& outputShapeInfo)
{
    // input data Format
    auto inputXdesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXdesc);
    auto inputXDataFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(inputXdesc->GetStorageFormat()));
    auto inputXDataType = inputXdesc->GetDataType();
    OP_CHECK_IF(
        supportDtype.count(inputXDataType) == 0, OP_LOGE(context->GetNodeName(), "X not support data type"),
        return ge::GRAPH_FAILED);
    // get input x shape info
    auto inputX = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_IF(inputX == nullptr, OP_LOGE(context->GetNodeName(), "Get input x failed"), return ge::GRAPH_FAILED);
    inputXShape = inputX->GetStorageShape();
    int64_t inputXShapeDims = inputXShape.GetDimNum();

    // get input offset shape info
    auto inputOffsetdesc = context->GetInputDesc(INPUT_OFFSET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputOffsetdesc);
    auto inputYDataType = inputOffsetdesc->GetDataType();
    OP_CHECK_IF(
        supportDtype.count(inputYDataType) == 0, OP_LOGE(context->GetNodeName(), "offset not support data type"),
        return ge::GRAPH_FAILED);
    auto inputOffset = context->GetInputShape(INPUT_OFFSET_INDEX);
    OP_CHECK_IF(
        inputOffset == nullptr, OP_LOGE(context->GetNodeName(), "Get input offset failed"), return ge::GRAPH_FAILED);
    inputOffsetShape = inputOffset->GetStorageShape();
    int64_t inputOffsetShapeDims = inputOffsetShape.GetDimNum();

    // get output shape info
    auto outputYdesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYdesc);
    auto outputYDataType = outputYdesc->GetDataType();
    OP_CHECK_IF(
        supportDtype.count(outputYDataType) == 0, OP_LOGE(context->GetNodeName(), "Not support output data type"),
        return ge::GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShapeInfo = outputShape->GetStorageShape();

    // check shape and format
    OP_CHECK_IF(
        inputXShapeDims != DIM_NUM_4D || inputOffsetShapeDims != DIM_NUM_4D,
        OP_LOGE(context->GetNodeName(), "X offset shape check failed"), return ge::GRAPH_FAILED);
    ge::Format inputOffsetDataFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(inputOffsetdesc->GetStorageFormat()));
    OP_CHECK_IF(
        inputXDataFormat != ge::FORMAT_NHWC || inputOffsetDataFormat != ge::FORMAT_NHWC,
        OP_LOGE(context->GetNodeName(), "X offset format support [NHWC]"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateStrideAndDilationByFormat(
    gert::TilingContext* context, const std::string format, DeformableOffsetAttr& deformableOffsetAttrInfo,
    const int64_t* stridesData, const int64_t* dilatesData)
{
    if (format == "NCHW") {
        deformableOffsetAttrInfo.strideH = stridesData[LIST_INDEX_2];
        deformableOffsetAttrInfo.strideW = stridesData[LIST_INDEX_3];
        deformableOffsetAttrInfo.dilationH = dilatesData[LIST_INDEX_2];
        deformableOffsetAttrInfo.dilationW = dilatesData[LIST_INDEX_3];
        OP_CHECK_IF(
            stridesData[LIST_INDEX_0] != 1 || stridesData[LIST_INDEX_1] != 1 || dilatesData[LIST_INDEX_0] != 1 ||
                dilatesData[LIST_INDEX_1] != 1,
            OP_LOGE(context->GetNodeName(), "Stride or dilation data index 0 and 1 value must equal 1"),
            return ge::GRAPH_FAILED);
    } else if (format == "NHWC") {
        deformableOffsetAttrInfo.strideH = stridesData[LIST_INDEX_1];
        deformableOffsetAttrInfo.strideW = stridesData[LIST_INDEX_2];
        deformableOffsetAttrInfo.dilationH = dilatesData[LIST_INDEX_1];
        deformableOffsetAttrInfo.dilationW = dilatesData[LIST_INDEX_2];
        OP_CHECK_IF(
            stridesData[LIST_INDEX_0] != 1 || stridesData[LIST_INDEX_3] != 1 || dilatesData[LIST_INDEX_0] != 1 ||
                dilatesData[LIST_INDEX_3] != 1,
            OP_LOGE(context->GetNodeName(), "Stride or dilation data index 0 and 3 value must equal 1"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context->GetNodeName(), "dataFormat %s is not support", format.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        deformableOffsetAttrInfo.strideH == 0 || deformableOffsetAttrInfo.strideW == 0,
        OP_LOGE(context->GetNodeName(), "StrideH not equal to 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDeformableOffsetAttrs(gert::TilingContext* context, DeformableOffsetAttr& deformableOffsetAttrInfo)
{
    // attr info
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(strides->GetSize() != DIM_NUM_4D, OP_LOGE("DeformableOffsetTiling", "Not stride support len"),
        return ge::GRAPH_FAILED);
    const int64_t* stridesData = reinterpret_cast<const int64_t*>(strides->GetData());

    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(pads->GetSize() != DIM_NUM_4D, OP_LOGE("DeformableOffsetTiling", "Not pads support len"),
        return ge::GRAPH_FAILED);
    const int64_t* padsData = reinterpret_cast<const int64_t*>(pads->GetData());

    auto ksizes = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksizes);
    OP_CHECK_IF(ksizes->GetSize() != EXCEPTED_KERNEL_SIZE, OP_LOGE("DeformableOffsetTiling", "Not ksize support len"),
        return ge::GRAPH_FAILED);
    const int64_t* ksizesData = reinterpret_cast<const int64_t*>(ksizes->GetData());

    auto dilates = attrs->GetAttrPointer<gert::ContinuousVector>(DILATIONS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilates);
    OP_CHECK_IF(
        dilates->GetSize() != DIM_NUM_4D, OP_LOGE("DeformableOffsetTiling", "Not dilation support len"),
        return ge::GRAPH_FAILED);
    const int64_t* dilatesData = reinterpret_cast<const int64_t*>(dilates->GetData());

    auto modulatePtr = attrs->GetAttrPointer<bool>(MODULATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, modulatePtr);
    bool isModulated = *modulatePtr;

    auto deformableGroupsPtr = attrs->GetAttrPointer<int>(DEFORMABLE_GROUPS_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, deformableGroupsPtr);
    deformableOffsetAttrInfo.deformableGroupsAttr = *deformableGroupsPtr;
    const std::string dataFormat = std::string(attrs->GetAttrPointer<char>(DATA_FORMAT_INDEX));
    OP_CHECK_IF(
        UpdateStrideAndDilationByFormat(context, dataFormat, deformableOffsetAttrInfo, stridesData, dilatesData) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get stride failed."), return ge::GRAPH_FAILED);
    // get attr value
    deformableOffsetAttrInfo.padsHeightUp = padsData[LIST_INDEX_0];
    deformableOffsetAttrInfo.padsHeightDown = padsData[LIST_INDEX_1];
    deformableOffsetAttrInfo.padsWidthLeft = padsData[LIST_INDEX_2];
    deformableOffsetAttrInfo.padsWidthRight = padsData[LIST_INDEX_3];
    deformableOffsetAttrInfo.dimKh = ksizesData[LIST_INDEX_0];
    deformableOffsetAttrInfo.dimKw = ksizesData[LIST_INDEX_1];
    deformableOffsetAttrInfo.offsetValueDim =
        isModulated ? POINT_WEIGHT_SIZE : POINT_NOT_WEIGHT_SIZE;
    OP_CHECK_IF(
        !isModulated, OP_LOGE(context->GetNodeName(), "Deformable modulate attr only support true."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOffsetArgs(
    gert::TilingContext* context, DeformableOffsetsOffset& deformableOffsetsOffset, const gert::Shape inputOffsetShape,
    const gert::Shape outputShapeInfo, DeformableOffsetAttr deformableOffsetAttrInfo) {
    // input offset info
    int64_t offsetWidth = inputOffsetShape.GetDim(LIST_INDEX_2);
    int64_t offsetHeight = inputOffsetShape.GetDim(LIST_INDEX_1);
    int64_t offsetChannel = inputOffsetShape.GetDim(LIST_INDEX_3);
    deformableOffsetsOffset.offsetBatchStride = offsetHeight * offsetWidth * offsetChannel;
    deformableOffsetsOffset.deformableGroups = offsetChannel /
        (deformableOffsetAttrInfo.dimKh * deformableOffsetAttrInfo.dimKw * deformableOffsetAttrInfo.offsetValueDim);
    OP_CHECK_IF(deformableOffsetsOffset.imgChannel % deformableOffsetsOffset.deformableGroups != 0,
        OP_LOGE(context->GetNodeName(), "Img channel(%u) not equal to %d", deformableOffsetsOffset.imgChannel,
            deformableOffsetsOffset.deformableGroups),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        deformableOffsetAttrInfo.deformableGroupsAttr != deformableOffsetsOffset.deformableGroups,
        OP_LOGE(
            context->GetNodeName(), "Deformable groups attr is %u is invalid when deformableGroups is %u",
            deformableOffsetAttrInfo.deformableGroupsAttr, deformableOffsetsOffset.deformableGroups),
        return ge::GRAPH_FAILED);

    int64_t outputHeight = outputShapeInfo.GetDim(LIST_INDEX_1);
    int64_t outputWidth = outputShapeInfo.GetDim(LIST_INDEX_2);
    deformableOffsetsOffset.outputBatchStride = outputHeight * outputWidth * deformableOffsetsOffset.imgChannel;
    int64_t offsetHeightSame = offsetHeight * deformableOffsetAttrInfo.dimKh;
    int64_t offsetWidthSame = offsetWidth * deformableOffsetAttrInfo.dimKw;

    OP_CHECK_IF(outputShapeInfo.GetDim(LIST_INDEX_1) != offsetHeightSame,
        OP_LOGE(context->GetNodeName(), "Img out height not equal to offsetHeight muti dimkh"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputShapeInfo.GetDim(LIST_INDEX_2) != offsetWidthSame,
        OP_LOGE(context->GetNodeName(), "Img out width not equal to offsetWidth muti dimkw"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(deformableOffsetsOffset.imgOutHeight != inputOffsetShape.GetDim(LIST_INDEX_1),
        OP_LOGE(context->GetNodeName(), "Img out height not equal to offsetHeight"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(deformableOffsetsOffset.imgOutWidth != inputOffsetShape.GetDim(LIST_INDEX_2),
        OP_LOGE(context->GetNodeName(), "Img out width not equal to offsetWidth"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(deformableOffsetsOffset.imgBatchNum != inputOffsetShape.GetDim(LIST_INDEX_0),
        OP_LOGE(context->GetNodeName(), "Offset batchNum not equal to input"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(deformableOffsetsOffset.imgBatchNum != outputShapeInfo.GetDim(LIST_INDEX_0),
        OP_LOGE(context->GetNodeName(), "Output batchNum not equal to input"), return ge::GRAPH_FAILED);
    // offset NHC (N, H, W, offsetValueDim, group=1, k_h, k_w)
    deformableOffsetsOffset.offsetKernelElementStride =
        deformableOffsetsOffset.deformableGroups * deformableOffsetAttrInfo.dimKh * deformableOffsetAttrInfo.dimKw;
    deformableOffsetsOffset.offsetPointStride =
        deformableOffsetAttrInfo.offsetValueDim * deformableOffsetsOffset.offsetKernelElementStride;
    deformableOffsetsOffset.offsetWidthStride = offsetWidth * deformableOffsetsOffset.offsetPointStride;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalDeformableOffsetsOffset(
    gert::TilingContext* context, gert::Shape& inputOffsetShape, gert::Shape& outputShapeInfo,
    DeformableOffsetAttr& deformableOffsetAttrInfo, DeformableOffsetsOffset& deformableOffsetsOffset)
{
    deformableOffsetsOffset.imgWidthStride = deformableOffsetsOffset.imgWidth * deformableOffsetsOffset.imgChannel;
    deformableOffsetsOffset.imgBatchStride =
        deformableOffsetsOffset.imgHeight * deformableOffsetsOffset.imgWidth * deformableOffsetsOffset.imgChannel;
    // conved img
    deformableOffsetsOffset.imgOutHeight =
        (deformableOffsetsOffset.imgHeight + deformableOffsetAttrInfo.padsHeightUp +
         deformableOffsetAttrInfo.padsHeightDown -
         (deformableOffsetAttrInfo.dilationH * (deformableOffsetAttrInfo.dimKh - 1) + 1)) /
            deformableOffsetAttrInfo.strideH + 1;
    deformableOffsetsOffset.imgOutWidth =
        (deformableOffsetsOffset.imgWidth + deformableOffsetAttrInfo.padsWidthLeft +
         deformableOffsetAttrInfo.padsWidthRight -
         (deformableOffsetAttrInfo.dilationW * (deformableOffsetAttrInfo.dimKw - 1) + 1)) /
            deformableOffsetAttrInfo.strideW + 1;

    OP_CHECK_IF(
        deformableOffsetsOffset.imgOutHeight <= 0, OP_LOGE(context->GetNodeName(), "ImgOutHeight must greate 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        deformableOffsetsOffset.imgOutWidth <= 0, OP_LOGE(context->GetNodeName(), "ImgOutWidth must greate 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        deformableOffsetsOffset.imgChannel <= 0, OP_LOGE(context->GetNodeName(), "Img channel must greate 0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckOffsetArgs(
            context, deformableOffsetsOffset, inputOffsetShape, outputShapeInfo, deformableOffsetAttrInfo) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check args failed."), return ge::GRAPH_FAILED);
    // output NHWC
    // real is (N, H, K_h, W, k_w, c)
    deformableOffsetsOffset.outputPointWidthStride =
        deformableOffsetAttrInfo.dimKw * deformableOffsetsOffset.imgChannel;
    deformableOffsetsOffset.outputWidthStride =
        deformableOffsetsOffset.imgOutWidth * deformableOffsetsOffset.outputPointWidthStride;
    deformableOffsetsOffset.outputKernelWidthStride =
        deformableOffsetAttrInfo.dimKh * deformableOffsetsOffset.outputWidthStride;
    deformableOffsetsOffset.numKernels = deformableOffsetsOffset.imgOutWidth * deformableOffsetsOffset.imgOutHeight *
                                         deformableOffsetsOffset.imgChannel * deformableOffsetsOffset.imgBatchNum;
    deformableOffsetsOffset.blockDimValue =
        (deformableOffsetsOffset.numKernels / VF_MAX_THREAD_NUM) +
        ((deformableOffsetsOffset.numKernels % VF_MAX_THREAD_NUM) == 0 ? 0 : 1);
    return ge::GRAPH_SUCCESS;
}

void SetDeformableOffsetsTilingData(
    gert::TilingContext* context, DeformableOffsetsTilingDataSimt& deformableOffsetTilingData,
    const DeformableOffsetsOffset deformableOffsetsOffset, const DeformableOffsetAttr deformableOffsetAttrInfo)
{
    deformableOffsetTilingData.set_blockNum(deformableOffsetsOffset.blockDimValue);
    deformableOffsetTilingData.set_strideHeight(deformableOffsetAttrInfo.strideH);
    deformableOffsetTilingData.set_strideWidth(deformableOffsetAttrInfo.strideW);
    deformableOffsetTilingData.set_dilationHeight(deformableOffsetAttrInfo.dilationH);
    deformableOffsetTilingData.set_dilationWidth(deformableOffsetAttrInfo.dilationW);
    deformableOffsetTilingData.set_padsHeight(deformableOffsetAttrInfo.padsHeightUp);
    deformableOffsetTilingData.set_padsWidth(deformableOffsetAttrInfo.padsWidthLeft);
    deformableOffsetTilingData.set_dimKHeight(deformableOffsetAttrInfo.dimKh);
    deformableOffsetTilingData.set_dimKWidth(deformableOffsetAttrInfo.dimKw);
    deformableOffsetTilingData.set_imgChannel(deformableOffsetsOffset.imgChannel);
    deformableOffsetTilingData.set_imgWidth(deformableOffsetsOffset.imgWidth);
    deformableOffsetTilingData.set_imgHeight(deformableOffsetsOffset.imgHeight);
    deformableOffsetTilingData.set_imgWidthStride(deformableOffsetsOffset.imgWidthStride);
    deformableOffsetTilingData.set_imgOutHeight(deformableOffsetsOffset.imgOutHeight);
    deformableOffsetTilingData.set_imgOutWidth(deformableOffsetsOffset.imgOutWidth);
    deformableOffsetTilingData.set_offsetKernelElementStride(deformableOffsetsOffset.offsetKernelElementStride);
    deformableOffsetTilingData.set_offsetPointStride(deformableOffsetsOffset.offsetPointStride);
    deformableOffsetTilingData.set_offsetWidthStride(deformableOffsetsOffset.offsetWidthStride);
    deformableOffsetTilingData.set_offsetValueDim(deformableOffsetAttrInfo.offsetValueDim);
    deformableOffsetTilingData.set_deformableGroups(deformableOffsetsOffset.deformableGroups);
    deformableOffsetTilingData.set_outputPointWidthStride(deformableOffsetsOffset.outputPointWidthStride);
    deformableOffsetTilingData.set_outputWidthStride(deformableOffsetsOffset.outputWidthStride);
    deformableOffsetTilingData.set_outputKernelWidthStride(deformableOffsetsOffset.outputKernelWidthStride);
    deformableOffsetTilingData.set_numKernels(deformableOffsetsOffset.numKernels);
    deformableOffsetTilingData.set_imgBatchStride(deformableOffsetsOffset.imgBatchStride);
    deformableOffsetTilingData.set_offsetBatchStride(deformableOffsetsOffset.offsetBatchStride);
    deformableOffsetTilingData.set_outputBatchStride(deformableOffsetsOffset.outputBatchStride);
    deformableOffsetTilingData.set_imgBatchNum(deformableOffsetsOffset.imgBatchNum);
    deformableOffsetTilingData.SaveToBuffer(
        context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
}

ge::graphStatus DeformableOffsetTiling(gert::TilingContext* context, int64_t maxCoreNum)
{
    OP_LOGD("DeformableOffsetTilingForAscendC", "DeformableOffsetTiling start");
    DeformableOffsetsTilingDataSimt deformableOffsetTilingData;
    gert::Shape inputXShape, inputOffsetShape, outputShapeInfo;

    OP_CHECK_IF(
        CheckDeformableOffsetParams(context, inputXShape, inputOffsetShape, outputShapeInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check params failed."), return ge::GRAPH_FAILED);
    DeformableOffsetAttr deformableOffsetAttrInfo;
    OP_CHECK_IF(
        CheckDeformableOffsetAttrs(context, deformableOffsetAttrInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check attr failed."), return ge::GRAPH_FAILED);

    DeformableOffsetsOffset deformableOffsetsOffset;
    // input x info
    deformableOffsetsOffset.imgBatchNum = inputXShape.GetDim(LIST_INDEX_0);
    deformableOffsetsOffset.imgChannel = inputXShape.GetDim(LIST_INDEX_3);
    deformableOffsetsOffset.imgWidth = inputXShape.GetDim(LIST_INDEX_2);
    deformableOffsetsOffset.imgHeight = inputXShape.GetDim(LIST_INDEX_1);
    OP_CHECK_IF(
        CalDeformableOffsetsOffset(
            context, inputOffsetShape, outputShapeInfo, deformableOffsetAttrInfo, deformableOffsetsOffset) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get offsets failed."), return ge::GRAPH_FAILED);
    deformableOffsetsOffset.blockDimValue =
        std::min(deformableOffsetsOffset.blockDimValue, maxCoreNum);
    int64_t input0Size = inputXShape.GetShapeSize();
    int64_t input1Size = inputOffsetShape.GetShapeSize();
    int64_t outputSize = outputShapeInfo.GetShapeSize();
    int64_t tilingKey = SIMT_COMMON_TILING_KEY;
    if (input0Size > DATA_INT32_MAX || input1Size > DATA_INT32_MAX || outputSize > DATA_INT32_MAX) {
        tilingKey = tilingKey + 1;
    }
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(deformableOffsetsOffset.blockDimValue);
    SetDeformableOffsetsTilingData(
        context, deformableOffsetTilingData, deformableOffsetsOffset, deformableOffsetAttrInfo);
    context->GetRawTilingData()->SetDataSize(deformableOffsetTilingData.GetDataSize());

    size_t usrSize = 0;
    size_t sysWorkSpaceSize = 16 * 1024 * 1024;

    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = usrSize + sysWorkSpaceSize;
    OP_LOGD("DeformableOffsetTilingForAscendC", "DeformableOffsetTiling end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableOffsetTilingSimt(gert::TilingContext* context, int64_t maxCoreNum)
{
    return DeformableOffsetTiling(context, maxCoreNum);
}

ge::graphStatus Tiling4DeformableOffsets(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Tiling4DeformableOffsets running.");
    const TilingPrepareForDeformableOffsetsCompileInfo* compileInfo =
        context->GetCompileInfo<TilingPrepareForDeformableOffsetsCompileInfo>();

    int64_t maxCoreNum = compileInfo->coreNum;
    OP_CHECK_IF(
        DeformableOffsetTilingSimt(context, maxCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The simd tiling function failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4PrepareDeformableOffsets(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4DeformableOffsets.");

    auto compileInfo = context->GetCompiledInfo<TilingPrepareForDeformableOffsetsCompileInfo>();
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

IMPL_OP_OPTILING(DeformableOffsets)
    .Tiling(Tiling4DeformableOffsets)
    .TilingParse<TilingPrepareForDeformableOffsetsCompileInfo>(Tiling4PrepareDeformableOffsets);
} // namespace optiling
