/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv2d_v2_base_tiling_check_input.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv2dBaseTiling::CheckOptionalInputLeagal()
{
    if (ParseBiasShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantScaleLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExtendScaleLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExtendReluWeightLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExtendClipValueLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetDisContinuousFlag()
{
    auto viewShapePtr = context_->GetInputShape(INPUT_FMAP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, viewShapePtr);
    auto viewShape = viewShapePtr->GetOriginShape();
    if (!context_->InputIsView(INPUT_FMAP_INDEX)) {
        return ge::GRAPH_SUCCESS;
    }
    auto viewStridePtr = context_->GetInputStride(INPUT_FMAP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, viewStridePtr);
    // check continuous
    if (viewStridePtr->GetStride(FORMAT_NCHW_N_INDEX) == 0) {
        return ge::GRAPH_SUCCESS;
    }

    if (descInfo_.fMapDtype != ge::DataType::DT_FLOAT && descInfo_.fMapDtype != ge::DataType::DT_FLOAT16 &&
        descInfo_.fMapDtype != ge::DataType::DT_BF16) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: disContinuous dtype: %d, should be fp16/bf16/fp32",
                paramInfo_.nodeType.c_str(), descInfo_.fMapDtype);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeType(), "x",
            GeDtypeToString(descInfo_.fMapDtype).c_str(),
            FormatString("If input x is a non-contiguous tensor, the dtype of parameter %s must be one of %s",
                "x", "{float16, float32, bfloat16}").c_str());
        return ge::GRAPH_FAILED;
    }
        
    if (context_->GetInputOffset(0) != 0) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: disContinuous inputOffset should be 0",
        paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }
    // check view_shape is HWNC & storage_shape is NCHW
    std::vector<int64_t> expectStride = {viewShape[FORMAT_NCHW_C_INDEX], 1,
        viewShape[FORMAT_NCHW_W_INDEX] * viewShape[FORMAT_NCHW_N_INDEX] * viewShape[FORMAT_NCHW_C_INDEX],
        viewShape[FORMAT_NCHW_N_INDEX] * viewShape[FORMAT_NCHW_C_INDEX]};
    for (size_t i = 0; i < CONV2D_DIM_SIZE_LIMIT; i++) {
        if (viewStridePtr->GetStride(i) != expectStride[i]) {
            OP_LOGE(context_->GetNodeName(),
                    "%s AscendC: disContinuous Input not satisfy viewShape(HWNC) storageShape(NCHW)",
                    paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    flagInfo_.disContinuousFlag = true;
    OP_LOGD(context_->GetNodeName(), "%s AscendC: disContinuous HWNC input", paramInfo_.nodeType.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::ParseFmapShape()
{
    auto fMapShapePtr = context_->GetInputShape(INPUT_FMAP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, fMapShapePtr);
    auto fMapShape = fMapShapePtr->GetStorageShape();
    if (fMapShape.GetDimNum() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input feature map shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), fMapShape.GetDimNum(), CONV2D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "x",
            std::to_string(fMapShape.GetDimNum()).c_str(),
            std::to_string(CONV2D_DIM_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }

    if (flagInfo_.disContinuousFlag) {
        // fMapShape is storage shape, but paramsIdxVec is from view format
        oriShapeAttrInfo_.oriFmapN = fMapShape.GetDim(DIS_CONTINUOUS_N_IDX);
        oriShapeAttrInfo_.oriFmapC = fMapShape.GetDim(DIS_CONTINUOUS_C_IDX);
        oriShapeAttrInfo_.oriFmapH = fMapShape.GetDim(DIS_CONTINUOUS_H_IDX);
        oriShapeAttrInfo_.oriFmapW = fMapShape.GetDim(DIS_CONTINUOUS_W_IDX);
        return ge::GRAPH_SUCCESS;
    }

    oriShapeAttrInfo_.oriFmapN = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_N_IDX]);
    oriShapeAttrInfo_.oriFmapC = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_C_IDX]);
    oriShapeAttrInfo_.oriFmapH = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_H_IDX]);
    oriShapeAttrInfo_.oriFmapW = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_W_IDX]);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckFmapShape()
{
    uint64_t batchMaxSize = shapeBoundTab.at("N").GetUpperBound(descInfo_.fMapDtype);
    uint64_t cInMaxSize = shapeBoundTab.at("Ci").GetUpperBound(descInfo_.fMapDtype);
    uint64_t hInMaxSize = shapeBoundTab.at("H").GetUpperBound(descInfo_.fMapDtype);
    uint64_t wInMaxSize = shapeBoundTab.at("W").GetUpperBound(descInfo_.fMapDtype);
    if (!CheckDim(oriShapeAttrInfo_.oriFmapN, batchMaxSize)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Batch (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapN, batchMaxSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_N_IDX], 1, batchMaxSize).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!CheckDim(oriShapeAttrInfo_.oriFmapC, cInMaxSize)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Cin (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapC, cInMaxSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_C_IDX], 1, cInMaxSize).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!CheckDim(oriShapeAttrInfo_.oriFmapH, hInMaxSize)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Hin (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapH, hInMaxSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_H_IDX], 1, hInMaxSize).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!CheckDim(oriShapeAttrInfo_.oriFmapW, wInMaxSize)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Win (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapW, wInMaxSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_W_IDX], 1, wInMaxSize).c_str());
        return ge::GRAPH_FAILED;
    }


    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::ParseWeightShape()
{
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShapePtr);
    auto weightShape = GetWeightShape(weightShapePtr);
    if (weightShape.GetDimNum() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input weight shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), weightShape.GetDimNum(), CONV2D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "filter",
            std::to_string(weightShape.GetDimNum()).c_str(),
            std::to_string(CONV2D_DIM_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }

    oriShapeAttrInfo_.oriWeightN =
        weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX]);
    oriShapeAttrInfo_.oriWeightC =
        weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_C_IDX]);
    oriShapeAttrInfo_.oriWeightH =
        weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_H_IDX]);
    oriShapeAttrInfo_.oriWeightW =
        weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_W_IDX]);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckWeightShape()
{
    uint64_t cOutMaxSize = shapeBoundTab.at("Co").GetUpperBound(descInfo_.weightDtype);
    uint64_t kHMaxSize = shapeBoundTab.at("kH").GetUpperBound(descInfo_.weightDtype);
    uint64_t kWMaxSize = shapeBoundTab.at("kW").GetUpperBound(descInfo_.weightDtype);
    if (!CheckDim(oriShapeAttrInfo_.oriWeightN, cOutMaxSize)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Cout (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightN, cOutMaxSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "filter",
            VectorToString(GetInputShapeVec(context_, INPUT_WEIGHT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX], 1, cOutMaxSize).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!CheckDim(oriShapeAttrInfo_.oriWeightH, kHMaxSize)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: kH (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightH, kHMaxSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "filter",
            VectorToString(GetInputShapeVec(context_, INPUT_WEIGHT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_H_IDX], 1, kHMaxSize).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!CheckDim(oriShapeAttrInfo_.oriWeightW, kWMaxSize)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: kW (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightW, kWMaxSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "filter",
            VectorToString(GetInputShapeVec(context_, INPUT_WEIGHT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_W_IDX], 1, kWMaxSize).c_str());
        return ge::GRAPH_FAILED;
    }


    auto k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_K_IDX);
    if (k0 == 0) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Get k0 = 0", paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckBiasShapeLegal(size_t idxC, uint32_t biasDimNum)
{
    auto biasShapePtr = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (flagInfo_.quantFlag) {
        biasShapePtr = context_->GetOptionalInputShape(QUANT_INPUT_BIAS_INDEX);
    }
    if (biasShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    auto weightShape = GetWeightShape(weightShapePtr);
    // when biasDimNum is 1, idxC is 0
    for (size_t i = 0; i < biasDimNum; i++) {
        if (i == idxC) {
            if (biasShapePtr->GetStorageShape().GetDim(i) !=
                weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX])) {
                OP_LOGE(context_->GetNodeName(),
                        "%s AscendC: input illegal bias shape: %ld, which must equal to Cout: %ld.",
                        paramInfo_.nodeType.c_str(), biasShapePtr->GetStorageShape().GetDim(i),
                        weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX]));
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeType(), "bias, filter",
                    VectorsToString(std::vector<std::vector<int64_t>>{
                        GetInputShapeVec(context_, INPUT_BIAS_INDEX),
                        GetInputShapeVec(context_, INPUT_WEIGHT_INDEX)}, IntToString<int64_t>).c_str(),
                    FormatString("Shape[%zu] of %s must be equal to shape[%zu] of %s",
                        i, "bias",
                        paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX], "filter").c_str());
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (biasShapePtr->GetStorageShape().GetDim(i) != 1) {
            OP_LOGE(context_->GetNodeName(), "%s AscendC: input bias shape dim %zu: %ld, but only support 1.",
                paramInfo_.nodeType.c_str(), i, biasShapePtr->GetStorageShape().GetDim(i));
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "bias",
                VectorToString(GetInputShapeVec(context_, INPUT_BIAS_INDEX), IntToString<int64_t>).c_str(),
                FormatString("Shape[%zu] of this parameter must be equal to %d", i, 1).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::ParseBiasShape()
{
    auto biasShapePtr = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (flagInfo_.quantFlag) {
        biasShapePtr = context_->GetOptionalInputShape(QUANT_INPUT_BIAS_INDEX);
    }
    if (biasShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasDimNum = biasShapePtr->GetStorageShape().GetDimNum();
    if (biasDimNum != FORMAT_ND_DIM && biasDimNum != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input bias shape dim num: %zu , but only support %u and %u.",
                paramInfo_.nodeType.c_str(), biasShapePtr->GetStorageShape().GetDimNum(),
                FORMAT_ND_DIM, CONV2D_DIM_SIZE_LIMIT);
        string expectDims = std::to_string(FORMAT_ND_DIM) + " or " + std::to_string(CONV2D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "bias",
            std::to_string(biasShapePtr->GetStorageShape().GetDimNum()).c_str(), expectDims.c_str());
        return ge::GRAPH_FAILED;
    }

    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    auto weightShape = GetWeightShape(weightShapePtr);
    size_t idxC = 0;
    auto biasDesc = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasFormat = biasDesc->GetStorageFormat();
    const string biasStr = "Bias";
    // get the index of C dim in 4D
    if (biasDimNum == CONV2D_DIM_SIZE_LIMIT) {  // bias shape is 4D
        if (!GetPosByFormat(biasFormat, "C", biasStr, idxC)) {
            return ge::GRAPH_FAILED;
        }
    }

    if (CheckBiasShapeLegal(idxC, biasDimNum) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckBiasShape() const
{
    return ge::GRAPH_SUCCESS;
}

bool Conv2dBaseTiling::GetPosByFormat(const ge::Format format, const std::string &pos,
                                      const std::string &inputStr, size_t &posIdx) const
{
    string formatStr = formatToStrTab[format];
    // util func illegal scene protect, not a useful dfx
    OP_LOGE_IF(formatStr.length() != CONV2D_DIM_SIZE_LIMIT, false, context_->GetNodeName(),
        "%s AscendC: %s format is not 4D.", paramInfo_.nodeType.c_str(), inputStr.c_str());
    OP_LOGE_IF(pos.length() != 1 || formatStr.find(pos) == string::npos, false, context_->GetNodeName(),
        "%s AscendC: %s pos %s not in 4d format: %s.", paramInfo_.nodeType.c_str(), inputStr.c_str(), pos.c_str(), formatStr.c_str());
    posIdx = formatStr.find(pos);
    return true;
}

ge::graphStatus Conv2dBaseTiling::ParseOutputShape()
{
    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShapePtr);
    auto outputShape = outputShapePtr->GetStorageShape();
    if (outputShape.GetDimNum() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: output shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), outputShape.GetDimNum(), CONV2D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "y",
            std::to_string(outputShape.GetDimNum()).c_str(),
            std::to_string(CONV2D_DIM_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriOutputN =
        outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_N_IDX]);
    oriShapeAttrInfo_.oriOutputC =
        outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_C_IDX]);
    oriShapeAttrInfo_.oriOutputH =
        outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_H_IDX]);
    oriShapeAttrInfo_.oriOutputW =
        outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_W_IDX]);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckOutputShape()
{
    if (!CheckDim(oriShapeAttrInfo_.oriOutputH, MAX_OUT_SHAPE)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Hout (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputH, MAX_OUT_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "y",
            VectorToString(GetOutputShapeVec(context_, OUTPUT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_H_IDX], 1, MAX_OUT_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!CheckDim(oriShapeAttrInfo_.oriOutputW, MAX_OUT_SHAPE)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Wout (%ld) is out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputW, MAX_OUT_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "y",
            VectorToString(GetOutputShapeVec(context_, OUTPUT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%lu, %lu]",
                paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_W_IDX], 1, MAX_OUT_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }

    if (flagInfo_.extendConvFlag && attrInfo_.dualOutput == 1 &&
        CheckExtendDualOutputShape() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::ParseExtendDualOutputShape()
{
    if (!(flagInfo_.extendConvFlag && attrInfo_.dualOutput == 1)) {
        return ge::GRAPH_SUCCESS;
    }
    auto output1ShapePtr = context_->GetOutputShape(EXTENDCONV_OUTPUT_1_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, output1ShapePtr);
    auto output1Shape = output1ShapePtr->GetStorageShape();
    if (output1Shape.GetDimNum() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: output1 shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), output1Shape.GetDimNum(), CONV2D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "y1",
            std::to_string(output1Shape.GetDimNum()).c_str(),
            std::to_string(CONV2D_DIM_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }

    oriShapeAttrInfo_.oriOutput1N =
        output1Shape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_N_IDX]);
    oriShapeAttrInfo_.oriOutput1C =
        output1Shape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_C_IDX]);
    oriShapeAttrInfo_.oriOutput1H =
        output1Shape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_H_IDX]);
    oriShapeAttrInfo_.oriOutput1W =
        output1Shape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_W_IDX]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendDualOutputShape()
{
    const std::vector<int64_t> output1Dims = {oriShapeAttrInfo_.oriOutput1N, oriShapeAttrInfo_.oriOutput1C,
                                              oriShapeAttrInfo_.oriOutput1H, oriShapeAttrInfo_.oriOutput1W};
    const std::vector<int64_t> expectedDims = {oriShapeAttrInfo_.oriOutputN, oriShapeAttrInfo_.oriOutputC,
                                               oriShapeAttrInfo_.oriOutputH, oriShapeAttrInfo_.oriOutputW};
    const std::vector<std::string> dimNames = {"N", "C", "H", "W"};

    for (size_t i = 0; i < CONV2D_DIM_SIZE_LIMIT; ++i) {
        if (output1Dims[i] != expectedDims[i]) {
            OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Output1 %s dimension (%ld) is not equal to expected (%ld).",
                paramInfo_.nodeType.c_str(), dimNames[i].c_str(), output1Dims[i], expectedDims[i]);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeType(), "y0, y1",
                VectorsToString(std::vector<std::vector<int64_t>>{expectedDims, output1Dims}, IntToString<int64_t>).c_str(),
                FormatString("When the dual_output attribute is true, the shapes of %s and %s must be the same",
                    "y0", "y1").c_str());
            return ge::GRAPH_FAILED;      
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckInputDesc()
{
    bool formatMatchTag = false;
    std::vector<std::vector<ge::Format>> supportFormats;
    std::stringstream ss;
    convBase_.GetSupportedFormats(flagInfo_.quantFlag, true, ss, supportFormats);
    for (uint8_t kindId = 0; kindId < supportFormats.size(); kindId++) {
        if (ConvArrMatch(paramInfo_.paramsFormat, supportFormats[kindId], paramInfo_.paramsFormat.size())) {
            formatMatchTag = true;
            break;
        }
    }
    if (!formatMatchTag) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params format [fmap, weight, output]: [%s, %s, %s]. only support %s",
            paramInfo_.nodeType.c_str(),
            formatToStrTab.at(descInfo_.fMapFormat).c_str(), formatToStrTab.at(descInfo_.weightFormat).c_str(),
            formatToStrTab.at(descInfo_.outFormat).c_str(), ss.str().c_str());
        string incorrectFormats = formatToStrTab.at(descInfo_.fMapFormat) + ", " +
                                  formatToStrTab.at(descInfo_.weightFormat) + ", " +
                                  formatToStrTab.at(descInfo_.outFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeType(), "x, filter, y", incorrectFormats.c_str(),
            FormatString("The formats of these parameters support only the following combinations: %s",
                ss.str().c_str()).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckParamsDtypeWithBias(std::vector<std::vector<ge::DataType>> supportedTypesList)
{
    if (!flagInfo_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    vector<ge::DataType> paramsType = {descInfo_.fMapDtype, descInfo_.weightDtype,
        descInfo_.outDtype, descInfo_.biasDtype};
    for (uint64_t kindsId = 0; kindsId < supportedTypesList.size(); kindsId++) {
        if (ConvArrMatchWithSize(paramsType, supportedTypesList[kindsId], paramsType.size())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params data type [fmap, weight, bias, output]: [%s, %s, %s, %s]",
            paramInfo_.nodeType.c_str(),
            GeDtypeToString(descInfo_.fMapDtype).c_str(),
            GeDtypeToString(descInfo_.weightDtype).c_str(),
            GeDtypeToString(descInfo_.biasDtype).c_str(),
            GeDtypeToString(descInfo_.outDtype).c_str());
    string incorrectDtypes = GeDtypeToString(descInfo_.fMapDtype) + ", " + 
                                GeDtypeToString(descInfo_.weightDtype) + ", " +
                                GeDtypeToString(descInfo_.biasDtype) + ", " +
                                GeDtypeToString(descInfo_.outDtype);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeType(), "x, filter, bias, y",
        incorrectDtypes.c_str(),
        FormatString("The dtypes of these parameters support only the following combinations: %s",
            VectorsToString(supportedTypesList, GeDtypeToString)).c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus Conv2dBaseTiling::CheckParamsDtypeWithoutBias(std::vector<std::vector<ge::DataType>> supportedTypesList)
{
    if (flagInfo_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    vector<ge::DataType> paramsType = {descInfo_.fMapDtype, descInfo_.weightDtype, descInfo_.outDtype};
    for (uint64_t kindsId = 0; kindsId < supportedTypesList.size(); kindsId++) {
        if (ConvArrMatchWithSize(paramsType, supportedTypesList[kindsId], paramsType.size())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params data type [fmap, weight, output]: [%s, %s, %s]",
            paramInfo_.nodeType.c_str(),
            GeDtypeToString(descInfo_.fMapDtype).c_str(),
            GeDtypeToString(descInfo_.weightDtype).c_str(),
            GeDtypeToString(descInfo_.outDtype).c_str());
    string incorrectDtypes = GeDtypeToString(descInfo_.fMapDtype) + ", " + 
                             GeDtypeToString(descInfo_.weightDtype) + ", " +
                             GeDtypeToString(descInfo_.outDtype);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeType(), "x, filter, y",
        incorrectDtypes.c_str(),
        FormatString("The dtypes of these parameters support only the following combinations: %s",
            VectorsToString(supportedTypesList, GeDtypeToString).c_str()).c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus Conv2dBaseTiling::CheckParamsDtype()
{
    // check int8 input not support  c04
    if (IsMdcSoc(opInfo_->npuArch) && descInfo_.weightFormat == ge::Format::FORMAT_FRACTAL_Z_C04 &&
        dtypeMap.at(descInfo_.fMapDtype) == ConvDtype::INT8) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: int8 input not support C04.", context_->GetNodeType());
        return ge::GRAPH_FAILED;
    }

    std::vector<std::vector<ge::DataType>> supportedTypesList;
    GetSupportedDataTypes(apiInputPlatformInfo.npuArch, flagInfo_.quantFlag,
                          descInfo_.fMapFormat, flagInfo_.extendConvFlag, supportedTypesList);
    OP_TILING_CHECK(supportedTypesList.size() == 0,
                    OP_LOGE(context_->GetNodeName(), "%s AscendC: Get supported types list fail.",
                    paramInfo_.nodeType.c_str()), return ge::GRAPH_FAILED);

    if (flagInfo_.hasBias) {
        return CheckParamsDtypeWithBias(supportedTypesList);
    } else {
        return CheckParamsDtypeWithoutBias(supportedTypesList);
    }
}

ge::graphStatus Conv2dBaseTiling::CheckQuantScaleLegal()
{
    if (!flagInfo_.quantFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto scaleDesc = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX);
    auto scaleDtype = scaleDesc->GetDataType();
    if (scaleDtype != ge::DataType::DT_INT64 && scaleDtype != ge::DataType::DT_UINT64) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: unSupported scale datatype: %s, only support [INT64] or [UINT64].",
                paramInfo_.nodeType.c_str(), dtypeToStrTab.at(scaleDtype).c_str());
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeType(), "scale",
            dtypeToStrTab.at(scaleDtype).c_str(), "int64 or uint64");
        return ge::GRAPH_FAILED;
    }
    auto scaleShapePtr = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    if (scaleShapePtr->GetStorageShape().GetDimNum() != FORMAT_ND_DIM) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input scale shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), scaleShapePtr->GetStorageShape().GetDimNum(), FORMAT_ND_DIM);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "scale",
            std::to_string(scaleShapePtr->GetStorageShape().GetDimNum()).c_str(),
            std::to_string(FORMAT_ND_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    auto weightShape = GetWeightShape(weightShapePtr);
    if (scaleShapePtr->GetStorageShape().GetDim(0) !=
        weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX])) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: input illegal scale shape: %ld, which must equal to Cout: %ld.",
                paramInfo_.nodeType.c_str(), scaleShapePtr->GetStorageShape().GetDim(0),
                weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_N_IDX]));
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeType(), "scale, filter",
            VectorsToString(std::vector<std::vector<int64_t>>{
                GetInputShapeVec(context_, INPUT_SCALE_INDEX),
                GetInputShapeVec(context_, INPUT_WEIGHT_INDEX)}, IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of %s must be equal to shape[%zu] of %s",
                0, "scale", paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX], "filter").c_str());
        return ge::GRAPH_FAILED;
    }
    fixpipeInfo_.channelWiseCoeff += INT64_DTYPE_SIZE_COMPARE_FP16;
    fixpipeInfo_.quantMode0 = static_cast<uint8_t>(optiling::conv_ops_tiling::QuantMode::VECTOR_QUANT);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendScaleLegal()
{
    if (!flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }

    // check scale0
    auto scale0Desc = context_->GetOptionalInputDesc(EXTENDCONV_INPUT_SCALE_0_INDEX);
    if (scale0Desc != nullptr &&
        !CheckScaleLegal(EXTENDCONV_INPUT_SCALE_0_INDEX, fixpipeInfo_.quantMode0, "scale0")) {
        return ge::GRAPH_FAILED;
    }

    // check scale1
    auto scale1Desc = context_->GetOptionalInputDesc(EXTENDCONV_INPUT_SCALE_1_INDEX);
    if (scale1Desc != nullptr &&
        !CheckScaleLegal(EXTENDCONV_INPUT_SCALE_1_INDEX, fixpipeInfo_.quantMode1, "scale1")) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool Conv2dBaseTiling::CheckScaleLegal(uint32_t scaleIndex, uint8_t& quantMode, const std::string& scaleType)
{
    auto scaleDesc = context_->GetOptionalInputDesc(scaleIndex);
    auto scaleDtype = scaleDesc->GetDataType();
    if (scaleDtype != ge::DataType::DT_INT64 && scaleDtype != ge::DataType::DT_UINT64) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: unSupported %s datatype: %s, only support [INT64] or [UINT64].",
                paramInfo_.nodeType.c_str(), scaleType.c_str(), dtypeToStrTab.at(scaleDtype).c_str());
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeType(), scaleType.c_str(),
            dtypeToStrTab.at(scaleDtype).c_str(), "int64 or uint64");
        return false;
    }
    if (scaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: unSupported %s format: %s, only support [ND].",
                paramInfo_.nodeType.c_str(), scaleType.c_str(),
                formatToStrTab.at(scaleDesc->GetStorageFormat()).c_str());
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeType(), scaleType.c_str(),
            formatToStrTab.at(scaleDesc->GetStorageFormat()).c_str(), "ND");
        return false;
    }
    auto scaleShapePtr = context_->GetOptionalInputShape(scaleIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    if (scaleShapePtr->GetStorageShape().GetDimNum() != FORMAT_ND_DIM) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input %s shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), scaleType.c_str(),
                scaleShapePtr->GetStorageShape().GetDimNum(), FORMAT_ND_DIM);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), scaleType.c_str(),
            std::to_string(scaleShapePtr->GetStorageShape().GetDimNum()).c_str(),
            std::to_string(FORMAT_ND_DIM).c_str());
        return false;
    }
    size_t scaleShapeLen = scaleShapePtr->GetStorageShape().GetDim(0);
    if (scaleShapeLen == 1) {
        // the type of scale is scalar when scaleShapeLen is 1
        quantMode = static_cast<uint8_t>(optiling::conv_ops_tiling::QuantMode::SCALAR_QUANT);
    } else {
        fixpipeInfo_.channelWiseCoeff += INT64_DTYPE_SIZE_COMPARE_FP16;
        quantMode = static_cast<uint8_t>(optiling::conv_ops_tiling::QuantMode::VECTOR_QUANT);
    }

    return true;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendReluWeightLegal()
{
    if (!flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendClipValueLegal()
{
    if (!flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendDualOutputSpecial()
{
    if (!flagInfo_.extendConvFlag || attrInfo_.dualOutput == 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (descInfo_.out1Format != descInfo_.outFormat) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: dual output1 format: %s should be same as output format: %s.",
                paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.out1Format).c_str(),
                formatToStrTab.at(descInfo_.outFormat).c_str());
        string incorrectFormats = formatToStrTab.at(descInfo_.outFormat) + ", " +
                                  formatToStrTab.at(descInfo_.out1Format);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeType(), "y0, y1", incorrectFormats.c_str(),
            FormatString("When the dual_output attribute is true, the formats of %s and %s must be the same",
                "y0", "y1").c_str());
        return ge::GRAPH_FAILED;
    }
    if (!(descInfo_.out1Dtype == ge::DT_FLOAT16 && descInfo_.outDtype == ge::DT_INT8) &&
        !(descInfo_.out1Dtype == ge::DT_INT8 && descInfo_.outDtype == ge::DT_FLOAT16) &&
        !(descInfo_.out1Dtype == ge::DT_FLOAT16 && descInfo_.outDtype == ge::DT_FLOAT16) &&
        !(descInfo_.out1Dtype == ge::DT_INT8 && descInfo_.outDtype == ge::DT_INT8)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Output data types must be one INT8 and one FLOAT16 or both INT8",
            paramInfo_.nodeType.c_str());
        string incorrectDtypes = GeDtypeToString(descInfo_.outDtype) + ", " + 
                                 GeDtypeToString(descInfo_.out1Dtype);
        std::stringstream ss;
        ss << "When the dual_output attribute is true, the dtypes of %s and %s must be one of %s";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeType(), "y0, y1",
            incorrectDtypes.c_str(),
            FormatString(ss.str().c_str(), "y0", "y1", "{int8, float16}").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}
}