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
 * \file conv3d_v2_base_tiling_check_input.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
 
namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv3dBaseTilingV2::ParseFmapShape()
{
    auto fMapShapePtr = context_->GetInputShape(INPUT_FMAP_INDEX);
    auto fMapShape = fMapShapePtr->GetStorageShape();
    if (fMapShape.GetDimNum() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input feature map shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), fMapShape.GetDimNum(), CONV3D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "x",
            std::to_string(fMapShape.GetDimNum()).c_str(),
            std::to_string(CONV3D_DIM_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriFmapN = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_N_IDX]);
    oriShapeAttrInfo_.oriFmapC = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_C_IDX]);
    oriShapeAttrInfo_.oriFmapD = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_D_IDX]);
    oriShapeAttrInfo_.oriFmapH = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_H_IDX]);
    oriShapeAttrInfo_.oriFmapW = fMapShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_W_IDX]);
    shapeInfo_.batch = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapN);
    shapeInfo_.ci = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapC);
    shapeInfo_.di = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapD);
    shapeInfo_.hi = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapH);
    shapeInfo_.wi = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapW);
    // note: cin will be checked in weightshape
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseWeightShape()
{
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    auto weightShape = weightShapePtr->GetStorageShape();
    if (weightShape.GetDimNum() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input weight shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), weightShape.GetDimNum(), CONV3D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "filter",
            std::to_string(weightShape.GetDimNum()).c_str(),
            std::to_string(CONV3D_DIM_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriWeightN = weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX]);
    oriShapeAttrInfo_.oriWeightC = weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_C_IDX]);
    oriShapeAttrInfo_.oriWeightD = weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_D_IDX]);
    oriShapeAttrInfo_.oriWeightH = weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_H_IDX]);
    oriShapeAttrInfo_.oriWeightW = weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_W_IDX]);
    shapeInfo_.kd = static_cast<uint64_t>(oriShapeAttrInfo_.oriWeightD);
    shapeInfo_.kh = static_cast<uint64_t>(oriShapeAttrInfo_.oriWeightH);
    shapeInfo_.kw = static_cast<uint64_t>(oriShapeAttrInfo_.oriWeightW);
    shapeInfo_.co = static_cast<uint64_t>(oriShapeAttrInfo_.oriWeightN);

    auto k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_K_IDX);
    if (k0 == 0) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Get k0 = 0.", paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckFmapShape()
{
    if (oriShapeAttrInfo_.oriFmapN < 1 || oriShapeAttrInfo_.oriFmapN > MAX_N_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Batch (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapN, MAX_N_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_N_IDX], 1, MAX_N_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapC < 1 || oriShapeAttrInfo_.oriFmapC > MAX_CIN_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Cin (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapC, MAX_CIN_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_C_IDX], 1, MAX_CIN_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapD < 1 || oriShapeAttrInfo_.oriFmapD > MAX_FM_D_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Din (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapD, MAX_FM_D_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_D_IDX], 1, MAX_FM_D_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapH < 1 || oriShapeAttrInfo_.oriFmapH > MAX_FM_H_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Hin (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapH, MAX_FM_H_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_H_IDX], 1, MAX_FM_H_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapW < 1 || oriShapeAttrInfo_.oriFmapW > MAX_FM_W_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Win (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapW, MAX_FM_W_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "x",
            VectorToString(GetInputShapeVec(context_, INPUT_FMAP_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.FMAP_PARAM_IDX][IDX_LIST_W_IDX], 1, MAX_FM_W_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckWeightShape()
{
    if (oriShapeAttrInfo_.oriWeightD < 1 || oriShapeAttrInfo_.oriWeightD > MAX_KD_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: KD (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightD, MAX_KD_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "filter",
            VectorToString(GetInputShapeVec(context_, INPUT_WEIGHT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_D_IDX], 1, MAX_KD_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriWeightH < 1 || oriShapeAttrInfo_.oriWeightH > MAX_KH_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: KH (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightH, MAX_KH_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "filter",
            VectorToString(GetInputShapeVec(context_, INPUT_WEIGHT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_H_IDX], 1, MAX_KH_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriWeightW < 1 || oriShapeAttrInfo_.oriWeightW > MAX_KW_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: KW (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightW, MAX_KW_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "filter",
            VectorToString(GetInputShapeVec(context_, INPUT_WEIGHT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_W_IDX], 1, MAX_KW_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriWeightN < 1 || oriShapeAttrInfo_.oriWeightN > MAX_COUT_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Cout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightN, MAX_COUT_BF16_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "filter",
            VectorToString(GetInputShapeVec(context_, INPUT_WEIGHT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX], 1, MAX_COUT_BF16_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool Conv3dBaseTilingV2::GetPosByFormat(const ge::Format format, const std::string &pos,
                                      const std::string &inputStr, size_t &posIdx) const
{
    string formatStr = formatToStrTab[format];
    // util func illegal scene protect, not a useful dfx
    OP_LOGE_IF(formatStr.length() != CONV3D_DIM_SIZE_LIMIT, false, context_->GetNodeName(),
        "%s AscendC: %s format is not 5D.", paramInfo_.nodeType.c_str(), inputStr.c_str());
    OP_LOGE_IF(pos.length() != 1 || formatStr.find(pos) == string::npos, false, context_->GetNodeName(),
        "%s AscendC: pos %s not in 5d format: %s", paramInfo_.nodeType.c_str(), pos.c_str(), formatStr.c_str());
    posIdx = formatStr.find(pos);
    return true;
}

ge::graphStatus Conv3dBaseTilingV2::ParseBiasShape()
{
    if (flagInfo_.isConv3dDequant && !flagInfo_.hasBias) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeType(), "bias", "nullptr",
            FormatString("If the dtype of input x is int8, parameter %s must be set", "bias").c_str()
        );
        return ge::GRAPH_FAILED;
    }
    auto biasShapePtr = flagInfo_.quantFlag ?
        context_->GetOptionalInputShape(QUANT_INPUT_BIAS_INDEX) : context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (!flagInfo_.hasBias || biasShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasDimNum = biasShapePtr->GetStorageShape().GetDimNum();
    if (biasDimNum != FORMAT_ND_DIM && biasDimNum != CONV3D_DIM_SIZE_LIMIT) {
        string expectDims = std::to_string(FORMAT_ND_DIM) + " or " + std::to_string(CONV3D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "bias",
            std::to_string(biasDimNum).c_str(), expectDims.c_str());
        return ge::GRAPH_FAILED;
    }
    size_t idxC = 0;
    auto biasDesc = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasFormat = static_cast<ge::Format>(GetPrimaryFormat(biasDesc->GetStorageFormat()));
    if (biasDimNum == CONV3D_DIM_SIZE_LIMIT && !GetPosByFormat(biasFormat, "C", "Bias", idxC)) {
        return ge::GRAPH_FAILED;
    }
    auto weightShape = context_->GetInputShape(INPUT_WEIGHT_INDEX)->GetStorageShape();
    size_t weightIdx = paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX];
    for (size_t i = 0; i < biasDimNum; i++) {
        if (i == idxC) {
            if (biasShapePtr->GetStorageShape().GetDim(i) != weightShape.GetDim(weightIdx)) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeType(), "bias, filter",
                    VectorsToString(std::vector<std::vector<int64_t>>{GetInputShapeVec(context_, INPUT_BIAS_INDEX),
                        GetInputShapeVec(context_, INPUT_WEIGHT_INDEX)}, IntToString<int64_t>).c_str(),
                    FormatString("Shape[%zu] of %s must be equal to shape[%zu] of %s",
                        i, "bias", weightIdx, "filter").c_str());
                return ge::GRAPH_FAILED;
            }
        } else if (biasShapePtr->GetStorageShape().GetDim(i) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "bias",
                VectorToString(GetInputShapeVec(context_, INPUT_BIAS_INDEX), IntToString<int64_t>).c_str(),
                FormatString("Shape[%zu] of this parameter must be equal to %d", i, 1).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckOutputShape()
{
    if (oriShapeAttrInfo_.oriOutputD < 1 || oriShapeAttrInfo_.oriOutputD > MAX_DOUT_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Dout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputD, MAX_DOUT_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "y",
            VectorToString(GetOutputShapeVec(context_, OUTPUT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_D_IDX], 1, MAX_DOUT_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriOutputH < 1 || oriShapeAttrInfo_.oriOutputH > MAX_HOUT_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Hout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputH, MAX_HOUT_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "y",
            VectorToString(GetOutputShapeVec(context_, OUTPUT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_H_IDX], 1, MAX_HOUT_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriOutputW < 1 || oriShapeAttrInfo_.oriOutputW > MAX_WOUT_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Wout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputW, MAX_WOUT_SHAPE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeType(), "y",
            VectorToString(GetOutputShapeVec(context_, OUTPUT_INDEX), IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of this parameter must be within the range [%ld, %ld]",
                paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_W_IDX], 1, MAX_WOUT_SHAPE).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckInputDesc()
{
    bool isConv3dDequantFormatLegal = descInfo_.fMapFormat == ge::FORMAT_NCDHW &&
                                        descInfo_.weightFormat == ge::FORMAT_NCDHW &&
                                        descInfo_.outFormat == ge::FORMAT_NDHWC;
    std::stringstream ss;
    ss << "[NCDHW, NCDHW, NDHWC]";
    if (flagInfo_.isConv3dDequant && !isConv3dDequantFormatLegal) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params format [fmap, weight, output]: [%s, %s, %s]. %s",
            paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.fMapFormat).c_str(),
            formatToStrTab.at(descInfo_.weightFormat).c_str(), formatToStrTab.at(descInfo_.outFormat).c_str(),
            ss.str().c_str());
        string incorrectFormats = formatToStrTab.at(descInfo_.fMapFormat) + ", " +
                                  formatToStrTab.at(descInfo_.weightFormat) + ", " +
                                  formatToStrTab.at(descInfo_.outFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeType(), "x, filter, y", incorrectFormats.c_str(),
            FormatString("If the dtype of input x is int8, the formats of these parameters must be %s",
                ss.str().c_str()).c_str());
        return ge::GRAPH_FAILED;
    }

    auto res = CheckInputDescForND();
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseOutputShape()
{
    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    auto outputShape = outputShapePtr->GetStorageShape();
    if (outputShape.GetDimNum() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: output shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), outputShape.GetDimNum(), CONV3D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "y",
            std::to_string(outputShape.GetDimNum()).c_str(),
            std::to_string(CONV3D_DIM_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriOutputN = outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_N_IDX]);
    oriShapeAttrInfo_.oriOutputC = outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_C_IDX]);
    oriShapeAttrInfo_.oriOutputD = outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_D_IDX]);
    oriShapeAttrInfo_.oriOutputH = outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_H_IDX]);
    oriShapeAttrInfo_.oriOutputW = outputShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.OUT_PARAM_IDX][IDX_LIST_W_IDX]);
    shapeInfo_.dout = static_cast<uint64_t>(oriShapeAttrInfo_.oriOutputD);
    shapeInfo_.ho = static_cast<uint64_t>(oriShapeAttrInfo_.oriOutputH);
    shapeInfo_.wo = static_cast<uint64_t>(oriShapeAttrInfo_.oriOutputW);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckQuantScaleDtype()
{
    if (flagInfo_.quantFlag) {
        if (descInfo_.scaleDtype != ge::DataType::DT_INT64 && descInfo_.scaleDtype != ge::DataType::DT_UINT64) {
            OP_LOGE(context_->GetNodeName(),
                    "%s AscendC: unSupported scale datatype: %s, only support int64 and uint64.",
                    paramInfo_.nodeType.c_str(), dtypeToStrTab.at(descInfo_.scaleDtype).c_str());
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeType(), "scale",
                dtypeToStrTab.at(descInfo_.scaleDtype).c_str(), "int64 or uint64");
            return ge::GRAPH_FAILED;
        }
        fixpipeInfo_.channelWiseCoeff += INT64_DTYPE_SIZE_COMPARE_FP16;
    } else {
        // conv3d
        if (descInfo_.scaleDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context_->GetNodeName(),
                    "%s AscendC: unSupported scale datatype: %s, only support float32.",
                    paramInfo_.nodeType.c_str(), dtypeToStrTab.at(descInfo_.scaleDtype).c_str());
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeType(), "scale",
                GeDtypeToString(descInfo_.scaleDtype).c_str(),
                FormatString("If the dtype of input x is int8, the dtype of parameter %s must be %s",
                    "scale", "float32").c_str());
            return ge::GRAPH_FAILED;
        }
        // scale will in UB, skip fbsize calc
        fixpipeInfo_.channelWiseCoeff = 0;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckQuantScaleDesc()
{
    if (!(flagInfo_.isConv3dDequant || flagInfo_.quantFlag)) {
        return ge::GRAPH_SUCCESS;
    }

    if (flagInfo_.isConv3dDequant && !flagInfo_.hasScale) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: conv3d scale can not be null when input is int8.",
                paramInfo_.nodeType.c_str());
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeType(), "scale", "nullptr",
            FormatString("If the dtype of input x is int8, parameter %s must be set", "scale").c_str()
        );
        return ge::GRAPH_FAILED;
    }

    if (static_cast<ge::Format>(descInfo_.scaleFormat) != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: unSupported scale format: %s, only support [ND].",
                paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.scaleFormat).c_str());
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeType(), "scale",
            formatToStrTab.at(descInfo_.scaleFormat).c_str(), "ND");
        return ge::GRAPH_FAILED;
    }
    auto tmpScaleIdx = flagInfo_.quantFlag ? QUANT_CONV_SCALE_IDX : CONV_SCALE_IDX;
    auto scaleShapePtr = context_->GetOptionalInputShape(tmpScaleIdx);
    if (scaleShapePtr->GetStorageShape().GetDimNum() != FORMAT_ND_DIM) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input scale shape dim num: %zu != %u.",
                paramInfo_.nodeType.c_str(), scaleShapePtr->GetStorageShape().GetDimNum(), FORMAT_ND_DIM);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeType(), "scale",
            std::to_string(scaleShapePtr->GetStorageShape().GetDimNum()).c_str(),
            std::to_string(FORMAT_ND_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    size_t scaleShape = scaleShapePtr->GetStorageShape().GetDim(0);
    if (scaleShape != static_cast<size_t>(oriShapeAttrInfo_.oriWeightN)) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: input illegal scale shape: %zu, which must equal to Cout: %ld.",
                paramInfo_.nodeType.c_str(), scaleShape, oriShapeAttrInfo_.oriWeightN);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeType(), "scale, filter",
            VectorsToString(std::vector<std::vector<int64_t>>{
                GetInputShapeVec(context_, tmpScaleIdx),
                GetInputShapeVec(context_, INPUT_WEIGHT_INDEX)}, IntToString<int64_t>).c_str(),
            FormatString("Shape[%zu] of %s must be equal to shape[%zu] of %s",
                0, "scale", paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX], "filter").c_str());
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantScaleDtype() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckParamsDtype()
{
    std::vector<ge::DataType> paramsType;
    if (flagInfo_.hasBias) {
        paramsType = {descInfo_.fMapDtype, descInfo_.weightDtype, descInfo_.biasDtype, descInfo_.outDtype};
    } else {
        paramsType = {descInfo_.fMapDtype, descInfo_.weightDtype, descInfo_.outDtype};
    }
    std::vector<std::vector<ge::DataType>> supportedTypesList;
    GetSupportedDataTypes(flagInfo_.hasBias, flagInfo_.quantFlag, supportedTypesList);
    for (uint64_t kindsId = 0; kindsId < supportedTypesList.size(); ++kindsId) {
        if (supportedTypesList[kindsId].size() == 0 || paramsType.size() != supportedTypesList[kindsId].size()) {
            continue;
        }
        if (ConvArrMatch(paramsType, supportedTypesList[kindsId], supportedTypesList[kindsId].size())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    if (flagInfo_.hasBias) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params data type [fmap, weight, bias, output]: [%s, %s, %s, %s].",
            paramInfo_.nodeType.c_str(), 
            GeDtypeToString(descInfo_.fMapDtype).c_str(), GeDtypeToString(descInfo_.weightDtype).c_str(),
            GeDtypeToString(descInfo_.biasDtype).c_str(), GeDtypeToString(descInfo_.outDtype).c_str());
            string incorrectDtypes = GeDtypeToString(descInfo_.fMapDtype) + ", " + 
                GeDtypeToString(descInfo_.weightDtype) + ", " + GeDtypeToString(descInfo_.biasDtype) + ", " +
                GeDtypeToString(descInfo_.outDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeType(), "x, filter, bias, y",
                incorrectDtypes.c_str(),
                FormatString("The dtypes of these parameters support only the following combinations: %s",
                    VectorsToString(supportedTypesList, GeDtypeToString).c_str()).c_str());
    } else { 
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params data type [fmap, weight, output]: [%s, %s, %s].",
            paramInfo_.nodeType.c_str(), 
            GeDtypeToString(descInfo_.fMapDtype).c_str(), GeDtypeToString(descInfo_.weightDtype).c_str(),
            GeDtypeToString(descInfo_.outDtype).c_str());
            string incorrectDtypes = GeDtypeToString(descInfo_.fMapDtype) + ", " + 
                GeDtypeToString(descInfo_.weightDtype) + ", " + GeDtypeToString(descInfo_.outDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeType(), "x, filter, y", incorrectDtypes.c_str(),
                FormatString("The dtypes of these parameters support only the following combinations: %s",
                    VectorsToString(supportedTypesList, GeDtypeToString).c_str()).c_str());
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus Conv3dBaseTilingV2::CheckInputDescForND()
{
    bool formatMatchTag = false;
    std::vector<std::vector<ge::Format>> supportFormats;
    std::stringstream ss;
    convBase_.GetSupportedFormats(flagInfo_.quantFlag, false, ss, supportFormats);
    for (uint8_t kindId = 0; kindId < supportFormats.size(); kindId++) {
        if (ConvArrMatch(paramInfo_.paramsFormat, supportFormats[kindId], paramInfo_.paramsFormat.size())) {
            formatMatchTag = true;
            break;
        }
    }
    if (!formatMatchTag) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params format [fmap, weight, output]: [%s, %s, %s]. only support %s",
            paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.fMapFormat).c_str(),
            formatToStrTab.at(descInfo_.weightFormat).c_str(), formatToStrTab.at(descInfo_.outFormat).c_str(),
            ss.str().c_str());
        string incorrectFormats = formatToStrTab.at(descInfo_.fMapFormat) + ", " +
            formatToStrTab.at(descInfo_.weightFormat) + ", " + formatToStrTab.at(descInfo_.outFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeType(), "x, filter, y", incorrectFormats.c_str(),
            FormatString("The formats of these parameters support only the following combinations: %s",
                ss.str().c_str()).c_str());
        return ge::GRAPH_FAILED;
    }


    return ge::GRAPH_SUCCESS;
}
}
}