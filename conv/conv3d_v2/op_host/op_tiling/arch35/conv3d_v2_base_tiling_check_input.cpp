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
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Batch (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapN, MAX_N_BF16_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapC < 1 || oriShapeAttrInfo_.oriFmapC > MAX_CIN_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Cin (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapC, MAX_CIN_BF16_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapD < 1 || oriShapeAttrInfo_.oriFmapD > MAX_FM_D_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Din (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapD, MAX_FM_D_BF16_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapH < 1 || oriShapeAttrInfo_.oriFmapH > MAX_FM_H_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Hin (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapH, MAX_FM_H_BF16_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriFmapW < 1 || oriShapeAttrInfo_.oriFmapW > MAX_FM_W_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Win (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriFmapW, MAX_FM_W_BF16_SHAPE);
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
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriWeightH < 1 || oriShapeAttrInfo_.oriWeightH > MAX_KH_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: KH (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightH, MAX_KH_BF16_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriWeightW < 1 || oriShapeAttrInfo_.oriWeightW > MAX_KW_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: KW (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightW, MAX_KW_BF16_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriWeightC < 1 || oriShapeAttrInfo_.oriWeightC > MAX_CIN_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Cin (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightC, MAX_CIN_BF16_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriWeightN < 1 || oriShapeAttrInfo_.oriWeightN > MAX_COUT_BF16_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Cout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriWeightN, MAX_COUT_BF16_SHAPE);
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
        OP_LOGE(context_->GetNodeName(), "%s AscendC: not support conv3d int8 bias null.", paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }

    auto biasShapePtr = flagInfo_.quantFlag ?
        context_->GetOptionalInputShape(QUANT_INPUT_BIAS_INDEX) : context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (!flagInfo_.hasBias || biasShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasDimNum = biasShapePtr->GetStorageShape().GetDimNum();
    if (biasDimNum != FORMAT_ND_DIM && biasDimNum != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input bias shape dim num: %zu , but only support %u and %u.",
                paramInfo_.nodeType.c_str(), biasDimNum, FORMAT_ND_DIM, CONV3D_DIM_SIZE_LIMIT);
        return ge::GRAPH_FAILED;
    }
 
    size_t idxC = 0;
    auto biasDesc = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasFormat = static_cast<ge::Format>(GetPrimaryFormat(biasDesc->GetStorageFormat()));
    const string biasStr = "Bias";
    if (biasDimNum == CONV3D_DIM_SIZE_LIMIT) {
        if (!GetPosByFormat(biasFormat, "C", biasStr, idxC)) {
            return ge::GRAPH_FAILED;
        }
    }
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    auto weightShape = weightShapePtr->GetStorageShape();
    // when biasDimNum is 1, idxC is 0
    for (size_t i = 0; i < biasDimNum; i++) {
        if (i == idxC) {
            if (biasShapePtr->GetStorageShape().GetDim(i) !=
                weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX])) {
                OP_LOGE(context_->GetNodeName(),
                        "%s AscendC: input illegal bias shape: %ld, which must equal to Cout: %ld.",
                        paramInfo_.nodeType.c_str(), biasShapePtr->GetStorageShape().GetDim(i),
                        weightShape.GetDim(paramInfo_.paramsIdxVec[paramInfo_.WEIGHT_PARAM_IDX][IDX_LIST_N_IDX]));
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (biasShapePtr->GetStorageShape().GetDim(i) != 1) {
            OP_LOGE(context_->GetNodeName(), "%s AscendC: input bias shape dim %zu: %ld, but only support 1.",
                paramInfo_.nodeType.c_str(), i, biasShapePtr->GetStorageShape().GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
 
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckBiasShape()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckOutputShape()
{
    if (oriShapeAttrInfo_.oriOutputD < 1 || oriShapeAttrInfo_.oriOutputD > MAX_DOUT_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Dout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputD, MAX_DOUT_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriOutputH < 1 || oriShapeAttrInfo_.oriOutputH > MAX_HOUT_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Hout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputH, MAX_HOUT_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriOutputW < 1 || oriShapeAttrInfo_.oriOutputW > MAX_WOUT_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Wout (%ld) is out of range[1, %ld].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOutputW, MAX_WOUT_SHAPE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckInputDesc()
{
    if (socVersion == SocVersion::ASCEND910_95 || socVersion == SocVersion::ASCEND910_55) {
        bool isConv3dDequantFormatLegal = descInfo_.fMapFormat == ge::FORMAT_NCDHW &&
                                          descInfo_.weightFormat == ge::FORMAT_NCDHW &&
                                          descInfo_.outFormat == ge::FORMAT_NDHWC;
        std::stringstream ss;
        ss <<"The support params format list [fmap, weight, output] for scene ";
        ss <<"(conv3d dequant) is [NCDHW, NCDHW, NDHWC]";
        if (flagInfo_.isConv3dDequant && !isConv3dDequantFormatLegal) {
            OP_LOGE(context_->GetNodeName(),
                "%s AscendC: unSupported params format [fmap, weight, output]: [%s, %s, %s]. %s",
                paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.fMapFormat).c_str(),
                formatToStrTab.at(descInfo_.weightFormat).c_str(), formatToStrTab.at(descInfo_.outFormat).c_str(),
                ss.str().c_str());
            return ge::GRAPH_FAILED;
        }

        auto res = CheckInputDescForND();
        if (res != ge::GRAPH_SUCCESS) {
            return res;
        }
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

ge::graphStatus Conv3dBaseTilingV2::CheckQuantScaleDesc()
{
    if (!(flagInfo_.isConv3dDequant || flagInfo_.quantFlag)) {
        return ge::GRAPH_SUCCESS;
    }

    if (flagInfo_.isConv3dDequant && !flagInfo_.hasScale) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: conv3d scale can not be null when input is int8.",
                paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }

    if (static_cast<ge::Format>(descInfo_.scaleFormat) != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: unSupported scale format: %s, only support [ND].",
                paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.scaleFormat).c_str());
        return ge::GRAPH_FAILED;
    }
    auto tmpScaleIdx = flagInfo_.quantFlag ? QUANT_CONV_SCALE_IDX : CONV_SCALE_IDX;
    auto scaleShapePtr = context_->GetOptionalInputShape(tmpScaleIdx);
    OP_TILING_CHECK(scaleShapePtr->GetStorageShape().GetDimNum() != FORMAT_ND_DIM,
                    OP_LOGE(context_->GetNodeName(), "%s AscendC: input scale shape dim num: %zu != %u.",
                            paramInfo_.nodeType.c_str(), scaleShapePtr->GetStorageShape().GetDimNum(), FORMAT_ND_DIM),
                    return ge::GRAPH_FAILED);
    size_t scaleShape = scaleShapePtr->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(scaleShape != static_cast<size_t>(oriShapeAttrInfo_.oriWeightN),
                    OP_LOGE(context_->GetNodeName(),
                            "%s AscendC: input illegal scale shape: %zu, which must equal to Cout: %ld.",
                            paramInfo_.nodeType.c_str(), scaleShape, oriShapeAttrInfo_.oriWeightN),
                    return ge::GRAPH_FAILED);

    if (flagInfo_.quantFlag) {
        if (descInfo_.scaleDtype != ge::DataType::DT_INT64 && descInfo_.scaleDtype != ge::DataType::DT_UINT64) {
            OP_LOGE(context_->GetNodeName(),
                    "%s AscendC: unSupported scale datatype: %s, only support int64 and uint64.",
                    paramInfo_.nodeType.c_str(), dtypeToStrTab.at(descInfo_.scaleDtype).c_str());
            return ge::GRAPH_FAILED;
        }
        fixpipeInfo_.channelWiseCoeff += INT64_DTYPE_SIZE_COMPARE_FP16;
    } else {
        // conv3d
        if (descInfo_.scaleDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context_->GetNodeName(),
                    "%s AscendC: unSupported scale datatype: %s, only support float32.",
                    paramInfo_.nodeType.c_str(), dtypeToStrTab.at(descInfo_.scaleDtype).c_str());
            return ge::GRAPH_FAILED;
        }
        // scale will in UB, skip fbsize calc
        fixpipeInfo_.channelWiseCoeff = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckParamsDtype()
{
    std::vector<ConvDtype> paramsType;
    if (flagInfo_.hasBias) {
        paramsType = {
            dtypeMap[descInfo_.fMapDtype], dtypeMap[descInfo_.weightDtype],
            dtypeMap[descInfo_.biasDtype], dtypeMap[descInfo_.outDtype]};
    } else {
        paramsType = {
            dtypeMap[descInfo_.fMapDtype], dtypeMap[descInfo_.weightDtype],
            dtypeMap[descInfo_.outDtype]};
    }
    std::vector<std::vector<ConvDtype>> supportedTypesList;
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
            DTYPE_TO_STR.at(dtypeMap.at(descInfo_.fMapDtype)).c_str(),
            DTYPE_TO_STR.at(dtypeMap.at(descInfo_.weightDtype)).c_str(),
            DTYPE_TO_STR.at(dtypeMap.at(descInfo_.biasDtype)).c_str(),
            DTYPE_TO_STR.at(dtypeMap.at(descInfo_.outDtype)).c_str());
    } else { 
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params data type [fmap, weight, output]: [%s, %s, %s].",
            paramInfo_.nodeType.c_str(), 
            DTYPE_TO_STR.at(dtypeMap.at(descInfo_.fMapDtype)).c_str(),
            DTYPE_TO_STR.at(dtypeMap.at(descInfo_.weightDtype)).c_str(),
            DTYPE_TO_STR.at(dtypeMap.at(descInfo_.outDtype)).c_str());
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus Conv3dBaseTilingV2::CheckInputDescForND()
{
    bool formatMatchTag = false;
    std::vector<std::vector<ge::Format>> supportFormats;
    std::stringstream ss;
    convBase_.GetSupportedFormats(flagInfo_.quantFlag, false, apiInputPlatformInfo_.socVersion, ss, supportFormats);
    for (uint8_t kindId = 0; kindId < supportFormats.size(); kindId++) {
        if (ConvArrMatch(paramInfo_.paramsFormat, supportFormats[kindId], paramInfo_.paramsFormat.size())) {
            formatMatchTag = true;
            break;
        }
    }
    if (!formatMatchTag) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: unSupported params format [fmap, weight, output]: [%s, %s, %s]. %s",
            paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.fMapFormat).c_str(),
            formatToStrTab.at(descInfo_.weightFormat).c_str(), formatToStrTab.at(descInfo_.outFormat).c_str(),
            ss.str().c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
}
}