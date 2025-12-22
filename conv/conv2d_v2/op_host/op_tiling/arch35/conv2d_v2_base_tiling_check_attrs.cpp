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
 * \file conv2d_v2_base_tiling_check_attrs.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"
 
namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv2dBaseTiling::CheckStrideLegal()
{
    uint32_t attrStrideIndex = flagInfo_.quantFlag ? ATTR_QUANT_STRIDE_INDEX : ATTR_STRIDE_INDEX;
    attrStrideIndex = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_STRIDES_INDEX : attrStrideIndex;
    auto stridePtr = context_->GetAttrs()->GetListInt(attrStrideIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
    if (stridePtr->GetSize() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input attr stride dim: %zu != %u.",
            paramInfo_.nodeType.c_str(), stridePtr->GetSize(), CONV2D_DIM_SIZE_LIMIT);
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriStrideN = stridePtr->GetData()[conv2dOriginFormatAixsPosInfo_.nIndex];
    oriShapeAttrInfo_.oriStrideC = stridePtr->GetData()[conv2dOriginFormatAixsPosInfo_.cIndex];
    oriShapeAttrInfo_.oriStrideH = stridePtr->GetData()[conv2dOriginFormatAixsPosInfo_.hIndex];
    oriShapeAttrInfo_.oriStrideW = stridePtr->GetData()[conv2dOriginFormatAixsPosInfo_.wIndex];
    if (oriShapeAttrInfo_.oriStrideH <= 0 || oriShapeAttrInfo_.oriStrideW <= 0 ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriStrideH) > MAX_ATTRS_SHAPE ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriStrideW) > MAX_ATTRS_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: stride (H: %ld, W: %ld) is out of range[1, %lu].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriStrideH, oriShapeAttrInfo_.oriStrideW, MAX_ATTRS_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriStrideN != 1 || oriShapeAttrInfo_.oriStrideC != 1) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: stride (N: %ld, C: %ld) should be 1.",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriStrideN, oriShapeAttrInfo_.oriStrideC);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckDilationLegal()
{
    uint32_t attrDilationIndex = flagInfo_.quantFlag ? ATTR_QUANT_DILATION_INDEX : ATTR_DILATION_INDEX;
    attrDilationIndex = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_DILATIONS_INDEX : attrDilationIndex;
    auto dilationPtr = context_->GetAttrs()->GetListInt(attrDilationIndex);
    if (dilationPtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (dilationPtr->GetSize() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input attr dilation dim: %zu != %u.",
                paramInfo_.nodeType.c_str(), dilationPtr->GetSize(), CONV2D_DIM_SIZE_LIMIT);
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriDilationN = dilationPtr->GetData()[conv2dOriginFormatAixsPosInfo_.nIndex];
    oriShapeAttrInfo_.oriDilationC = dilationPtr->GetData()[conv2dOriginFormatAixsPosInfo_.cIndex];
    oriShapeAttrInfo_.oriDilationH = dilationPtr->GetData()[conv2dOriginFormatAixsPosInfo_.hIndex];
    oriShapeAttrInfo_.oriDilationW = dilationPtr->GetData()[conv2dOriginFormatAixsPosInfo_.wIndex];
    if (oriShapeAttrInfo_.oriDilationH <= 0 || oriShapeAttrInfo_.oriDilationW <= 0 ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriDilationH) > MAX_ATTRS_SHAPE ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriDilationW) > MAX_ATTRS_SHAPE) {
        OP_LOGE(context_->GetNodeName(), 
                "%s AscendC: dilation (H: %ld, W: %ld) is out of range[1, %lu].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriDilationH,
                oriShapeAttrInfo_.oriDilationW, MAX_ATTRS_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriDilationN != 1 || oriShapeAttrInfo_.oriDilationC != 1) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: dilation (N: %ld, C: %ld) should be 1.",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriDilationN, oriShapeAttrInfo_.oriDilationC);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckPadLegal()
{
    uint32_t attrPadIndex = flagInfo_.quantFlag ? ATTR_QUANT_PAD_INDEX : ATTR_PAD_INDEX;
    attrPadIndex = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_PADS_INDEX : attrPadIndex;
    auto padPtr = context_->GetAttrs()->GetListInt(attrPadIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padPtr);
    if (padPtr->GetSize() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input attr pad dim: %zu != %u.", paramInfo_.nodeType.c_str(),
                padPtr->GetSize(), CONV2D_DIM_SIZE_LIMIT);
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriPadTop = padPtr->GetData()[PAD_TOP_INDEX];
    oriShapeAttrInfo_.oriPadBottom = padPtr->GetData()[PAD_BOTTOM_INDEX];
    oriShapeAttrInfo_.oriPadLeft = padPtr->GetData()[PAD_LEFT_INDEX];
    oriShapeAttrInfo_.oriPadRight = padPtr->GetData()[PAD_RIGHT_INDEX];

    OP_LOGE_IF(!UpdateOriPadFromPadMode(), ge::GRAPH_FAILED,  context_->GetNodeName(),
        "%s AscendC: UpdateOriPadFromPadMode Failed.", paramInfo_.nodeType.c_str());

    if (oriShapeAttrInfo_.oriPadTop < 0 || oriShapeAttrInfo_.oriPadBottom < 0 ||
        oriShapeAttrInfo_.oriPadLeft < 0 || oriShapeAttrInfo_.oriPadRight < 0 ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriPadTop) > MAX_ATTRS_SHAPE ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriPadBottom) > MAX_ATTRS_SHAPE ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriPadLeft) > MAX_ATTRS_SHAPE ||
        static_cast<uint64_t>(oriShapeAttrInfo_.oriPadRight) > MAX_ATTRS_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: pad (top: %ld, bottom: %ld, left: %ld, right: %ld) is out of range[0, %lu].",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriPadTop, oriShapeAttrInfo_.oriPadBottom,
                oriShapeAttrInfo_.oriPadLeft, oriShapeAttrInfo_.oriPadRight,
                MAX_ATTRS_SHAPE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckRoundModeLegal()
{
    if (!flagInfo_.quantFlag && !flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t roundModeIndex = ATTR_QUANT_ROUNDMODE_INDEX;
    roundModeIndex = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_ROUND_MODE_INDEX : roundModeIndex;
    auto roundModePtr = context_->GetAttrs()->GetStr(roundModeIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, roundModePtr);
    string roundMode(roundModePtr);
    auto outputDesc = context_->GetOutputDesc(OUTPUT_INDEX);
    if (outputDesc->GetDataType() == ge::DataType::DT_INT8 ||
        outputDesc->GetDataType() == ge::DataType::DT_FLOAT8_E4M3FN) {
        if (roundMode != "rint") {
            OP_LOGE(context_->GetNodeName(),
                "%s AscendC: when outputType is int8 or float8_e4m3fn, round_mode only support rint, actually is %s",
                paramInfo_.nodeType.c_str(), roundMode.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (outputDesc->GetDataType() == ge::DataType::DT_HIFLOAT8) {
        if (roundMode != "round") {
            OP_LOGE(context_->GetNodeName(),
                "%s AscendC: when outputType is hifloat8, round_mode only support round, actually is %s",
                paramInfo_.nodeType.c_str(), roundMode.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (roundMode.empty()) {
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGW(context_->GetNodeName(), "%s AscendC: the input round_mode is suggested to be set as an empty str",
            paramInfo_.nodeType.c_str());
        if (!convBase_.CheckValidString(roundMode, context_)) {
            OP_LOGE(context_->GetNodeName(),
                "%s AscendC: the input round_mode has invalid str", paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckGroupsLegal()
{
    auto attrGroupsIndex = flagInfo_.quantFlag ? ATTR_QUANT_GROUP_INDEX : ATTR_GROUP_INDEX;
    attrGroupsIndex = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_GROUPS_INDEX : attrGroupsIndex;
    auto groupsPtr = context_->GetAttrs()->GetInt(attrGroupsIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, groupsPtr);
    oriShapeAttrInfo_.oriGroups = *groupsPtr;

    if (oriShapeAttrInfo_.oriGroups < 1 || static_cast<uint64_t>(oriShapeAttrInfo_.oriGroups) > MAX_GROUP_SHAPE) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: groups(%ld) from attr are out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriGroups, MAX_GROUP_SHAPE);
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.nodeType == "Conv2DV2" && oriShapeAttrInfo_.oriGroups == 1 && oriShapeAttrInfo_.oriWeightC != 0) {
        if (oriShapeAttrInfo_.oriFmapC % oriShapeAttrInfo_.oriWeightC == 0) {
            oriShapeAttrInfo_.oriGroups = oriShapeAttrInfo_.oriFmapC / oriShapeAttrInfo_.oriWeightC;
            OP_LOGD(context_->GetNodeName(),
                "%s AscendC: Attr groups is implicitly changed. ori groups %lu actual groups %lu",
                paramInfo_.nodeType.c_str(), *groupsPtr, oriShapeAttrInfo_.oriGroups);
        }
    }
    
    return ge::GRAPH_SUCCESS;
}

// optiling recaculate pad for kernel directed call process
void Conv2dBaseTiling::GetOriPadFromPadMode(const string& padMode)
{
    if (padMode == "SPECIFIC") {
        return;
    }

    if (padMode == "VALID") {
        oriShapeAttrInfo_.oriPadTop = 0;
        oriShapeAttrInfo_.oriPadBottom = 0;
        oriShapeAttrInfo_.oriPadLeft = 0;
        oriShapeAttrInfo_.oriPadRight = 0;
        return;
    } else {
        int64_t padH = (ConvCeilDiv(oriShapeAttrInfo_.oriFmapH, oriShapeAttrInfo_.oriStrideH) - 1) *
                    oriShapeAttrInfo_.oriStrideH + oriShapeAttrInfo_.oriDilationH * (oriShapeAttrInfo_.oriWeightH - 1) -
                    oriShapeAttrInfo_.oriFmapH + 1;
        int64_t padW = (ConvCeilDiv(oriShapeAttrInfo_.oriFmapW, oriShapeAttrInfo_.oriStrideW) - 1) *
                    oriShapeAttrInfo_.oriStrideW + oriShapeAttrInfo_.oriDilationW * (oriShapeAttrInfo_.oriWeightW - 1) -
                    oriShapeAttrInfo_.oriFmapW + 1;
        if (padMode == "SAME" || padMode == "SAME_UPPER") {
            if (padMode == "SAME") {
                padH = padH < 0 ? 0 : padH;
                padW = padW < 0 ? 0 : padW;
            }
            oriShapeAttrInfo_.oriPadBottom = ConvCeilDiv(padH, PAD_MODE_DIV_FACTOR);
            oriShapeAttrInfo_.oriPadTop = padH - oriShapeAttrInfo_.oriPadBottom;
            oriShapeAttrInfo_.oriPadRight = ConvCeilDiv(padW, PAD_MODE_DIV_FACTOR);
            oriShapeAttrInfo_.oriPadLeft = padW - oriShapeAttrInfo_.oriPadRight;
        } else {
            // padMode is "SAME_LOWER"
            oriShapeAttrInfo_.oriPadTop = ConvCeilDiv(padH, PAD_MODE_DIV_FACTOR);
            oriShapeAttrInfo_.oriPadBottom = padH - oriShapeAttrInfo_.oriPadTop;
            oriShapeAttrInfo_.oriPadLeft = ConvCeilDiv(padW, PAD_MODE_DIV_FACTOR);
            oriShapeAttrInfo_.oriPadRight = padW - oriShapeAttrInfo_.oriPadLeft;
        }
    }
    return;
}

bool Conv2dBaseTiling::UpdateOriPadFromPadMode()
{
    if (flagInfo_.quantFlag) {
        return true;
    }
     // Conv2DV2
    uint32_t padModeIndex = ATTR_PAD_MODE_INDEX;
    padModeIndex = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_PAD_MODE_INDEX : padModeIndex;
    auto padModePtr = context_->GetAttrs()->GetStr(padModeIndex);
    if (padModePtr == nullptr) {
        return true; // skip udpate
    }
    string padMode(padModePtr);
    auto iter = find(PADMODE_WHITELIST.begin(), PADMODE_WHITELIST.end(), padMode);
    if (iter == PADMODE_WHITELIST.end()) {
        std::stringstream ss;
        ss << paramInfo_.nodeType.c_str() <<" AscendC: Only support pad_mode in [VALID, SPECIFIC, SAME, SAME_UPPER, "
           << "SAME_LOWER], actually is: " << padMode.c_str();
        OP_LOGE(context_->GetNodeName(), "%s", ss.str().c_str());
        return false;
    }
    GetOriPadFromPadMode(padMode);
    return true;
}

ge::graphStatus Conv2dBaseTiling::CheckQuantDtypeLegal()
{
    if (!flagInfo_.quantFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto quantDtypePtr = context_->GetAttrs()->GetInt(ATTR_QUANT_DTYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, quantDtypePtr);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus Conv2dBaseTiling::CheckDataFormatLegal()
{
    auto attrDataFormatIndex = flagInfo_.quantFlag ? ATTR_QUANT_DATAFORMAT_INDEX : ATTR_DATAFORMAT_INDEX;
    attrDataFormatIndex =  flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_DATA_FORMAT_INDEX : attrDataFormatIndex;
    auto dataFormatPtr = context_->GetAttrs()->GetStr(attrDataFormatIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dataFormatPtr);
    string dataFormat(dataFormatPtr);
    if (dataFormat != "NCHW" && dataFormat != "NHWC") {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: unsupported dataFormat attr: %s",
                paramInfo_.nodeType.c_str(), dataFormat.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckOffsetXLegal()
{
    if (!flagInfo_.quantFlag && !flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t offsetXIndex  = ATTR_QUANT_OFFSETX_INDEX;
    offsetXIndex =  flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_OFFSET_X_INDEX : offsetXIndex;
    auto offsetXPtr = context_->GetAttrs()->GetInt(offsetXIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, offsetXPtr);
    oriShapeAttrInfo_.oriOffsetX = *offsetXPtr;
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    bool hif8Fp8Mode = ((fMapDesc->GetDataType() == ge::DataType::DT_HIFLOAT8) ||
        (fMapDesc->GetDataType() == ge::DataType::DT_FLOAT8_E4M3FN));
    if (hif8Fp8Mode && (oriShapeAttrInfo_.oriOffsetX != 0)) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Only support offset_x = 0 in hif8 or fp8 mode, actually is %ld",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOffsetX);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriOffsetX > OFFSET_X_MAX_VALUE || oriShapeAttrInfo_.oriOffsetX < OFFSET_X_MIN_VALUE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Only support offset_x in ([%d, %d]), actually is %ld",
                paramInfo_.nodeType.c_str(), OFFSET_X_MIN_VALUE, OFFSET_X_MAX_VALUE, oriShapeAttrInfo_.oriOffsetX);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckEnableHf32Legal()
{
    if (flagInfo_.quantFlag) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t enableHf32Index = ATTR_ENABLE_HF32_INDEX;
    enableHf32Index = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_ENABLE_HF32_INDEX :  enableHf32Index;
    auto enableHf32Ptr = context_->GetAttrs()->GetBool(enableHf32Index);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, enableHf32Ptr);
    bool IsFp32InputFp32OutputFlag = descInfo_.fMapDtype == ge::DataType::DT_FLOAT &&
        descInfo_.weightDtype == ge::DataType::DT_FLOAT &&
        descInfo_.outDtype == ge::DataType::DT_FLOAT;
 
    if (flagInfo_.hasBias) {
        IsFp32InputFp32OutputFlag &= descInfo_.biasDtype == ge::DataType::DT_FLOAT;
    }
    bool enbaleHf32 = *enableHf32Ptr;
    OP_TILING_CHECK(enbaleHf32 && !IsFp32InputFp32OutputFlag,
                    OP_LOGE(context_->GetNodeName(),
                            "%s AscendC: enable_hf32 can be true only when both input and output are FP32",
                            paramInfo_.nodeType.c_str()),
                    return ge::GRAPH_FAILED);
    attrInfo_.hf32Mode = static_cast<uint32_t>(enbaleHf32 ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendConv2dReluWeightAndClipValue(const uint32_t outputIdx, uint8_t& reluMode)
{
    // get input and attr desc according to outputIdx
    uint32_t reluWeightInputIdx =
        (outputIdx == 0 ? EXTENDCONV_INPUT_RELU_WIGHT_0_INDEX : EXTENDCONV_INPUT_RELU_WIGHT_1_INDEX);
    uint32_t clipValueInputIdx =
        (outputIdx == 0 ? EXTENDCONV_INPUT_CLIP_VALUE_0_INDEX : EXTENDCONV_INPUT_CLIP_VALUE_1_INDEX);
    uint32_t enabelReluInputIdx =
        (outputIdx == 0 ? EXTENDCONV_ATTR_ENABLE_RELU_0_INDEX : EXTENDCONV_ATTR_ENABLE_RELU_1_INDEX);
    // get enable relu and input ptr
    auto enableReluPtr = context_->GetAttrs()->GetBool(enabelReluInputIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, enableReluPtr);
    auto reluWeightPtr = context_->GetOptionalInputShape(reluWeightInputIdx);
    auto clipValuePtr = context_->GetOptionalInputShape(clipValueInputIdx);
    if (*enableReluPtr) {
        reluMode = static_cast<uint8_t>(ReluMode::NORMALRELU);
    }
    OP_LOGD(context_->GetNodeName(),
            "%s AscendC: reluMode is %u, enableRelu[%d] enableReluWeight[%d] enableClipValue[%d].",
            context_->GetNodeType(), reluMode, *enableReluPtr, reluWeightPtr == nullptr, clipValuePtr == nullptr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendReluLegal()
{
    if (!flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto enableRelu0Ptr = context_->GetAttrs()->GetBool(EXTENDCONV_ATTR_ENABLE_RELU_0_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, enableRelu0Ptr);
    auto enableRelu1Ptr = context_->GetAttrs()->GetBool(EXTENDCONV_ATTR_ENABLE_RELU_1_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, enableRelu1Ptr);

    // check scale1 and enablerelu1 when dualoutput is false
    if (!attrInfo_.dualOutput) {
        auto scale1Desc = context_->GetOptionalInputDesc(EXTENDCONV_INPUT_SCALE_1_INDEX);
        if (*enableRelu1Ptr || scale1Desc != nullptr) {
            OP_LOGE(context_->GetNodeName(),
                    "%s AscendC: When dualoutput is false, the second related input and attr is not allowed.",
                    context_->GetNodeType());
            return ge::GRAPH_FAILED;
        }
    }

    // check and get relumode/slipmode
    if (this->CheckExtendConv2dReluWeightAndClipValue(0, fixpipeInfo_.reluMode0)
        != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (attrInfo_.dualOutput) {
        if (this->CheckExtendConv2dReluWeightAndClipValue(1, fixpipeInfo_.reluMode1)
            != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    OP_LOGD(context_->GetNodeName(),
            "%s AscendC: reluMode0 is %u, reluMode1 is %u.",
            context_->GetNodeType(), fixpipeInfo_.reluMode0, fixpipeInfo_.reluMode1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendDualOutputLegal()
{
    if (!flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto dualOutputPtr = context_->GetAttrs()->GetBool(EXTENDCONV_ATTR_DUAL_OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dualOutputPtr);
    attrInfo_.dualOutput = static_cast<uint8_t>(*dualOutputPtr);
    fixpipeInfo_.dualOutput = attrInfo_.dualOutput;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckExtendDtypeLegal()
{
    if (!flagInfo_.extendConvFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto extendDtype0Ptr = context_->GetAttrs()->GetInt(EXTENDCONV_ATTR_DTYPE_0_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, extendDtype0Ptr);
    int64_t extendDtype0 = *extendDtype0Ptr;
    auto iter = std::find(EXTENDCONV2D_SUPPORTED_ATTR_DTYPE.begin(), EXTENDCONV2D_SUPPORTED_ATTR_DTYPE.end(), extendDtype0);
    if (iter == EXTENDCONV2D_SUPPORTED_ATTR_DTYPE.end()) {
         OP_LOGE(context_->GetNodeName(), 
            "%s AscendC: unSupported dtype0: %ld, only support [default, float, float16, int8, bfloat16, hifloat8, float8_e4m3fn].",
            paramInfo_.nodeType.c_str(), extendDtype0);
         return ge::GRAPH_FAILED;
    }
    auto extendDtype1Ptr = context_->GetAttrs()->GetInt(EXTENDCONV_ATTR_DTYPE_1_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, extendDtype1Ptr);
    int64_t extendDtype1 = *extendDtype1Ptr;
    iter = std::find(EXTENDCONV2D_SUPPORTED_ATTR_DTYPE.begin(), EXTENDCONV2D_SUPPORTED_ATTR_DTYPE.end(), extendDtype1);
    if (iter == EXTENDCONV2D_SUPPORTED_ATTR_DTYPE.end()) {
         OP_LOGE(context_->GetNodeName(), 
            "%s AscendC: unSupported dtype1: %ld, only support [default, float, float16, int8, bfloat16, hifloat8, float8_e4m3fn].",
            paramInfo_.nodeType.c_str(), extendDtype1);
         return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckAttrsLeagal()
{
    if (CheckExtendDualOutputLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantDtypeLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDilationLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckPadLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckGroupsLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDataFormatLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOffsetXLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRoundModeLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckEnableHf32Legal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExtendReluLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExtendDtypeLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}
}