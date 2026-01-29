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
 * \file conv3d_v2_base_tiling_check_attrs.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
 
namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv3dBaseTilingV2::ParseQuantDtypeLegal()
{
    if (!flagInfo_.quantFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto quantDtypePtr = context_->GetAttrs()->GetInt(ATTR_QUANT_DTYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, quantDtypePtr);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseStrideLegal()
{
    auto attrStrideIndex = flagInfo_.quantFlag ? ATTR_QUANT_STRIDE_INDEX : ATTR_STRIDE_INDEX;
    auto stridePtr = context_->GetAttrs()->GetListInt(attrStrideIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
    if (stridePtr->GetSize() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input attr stride dim: %zu != %u.", paramInfo_.nodeType.c_str(),
                stridePtr->GetSize(), CONV3D_DIM_SIZE_LIMIT);
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriStrideN = stridePtr->GetData()[conv3dOriginFormatAixsPosInfo_.nIndex];
    oriShapeAttrInfo_.oriStrideC = stridePtr->GetData()[conv3dOriginFormatAixsPosInfo_.cIndex];
    oriShapeAttrInfo_.oriStrideD = stridePtr->GetData()[conv3dOriginFormatAixsPosInfo_.dIndex];
    oriShapeAttrInfo_.oriStrideH = stridePtr->GetData()[conv3dOriginFormatAixsPosInfo_.hIndex];
    oriShapeAttrInfo_.oriStrideW = stridePtr->GetData()[conv3dOriginFormatAixsPosInfo_.wIndex];
    if (oriShapeAttrInfo_.oriStrideD < 1 || oriShapeAttrInfo_.oriStrideD > MAX_STRIDE_D_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Stride_D (%ld) is out of range[1, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriStrideD, MAX_STRIDE_D_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriStrideH < 1 || oriShapeAttrInfo_.oriStrideH > MAX_STRIDE_H_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Stride_H (%ld) is out of range[1, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriStrideH, MAX_STRIDE_H_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriStrideW < 1 || oriShapeAttrInfo_.oriStrideW > MAX_STRIDE_W_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Stride_W (%ld) is out of range[1, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriStrideW, MAX_STRIDE_W_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriStrideN != 1 || oriShapeAttrInfo_.oriStrideC != 1) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: stride (N: %ld, C: %ld) should be 1.",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriStrideN, oriShapeAttrInfo_.oriStrideC);
        return ge::GRAPH_FAILED;
    }
    attrInfo_.strideD = static_cast<uint64_t>(oriShapeAttrInfo_.oriStrideD);
    attrInfo_.strideH = static_cast<uint64_t>(oriShapeAttrInfo_.oriStrideH);
    attrInfo_.strideW = static_cast<uint64_t>(oriShapeAttrInfo_.oriStrideW);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseDilationLegal()
{
    auto attrDilationIndex = flagInfo_.quantFlag ? ATTR_QUANT_DILATION_INDEX : ATTR_DILATION_INDEX;
    auto dilationPtr = context_->GetAttrs()->GetListInt(attrDilationIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dilationPtr);
    if (dilationPtr->GetSize() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input attr dilation dim: %zu != %u.",
                paramInfo_.nodeType.c_str(), dilationPtr->GetSize(), CONV3D_DIM_SIZE_LIMIT);
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriDilationN = dilationPtr->GetData()[conv3dOriginFormatAixsPosInfo_.nIndex];
    oriShapeAttrInfo_.oriDilationC = dilationPtr->GetData()[conv3dOriginFormatAixsPosInfo_.cIndex];
    oriShapeAttrInfo_.oriDilationD = dilationPtr->GetData()[conv3dOriginFormatAixsPosInfo_.dIndex];
    oriShapeAttrInfo_.oriDilationH = dilationPtr->GetData()[conv3dOriginFormatAixsPosInfo_.hIndex];
    oriShapeAttrInfo_.oriDilationW = dilationPtr->GetData()[conv3dOriginFormatAixsPosInfo_.wIndex];
    if (oriShapeAttrInfo_.oriDilationD < 1 || oriShapeAttrInfo_.oriDilationD > MAX_DILATION_D_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Dilation_D (%ld) is out of range[1, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriDilationD, MAX_DILATION_D_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriDilationH < 1 || oriShapeAttrInfo_.oriDilationH > MAX_DILATION_H_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Dilation_H (%ld) is out of range[1, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriDilationH, MAX_DILATION_H_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriDilationW < 1 || oriShapeAttrInfo_.oriDilationW > MAX_DILATION_W_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Dilation_W (%ld) is out of range[1, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriDilationW, MAX_DILATION_W_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriDilationN != 1 || oriShapeAttrInfo_.oriDilationC != 1) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: dilation (N: %ld, C: %ld) should be 1.",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriDilationN, oriShapeAttrInfo_.oriDilationC);
        return ge::GRAPH_FAILED;
    }
    attrInfo_.dilationD = static_cast<uint64_t>(oriShapeAttrInfo_.oriDilationD);
    attrInfo_.dilationH = static_cast<uint64_t>(oriShapeAttrInfo_.oriDilationH);
    attrInfo_.dilationW = static_cast<uint64_t>(oriShapeAttrInfo_.oriDilationW);
    return ge::GRAPH_SUCCESS;
}

// optiling recaculate pad for kernel directed call process
ge::graphStatus Conv3dBaseTilingV2::GetOriPadFromAttrPad()
{
    auto attrPadIndex = flagInfo_.quantFlag ? ATTR_QUANT_PAD_INDEX : ATTR_PAD_INDEX;
    auto padPtr = context_->GetAttrs()->GetListInt(attrPadIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padPtr);
    if (padPtr->GetSize() != FORMAT_PAD_DIM) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: input attr pad dim: %zu != %u.", paramInfo_.nodeType.c_str(), 
                padPtr->GetSize(), FORMAT_PAD_DIM);
        return ge::GRAPH_FAILED;
    }
    oriShapeAttrInfo_.oriPadHead = padPtr->GetData()[PAD_HEAD_INDEX];
    oriShapeAttrInfo_.oriPadTail = padPtr->GetData()[PAD_TAIL_INDEX];
    oriShapeAttrInfo_.oriPadTop = padPtr->GetData()[PAD_TOP_INDEX];
    oriShapeAttrInfo_.oriPadBottom = padPtr->GetData()[PAD_BOTTOM_INDEX];
    oriShapeAttrInfo_.oriPadLeft = padPtr->GetData()[PAD_LEFT_INDEX];
    oriShapeAttrInfo_.oriPadRight = padPtr->GetData()[PAD_RIGHT_INDEX];
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus Conv3dBaseTilingV2::CheckOriPadLegal()
{
    if (oriShapeAttrInfo_.oriPadHead < 0 || oriShapeAttrInfo_.oriPadHead > MAX_PAD_D_SHAPE ||
        oriShapeAttrInfo_.oriPadTail < 0 || oriShapeAttrInfo_.oriPadTail > MAX_PAD_D_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Pad_Head (%ld) or Pad_Tail (%ld) is out of range[0, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriPadHead, oriShapeAttrInfo_.oriPadTail, MAX_PAD_D_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriPadTop < 0 || oriShapeAttrInfo_.oriPadTop > MAX_PAD_H_SHAPE ||
        oriShapeAttrInfo_.oriPadBottom < 0 || oriShapeAttrInfo_.oriPadBottom > MAX_PAD_H_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Pad_Top (%ld) or Pad_Bottom (%ld) is out of range[0, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriPadTop, oriShapeAttrInfo_.oriPadBottom, MAX_PAD_H_SHAPE);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriPadLeft < 0 || oriShapeAttrInfo_.oriPadLeft > MAX_PAD_W_SHAPE ||
        oriShapeAttrInfo_.oriPadRight < 0 || oriShapeAttrInfo_.oriPadRight > MAX_PAD_W_SHAPE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Pad_Left (%ld) or Pad_Right (%ld) is out of range[0, %ld].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriPadLeft, oriShapeAttrInfo_.oriPadRight, MAX_PAD_W_SHAPE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus Conv3dBaseTilingV2::GetOriPadFromPadMode()
{
    auto attrPadModeIndex = flagInfo_.quantFlag ? ATTR_QUANT_PADMODE_INDEX : ATTR_PAD_MODE_INDEX;
    auto padModePtr = context_->GetAttrs()->GetStr(attrPadModeIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padModePtr);
    string padMode(padModePtr);
    auto iter = find(PADMODE_WHITELIST.begin(), PADMODE_WHITELIST.end(), padMode);
    if (iter == PADMODE_WHITELIST.end()) {
        std::stringstream ss;
        ss << paramInfo_.nodeType.c_str() <<" AscendC: Only support pad_mode in [VALID, SPECIFIC, SAME, SAME_UPPER, "
            << "SAME_LOWER], actually is: " << padMode.c_str();
        OP_LOGE(context_->GetNodeName(), "%s", ss.str().c_str());
        return ge::GRAPH_FAILED;
    }

    if (padMode == "SPECIFIC") {
        return ge::GRAPH_SUCCESS;
    }

    if (padMode == "VALID") {
        oriShapeAttrInfo_.oriPadHead = 0;
        oriShapeAttrInfo_.oriPadTail = 0;
        oriShapeAttrInfo_.oriPadTop = 0;
        oriShapeAttrInfo_.oriPadBottom = 0;
        oriShapeAttrInfo_.oriPadLeft = 0;
        oriShapeAttrInfo_.oriPadRight = 0;
        return ge::GRAPH_SUCCESS;
    } else {
        return ApplySamesPad(padMode); 
    }
}

ge::graphStatus Conv3dBaseTilingV2::ApplySamesPad(const string& padMode)
{
    int64_t padD = (ConvCeilDiv(shapeInfo_.di, attrInfo_.strideD) - 1) * attrInfo_.strideD +
                attrInfo_.dilationD * (shapeInfo_.kd - 1) - shapeInfo_.di + 1;
    int64_t padH = (ConvCeilDiv(shapeInfo_.hi, attrInfo_.strideH) - 1) * attrInfo_.strideH +
                attrInfo_.dilationH * (shapeInfo_.kh - 1) - shapeInfo_.hi + 1;
    int64_t padW = (ConvCeilDiv(shapeInfo_.wi, attrInfo_.strideW) - 1) * attrInfo_.strideW +
                attrInfo_.dilationW * (shapeInfo_.kw - 1) - shapeInfo_.wi + 1;
    if (padMode == "SAME" || padMode == "SAME_UPPER") {
        if (padMode == "SAME") {
            padD = padD < 0 ? 0 : padD;
            padH = padH < 0 ? 0 : padH;
            padW = padW < 0 ? 0 : padW;                
        }
        oriShapeAttrInfo_.oriPadTail = ConvCeilDiv(padD, PAD_MODE_DIV_FACTOR);
        oriShapeAttrInfo_.oriPadHead = padD - oriShapeAttrInfo_.oriPadTail;
        oriShapeAttrInfo_.oriPadBottom = ConvCeilDiv(padH, PAD_MODE_DIV_FACTOR);
        oriShapeAttrInfo_.oriPadTop = padH - oriShapeAttrInfo_.oriPadBottom;
        oriShapeAttrInfo_.oriPadRight = ConvCeilDiv(padW, PAD_MODE_DIV_FACTOR);
        oriShapeAttrInfo_.oriPadLeft = padW - oriShapeAttrInfo_.oriPadRight;
    } else {
        // padMode is "SAME_LOWER"
        oriShapeAttrInfo_.oriPadHead = ConvCeilDiv(padD, PAD_MODE_DIV_FACTOR);
        oriShapeAttrInfo_.oriPadTail = padD - oriShapeAttrInfo_.oriPadHead;
        oriShapeAttrInfo_.oriPadTop = ConvCeilDiv(padH, PAD_MODE_DIV_FACTOR);
        oriShapeAttrInfo_.oriPadBottom = padH - oriShapeAttrInfo_.oriPadTop;
        oriShapeAttrInfo_.oriPadLeft = ConvCeilDiv(padW, PAD_MODE_DIV_FACTOR);
        oriShapeAttrInfo_.oriPadRight = padW - oriShapeAttrInfo_.oriPadLeft;                    
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParsePadLegal()
{
    OP_LOGE_IF(GetOriPadFromAttrPad() != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED,  context_->GetNodeName(),
        "%s AscendC: GetOriPadFromAttrPad Failed.", paramInfo_.nodeType.c_str());

    OP_LOGE_IF(GetOriPadFromPadMode() != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED,  context_->GetNodeName(),
        "%s AscendC: GetOriPadFromPadMode Failed.", paramInfo_.nodeType.c_str());

    if (CheckOriPadLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED; 
    }

    attrInfo_.padHead = static_cast<uint64_t>(oriShapeAttrInfo_.oriPadHead);
    attrInfo_.padTail = static_cast<uint64_t>(oriShapeAttrInfo_.oriPadTail);
    attrInfo_.padTop = static_cast<uint64_t>(oriShapeAttrInfo_.oriPadTop);
    attrInfo_.padBottom = static_cast<uint64_t>(oriShapeAttrInfo_.oriPadBottom);
    attrInfo_.padLeft = static_cast<uint64_t>(oriShapeAttrInfo_.oriPadLeft);
    attrInfo_.padRight = static_cast<uint64_t>(oriShapeAttrInfo_.oriPadRight);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseGroupLegal()
{
    auto attrGroupIndex = flagInfo_.quantFlag ? ATTR_QUANT_GROUP_INDEX : ATTR_GROUP_INDEX;
    auto groupsPtr = context_->GetAttrs()->GetInt(attrGroupIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, groupsPtr);
    oriShapeAttrInfo_.oriGroups = *groupsPtr;
    if (flagInfo_.isConv3dDequant && oriShapeAttrInfo_.oriGroups != 1) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: unsupport groups (%ld)! = 1.",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriGroups);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriGroups < 1 || static_cast<uint64_t>(oriShapeAttrInfo_.oriGroups) > MAX_GROUP_SHAPE) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: groups(%ld) from attr are out of range[1, %lu].",
            paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriGroups, MAX_GROUP_SHAPE);
        return ge::GRAPH_FAILED;
    }

    attrInfo_.groups = static_cast<uint32_t>(oriShapeAttrInfo_.oriGroups);
    flagInfo_.convGroupType = GetGroupsInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseQuantDataFormatLegal()
{
    if (!flagInfo_.quantFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto DataFormatPtr = context_->GetAttrs()->GetStr(ATTR_QUANT_DATAFORMAT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, DataFormatPtr);
    string dataFormat(DataFormatPtr);
    if (dataFormat != "NCDHW") {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: Only support dataFormat(NCDHW), actually is %s",
                paramInfo_.nodeType.c_str(), dataFormat.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseQuantOffsetXLegal()
{
    if (!flagInfo_.quantFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto offsetXPtr = context_->GetAttrs()->GetInt(ATTR_QUANT_OFFSETX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, offsetXPtr);
    oriShapeAttrInfo_.oriOffsetX = *offsetXPtr;
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    bool hif8Fp8Mode = ((fMapDesc->GetDataType() == ge::DataType::DT_HIFLOAT8) ||
        (fMapDesc->GetDataType() == ge::DataType::DT_FLOAT8_E4M3FN));
    if (hif8Fp8Mode && (oriShapeAttrInfo_.oriOffsetX != 0)) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Only support offsetX = 0 in hif8 or fp8 mode, actually is %ld",
                paramInfo_.nodeType.c_str(), oriShapeAttrInfo_.oriOffsetX);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo_.oriOffsetX > OFFSET_X_MAX_VALUE || oriShapeAttrInfo_.oriOffsetX < OFFSET_X_MIN_VALUE) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Only support offsetX in ([%d, %d]), actually is %ld",
                paramInfo_.nodeType.c_str(), OFFSET_X_MIN_VALUE, OFFSET_X_MAX_VALUE, oriShapeAttrInfo_.oriOffsetX);
        return ge::GRAPH_FAILED;
    }
    attrInfo_.offsetx = static_cast<int32_t>(oriShapeAttrInfo_.oriOffsetX);   
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::ParseQuantRoundModeLegal()
{
    if (!flagInfo_.quantFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto roundModePtr = context_->GetAttrs()->GetStr(ATTR_QUANT_ROUNDMODE_INDEX);
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
                "%s AscendC: the input round_mode has invalid str",
                paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (STR_TO_ROUNDMODE.find(roundMode) != STR_TO_ROUNDMODE.end()) {
        attrInfo_.roundMode = STR_TO_ROUNDMODE.at(roundMode);
    }
    return ge::GRAPH_SUCCESS;
}

}
}