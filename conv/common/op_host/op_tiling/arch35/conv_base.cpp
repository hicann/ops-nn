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
 * \file conv_base.cpp
 * \brief
 */
#include "conv_base.h"
#include <cmath>
#include <set>
#include "conv_api_tiling_base.h"
#include "conv_base_utils.h"

using namespace platform_ascendc;
using namespace conv_tiling;

namespace optiling {
namespace conv_ops_tiling {

ge::graphStatus ShapeAttrSynthesisCheck(ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo, gert::TilingContext* context)
{
    int64_t cmpHo = ConvComputeHo(oriShapeAttrInfo.oriFmapH, oriShapeAttrInfo.oriWeightH, oriShapeAttrInfo.oriPadTop,
        oriShapeAttrInfo.oriPadBottom, oriShapeAttrInfo.oriDilationH, oriShapeAttrInfo.oriStrideH);
    int64_t cmpWo = ConvComputeWo(oriShapeAttrInfo.oriFmapW, oriShapeAttrInfo.oriWeightW, oriShapeAttrInfo.oriPadLeft,
        oriShapeAttrInfo.oriPadRight, oriShapeAttrInfo.oriDilationW, oriShapeAttrInfo.oriStrideW);
    if (oriShapeAttrInfo.oriFmapN != oriShapeAttrInfo.oriOutputN) {
        OP_LOGE(context->GetNodeName(), "%s AscendC: ShapeAttrSynthesisCheck failed, " \
            "oriFmapN(%ld) != oriOutputN(%ld)", context->GetNodeType(), oriShapeAttrInfo.oriFmapN,
            oriShapeAttrInfo.oriOutputN);
        return ge::GRAPH_FAILED;
    }
    if (cmpHo != oriShapeAttrInfo.oriOutputH) {
        OP_LOGE(context->GetNodeName(), "%s AscendC: ShapeAttrSynthesisCheck failed, " \
            "cmpHo(%ld) != oriOutputH(%ld)", context->GetNodeType(), cmpHo, oriShapeAttrInfo.oriOutputH);
        return ge::GRAPH_FAILED;
    }
    if (cmpWo != oriShapeAttrInfo.oriOutputW) {
        OP_LOGE(context->GetNodeName(), "%s AscendC: ShapeAttrSynthesisCheck failed, " \
            "cmpWo(%ld) != oriOutputW(%ld)", context->GetNodeType(), cmpWo, oriShapeAttrInfo.oriOutputW);
        return ge::GRAPH_FAILED;
    }
    return ShapeAttrSynthesisCheckAux(oriShapeAttrInfo, context);
}

ge::graphStatus ShapeAttrSynthesisCheckAux(ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo,
                                           gert::TilingContext* context)
{
    // 0 means INPUT_FMAP_INDEX
    auto fmStorageFormat = static_cast<ge::Format>(GetPrimaryFormat(context->GetInputDesc(0)->GetStorageFormat()));
    if (fmStorageFormat == ge::Format::FORMAT_NCDHW || fmStorageFormat == ge::Format::FORMAT_NDHWC) {
        int64_t cmpDo = ConvComputeDo(oriShapeAttrInfo.oriFmapD, oriShapeAttrInfo.oriWeightD,
        oriShapeAttrInfo.oriPadHead, oriShapeAttrInfo.oriPadTail,
        oriShapeAttrInfo.oriDilationD, oriShapeAttrInfo.oriStrideD);
        if (cmpDo != oriShapeAttrInfo.oriOutputD) {
            OP_LOGE(context->GetNodeName(),
                "%s AscendC: ShapeAttrSynthesisCheck failed, cmpDo(%ld) != oriOutputD(%ld)",
                context->GetNodeType(), cmpDo, oriShapeAttrInfo.oriOutputD);
            return ge::GRAPH_FAILED;
        }
    }
    if (oriShapeAttrInfo.oriFmapC % oriShapeAttrInfo.oriGroups != 0) {
        OP_LOGE(context->GetNodeName(), "%s AscendC: featureMap Cin(%ld) mod groups(%ld) != 0",
            context->GetNodeType(), oriShapeAttrInfo.oriFmapC, oriShapeAttrInfo.oriGroups);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo.oriWeightN % oriShapeAttrInfo.oriGroups != 0) {
        OP_LOGE(context->GetNodeName(), "%s AscendC: weight Cout (%ld) mod groups (%ld) != 0",
            context->GetNodeType(), oriShapeAttrInfo.oriWeightN, oriShapeAttrInfo.oriGroups);
        return ge::GRAPH_FAILED;
    }
    if (oriShapeAttrInfo.oriFmapC != oriShapeAttrInfo.oriWeightC * oriShapeAttrInfo.oriGroups) {
        OP_LOGE(context->GetNodeName(), "%s AscendC: featureMap Cin(%ld) != weight Cin(%ld) * groups(%ld).",
            context->GetNodeType(), oriShapeAttrInfo.oriFmapC, oriShapeAttrInfo.oriWeightC,
            oriShapeAttrInfo.oriGroups);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void GetSupportedDataTypes(bool hasBias, bool quantFlag, std::vector<std::vector<ConvDtype>>& supportTypes)
{
    if (hasBias) {
        if (quantFlag) {
            supportTypes = QUANTCONV_SUPPORTED_TYPES_WITH_BIAS;
        } else {
            supportTypes = CONV_SUPPORTED_TYPES_WITH_BIAS;
        }
    } else {
        if (quantFlag) {
            supportTypes = QUANTCONV_SUPPORTED_TYPES_WITHOUT_BIAS;
        } else {
            supportTypes = CONV_SUPPORTED_TYPES_WITHOUT_BIAS;
        }
    }
}

void GetSupportedDataTypes(const platform_ascendc::SocVersion& socVersion, bool quantFlag, 
    ge::Format fMapFormat, bool exendConvFlag, std::vector<std::vector<ConvDtype>>& supportTypes)
{
    if (exendConvFlag) {
        if (fMapFormat == ge::Format::FORMAT_NCHW &&
            SOC_EXTENDCONV_SUPPORTED_TYPES_NCHW.find(socVersion) != SOC_EXTENDCONV_SUPPORTED_TYPES_NCHW.end()) {
            supportTypes = SOC_EXTENDCONV_SUPPORTED_TYPES_NCHW.at(socVersion);
        } else if (fMapFormat == ge::Format::FORMAT_NHWC &&
            SOC_EXTENDCONV_SUPPORTED_TYPES_NHWC.find(socVersion) != SOC_EXTENDCONV_SUPPORTED_TYPES_NHWC.end()) {
            supportTypes = SOC_EXTENDCONV_SUPPORTED_TYPES_NHWC.at(socVersion);
        }
    } else if (quantFlag) {
        supportTypes = QUANTCONV_SUPPORTED_TYPES;
    } else {
        if (SOC_CONV_SUPPORTED_TYPES.find(socVersion) != SOC_CONV_SUPPORTED_TYPES.end()) {
            supportTypes = SOC_CONV_SUPPORTED_TYPES.at(socVersion);
        }
    }
}

bool GetConvParamsIdx(const std::vector<ge::Format> formatVec, std::vector<std::vector<std::size_t>>& idxVec)
{
    ge::Format format;
    if (formatVec.size() != idxVec.size()) {
        return false;
    }
    for (size_t i = 0; i < static_cast<size_t>(formatVec.size()); ++i) {
        if (idxVec[i].size() != IDX_LIST_END_IDX) {
            return false;
        }
        format = formatVec[i];
        string fmtStr = ge::TypeUtils::FormatToSerialString(format);
        // all support format check in pre process
        idxVec[i][IDX_LIST_N_IDX] = fmtStr.find('N') != std::string::npos ? fmtStr.find('N') : IDX_LIST_END_IDX;
        idxVec[i][IDX_LIST_C_IDX] = fmtStr.find('C') != std::string::npos ? fmtStr.find('C') : IDX_LIST_END_IDX;
        idxVec[i][IDX_LIST_H_IDX] = fmtStr.find('H') != std::string::npos ? fmtStr.find('H') : IDX_LIST_END_IDX;
        idxVec[i][IDX_LIST_W_IDX] = fmtStr.find('W') != std::string::npos ? fmtStr.find('W') : IDX_LIST_END_IDX;
        idxVec[i][IDX_LIST_D_IDX] = fmtStr.find('D') != std::string::npos ? fmtStr.find('D') : IDX_LIST_END_IDX;
    }
    return true;
}

void ConvBase::SetBitsFromBool(uint64_t& number, const std::array<bool, UINT64_BIT_COUNT>& bits) const
{
    number = static_cast<uint64_t>(0);
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) {
            number |= (static_cast<uint64_t>(1) << i);
        }
    }
}
 
void ConvBase::SetBytesFromUint8(uint64_t& number, const std::array<uint8_t, UINT64_BYTE_COUNT>& bytes) const
{
    number = static_cast<uint64_t>(0);
    for (size_t i = 0; i < UINT64_BYTE_COUNT; ++i) {
        number |= static_cast<uint64_t>(bytes[i]) << (i * BITS_PER_BYTE);
    }
}
 
void ConvBase::SetBytesFromUint32(uint64_t& number, uint32_t highPart, uint32_t lowPart) const
{
    number = (static_cast<uint64_t>(highPart) << UINT32_BIT_COUNT) | lowPart;
}

void ConvBase::ConvBaseInit(ConvAscendcShapesInfo shapeInfo, ConvAscendcDescInfo descInfo,
                            ConvAscendcTilingFlag flagInfo, gert::TilingContext* context)
{
    shapeInfo_ = shapeInfo;
    descInfo_ = descInfo;
    flagInfo_ = flagInfo;
    context_ = context;
}

void ConvBase::UpdateFlagInfo(const ConvAscendcTilingFlag& flagInfo)
{
    flagInfo_ = flagInfo;
}

void ConvBase::ConvBaseInitFixpipeInfo(const FixpipeInfo& fixpipeInfo)
{
    fixpipeInfo_ = fixpipeInfo;
}

void ConvBase::ConvBaseInitOpInfo(const ConvTilingParseInfo* opInfo)
{
    opInfo_ = opInfo;
}

void ConvBase::ConvBaseInitFeatureFlag(const ConvAscendcFeatureFlag featureFlagInfo)
{
    featureFlagInfo_ = featureFlagInfo;
}

void ConvBase::InitGroupInfo(ConvOriGroupInfo convOriGroupInfo,
                             ConvOptGroupInfo convOptGroupInfo)
{
    oriGroupInfo_ = convOriGroupInfo;
    optGroupInfo_ = convOptGroupInfo;
}

void ConvBase::updatePlatformInfoFromOpInfo()
{
    platformInfo_.aicoreNum = opInfo_->aicoreNum;
    platformInfo_.l1Size = opInfo_->l1Size;
    platformInfo_.l0aSize = opInfo_->l0aSize;
    platformInfo_.l0bSize = opInfo_->l0bSize;
    platformInfo_.l0cSize = opInfo_->l0cSize;
}

void ConvBase::SetParams(uint64_t l2Rate)
{
    l2Rate_ = l2Rate;
}

ge::graphStatus ConvBase::CheckC04L1SizeLimitsInHWSplitMode()
{
    // c04 require wi fulload L1 if hi > 1
    uint64_t tempWi = shapeInfo_.hi > 1 ? shapeInfo_.wi : ConvInferWiL1(convOpsConstParams_.m0,
        shapeInfo_.wi, shapeInfo_.kw, attrInfo_.dilationW, attrInfo_.strideW);
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(C04_CIN_SIZE, ConvAlignB(
        C04_CIN_SIZE * shapeInfo_.kh * shapeInfo_.kw, convOpsConstParams_.k0), tempWi);
    if (usdL1SizeUnderMinHWtiling > opInfo_->l1Size) {
        OP_LOGD(context_->GetNodeName(),
            "%s AscendC: c04 minUsedL1SizeInHWmode: %lu > maxL1Size: %lu, c04 mode cannot be enabled.",
            context_->GetNodeType(), usdL1SizeUnderMinHWtiling, opInfo_->l1Size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus ConvBase::CheckC04L1SizeLimitsInMsplitMode()
{
    uint64_t c04UsdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(C04_CIN_SIZE,
        ConvAlignB(C04_CIN_SIZE * shapeInfo_.kh * shapeInfo_.kw, convOpsConstParams_.k0));
    if (c04UsdL1SizeUnderMinMtiling > opInfo_->l1Size) {
        OP_LOGD(context_->GetNodeName(),
            "%s AscendC: c04 minUsedL1SizeInMmode: %lu > maxL1Size: %lu, c04 mode cannot be enabled.",
            context_->GetNodeType(), c04UsdL1SizeUnderMinMtiling, opInfo_->l1Size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool ConvBase::IsFp32InputFp32Output()
{
    bool ret = descInfo_.fMapDtype == ge::DataType::DT_FLOAT &&
        descInfo_.weightDtype == ge::DataType::DT_FLOAT &&
        descInfo_.outDtype == ge::DataType::DT_FLOAT;
 
    if (flagInfo_.hasBias) {
        ret &= descInfo_.biasDtype == ge::DataType::DT_FLOAT;
    }
    return ret;
}

bool ConvBase::GetConvParasHf32Mode(const uint32_t enableHf32Idx, uint32_t& hf32Mode)
{
    if (!IsFp32InputFp32Output()) {
        return true; // hf32Mode is default 0 (means not enable).
    }
    auto enableHf32Ptr = context_->GetAttrs()->GetBool(enableHf32Idx);
    if (enableHf32Ptr == nullptr) {
        hf32Mode = 0;
        return true;
    }
    hf32Mode = static_cast<uint32_t>(*enableHf32Ptr ? 1 : 0);
    return true;
}

void ConvBase::GetSupportedFormats(bool quantFlag, bool is2dFlag, const platform_ascendc::SocVersion socVersion,
    std::stringstream& ss, std::vector<std::vector<ge::Format>>& supportFormats)
{
    const std::string& nodeType = context_->GetNodeType();
    bool extendConvFlag = nodeType == "ExtendConv2D";
    ss <<"The support params format list [fmap, weight, output] for scene ";
    if (socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        if (extendConvFlag) {
            supportFormats = EXTENDCONV2D_SUPPORT_FORMAT_LIST;
            ss << "(extendConvFlag is enable) is ";
        } else if (quantFlag) {
            supportFormats = is2dFlag ? SUPPORT_QUANT_CONV2D_FORMAT_LIST : SUPPORT_QUANT_CONV3D_FORMAT_LIST;
            ss << "(quantFlag is enable) is ";
        } else if (!quantFlag &&
            (descInfo_.fMapDtype != ge::DataType::DT_HIFLOAT8 || descInfo_.weightDtype != ge::DataType::DT_HIFLOAT8)) {
            supportFormats = is2dFlag ? SUPPORT_CONV2D_FORMAT_LIST : SUPPORT_CONV3D_FORMAT_LIST;
            ss << "(quantFlag is disable and inputDtype is not HIFLOAT8) is ";
        } else {
            supportFormats = is2dFlag ? SUPPORT_CONV2D_DEFAULT_FORMAT_LIST : SUPPORT_CONV3D_DEFAULT_FORMAT_LIST;
            ss << "(quantFlag is disable and inputDtype is HIFLOAT8) is ";
        }
    } else if (socVersion == platform_ascendc::SocVersion::MC62CM12A) {
        if (extendConvFlag) {
            supportFormats = EXTENDCONV2D_SUPPORT_FORMAT_LIST_MC62CM12A;
            ss << "(extendConvFlag is enable) is ";
        } else {
            supportFormats = SUPPORT_CONV2D_FORMAT_LIST_MC62CM12A;
            ss << "(extendConvFlag is disable) is ";
        }
    } else {
        supportFormats = is2dFlag ? SUPPORT_CONV2D_DEFAULT_FORMAT_LIST : SUPPORT_CONV3D_DEFAULT_FORMAT_LIST;
        ss << "(default soc version) is ";
    }

    for (size_t row = 0; row < supportFormats.size(); ++row) {
        ss << "[";
        for (size_t elem = 0; elem < supportFormats[row].size(); ++elem) {
            ss << formatToStrTab.at(supportFormats[row][elem]).c_str();
             if (elem != supportFormats[row].size() - 1) {
                ss << ", ";
             }
        }
        ss << "]";
        if (row != supportFormats.size() - 1) {
            ss << ", ";
        }
    }
}

bool ConvBase::CheckValidString(const string &inputStr, const gert::TilingContext* context) const
{
    if (inputStr.empty()) {
        return true;
    }

    if (inputStr.size() > MAX_STR_LEN) {
        OP_LOGE(context->GetNodeName(),
            "%s AscendC: check input string length: %zu failed, string length exceed %zu.",
            context->GetNodeType(), inputStr.size(), MAX_STR_LEN);
        return false;
    }
    if (!std::all_of(inputStr.begin(), inputStr.end(),
                     [](char c) { return std::isalnum(c) || c == '_'; })) {
        OP_LOGE(context->GetNodeName(),
            "%s AscendC: check input string: %s failed, only support 0-9, a-z, A-Z and '_'.",
            context->GetNodeType(), inputStr.c_str());
        return false;
    }

    return true;
}
}
}