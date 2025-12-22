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
uint32_t Gcd(uint32_t i, uint32_t j)
{
    return j > 0 ? Gcd(j, i % j) : i;
}

uint64_t ConvCeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

void ConvCalcCommFactor(const uint64_t num, const uint32_t numMax, std::vector<uint32_t> &reslist)
{
    uint32_t sqrtMax = static_cast<uint32_t>(sqrt(num));
    for (uint32_t i = 1; i <= sqrtMax; ++i) {
        if (num % i == 0) {
            if (i <= numMax) {
                reslist.emplace_back(i);
            }
            uint32_t right = num / i;
            if (right != i && right <= numMax) {
                reslist.emplace_back(right);
            }
        }
    }
    sort(reslist.begin(), reslist.end());
}

uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

uint64_t ConvInferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH)
{
    uint64_t khDilated = (singlekH - 1) * dilationH + 1;
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

uint64_t ConvInferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW)
{
    uint64_t kwDilated = (singlekW - 1) * dilationW + 1;
    uint64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

int64_t ConvComputeHo(int64_t hi, int64_t hk, int64_t padTop, int64_t padBottom, int64_t dilationH, int64_t strideH)
{
    if (strideH == 0) {
        return 1;
    }
    int64_t cmpHo = (hi + padTop + padBottom - dilationH * (hk - 1) - 1) / strideH + 1;
    return cmpHo;
}

int64_t ConvComputeWo(int64_t wi, int64_t wk, int64_t padLeft, int64_t padRight, int64_t dilationW, int64_t strideW)
{
    if (strideW == 0) {
        return 1;
    }
    int64_t cmpWo = (wi + padLeft + padRight - dilationW * (wk - 1) - 1) / strideW + 1;
    return cmpWo;
}

int64_t ConvComputeDo(int64_t di, int64_t dk, int64_t padHead, int64_t padTail, int64_t dilationD, int64_t strideD)
{
    if (strideD == 0) {
        return 1;
    }
    int64_t cmpDo = (di + padHead + padTail - dilationD * (dk - 1) - 1) / strideD + 1;
    return cmpDo;
}

void ConvBlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange, const std::vector<uint32_t> &mixRange)
{
    std::vector<uint32_t> tmpSelectMixRange;
    for (auto v : mixRange) {
        if (v <= orgDim) {
            tmpSelectMixRange.push_back(v);
        }
    }
    std::set<uint32_t>tmpRanges(inputRange.begin(), inputRange.end());
    tmpRanges.insert(tmpSelectMixRange.begin(), tmpSelectMixRange.end());
    inputRange.assign(tmpRanges.begin(), tmpRanges.end());
}

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

bool IsWeightNZFormat(ge::Format weightFormat)
{
    return weightFormat == ge::Format::FORMAT_FRACTAL_Z || weightFormat == ge::Format::FORMAT_FRACTAL_Z_C04;
}

void InitblockDimConstParas(ConvOpsConstParams& convOpsConstParams,
                            const ConvAscendcDescInfo& descInfo,
                            const ConvAscendcShapesInfo& shapeInfo)
{
    convOpsConstParams.m0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo.fMapDtype), MKN_M_IDX);
    convOpsConstParams.k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo.weightDtype), MKN_K_IDX);
    convOpsConstParams.n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo.fMapDtype), MKN_N_IDX);
    convOpsConstParams.ci1 = ConvCeilDiv(shapeInfo.ci, convOpsConstParams.k0);
    convOpsConstParams.co1 = ConvCeilDiv(shapeInfo.co, convOpsConstParams.n0);
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

void ConvBase::ConvBaseInitAttrInfo(ConvAscendcAttrInfo attrInfo)
{
    attrInfo_ = attrInfo;
}

void ConvBase::ConvBaseInitFixpipeInfo(const conv_tiling::FixpipeInfo& fixpipeInfo)
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

void ConvBase::GetConvBaseCoreInfo(ConvOpsConstParams& convOpsConstParams)
{
    convOpsConstParams = convOpsConstParams_;
}

void ConvBase::InitGroupInfo(conv_tiling::ConvOriGroupInfo convOriGroupInfo,
                             conv_tiling::ConvOptGroupInfo convOptGroupInfo)
{
    oriGroupInfo_ = convOriGroupInfo;
    optGroupInfo_ = convOptGroupInfo;
}

uint64_t ConvBase::CalcTotalCostMsplitMode(uint32_t batchDim, uint32_t mDim,
                                           uint32_t nDim, uint32_t doDim, uint32_t groupDim)
{
    uint64_t curCi = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.ciPerGroup : optGroupInfo_.cinOpt) : shapeInfo_.ci;
    uint64_t ci1 = ConvCeilDiv(curCi, k0_);

    uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    uint64_t co1 = ConvCeilDiv(curCo, n0_);

    uint64_t curGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.groupOpt : attrInfo_.groups;

    uint64_t loadFeatureMapCost = ConvCeilDiv(shapeInfo_.batch, batchDim) * ConvCeilDiv(curGroups, groupDim) *
                                  ConvCeilDiv(shapeInfo_.dout, doDim) *
                                  ConvCeilDiv(ConvAlignB(shapeInfo_.hi * shapeInfo_.wi, m0_), mDim) *
                                  shapeInfo_.kd * ci1 * k0_;

    uint64_t loadWeightCost = ConvCeilDiv(curGroups, groupDim) *
                              shapeInfo_.kd * ci1 * shapeInfo_.kh * shapeInfo_.kw * k0_ *
                              ConvCeilDiv(shapeInfo_.batch, batchDim);
    // N dim full load to UB in opt group mode.
    if (flagInfo_.convGroupType != ConvGroupType::OPT_GROUP_CONV) {
        loadWeightCost *= ConvCeilDiv(co1 * n0_, nDim);
    }

    uint64_t loadOutputCost = ConvCeilDiv(shapeInfo_.batch, batchDim) * ConvCeilDiv(curGroups, groupDim) *
                              ConvCeilDiv(co1 * n0_, nDim) *
                              ConvCeilDiv(shapeInfo_.dout, doDim) *
                              ConvCeilDiv(ConvAlignB(shapeInfo_.ho * shapeInfo_.wo, m0_), mDim);

    uint64_t cubeCalcCost = ConvCeilDiv(shapeInfo_.batch, batchDim) * ConvCeilDiv(curGroups, groupDim) *
                            ConvCeilDiv(co1, nDim) * ConvCeilDiv(shapeInfo_.dout, doDim) *
                            shapeInfo_.kd * ci1 * shapeInfo_.kh * shapeInfo_.kw *
                            ConvCeilDiv(ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, m0_), mDim);

    uint32_t curBwCoeff = GetWeightBandWidthCoeff();

    return (loadFeatureMapCost + (loadWeightCost * curBwCoeff) + loadOutputCost) / MIN_L2_BAND_WIDTH + cubeCalcCost;
}

void ConvBase::SetParams(uint64_t l2Rate)
{
    l2Rate_ = l2Rate;
}

void ConvBase::SetAiCoreNum(uint32_t aicoreNum)
{
    aicoreNum_ = aicoreNum;
}

void ConvBase::SetMKN(uint32_t m0, uint32_t k0, uint32_t n0)
{
    m0_ = m0;
    k0_ = k0;
    n0_ = n0;
}

void ConvBase::InitblockDimConstParas()
{
    convOpsConstParams_.m0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_M_IDX);
    convOpsConstParams_.k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_K_IDX);
    convOpsConstParams_.n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX);
    convOpsConstParams_.ci1 = ConvCeilDiv(shapeInfo_.ci, convOpsConstParams_.k0);
    convOpsConstParams_.co1 = ConvCeilDiv(shapeInfo_.co, convOpsConstParams_.n0);
	SetAiCoreNum(opInfo_->aicoreNum);
    SetMKN(CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_M_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_K_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX));
}
 
uint64_t ConvBase::CalcMinUsedL1SizeInMsplitMode(uint64_t kAL1min, uint64_t kBL1min)
{
    uint64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
    uint64_t biasDtypeSize = dtypeSizeTab.at(descInfo_.biasDtype);
    uint64_t weightDtypeSize = dtypeSizeTab.at(descInfo_.weightDtype);

    // ensure 32-byte alignment
    uint64_t nBL1min = convOpsConstParams_.n0;
    uint64_t biasUsedL1Size = flagInfo_.hasBias ? ConvAlignB(nBL1min * biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = ConvAlignB(nBL1min * fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE, C0_SIZE);
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);
    uint64_t hoAL1min = std::min(convOpsConstParams_.m0 / shapeInfo_.wo + CONST_VALUE_2, shapeInfo_.ho);
    uint64_t hiAL1min = ConvInferHiL1(hoAL1min, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
    uint64_t fmapUsedL1Size =
        ConvAlignB(hiAL1min * shapeInfo_.wi * kAL1min * fMapDtypeSize, C0_SIZE);

    uint64_t minL1LoadSize = biasUsedL1Size + fmapUsedL1Size + weightUsedL1Size + scaleUsedL1Size;
    return minL1LoadSize;
}

uint64_t ConvBase::CalcMinUsedL1SizeInHWsplitMode(uint64_t kAL1min, uint64_t kBL1min, uint64_t wiAL1min)
{
    uint64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
    uint64_t biasDtypeSize = dtypeSizeTab.at(descInfo_.biasDtype);
    uint64_t weightDtypeSize = dtypeSizeTab.at(descInfo_.weightDtype);
    uint64_t nBL1min = convOpsConstParams_.n0;
    uint64_t biasUsedL1Size = flagInfo_.hasBias ? ConvAlignB(nBL1min * biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = ConvAlignB(nBL1min * fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE, C0_SIZE);
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);

    uint64_t fmapUsedL1Size = 0;
    uint64_t hoAL1min = shapeInfo_.wo < convOpsConstParams_.m0 ? ConvCeilDiv(convOpsConstParams_.m0, shapeInfo_.wo) : 1;
    uint64_t hiAL1min = ConvInferHiL1(hoAL1min, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
    fmapUsedL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * fMapDtypeSize, C0_SIZE);

    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;
    return minL1LoadSize;
}

ge::graphStatus ConvBase::CheckL1SizeLimitsInHWsplitMode()
{
    // require hiL1 * wiL1 >= m0
    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t woAL1min = convOpsConstParams_.m0;
    uint64_t wiAL1min = ConvInferWiL1(woAL1min, shapeInfo_.wi, shapeInfo_.kw, attrInfo_.dilationW, attrInfo_.strideW);
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(convOpsConstParams_.k0,
        shapeInfo_.kh * shapeInfo_.kw * convOpsConstParams_.k0, wiAL1min);
    if (usdL1SizeUnderMinHWtiling > opInfo_->l1Size) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: MinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                context_->GetNodeType(), usdL1SizeUnderMinHWtiling, opInfo_->l1Size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus ConvBase::CheckL1SizeLimitsInMSplitMode()
{
    uint64_t usdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(convOpsConstParams_.k0,
        shapeInfo_.kh * shapeInfo_.kw * convOpsConstParams_.k0);
    if (usdL1SizeUnderMinMtiling > opInfo_->l1Size) {
        OP_LOGD(context_->GetNodeName(),
            "%s AscendC: MinL1LoadSizeInMmode > L1size, current L1size: %lu, maxL1Size: %lu",
            context_->GetNodeType(), usdL1SizeUnderMinMtiling, opInfo_->l1Size);
        return ge::GRAPH_FAILED;
    }
    // load3dv2 win max value
    if (shapeInfo_.wi > LOAD3DV2_WIN_LIMIT_VALUE) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
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

BlockDimRes ConvBase::BlockDimDecisionMsplitMode()
{
    GetBlockDimRangeMsplitMode();
    GetBlockDimInitMsplitMode();
    CoreBlockDimDecisionMsplitMode();
    return blockDimRes_;
}

void ConvBase::GetBlockDimRangeMsplitMode()
{
    GetBlockDimRangeCommon();
    uint64_t m1 = ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, m0_); // mRange
    ConvCalcCommFactor(m1, aicoreNum_, blockDimRanges_.mRange);
    ConvBlockDimFactorMix(m1, blockDimRanges_.mRange, blockDimRanges_.aicNumRange);
}

void ConvBase::GetBlockDimInitMsplitMode()
{
    blockDimInit_.resize(BLOCKDIM_MSPLIT_DEC_NUM, 1);
    blockDimInit_[BLOCKDIM_MSPLIT_BATCH_IDX] = blockDimRanges_.batchRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_M_IDX] = blockDimRanges_.mRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_N_IDX] = blockDimRanges_.nRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_DO_IDX] = blockDimRanges_.doRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_GROUP_IDX] = blockDimRanges_.groupRange[0];
}

void ConvBase::CoreBlockDimDecisionMsplitMode()
{
    blockDimRes_.batchDim = 1;
    blockDimRes_.mDim = 1;
    blockDimRes_.nDim = 1;
    blockDimRes_.doDim = 1;
    blockDimRes_.groupDim = 1;
    blockDimRes_.minCost = CalcTotalCostMsplitMode(blockDimRes_.batchDim,
                                        blockDimRes_.mDim, blockDimRes_.nDim,
                                        blockDimRes_.doDim, blockDimRes_.groupDim);

    vector<vector<uint32_t>> allRanges(BLOCKDIM_MSPLIT_DEC_NUM, vector<uint32_t>(1, 1));
    allRanges[BLOCKDIM_MSPLIT_BATCH_IDX] = blockDimRanges_.batchRange;
    allRanges[BLOCKDIM_MSPLIT_M_IDX] = blockDimRanges_.mRange;
    allRanges[BLOCKDIM_MSPLIT_N_IDX] = blockDimRanges_.nRange;
    allRanges[BLOCKDIM_MSPLIT_DO_IDX] = blockDimRanges_.doRange;
    allRanges[BLOCKDIM_MSPLIT_GROUP_IDX] = blockDimRanges_.groupRange;
    vector<uint32_t> dimsRecord;
    BlockDimDecisionBackTrackMsplitMode(allRanges, BLOCKDIM_MSPLIT_BATCH_IDX, dimsRecord);
}


void ConvBase::BlockDimDecisionBackTrackMsplitMode(const vector<vector<uint32_t>> &inputRanges,
                                                 uint32_t rangeIdx, vector<uint32_t> &record)
{
    if (record.size() == inputRanges.size()) {
        uint32_t curBlockDim = record[BLOCKDIM_MSPLIT_BATCH_IDX] * record[BLOCKDIM_MSPLIT_M_IDX] *
                               record[BLOCKDIM_MSPLIT_N_IDX] * record[BLOCKDIM_MSPLIT_DO_IDX] *
                               record[BLOCKDIM_MSPLIT_GROUP_IDX];
        if (curBlockDim > aicoreNum_) {
            return;
        }
        bool  update_flag = false;
        uint64_t curCost = CalcTotalCostMsplitMode(record[BLOCKDIM_MSPLIT_BATCH_IDX],
                                         record[BLOCKDIM_MSPLIT_M_IDX], record[BLOCKDIM_MSPLIT_N_IDX],
                                         record[BLOCKDIM_MSPLIT_DO_IDX], record[BLOCKDIM_MSPLIT_GROUP_IDX]);
        if (curCost < blockDimRes_.minCost) {
            update_flag = true;
        } else if (curCost == blockDimRes_.minCost) {
            update_flag = CmpCoreUtilizeMsplitMode(record[BLOCKDIM_MSPLIT_BATCH_IDX], record[BLOCKDIM_MSPLIT_M_IDX],
                record[BLOCKDIM_MSPLIT_N_IDX], record[BLOCKDIM_MSPLIT_DO_IDX], record[BLOCKDIM_MSPLIT_GROUP_IDX]);
        }
        if (update_flag) {
            SetBlockDimMsplitMode(record, curCost);
        }
        return;
    }

    if (rangeIdx >= inputRanges.size() || rangeIdx >= blockDimInit_.size()) {
        return;
    }

    for (uint32_t i = 0; i < inputRanges[rangeIdx].size(); i++) {
        record.push_back(inputRanges[rangeIdx][i]);
        BlockDimDecisionBackTrackMsplitMode(inputRanges, rangeIdx + 1, record);
        record.pop_back();
    }
}

// hw split mode
BlockDimRes ConvBase::BlockDimDecisionHWsplitMode()
{
    GetBlockDimRangeHWsplitMode();
    GetBlockDimInitHWsplitMode();
    CoreBlockDimDecisionHWsplitMode();
    CheckCoreUsedupHWsplitMode();
	return blockDimRes_;
}

uint64_t ConvBase::CalcCostHWsplitMode(const BlockDimRes &blockDimRes,
    const uint64_t ci1, const uint64_t ci0, const uint64_t co1)
{
    const uint64_t curGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.groupOpt : attrInfo_.groups;

    const uint64_t loadFeatureMapCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * ConvCeilDiv(shapeInfo_.dout, blockDimRes.doDim) *
        shapeInfo_.kd * ci1 * ConvCeilDiv(shapeInfo_.hi, blockDimRes.hoDim) *
        ConvCeilDiv(shapeInfo_.wi, blockDimRes.woDim) * ci0;

    const uint64_t weightKSize = flagInfo_.enableC04Flag ?
        static_cast<uint64_t>(ConvAlignB(ci1 * shapeInfo_.kh * shapeInfo_.kw, static_cast<uint64_t>(k0_))) :
        ci1 * shapeInfo_.kh * shapeInfo_.kw * static_cast<uint64_t>(k0_);
   uint64_t loadWeightCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * shapeInfo_.kd * weightKSize;

    // N dim full load to UB in opt group mode.
    if (flagInfo_.convGroupType != ConvGroupType::OPT_GROUP_CONV) {
        loadWeightCost *= ConvCeilDiv(co1 * n0_, blockDimRes.nDim);
    }

    const uint64_t loadOutputCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * ConvCeilDiv(shapeInfo_.dout, blockDimRes.doDim) *
        ConvCeilDiv(co1 * n0_, blockDimRes.nDim) * ConvCeilDiv(shapeInfo_.ho, blockDimRes.hoDim) *
        ConvCeilDiv(shapeInfo_.wo, blockDimRes.woDim);

    const uint64_t cubeCalcCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * ConvCeilDiv(co1, blockDimRes.nDim) *
        ConvCeilDiv(shapeInfo_.dout, blockDimRes.doDim) * shapeInfo_.kd * ci1 * shapeInfo_.kh * shapeInfo_.kw *
        ConvCeilDiv(ConvCeilDiv(shapeInfo_.ho, blockDimRes.hoDim) * ConvCeilDiv(shapeInfo_.wo, blockDimRes.woDim),
        m0_);

    uint32_t curBwCoeff = GetWeightBandWidthCoeff();
    return (loadFeatureMapCost + (loadWeightCost * curBwCoeff) + loadOutputCost) / MIN_L2_BAND_WIDTH + cubeCalcCost;
}

uint64_t ConvBase::CalcTotalCostHWsplitMode(const BlockDimRes &blockDimRes)
{
    uint64_t ci0 = k0_;
    uint64_t curCi = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.ciPerGroup : optGroupInfo_.cinOpt) : shapeInfo_.ci;
    uint64_t ci1 = ConvCeilDiv(curCi, ci0);

    uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    uint64_t co1 = ConvCeilDiv(curCo, n0_);

    if (flagInfo_.enableC04Flag) {
        ci1 = C04_CI1_SIZE;
        ci0 = C04_CIN_SIZE;
    }

    return CalcCostHWsplitMode(blockDimRes, ci1, ci0, co1);
}

uint64_t ConvBase::GetMinBurstNum()
{
    return MIN_L2_BAND_WIDTH / dtypeSizeTab.at(descInfo_.fMapDtype);
}

uint32_t ConvBase::GetWeightBandWidthCoeff()
{
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NCDHW || descInfo_.fMapFormat == ge::Format::FORMAT_NCHW) {
        if (IsWeightNZFormat(descInfo_.weightFormat)) {
            return BW_COEFF_UB;
        }
        if (flagInfo_.convGroupType != ConvGroupType::OPT_GROUP_CONV) {
            return BW_COEFF;
        }
    }
    return BW_COEFF_UB;
}

void ConvBase::SeperateHoRangeHWsplitMode()
{
    std::vector<uint32_t> tmpHoRange = blockDimRanges_.hoRange;
    uint64_t minValue = GetMinBurstNum() / shapeInfo_.wo;
    uint32_t firstValidIdx = tmpHoRange.size();
    for (uint32_t i = 0; i < tmpHoRange.size(); i++) {
        if (ConvCeilDiv(shapeInfo_.ho, tmpHoRange[i]) >= minValue) {
            firstValidIdx = i;
            break;
        }
    }
    blockDimRanges_.hoRange.clear();
    blockDimRanges_.hoSpareRange.clear();
    blockDimRanges_.hoRange.assign(tmpHoRange.begin() + firstValidIdx, tmpHoRange.end());
    blockDimRanges_.hoSpareRange.assign(tmpHoRange.begin(), tmpHoRange.begin() + firstValidIdx);
}

void ConvBase::GetBlockDimRangeCommon()
{
    // aicoreRange
    ConvCalcCommFactor(aicoreNum_, aicoreNum_, blockDimRanges_.aicNumRange);
    // batchRange
    ConvCalcCommFactor(shapeInfo_.batch, aicoreNum_, blockDimRanges_.batchRange);

    if (shapeInfo_.batch >= BATCH_AICORE_COF * aicoreNum_) {
        blockDimRanges_.batchRange = blockDimRanges_.aicNumRange;
    } else {
        ConvBlockDimFactorMix(shapeInfo_.batch, blockDimRanges_.batchRange, blockDimRanges_.aicNumRange);
    }
    // nRange
    uint64_t curCo = shapeInfo_.co;
    if (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV) {
        curCo = oriGroupInfo_.coPerGroup;
    } else if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        curCo = optGroupInfo_.coutOpt;
    }
    ConvCalcCommFactor(ConvCeilDiv(curCo, n0_), aicoreNum_, blockDimRanges_.nRange);
    ConvBlockDimFactorMix(ConvCeilDiv(curCo, n0_), blockDimRanges_.nRange, blockDimRanges_.aicNumRange);

    // doRange
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NCDHW || descInfo_.fMapFormat == ge::Format::FORMAT_NDHWC) {
        ConvCalcCommFactor(shapeInfo_.dout, aicoreNum_, blockDimRanges_.doRange);
        ConvBlockDimFactorMix(shapeInfo_.dout, blockDimRanges_.doRange, blockDimRanges_.aicNumRange);
    } else {
        blockDimRanges_.doRange.assign(1, 1);
    }
    // groupRange
    uint64_t curGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.groupOpt : attrInfo_.groups;
    ConvCalcCommFactor(curGroups, aicoreNum_, blockDimRanges_.groupRange);
    ConvBlockDimFactorMix(curGroups, blockDimRanges_.groupRange, blockDimRanges_.aicNumRange);
}

void ConvBase::GetBlockDimRangeHWsplitMode()
{
    GetBlockDimRangeCommon();
    ConvCalcCommFactor(shapeInfo_.ho, aicoreNum_, blockDimRanges_.hoRange);
    ConvBlockDimFactorMix(shapeInfo_.ho, blockDimRanges_.hoRange, blockDimRanges_.aicNumRange);
    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_CONV1D_FLAG) {
        // 如果 L1 全载，优先使用在 batch 轴绑核, wo 最大的绑轴数量为 aicoreNum_ / max(batchRange),
        uint64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
        uint64_t kAL1max = ConvAlignB(shapeInfo_.ci, convOpsConstParams_.k0);
        uint64_t fmapUsedL1Size = ConvAlignB(shapeInfo_.wi * kAL1max * fMapDtypeSize, C0_SIZE);
        if (fmapUsedL1Size < ConvCeilDiv(opInfo_->l1Size, CONST_VALUE_2)) {
            uint32_t woRangeMaxConv1D = ConvCeilDiv(aicoreNum_, shapeInfo_.batch);
            ConvCalcCommFactor(shapeInfo_.wo, woRangeMaxConv1D, blockDimRanges_.woRange);
        } else {
            ConvCalcCommFactor(shapeInfo_.wo, aicoreNum_, blockDimRanges_.woRange);
            ConvBlockDimFactorMix(shapeInfo_.wo, blockDimRanges_.woRange, blockDimRanges_.aicNumRange);
        }
        return;
    }                                         
    blockDimRanges_.woRange.emplace_back(1);

    if (shapeInfo_.wo < GetMinBurstNum()) {
        SeperateHoRangeHWsplitMode();
    }
}

bool ConvBase::CmpCoreUtilize(const uint32_t curCoreUtilize, const uint32_t minCostCoreUtilize,
    const uint32_t batchDim, const uint32_t doDim)
{
    if (curCoreUtilize < minCostCoreUtilize) {
        return false;
    } else if (curCoreUtilize == minCostCoreUtilize) {
        // for same cost, preference: batch > dout
        bool updateFlag = std::make_tuple(batchDim, doDim) > std::make_tuple(blockDimRes_.batchDim, blockDimRes_.doDim);
        return updateFlag;
    }
    return true;
}

bool ConvBase::CmpCoreUtilizeMsplitMode(uint32_t batchDim, uint32_t mDim, uint32_t nDim, uint32_t doDim, uint32_t groupDim)
{
    const uint32_t curCoreUtilize = batchDim * mDim * nDim * doDim * groupDim / aicoreNum_;
    const uint32_t minCostCoreUtilize = blockDimRes_.batchDim * blockDimRes_.mDim * blockDimRes_.nDim *
                                  blockDimRes_.doDim * blockDimRes_.groupDim / aicoreNum_;
    return CmpCoreUtilize(curCoreUtilize, minCostCoreUtilize, batchDim, doDim);
}

bool ConvBase::CmpCoreUtilizeHWsplitMode(const vector<uint32_t> &record)
{
    auto batchDim = record[BLOCKDIM_HWSPLIT_BATCH_IDX];
    auto hoDim = record[BLOCKDIM_HWSPLIT_HO_IDX];
    auto woDim = record[BLOCKDIM_HWSPLIT_WO_IDX];
    auto nDim = record[BLOCKDIM_HWSPLIT_N_IDX];
    auto doDim = record[BLOCKDIM_HWSPLIT_DO_IDX];
    auto groupDim = record[BLOCKDIM_HWSPLIT_GROUP_IDX];
    const uint32_t curCoreUtilize = batchDim * hoDim * woDim * nDim * doDim * groupDim / aicoreNum_;
    const uint32_t minCostCoreUtilize = blockDimRes_.batchDim * blockDimRes_.hoDim * blockDimRes_.woDim *
                                        blockDimRes_.nDim * blockDimRes_.doDim * blockDimRes_.groupDim / aicoreNum_;
    return CmpCoreUtilize(curCoreUtilize, minCostCoreUtilize, batchDim, doDim);
}

void ConvBase::SetBlockDimHWsplitMode(const vector<uint32_t> &record, const uint64_t curCost,
    BlockDimRes &blockDimRes) const
{
    blockDimRes.batchDim = record[BLOCKDIM_HWSPLIT_BATCH_IDX];
    blockDimRes.hoDim = record[BLOCKDIM_HWSPLIT_HO_IDX];
    blockDimRes.woDim = record[BLOCKDIM_HWSPLIT_WO_IDX];
    blockDimRes.nDim = record[BLOCKDIM_HWSPLIT_N_IDX];
    blockDimRes.doDim = record[BLOCKDIM_HWSPLIT_DO_IDX];
    blockDimRes.groupDim = record[BLOCKDIM_HWSPLIT_GROUP_IDX];
    blockDimRes.minCost = curCost;
}

void ConvBase::SetBlockDimMsplitMode(const vector<uint32_t> &record, uint64_t curCost)
{
    blockDimRes_.batchDim = record[BLOCKDIM_MSPLIT_BATCH_IDX];
    blockDimRes_.mDim = record[BLOCKDIM_MSPLIT_M_IDX];
    blockDimRes_.nDim = record[BLOCKDIM_MSPLIT_N_IDX];
    blockDimRes_.doDim = record[BLOCKDIM_MSPLIT_DO_IDX];
    blockDimRes_.groupDim = record[BLOCKDIM_MSPLIT_GROUP_IDX];
    blockDimRes_.minCost = curCost;
}

void ConvBase::GetBlockDimInitHWsplitMode()
{
    blockDimInit_.resize(BLOCKDIM_HWSPLIT_DEC_NUM, 1);
    blockDimInit_[BLOCKDIM_HWSPLIT_BATCH_IDX] = blockDimRanges_.batchRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_HO_IDX] = blockDimRanges_.hoRange.empty() ? 1 : blockDimRanges_.hoRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_WO_IDX] = blockDimRanges_.woRange.empty() ? 1 : blockDimRanges_.woRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_N_IDX] = blockDimRanges_.nRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_DO_IDX] = blockDimRanges_.doRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_GROUP_IDX] = blockDimRanges_.groupRange[0];
}

void ConvBase::BlockDimDecisionBackTrackHWsplitMode(const vector<vector<uint32_t>> &inputRanges,
                                                    uint32_t rangeIdx, vector<uint32_t> &record)
{
    if (record.size() == inputRanges.size()) {
        BlockDimRes blockDimResTemp;
        SetBlockDimHWsplitMode(record, 0, blockDimResTemp);
        const uint32_t curBlockDim = blockDimResTemp.batchDim * blockDimResTemp.hoDim *
            blockDimResTemp.woDim * blockDimResTemp.nDim * blockDimResTemp.doDim *
            blockDimResTemp.groupDim;
        if (curBlockDim > aicoreNum_) {
            return;
        }

        bool updateFlag = false;
        const uint64_t curCost = CalcTotalCostHWsplitMode(blockDimResTemp);
        if (curCost < blockDimRes_.minCost) {
            updateFlag = true;
        } else if (curCost == blockDimRes_.minCost) {
            updateFlag = CmpCoreUtilizeHWsplitMode(record);
        }
        if (updateFlag) {
            SetBlockDimHWsplitMode(record, curCost, blockDimRes_);
        }
        return;
    }

    if (rangeIdx >= inputRanges.size() || rangeIdx >= blockDimInit_.size()) {
        return;
    }

    for (uint32_t i = 0; i < inputRanges[rangeIdx].size(); i++) {
        record.push_back(inputRanges[rangeIdx][i]);
        BlockDimDecisionBackTrackHWsplitMode(inputRanges, rangeIdx + 1, record);
        record.pop_back();
    }
}

void ConvBase::CoreBlockDimDecisionHWsplitMode()
{
    blockDimRes_.batchDim = static_cast<uint32_t>(1);
    blockDimRes_.hoDim = static_cast<uint32_t>(1);
    blockDimRes_.woDim = static_cast<uint32_t>(1);
    blockDimRes_.nDim = static_cast<uint32_t>(1);
    blockDimRes_.doDim = static_cast<uint32_t>(1);
    blockDimRes_.groupDim = static_cast<uint32_t>(1);
    blockDimRes_.minCost = CalcTotalCostHWsplitMode(blockDimRes_);

    vector<vector<uint32_t>> allRanges(BLOCKDIM_HWSPLIT_DEC_NUM, vector<uint32_t>(1, 1));
    vector<uint32_t> tmpHoRange = {static_cast<uint32_t>(1)};
    allRanges[BLOCKDIM_HWSPLIT_BATCH_IDX] = blockDimRanges_.batchRange;
    allRanges[BLOCKDIM_HWSPLIT_HO_IDX] = blockDimRanges_.hoRange.empty() ? tmpHoRange : blockDimRanges_.hoRange;
    allRanges[BLOCKDIM_HWSPLIT_WO_IDX] = blockDimRanges_.woRange;
    allRanges[BLOCKDIM_HWSPLIT_N_IDX] = blockDimRanges_.nRange;
    allRanges[BLOCKDIM_HWSPLIT_DO_IDX] = blockDimRanges_.doRange;
    allRanges[BLOCKDIM_HWSPLIT_GROUP_IDX] = blockDimRanges_.groupRange;
    vector<uint32_t> dimsRecord;
    BlockDimDecisionBackTrackHWsplitMode(allRanges, BLOCKDIM_HWSPLIT_BATCH_IDX, dimsRecord);
}

void ConvBase::CheckCoreUsedupHWsplitMode()
{
    // woDim is not considered because it is only used in Conv1d scene and hoDim is always 1
    if (blockDimRes_.batchDim * blockDimRes_.hoDim * blockDimRes_.nDim *
        blockDimRes_.doDim * blockDimRes_.groupDim == aicoreNum_) {
        return;
    }

    for (auto newHoDim : blockDimRanges_.hoSpareRange) {
        if (blockDimRes_.batchDim * blockDimRes_.groupDim * newHoDim *
            blockDimRes_.nDim * blockDimRes_.doDim <= aicoreNum_) {
            blockDimRes_.hoDim = std::max(blockDimRes_.hoDim, newHoDim);
        }
    }
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