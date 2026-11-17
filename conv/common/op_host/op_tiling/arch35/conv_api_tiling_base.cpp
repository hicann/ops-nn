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
 * \file conv_api_tiling_base.cpp
 * \brief
 */

#include "conv_api_tiling_base.h"
#include <cstdint>
#include <unordered_set>
#include <sstream>
#include "conv_api_tiling_algorithm_BBmode.h"

using namespace std;
using namespace conv_tiling_algo_bb;

namespace conv_tiling {
ConvTilingBase::ConvTilingBase(const PlatformInfo& platform)
{
    platformInfo.npuArch = platform.npuArch;
    platformInfo.isCubeVectorFuse = platform.isCubeVectorFuse;
    platformInfo.l1Size = platform.l1Size;
    platformInfo.l0ASize = platform.l0ASize;
    platformInfo.l0BSize = platform.l0BSize;
    platformInfo.l0CSize = platform.l0CSize;
    platformInfo.ubSize = platform.ubSize;
    platformInfo.btSize = platform.btSize;
    platformInfo.fbSize = platform.fbSize;
    platformInfo.aivPerAic = platform.aivPerAic;
    platformInfo.aicoreNum = platform.aicoreNum;
}

void ConvTilingBase::SetNodeType(string inType) { nodeType = inType; }

void ConvTilingBase::InferCubeInfo()
{
    cubeInfo.m0 = CUBE_MKN_TAB.GetMKN(descInfo.fMapType.dtype, MKN_M_INDEX);
    cubeInfo.k0 = CUBE_MKN_TAB.GetMKN(descInfo.weightType.dtype, MKN_K_INDEX);
    cubeInfo.n0 = CUBE_MKN_TAB.GetMKN(descInfo.fMapType.dtype, MKN_N_INDEX);
    cubeInfo.biasType = CUBE_TYPE_TAB.ToBiasType(descInfo.fMapType.dtype);
    cubeInfo.madType = CUBE_TYPE_TAB.ToMadType(descInfo.fMapType.dtype);
}

int64_t ConvTilingBase::CheckTilingRes()
{
    if (outputOrder == static_cast<int8_t>(OutputOrder::M) && (l1TilingInfo.mAL1 == 0 || l0TilingInfo.mL0 == 0)) {
        OP_LOGE(nodeType, "Check tiling res failed: mAL1: %lu, mL0: %lu.", l1TilingInfo.mAL1, l0TilingInfo.mL0);
        return -1;
    }
    if (outputOrder == static_cast<int8_t>(OutputOrder::HW) &&
        (l1TilingInfo.hoAL1 == 0 || l1TilingInfo.woAL1 == 0 || l0TilingInfo.hoL0 == 0 || l0TilingInfo.woL0 == 0)) {
        OP_LOGE(nodeType, "Check tiling res failed: hoAL1: %lu, woAL1: %lu, hoL0: %lu, woL0: %lu.", l1TilingInfo.hoAL1,
                l1TilingInfo.woAL1, l0TilingInfo.hoL0, l0TilingInfo.woL0);
        return -1;
    }

    if (l1TilingInfo.kAL1 == 0 || l1TilingInfo.kBL1 == 0 || l1TilingInfo.nBL1 == 0) {
        OP_LOGE(nodeType, "Check tiling res failed: kAL1: %lu, kBL1: %lu, nBL1: %lu.", l1TilingInfo.kAL1,
                l1TilingInfo.kBL1, l1TilingInfo.nBL1);
        return -1;
    }

    if (l0TilingInfo.kL0 == 0 || l0TilingInfo.nL0 == 0) {
        OP_LOGE(nodeType, "Check tiling res failed: kL0: %lu, nL0: %lu.", l0TilingInfo.kL0, l0TilingInfo.nL0);
        return -1;
    }

    return 0;
}

uint32_t ConvTilingBase::GetBandWidthCof() const
{
    if (descInfo.weightType.format == ConvFormat::FRACTAL_Z ||
        descInfo.weightType.format == ConvFormat::FRACTAL_Z_C04 ||
        descInfo.weightType.format == ConvFormat::FRACTAL_Z_3D) {
        return 1;
    }

    if (platformInfo.npuArch == NpuArch::DAV_9201 || platformInfo.npuArch == NpuArch::DAV_9202) {
        return 1;
    }

    if (descInfo.fMapType.format == ConvFormat::NCHW || descInfo.fMapType.format == ConvFormat::NCDHW) {
        return BAND_WIDTH_COEFF;
    }
    return 1;
}

bool ConvTilingBase::CheckQuantUniqueAttr()
{
    if (quantConvFlag) {
        ConvDtype fMapDtype = descInfo.fMapType.dtype;
        ConvDtype outputDtype = descInfo.outputType.dtype;
        if (attrInfo.offsetx != 0 && (fMapDtype == ConvDtype::HIFLOAT8 || fMapDtype == ConvDtype::FLOAT8_E4M3FN)) {
            OP_LOGE(nodeType, "Only support offsetX = 0 in hif8 or fp8 mode, actually is %d", attrInfo.offsetx);
            return false;
        }
        if (outputDtype == ConvDtype::HIFLOAT8) {
            if (attrInfo.roundMode != conv_tiling::ROUND_MODE_ROUND) {
                OP_LOGE(nodeType, "Only support round in hif8 dtype, actually is %d", attrInfo.roundMode);
                return false;
            }
        } else if (outputDtype == ConvDtype::INT8 || outputDtype == ConvDtype::FLOAT8_E4M3FN) {
            if (attrInfo.roundMode != conv_tiling::ROUND_MODE_RINT) {
                OP_LOGE(nodeType, "Only support rint in not_hif8 dtype, actually is %d", attrInfo.roundMode);
                return false;
            }
        }
    }
    return true;
}

bool ConvTilingBase::CheckGroups()
{
    if (attrInfo.groups <= 0) {
        OP_LOGE(nodeType, "Illegal attrs have set: groups=%d which must > 0.", attrInfo.groups);
        return false;
    }
    if (optGroupFlag) { // this->optGroupFlag
        if (shapeInfo.singleGroups <= 0) {
            OP_LOGE(nodeType, "Illegal attrs have set: singleGroups=%ld which must > 0.", shapeInfo.singleGroups);
            return false;
        }
        if (shapeInfo.enlarge <= 0) {
            OP_LOGE(nodeType, "Illegal attrs have set: enlarge=%d which must > 0.", shapeInfo.enlarge);
            return false;
        }
        if (shapeInfo.singleGroupOpt <= 0) {
            OP_LOGE(nodeType, "Input illegal singleGroupOpt: %ld, which must > 0.", shapeInfo.singleGroupOpt);
            return false;
        }
    }
    return true;
}

bool ConvTilingBase::CheckDtype()
{
    vector<ConvDtype> paramsType;
    if (hasBias) {
        paramsType = {descInfo.fMapType.dtype, descInfo.weightType.dtype, descInfo.biasType.dtype,
                      descInfo.outputType.dtype};
    } else {
        paramsType = {descInfo.fMapType.dtype, descInfo.weightType.dtype, descInfo.outputType.dtype};
    }
    auto supportedTypesList = GetSupportedDataTypes();
    for (uint64_t kindsId = 0; kindsId < supportedTypesList.size(); ++kindsId) {
        if (supportedTypesList[kindsId].size() == 0) {
            continue;
        }
        if (IsEqual(paramsType, supportedTypesList[kindsId], supportedTypesList[kindsId].size())) {
            return true;
        }
    }
    if (hasBias) {
        OP_LOGE(nodeType, "unSupported params data type [fmap, weight, bias, output]: [%s, %s, %s, %s].",
                DTYPE_TO_STR.at(descInfo.fMapType.dtype).c_str(), DTYPE_TO_STR.at(descInfo.weightType.dtype).c_str(),
                DTYPE_TO_STR.at(descInfo.biasType.dtype).c_str(), DTYPE_TO_STR.at(descInfo.outputType.dtype).c_str());
    } else {
        OP_LOGE(nodeType, "unSupported params data type [fmap, weight, output]: [%s, %s, %s].",
                DTYPE_TO_STR.at(descInfo.fMapType.dtype).c_str(), DTYPE_TO_STR.at(descInfo.weightType.dtype).c_str(),
                DTYPE_TO_STR.at(descInfo.outputType.dtype).c_str());
    }
    return false;
}

vector<vector<ConvDtype>> ConvTilingBase::GetSupportedDataTypes() const
{
    vector<vector<ConvDtype>> res;
    if (hasBias) {
        // [fmap, weight, bias, output]
        if (extendConvFlag || quantConvFlag) {
            res = EXTENDCONV_QUANTCONV_SUPPORTED_TYPES_WITH_BIAS;
        } else {
            res = CONV_SUPPORTED_TYPES_WITH_BIAS;
        }
    } else {
        // [fmap, weight, output]
        if (extendConvFlag || quantConvFlag) {
            res = EXTENDCONV_QUANTCONV_SUPPORTED_TYPES_WITHOUT_BIAS;
        } else {
            res = CONV_SUPPORTED_TYPES_WITHOUT_BIAS;
        }
    }

    return res;
}

void ConvTilingBase::LogLoad3DLimitError(const std::string& paramName, const std::string& actualValue,
                                         const std::string& reason) const
{
    if (platformInfo.isCubeVectorFuse) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeType.c_str(), paramName.c_str(), actualValue.c_str(), reason.c_str());
    } else {
        OP_LOGD(nodeType.c_str(), "AscendC: %s", reason.c_str());
    }
}

bool ConvTilingBase::CheckLoad3DAttrLimits()
{
    if (static_cast<uint32_t>(attrInfo.strideH) > LOAD3D_MAX_STRIDE_H_W ||
        static_cast<uint32_t>(attrInfo.strideW) > LOAD3D_MAX_STRIDE_H_W) {
        std::stringstream ssActual, ssReason;
        ssActual << "strideH=" << attrInfo.strideH << ", strideW=" << attrInfo.strideW;
        ssReason << "Attrs does not satisfy Load3D's limits: strideH=" << attrInfo.strideH
                 << ", strideW=" << attrInfo.strideW << ", which must <= " << LOAD3D_MAX_STRIDE_H_W;
        LogLoad3DLimitError("strides", ssActual.str(), ssReason.str());
        return false;
    }
    if (static_cast<uint32_t>(attrInfo.dilationH) > LOAD3D_MAX_DILATION_H_W ||
        static_cast<uint32_t>(attrInfo.dilationW) > LOAD3D_MAX_DILATION_H_W) {
        std::stringstream ssActual, ssReason;
        ssActual << "dilationH=" << attrInfo.dilationH << ", dilationW=" << attrInfo.dilationW;
        ssReason << "Attrs does not satisfy Load3D's limits: dilationH=" << attrInfo.dilationH
                 << ", dilationW=" << attrInfo.dilationW << ", which must <= " << LOAD3D_MAX_DILATION_H_W;
        LogLoad3DLimitError("dilations", ssActual.str(), ssReason.str());
        return false;
    }
    if (static_cast<uint32_t>(attrInfo.padLeft) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(attrInfo.padRight) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(attrInfo.padTop) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(attrInfo.padBottom) > LOAD3D_MAX_PAD) {
        std::stringstream ssActual, ssReason;
        ssActual << "padTop=" << attrInfo.padTop << ", padBottom=" << attrInfo.padBottom
                 << ", padLeft=" << attrInfo.padLeft << ", padRight=" << attrInfo.padRight;
        ssReason << "Attrs does not satisfy Load3D's limits: padTop=" << attrInfo.padTop
                 << ", padBottom=" << attrInfo.padBottom << ", padLeft=" << attrInfo.padLeft
                 << ", padRight=" << attrInfo.padRight << ", which must <= " << LOAD3D_MAX_PAD;
        LogLoad3DLimitError("pads", ssActual.str(), ssReason.str());
        return false;
    }
    return true;
}

bool ConvTilingBase::CheckLoad3DWeightLimits()
{
    if (static_cast<uint64_t>(shapeInfo.orgkH) > LOAD3D_MAX_FILTER_H_W ||
        static_cast<uint64_t>(shapeInfo.orgkW) > LOAD3D_MAX_FILTER_H_W) {
        std::stringstream ssActual, ssReason;
        ssActual << "kh=" << shapeInfo.orgkH << ", kw=" << shapeInfo.orgkW;
        ssReason << "Weight shape does not satisfy Load3D's limits: kh=" << shapeInfo.orgkH
                 << ", kw=" << shapeInfo.orgkW << ", which must <= " << LOAD3D_MAX_FILTER_H_W;
        LogLoad3DLimitError("filter", ssActual.str(), ssReason.str());
        return false;
    }
    auto k0 = CUBE_MKN_TAB.GetMKN(descInfo.weightType.dtype, MKN_K_INDEX);
    uint64_t tmpkHWSize = static_cast<uint64_t>(shapeInfo.orgkH) * static_cast<uint64_t>(shapeInfo.orgkW) * k0;
    if (tmpkHWSize > LOAD3D_MAX_DDR2L1_SIZE) {
        std::stringstream ssActual, ssReason;
        ssActual << "kH*kW*k0=" << tmpkHWSize;
        ssReason << "Weight shape does not satisfy Load3D's limits: kH*kW*k0=" << tmpkHWSize
                 << ", which must <= " << LOAD3D_MAX_DDR2L1_SIZE;
        LogLoad3DLimitError("filter", ssActual.str(), ssReason.str());
        return false;
    }
    return true;
}

bool ConvTilingBase::CheckLoad3DAllPadLimits()
{
    uint64_t dilatedKernelH = (static_cast<uint64_t>(shapeInfo.orgkH) - 1) * static_cast<uint64_t>(attrInfo.dilationH) +
                              1;
    uint64_t dilatedKernelW = (static_cast<uint64_t>(shapeInfo.orgkW) - 1) * static_cast<uint64_t>(attrInfo.dilationW) +
                              1;
    bool allPadH = static_cast<uint64_t>(attrInfo.padTop) >= dilatedKernelH ||
                   static_cast<uint64_t>(attrInfo.padBottom) >= dilatedKernelH;
    bool allPadW = static_cast<uint64_t>(attrInfo.padLeft) >= dilatedKernelW ||
                   static_cast<uint64_t>(attrInfo.padRight) >= dilatedKernelW;
    if ((allPadH || allPadW) &&
        (dilatedKernelH > LOAD3D_ALLPAD_VIRTUAL_LIMIT || dilatedKernelW > LOAD3D_ALLPAD_VIRTUAL_LIMIT)) {
        std::stringstream ssActual, ssReason;
        ssActual << "dilatedKernelH=" << dilatedKernelH << ", dilatedKernelW=" << dilatedKernelW;
        ssReason << "Attrs does not satisfy Load3D's limits: Load3D allPad overflow (dilatedKernel > 511), "
                    "fall back to DMA.";
        LogLoad3DLimitError("pads", ssActual.str(), ssReason.str());
        return false;
    }
    return true;
}

bool ConvTilingBase::CheckLoad3DLimits()
{
    return CheckLoad3DAttrLimits() && CheckLoad3DWeightLimits() && CheckLoad3DAllPadLimits();
}

bool ConvTilingBase::CheckTilingAlgorithmType(BasicBlockInfo& basicBlockInfo, int8_t phase)
{
    if (phase == BB_PHASE_1) {
        if (!CheckParamsBBModePhase1()) {
            OP_LOGE(nodeType, "conv api tiling general params check failed in phase 1.");
            return false;
        }
    } else {
        if (!CheckParams()) {
            OP_LOGE(nodeType, "conv api tiling general params check failed in phase 2.");
            return false;
        }
    }
    InferCubeInfo();
    if (!CheckL1SizeLimit()) {
        OP_LOGE(nodeType, "conv api tiling L1size check failed.");
        return false;
    }

    if (phase == BB_PHASE_1) {
        if (!CheckBasicBlockInfoPhase1(basicBlockInfo)) {
            OP_LOGE(nodeType, "conv basic block info check failed in phase 1.");
            return false;
        }
    } else {
        if (!CheckBasicBlockInfoPhase2(basicBlockInfo)) {
            OP_LOGE(nodeType, "conv basic block info check failed in phase 2.");
            return false;
        }
    }
    if (CheckBBmodeLimits(basicBlockInfo)) {
        tilingAlgorithmType = static_cast<int8_t>(TilingAlgorithmType::BASIC_BLOCK);
    }
    switch (tilingAlgorithmType) {
        case static_cast<int8_t>(TilingAlgorithmType::BASIC_BLOCK):
            return true;
        case static_cast<int8_t>(TilingAlgorithmType::FORMULAS):
        default:
            OP_LOGE(nodeType, "Unsupported tiling algorithm type %d, only support BASIC_BLOCK(%d).",
                    tilingAlgorithmType, static_cast<int8_t>(TilingAlgorithmType::BASIC_BLOCK));
            return false;
    }
}

bool ConvTilingBase::CheckL1SizeLimit()
{
    uint32_t fMapDtypeSize = DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);
    uint32_t weightDtypeSize = DTYPE_SIZE_TAB.at(descInfo.weightType.dtype);
    // ensure 32-byte alignment
    uint64_t minBiasSize = hasBias ? AlignB(cubeInfo.n0 * DTYPE_SIZE_TAB.at(descInfo.biasType.dtype), C0_SIZE) : 0;
    uint64_t minScaleSize = 0;
    if (hasScale) {
        minScaleSize = static_cast<uint64_t>(shapeInfo.channelWiseCoeff * cubeInfo.n0 *
                                             DTYPE_SIZE_TAB.at(descInfo.scaleType.dtype));
    }
    uint64_t minBL1Size = static_cast<uint64_t>(shapeInfo.orgkH) * static_cast<uint64_t>(shapeInfo.orgkW) *
                          cubeInfo.n0 * cubeInfo.k0 * weightDtypeSize;
    uint64_t minAL1Size = 0;
    uint64_t minHoAL1 = min(MIN_M_L1_SIZE / static_cast<uint64_t>(shapeInfo.orgWo) + CONST_VALUE_2,
                            static_cast<uint64_t>(shapeInfo.orgHo));
    uint64_t khd = static_cast<uint64_t>(attrInfo.dilationH) * (static_cast<uint64_t>(shapeInfo.orgkH) - 1) + 1;
    uint64_t minHiAL1 = (minHoAL1 - 1) * static_cast<uint64_t>(attrInfo.strideH) + khd;
    minHiAL1 = (minHiAL1 < static_cast<uint64_t>(shapeInfo.orgHi)) ? minHiAL1 : static_cast<uint64_t>(shapeInfo.orgHi);
    minAL1Size = minHiAL1 * static_cast<uint64_t>(shapeInfo.orgWi) * cubeInfo.k0 * fMapDtypeSize * innerBatch;
    uint64_t minL1LoadSize = minAL1Size + minBL1Size + minBiasSize + minScaleSize;
    if (minL1LoadSize > platformInfo.l1Size) {
        OP_LOGD(nodeType, "CheckL1SizeLimit, minimum L1 size: %lu, maxL1Size: %lu", minL1LoadSize, platformInfo.l1Size);
        return false;
    }
    return true;
}

bool ConvTilingBase::CheckBasicBlockInfoPhase1(BasicBlockInfo& basicBlockInfo) const
{
    if (basicBlockInfo.fDim == 0) {
        OP_LOGE(nodeType, "fDim is illegal which is: %u", basicBlockInfo.fDim);
        return false;
    }
    if (basicBlockInfo.nDim == 0) {
        OP_LOGE(nodeType, "nDim is illegal which is: %u", basicBlockInfo.nDim);
        return false;
    }
    if (basicBlockInfo.groupDim == 0) {
        OP_LOGE(nodeType, "groupDim is illegal which is: %u", basicBlockInfo.groupDim);
        return false;
    }
    if (basicBlockInfo.aicoreNum == 0) {
        OP_LOGE(nodeType, "aicoreNum is illegal which is: %u", basicBlockInfo.aicoreNum);
        return false;
    }
    if (basicBlockInfo.mIn == 0) {
        OP_LOGE(nodeType, "mIn is illegal which is: %lu", basicBlockInfo.mIn);
        return false;
    }
    if (basicBlockInfo.mTile == 0) {
        OP_LOGE(nodeType, "mTile is illegal which is: %u", basicBlockInfo.mTile);
        return false;
    }
    if (basicBlockInfo.nTile == 0) {
        OP_LOGE(nodeType, "nTile is illegal which is: %u", basicBlockInfo.nTile);
        return false;
    }
    if (basicBlockInfo.mCut == 0) {
        OP_LOGE(nodeType, "mCut is illegal which is: %u", basicBlockInfo.mCut);
        return false;
    }
    if (basicBlockInfo.nCut == 0) {
        OP_LOGE(nodeType, "nCut is illegal which is: %u", basicBlockInfo.nCut);
        return false;
    }
    if (basicBlockInfo.batch == 0) {
        OP_LOGE(nodeType, "batch is illegal which is: %u", basicBlockInfo.batch);
        return false;
    }
    if (basicBlockInfo.doCut == 0) {
        OP_LOGE(nodeType, "doCut is illegal which is: %u", basicBlockInfo.doCut);
        return false;
    }
    return true;
}

bool ConvTilingBase::CheckBasicBlockInfoPhase2(BasicBlockInfo& basicBlockInfo)
{
    if (basicBlockInfo.batchDim == 0) {
        OP_LOGE(nodeType, "batchDim is illegal which is: %u", basicBlockInfo.batchDim);
        return false;
    }
    if (basicBlockInfo.mDim == 0) {
        OP_LOGE(nodeType, "mDim is illegal which is: %u", basicBlockInfo.mDim);
        return false;
    }
    if (basicBlockInfo.nDim == 0) {
        OP_LOGE(nodeType, "nDim is illegal which is: %u", basicBlockInfo.nDim);
        return false;
    }
    if (basicBlockInfo.mTile == 0) {
        OP_LOGE(nodeType, "mTile is illegal which is: %u", basicBlockInfo.mTile);
        return false;
    }
    if (basicBlockInfo.nTile == 0) {
        OP_LOGE(nodeType, "nTile is illegal which is: %u", basicBlockInfo.nTile);
        return false;
    }
    if (basicBlockInfo.mIn == 0) {
        OP_LOGE(nodeType, "mIn is illegal which is: %lu", basicBlockInfo.mIn);
        return false;
    }
    if (static_cast<uint8_t>(basicBlockInfo.iterateMNOrder) == static_cast<uint8_t>(IterateMNOrder::INVALID)) {
        OP_LOGE(nodeType, "iterateMNOrder is illegal which is: %u (INVALID)",
                static_cast<uint8_t>(basicBlockInfo.iterateMNOrder));
        return false;
    }
    return true;
}

bool ConvTilingBase::CheckParams()
{
    if (!CheckAlgorithmLimit()) {
        return false;
    }
    if (!CheckAttr()) {
        return false;
    }
    if (!CheckShape()) {
        return false;
    }
    if (!CheckFormat()) {
        return false;
    }
    if (!CheckDtype()) {
        return false;
    }
    if (!CheckInstructionLimits()) {
        return false;
    }
    return true;
}

bool ConvTilingBase::CheckParamsBBModePhase1()
{
    if (!CheckAlgorithmLimit()) {
        return false;
    }
    if (!CheckAttrBBModePhase1()) {
        return false;
    }
    if (!CheckShapeBBModePhase1()) {
        return false;
    }
    if (!CheckFormat()) {
        return false;
    }
    if (!CheckDtype()) {
        return false;
    }
    if (!CheckInstructionLimits()) {
        return false;
    }
    return true;
}

bool ConvTilingBase::EvalCandidateCoreBinding(BasicBlockInfo& basicBlockInfo)
{
    if (!CheckTilingAlgorithmType(basicBlockInfo, BB_PHASE_1)) {
        return false;
    }
    Infer5hdShape();
    ConvTilingAlgorithmBBmode algoBB(this, basicBlockInfo);
    algoBB.AdjustM();
    algoBB.AdjustN();
    algoBB.CalcBestL1LoadStratgy();
    algoBB.CalcCoreUtilization();
    return true;
}
} // namespace conv_tiling
