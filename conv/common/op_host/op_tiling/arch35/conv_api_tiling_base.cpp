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

#include <cstdint>
#include <unordered_set>
#include "conv_api_tiling_base.h"

using namespace std;

namespace conv_tiling {
ConvTilingBase::ConvTilingBase(const PlatformInfo& platform)
{
    platformInfo.socVersion = platform.socVersion;
    platformInfo.l1Size = platform.l1Size;
    platformInfo.l0ASize = platform.l0ASize;
    platformInfo.l0BSize = platform.l0BSize;
    platformInfo.l0CSize = platform.l0CSize;
    platformInfo.ubSize = platform.ubSize;
    platformInfo.btSize = platform.btSize;
    platformInfo.fbSize = platform.fbSize;
}

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
    if (outputOrder == static_cast<int8_t>(OutputOrder::M) &&
        (l1TilingInfo.mAL1 == 0 || l0TilingInfo.mL0 == 0)) {
        TILING_LOG_ERROR("Check tiling res failed: mAL1: %lu, mL0: %lu.", l1TilingInfo.mAL1, l0TilingInfo.mL0);
        return -1;
    }
    if (outputOrder == static_cast<int8_t>(OutputOrder::HW) &&
        (l1TilingInfo.hoAL1 == 0 || l1TilingInfo.woAL1 == 0 || l0TilingInfo.hoL0 == 0 || l0TilingInfo.woL0 == 0)) {
        TILING_LOG_ERROR("Check tiling res failed: hoAL1: %lu, woAL1: %lu, hoL0: %lu, woL0: %lu.",
            l1TilingInfo.hoAL1, l1TilingInfo.woAL1, l0TilingInfo.hoL0, l0TilingInfo.woL0);
        return -1;
    }

    if (l1TilingInfo.kAL1 == 0 || l1TilingInfo.kBL1 == 0 || l1TilingInfo.nBL1 == 0) {
        TILING_LOG_ERROR("Check tiling res failed: kAL1: %lu, kBL1: %lu, nBL1: %lu.",
            l1TilingInfo.kAL1, l1TilingInfo.kBL1, l1TilingInfo.nBL1);
        return -1;
    }

    if (l0TilingInfo.kL0 == 0 || l0TilingInfo.nL0 == 0) {
        TILING_LOG_ERROR("Check tiling res failed: kL0: %lu, nL0: %lu.", l0TilingInfo.kL0, l0TilingInfo.nL0);
        return -1;
    }

    return 0;
}

uint32_t ConvTilingBase::GetBandWidthCof(platform_ascendc::SocVersion curSoc) const
{
    if (descInfo.weightType.format == ConvFormat::FRACTAL_Z ||
        descInfo.weightType.format == ConvFormat::FRACTAL_Z_C04) {
        return 1;
    }
    switch (curSoc) {
        case platform_ascendc::SocVersion::ASCEND910B:
            return 1;
        case platform_ascendc::SocVersion::ASCEND950:
            if (descInfo.fMapType.format == ConvFormat::NCHW || descInfo.fMapType.format == ConvFormat::NCDHW) {
                return BAND_WIDTH_COEFF;
            }
            return 1;
        case platform_ascendc::SocVersion::ASCEND910_55:
            return BAND_WIDTH_COEFF;
        default:
            return 0;
    }
}

bool ConvTilingBase::CheckQuantUniqueAttr()
{
    if (quantConvFlag) {
        ConvDtype fMapDtype = descInfo.fMapType.dtype;
        ConvDtype outputDtype = descInfo.outputType.dtype;
        if (attrInfo.offsetx != 0 &&
            (fMapDtype == ConvDtype::HIFLOAT8 || fMapDtype == ConvDtype::FLOAT8_E4M3FN)) {
                TILING_LOG_ERROR(
                    "Only support offsetX = 0 in hif8 or fp8 mode, actually is %d",
                    attrInfo.offsetx);
                return false;
        }
        if (outputDtype == ConvDtype::HIFLOAT8) {
            if (attrInfo.roundMode != conv_tiling::ROUND_MODE_ROUND) {
                TILING_LOG_ERROR(
                    "Only support round in hif8 dtype, actually is %d",
                    attrInfo.roundMode);
                return false;
            }
        } else if (outputDtype == ConvDtype::INT8 || outputDtype == ConvDtype::FLOAT8_E4M3FN) {
            if (attrInfo.roundMode != conv_tiling::ROUND_MODE_RINT) {
                TILING_LOG_ERROR("Only support rint in not_hif8 dtype, actually is %d",
                    attrInfo.roundMode);
                return false;
            }
        }
    }
    return true;
}

bool ConvTilingBase::CheckGroups()
{
    if (attrInfo.groups <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: groups=%d which must > 0.", attrInfo.groups);
        return false;
    }
    if (optGroupFlag) {//this->optGroupFlag
        if (shapeInfo.singleGroups <= 0) {
            TILING_LOG_ERROR("Illegal attrs have set: singleGroups=%ld which must > 0.", shapeInfo.singleGroups);
            return false;
        }
        if (shapeInfo.enlarge <= 0) {
            TILING_LOG_ERROR("Illegal attrs have set: enlarge=%d which must > 0.", shapeInfo.enlarge);
            return false;
        }
        if (shapeInfo.singleGroupOpt <= 0) {
            TILING_LOG_ERROR("Input illegal singleGroupOpt: %ld, which must > 0.", shapeInfo.singleGroupOpt);
            return false;
        }
    }
    return true;
}

bool ConvTilingBase::CheckDtype()
{
    vector<ConvDtype> paramsType;
    if (hasBias) {
        paramsType = {descInfo.fMapType.dtype, descInfo.weightType.dtype,
                      descInfo.biasType.dtype, descInfo.outputType.dtype};
    } else {
        paramsType = {descInfo.fMapType.dtype, descInfo.weightType.dtype,
                      descInfo.outputType.dtype};
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
        TILING_LOG_ERROR("unSupported params data type [fmap, weight, bias, output]: [%s, %s, %s, %s].",
            DTYPE_TO_STR.at(descInfo.fMapType.dtype).c_str(),
            DTYPE_TO_STR.at(descInfo.weightType.dtype).c_str(),
            DTYPE_TO_STR.at(descInfo.biasType.dtype).c_str(),
            DTYPE_TO_STR.at(descInfo.outputType.dtype).c_str());
    } else {
        TILING_LOG_ERROR("unSupported params data type [fmap, weight, output]: [%s, %s, %s].",
            DTYPE_TO_STR.at(descInfo.fMapType.dtype).c_str(),
            DTYPE_TO_STR.at(descInfo.weightType.dtype).c_str(),
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

bool ConvTilingBase::CheckLoad3DLimits()
{
    if (static_cast<uint32_t>(attrInfo.strideH) > LOAD3D_MAX_STRIDE_H_W ||
        static_cast<uint32_t>(attrInfo.strideW) > LOAD3D_MAX_STRIDE_H_W) {
        TILING_LOG_DEBUG(
            "Attrs not satisfying load3d's limits: strideH=%d, strideW=%d, which must <= %u.",
            attrInfo.strideH, attrInfo.strideW, LOAD3D_MAX_STRIDE_H_W);
        return false;
    }
    if (static_cast<uint32_t>(attrInfo.dilationH) > LOAD3D_MAX_DILATION_H_W ||
        static_cast<uint32_t>(attrInfo.dilationW) > LOAD3D_MAX_DILATION_H_W) {
        TILING_LOG_DEBUG(
            "Attrs not satisfying load3d's limits: dilationH=%d, dilationW=%d, which must <= %u.",
            attrInfo.dilationH, attrInfo.dilationW, LOAD3D_MAX_DILATION_H_W);
        return false;
    }
    if (static_cast<uint32_t>(attrInfo.padLeft) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(attrInfo.padRight) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(attrInfo.padTop) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(attrInfo.padBottom) > LOAD3D_MAX_PAD) {
        TILING_LOG_DEBUG(
            "Attrs not satisfying load3d's limits: padTop=%d, padBottom=%d, padLeft=%d, padRight=%d, which must <= %u.",
            attrInfo.padTop, attrInfo.padBottom, attrInfo.padLeft, attrInfo.padRight,
            LOAD3D_MAX_PAD);
        return false;
    }
    if (static_cast<uint64_t>(shapeInfo.orgkH) > LOAD3D_MAX_FILTER_H_W ||
        static_cast<uint64_t>(shapeInfo.orgkW) > LOAD3D_MAX_FILTER_H_W) {
        TILING_LOG_DEBUG("Weight shpae not satisfy Load3D's limits: kh=%ld, kw=%ld, which must <= %u.",
                         shapeInfo.orgkH, shapeInfo.orgkW, LOAD3D_MAX_FILTER_H_W);
        return false;
    }
    uint64_t tmpkHWSize = static_cast<uint64_t>(shapeInfo.orgkH) * static_cast<uint64_t>(shapeInfo.orgkW) *
                          cubeInfo.k0;
    if (tmpkHWSize > LOAD3D_MAX_DDR2L1_SIZE) {
        TILING_LOG_DEBUG("Weight shape not satisfying load3d's limits: kH*kW*k0=%lu, which must <= %u.",
                         tmpkHWSize, LOAD3D_MAX_DDR2L1_SIZE);
        return false;
    }
    return true;
}
 
bool ConvTilingBase::CheckDmaLimits()
{
    if (attrInfo.groups != 1) {
        TILING_LOG_ERROR("DMA only support groups=1, actual groups=%u.", attrInfo.groups);
        return false;
    }

    uint64_t woAL1Min = cubeInfo.m0;
    uint64_t hoAL1Min = 1;
 
    uint64_t fmapUbSizeMin =
        AlignB(hoAL1Min * woAL1Min * cubeInfo.k0 * DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype), C0_BYTE_SIZE);
    if (fmapUbSizeMin > platformInfo.ubSize) {
        TILING_LOG_ERROR("DMA min ub size not enough: platformInfo.ubSize=%lu, fmapUbSizeMin=%lu.",
            platformInfo.ubSize, fmapUbSizeMin);
        return false;
    }
 
    return true;
}

}