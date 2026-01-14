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
 * \file conv_api_tiling_base.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV_API_TILING_BASE_H
#define ASCENDC_TILING_CONV_API_TILING_BASE_H

#include "conv_api_tiling_util.h"

namespace conv_tiling {
// [fmap, weight, bias, output]
const std::vector<std::vector<ConvDtype>> CONV_SUPPORTED_TYPES_WITH_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::HIFLOAT8},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT32, ConvDtype::BFLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16}
};

// [fmap, weight, output]
const std::vector<std::vector<ConvDtype>> CONV_SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8}
};

// [fmap, weight, bias, output]
const std::vector<std::vector<ConvDtype>> EXTENDCONV_QUANTCONV_SUPPORTED_TYPES_WITH_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::INT8},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32, ConvDtype::INT8},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::HIFLOAT8},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::BFLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT8_E4M3FN},
    {ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::INT8}
};

// [fmap, weight, output]
const std::vector<std::vector<ConvDtype>> EXTENDCONV_QUANTCONV_SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::INT8},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT8},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::BFLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN},
    {ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::INT8}
};

struct L1TilingRes {
    uint64_t kAL1 = 0; // with khkw
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    uint64_t mAL1 = 0;
    uint64_t khL1 = 0;
    uint64_t kwL1 = 0;

    IterateMNOrder iterateMNOrder = IterateMNOrder::INVALID;
    bool biasFullLoadFlag = false;
    bool fixpParamsFullLoadFlag = false;
    bool al1FullLoad = false;
    bool bl1FullLoad = false;
};

struct L0TilingRes {
    uint64_t hoL0 = 0;
    uint64_t woL0 = 0;
    uint64_t kL0 = 0;
    uint64_t nL0 = 0;
    uint64_t mL0 = 0;
};

struct UbTilingRes {
    uint64_t mUb = 0;
    uint64_t nUb = 0;
};

struct DoubleBufferRes {
    uint8_t pbAL1 = 1;
    uint8_t pbBL1 = 1;
    uint8_t pbAL0 = 1;
    uint8_t pbBL0 = 1;
    uint8_t pbCL0 = 1;
    uint64_t pBufferFlag = 0;
};

struct ShapeInfo {
    int64_t orgkH = -1;
    int64_t orgkW = -1;
    int64_t orgkD = -1;
    int64_t orgCo = -1;
    int64_t orgCi = -1;
    int64_t orgHi = -1;
    int64_t orgWi = -1;
    int64_t orgDi = -1;
    int64_t orgHo = 0;
    int64_t orgWo = 0;
    int64_t orgDo = 0;

    int64_t singlekH = -1;
    int64_t singlekW = -1;
    int64_t singlekD = -1;
    int64_t singleCo = -1;
    int64_t singleCi = -1;
    int64_t singleDo = -1;
    int64_t singleHo = -1;
    int64_t singleWo = -1;
    int64_t singleM = -1;
    int64_t singleBatch = -1;

    uint64_t singleCi1 = 0;
    uint64_t singleCo1 = 0;
    uint64_t singleM1 = 0;

    int64_t singleGroups = -1;
    int64_t singleGroupOpt = -1;
    int32_t enlarge = 0;

    // extend conv2d fixpipe params coefficient
    float channelWiseCoeff = 0;
    uint8_t quantMode0 = 0;
    uint8_t reluMode0 = 0;
    uint8_t clipMode0 = 0;
    uint8_t quantMode1 = 0;
    uint8_t reluMode1 = 0;
    uint8_t clipMode1 = 0;
    uint8_t dualOutput = 0;
};

struct AttrInfo {
    int32_t padHead = 0;
    int32_t padTail = 0;
    int32_t padTop = 0;
    int32_t padBottom = 0;
    int32_t padLeft = 0;
    int32_t padRight = 0;

    int32_t strideH = 1;
    int32_t strideW = 1;
    int32_t strideD = 1;

    int32_t dilationH = 1;
    int32_t dilationW = 1;
    int32_t dilationD = 1;

    int32_t groups = 1;
    int8_t offsetx = 0;
    int8_t roundMode = 0;
};

struct DescInfo {
    ConvType weightType = {ConvFormat::UNDEFINED, ConvDtype::UNDEFINED, TPosition::GM};
    ConvType fMapType = {ConvFormat::UNDEFINED, ConvDtype::UNDEFINED, TPosition::GM};
    ConvType biasType = {ConvFormat::UNDEFINED, ConvDtype::UNDEFINED, TPosition::GM};
    ConvType outputType = {ConvFormat::UNDEFINED, ConvDtype::UNDEFINED, TPosition::CO1};
    ConvType scaleType = {ConvFormat::UNDEFINED, ConvDtype::UNDEFINED, TPosition::GM};
};

struct CubeInfo {
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    ConvDtype madType = ConvDtype::UNDEFINED;
    ConvDtype biasType = ConvDtype::UNDEFINED;
};

class __attribute__((visibility("default"))) ConvTilingBase {
public:
    ConvTilingBase() {};
    explicit ConvTilingBase(const PlatformInfo& platform);
    virtual ~ConvTilingBase() = default;

    DescInfo descInfo;
    ShapeInfo shapeInfo;
    AttrInfo attrInfo;
    CubeInfo cubeInfo;
    PlatformInfo platformInfo;
    L1TilingRes l1TilingInfo;
    L0TilingRes l0TilingInfo;
    UbTilingRes ubTilingInfo;
    DoubleBufferRes dbValue;

    // common usd
    bool hasBias = false;
    bool hasScale = false;
    bool isScaleBiasInUb = false;
    int8_t outputOrder = static_cast<int8_t>(OutputOrder::M);
    int8_t tilingAlgorithmType = static_cast<int8_t>(TilingAlgorithmType::FORMULAS);

    // usd for conv2d
    bool isC04Flag = false;
    bool isDmaFlag = false;
    bool hf32Enable = false;
    bool hf32TransMode = false;
    bool optGroupFlag = false;
    bool enableInnerBatch = false;
    uint32_t innerBatch = 1;

    // used for extendconv2d
    bool extendConvFlag = false;
    bool quantConvFlag = false;

protected:
    virtual int64_t Compute() = 0;
    void InferCubeInfo();
    int64_t CheckTilingRes();
    bool CheckQuantUniqueAttr();
    bool CheckGroups();
    bool CheckDtype();
    std::vector<std::vector<ConvDtype>> GetSupportedDataTypes() const;
    bool CheckLoad3DLimits();
    bool CheckDmaLimits();
    uint32_t GetBandWidthCof(platform_ascendc::SocVersion curSoc) const;
};
} // namespace conv_tiling

#endif // ASCENDC_TILING_CONV_API_TILING_BASE_H