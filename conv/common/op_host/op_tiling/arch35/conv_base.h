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
 * \file conv_base.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_BASE_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_BASE_H

#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "log/log.h"
#include "../cube_tiling.h"
#include "conv_base_utils.h"
#include "conv_api_tiling_util.h"
namespace optiling {
namespace conv_ops_tiling {
using conv_tiling::ConvDtype;
using conv_tiling::ConvFormat;
using conv_tiling::ConvGroupType;

static std::map<platform_ascendc::SocVersion, std::string> socNameTab = {
    {platform_ascendc::SocVersion::ASCEND910_95, "Ascend910_95"},
    {platform_ascendc::SocVersion::ASCEND910_55, "Ascend910_55"},
    {platform_ascendc::SocVersion::RESERVED_VERSION, "RESERVED_VERSION"}
};

static map<string, platform_ascendc::SocVersion> socConvertMap = {
    {"Ascend910_9589", platform_ascendc::SocVersion::ASCEND910_95},
    {"Ascend910_5591", platform_ascendc::SocVersion::ASCEND910_55},
};

static map<platform_ascendc::SocVersion, uint64_t> socBTsizeMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, BT_SIZE_910_95},
    {platform_ascendc::SocVersion::ASCEND910_55, BT_SIZE_910_55},
    {platform_ascendc::SocVersion::RESERVED_VERSION, 0}
};

static map<platform_ascendc::SocVersion, uint64_t> socFBsizeMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, FB_SIZE_910_95},
    {platform_ascendc::SocVersion::ASCEND910_55, FB_SIZE_910_55},
    {platform_ascendc::SocVersion::RESERVED_VERSION, 0}
};

static std::map<ge::DataType, std::string> dtypeToStrTab = {
    {ge::DataType::DT_FLOAT, "float32"}, {ge::DataType::DT_FLOAT16, "float16"},
    {ge::DataType::DT_BF16, "bfloat16"}, {ge::DataType::DT_INT64, "int64"},
    {ge::DataType::DT_UINT64, "uint64"}, {ge::DataType::DT_INT32, "int32"},
    {ge::DataType::DT_INT8, "int8"}, {ge::DataType::DT_HIFLOAT8, "hifloat8"},
    {ge::DataType::DT_FLOAT8_E4M3FN, "float8_e4m3fn"},
    {ge::DataType::DT_INT16, "int16"},
    {ge::DataType::DT_UINT8, "uint8"},
    {ge::DataType::DT_UINT16, "uint16"},
    {ge::DataType::DT_UINT32, "uint32"},
    {ge::DataType::DT_DOUBLE, "double"},
    {ge::DataType::DT_INT4, "int4"},
    {ge::DataType::DT_UNDEFINED, "undefined"}
};

static std::map<ge::DataType, ConvDtype> dtypeMap = {
    {ge::DT_FLOAT16, ConvDtype::FLOAT16}, {ge::DT_FLOAT, ConvDtype::FLOAT32},
    {ge::DT_BF16, ConvDtype::BFLOAT16}, {ge::DT_INT8, ConvDtype::INT8},
    {ge::DT_INT64, ConvDtype::INT64}, {ge::DT_UINT64, ConvDtype::UINT64},
    {ge::DT_INT32, ConvDtype::INT32}, {ge::DT_HIFLOAT8, ConvDtype::HIFLOAT8},
    {ge::DT_FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN},
    {ge::DT_INT16, ConvDtype::INT16},
    {ge::DT_UINT8, ConvDtype::UINT8},
    {ge::DT_UINT16, ConvDtype::UINT16},
    {ge::DT_UINT32, ConvDtype::UINT32},
    {ge::DT_DOUBLE, ConvDtype::DOUBLE},
    {ge::DT_INT4, ConvDtype::INT4},
    {ge::DT_UNDEFINED, ConvDtype::UNDEFINED}
};

static std::map<ge::Format, std::string> formatToStrTab = {
    {ge::FORMAT_NCHW, "NCHW"}, {ge::FORMAT_NHWC, "NHWC"},
    {ge::FORMAT_HWCN, "HWCN"}, {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"},
    {ge::FORMAT_NCDHW, "NCDHW"}, {ge::FORMAT_NC1HWC0, "NC1HWC0"},
    {ge::FORMAT_ND, "ND"}, {ge::FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
    {ge::FORMAT_FRACTAL_Z, "FRACTAL_Z"}
};

static std::map<ge::Format, ConvFormat> formatMap = {
    {ge::FORMAT_ND, ConvFormat::ND}, {ge::FORMAT_NCHW, ConvFormat::NCHW},
    {ge::FORMAT_NHWC, ConvFormat::NHWC}, {ge::FORMAT_HWCN, ConvFormat::HWCN},
    {ge::FORMAT_DHWNC, ConvFormat::DHWNC}, {ge::FORMAT_DHWCN, ConvFormat::DHWCN},
    {ge::FORMAT_NDHWC, ConvFormat::NDHWC}, {ge::FORMAT_NCDHW, ConvFormat::NCDHW},
    {ge::FORMAT_NC1HWC0, ConvFormat::NC1HWC0}, {ge::FORMAT_FRACTAL_Z_C04, ConvFormat::FRACTAL_Z_C04},
    {ge::FORMAT_FRACTAL_Z, ConvFormat::FRACTAL_Z}
};

// [fmap, weight, output, bias]
const std::vector<std::vector<ConvDtype>> CONV_SUPPORTED_TYPES_910_95 = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32}
};

const std::map<platform_ascendc::SocVersion, std::vector<std::vector<ConvDtype>>> SOC_CONV_SUPPORTED_TYPES =
{
    {platform_ascendc::SocVersion::ASCEND910_95, CONV_SUPPORTED_TYPES_910_95}
};

// [fmap, weight, output, bias]
const std::vector<std::vector<ConvDtype>> QUANTCONV_SUPPORTED_TYPES_WITH_BIAS = {
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32, ConvDtype::FLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::HIFLOAT8},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::BFLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT8_E4M3FN}
};

// [fmap, weight, output]
const std::vector<std::vector<ConvDtype>> QUANTCONV_SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::BFLOAT16},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN}
};

// [fmap, weight, output, bias]
const std::vector<std::vector<ConvDtype>> QUANTCONV_SUPPORTED_TYPES = {
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::INT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::BFLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::BFLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32}
};

// [fmap, weight, output, bias]
const std::vector<std::vector<ConvDtype>> EXTENDCONV_SUPPORTED_TYPES_NCHW = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::INT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::BFLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::BFLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32}
};

// [fmap, weight, output, bias]
const std::vector<std::vector<ConvDtype>> EXTENDCONV_SUPPORTED_TYPES_NHWC = {
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::INT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32},
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::FLOAT16}
};

const std::map<platform_ascendc::SocVersion,
    std::vector<std::vector<ConvDtype>>> SOC_EXTENDCONV_SUPPORTED_TYPES_NCHW = {
    {platform_ascendc::SocVersion::ASCEND910_95, EXTENDCONV_SUPPORTED_TYPES_NCHW}
};

const std::map<platform_ascendc::SocVersion,
    std::vector<std::vector<ConvDtype>>> SOC_EXTENDCONV_SUPPORTED_TYPES_NHWC = {
    {platform_ascendc::SocVersion::ASCEND910_95, EXTENDCONV_SUPPORTED_TYPES_NHWC}
};

struct ShapeBound {
  std::map<ge::DataType, uint64_t> boundTab;
  ShapeBound(uint64_t b8UpperBd, uint64_t b16UpperBd, uint64_t b32UpperBd) {
    boundTab[ge::DataType::DT_INT8] = b8UpperBd;
    boundTab[ge::DataType::DT_FLOAT16] = b16UpperBd;
    boundTab[ge::DataType::DT_FLOAT] = b32UpperBd;
    boundTab[ge::DataType::DT_BF16] = b16UpperBd;
    boundTab[ge::DataType::DT_HIFLOAT8] = b8UpperBd;
    boundTab[ge::DataType::DT_FLOAT8_E4M3FN] = b8UpperBd;
  }
  uint64_t GetUpperBound(ge::DataType dType) const {
    return boundTab.at(dType);
  }
};
 
const std::map<std::string, ShapeBound> shapeBoundTab = {
    {"N", {MAX_N_B8_SHAPE, MAX_N_B16_SHAPE, MAX_N_B32_SHAPE}},
    {"Ci", {MAX_CIN_B8_SHAPE, MAX_CIN_B16_SHAPE, MAX_CIN_B32_SHAPE}},
    {"H", {MAX_FM_H_B8_SHAPE, MAX_FM_H_B16_SHAPE, MAX_FM_H_B32_SHAPE}},
    {"W", {MAX_FM_W_B8_SHAPE, MAX_FM_W_B16_SHAPE, MAX_FM_W_B32_SHAPE}},
    {"Co", {MAX_COUT_B8_SHAPE, MAX_COUT_B16_SHAPE, MAX_COUT_B32_SHAPE}},
    {"kH", {MAX_KH_B8_SHAPE, MAX_KH_B16_SHAPE, MAX_KH_B32_SHAPE}},
    {"kW", {MAX_KW_B8_SHAPE, MAX_KW_B16_SHAPE, MAX_KW_B32_SHAPE}}
};

const vector<string> PADMODE_WHITELIST = {
    "SPECIFIC",
    "SAME",
    "VALID",
    "SAME_UPPER",
    "SAME_LOWER"
};

struct ascendOpsCubeTypeMap {
    struct {
        ConvDtype madType;
        ConvDtype biasType;
    } typeMaps[static_cast<uint8_t>(ConvDtype::UNDEFINED) + 1] =
        {{ConvDtype::UNDEFINED, ConvDtype::UNDEFINED}};

    ConvDtype ToBiasType(ConvDtype type) const {
        return typeMaps[static_cast<uint8_t>(type)].biasType;
    }
    ConvDtype ToMadType(ConvDtype type) const {
        return typeMaps[static_cast<uint8_t>(type)].madType;
    }
    
    ascendOpsCubeTypeMap() {
        SetMapping(ConvDtype::INT4, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::INT8, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::FLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::BFLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
    }
    
    private:
    void SetMapping(ConvDtype key, ConvDtype madType, ConvDtype biasType) {
        typeMaps[static_cast<uint8_t>(key)].madType = madType;
        typeMaps[static_cast<uint8_t>(key)].biasType = biasType;    // bias dtype in bt
    }
};

const ascendOpsCubeTypeMap CUBE_TYPE_MAP = ascendOpsCubeTypeMap();

struct AscendOpsMknMap {
    int32_t getByIndex(uint32_t idx) const {
        if (idx > MKN_MAX_SIZE - 1) {
            return MKN_VALUE_DEFAULT;
        }
        return map[idx];
    }
    int8_t map[MKN_MAX_SIZE];
};
 
struct ascendOpsCubeMkn {
    int8_t toIdx[static_cast<uint8_t>(ConvDtype::UNDEFINED) + 1] = {0};
    AscendOpsMknMap maps[3] = {{CUBE_UNIT, FP16_CUBE_UNIT, CUBE_UNIT}, // fp16
                        {CUBE_UNIT, FP32_CUBE_UNIT, CUBE_UNIT}, // fp32
                        {CUBE_UNIT, INT8_CUBE_UNIT, CUBE_UNIT}}; // int8 hif8 fp8
    uint32_t GetMKN(ConvDtype dType, uint32_t idx) const {
        return maps[toIdx[static_cast<uint8_t>(dType)]].getByIndex(idx);
    }
    ascendOpsCubeMkn() {
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT32)] = 1;
        toIdx[static_cast<uint8_t>(ConvDtype::BFLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::INT8)] = CONST_VALUE_2;
        toIdx[static_cast<uint8_t>(ConvDtype::HIFLOAT8)] = CONST_VALUE_2;
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT8_E4M3FN)] = CONST_VALUE_2;
    }
};

const ascendOpsCubeMkn CUBE_MKN_MAP = ascendOpsCubeMkn();

struct ConvAscendcDescInfo {
    ge::DataType weightDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType fMapDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType outDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType out1Dtype = ge::DataType::DT_UNDEFINED;
    ge::DataType quantScaleDtype = ge::DataType::DT_INT64;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format fMapFormat = ge::FORMAT_NCHW;
    ge::Format biasFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_NCHW;
    ge::Format out1Format = ge::FORMAT_NCHW;
    ge::Format quantScaleFormat = ge::FORMAT_ND;
};

struct ConvAscendcTilingFlag {
    bool hasBias = false;
    bool quantFlag = false; // hif8, fp8, int8 with scale is true; hif8 without scale is false
    bool extendConvFlag = false;
    bool enableC04Flag = false;
    bool mSplitModeFlag = false;
    ConvGroupType convGroupType = ConvGroupType::NORMAL_CONV;
    bool mBasicBlockFlag = false;
    bool useTilingRepo = false;
    bool useTilingCache = false;
};

enum class ConvAscendcFeatureFlag : uint8_t {
    IS_LOAD3D_FLAG = 0,
    IS_CONV1D_FLAG,
    IS_DMA_FLAG,
    INVALID
};

constexpr const char* FeatureFlagEnumToString(const ConvAscendcFeatureFlag convFeatureFlag) {
    constexpr std::array<const char*, static_cast<std::size_t>(ConvAscendcFeatureFlag::INVALID) + 1>
        convFeatureFlagStrings = {"IS_LOAD3D_FLAG", "IS_CONV1D_FLAG", "IS_DMA_FLAG", "INVALID"
    };
    auto index = static_cast<std::size_t>(convFeatureFlag);
    return (index < convFeatureFlagStrings.size()) ? convFeatureFlagStrings[index] : "INVALID";
}

struct BlockDimRange {
  std::vector<uint32_t> aicNumRange;
  std::vector<uint32_t> batchRange;
  std::vector<uint32_t> mRange;
  std::vector<uint32_t> nRange;
  std::vector<uint32_t> doRange;
  std::vector<uint32_t> hoRange;
  std::vector<uint32_t> hoSpareRange;
  std::vector<uint32_t> woRange;
  std::vector<uint32_t> groupRange;
};

struct BlockDimRes {
  uint32_t batchDim = 0;
  uint32_t mDim = 0;
  uint32_t nDim = 0;
  uint32_t doDim = 0;
  uint32_t hoDim = 0;
  uint32_t woDim = 0;
  uint32_t groupDim = 0;
  uint64_t minCost = 0;
};

struct ConvTilingKeyPara {
    uint64_t fmpTiling = 0;
    uint64_t weightTiling = 0;
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;
    uint64_t outputOrder = 0;
    uint64_t iterOrder = 0;
    uint64_t groupType = 0;
    uint64_t enableSmallChannel = 0;
    uint64_t weightUbTrans = 0;
    uint64_t fmapCppyMode = 0;
    uint64_t innerBatch = 0;
};

static std::map<ge::DataType, uint32_t> dtypeSizeTab = {
    {ge::DataType::DT_FLOAT16, 2},
    {ge::DataType::DT_FLOAT, 4},
    {ge::DataType::DT_BF16, 2},
    {ge::DataType::DT_INT8, 1},
    {ge::DataType::DT_INT64, 8},
    {ge::DataType::DT_UINT64, 8},
    {ge::DataType::DT_INT32, 4},
    {ge::DataType::DT_HIFLOAT8, 1},
    {ge::DataType::DT_FLOAT8_E4M3FN, 1},
    {ge::DataType::DT_INT16, 2},
    {ge::DataType::DT_UINT8, 1},
    {ge::DataType::DT_UINT16, 2},
    {ge::DataType::DT_UINT32, 4},
    {ge::DataType::DT_UNDEFINED, 0}
};

struct ConvOpsConstParams {
  uint64_t m0 = 0;
  uint64_t n0 = 0;
  uint64_t k0 = 0;
  uint64_t ci1 = 0;
  uint64_t co1 = 0;
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

uint32_t Gcd(uint32_t i, uint32_t j);

uint64_t ConvCeilDiv(uint64_t a, uint64_t b);
void ConvCalcCommFactor(const uint64_t num, const uint32_t numMax, std::vector<uint32_t> &reslist);
uint64_t ConvAlignB(uint64_t a, uint64_t b);
uint64_t ConvInferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH);
uint64_t ConvInferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW);
int64_t ConvComputeHo(int64_t hi, int64_t hk, int64_t padTop, int64_t padBottom, int64_t dilationH, int64_t strideH);
int64_t ConvComputeWo(int64_t wi, int64_t wk, int64_t padLeft, int64_t padRight, int64_t dilationW, int64_t strideW);
void ConvBlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange, const std::vector<uint32_t> &mixRange);
ge::graphStatus ShapeAttrSynthesisCheck(ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo, gert::TilingContext* context);
ge::graphStatus ShapeAttrSynthesisCheckAux(ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo,
                                           gert::TilingContext* context);
void GetSupportedDataTypes(bool hasBias, bool quantFlag, std::vector<std::vector<ConvDtype>>& supportTypes);
void GetSupportedDataTypes(const platform_ascendc::SocVersion& socVersion, bool quantFlag,
                           ge::Format fMapFormat, bool exendConvFlag,
                           std::vector<std::vector<ConvDtype>>& supportTypes);
bool GetConvParamsIdx(const std::vector<ge::Format> formatVec, std::vector<std::vector<std::size_t>>& idxVec);
void InitblockDimConstParas(ConvOpsConstParams& convOpsConstParams,
                            const ConvAscendcDescInfo& descInfo,
                            const ConvAscendcShapesInfo& shapeInfo);
bool IsWeightNZFormat(ge::Format weightFormat);

template <typename T>
bool ConvArrMatch(T& arr1, const T& arr2, size_t size)
{
    if (arr1.size() != size || arr2.size() != size) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool ConvArrMatchWithSize(T& arr1, const T& arr2, size_t size)
{
    if (arr1.size() < size || arr2.size() < size) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

class ConvBase {
public:
    ConvBase() {};
    explicit ConvBase(gert::TilingContext* context) : context_(context) {};
    void ConvBaseInit(ConvAscendcShapesInfo shapeInfo, ConvAscendcDescInfo descInfo, ConvAscendcTilingFlag flagInfo,
        gert::TilingContext* context);
    void ConvBaseInitAttrInfo(ConvAscendcAttrInfo attrInfo);
    void ConvBaseInitOpInfo(const ConvTilingParseInfo* opInfo);
    void ConvBaseInitFeatureFlag(const ConvAscendcFeatureFlag featureFlagInfo);
    void GetConvBaseCoreInfo(ConvOpsConstParams& convOpsConstParams);
    void InitblockDimConstParas();
    void InitGroupInfo(conv_tiling::ConvOriGroupInfo convOriGroupInfo,
                       conv_tiling::ConvOptGroupInfo convOptGroupInfo);
    BlockDimRes BlockDimDecisionMsplitMode();
    void GetBlockDimRangeCommon();
    void GetBlockDimRangeMsplitMode();
    void GetBlockDimInitMsplitMode();
    void CoreBlockDimDecisionMsplitMode();
    void BlockDimDecisionBackTrackMsplitMode(const vector<vector<uint32_t>> &inputRanges,
                                             uint32_t rangeIdx, vector<uint32_t> &record);
    uint64_t CalcTotalCostMsplitMode(uint32_t batchDim, uint32_t mDim,
                                     uint32_t nDim, uint32_t doDim, uint32_t groupDim);
    BlockDimRes BlockDimDecisionHWsplitMode();
    bool CmpCoreUtilize(const uint32_t curCoreUtilize, const uint32_t minCostCoreUtilize,
                        const uint32_t batchDim, const uint32_t doDim);
    bool CmpCoreUtilizeMsplitMode(uint32_t batchDim, uint32_t hoDim, uint32_t nDim, uint32_t doDim, uint32_t groupDim);
    bool CmpCoreUtilizeHWsplitMode(const vector<uint32_t> &record);
    uint64_t CalcCostHWsplitMode(const BlockDimRes &blockDimRes, const uint64_t ci1, const uint64_t ci0,
                                 const uint64_t co1);
    uint64_t CalcTotalCostHWsplitMode(const BlockDimRes &blockDimRes);
    void SeperateHoRangeHWsplitMode();
    void GetBlockDimRangeHWsplitMode();
    void GetBlockDimInitHWsplitMode();
    void SetBlockDimHWsplitMode(const vector<uint32_t> &record, const uint64_t curCost,
                                BlockDimRes &blockDimRes) const;
    void SetBlockDimMsplitMode(const vector<uint32_t> &record, uint64_t curCost);
    void BlockDimDecisionBackTrackHWsplitMode(const std::vector<vector<uint32_t>> &inputRanges,
                                              uint32_t rangeIdx, std::vector<uint32_t> &record);
    void CoreBlockDimDecisionHWsplitMode();
    void CheckCoreUsedupHWsplitMode();
    void SetAiCoreNum(uint32_t aicoreNum);
    void InitBlockDimConstParas();
    void SetMKN(uint32_t m0, uint32_t k0, uint32_t n0);

    void SetParams(uint64_t l2Rate);
    uint64_t GetMinBurstNum();
    uint64_t CalcMinUsedL1SizeInMsplitMode(uint64_t kAL1min, uint64_t kBL1min);
    uint64_t CalcMinUsedL1SizeInHWsplitMode(uint64_t kAL1min, uint64_t kBL1min, uint64_t wiAL1min);
    ge::graphStatus CheckL1SizeLimitsInMSplitMode();
    ge::graphStatus CheckL1SizeLimitsInHWsplitMode();
    ge::graphStatus CheckC04L1SizeLimitsInMsplitMode();
    ge::graphStatus CheckC04L1SizeLimitsInHWSplitMode();
    bool IsFp32InputFp32Output();
    void SetBitsFromBool(uint64_t& number, const std::array<bool, UINT64_BIT_COUNT>& bits) const;
    void SetBytesFromUint8(uint64_t& number, const std::array<uint8_t, UINT64_BYTE_COUNT>& bytes) const;
    void SetBytesFromUint32(uint64_t& number, uint32_t highPart, uint32_t lowPart) const;
    bool GetConvParasHf32Mode(const uint32_t enableHf32Idx, uint32_t& hf32Mode);
    uint32_t GetWeightBandWidthCoeff();
    void GetSupportedFormats(bool quantFlag, bool is2dFlag, const platform_ascendc::SocVersion socVersion,
                             std::stringstream& ss, std::vector<std::vector<ge::Format>>& supportFormats);
    void ConvBaseInitFixpipeInfo(const conv_tiling::FixpipeInfo& fixpipeInfo);
    bool CheckValidString(const string &inputStr, const gert::TilingContext* context) const;
private:
    ConvAscendcShapesInfo shapeInfo_;
    ConvAscendcAttrInfo attrInfo_;
    ConvAscendcDescInfo descInfo_;
    ConvAscendcTilingFlag flagInfo_;
    ConvAscendcFeatureFlag featureFlagInfo_;
    conv_tiling::ConvOriGroupInfo oriGroupInfo_;
    conv_tiling::ConvOptGroupInfo optGroupInfo_;
    conv_tiling::FixpipeInfo fixpipeInfo_;

    gert::TilingContext* context_ = nullptr;
    uint64_t l2Rate_ = 1;

    uint32_t aicoreNum_ = 0;
    uint32_t m0_ = 1;
    uint32_t k0_ = 1;
    uint32_t n0_ = 1;
    BlockDimRes blockDimRes_;
    BlockDimRange blockDimRanges_;
    std::vector<uint32_t> blockDimInit_;
    ConvOpsConstParams convOpsConstParams_;
    const ConvTilingParseInfo* opInfo_ = nullptr;
};
}
}
#endif