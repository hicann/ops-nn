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
#include "conv_base_blockdim_decision.h"
#include "conv_template_utils.h"
namespace optiling {
namespace conv_ops_tiling {

static std::map<NpuArch, std::string> socNameTab = {
    {NpuArch::DAV_3510, "Ascend950"},
    {NpuArch::DAV_5102, "MC62CM12A"}
};

static std::map<string, NpuArch> socConvertMap = {
    {"DAV_3510", NpuArch::DAV_3510},
    {"Ascend950", NpuArch::DAV_3510},
    {"Ascend910_95", NpuArch::DAV_3510},
    {"MC62CM12AA", NpuArch::DAV_5102},
    {"DAV_5102", NpuArch::DAV_5102}
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
const std::vector<std::vector<ConvDtype>> CONV_SUPPORTED_TYPES_DAV = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16},
    {ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::HIFLOAT8, ConvDtype::FLOAT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::FLOAT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::BFLOAT16, ConvDtype::FLOAT32}
};

// [fmap, weight, output, bias]
const std::vector<std::vector<ConvDtype>> CONV_SUPPORTED_TYPES_MDC = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16}
};

const std::map<NpuArch, std::vector<std::vector<ConvDtype>>> SOC_CONV_SUPPORTED_TYPES = {
    {NpuArch::DAV_3510, CONV_SUPPORTED_TYPES_DAV},
    {NpuArch::DAV_5102, CONV_SUPPORTED_TYPES_MDC}
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
const std::vector<std::vector<ConvDtype>> EXTENDCONV2D_SUPPORTED_TYPES_MDC = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::INT32},
    {ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::INT32},
    {ConvDtype::FLOAT16, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32}
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

const std::map<NpuArch,
    std::vector<std::vector<ConvDtype>>> SOC_EXTENDCONV_SUPPORTED_TYPES_NCHW = {
    {NpuArch::DAV_5102, EXTENDCONV2D_SUPPORTED_TYPES_MDC},
    {NpuArch::DAV_3510, EXTENDCONV_SUPPORTED_TYPES_NCHW}
};

const std::map<NpuArch,
    std::vector<std::vector<ConvDtype>>> SOC_EXTENDCONV_SUPPORTED_TYPES_NHWC = {
    {NpuArch::DAV_5102, EXTENDCONV2D_SUPPORTED_TYPES_MDC},
    {NpuArch::DAV_3510, EXTENDCONV_SUPPORTED_TYPES_NHWC}
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

constexpr const char* FeatureFlagEnumToString(const ConvAscendcFeatureFlag convFeatureFlag) {
    constexpr std::array<const char*, static_cast<std::size_t>(ConvAscendcFeatureFlag::INVALID) + 1>
        convFeatureFlagStrings = {"IS_LOAD3D_FLAG", "IS_CONV1D_FLAG", "IS_DMA_FLAG", "INVALID"
    };
    auto index = static_cast<std::size_t>(convFeatureFlag);
    return (index < convFeatureFlagStrings.size()) ? convFeatureFlagStrings[index] : "INVALID";
}

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

ge::graphStatus ShapeAttrSynthesisCheck(ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo, gert::TilingContext* context);
ge::graphStatus ShapeAttrSynthesisCheckAux(ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo,
                                           gert::TilingContext* context);
void GetSupportedDataTypes(bool hasBias, bool quantFlag, std::vector<std::vector<ConvDtype>>& supportTypes);
void GetSupportedDataTypes(const NpuArch& socVersion, bool quantFlag,
                           ge::Format fMapFormat, bool exendConvFlag,
                           std::vector<std::vector<ConvDtype>>& supportTypes);
bool GetConvParamsIdx(const std::vector<ge::Format> formatVec, std::vector<std::vector<std::size_t>>& idxVec);
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

class ConvBase : public ConvBaseDeci {
public:
    ConvBase() {};
    explicit ConvBase(gert::TilingContext* context) : context_(context) {};
    void ConvBaseInit(ConvAscendcShapesInfo shapeInfo, ConvAscendcDescInfo descInfo, ConvAscendcTilingFlag flagInfo,
                      gert::TilingContext* context);
    void ConvBaseInitOpInfo(const ConvTilingParseInfo* opInfo);
    void ConvBaseInitFeatureFlag(const ConvAscendcFeatureFlag featureFlagInfo);
    void InitGroupInfo(ConvOriGroupInfo convOriGroupInfo,
                       ConvOptGroupInfo convOptGroupInfo);
    void InitBlockDimConstParas();
    void UpdateFlagInfo(const ConvAscendcTilingFlag& flagInfo);
    void updatePlatformInfoFromOpInfo();

    void SetParams(uint64_t l2Rate);
    ge::graphStatus CheckC04L1SizeLimitsInMsplitMode();
    ge::graphStatus CheckC04L1SizeLimitsInHWSplitMode();
    bool IsFp32InputFp32Output();
    void SetBitsFromBool(uint64_t& number, const std::array<bool, UINT64_BIT_COUNT>& bits) const;
    void SetBytesFromUint8(uint64_t& number, const std::array<uint8_t, UINT64_BYTE_COUNT>& bytes) const;
    void SetBytesFromUint32(uint64_t& number, uint32_t highPart, uint32_t lowPart) const;
    bool GetConvParasHf32Mode(const uint32_t enableHf32Idx, uint32_t& hf32Mode);
    uint32_t GetWeightBandWidthCoeff();
    void GetSupportedFormats(bool quantFlag, bool is2dFlag,
                             std::stringstream& ss, std::vector<std::vector<ge::Format>>& supportFormats);
    void ConvBaseInitFixpipeInfo(const FixpipeInfo& fixpipeInfo);
    bool CheckValidString(const string &inputStr, const gert::TilingContext* context) const;

private:
    gert::TilingContext* context_ = nullptr;
    const ConvTilingParseInfo* opInfo_ = nullptr;
};
}
}
#endif