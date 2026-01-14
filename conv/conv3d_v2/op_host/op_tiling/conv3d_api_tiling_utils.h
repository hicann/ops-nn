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
 * \file conv3d_tiling_util.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV3D_TILING_UTIL_H
#define ASCENDC_TILING_CONV3D_TILING_UTIL_H

#include <iostream>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"

namespace Conv3dApiTiling {

#ifdef ASC_OP_DEBUG_TEST
#define TILING_CUBE_LOG(level, format, ...)                               \
    do {                                                                \
        fprintf(stdout, "[LOG] %s " format "\n", level, ##__VA_ARGS__); \
    } while (0)
#else
#define TILING_CUBE_LOG(level, format, ...)
#endif

#define TILING_DEBUG_LOG(format, ...) TILING_CUBE_LOG("[DEBUG]", format, ##__VA_ARGS__)
#define TILING_INFO_LOG(format, ...) TILING_CUBE_LOG("[INFO] ", format, ##__VA_ARGS__)
#define TILING_WARNING_LOG(format, ...) TILING_CUBE_LOG("[WARN] ", format, ##__VA_ARGS__)
#define TILING_ERROR_LOG(format, ...) TILING_CUBE_LOG("[ERROR]", format, ##__VA_ARGS__)

enum class ConvDtype : std::uint8_t {
    FLOAT16 = 0,
    FLOAT32,
    BF16,
    INT4,
    INT8,
    UINT8,
    INT32,
    INT64,
    UINT64,
    CONVDTYPEMAX
};

enum class ConvFormat : std::uint8_t {
    ND = 0,
    NCHW,
    NHWC,
    HWCN,
    DHWNC,
    DHWCN,
    NDHWC,
    NCDHW,
    NC1HWC0,
    NDC1HWC0,
    FRACTAL_Z_3D
};

enum class TPosition : std::uint8_t {
    GM = 0,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    MAX
};

enum class IterateMNOrder : std::uint8_t {
    ITER_M_FST = 0,
    ITER_N_FST,
    INVALID
};

enum class PadIndex : std::uint8_t {
    PAD_HEAD = 0,
    PAD_TAIL,
    PAD_TOP,
    PAD_BOTTOM,
    PAD_LEFT,
    PAD_RIGHT
};

const std::map<ConvDtype, std::string> g_dtypeToStr = {
    {ConvDtype::FLOAT16, "float16"},
    {ConvDtype::FLOAT32, "float32"},
    {ConvDtype::BF16, "bfloat16"},
    {ConvDtype::INT4, "int4"},
    {ConvDtype::INT8, "int8"},
    {ConvDtype::UINT8, "uint8"},
    {ConvDtype::INT64, "int64"},
    {ConvDtype::UINT64, "uint64"},
    {ConvDtype::INT32, "int32"},
};

const std::map<ConvFormat, std::string> g_formatToStr = {
    {ConvFormat::NCHW, "NCHW"},
    {ConvFormat::NHWC, "NHWC"},
    {ConvFormat::HWCN, "HWCN"},
    {ConvFormat::DHWNC, "DHWNC"},
    {ConvFormat::DHWCN, "DHWCN"},
    {ConvFormat::NDHWC, "NDHWC"},
    {ConvFormat::NCDHW, "NCDHW"},
    {ConvFormat::NC1HWC0, "NC1HWC0"},
    {ConvFormat::ND, "ND"}
};

const std::map<ConvDtype, uint32_t> g_dtypeSizeTab = {
    {ConvDtype::FLOAT16, 2},
    {ConvDtype::FLOAT32, 4},
    {ConvDtype::BF16, 2},
    {ConvDtype::INT8, 1},
    {ConvDtype::UINT8, 1},
    {ConvDtype::INT64, 8},
    {ConvDtype::UINT64, 8},
    {ConvDtype::INT32, 4}
};

constexpr uint32_t COUNT_PARAMS_BIAS_SCALE = 5; // [fmap, weight, bias, scale, output]
constexpr uint32_t COUNT_PARAMS_BIAS = 4; // [fmap, weight, bias, output]
constexpr uint32_t COUNT_PARAMS_NO_BIAS = 3; // [fmap, weight, output]

const std::vector<std::vector<ConvDtype>> SUPPORTED_TYPES_WITH_BIAS_SCALE = {
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::BF16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::BF16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT16}
};

const std::vector<std::vector<ConvDtype>> SUPPORTED_TYPES_WITH_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32, ConvDtype::FLOAT16},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::BF16}
};
const std::vector<std::vector<ConvDtype>> SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16}
};

const std::vector<std::vector<ConvDtype>> POINTWISE_SUPPORTED_TYPES_WITH_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::BF16}
};
const std::vector<std::vector<ConvDtype>> POINTWISE_SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16}
};

constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr int8_t M_Mode = 0;
constexpr int8_t HW_Mode = 1;

constexpr uint32_t C0_BYTE_SIZE = 32;
constexpr uint32_t MIN_BURST_SIZE = 128;
constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_FILTER_H_W = 511;
constexpr uint32_t LOAD3D_MAX_DDR2L1_SIZE = 65535;

constexpr uint32_t CUBE_UNIT = 16;
constexpr uint32_t FP16_CUBE_UNIT = 16;
constexpr uint32_t FP32_CUBE_UNIT = 8;
constexpr uint32_t INT8_CUBE_UNIT = 32;
constexpr uint32_t MKN_MAX_SIZE = 3;
constexpr uint32_t MKN_M_INDEX = 0;
constexpr uint32_t MKN_K_INDEX = 1;
constexpr uint32_t MKN_N_INDEX = 2;
constexpr uint32_t MKN_VALUE_DEFAULT = 16;

constexpr uint64_t MAX_64_BIT_NUM = 0xFFFFFFFFFFFFFFFFU;

constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t CONST_HO_1 = 1;
constexpr uint32_t C0_SIZE = 32;

constexpr int64_t INVALID_VALUE = -1L;
constexpr int64_t VALID_VALUE = 0;

constexpr uint64_t ENABLE_DOUBLE_BUFFER = 1;
constexpr uint64_t DISABLE_DOUBLE_BUFFER = 0;

constexpr uint64_t ENABLE_HF32 = 1;
constexpr uint64_t DISABLE_HF32 = 0;

constexpr int64_t INITIAL_TILING = 0;
constexpr int8_t INITIAL_OFFSET = 0;
constexpr uint8_t INITIAL_PING_PONG_AL1 = 1;
constexpr uint8_t INITIAL_PING_PONG_BL1 = 1;
constexpr uint8_t INITIAL_PING_PONG_AL0 = 2;
constexpr uint8_t INITIAL_PING_PONG_BL0 = 2;
constexpr uint8_t INITIAL_PING_PONG_CL0 = 1;
constexpr uint8_t INITIAL_PING_PONG_UB = 1;
constexpr uint32_t INITIAL_INDEX = 0;
constexpr uint64_t INITIAL_SIZE = 0;
constexpr uint64_t INITIAL_DIM = 0;
constexpr uint64_t INITIAL_WEIGHT_DIM = 1;

constexpr uint8_t PING_PONG_AL1 = 2;

constexpr uint8_t DISABLE_QUANT = 0;
constexpr uint32_t QUANT_VEC_MAX_LOAD = 8192;

// load3dv2 postk limit
constexpr uint32_t POSTK_LIMIT = 65535;

struct ConvType {
    ConvFormat format;
    ConvDtype dtype;
    TPosition pos;
};

struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = INITIAL_SIZE;
    uint64_t l0CSize = INITIAL_SIZE;
    uint64_t ubSize = INITIAL_SIZE;
    uint64_t l0ASize = INITIAL_SIZE;
    uint64_t l0BSize = INITIAL_SIZE;
    uint64_t btSize = INITIAL_SIZE;
};

struct Conv3DOriGroupInfo {
    int64_t groups = INVALID_VALUE;
    int64_t cin = INVALID_VALUE;
    int64_t cout = INVALID_VALUE;
    ConvDtype dtype = ConvDtype::CONVDTYPEMAX;
};

struct Conv3DGroupOptInfo {
    int64_t groupOpt = INVALID_VALUE;
    int64_t cinOpt = INVALID_VALUE;
    int64_t coutOpt = INVALID_VALUE;
};

struct AscendApiCubeTypeMap {
public:
    struct {
        ConvDtype madType;
        ConvDtype biasType;
    } typeMaps[static_cast<uint8_t>(ConvDtype::CONVDTYPEMAX) + 1] = {
        {ConvDtype::CONVDTYPEMAX, ConvDtype::CONVDTYPEMAX}
    };

    ConvDtype ToBiasType(ConvDtype type) const
    {
        return typeMaps[static_cast<uint8_t>(type)].biasType;
    }
    ConvDtype ToMadType(ConvDtype type) const
    {
        return typeMaps[static_cast<uint8_t>(type)].madType;
    }

    AscendApiCubeTypeMap()
    {
        SetMapping(ConvDtype::INT4, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::INT8, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::UINT8, ConvDtype::INT32, ConvDtype::INT32);
        SetMapping(ConvDtype::FLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
    }

private:
    void SetMapping(ConvDtype key, ConvDtype madType, ConvDtype biasType)
    {
        typeMaps[static_cast<uint8_t>(key)].madType = madType;
        typeMaps[static_cast<uint8_t>(key)].biasType = biasType;
    }
};

struct AscendApiMknMap {
    uint32_t GetByIndex(uint32_t idx) const
    {
        if (idx > MKN_MAX_SIZE - 1) {
        return MKN_VALUE_DEFAULT;
        }
        return map[idx];
    }
    uint8_t map[MKN_MAX_SIZE];
};

struct AscendApiCubeMkn {
    int8_t toIdx[static_cast<uint8_t>(ConvDtype::CONVDTYPEMAX) + 1] = {0};
    AscendApiMknMap maps[3] = {{CUBE_UNIT, FP16_CUBE_UNIT, CUBE_UNIT}, // fp16
                               {CUBE_UNIT, FP32_CUBE_UNIT, CUBE_UNIT}, // fp32
                               {CUBE_UNIT, INT8_CUBE_UNIT, CUBE_UNIT}}; // int8
    uint32_t GetMKN(ConvDtype dType, uint32_t idx) const
    {
        return maps[toIdx[static_cast<uint8_t>(dType)]].GetByIndex(idx);
    }
    AscendApiCubeMkn()
    {
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT16)] = static_cast<int8_t>(0);
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT32)] = static_cast<int8_t>(1);
        toIdx[static_cast<uint8_t>(ConvDtype::BF16)] = static_cast<int8_t>(0);
        toIdx[static_cast<uint8_t>(ConvDtype::INT8)] = static_cast<int8_t>(CONST_VALUE_2);
    }
};

const AscendApiCubeTypeMap CUBE_TYPE_TAB = AscendApiCubeTypeMap();
const AscendApiCubeMkn CUBE_MKN_TAB = AscendApiCubeMkn();

int64_t LCM(int64_t numL, int64_t numR);
uint64_t CeilDiv(uint64_t a, uint64_t b);
uint64_t AlignB(uint64_t a, uint64_t b);
uint64_t Gcd(uint64_t a, uint64_t b);
void FindDivisorsUpTo(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList);
void CalcCommFactorWithPowerOfTwo(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList);
void CalcCommFactor(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList);
void CalcFactorPointWise(uint64_t numMax, std::vector<uint64_t> &resList);
void VectorElementMultip(std::vector<uint64_t> &range, const uint64_t value);
bool IsArrayEqual(const std::vector<ConvDtype>& arr1, const std::vector<ConvDtype>& arr2, uint32_t size);

template <typename T>
bool AddWithOverflowCheck(T &res, T a, T b)
{
    T tmpRes = a + b;
    if (tmpRes < a || tmpRes < b) {
        return true;
    }
    res = tmpRes;
    return false;
}

} // namespace Conv3dApiTiling

#endif // ASCENDC_TILING_CONV3D_TILING_UTIL_H