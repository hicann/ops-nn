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
 * \file conv_api_tiling_util.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV_API_TILING_UTIL_H
#define ASCENDC_TILING_CONV_API_TILING_UTIL_H

#include <iostream>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <tiling/platform/platform_ascendc.h>

using namespace std;

namespace conv_tiling {
#ifdef ENABLE_CONV_TILING_DEBUG
#define LOG(level, format, ...)                                         \
    do {                                                                \
        fprintf(stdout, "[LOG] %s " format "\n", level, ##__VA_ARGS__); \
    } while (0)
#else
#define LOG(level, format, ...)
#endif

#define TILING_LOG_DEBUG(format, ...) LOG("[DEBUG]", format, ##__VA_ARGS__)
#define TILING_LOG_INFO(format, ...) LOG("[INFO] ", format, ##__VA_ARGS__)
#define TILING_LOG_WARNING(format, ...) LOG("[WARN] ", format, ##__VA_ARGS__)
#define TILING_LOG_ERROR(format, ...) LOG("[ERROR]", format, ##__VA_ARGS__)

enum class ConvDtype {
    FLOAT16 = 0,
    FLOAT32,
    BFLOAT16,
    INT4,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT64,
    UINT8,
    UINT16,
    UINT32,
    HIFLOAT8,
    FLOAT8_E4M3FN,
    DOUBLE,
    UNDEFINED
};

enum class ConvFormat {
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
    FRACTAL_Z_C04,
    FRACTAL_Z_3D,
    FRACTAL_Z,
    UNDEFINED
};

enum class TPosition {
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

struct ConvType {
    ConvFormat format = ConvFormat::UNDEFINED;
    ConvDtype dtype = ConvDtype::UNDEFINED;
    TPosition pos = TPosition::MAX;
};

struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint32_t abL1mte2BandWidthCof = 0;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t btSize = 0;
    uint64_t fbSize = 0;
};

enum class BoundType {
    CUBE_BOUND = 0,
    MEMORY_BOUND,
    INVALID
};

enum class IterateMNOrder {
    ITER_M_FST = 0,
    ITER_N_FST,
    INVALID
};

enum class OutputOrder {
    HW = 0,
    M,
    INVALID
};

enum class ConvGroupType {
    NORMAL_CONV = 0,
    ORI_GROUP_CONV,
    OPT_GROUP_CONV
};

struct ConvOriGroupInfo {
    uint64_t ciPerGroup = 0;
    uint64_t coPerGroup = 0;
    uint64_t groups = 0;
    ConvDtype weightDtype = ConvDtype::UNDEFINED;
};

struct ConvOptGroupInfo {
    uint64_t enlarge = 0;
    uint64_t groupOpt = 0;
    uint64_t cinOpt = 0;
    uint64_t coutOpt = 0;
};

struct ConvC04Info {
    uint64_t curNBL1 = 0;
    uint64_t orgkH = 0;
    uint64_t orgkW = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint8_t pbBL1 = 0;
    ConvDtype weightDtype = ConvDtype::UNDEFINED;
};

struct ConvWeightUbTransParams {
    uint64_t curNBL1 = 0;
    uint64_t curKBL1 = 0;
    uint64_t orgkH = 0;
    uint64_t orgkW = 0;
    uint64_t k0 = 0;
    uint64_t n0 = 0;
    ConvDtype weightDtype = ConvDtype::UNDEFINED;
 
    uint32_t bUbNStep = 0;
    uint32_t bUbKStep = 0;
};

struct ConvDmaParams {
    uint64_t curHoL1 = 0;
    uint64_t curWoL1 = 0;
    uint64_t khL1 = 0;
    uint64_t kwL1 = 0;
    uint64_t k0 = 0;
    ConvDtype fmapDtype = ConvDtype::UNDEFINED;

    uint32_t khUb = 0;
    uint32_t kwUb = 0;
};

struct FixpipeInfo {
    uint8_t quantMode0 = 0;
    uint8_t reluMode0 = 0;
    uint8_t clipMode0 = 0;
    uint8_t quantMode1 = 0;
    uint8_t reluMode1 = 0;
    uint8_t clipMode1 = 0;
    uint8_t dualOutput = 0;
    float channelWiseCoeff = 0;
// channelWiseCoeff represents the sum of the multiples of the byte width of the data type of the channelwise
// parameter in fixpipe compared to the byte width of FP16 (two bytes).
};

enum class TilingAlgorithmType {
    FORMULAS = 0,
    BASIC_BLOCK,
    INVALID
};

const std::map<ConvDtype, std::string> DTYPE_TO_STR = {
    {ConvDtype::FLOAT16, "float16"},
    {ConvDtype::FLOAT32, "float32"},
    {ConvDtype::BFLOAT16, "bfloat16"},
    {ConvDtype::INT4, "int4"},
    {ConvDtype::INT8, "int8"},
    {ConvDtype::UINT8, "uint8"},
    {ConvDtype::INT16, "int16"},
    {ConvDtype::UINT16, "uint16"},
    {ConvDtype::INT32, "int32"},
    {ConvDtype::UINT32, "uint32"},
    {ConvDtype::INT64, "int64"},
    {ConvDtype::UINT64, "uint64"},
    {ConvDtype::HIFLOAT8, "hifloat8"},
    {ConvDtype::FLOAT8_E4M3FN, "float8_e4m3fn"},
    {ConvDtype::DOUBLE, "double"},
    {ConvDtype::UNDEFINED, "undefined"},
};

const std::map<ConvFormat, std::string> FORMAT_TO_STR = {
    {ConvFormat::NCHW, "NCHW"},
    {ConvFormat::NHWC, "NHWC"},
    {ConvFormat::HWCN, "HWCN"},
    {ConvFormat::DHWNC, "DHWNC"},
    {ConvFormat::DHWCN, "DHWCN"},
    {ConvFormat::NDHWC, "NDHWC"},
    {ConvFormat::NCDHW, "NCDHW"},
    {ConvFormat::NC1HWC0, "NC1HWC0"},
    {ConvFormat::NDC1HWC0, "NDC1HWC0"},
    {ConvFormat::ND, "ND"},
    {ConvFormat::FRACTAL_Z_C04, "FRACTAL_Z_C04"},
    {ConvFormat::FRACTAL_Z_3D, "FRACTAL_Z_3D"},
    {ConvFormat::UNDEFINED, "UNDEFINED"},
};

const std::map<ConvDtype, uint32_t> DTYPE_SIZE_TAB = {
    {ConvDtype::FLOAT16, 2},
    {ConvDtype::FLOAT32, 4},
    {ConvDtype::BFLOAT16, 2},
    {ConvDtype::INT4, 1},
    {ConvDtype::INT8, 1},
    {ConvDtype::INT32, 4},
    {ConvDtype::INT64, 8},
    {ConvDtype::UINT64, 8},
    {ConvDtype::HIFLOAT8, 1},
    {ConvDtype::FLOAT8_E4M3FN, 1}
};

constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t C0_BYTE_SIZE = 32;
constexpr uint32_t MIN_BURST_SIZE = 128;
constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_FILTER_H_W = 511;
constexpr uint32_t LOAD3D_MAX_DDR2L1_SIZE = 65535;
constexpr uint32_t LOAD3D_M_START_POS_LIMIT = 32767;
constexpr uint32_t DATACOPYPARAMS_BURSTLEN_MAX = 65535;

constexpr uint32_t M0 = 16;
constexpr uint32_t N0 = 16;
constexpr uint32_t B16_K0 = 16;
constexpr uint32_t B32_K0 = 8;
constexpr uint32_t B8_K0 = 32;
constexpr uint32_t MKN_MAX_SIZE = 3;
constexpr uint32_t MKN_M_INDEX = 0;
constexpr uint32_t MKN_K_INDEX = 1;
constexpr uint32_t MKN_N_INDEX = 2;
constexpr uint32_t MKN_VALUE_DEFAULT = 16;

constexpr uint64_t MAX_40_BIT_NUM = 1099511627775;
constexpr uint64_t MAX_32_BIT_NUM = 4294967295;
constexpr uint64_t MAX_16_BIT_NUM = 65535;
constexpr uint64_t MAX_31_BIT_NUM = 2147483647;

constexpr uint64_t CONST_VALUE_1 = 1;
constexpr uint64_t C04_CIN_SIZE = 4;
constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t C04_VEC_USE_BUFF_NUM = 2;
constexpr uint64_t WEIGHT_UB_BUFF_NUMS = 2;
constexpr uint32_t VEC_NUM_PER_CUBE_910D = 2;
constexpr uint32_t VGATHER_REGISTER_SIZE = 256;
constexpr uint32_t VGATHER_PERF_LIMIT = 8;
constexpr uint32_t BAND_WIDTH_COEFF = 4;
constexpr uint32_t C0_SIZE = 32;

constexpr uint32_t POSTK_LIMIT = 65535;

constexpr int8_t ROUND_MODE_RINT = 1;
constexpr int8_t ROUND_MODE_ROUND = 2;
constexpr int8_t ROUND_MODE_HYBRID = 3;

constexpr uint32_t FORMAT_FM_INDEX = 0;
constexpr uint32_t FORMAT_WEIGHT_INDEX = 1;
constexpr uint32_t FORMAT_BIAS_INDEX = 2;

constexpr uint32_t MIN_M_L1_SIZE = 16;

constexpr uint32_t L1_SCORE_BASE_5_POINTS = 5;
constexpr uint32_t L1_SCORE_BASE_4_POINTS = 4;
constexpr uint32_t L1_SCORE_BASE_3_POINTS = 3;
constexpr uint32_t L1_SCORE_BASE_0_POINTS = 0;

constexpr uint32_t KERNEL_1X1 = 1;
constexpr uint32_t KERNEL_3X3 = 9;

constexpr uint32_t FMAP_WEIGHT_BANDWIDTH_COEFF_1V1 = 1;
constexpr uint32_t FMAP_WEIGHT_BANDWIDTH_COEFF_4V1 = 4;
constexpr uint32_t FMAP_WEIGHT_BANDWIDTH_COEFF_6V1 = 6;

constexpr int8_t BB_PHASE_1 = 1;
constexpr int8_t BB_PHASE_2 = 2;
constexpr uint32_t FP16_DTYPE_SIZE = 2;

// load3dv2 hin/win max value
constexpr uint64_t LOAD3DV2_HIN_WIN_LIMIT_VALUE = 32767;

struct AscendApiCubeTypeMap {
public:
    struct {
        ConvDtype madType;
        ConvDtype biasType;
    } typeMaps[static_cast<uint8_t>(ConvDtype::UNDEFINED) + 1] = {
        {ConvDtype::UNDEFINED, ConvDtype::UNDEFINED}
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
        SetMapping(ConvDtype::FLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::BFLOAT16, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::HIFLOAT8, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
        SetMapping(ConvDtype::FLOAT8_E4M3FN, ConvDtype::FLOAT32, ConvDtype::FLOAT32);
    }

private:
    void SetMapping(ConvDtype key, ConvDtype madType, ConvDtype biasType)
    {
        typeMaps[static_cast<uint8_t>(key)].madType = madType;
        typeMaps[static_cast<uint8_t>(key)].biasType = biasType;
    }
};

struct AscendApiMknMap {
    int32_t GetByIndex(uint32_t idx) const
    {
        if (idx > MKN_MAX_SIZE - 1) {
        return MKN_VALUE_DEFAULT;
        }
        return map[idx];
    }
    int8_t map[MKN_MAX_SIZE];
};
 
struct AscendApiCubeMkn {
    int8_t toIdx[static_cast<uint8_t>(ConvDtype::UNDEFINED) + 1] = {0};
    AscendApiMknMap maps[3] = {{M0, B16_K0, N0}, // fp16
                               {M0, B32_K0, N0}, // fp32
                               {M0, B8_K0, N0}}; // int8 hif8 fp8
    uint32_t GetMKN(ConvDtype dType, uint32_t idx) const
    {
        return maps[toIdx[static_cast<uint8_t>(dType)]].GetByIndex(idx);
    }
    AscendApiCubeMkn()
    {
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT32)] = 1;
        toIdx[static_cast<uint8_t>(ConvDtype::BFLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::INT8)] = CONST_VALUE_2;
        toIdx[static_cast<uint8_t>(ConvDtype::HIFLOAT8)] = CONST_VALUE_2;
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT8_E4M3FN)] = CONST_VALUE_2;
    }
};

const AscendApiCubeTypeMap CUBE_TYPE_TAB = AscendApiCubeTypeMap();
const AscendApiCubeMkn CUBE_MKN_TAB = AscendApiCubeMkn();

uint64_t AlignB(uint64_t a, uint64_t b);
uint64_t CeilDiv(uint64_t a, uint64_t b);
uint64_t FloorAlign(uint64_t a, uint64_t b);
uint64_t Gcd(uint64_t a, uint64_t b);
void CalcCommFactor(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList);
void CalcCommFactorWithPowerOfTwo(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList);
void CalcCommFactorOfTwoNum(const uint64_t num1, const uint64_t num2, std::vector<uint64_t> &resList);
void VectorElementMultip(std::vector<uint64_t> &range, const uint64_t value);
bool IsEqual(std::vector<ConvDtype>& arr1, const std::vector<ConvDtype>& arr2, uint32_t size);
uint64_t CalcKhDilated(uint64_t singlekH, uint64_t dilationH);
uint64_t Conv2DInferHiL1(uint64_t inputHoL1, uint64_t khDilated, uint64_t hi, uint64_t strideH);
int64_t FindMaxProductUnderLimit(int64_t a, int64_t limit);
uint64_t DivideAndAlign(uint64_t num, uint64_t b, uint64_t c);
uint64_t Lcm(const uint64_t valueA, const uint64_t valueB);
} // namespace conv_tiling

#endif // ASCENDC_TILING_CONV_API_TILING_UTIL_H