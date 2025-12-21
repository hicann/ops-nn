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
 * \file conv3d_tiling_utils.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_UTILS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_UTILS_H

#include <vector>
#include "conv3d_api_tiling.h"
#include "conv3d_api_tiling_utils.h"

namespace optiling {
namespace Conv3dOpsTiling {

constexpr uint32_t INPUT_FMAP_INDEX = 0;
constexpr uint32_t INPUT_WEIGHT_INDEX = 1;
constexpr uint32_t INPUT_BIAS_INDEX = 2;
constexpr uint32_t INPUT_SCALE_INDEX = 3;
constexpr uint32_t INPUT_OFFSET_INDEX = 4;
constexpr uint32_t INPUT_PERTOKEN_SCALE_INDEX = 5;
constexpr uint32_t INPUT_OFFSET_W_INDEX = 6;
constexpr uint32_t OUTPUT_INDEX = 0;

constexpr uint32_t FORMAT_NCDHW_N_INDEX = 0;
constexpr uint32_t FORMAT_NCDHW_C_INDEX = 1;
constexpr uint32_t FORMAT_NCDHW_D_INDEX = 2;
constexpr uint32_t FORMAT_NCDHW_H_INDEX = 3;
constexpr uint32_t FORMAT_NCDHW_W_INDEX = 4;
constexpr uint32_t FORMAT_NCDHW_DIM = 5;

constexpr uint32_t FORMAT_NDC1HWC0_N_INDEX = 0;
constexpr uint32_t FORMAT_NDC1HWC0_D_INDEX = 1;
constexpr uint32_t FORMAT_NDC1HWC0_C1_INDEX = 2;
constexpr uint32_t FORMAT_NDC1HWC0_H_INDEX = 3;
constexpr uint32_t FORMAT_NDC1HWC0_W_INDEX = 4;
constexpr uint32_t FORMAT_NDC1HWC0_C0_INDEX = 5;
constexpr uint32_t FORMAT_NDC1HWC0_DIM = 6;

constexpr uint32_t FORMAT_ND_DIM = 1;

constexpr uint32_t FORMAT_FRACTAL_3D_DKCIN1KHKW_INDEX = 0;
constexpr uint32_t FORMAT_FRACTAL_3D_N1_INDEX = 1;
constexpr uint32_t FORMAT_FRACTAL_3D_N0_INDEX = 2;
constexpr uint32_t FORMAT_FRACTAL_3D_C0_INDEX = 3;
constexpr uint32_t FORMAT_FRACTAL_3D_DIM = 4;

constexpr uint32_t ATTR_STRIDE_INDEX = 0;
constexpr uint32_t ATTR_PAD_INDEX = 1;
constexpr uint32_t ATTR_DILATION_INDEX = 2;
constexpr uint32_t ATTR_GROUP_INDEX = 3;
constexpr uint32_t ATTR_DATA_FORMAT_INDEX = 4;
constexpr uint32_t ATTR_HF32_FLAG_INDEX = 7;
constexpr uint32_t ATTR_OP_IMPL_MODE_INDEX = 8;

constexpr uint32_t PAD_HEAD_INDEX = 0;
constexpr uint32_t PAD_TAIL_INDEX = 1;
constexpr uint32_t PAD_UP_INDEX = 2;
constexpr uint32_t PAD_DOWN_INDEX = 3;
constexpr uint32_t PAD_LEFT_INDEX = 4;
constexpr uint32_t PAD_RIGHT_INDEX = 5;
constexpr uint32_t FORMAT_PAD_DIM = 6;

constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_FILTER_H_W = 511;

constexpr uint32_t MAX_16_BIT_NUM = 65535;

constexpr uint32_t MIN_WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr uint32_t WORKSPACE_NUM = 4;
constexpr uint32_t C0_BYTE_SIZE = 32;

constexpr uint32_t CUBE_UNIT = 16;
constexpr uint32_t FP16_CUBE_UNIT = 16;
constexpr uint32_t FP32_CUBE_UNIT = 8;
constexpr uint32_t INT8_CUBE_UNIT = 32;
constexpr uint32_t MKN_MAX_SIZE = 3;
constexpr uint32_t MKN_M_INDEX = 0;
constexpr uint32_t MKN_K_INDEX = 1;
constexpr uint32_t MKN_N_INDEX = 2;
constexpr uint32_t MKN_VALUE_DEFAULT = 16;

constexpr uint32_t CONST_VALUE_2 = 2;

constexpr uint32_t BATCH_AICORE_COF = 2;
constexpr uint32_t C0_SIZE = 32;

//blockdim: [batchDim, mdim, nDim, doDim, groupDim]
constexpr uint32_t BLOCKDIM_DEC_NUM = 5;
constexpr uint32_t BLOCKDIM_BATCH_IDX = 0;
constexpr uint32_t BLOCKDIM_M_IDX = 1;
constexpr uint32_t BLOCKDIM_N_IDX = 2;
constexpr uint32_t BLOCKDIM_DO_IDX = 3;
constexpr uint32_t BLOCKDIM_GROUP_IDX = 4;

constexpr uint32_t MKN_M_IDX = 0;
constexpr uint32_t MKN_K_IDX = 1;
constexpr uint32_t MKN_N_IDX = 2;

constexpr uint32_t COUNT_PARAMS_WITH_BIAS = 4; // [fmap, weight, bias, output]
constexpr uint32_t COUNT_PARAMS_WITHOUT_BIAS = 3; // [fmap, weight, output]

constexpr uint64_t INITIAL_AICORE_ZERO = 0;
constexpr uint64_t INITIAL_L2_RATE_ZERO = 0;

constexpr uint64_t MAX_ORI_ONE_DIM_SIZE = 1000000;
constexpr uint64_t MAX_ORI_FMAP_SIZE = 10000000;

constexpr uint32_t PBUFFERFLAG_L0A_MASK = 1;
constexpr uint32_t PBUFFERFLAG_L0B_MASK = 2;
constexpr uint32_t PBUFFERFLAG_L0C_MASK = 4;

const std::vector<std::vector<Conv3dApiTiling::ConvDtype>> SUPPORTED_TYPES_WITH_BIAS = {
    {Conv3dApiTiling::ConvDtype::FLOAT16, Conv3dApiTiling::ConvDtype::FLOAT16,
     Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT16},
    {Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32,
     Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32},
    {Conv3dApiTiling::ConvDtype::INT8, Conv3dApiTiling::ConvDtype::INT8,
     Conv3dApiTiling::ConvDtype::INT32, Conv3dApiTiling::ConvDtype::INT32},
    {Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16,
     Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::BF16}
};
const std::vector<std::vector<Conv3dApiTiling::ConvDtype>> SUPPORTED_TYPES_WITHOUT_BIAS = {
    {Conv3dApiTiling::ConvDtype::FLOAT16, Conv3dApiTiling::ConvDtype::FLOAT16, Conv3dApiTiling::ConvDtype::FLOAT16},
    {Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32},
    {Conv3dApiTiling::ConvDtype::INT8, Conv3dApiTiling::ConvDtype::INT8, Conv3dApiTiling::ConvDtype::INT32},
    {Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16},
};

struct Conv3DAscendcShapesInfo {
    uint32_t batch = 1;
    uint32_t cIn = 1;
    uint32_t di = 1;
    uint64_t hi = 1;
    uint64_t wi = 1;
    uint32_t cOut = 1;
    uint32_t kd = 1;
    uint32_t kh = 1;
    uint32_t kw = 1;
    uint32_t dOut = 1;
    uint64_t ho = 1;
    uint64_t wo = 1;
    uint64_t cinOpt = 1;
    uint64_t coutOpt = 1;
};

struct Conv3DTilingFlag {
    bool hasBias = false;
    bool hasScale = false;
};

struct BlockDimRange {
  std::vector<uint32_t> aicNumRange;
  std::vector<uint32_t> batchRange;
  std::vector<uint32_t> mRange;
  std::vector<uint32_t> nRange;
  std::vector<uint32_t> doRange;
  std::vector<uint32_t> groupRange;
};

struct BlockDimConstParas {
  uint64_t m0;
  uint64_t n0;
  uint64_t k0;
  uint64_t ci1;
  uint64_t co1;
};

struct BlockDimRes {
  uint32_t batchDim = 0;
  uint32_t mDim = 0;
  uint32_t nDim = 0;
  uint32_t doDim = 0;
  uint32_t groupDim = 0;
  uint64_t minCost = 0;
};

static std::map<Conv3dApiTiling::ConvDtype, std::string> g_convDtypeToStr = {
    {Conv3dApiTiling::ConvDtype::FLOAT16, "float16"},
    {Conv3dApiTiling::ConvDtype::FLOAT32, "float32"},
    {Conv3dApiTiling::ConvDtype::BF16, "bfloat16"},
    {Conv3dApiTiling::ConvDtype::INT4, "int4"},
    {Conv3dApiTiling::ConvDtype::INT8, "int8"},
    {Conv3dApiTiling::ConvDtype::UINT8, "uint8"},
    {Conv3dApiTiling::ConvDtype::INT64, "int64"},
    {Conv3dApiTiling::ConvDtype::UINT64, "uint64"},
    {Conv3dApiTiling::ConvDtype::INT32, "int32"},
};

struct AscendOpsCubeTypeMap {
    struct {
        Conv3dApiTiling::ConvDtype madType;
        Conv3dApiTiling::ConvDtype biasType;
    } typeMaps[static_cast<uint8_t>(Conv3dApiTiling::ConvDtype::CONVDTYPEMAX) + 1] =
        {{Conv3dApiTiling::ConvDtype::CONVDTYPEMAX, Conv3dApiTiling::ConvDtype::CONVDTYPEMAX}};

    Conv3dApiTiling::ConvDtype ToBiasType(Conv3dApiTiling::ConvDtype type) const {
        return typeMaps[static_cast<uint8_t>(type)].biasType;
    }
    Conv3dApiTiling::ConvDtype ToMadType(Conv3dApiTiling::ConvDtype type) const {
        return typeMaps[static_cast<uint8_t>(type)].madType;
    }

    AscendOpsCubeTypeMap() {
        SetMapping(Conv3dApiTiling::ConvDtype::INT4, Conv3dApiTiling::ConvDtype::INT32, Conv3dApiTiling::ConvDtype::INT32);
        SetMapping(Conv3dApiTiling::ConvDtype::INT8, Conv3dApiTiling::ConvDtype::INT32, Conv3dApiTiling::ConvDtype::INT32);
        SetMapping(Conv3dApiTiling::ConvDtype::UINT8, Conv3dApiTiling::ConvDtype::INT32, Conv3dApiTiling::ConvDtype::INT32);
        SetMapping(Conv3dApiTiling::ConvDtype::FLOAT16, Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32);
        SetMapping(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32);
        SetMapping(Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32);
    }

    private:
    void SetMapping(Conv3dApiTiling::ConvDtype key, Conv3dApiTiling::ConvDtype madType, Conv3dApiTiling::ConvDtype biasType) {
        typeMaps[static_cast<uint8_t>(key)].madType = madType;
        typeMaps[static_cast<uint8_t>(key)].biasType = biasType;    // bias dtype in bt
    }
};

const AscendOpsCubeTypeMap CUBE_TYPE_MAP = AscendOpsCubeTypeMap();

struct AscendOpsMknMap {
    uint32_t GetByIndex(uint32_t idx) const {
        if (idx > MKN_MAX_SIZE - 1) {
            return MKN_VALUE_DEFAULT;
        }
        return static_cast<uint32_t>(map[idx]);
    }
    int8_t map[MKN_MAX_SIZE];
};

struct AscendOpsCubeMkn {
    int8_t toIdx[static_cast<uint8_t>(Conv3dApiTiling::ConvDtype::CONVDTYPEMAX) + 1] = {0};
    AscendOpsMknMap maps[3] = {{CUBE_UNIT, FP16_CUBE_UNIT, CUBE_UNIT}, // fp16
                              {CUBE_UNIT, FP32_CUBE_UNIT, CUBE_UNIT}, // fp32
                              {CUBE_UNIT, INT8_CUBE_UNIT, CUBE_UNIT}}; // int8
    uint32_t GetMKN(Conv3dApiTiling::ConvDtype dType, uint32_t idx) {
        return maps[toIdx[static_cast<uint8_t>(dType)]].GetByIndex(idx);
    }
    AscendOpsCubeMkn() {
        toIdx[static_cast<uint8_t>(Conv3dApiTiling::ConvDtype::FLOAT16)] = static_cast<int8_t>(0);
        toIdx[static_cast<uint8_t>(Conv3dApiTiling::ConvDtype::FLOAT32)] = static_cast<int8_t>(1);
        toIdx[static_cast<uint8_t>(Conv3dApiTiling::ConvDtype::BF16)] = static_cast<int8_t>(0);
        toIdx[static_cast<uint8_t>(Conv3dApiTiling::ConvDtype::INT8)] = static_cast<int8_t>(CONST_VALUE_2);
    }
};

static AscendOpsCubeMkn g_cubeMknMap = AscendOpsCubeMkn();

uint64_t CeilDiv(uint64_t a, uint64_t b);
void CalcCommFactor(const uint64_t num, const uint32_t numMax, std::vector<uint32_t> &reslist);
bool IsArrayEqual(std::vector<Conv3dApiTiling::ConvDtype>& arr1,
                  const std::vector<Conv3dApiTiling::ConvDtype>& arr2,
                  uint32_t size);
uint64_t AlignB(uint64_t a, uint64_t b);
uint64_t InferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH);
uint64_t InferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW);

} // namespace Conv3dOpsTiling

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_UTILS_H