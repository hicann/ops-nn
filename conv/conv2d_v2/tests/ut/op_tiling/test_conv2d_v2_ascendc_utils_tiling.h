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
 * \file test_conv2d_v2_tiling_utils.h.h
 * \brief
 */

#ifndef TEST_CONV2D_V2_ASCENDC_UTILS_TILING_H
#define TEST_CONV2D_V2_ASCENDC_UTILS_TILING_H

#include "platform/platform_info.h"
#include "../../../op_host/op_tiling/conv2d_v2_tiling.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_HWmode.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_Mmode.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_BBmode.h"

namespace conv_tiling_utils {
using namespace std;
using namespace conv_tiling_algo_hw;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;

// 硬件相关常量
constexpr uint32_t CUBE_M0 = 16;
constexpr uint32_t CUBE_N0 = 16;
constexpr uint32_t CUBE_K0_32 = 32;
constexpr uint32_t CUBE_K0_16 = 16;
constexpr uint32_t CUBE_K0_8 = 8;
constexpr uint32_t CUBE_C0_SIZE = 32;
constexpr uint32_t CUBE_C04_SIZE = 4;
constexpr uint32_t AIC_NUM = 32;

// 内存相关常量
constexpr uint32_t MEM_SIZE_64B = 64;
constexpr uint32_t MEM_SIZE_128B = 128;
constexpr uint32_t MEM_SIZE_1K = 1024;
constexpr uint32_t MEM_SIZE_4K = 4096;
constexpr uint32_t MEM_SIZE_64K = 65536;
constexpr uint32_t MEM_SIZE_100K = 102400;
constexpr uint32_t MEM_SIZE_200K = 204800;
constexpr uint32_t MEM_SIZE_212K = 217088;
constexpr uint32_t MEM_SIZE_256K = 262144;
constexpr uint32_t MEM_SIZE_512K = 524288;
constexpr uint32_t MEM_SIZE_300K = 307200;

// 基本块相关常量
constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_64 = 64;
constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_128 = 128;
constexpr uint32_t BASICBLOCK_INIT_VALUE_64 = 64;
constexpr uint32_t BASICBLOCK_INIT_VALUE_128 = 128;
constexpr uint32_t BASICBLOCK_INIT_VALUE_256 = 256;
constexpr uint32_t BASICBLOCK_INIT_VALUE_512 = 512;
constexpr uint32_t BASICBLOCK_INIT_VALUE_1024 = 1024;

// 数值常量
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_3 = 3;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t NUM_9 = 9;
constexpr uint32_t NUM_10 = 10;
constexpr uint32_t NUM_14 = 14;
constexpr uint32_t DTYPESIZE_2 = 2;
constexpr uint32_t DTYPESIZE_4 = 4;
constexpr uint32_t DTYPESIZE_8 = 8;

// Dtype相关常量组合
const std::vector<std::vector<ConvDtype>> SUPPORTED_QUANT_TYPES_WITHOUT_BIAS = {
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
const std::vector<std::vector<ConvDtype>> SUPPORTED_QUANT_TYPES_WITH_BIAS = {
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
const std::vector<std::vector<ConvDtype>> SUPPORTED_CONV2D_TYPES_WITH_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16}
};
const std::vector<std::vector<ConvDtype>> SUPPORTED_CONV2D_TYPES_WITHOUT_BIAS = {
    {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16},
    {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32},
    {ConvDtype::BFLOAT16, ConvDtype::BFLOAT16, ConvDtype::BFLOAT16}
};

// ConvShape 结构体定义
struct ConvShape {
    uint64_t inputV;
    uint64_t kernelV;
    uint64_t padone;
    uint64_t padtwo;
    uint64_t dilationV;
    uint64_t strideV;
};

uint64_t InferOut(ConvShape convShape);

int64_t InferHiL1(uint64_t inputHoL1, int64_t hi, uint64_t singlekH, uint64_t dilationH, uint64_t strideH);

int64_t InferWiL1(uint64_t inputWoL1, int64_t wi, uint64_t singlekW, uint64_t dilationW, uint64_t strideW);

uint64_t ConvCeilDiv(uint64_t a, uint64_t b);

uint64_t ConvGcd(uint64_t a, uint64_t b);

uint64_t ConvAlignB(uint64_t a, uint64_t b);
}
#endif