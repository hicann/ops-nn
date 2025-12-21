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
 * \file test_conv_ascendc_common_utils.h
 * \brief
 */

#ifndef ASCENDC_COMMON_UTILS_H
#define ASCENDC_COMMON_UTILS_H


#include "platform/platform_info.h"
#include "../../../op_host/op_tiling/conv2d_tiling.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_HWmode.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_Mmode.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_BBmode.h"

namespace conv_api_tiling_test {

using namespace std;
using namespace conv_tiling_algo_hw;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;

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
uint64_t InferHo(uint64_t inputHi, uint64_t kH, uint64_t padTop, uint64_t padBottom, uint64_t dilationH, uint64_t strideH);
uint64_t InferWo(uint64_t inputWi, uint64_t kW, uint64_t padLeft, uint64_t padRight, uint64_t dilationW, uint64_t strideW);
uint64_t InferHi(uint64_t inputHo, uint64_t kH, uint64_t padTop, uint64_t padBottom, uint64_t dilationH, uint64_t strideH);
uint64_t InferWi(uint64_t inputWo, uint64_t kW, uint64_t padLeft, uint64_t padRight, uint64_t dilationW, uint64_t strideW);
int64_t InferHiL1(uint64_t inputHoL1, int64_t hi, uint64_t singlekH, uint64_t dilationH, uint64_t strideH);
int64_t InferWiL1(uint64_t inputWoL1, int64_t wi, uint64_t singlekW, uint64_t dilationW, uint64_t strideW);


uint64_t ConvCeilDiv(uint64_t a, uint64_t b);

void Conv2DCase(vector<uint64_t> fmShape, vector<uint64_t> weightShape,
                vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations,
                vector<uint32_t> blockDims, std::vector<ConvDtype> dtypes,
                std::vector<bool> flags,
                std::vector<uint8_t> modes, uint32_t groups);
void CheckValidTilingData(optiling::TConv2DTiling &tilingData,
                          Conv2dTiling &tiling);

uint64_t CalcUsdL1Size(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbAL1,
                       int8_t pbBL1);

uint64_t CalcUsdL0ASize(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbAL0);

uint64_t CalcUsdL0BSize(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbBL0);

uint64_t CalcUsdL0CSize(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbCL0);
#endif

}