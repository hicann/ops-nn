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
 * \file test_conv2d_v2_tiling_utils.h.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <platform/soc_spec.h>
#include "platform/platform_info.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"

namespace conv_tiling_utils {
using namespace std;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;

uint64_t InferOut(ConvShape convShape)
{
    if (convShape.strideV == 0) {
        return 0;
    }
    return (convShape.inputV + convShape.padone + convShape.padtwo - convShape.dilationV * (convShape.kernelV - 1) - 1) / convShape.strideV + 1;
}

int64_t InferHiL1(uint64_t inputHoL1, int64_t hi, uint64_t singlekH,
                  uint64_t dilationH, uint64_t strideH)
{
    int64_t khDilated = (singlekH - 1) * dilationH + 1;
    int64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }
    return tmpHiL1;
}

int64_t InferWiL1(uint64_t inputWoL1, int64_t wi, uint64_t singlekW, uint64_t dilationW, uint64_t strideW)
{
    int64_t kwDilated = (singlekW - 1) * dilationW + 1;
    int64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }
    return tmpWiL1;
}

uint64_t ConvCeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

uint64_t ConvGcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
   if (b == 0) {
      return 0;
   }
   return ((a + b - 1) / b) * b;
}

} // conv_tiling_utils