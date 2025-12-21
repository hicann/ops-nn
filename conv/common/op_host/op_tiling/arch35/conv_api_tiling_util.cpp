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
 * \file conv_api_tiling_util.cpp
 * \brief
 */

#include "conv_api_tiling_util.h"

namespace conv_tiling {
uint64_t AlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        TILING_LOG_DEBUG("The denominator cannot be zero!");
        return 0;
    }

    return ((a + b - 1) / b) * b;
}

uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        TILING_LOG_DEBUG("The denominator cannot be zero!");
        return 0;
    }

    return (a + b - 1) / b;
}

uint64_t FloorAlign(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }

    return a - (a % b);
}

uint64_t Gcd(uint64_t a, uint64_t b)
{
    if (b == 0) {
        TILING_LOG_DEBUG("The denominator cannot be zero!");
        return 0;
    }

    uint64_t c;
    if (a < b) {
        c = a;
        a = b;
        b = c;
    }

    while (a % b != 0) {
        c = a % b;
        a = b;
        b = c;
    }

    return b;
}

uint64_t Lcm(const uint64_t valueA, const uint64_t valueB)
{
    if (valueB == 0 || valueA == 0) {
        TILING_LOG_DEBUG("The denominator cannot be zero!");
        return 0;
    }

    uint64_t para1 = valueA;
    uint64_t para2 = valueB;
    uint64_t tmpValue = para1;
    while (para1 % para2 != 0) {
        tmpValue = para1;
        para1 = para2;
        para2 = tmpValue % para2;
    }

    return (static_cast<uint64_t>(valueA) * valueB) / para2;
}

void CalcCommFactor(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList)
{
    uint64_t sqrtMax = static_cast<uint64_t>(sqrt(num));
    for (uint64_t i = 1; i <= sqrtMax; ++i) {
        if (num % i == 0) {
            if (i <= numMax) {
                resList.emplace_back(i);
            }
            uint64_t right = num / i;
            if (right != i && right <= numMax) {
                resList.emplace_back(right);
            }
        }
    }
    sort(resList.begin(), resList.end());
}

void CalcCommFactorWithPowerOfTwo(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList)
{
    CalcCommFactor(num, numMax, resList);
    for (uint64_t i = 2; i <= min(num, numMax); i *= CONST_VALUE_2) {
        if (std::find(resList.begin(), resList.end(), i) == resList.end()) {
            resList.emplace_back(i);
        }
    }
    sort(resList.begin(), resList.end());
}

void CalcCommFactorOfTwoNum(const uint64_t num1, const uint64_t num2, std::vector<uint64_t> &resList)
{
    uint64_t minNum = min(num1, num2);
    for (uint64_t i = 1; i <= minNum; ++i) {
        if (num1 % i == 0 && num2 % i == 0) {
                resList.emplace_back(i);
        }
    }
    sort(resList.begin(), resList.end());
}

void VectorElementMultip(std::vector<uint64_t>& range, const uint64_t value)
{
    for (auto& factor : range) {
        factor *= value;
    }
}

bool IsEqual(std::vector<ConvDtype>& arr1, const std::vector<ConvDtype>& arr2, uint32_t size)
{
    if (std::min(arr1.size(), arr2.size()) > size) {
        return false;
    }

    for (uint32_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

uint64_t CalcKhDilated(uint64_t singlekH, uint64_t dilationH)
{
    uint64_t khDilated = (singlekH - 1) * dilationH + 1;
    return khDilated;
}

uint64_t Conv2DInferHiL1(uint64_t inputHoL1, uint64_t khDilated, uint64_t hi, uint64_t strideH)
{
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

int64_t FindMaxProductUnderLimit(int64_t a, int64_t limit)
{
    int64_t mutiple = 1;

    while (a * mutiple <= limit) {
        mutiple++;
    }

    return mutiple - 1;
}

uint64_t DivideAndAlign(uint64_t num, uint64_t b, uint64_t c) {
    if (c == 0) {
        return num;
    }

    while (num > b || (num % c != 0)) {
        num /= CONST_VALUE_2;
    }
    return FloorAlign(num, c);
}
} // namespace conv_tiling
