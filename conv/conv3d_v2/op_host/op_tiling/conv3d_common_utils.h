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
 * \file conv3d_common_utils.h
 * \brief Common utility functions for conv3d operations
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_COMMON_UTILS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_COMMON_UTILS_H

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

namespace Conv3dCommon {

// Shared constexpr constants used across conv3d tiling modules.
constexpr uint32_t C0_BYTE_SIZE = 32;
constexpr uint32_t C0_SIZE = 32;
constexpr uint32_t CONST_VALUE_2 = 2;

constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_FILTER_H_W = 511;

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

// Common utility function for multiplication with overflow check
template <typename T>
inline bool MulWithOverflowCheck(T &res, T a, T b)
{
    if (a == 0 || b == 0) {
        res = 0;
        return false;
    }
    T tmpRes = a * b;
    if (tmpRes / a != b) {
        return true;
    }
    res = tmpRes;
    return false;
}

// 调用时控制传参个数，避免栈溢出
template <typename T, typename... Args>
inline bool MulWithOverflowCheck(T &res, T a, T b, Args... args)
{
    T tmp;
    return MulWithOverflowCheck(tmp, a, b) || MulWithOverflowCheck(res, tmp, args...);
}

// Safely compute ceil(a / b). When b == 0, returns 0 to avoid division by zero.
inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - static_cast<uint64_t>(1)) / b;
}

// Align value up to "align" (multiple of align). When align == 0, returns 0.
inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return 0;
    }
    return CeilDiv(value, align) * align;
}

// Infer input H size (HiL1) from output Ho tile (HoL1) and conv attributes.
inline uint64_t InferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH)
{
    uint64_t khDilated = (singlekH - static_cast<uint64_t>(1)) * static_cast<uint64_t>(dilationH) + static_cast<uint64_t>(1);
    uint64_t tmpHiL1 = (inputHoL1 - static_cast<uint64_t>(1)) * static_cast<uint64_t>(strideH) + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }
    return tmpHiL1;
}

// Infer input W size (WiL1) from output Wo tile (WoL1) and conv attributes.
inline uint64_t InferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW)
{
    uint64_t kwDilated = (singlekW - static_cast<uint64_t>(1)) * static_cast<uint64_t>(dilationW) + static_cast<uint64_t>(1);
    uint64_t tmpWiL1 = (inputWoL1 - static_cast<uint64_t>(1)) * static_cast<uint64_t>(strideW) + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }
    return tmpWiL1;
}

// Internal helper for divisor enumeration; shared by overloads below.
template <typename TNum, typename TMax, typename TOut>
inline void CalcCommFactorImpl(TNum num, TMax numMax, std::vector<TOut> &resList)
{
    if (num == 0) {
        return;
    }

    TNum sqrtMax = static_cast<TNum>(std::sqrt(static_cast<double>(num)));
    for (TNum i = 1; i <= sqrtMax; ++i) {
        if (num % i == 0) {
            if (i <= static_cast<TNum>(numMax)) {
                resList.emplace_back(static_cast<TOut>(i));
            }
            TNum right = num / i;
            if (right != i && right <= static_cast<TNum>(numMax)) {
                resList.emplace_back(static_cast<TOut>(right));
            }
        }
    }

    std::sort(resList.begin(), resList.end());
}

// Enumerate common factors up to numMax; result stored as uint32_t (conv3d_base_tiling).
inline void CalcCommFactor(uint64_t num, uint32_t numMax, std::vector<uint32_t> &resList)
{
    CalcCommFactorImpl<uint64_t, uint32_t, uint32_t>(num, numMax, resList);
}

// Enumerate common factors up to numMax; result stored as uint64_t（conv3d_api_tiling）.
inline void CalcCommFactor(uint64_t num, uint64_t numMax, std::vector<uint64_t> &resList)
{
    CalcCommFactorImpl<uint64_t, uint64_t, uint64_t>(num, numMax, resList);
}

// Generic array equality check with size constraint.
template <typename T>
inline bool IsArrayEqual(const std::vector<T> &arr1, const std::vector<T> &arr2, uint32_t size)
{
    if (arr1.size() < size || arr2.size() < size) {
        return false;
    }
    for (uint32_t i = 0; i < size; ++i) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

} // namespace Conv3dCommon

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_COMMON_UTILS_H
