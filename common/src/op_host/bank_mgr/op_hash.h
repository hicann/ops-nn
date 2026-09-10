/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_OP_HASH_H_
#define RUNTIME_KB_OP_HASH_H_
#include <cstddef>
#include <cstdint>
#include <functional>
namespace RuntimeKb {
namespace {
constexpr size_t kGoldenRatio = 0x9e3779b9;
constexpr uint32_t kSixBitShift = 6U;
constexpr uint32_t kTwoBitShift = 2U;
constexpr uint32_t kEightBitShift = 8U;
constexpr uint32_t kSixtennBitShift = 16U;
constexpr uint32_t kHashSeed = 271828U;
} // namespace

template <typename T>
struct Hash {
    size_t operator()(const T& tt) const { return std::hash<T>()(tt); }
};

template <typename T>
inline void HashCombine(const T& val, uint32_t& out)
{
    out ^= Hash<T>()(val) + kGoldenRatio + (out << kSixBitShift) + (out >> kTwoBitShift);
}

uint32_t CommonHash(const void* src, uint32_t len, uint32_t seed = kHashSeed);
} // namespace RuntimeKb
#endif
