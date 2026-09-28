/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SINGLE_LAYER_LSTM_GRAD_WIDE_WORKSPACE_H
#define SINGLE_LAYER_LSTM_GRAD_WIDE_WORKSPACE_H
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define LSTM_WIDE_INLINE __aicore__ inline
#else
#define LSTM_WIDE_INLINE inline
#endif

namespace LstmGradWide {
// Private FP32 storage only: none of these regions is an operator output tensor.
struct Workspace {
    static constexpr uint64_t ALIGN = 512;
    static constexpr uint64_t FLOAT_BYTES = 4;
    static constexpr uint64_t PLANES = 7;
    uint64_t x, w, bias, initH, initC, dy, dh, dc, cache;
    uint64_t dx, dw, db, dhPrev, dcPrev, legacy, bytes;
    uint64_t inputElements, weightElements, stateElements, planeElements;

    LSTM_WIDE_INLINE bool Region(uint64_t elements, uint64_t& offset)
    {
        constexpr uint64_t limit = (uint64_t{1} << 63) - 1;
        if (elements > (limit - ALIGN) / FLOAT_BYTES || bytes > limit - ALIGN) {
            return false;
        }
        const uint64_t length = (elements * FLOAT_BYTES + ALIGN - 1) / ALIGN * ALIGN;
        if (length > limit - bytes) {
            return false;
        }
        offset = bytes;
        bytes += length;
        return true;
    }

    LSTM_WIDE_INLINE bool Fill(int64_t t, int64_t b, int64_t i, int64_t h, int64_t biasParts)
    {
        // Kernel dimensions are signed 32-bit; reject overflow before evaluating products.
        constexpr uint64_t extentLimit = (uint64_t{1} << 31) - 1;
        if (t <= 0 || b <= 0 || i <= 0 || h <= 0 || biasParts < 0 || biasParts > 2 ||
            static_cast<uint64_t>(t) > extentLimit || static_cast<uint64_t>(b) > extentLimit ||
            static_cast<uint64_t>(i) + h > extentLimit / FLOAT_BYTES || static_cast<uint64_t>(h) > extentLimit / 4 ||
            static_cast<uint64_t>(t) * b > extentLimit) {
            return false;
        }
        const uint64_t rows = static_cast<uint64_t>(t) * b;
        stateElements = static_cast<uint64_t>(b) * h;
        planeElements = rows * h;
        inputElements = rows * i;
        weightElements = uint64_t{4} * h * (i + h);
        bytes = 0;
        if (!Region(inputElements, x) || !Region(weightElements, w) || !Region(uint64_t{4} * h * biasParts, bias) ||
            !Region(stateElements, initH) || !Region(stateElements, initC) || !Region(planeElements, dy) ||
            !Region(stateElements, dh) || !Region(stateElements, dc) || !Region(PLANES * planeElements, cache) ||
            !Region(inputElements, dx) || !Region(weightElements, dw) || !Region(uint64_t{4} * h, db) ||
            !Region(stateElements, dhPrev) || !Region(stateElements, dcPrev)) {
            return false;
        }
        // Existing FP32 matmul kernel: dgate + dxh + xh, in that order.
        const uint64_t legacyElements = rows * uint64_t{4} * h + (rows + b) * (static_cast<uint64_t>(i) + h);
        return Region(legacyElements, legacy);
    }
};
} // namespace LstmGradWide
#undef LSTM_WIDE_INLINE
#endif
