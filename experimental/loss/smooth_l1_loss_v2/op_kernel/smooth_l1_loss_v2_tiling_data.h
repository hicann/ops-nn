/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_v2_tiling_data.h
 * \brief tiling data struct for SmoothL1LossV2
 */
#include <cstdint>
#include <stdint.h>
#include <cctype>
#include <cmath>
#include <type_traits>
#ifndef SMOOTH_L1_LOSS_V2_TILING_DATA_H_
#define SMOOTH_L1_LOSS_V2_TILING_DATA_H_

struct SmoothL1LossV2TilingData {
    uint64_t smallCoreDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t finalSmallTileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t smallTailDataNum = 0;
    uint64_t bigTailDataNum = 0;
    uint64_t tailBlockNum = 0;
    uint64_t totalDataNum = 0; // 实际元素总数
    uint32_t blockBytes = 0;   // 平台 block bytes
    float sigma = 1.0f;        // 平滑系数
    int reduction = 0;         // 归约模式（0:none, 1:sum, 2:mean）
};

namespace smooth_l1_loss_v2_host {

template <typename StrAttrPtrT>
inline const char* AsCStr(StrAttrPtrT attr)
{
    if (attr == nullptr) {
        return nullptr;
    }
    using PointeeT = std::remove_cv_t<std::remove_pointer_t<StrAttrPtrT>>;
    if constexpr (std::is_same_v<PointeeT, char>) {
        return attr;
    } else {
        return attr->c_str();
    }
}

inline int32_t ParseReductionTypeFromToken(const char* token)
{
    if (token == nullptr) {
        return -1;
    }

    auto eq_icase = [](const char* lhs, const char* rhs) -> bool {
        if (lhs == nullptr || rhs == nullptr) {
            return false;
        }
        while (*lhs != '\0' && *rhs != '\0') {
            char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(*lhs)));
            char rc = static_cast<char>(std::tolower(static_cast<unsigned char>(*rhs)));
            if (lc != rc) {
                return false;
            }
            ++lhs;
            ++rhs;
        }
        return *lhs == '\0' && *rhs == '\0';
    };

    if (eq_icase(token, "none") || eq_icase(token, "n")) {
        return 0;
    }
    if (eq_icase(token, "sum") || eq_icase(token, "s")) {
        return 1;
    }
    if (eq_icase(token, "mean") || eq_icase(token, "m")) {
        return 2;
    }
    return -1;
}

template <typename AttrsT>
inline bool TryGetReductionTypeFromStringAttrs(const AttrsT* attrs, int32_t& reductionType)
{
    // Prefer string attr: op_def order is sigma(0), reduction(1).
    auto attr = attrs->GetStr(1);
    if (attr == nullptr) {
        // Compatibility with UT/runtime paths that may only provide one string attr.
        attr = attrs->GetStr(0);
    }
    if (attr == nullptr) {
        return false;
    }
    int32_t reduction = ParseReductionTypeFromToken(AsCStr(attr));
    if (reduction < 0) {
        return false;
    }
    reductionType = reduction;
    return true;
}

template <typename AttrsT>
inline bool TryGetRawReductionTypeFromIntAttrs(const AttrsT* attrs, int32_t& rawReduction)
{
    if (attrs == nullptr) {
        return false;
    }

    auto pickReductionFromIndex = [&](int idx, bool& ok) -> int32_t {
        ok = false;
        auto iv = attrs->GetInt(idx);
        if (iv == nullptr) {
            return -1;
        }
        int32_t v = static_cast<int32_t>(*iv);
        if (v < 0 || v > 2) {
            return -1;
        }
        auto fv = attrs->GetFloat(idx);
        if (fv != nullptr) {
            float f = *fv;
            if (std::fabs(f - static_cast<float>(v)) > 1e-6f) {
                return -1;
            }
        }
        ok = true;
        return v;
    };

    bool ok1 = false;
    int32_t raw1 = pickReductionFromIndex(1, ok1);
    bool ok0 = false;
    int32_t raw0 = pickReductionFromIndex(0, ok0);

    if (ok1) {
        rawReduction = raw1;
    } else if (ok0) {
        rawReduction = raw0;
    } else {
        return false;
    }
    return rawReduction >= 0;
}

inline int32_t MapRawReductionType(int32_t rawReduction, bool remapIntReductionOrder)
{
    if (rawReduction < 0) {
        return -1;
    }

    if (!remapIntReductionOrder) {
        return rawReduction;
    }
    if (rawReduction == 0) {
        return 0;
    }
    if (rawReduction == 1) {
        return 2;
    }
    return 1;
}

template <typename AttrsT>
inline int32_t GetReductionTypeFromAttrs(const AttrsT* attrs, bool remapIntReductionOrder)
{
    if (attrs == nullptr) {
        return 2;
    }

    int32_t reductionType = -1;
    if (TryGetReductionTypeFromStringAttrs(attrs, reductionType)) {
        return reductionType;
    }

    int32_t rawReduction = -1;
    if (!TryGetRawReductionTypeFromIntAttrs(attrs, rawReduction)) {
        return -1;
    }
    return MapRawReductionType(rawReduction, remapIntReductionOrder);
}

template <typename AttrsT>
inline float GetSigmaFromAttrs(const AttrsT* attrs)
{
    float sigma = 1.0f;
    if (attrs == nullptr) {
        return sigma;
    }

    auto pickSigmaFromIndex = [&](int idx, bool& ok) -> float {
        ok = false;
        auto fv = attrs->GetFloat(idx);
        if (fv == nullptr) {
            return 1.0f;
        }
        float f = *fv;
        auto iv = attrs->GetInt(idx);
        if (iv != nullptr) {
            int v = *iv;
            if (v >= 0 && v <= 2 && std::fabs(f - static_cast<float>(v)) <= 1e-6f) {
                return 1.0f;
            }
        }
        ok = true;
        return f;
    };

    bool ok0 = false;
    float s0 = pickSigmaFromIndex(0, ok0);
    bool ok1 = false;
    float s1 = pickSigmaFromIndex(1, ok1);

    if (ok0) {
        sigma = s0;
    } else if (ok1) {
        sigma = s1;
    }

    if (!std::isfinite(sigma) || sigma <= 0.0f) {
        sigma = 1.0f;
    }
    return sigma;
}

template <typename ContextT>
inline int32_t GetReductionTypeFromContext(const ContextT* context, bool remapIntReductionOrder)
{
    if (context == nullptr) {
        return 2;
    }
    return GetReductionTypeFromAttrs(context->GetAttrs(), remapIntReductionOrder);
}

template <typename ContextT>
inline float GetSigmaFromContext(const ContextT* context)
{
    if (context == nullptr) {
        return 1.0f;
    }
    return GetSigmaFromAttrs(context->GetAttrs());
}

} // namespace smooth_l1_loss_v2_host

#endif