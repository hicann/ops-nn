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
 * \file swiglu_group_quant_with_dual_axis_policy.h
 * \brief Compile-time scheduling policy for SwigluGroupQuantWithDualAxis.
 */
#ifndef SWIGLU_DUAL_AXIS_POLICY_H
#define SWIGLU_DUAL_AXIS_POLICY_H
#include <cstdint>

namespace SwigluDualAxisPolicy {
// Origin sharing depends on weight/output semantics, independently of H.
constexpr bool ShareOrigin(bool hasWeight, bool outputOrigin) { return !hasWeight && outputOrigin; }

// Independently qualified scheduling policy: do not derive it from ShareOrigin.
constexpr bool OwnWholeGroups(int64_t h, bool hasWeight, bool outputOrigin, bool hasGroup, int64_t groups,
                              int64_t cores)
{
    return !hasWeight && outputOrigin && h == 384 && hasGroup && groups >= cores;
}

} // namespace SwigluDualAxisPolicy
#endif
