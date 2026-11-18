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
 * \file foreach_log_regbase.h
 * \brief Register-only compute policy for ForeachLog on Ascend950.
 *        Computes y = ln(x) for each tensor in the list. FP16 and FP32 use the
 *        native dav_3510 Ln (mirrors the retired SIMT hlog/logf paths); BF16 is
 *        promoted to FP32 and rounded back by the shared unary policy.
 */

#ifndef FOREACH_LOG_REGBASE_H
#define FOREACH_LOG_REGBASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_log_tiling_data.h"
#include "../../foreach_utils/arch35/foreach_unary_regbase_common.h"

namespace ForeachLog {
using namespace AscendC;

struct LogOperation {
    static constexpr bool kPromoteHalf = false;

    template <typename T>
    __simd_callee__ static inline void Apply(Reg::RegTensor<T, Reg::RegTraitNumOne>& dstReg,
                                             Reg::RegTensor<T, Reg::RegTraitNumOne>& srcReg, Reg::MaskReg& mask)
    {
        Reg::Ln<T>(dstReg, srcReg, mask);
    }
};

template <typename T, typename Tiling>
using ForeachLogRegbase = ForeachUnaryRegbase::UnaryApplyKernel<T, Tiling, LogOperation>;
} // namespace ForeachLog

#endif // FOREACH_LOG_REGBASE_H
