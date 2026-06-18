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
 * \file fake_quant_with_min_max_vars_per_channel_base.h
 * \brief CastTrait 集中定义（DESIGN.md v2.1 §3.5.1）
 *
 * VF Compute 的 floor(x+0.5) 实现链路：
 *   fp32 (vregScaled) --CAST_FLOOR--> int32 (vregInt) --UNKNOWN--> fp32 (vregFloored)
 *
 * 字段值严格按 Cast-45.md 官方矩阵复制：
 *   - 表 3「浮点转整数」float → int32 行：layoutMode = UNKNOWN, satMode = NO_SAT,
 *     mrgMode = ZEROING, roundMode = CAST_FLOOR
 *   - 表 5「整数转浮点」int32 → float 行：layoutMode = UNKNOWN, satMode = UNKNOWN,
 *     mrgMode = ZEROING, roundMode = UNKNOWN
 */
#ifndef _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_BASE_H_
#define _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_BASE_H_

#include "kernel_operator.h"

namespace NsFakeQuantPerChannel {

// fp32 → int32, 向负无穷舍入（TF floor 语义）
// 来源：Cast-45.md 表 3「浮点转整数」`float → int32` 行
static constexpr AscendC::Reg::CastTrait kCastFp32ToInt32Floor = {
    AscendC::Reg::RegLayout::UNKNOWN,       // 表 3 规定 float→int32 = UNKNOWN
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_FLOOR
};

// int32 → fp32, round 选 CAST_NONE
// 来源：Cast-45.md 表 5「整数转浮点」`int32 → float` 行 +
//      实际验证 (dav_3510 后端 vcvt(s32→f32) 编译期强制要求 roundMode ∈ {R,A,F,C,Z}；
//       SDK kernel_reg_compute_vec_vconv_impl.h::CastOperator 对 Tuple<float,int32_t>
//       特殊处理：当 roundMode==CAST_NONE 时自动转 CAST_RINT，避免 static_assert 失败。
//       使用 CAST_NONE 是与 SDK 兼容的写法，等价于 round-to-nearest)
static constexpr AscendC::Reg::CastTrait kCastInt32ToFp32 = {
    AscendC::Reg::RegLayout::UNKNOWN,       // 表 5 规定 int32→float = UNKNOWN
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE           // 触发 SDK 的 conditionNoneToRint 路径
};

}  // namespace NsFakeQuantPerChannel

#endif  // _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_BASE_H_
