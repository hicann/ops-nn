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
 * \file bn3d_training_reduce_check_support.h
 * \brief BN3DTrainingReduce Ascend 950 custom-candidate origin-format policy.
 */

#ifndef BN3D_TRAINING_REDUCE_CHECK_SUPPORT_H
#define BN3D_TRAINING_REDUCE_CHECK_SUPPORT_H

#include "register/op_def_registry.h"

namespace ops {
// NDC1HWC0 的 InferShape、Tiling 与 Kernel 实现继续保留。本开关只控制当前 custom 候选；
// 平台级入口还需 built-in 同名候选启用等价检查。后续放开时须同步文档与图模式用例。
constexpr bool ENABLE_NDC1HWC0_PUBLIC_ORIGIN = false;

inline ge::graphStatus CheckSupport4BN3DTrainingReduce(const ge::Operator& op, ge::AscendString& result)
{
    const ge::Format originFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(op.GetInputDescByName("x").GetOriginFormat()));
    if (!ENABLE_NDC1HWC0_PUBLIC_ORIGIN && originFormat == ge::FORMAT_NDC1HWC0) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "NDC1HWC0 origin format is not publicly supported on Ascend 950."})");
        return ge::GRAPH_FAILED;
    }

    result = ge::AscendString(
        R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "BN3DTrainingReduce CheckSupport passed."})");
    return ge::GRAPH_SUCCESS;
}
} // namespace ops

#endif // BN3D_TRAINING_REDUCE_CHECK_SUPPORT_H
