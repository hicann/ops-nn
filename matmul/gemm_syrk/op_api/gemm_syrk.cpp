/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "gemm_syrk.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GemmSyrk);

aclTensor* GemmSyrk(const aclTensor* a, aclTensor* cRef, float alpha, float beta, bool transposeX, const char* fillMode,
                    aclOpExecutor* executor)
{
    L0_DFX(GemmSyrk, a, cRef, alpha, beta, transposeX);

    // No INFER_SHAPE call: the in-place cRef carries the complete output
    // descriptor (same-name input/output port), so the launcher consumes it
    // directly and the registered infer-shape hook is unnecessary here.
    // FLOAT32 only runs in HF32 cube mode; advertise 0x40 in op_impl_mode so
    // the tiling layer accepts it. FP16/BF16 keep the default 0x1.
    const int64_t opImplMode = a->GetDataType() == DataType::DT_FLOAT ? 0x40L : 0x1L;
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GemmSyrk, OP_INPUT(a, cRef), OP_OUTPUT(cRef),
                                           OP_ATTR(alpha, beta, transposeX, fillMode, opImplMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return cRef;
}

} // namespace l0op
