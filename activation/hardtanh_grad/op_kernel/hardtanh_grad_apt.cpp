/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file hardtanh_grad_apt.cpp
 * \brief torch.where((result <= min_val) | (result >= max_val), 0.0, grad)
 */
#include "kernel_operator.h"
#include "arch35/hardtanh_grad_dag.h"
#include "arch35/hardtanh_grad_struct.h"
#include "arch35/hardtanh_grad_tilingdata.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace HardtanhGradOp;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void hardtanh_grad(GM_ADDR result, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(HardtanhGradTilingData);

    TPipe pipe;
    if constexpr (dType == static_cast<uint64_t>(HardtanhGrad_TPL)) {
        ElementwiseSch<schMode, HardtanhGradOp::HardtanhGradDag<DTYPE_RESULT>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(static_cast<float>(tilingData.minVal));
        sch.template SetVar<float, 1>(static_cast<float>(tilingData.maxVal));
        sch.Init(result, grad, y);
        sch.Process();
    }
    return;
}
