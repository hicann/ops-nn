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
 * \file apply_adagrad_d.cpp
 * \brief apply_adagrad_d
 */
#include "kernel_operator.h"
#include "./arch35/apply_adagrad_d_tiling_key.h"
#include "./arch35/apply_adagrad_d_dag.h"
#include "apply_adagrad_d_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace ApplyAdagradDTilingData;

template <uint64_t schMode, uint64_t updateSlots, uint64_t dType>
__global__ __aicore__ void apply_adagrad_d(GM_ADDR var, 
                                           GM_ADDR accum, 
                                           GM_ADDR lr,
                                           GM_ADDR grad,
                                           GM_ADDR var_out,
                                           GM_ADDR accum_out,
                                           GM_ADDR workspace, 
                                           GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ApplyAdagradDTilingDataStruct);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdagradDTilingDataStruct, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if constexpr (static_cast<int>(updateSlots) > 0) {
        ElementwiseSch<schMode, ApplyAdagradDOp::ApplyAdagradDUpdateSlots<DTYPE_VAR>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(var, accum, lr, grad, var_out, accum_out);
        sch.Process();
    } else {
        ElementwiseSch<schMode, ApplyAdagradDOp::ApplyAdagradD<DTYPE_VAR>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(var, accum, lr, grad, var_out);
        sch.Process();  
    }
}
