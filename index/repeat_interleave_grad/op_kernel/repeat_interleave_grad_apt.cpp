/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad.cpp
 * \brief repeat interleave grad
 */

#include "v35/repeat_interleave_grad_base.h"
#include "v35/repeat_interleave_grad_david.h"
#include "v35/repeat_interleave_grad_david_MN.h"
#include "v35/repeat_interleave_grad_n_1.h"
#include "v35/repeat_interleave_grad_david_tiling_data.h"
#include "v35/repeat_interleave_grad_block_split_r.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "v35/repeat_interleave_grad_dag.h"
#include "v35/repeat_interleave_grad_tiling_key.h"
#include "atvoss/reduce/reduce_sch.h"

using namespace AscendC;
using namespace RepeatInterleaveGrad;
using namespace Ops::Base::ReduceOpTmpl;

template <REDUCE_TPL_PARAM, uint32_t TemplateNum>
__global__ __aicore__ void repeat_interleave_grad(
    GM_ADDR input_grad, GM_ADDR repeats, GM_ADDR output_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (TemplateNum == RIG::IS_REDUCE_T) {
        // repeat is int
        if (g_coreType == AIC) {
            return;
        }
        if (workspace == nullptr) {
            return;
        }
        SetSysWorkspace(workspace);
        GM_ADDR userWS = GetUserWorkspace(workspace);
        if (userWS == nullptr) {
            return;
        }
        REGISTER_TILING_DEFAULT(RepeatInterleaveGradDavidTilingData);
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveGradDavidTilingData, tilingData, tiling);

        TPipe pipe;
        using PromoteType = __reduceType::GetPromoteType<DTYPE_Y_GRAD>::T;
        using Op = ReduceSch<REDUCE_TPL_VALUE, RIG::RIGDag<DTYPE_Y_GRAD, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.Init(&pipe, input_grad, output_grad, userWS);
        op.Process();
    } else if constexpr (TemplateNum == RIG::BLOCK_SPLIT_M) {
        using PromoteType = __RIGType::GetPromoteType<DTYPE_Y_GRAD>::T;

        REGISTER_TILING_DEFAULT(RepeatInterleaveGradDavidTilingData);
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveGradDavidTilingData, tilingData, tiling);
        TPipe pipe;
        RepeatInterleaveGradDavid<DTYPE_Y_GRAD, PromoteType, DTYPE_REPEATS> op(pipe);
        op.Init(input_grad, repeats, output_grad, workspace, &tilingData);
        op.Process();
    } else if constexpr (TemplateNum == RIG::BLOCK_SPLIT_M_N_1) {
        using PromoteType = __RIGType::GetPromoteType<DTYPE_Y_GRAD>::T;

        REGISTER_TILING_DEFAULT(RepeatInterleaveGradDavidTilingData);
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveGradDavidTilingData, tilingData, tiling);
        TPipe pipe;
        RepeatInterleaveGradDimNOneDavid<DTYPE_Y_GRAD, PromoteType, DTYPE_REPEATS> op(pipe);
        op.Init(input_grad, repeats, output_grad, workspace, &tilingData);
        op.Process();
    } else if constexpr (TemplateNum == RIG::BLOCK_SPLIT_MN) {
        using PromoteType = __RIGType::GetPromoteType<DTYPE_Y_GRAD>::T;

        REGISTER_TILING_DEFAULT(RepeatInterleaveGradDavidTilingData);
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveGradDavidTilingData, tilingData, tiling);
        TPipe pipe;
        RepeatInterleaveGradDavidMN<DTYPE_Y_GRAD, PromoteType, DTYPE_REPEATS> op(pipe);
        op.Init(input_grad, repeats, output_grad, workspace, &tilingData);
        op.Process();
    } else if constexpr (TemplateNum == RIG::BLOCK_SPLIT_R) {
        using PromoteType = __RIGType::GetPromoteType<DTYPE_Y_GRAD>::T;

        REGISTER_TILING_DEFAULT(RepeatInterleaveGradDavidTilingData);
        GET_TILING_DATA_WITH_STRUCT(RepeatInterleaveGradDavidTilingData, tilingData, tiling);
        TPipe pipe;
        RepeatInterleaveGradBlockSplitR<DTYPE_Y_GRAD, PromoteType, DTYPE_REPEATS> op(pipe);
        op.Init(input_grad, repeats, output_grad, workspace, &tilingData);
        op.Process();
    }
}
