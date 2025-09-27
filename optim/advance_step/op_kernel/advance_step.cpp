/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file advance_step.cpp
 * \brief
 */

#include "advance_step.h"
#include "advance_step_spec.h"
using namespace AscendC;
using namespace AdvanceStepNs;

extern "C" __global__ __aicore__ void advance_step(
    GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
    GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1)) {
        KernelAdvanceStep<int64_t> op;
        op.Init(
            input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, userWs,
            &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        TPipe Pipe;
        KernelAdvanceStepSpec<int64_t> op;
        op.Init(
            input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, spec_token,
            accepted_num, userWs, &tilingData, &Pipe);
        op.Process();
    }

#endif
}