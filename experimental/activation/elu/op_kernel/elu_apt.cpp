/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file elu_apt.cpp
 * \brief
 */

#include "arch35/elu_dag.h"
#include "arch35/elu_struct.h"
#include "arch35/elu_tiling_struct.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "../inc/platform.h"

using namespace EluNs;
using namespace Ops::Base;
using namespace AscendC;
using namespace EluOp;

namespace {
template <uint64_t schMode, typename InTy>
__aicore__ inline void RunEluSch(EluTilingData& tilingData, TPipe& pipe, GM_ADDR x, GM_ADDR y)
{
    ElementwiseSch<schMode, typename EluDag<InTy, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.template SetVar<float, ELU_ATTR_ALPHA_INDEX>(tilingData.alpha);
    sch.template SetVar<float, ELU_ATTR_SCALE_INDEX>(tilingData.scale);
    sch.template SetVar<float, ELU_ATTR_INPUT_SCALE_INDEX>(tilingData.inputScale);
    sch.Init(x, y);
    sch.Process();
}
} // namespace

template <uint64_t tilingKey>
__global__ __aicore__ void elu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(EluTilingData);
    GET_TILING_DATA_WITH_STRUCT(EluTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (tilingKey == static_cast<uint64_t>(ELU_TPL_KEY_FP16_SCH0)) {
        RunEluSch<ELU_TPL_SCH_MODE_0, half>(tilingData, pipe, x, y);
    } else if constexpr (tilingKey == static_cast<uint64_t>(ELU_TPL_KEY_FP16_SCH1)) {
        RunEluSch<ELU_TPL_SCH_MODE_1, half>(tilingData, pipe, x, y);
    } else if constexpr (tilingKey == static_cast<uint64_t>(ELU_TPL_KEY_BF16_SCH0)) {
        RunEluSch<ELU_TPL_SCH_MODE_0, bfloat16_t>(tilingData, pipe, x, y);
    } else if constexpr (tilingKey == static_cast<uint64_t>(ELU_TPL_KEY_BF16_SCH1)) {
        RunEluSch<ELU_TPL_SCH_MODE_1, bfloat16_t>(tilingData, pipe, x, y);
    } else if constexpr (tilingKey == static_cast<uint64_t>(ELU_TPL_KEY_FP32_SCH0)) {
        RunEluSch<ELU_TPL_SCH_MODE_0, float>(tilingData, pipe, x, y);
    } else if constexpr (tilingKey == static_cast<uint64_t>(ELU_TPL_KEY_FP32_SCH1)) {
        RunEluSch<ELU_TPL_SCH_MODE_1, float>(tilingData, pipe, x, y);
    }
}
