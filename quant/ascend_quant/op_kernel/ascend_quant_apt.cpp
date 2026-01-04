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
 * \file ascend_quant_apt.cpp
 * \brief ascend_quant kernel enter
 */

#include "kernel_operator.h"
#include "arch35/ascend_quant_regbase.h"
#include "arch35/ascend_quant_struct.h"
#include "arch35/ascend_quant_tilingdata.h"

using namespace AscendC;
using namespace AscendQuantOp;
template <uint64_t RoundMode, uint64_t ExtraBit>
__global__ __aicore__ void ascend_quant(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AscendQuantTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendQuantPerTensorRegbase<DTYPE_X, DTYPE_Y, RoundMode> op;
    op.Init(x, y, &tilingData);
    op.Process();
}