/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_apt.cpp
 * \brief UpdateTensorDesc kernel 入口（非模板，tilingKey 恒 0）。
 *   x 为占位输入（不读取），y 为 128×int64 描述缓冲区 RMW 对象，
 *   workspace 恒 0（形参按框架契约保留，不传入 kernel）。
 */

#include "kernel_operator.h"
#include "arch35/update_tensor_desc_kernel.h"
#include "arch35/update_tensor_desc_tiling_data.h"

__global__ __aicore__ void update_tensor_desc(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GET_TILING_DATA_WITH_STRUCT(UpdateTensorDescTilingData, td, tiling);

    AscendC::TPipe pipe;
    UpdateTensorDescKernel kernel;
    kernel.Init(x, y, &td, &pipe);
    kernel.Process();

    (void)workspace;
}
