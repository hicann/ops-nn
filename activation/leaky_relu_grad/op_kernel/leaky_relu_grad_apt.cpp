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
 * \file leaky_relu_grad.cpp
 * \brief leaky_relu_grad
 */

 #include "kernel_operator.h"
 #include "arch35/leaky_relu_grad_dag.h"
 #include "arch35/leaky_relu_grad_struct.h"
 #include "atvoss/broadcast/broadcast_sch.h"
 
 using namespace AscendC;
 using namespace LeakyReluGradOp;
 
 template <uint64_t schMode, uint64_t dtype>
 __global__ __aicore__ void leaky_relu_grad(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace,
                                      GM_ADDR tiling)
 {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
     using OpDag = LeakyReluGradDag<DTYPE_GRADIENTS>::OpDag;
     BroadcastSch<schMode, OpDag> sch(tiling);
     sch.Process(gradients, features, backprops);
     return;
 }
 