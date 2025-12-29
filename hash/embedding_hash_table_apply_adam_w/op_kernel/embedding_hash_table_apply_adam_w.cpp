/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "arch35/embedding_hash_table_apply_adam_w_b16.h"
 #include "arch35/embedding_hash_table_apply_adam_w_b32.h"
 
 using namespace Hashtbl;
 namespace {
 #define BIT2WIDTH_TILING_KEY 102
 #define BIT4WIDTH_TILING_KEY 104
 }  // namespace
 
 extern "C" __global__ __aicore__ void embedding_hash_table_apply_adam_w(
     GM_ADDR tableIn, GM_ADDR keys, GM_ADDR m, GM_ADDR v, GM_ADDR beta1Power, GM_ADDR beta2Power, GM_ADDR lr,
     GM_ADDR weightDecay, GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad, GM_ADDR maxGradNorm, GM_ADDR mOut,
     GM_ADDR vOut, GM_ADDR beta1PowerOut, GM_ADDR beta2PowerOut, GM_ADDR maxGradNormOut,
     GM_ADDR workspace, GM_ADDR tiling) {
   KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); 
   if (workspace == nullptr) {
     return;
   }
   SetSysWorkspace(workspace);
   GM_ADDR userWS = GetUserWorkspace(workspace);
   if (userWS == nullptr) {
     return;
   }
   GET_TILING_DATA(tilingData, tiling);
 
   if (TILING_KEY_IS(BIT2WIDTH_TILING_KEY)) {
     KernelEmbeddingHashTableApplyAdamWB16<half> op;
     op.Init(tableIn, keys, m, v, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, epsilon, grad, maxGradNorm,
             mOut, vOut, beta1PowerOut, beta2PowerOut, maxGradNormOut, workspace, tilingData);
     op.Process();
     op.PostProcess(beta1PowerOut, beta2PowerOut);
   } else if (TILING_KEY_IS(BIT4WIDTH_TILING_KEY)) {
     KernelEmbeddingHashTableApplyAdamWB32<float> op;
     op.Init(tableIn, keys, m, v, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, epsilon, grad, maxGradNorm,
             mOut, vOut, beta1PowerOut, beta2PowerOut, maxGradNormOut, workspace, tilingData);
     op.Process();
     op.PostProcess(beta1PowerOut, beta2PowerOut);
   }
 }
 