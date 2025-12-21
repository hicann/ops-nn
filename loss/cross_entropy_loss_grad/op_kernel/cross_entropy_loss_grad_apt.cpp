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
 * \file cross_entropy_loss_grad_apt.cpp
 * \brief cross_entropy_loss_grad_apt
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/cross_entropy_loss_grad_tiling_key.h"
#include "arch35/cross_entropy_loss_grad_tiling_data.h"
#include "arch35/cross_entropy_loss_grad_weight_not_none.h"
#include "arch35/cross_entropy_loss_grad_weight_none.h"
#include "arch35/cross_entropy_loss_grad_full_load.h"

using namespace AscendC;
using namespace CrossEntropyLossGradRegbase;

template <uint64_t schId, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__global__ __aicore__ void cross_entropy_loss_grad(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                              GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                              GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  REGISTER_TILING_DEFAULT(CrossEntropyLossGradRegbaseTilingData);
  GET_TILING_DATA_WITH_STRUCT(CrossEntropyLossGradRegbaseTilingData, tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

  if constexpr (schId == 0 && isWeight == 0) {
    CrossEntropyLossGradWeightNone<DTYPE_GRAD_LOSS, DTYPE_TARGET> CrossEntropyLossGradWeightNoneOp;
    CrossEntropyLossGradWeightNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNoneOp.Process();
  }
  else if constexpr (schId == 0 && isWeight == 1) {
    CrossEntropyLossGradWeightNotNone<DTYPE_GRAD_LOSS, DTYPE_TARGET> CrossEntropyLossGradWeightNotNoneOp;
    CrossEntropyLossGradWeightNotNoneOp.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    CrossEntropyLossGradWeightNotNoneOp.Process();
  }
  else if constexpr (schId == 1) {
    CELossGradFullLoad<DTYPE_GRAD_LOSS, DTYPE_TARGET, reductionKey, isWeight, labelS, isIgnore> op;
    op.Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tilingData);
    op.Process();
  } else {
    return;
  }
}