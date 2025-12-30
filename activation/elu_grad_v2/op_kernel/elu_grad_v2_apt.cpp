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
 * \file elu_grad_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./arch35/elu_grad_v2_dag.h"
#include "./arch35/elu_grad_v2_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "elu_grad_v2_tiling_struct.h"

using namespace Ops::Base;
using namespace EluGradV2Op;
using namespace EluGradV2Ns;

template<uint64_t schMode, uint64_t dType>
__global__ __aicore__ void elu_grad_v2(GM_ADDR grads, GM_ADDR activations, GM_ADDR y,
                            GM_ADDR workspace, GM_ADDR tiling) 
{
  REGISTER_TILING_DEFAULT(EluGradV2TilingData);
  GET_TILING_DATA_WITH_STRUCT(EluGradV2TilingData, tilingData, tiling);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  if constexpr (dType == static_cast<uint64_t>(EluGradV2_TPL_FP16)) {
    ElementwiseSch<schMode, EluGradV2IsResultOp<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.template SetVar<float, 0>(tilingData.negcoef);
    sch.template SetVar<float, 1>(tilingData.scale);
    sch.template SetVar<float, PLACEHOLDER_INDEX_2>(tilingData.inputScale);
    sch.Init(grads, activations, y);
    sch.Process();
    return;
  } else if constexpr (dType == static_cast<uint64_t>(EluGradV2_TPL_BF16)) {
    ElementwiseSch<schMode, EluGradV2IsResultOp<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.template SetVar<float, 0>(tilingData.negcoef);
    sch.template SetVar<float, 1>(tilingData.scale);
    sch.template SetVar<float, PLACEHOLDER_INDEX_2>(tilingData.inputScale);
    sch.Init(grads, activations, y);
    sch.Process();
    return;
  } else if constexpr (dType == static_cast<uint64_t>(EluGradV2_TPL_FP32)) {
    ElementwiseSch<schMode, EluGradV2IsResultOp<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.template SetVar<float, 0>(tilingData.negcoef);
    sch.template SetVar<float, 1>(tilingData.scale);
    sch.template SetVar<float, PLACEHOLDER_INDEX_2>(tilingData.inputScale);
    sch.Init(grads, activations, y);
    sch.Process();
    return;
  } else if constexpr (dType == static_cast<uint64_t>(EluGradV2_TPL_FP16_N)) {
    ElementwiseSch<schMode, EluGradV2NoResultOp<half>::OpDag> sch(&(tilingData.baseTiling), &pipe); 
    sch.template SetVar<float, 0>(tilingData.negcoef);
    sch.template SetVar<float, 1>(tilingData.scale);
    sch.template SetVar<float, PLACEHOLDER_INDEX_2>(tilingData.inputScale);
    sch.Init(grads, activations, y);
    sch.Process();
    return;
  } else if constexpr (dType == static_cast<uint64_t>(EluGradV2_TPL_BF16_N)) {
    ElementwiseSch<schMode, EluGradV2NoResultOp<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe); 
    sch.template SetVar<float, 0>(tilingData.negcoef);
    sch.template SetVar<float, 1>(tilingData.scale);
    sch.template SetVar<float, PLACEHOLDER_INDEX_2>(tilingData.inputScale);
    sch.Init(grads, activations, y);
    sch.Process();
    return;
  } else if constexpr (dType == static_cast<uint64_t>(EluGradV2_TPL_FP32_N)) {
    ElementwiseSch<schMode, EluGradV2NoResultOp<float>::OpDag> sch(&(tilingData.baseTiling), &pipe); 
    sch.template SetVar<float, 0>(tilingData.negcoef);
    sch.template SetVar<float, 1>(tilingData.scale);
    sch.template SetVar<float, PLACEHOLDER_INDEX_2>(tilingData.inputScale);
    sch.Init(grads, activations, y);
    sch.Process();
    return;
  }
  return;
}