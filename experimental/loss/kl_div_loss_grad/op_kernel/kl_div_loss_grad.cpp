/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kl_div_loss_grad.cpp
 * \brief
 */

#include "kl_div_loss_grad.h"

using namespace NsKlDivLossGrad;

template <bool logTarget, bool broadcast>
struct PolicySelector;

template <>
struct PolicySelector<false, false> {
    template <class T>
    using type = ComputeImpl<T>;
};

template <>
struct PolicySelector<true, false> {
    template <class T>
    using type = ComputeImplLog<T>;
};

template <>
struct PolicySelector<false, true> {
    template <class T>
    using type = ComputeImplBroadCast<T>;
};

template <>
struct PolicySelector<true, true> {
    template <class T>
    using type = ComputeImplBroadCastLog<T>;
};

template <bool logTarget, bool broadcast>
__global__ __aicore__ void kl_div_loss_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR target, GM_ADDR y, GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KlDivLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(KlDivLossGradTilingData, tilingData, tiling);
    KernelKlDivLossGrad<DTYPE_GRAD, PolicySelector<logTarget, broadcast>::template type> op;
    AscendC::TPipe pipe;
    op.Init(grad, input, target, y, tilingData.bigCoreDataNum, tilingData.smallCoreDataNum, tilingData.tileDataNum,
            tilingData.bigCoreNum, tilingData.coff, &pipe);
    op.Process();
}
