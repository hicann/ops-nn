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
 * \file conv2d_v2_instr_fz_impl.h
 * \brief
 */

#ifndef CONV2D_V2_INSTR_FZ_IMPL_H
#define CONV2D_V2_INSTR_FZ_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class LoadBL1FZTools {
public:
    __aicore__ inline LoadBL1FZTools() {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        maxNBL1Iter_ = self->ctx.maxNBL1Iter;
        maxKBL1Iter_ = self->ctx.maxKBL1Iter;
        orgCoAlignN0 = AlignB(self_->ctx.convTiling->orgCo, BLOCK_L0_N);
        if constexpr (Intf::groupOptNZFlag) {
            self->ctx.coPerGroup = self->ctx.convTiling->orgCo / self->ctx.convTiling->groups;
            self->ctx.coOpt = self->ctx.coPerGroup * self->ctx.convTiling->enlarge;
            orgCoAlignN0 = AlignB(self->ctx.coOpt, BLOCK_L0_N);
        } else {
            orgCoAlignN0 = AlignB(self_->ctx.convTiling->orgCo, BLOCK_L0_N);
        }
    }

    __aicore__ inline void LoadBL1()
    {
        LoadBL1(self_->ctx.kBL1Iter, self_->ctx.nBL1Iter);
    }

    __aicore__ inline void LoadBL1(uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        uint64_t bL1GmOffset;
        if constexpr (!Intf::hasNL1IterFlag) {
            bL1GmOffset = kBL1Iter * self_->ctx.convTiling->kBL1 * orgCoAlignN0;
        } else {
            bL1GmOffset = kBL1Iter * self_->ctx.convTiling->kBL1 * orgCoAlignN0 +
                          nBL1Iter * self_->ctx.convTiling->nBL1 * Intf::k0;
            self_->ctx.currentNBL1 = nBL1Iter == maxNBL1Iter_ ? self_->ctx.nBL1Tail : self_->ctx.convTiling->nBL1;
        }

        uint64_t currentKBL1 = self_->ctx.convTiling->kBL1;
        if constexpr (!Intf::c04Flag) {
            currentKBL1 = kBL1Iter == maxKBL1Iter_ ? self_->ctx.kBL1AlignK0Tail : self_->ctx.convTiling->kBL1;
        }
        DataCopyPadExtParams<typename Intf::WeightT> padParams;
        DataCopyExtParams dataCopyParams;
        if (orgCoAlignN0 == self_->ctx.currentNBL1) {
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = self_->ctx.currentNBL1 * currentKBL1 * Intf::sizeOfWeight;
            dataCopyParams.srcStride = 0;
        } else {
            dataCopyParams.blockCount = CeilDiv(currentKBL1, Intf::k0);
            dataCopyParams.blockLen = self_->ctx.currentNBL1 * C0_SIZE;
            dataCopyParams.srcStride = (orgCoAlignN0 - self_->ctx.currentNBL1) * C0_SIZE;
            dataCopyParams.dstStride = self_->ctx.convTiling->nBL1 - self_->ctx.currentNBL1;
        }
        DataCopyPad<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], dataCopyParams, padParams);
    }

private:
    Intf *self_ = nullptr;
    uint64_t maxNBL1Iter_ = UINT64_MAX;
    uint64_t maxKBL1Iter_ = UINT64_MAX;
    uint32_t orgCoAlignN0 = 0;
};

};

#endif // CONV2D_V2_INSTR_IMPL_H