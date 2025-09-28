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
 * \file mse_loss_v2_mean.h
 * \brief
 */

#ifndef MSE_LOSS_V2_MEAN_H
#define MSE_LOSS_V2_MEAN_H

#include "mse_loss_v2_sum.h"

namespace MSELossV2Namespace {
template <typename T>
class MSELossV2Mean : public MSELossV2Sum<T>
{
public:
    __aicore__ inline MSELossV2Mean(AscendC::TPipe* pipe, const MSELossV2TilingData* tilingData)
        : MSELossV2Sum<T>(pipe, tilingData), scale(tilingData->scale)
    {}

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> cast0Local = this->castBuf0.template Get<float>();
        AscendC::LocalTensor<float> cast1Local = this->castBuf1.template Get<float>();

        for (uint64_t i = 0u; i < this->epochs; i++) {
            MSELossV2Base<T>::CopyIn(cast0Local, cast1Local, i * this->tileLength, this->tileLength);
            this->Compute(cast0Local, cast1Local, this->tileLength);
        }

        if (this->isLastCore && this->tailElems) {
            MSELossV2Base<T>::CopyIn(
                cast0Local, cast1Local, this->epochs * this->tileLength,
                this->tailTileLength + MSELossV2Base<T>::FLOAT_COUNT_PER_BLOCK);
            this->Compute(cast0Local, cast1Local, this->tailTileLength + this->tailElems);
        } else if (this->tailTileLength) {
            MSELossV2Base<T>::CopyIn(cast0Local, cast1Local, this->epochs * this->tileLength, this->tailTileLength);
            this->Compute(cast0Local, cast1Local, this->tailTileLength);
        }

        this->CopyOut();
    }

private:
    __aicore__ inline void Compute(
        const AscendC::LocalTensor<float>& cast0, const AscendC::LocalTensor<float>& cast1, uint64_t tileLength)
    {
        AscendC::Sub<float>(cast0, cast0, cast1, tileLength);
        AscendC::Mul<float>(cast1, cast0, cast0, tileLength);
        this->ReduceSumBisect(cast1, tileLength, this->scale);
    }

private:
    float scale = 0.f;
};

template <>
class MSELossV2Mean<float> : public MSELossV2Sum<float>
{
public:
    __aicore__ inline MSELossV2Mean(AscendC::TPipe* pipe, const MSELossV2TilingData* tilingData)
        : MSELossV2Sum<float>(pipe, tilingData), scale(tilingData->scale)
    {}

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0u; i < this->epochs; i++) {
            MSELossV2Base<float>::CopyIn(i * this->tileLength, this->tileLength);
            this->Compute(this->tileLength);
        }

        if (this->isLastCore && this->tailElems) {
            MSELossV2Base<float>::CopyIn(
                this->epochs * this->tileLength, this->tailTileLength + MSELossV2Base<float>::FLOAT_COUNT_PER_BLOCK);
            this->Compute(this->tailTileLength + this->tailElems);
        } else if (this->tailTileLength) {
            MSELossV2Base<float>::CopyIn(this->epochs * this->tileLength, this->tailTileLength);
            this->Compute(this->tailTileLength);
        }

        this->CopyOut();
    }

private:
    __aicore__ inline void Compute(uint64_t tileLength)
    {
        AscendC::LocalTensor<float> src0Local = this->inQue0.template DeQue<float>();
        AscendC::LocalTensor<float> src1Local = this->inQue1.template DeQue<float>();

        AscendC::Sub<float>(src0Local, src0Local, src1Local, tileLength);
        AscendC::Mul<float>(src1Local, src0Local, src0Local, tileLength);
        this->ReduceSumBisect(src1Local, tileLength, this->scale);

        this->inQue0.template FreeTensor<float>(src0Local);
        this->inQue1.template FreeTensor<float>(src1Local);
    }

private:
    float scale = 0.f;
};
} // namespace MSELossV2Namespace

#endif