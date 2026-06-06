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
 * \file kl_div_loss_grad.h
 * \brief
 */
#ifndef __KL_DIV_LOSS_GRAD_H__
#define __KL_DIV_LOSS_GRAD_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kl_div_loss_grad_tiling_data.h"
#include "kl_div_loss_grad_tiling_key.h"

namespace NsKlDivLossGrad {
template <class T, template <class> class Policy>
class KernelKlDivLossGrad {
    using PolicyType = Policy<float>;

public:
    __aicore__ inline KernelKlDivLossGrad() {}
    __aicore__ inline void Init(
        GM_ADDR grad, GM_ADDR input, GM_ADDR target, GM_ADDR y, uint32_t bigCoreDataNum, uint32_t smallCoreDataNum,
        uint32_t tileDataNum, uint32_t bigCoreNum, float coeff, AscendC::TPipe* pipe)
    {
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        if (AscendC::GetBlockIdx() < bigCoreNum) {
            this->coreDataNum = bigCoreDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - bigCoreNum);
        }
        if constexpr (!PolicyType::broadcast) {
            gradGm.SetGlobalBuffer((__gm__ T*)grad + globalBufferIndex, this->coreDataNum);
        } else {
            gradGm.SetGlobalBuffer((__gm__ T*)grad, 1);
        }
        targetGm.SetGlobalBuffer((__gm__ T*)target + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

        if constexpr (!PolicyType::broadcast) {
            pipe->InitBuffer(inQueGrad, 2, tileDataNum * sizeof(T));
            if constexpr (!std::is_same_v<T, float>) {
                pipe->InitBuffer(gradBuf, tileDataNum * sizeof(float));
            }
        }
        pipe->InitBuffer(inQueTarget, 2, tileDataNum * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe->InitBuffer(targetBuf, tileDataNum * sizeof(float));
        }
        pipe->InitBuffer(outQueY, 2, tileDataNum * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe->InitBuffer(yBuf, tileDataNum * sizeof(float));
        }

        this->tileDataNum = tileDataNum;
        if constexpr (PolicyType::broadcast) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                policy.grad = AscendC::ToFloat(gradGm.GetValue(0));
            } else {
                policy.grad = static_cast<float>(gradGm.GetValue(0));
            }
        }
        policy.negCoeff = -coeff;
    }
    __aicore__ inline void Process()
    {
        uint64_t coreDataNum = this->coreDataNum;
        uint64_t tileDataNum = this->tileDataNum;
        for (uint64_t offset = 0; offset < coreDataNum; offset += tileDataNum) {
            uint32_t processDataNum = AscendC::Std::min(tileDataNum, coreDataNum - offset);
            CopyIn(offset, processDataNum);
            Compute(processDataNum);
            CopyOut(offset, processDataNum);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t processDataNum)
    {
        AscendC::LocalTensor<T> targetLocal = inQueTarget.AllocTensor<T>();
        AscendC::DataCopy(targetLocal, targetGm[offset], processDataNum);
        inQueTarget.EnQue(targetLocal);
        if constexpr (!PolicyType::broadcast) {
            AscendC::LocalTensor<T> gradLocal = inQueGrad.AllocTensor<T>();
            AscendC::DataCopy(gradLocal, gradGm[offset], processDataNum);
            inQueGrad.EnQue(gradLocal);
        }
    }
    __aicore__ inline void Compute(uint32_t processDataNum)
    {
        AscendC::LocalTensor<T> yLocal = outQueY.AllocTensor<T>();
        AscendC::LocalTensor<T> targetLocal = inQueTarget.DeQue<T>();
        if constexpr (!std::is_same_v<T, float>) {
            AscendC::LocalTensor<float> targetFp32 = targetBuf.Get<float>();
            AscendC::LocalTensor<float> yFp32 = yBuf.Get<float>();
            AscendC::Cast(targetFp32, targetLocal, AscendC::RoundMode::CAST_NONE, processDataNum);
            if constexpr (!PolicyType::broadcast) {
                AscendC::LocalTensor<T> gradLocal = inQueGrad.DeQue<T>();
                AscendC::LocalTensor<float> gradFp32 = gradBuf.Get<float>();
                AscendC::Cast(gradFp32, gradLocal, AscendC::RoundMode::CAST_NONE, processDataNum);
                inQueGrad.FreeTensor(gradLocal);
                policy.compute(gradFp32, targetFp32, yFp32, processDataNum);
            } else {
                policy.compute(targetFp32, yFp32, processDataNum);
            }
            AscendC::Cast(yLocal, yFp32, AscendC::RoundMode::CAST_RINT, processDataNum);
        } else {
            if constexpr (!PolicyType::broadcast) {
                AscendC::LocalTensor<T> gradLocal = inQueGrad.DeQue<T>();
                policy.compute(gradLocal, targetLocal, yLocal, processDataNum);
                inQueGrad.FreeTensor(gradLocal);
            } else {
                policy.compute(targetLocal, yLocal, processDataNum);
            }
        }
        outQueY.EnQue(yLocal);
        inQueTarget.FreeTensor(targetLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t processDataNum)
    {
        AscendC::LocalTensor<T> yLocal = outQueY.DeQue<T>();
        AscendC::DataCopy(yGm[offset], yLocal, processDataNum);
        outQueY.FreeTensor(yLocal);
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueGrad, inQueTarget;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gradBuf, targetBuf, yBuf;
    AscendC::GlobalTensor<T> gradGm, targetGm;
    AscendC::GlobalTensor<T> yGm;

    uint32_t coreDataNum;
    uint32_t tileDataNum;

    PolicyType policy;
};

template <class T>
class ComputeImpl {
public:
    static constexpr bool broadcast = false;

    __aicore__ inline void compute(
        const AscendC::LocalTensor<T>& grad, const AscendC::LocalTensor<T>& target, const AscendC::LocalTensor<T>& y,
        uint32_t processDataNum) const
    {
        AscendC::Mul(y, grad, target, processDataNum);
        AscendC::Muls(y, y, negCoeff, processDataNum);
    }

    T negCoeff;
};

template <class T>
class ComputeImplLog {
public:
    static constexpr bool broadcast = false;

    __aicore__ inline void compute(
        const AscendC::LocalTensor<T>& grad, const AscendC::LocalTensor<T>& target, const AscendC::LocalTensor<T>& y,
        uint32_t processDataNum) const
    {
        AscendC::Exp(y, target, processDataNum);
        AscendC::Mul(target, y, grad, processDataNum);
        AscendC::Muls(y, target, negCoeff, processDataNum);
    }

    T negCoeff;
};

template <class T>
class ComputeImplBroadCast {
public:
    static constexpr bool broadcast = true;

    __aicore__ inline void compute(
        const AscendC::LocalTensor<T>& target, const AscendC::LocalTensor<T>& y, uint32_t processDataNum) const
    {
        AscendC::Muls(y, target, grad, processDataNum);
        AscendC::Muls(y, y, negCoeff, processDataNum);
    }

    T grad;
    T negCoeff;
};

template <class T>
class ComputeImplBroadCastLog {
public:
    static constexpr bool broadcast = true;

    __aicore__ inline void compute(
        const AscendC::LocalTensor<T>& target, const AscendC::LocalTensor<T>& y, uint32_t processDataNum) const
    {
        AscendC::Exp(y, target, processDataNum);
        AscendC::Muls(target, y, grad, processDataNum);
        AscendC::Muls(y, target, negCoeff, processDataNum);
    }

    T grad;
    T negCoeff;
};
} // namespace NsKlDivLossGrad
#endif // KL_DIV_LOSS_GRAD_H
