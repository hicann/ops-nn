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
 * \file reduce_buf_pool.h
 * \brief class ReduceBufPool
 */

#ifndef _REDUCE_BUF_POOL_H_
#define _REDUCE_BUF_POOL_H_

#include "kernel_operator.h"
#include "platform.h"
#include "kernel_utils.h"

namespace RepeatInterleaveGrad {
struct PoolManagerUnit {
    int32_t idx = -1;
    int32_t eleSize = 0;
    int32_t bufNum = 0;
    int32_t offset = 0;
};

class ReduceBufPool
{
    constexpr static int32_t MAX_INPUT_SIZE = 10;

private:
    bool memo_[MAX_INPUT_SIZE] = {0};
    PoolManagerUnit inputUnit_;
    PoolManagerUnit computeUnit_;
    event_t eventIdV2Mte2_[MAX_INPUT_SIZE];
    TBuf<> qQue_;
    TPipe* pipe_;
    int32_t basicNum_;

public:
    __aicore__ inline ReduceBufPool(){};

    template <class DataType, class PromoteDataType>
    __aicore__ inline void Init(TPipe* pipeIn, int32_t inputNum, int32_t computeNum, int32_t basicBlockLen)
    {
        pipe_ = pipeIn;
        int32_t inputEleSize = sizeof(DataType);
        int32_t computeEleSize = sizeof(PromoteDataType);
        basicNum_ = basicBlockLen / sizeof(DataType);
        int32_t poolSize = basicNum_ * inputEleSize * inputNum + basicNum_ * computeEleSize * computeNum;
        inputUnit_.bufNum = inputNum;
        inputUnit_.eleSize = inputEleSize;
        inputUnit_.offset = 0;
        computeUnit_.bufNum = computeNum;
        computeUnit_.eleSize = computeEleSize;
        computeUnit_.offset = basicNum_ * sizeof(DataType) * inputNum;

        // Init buffer
        pipe_->InitBuffer(qQue_, poolSize);
        LocalTensor<DataType> inputUb = qQue_.GetWithOffset<DataType>(basicNum_ * inputNum, 0);

        // Init event
        for (uint32_t i = 0; i < inputNum; i++) {
            eventIdV2Mte2_[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        }
    }

    __aicore__ inline void ResetEvent()
    {
        for (uint32_t i = 0; i < inputUnit_.bufNum; i++) {
            if (memo_[i]) {
                WaitFlag<HardEvent::V_MTE2>(eventIdV2Mte2_[i]);
            }
            memo_[i] = false;
        }
        inputUnit_.idx = -1;
        computeUnit_.idx = -1;
    }

    template <typename T>
    __aicore__ inline void ResetInputSize(int32_t inputNum)
    {
        inputUnit_.bufNum = inputNum;
        computeUnit_.offset = basicNum_ * sizeof(T) * inputNum;
    }

    __aicore__ inline void ResetComputeSize(int32_t computeNum)
    {
        computeUnit_.bufNum = computeNum;
    }

    __aicore__ inline void DeInit()
    {
        for (uint32_t i = 0; i < inputUnit_.bufNum; i++) {
            if (memo_[i]) {
                WaitFlag<HardEvent::V_MTE2>(eventIdV2Mte2_[i]);
            }
            memo_[i] = false;
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdV2Mte2_[i]);
        }
    }

    template <bool IsInput, typename T>
    __aicore__ inline const void AllocTensor(LocalTensor<T>& tensor)
    {
        if constexpr (IsInput) {
            int32_t idx = GetInputTensorId();
            if (memo_[idx]) {
                WaitFlag<HardEvent::V_MTE2>(eventIdV2Mte2_[idx]);
            }
            memo_[idx] = true;
            tensor = qQue_.GetWithOffset<T>(basicNum_, inputUnit_.offset + idx * basicNum_ * sizeof(T));
        } else {
            int32_t idx = GetComputeTensorId();
            tensor = qQue_.GetWithOffset<T>(basicNum_, computeUnit_.offset + idx * basicNum_ * sizeof(T));
        }
    }

    template <typename T>
    __aicore__ inline void FreeTensor(LocalTensor<T>& tensor)
    {
        LocalTensor<T> tmpBuf = qQue_.GetWithOffset<T>(basicNum_, 0);
        uint64_t start = (uint64_t)(tmpBuf.GetPhyAddr());
        uint64_t offset = (uint64_t)(tensor.GetPhyAddr());
        if (offset - start < computeUnit_.offset) {
            int32_t idx = (offset - start) / sizeof(T) / basicNum_;
            SetFlag<HardEvent::V_MTE2>(eventIdV2Mte2_[idx]);
        }
    }

private:
    __aicore__ inline int32_t GetComputeTensorId()
    {
        computeUnit_.idx = (computeUnit_.idx + 1) % computeUnit_.bufNum;
        return computeUnit_.idx;
    }

    __aicore__ inline int32_t GetInputTensorId()
    {
        inputUnit_.idx = (inputUnit_.idx + 1) % inputUnit_.bufNum;
        return inputUnit_.idx;
    }
}; // class ReduceBufPool
} // namespace RepeatInterleaveGrad
#endif