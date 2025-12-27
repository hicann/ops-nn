/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file ctc_loss_v2_base.h
 * \brief ctc_loss_v2_base
 */

#ifndef CTC_LOSS_V2_BASE
#define CTC_LOSS_V2_BASE

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctc_loss_v2_tiling_key.h"

#define INFINITY (__builtin_inff())
#define BSALIGNSIZE 32
using namespace AscendC;

namespace CTCLossV2 {
constexpr int64_t THREAD_NUM_1024 = 1024;
constexpr int64_t THREAD_NUM_512 = 512;
constexpr float neginf = -INFINITY;
template <typename T, typename DataType, typename ThreadType>
class CTCLossV2Base {
public:
    __aicore__ inline CTCLossV2Base(){};
    __aicore__ inline void BaseInit(
        GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths, GM_ADDR neg_log_likelihood,
        GM_ADDR log_alpha, GM_ADDR workspace, const CTCLossV2TilingData4AscendC* tilingData)
    {
        tdPtr = tilingData;
        logProbsDataGm.SetGlobalBuffer((__gm__ T*)(log_probs));
        targetsDataGm.SetGlobalBuffer((__gm__ DataType*)(targets));
        inputLengthsGm.SetGlobalBuffer((__gm__ DataType*)(input_lengths));
        targetLengthsGm.SetGlobalBuffer((__gm__ DataType*)(target_lengths));
        negLogLikelihoodDataGm.SetGlobalBuffer((__gm__ T*)(neg_log_likelihood));
        logAlphaDataGm.SetGlobalBuffer((__gm__ T*)(log_alpha));
        DataType batchSize = tdPtr->batchSize;
        DataType laBatchStride = tdPtr->laBatchStride;
        DataType laInputStride = tdPtr->laInputStride;
        DataType laGmSize = batchSize * laBatchStride;
        uint32_t blkIdx = GetBlockIdx();
        uint32_t blkNum = GetBlockNum();
        uint32_t singleCoreSize = (blkNum == 0) ? 0 : (laGmSize + blkNum - 1) / blkNum;
        int64_t tailSize = laGmSize - blkIdx * singleCoreSize;
        if (tailSize > 0) {
            int64_t singleCoreSz = tailSize < singleCoreSize ? tailSize : singleCoreSize;
            InitOutput<T>(logAlphaDataGm[blkIdx * singleCoreSize], singleCoreSz, neginf);
        }
        SyncAll();
    };

    __aicore__ inline void MTE2ToSSync()
    {
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    };

protected:
    GlobalTensor<T> logAlphaDataGm;
    GlobalTensor<T> logProbsDataGm;
    GlobalTensor<DataType> inputLengthsGm;
    GlobalTensor<DataType> targetsDataGm;
    GlobalTensor<DataType> targetLengthsGm;
    GlobalTensor<T> negLogLikelihoodDataGm;
    const CTCLossV2TilingData4AscendC* tdPtr = nullptr;
};
} // namespace CTCLossV2
#endif // CTC_LOSS_V2_BASE