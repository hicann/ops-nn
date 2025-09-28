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
 * \file quant_matmul_reduce_sum_init_output.h
 * \brief
 */
#ifndef QUANT_MATMUL_REDUCE_SUM_INIT_OUTPUT_H
#define QUANT_MATMUL_REDUCE_SUM_INIT_OUTPUT_H

#include "quant_matmul_reduce_sum_utils.h"

namespace AscendC {
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
  if (unlikely(b == 0)) {
    return a;
  }
  return (a + b - 1) / b;
}

template <typename yType>
class QuantMatmulReduceSumInitOutput
{
public:
    __aicore__ inline QuantMatmulReduceSumInitOutput()
    {}
    __aicore__ inline void Init(
        GM_ADDR y, GM_ADDR workSpace, const QuantMatmulReduceSumTilingData* tilingData, TPipe* tPipe)
    {
        InitTilingData(tilingData);
        // init global buffer
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        tPipe->InitBuffer(localBuffer_, TOTAL_L1_SIZE);
    }

    /** main logical function
     */
    __aicore__ inline void Process()
    {
        uint32_t usedCoreNum = GetBlockNum();
        uint64_t clearSizePerCore = AlignUp(CeilDiv(outputSize_, usedCoreNum), MIN_CLEAR_SIZE);
        usedCoreNum = CeilDiv(outputSize_, clearSizePerCore);
        if (GetBlockIdx() >= usedCoreNum) {
            SyncAllCores();
            return;
        }

        LocalTensor<yType> tmpBuf = localBuffer_.Get<yType>();

        uint64_t dstOffset = clearSizePerCore * GetBlockIdx();
        uint64_t realClearSize = QUANT_MATMUL_REDUCE_SUM::Min<uint64_t>(outputSize_ - dstOffset, clearSizePerCore);
        uint64_t clearBlkNum = realClearSize * sizeof(yType) / ONE_BLK_SIZE;

        DataCopyParams dataCopyParams(1, 1, 0, 0);
        if (unlikely(clearBlkNum == 0)) {
            ClearLocalTensor(tmpBuf, ONE_BLK_ITEM_NUM);
            // 只支持32B对齐搬出，这里尽量避免越界，往前凑。如果是小于32B的极小case，那只能越界了
            if (dstOffset > ONE_BLK_ITEM_NUM - realClearSize) {
                dstOffset -= (ONE_BLK_ITEM_NUM - realClearSize);
            }

            DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
        } else {
            dataCopyParams.blockLen =
                QUANT_MATMUL_REDUCE_SUM::Min<uint64_t>(tmpBuf.GetSize() * sizeof(yType) / ONE_BLK_SIZE, clearBlkNum);
            uint64_t burstItemNum = dataCopyParams.blockLen * ONE_BLK_ITEM_NUM;
            ClearLocalTensor(tmpBuf, burstItemNum);
            uint64_t loop = clearBlkNum / dataCopyParams.blockLen;
            for (uint64_t idx = 0; idx < loop; ++idx) {
                DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
                dstOffset += burstItemNum;
            }

            dataCopyParams.blockLen = clearBlkNum % dataCopyParams.blockLen;
            if (dataCopyParams.blockLen > 0) {
                DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
                dstOffset += (dataCopyParams.blockLen * ONE_BLK_ITEM_NUM);
            }

            uint64_t tailClearSize = realClearSize - clearBlkNum * ONE_BLK_ITEM_NUM;
            if (tailClearSize > 0) {
                dataCopyParams.blockLen = 1;
                // 同前，尽量避免越界，往前凑
                if (dstOffset > ONE_BLK_ITEM_NUM - tailClearSize) {
                    dstOffset -= (ONE_BLK_ITEM_NUM - tailClearSize);
                }

                DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
            }
        }

        SyncAllCores();
    }

    __aicore__ inline void ClearLocalTensor(LocalTensor<yType>& tmpBuf, uint32_t size)
    {
        uint16_t blockNum = size / ONE_BLK_ITEM_NUM;
        InitConstValue(tmpBuf, {1, blockNum, 0, 0U});

        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE3>();
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
    }

    __aicore__ inline void SyncAllCores()
    {
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIC_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_FLAG);
    }

private:
    uint64_t outputSize_;
    GlobalTensor<yType> yGm_;
    TBuf<TPosition::TSCM> localBuffer_;

    static constexpr uint64_t MIN_CLEAR_BYTE = 512UL;
    static constexpr uint64_t MIN_CLEAR_SIZE = MIN_CLEAR_BYTE / sizeof(yType);
    static constexpr uint64_t ONE_BLK_ITEM_NUM = ONE_BLK_SIZE / sizeof(yType);

    __aicore__ inline void InitTilingData(const QuantMatmulReduceSumTilingData* tilingData)
    {
        uint32_t mSize = tilingData->matmulTiling.M;
        uint32_t nSize = tilingData->matmulTiling.N;
        outputSize_ = mSize * nSize;
    }
};
} // namespace AscendC

#endif // QUANT_MATMUL_REDUCE_SUM_INIT_OUTPUT_H
