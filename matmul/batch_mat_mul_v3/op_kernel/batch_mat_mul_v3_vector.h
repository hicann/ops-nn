/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file batch_mat_mul_v3_vector.h
 * \brief vector kernel for batch_mat_mul_v3, used when B matrix last dim is 1, actual shape [1, k]
 * forcefully transposed before entering this kernel
 */
#ifndef BATCH_MAT_MUL_V3_VECTOR_H
#define BATCH_MAT_MUL_V3_VECTOR_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

template <typename A_TYPE, typename B_TYPE, typename C_TYPE>
class BatchMatmulVectorKernel {
public:
    __aicore__ inline BatchMatmulVectorKernel() {}
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;

    TBuf<TPosition::VECIN> aBuf, bBuf;
    TBuf<TPosition::VECOUT> cBuf;
    TBuf<TPosition::VECCALC> sharedTmpBuf, mulBuf;
    GlobalTensor<A_T> aGm;
    GlobalTensor<B_T> bGm;
    GlobalTensor<C_T> cGm;
    uint64_t coreNumber;
    uint64_t coreData;
    uint64_t rowsPerCore;
    uint64_t aTotalSize;
    uint64_t bTotalSize;
    uint64_t dimSizeSecondLast;
    uint64_t dimSizeLast;
    uint64_t alignedDimSizeLast;
    uint64_t bRowsPerBatch;
    uint64_t cTotalSize;
    LocalTensor<A_T> aUb;
    LocalTensor<B_T> bUb;
    LocalTensor<C_T> mulUb;
    LocalTensor<C_T> cUb;
    LocalTensor<C_T> sharedTmpUb;

    DataCopyPadExtParams<A_T> padParamsA;
    DataCopyPadExtParams<B_T> padParamsB;

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const BatchMatmulTilingData* tilingData, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->coreNumber = tilingData->vectorTilingInfo.coreNumber;
        this->coreData = tilingData->vectorTilingInfo.coreData;
        this->rowsPerCore = tilingData->vectorTilingInfo.rowsPerCore;
        this->aTotalSize = tilingData->vectorTilingInfo.aTotalSize;
        this->bTotalSize = tilingData->vectorTilingInfo.bTotalSize;
        this->cTotalSize = tilingData->vectorTilingInfo.cTotalSize;
        this->dimSizeSecondLast = tilingData->vectorTilingInfo.dimSizeSecondLast;
        this->dimSizeLast = tilingData->vectorTilingInfo.dimSizeLast;
        this->alignedDimSizeLast = tilingData->vectorTilingInfo.alignedDimSizeLast;
        this->bRowsPerBatch = tilingData->vectorTilingInfo.bRowsPerBatch;

        uint8_t rightPad = static_cast<uint8_t>(this->alignedDimSizeLast - this->dimSizeLast);
        this->padParamsA = DataCopyPadExtParams<A_T>{true, 0, rightPad, static_cast<A_T>(0)};
        this->padParamsB = DataCopyPadExtParams<B_T>{true, 0, rightPad, static_cast<B_T>(0)};

        aGm.SetGlobalBuffer((__gm__ A_T*)aGM, this->aTotalSize);
        bGm.SetGlobalBuffer((__gm__ B_T*)bGM, this->bTotalSize);
        cGm.SetGlobalBuffer((__gm__ C_T*)cGM, this->cTotalSize);

        pipe->InitBuffer(aBuf, this->rowsPerCore * alignedDimSizeLast * sizeof(A_T));
        pipe->InitBuffer(bBuf, this->rowsPerCore * alignedDimSizeLast * sizeof(B_T));
        pipe->InitBuffer(mulBuf, this->rowsPerCore * alignedDimSizeLast * sizeof(C_T));
        pipe->InitBuffer(sharedTmpBuf, this->rowsPerCore * alignedDimSizeLast * sizeof(C_T));
        pipe->InitBuffer(cBuf, this->rowsPerCore * sizeof(C_T));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        if (coreId >= this->coreNumber) {
            return;
        }
        uint64_t startAddress = coreId * this->coreData;
        if (startAddress >= this->cTotalSize) {
            return;
        }
        uint64_t totalRows = this->coreData;
        // 最后一个核处理剩余行
        if (coreId == this->coreNumber - 1) {
            totalRows = this->cTotalSize - startAddress;
        }

        // 按照rowsPerCore分块处理，每次迭代计算rowsPerCore行，即ub可操作的最大行数
        uint64_t fullIters = totalRows / this->rowsPerCore;
        uint64_t tailRows = totalRows % this->rowsPerCore;
        for (uint64_t i = 0; i < fullIters; i++) {
            uint64_t address = startAddress + i * this->rowsPerCore;
            indicesCompute(this->rowsPerCore, address);
        }
        // 处理尾块计算
        if (tailRows != 0) {
            uint64_t address = startAddress + fullIters * this->rowsPerCore;
            indicesCompute(tailRows, address);
        }
    }

private:
    __aicore__ inline void indicesCompute(int32_t tensorSize, uint64_t address)
    {
        aUb = aBuf.Get<A_T>();
        bUb = bBuf.Get<B_T>();
        mulUb = mulBuf.Get<C_T>();
        cUb = cBuf.Get<C_T>();
        sharedTmpUb = sharedTmpBuf.Get<C_T>();

        // A: 一条DataCopyPad拷贝所有行
        // rightPadding已将每个block扩展到alignedDimSizeLast，dstStride=0即可
        DataCopyExtParams copyParamsA{1, (uint32_t)(dimSizeLast * sizeof(A_T)), 0, 0, 0};
        copyParamsA.blockCount = tensorSize;
        DataCopyPad(aUb, aGm[address * dimSizeLast], copyParamsA, padParamsA);

        CopyAndBroadcastB(tensorSize, address);
        BmmByVec(tensorSize, address);
    }

    /**
     * @brief 复制并广播B向量: (1, dimSizeLast) -> (tensorSize, dimSizeLast)
     *
     * @param tensorSize 单核内矩阵行数
     * @param address 矩阵起始地址
     */
    __aicore__ inline void CopyAndBroadcastB(int32_t tensorSize, uint64_t address)
    {
        // 计算需要读取的B矩阵batch范围
        uint64_t firstBatchIdx = address / dimSizeSecondLast;
        uint64_t lastBatchIdx = (address + tensorSize - 1) / dimSizeSecondLast;
        int32_t numBatches = static_cast<int32_t>(lastBatchIdx - firstBatchIdx + 1);

        // Step 1: 一次性从GM拷贝所有B行到mulUb临时空间
        // B数据在GM中连续存储，每个batch一行(dimSizeLast个元素)
        DataCopyExtParams copyParamsB{static_cast<uint16_t>(numBatches), (uint32_t)(dimSizeLast * sizeof(B_T)), 0, 0,
                                      0};
        auto tempUb = mulUb.template ReinterpretCast<B_T>();
        DataCopyPad(tempUb, bGm[firstBatchIdx * bRowsPerBatch], copyParamsB, padParamsB);
        PipeBarrier<PIPE_ALL>();

        // Step 2: 从临时空间广播到bUb目标位置
        // 负责广播每个batch的单行到bUb目标位置
        // BLOCK_BYTE_SIZE = 32
        constexpr uint32_t elementsPerBlock = BLOCK_BYTE_SIZE / sizeof(B_T);
        for (int32_t i = 0; i < tensorSize;) {
            uint64_t curAddress = address + i;
            uint64_t bBatchIdx = curAddress / dimSizeSecondLast;
            uint64_t batchEndRow = (bBatchIdx + 1) * dimSizeSecondLast;
            int32_t batchCount = (tensorSize - i < static_cast<int32_t>(batchEndRow - address - i)) ?
                                     (tensorSize - i) :
                                     static_cast<int32_t>(batchEndRow - address - i);
            int32_t localBatchIdx = static_cast<int32_t>(bBatchIdx - firstBatchIdx);

            // 广播: 从tempUb[localBatchIdx行] 到 bUb[i行..i+batchCount-1行]
            uint64_t mask = alignedDimSizeLast;
            CopyRepeatParams copyRepeatParams;
            copyRepeatParams.srcStride = 0;
            copyRepeatParams.dstStride = 0;
            copyRepeatParams.srcRepeatSize = 0; // 源不前进
            copyRepeatParams.dstRepeatSize = alignedDimSizeLast / elementsPerBlock;
            Copy(bUb[i * alignedDimSizeLast], tempUb[localBatchIdx * alignedDimSizeLast], mask,
                 static_cast<uint8_t>(batchCount), copyRepeatParams);
            i += batchCount;
        }
        PipeBarrier<PIPE_ALL>();
    }

    /**
     * @brief 矩阵乘法: vector核内进行逐元素相乘，然后沿dimSizeLast维度进行归一化计算reduce sum
     *
     * @param tensorSize 单核内矩阵行数
     * @param address 矩阵起始地址
     */
    __aicore__ inline void BmmByVec(int32_t tensorSize, uint64_t address)
    {
        // 1. 逐元素相乘
        Mul(mulUb, aUb, bUb, tensorSize * alignedDimSizeLast);
        PipeBarrier<PIPE_ALL>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(tensorSize), static_cast<uint32_t>(alignedDimSizeLast)};

        // 2. 沿dimSizeLast维度进行归一化计算reduce sum
        ReduceSum<C_T, AscendC::Pattern::Reduce::AR, true>(cUb, mulUb, sharedTmpUb.template ReinterpretCast<uint8_t>(),
                                                           srcShape, true);
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams copyParamsC{1, (uint32_t)(tensorSize * sizeof(C_T)), 0, 0, 0};
        DataCopyPad(cGm[address], cUb, copyParamsC);
        PipeBarrier<PIPE_ALL>();
    }
};

#endif // BATCH_MAT_MUL_V3_VECTOR_H
