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
 * \file scatter_list_rsbse.h
 * \brief
 */
// 此文件实现所有芯片、-2轴更新、小batch小element、updates的后两维乘积对齐block场景，对应模板200
// R: row, S: small, B: batch(updates的前两维除以核数得到), E: element(updates的后两维乘积)
#ifndef SCATTER_LIST_RSBSE_H_
#define SCATTER_LIST_RSBSE_H_

#include "kernel_operator.h"
#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListRSBSE : public ScatterListBase<T1>
{
public:
    __aicore__ inline ScatterListRSBSE(){};
    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR varOut, GM_ADDR workspace,
        const ScatterListTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOutEqLen(const int64_t& eachCoreBatchNum);
    __aicore__ inline void CopyOutNotEqLen(const int64_t& eachCoreBatchNum);

    constexpr static uint32_t bufferNum = 1;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> indiceInQueue;
    TQue<QuePosition::VECIN, 1> maskInQueue;
    TQue<QuePosition::VECIN, bufferNum> updatesInQueue;

    GlobalTensor<T1> varGm;
    GlobalTensor<T1> updatesGm;
    GlobalTensor<T2> indiceGm;
    GlobalTensor<uint8_t> maskGm;

    GM_ADDR varPtr = nullptr;
    bool maskIsNull = false;
    int64_t blockIndex = 0;
    int64_t preCoreBatchIdx = 0;
    int64_t curCoreBatchIdx = 0;
    int64_t dim0Idx = 0;
    int64_t dim1Idx = 0;
    int64_t dim2OffsetIdx = 0;
    int64_t dim2UpdateLen = 0;
    int64_t dstGmOffset = 0;
    int64_t srcGmOffset = 0;

    // tiling params
    ScatterListTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void ScatterListRSBSE<T1, T2>::Init(
    GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR varOut, GM_ADDR workspace,
    const ScatterListTilingData* tilingData)
{
    blockIndex = GetBlockIdx();
    this->ParseTilingData(tilingData, m_tilingData);
    varPtr = var;
    indiceGm.SetGlobalBuffer((__gm__ T2*)indice);
    updatesGm.SetGlobalBuffer((__gm__ T1*)updates);

    preCoreBatchIdx = blockIndex * m_tilingData.preCoreBatchNum;
    pipe.InitBuffer(indiceInQueue, 1, m_tilingData.indiceUbSize);
    pipe.InitBuffer(updatesInQueue, bufferNum, m_tilingData.updatesUbSize);

    if (mask == nullptr) {
        maskIsNull = true;
    } else {
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask);
        pipe.InitBuffer(maskInQueue, 1, m_tilingData.maskUbSize);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRSBSE<T1, T2>::Process()
{
    if (blockIndex >= m_tilingData.useCoreNum) {
        return;
    }
    if (blockIndex == m_tilingData.useCoreNum - 1) {
        if (m_tilingData.indiceDims != 1) {
            CopyIn();
            CopyOutNotEqLen(m_tilingData.lastCoreBatchNum);
        } else {
            CopyIn();
            CopyOutEqLen(m_tilingData.lastCoreBatchNum);
        }
    } else {
        if (m_tilingData.indiceDims == 1) {
            CopyIn();
            CopyOutEqLen(m_tilingData.preCoreBatchNum);
        } else {
            CopyIn();
            CopyOutNotEqLen(m_tilingData.preCoreBatchNum);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRSBSE<T1, T2>::CopyIn()
{
    LocalTensor<T1> updatesUb = updatesInQueue.AllocTensor<T1>();
    DataCopy(updatesUb, updatesGm[preCoreBatchIdx * m_tilingData.srcBatchStride], m_tilingData.updatesCount);
    updatesInQueue.EnQue(updatesUb);

    LocalTensor<T2> indiceUb = indiceInQueue.AllocTensor<T2>();
    DataCopy(indiceUb, indiceGm, m_tilingData.indiceCount);
    indiceInQueue.EnQue(indiceUb);

    if (!maskIsNull) {
        LocalTensor<uint8_t> maskUb = maskInQueue.AllocTensor<uint8_t>();
        DataCopy(maskUb, maskGm, m_tilingData.maskCount);
        maskInQueue.EnQue(maskUb);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRSBSE<T1, T2>::CopyOutEqLen(const int64_t& eachCoreBatchNum)
{
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }
    this->Mte2ToS();
    this->Mte2ToMte3();

    for (int64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < eachCoreBatchNum; eachCoreBatchIdx++) {
        curCoreBatchIdx = preCoreBatchIdx + eachCoreBatchIdx;
        dim0Idx = curCoreBatchIdx / m_tilingData.dim1Count;
        if (maskIsNull || maskUb.GetValue(dim0Idx) == 1) {
            varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
            dim1Idx = curCoreBatchIdx % m_tilingData.dim1Count;
            dim2OffsetIdx = indiceUb.GetValue(dim0Idx);
            dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx * m_tilingData.dim3Count;
            srcGmOffset = eachCoreBatchIdx * m_tilingData.srcBatchStride;
            DataCopy(varGm[dstGmOffset], updatesUb[srcGmOffset], m_tilingData.srcBatchStride);
        }
    }

    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRSBSE<T1, T2>::CopyOutNotEqLen(const int64_t& eachCoreBatchNum)
{
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }
    this->Mte2ToS();
    this->Mte2ToMte3();

    for (int64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < eachCoreBatchNum; eachCoreBatchIdx++) {
        curCoreBatchIdx = preCoreBatchIdx + eachCoreBatchIdx;
        dim0Idx = curCoreBatchIdx / m_tilingData.dim1Count;
        if (maskIsNull || maskUb.GetValue(dim0Idx) == 1) {
            varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
            dim1Idx = curCoreBatchIdx % m_tilingData.dim1Count;
            dim2OffsetIdx = indiceUb.GetValue(dim0Idx * 2);
            dim2UpdateLen = indiceUb.GetValue(dim0Idx * 2 + 1);
            dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx * m_tilingData.dim3Count;
            srcGmOffset = eachCoreBatchIdx * m_tilingData.srcBatchStride;
            DataCopy(varGm[dstGmOffset], updatesUb[srcGmOffset], dim2UpdateLen * m_tilingData.dim3Count);
        }
    }

    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

} // namespace ScatterList

#endif // SCATTER_LIST_RSBSE_H_
