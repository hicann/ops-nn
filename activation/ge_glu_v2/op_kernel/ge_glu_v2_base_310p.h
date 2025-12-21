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
 * \file ge_glu_v2_base_310p.h
 * \brief
 */

#ifndef GeGluV2_BASE_310P_H
#define GeGluV2_BASE_310P_H

#include "ge_glu_v2_base.h"

namespace GeGluV2 {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t BUFFER_SIZE = 8192;
constexpr int32_t CHUNK_NUM = 2;
constexpr int32_t MASK_SIZE = 2;
constexpr int64_t MAX_UINT8 = 255;
constexpr int64_t NUM_ONE_BLOCK_INT = 8;
constexpr int64_t BYTE_ONE_BLOCK = 32;
constexpr float ERF_PARAM1 = -0.3512339572e-8f;
constexpr float ERF_PARAM2 = 0.2645266170e-6f;
constexpr float ERF_PARAM3 = -0.7929488134e-5f;
constexpr float ERF_PARAM4 = 0.1106123840e-3f;
constexpr float ERF_PARAM5 = 0.6518995814e-4f;
constexpr float ERF_PARAM6 = -0.7266616915e-1f;
constexpr float ERF_PARAM7 = -0.1595769883e1f;
constexpr float ERF_THRESHOLD = 5.75;

template <typename T>
class GeGluV2Base310P : public GeGluV2Base<T> {
public:
    __aicore__ inline GeGluV2Base310P(){};

protected:
    __aicore__ inline void BaseInit310P(GM_ADDR workspace);
    __aicore__ inline void CopyInX(
        const int64_t& index, const int64_t& blockCount, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2);
    __aicore__ inline void ComputeGeluBaseErf(
        LocalTensor<float>& ubx2_fp32, LocalTensor<float>& computeOut, LocalTensor<float>& x1,
        LocalTensor<float>& x_pow, const int64_t& length);
    __aicore__ inline void CopyOutBase(
        const int64_t& index, const int64_t& length, const int64_t& group, LocalTensor<T>& outLocal,
        GlobalTensor<T>& outGm, LocalTensor<T>& tmpLocal, int64_t flag);
    __aicore__ inline void CopyInXAlignLastBigWithoutPad(
        const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2,
        const int64_t& delta);
    __aicore__ inline void CopyOutGeluBaseLastBigWithoutPad(
        const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& outLocal,
        const int64_t& delta);
    __aicore__ inline void CopyOutMulBaseLastBigWithoutPad(
        const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& outLocal,
        const int64_t& delta);
    __aicore__ inline void CopyNew(
        int64_t saveToWsNum, const int64_t& group, LocalTensor<T>& outLocal, GlobalTensor<T>& outGm, int64_t flag,
        bool needTransWorkspace);

protected:
    uint64_t m_rightPadNum;
    int64_t m_splitSizeAlign;
    int64_t m_blockCount;
    uint64_t m_duplicateMask;
    int64_t m_duplicateOffset;
    int64_t tailBlock;
    int64_t alignLength;
    uint64_t curLoopDstAddr;
    bool m_isSplitSizeAlign;
    bool m_isLastCore;

private:
    GlobalTensor<int32_t> m_syncGlobal;
    TQue<QuePosition::VECIN, 1> m_workQueue;
    TPipe m_pipe;
    GlobalTensor<T> tailTmpWsSelf;
};

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::BaseInit310P(GM_ADDR workspace)
{
    if (this->blockIdx >= this->m_tilingData.realCoreNum) {
        return;
    }
    m_isLastCore = (this->blockIdx == this->m_tilingData.realCoreNum - 1) &&
                   (this->m_tilingData.tailLoopNum != 0 || this->m_tilingData.lastTailGroup != 0);
    m_isSplitSizeAlign = (this->m_tilingData.splitSize % this->m_tilingData.blockSize) == 0;
    if (!m_isSplitSizeAlign) {
        int64_t gmSizePerCore = this->m_tilingData.numPerCore * this->m_tilingData.splitSize;
        if (m_isLastCore) {
            gmSizePerCore = this->m_tilingData.tailLoopNum * this->m_tilingData.group * this->m_tilingData.splitSize +
                            this->m_tilingData.lastTailGroup * this->m_tilingData.splitSize;
        }
        tailTmpWsSelf.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(workspace + this->blockIdx * BYTE_ONE_BLOCK * 4),
            this->m_tilingData.blockSize * 4); // 扩大两倍，分别给两个输出
        m_splitSizeAlign = (this->m_tilingData.splitSize + this->m_tilingData.blockSize - 1) /
                           this->m_tilingData.blockSize * this->m_tilingData.blockSize;
        m_rightPadNum = m_splitSizeAlign - this->m_tilingData.splitSize;
        m_duplicateMask = ((1 << m_rightPadNum) - 1) << (this->m_tilingData.blockSize - m_rightPadNum);
        m_blockCount = m_splitSizeAlign / this->m_tilingData.blockSize;
        m_duplicateOffset = m_splitSizeAlign - this->m_tilingData.blockSize;
    }
}

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::CopyInX(
    const int64_t& index, const int64_t& group, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2)
{
    DataCopyParams intriParams;
    intriParams.blockCount = group;
    intriParams.dstStride = 0;
    uint64_t curLoopSrcAddr = this->gmXOffset + index * this->one_process_in_stride;
    if (m_isSplitSizeAlign) {
        // align case
        intriParams.blockLen = this->m_tilingData.splitSize / this->m_tilingData.blockSize;
        intriParams.srcStride = this->m_tilingData.splitSize / this->m_tilingData.blockSize;
        DataCopy(ubX1, this->xGm[curLoopSrcAddr + this->x1_stride], intriParams);
        DataCopy(ubX2, this->xGm[curLoopSrcAddr + this->x2_stride], intriParams);
    } else {
        // not align case
        for (uint64_t idx = 0; idx < group; ++idx) {
            uint64_t dstAddr = idx * m_splitSizeAlign;
            uint64_t srcAddrOffset = idx * this->m_tilingData.splitSize * CHUNK_NUM;
            DataCopy(ubX1[dstAddr], this->xGm[curLoopSrcAddr + this->x1_stride + srcAddrOffset], m_splitSizeAlign);
            DataCopy(ubX2[dstAddr], this->xGm[curLoopSrcAddr + this->x2_stride + srcAddrOffset], m_splitSizeAlign);
        }
    }
}

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::CopyNew(
    int64_t saveToWsNum, const int64_t& group, LocalTensor<T>& outLocal, GlobalTensor<T>& outGm, int64_t flag,
    bool needTransWorkspace)
{
    for (int64_t idx = 0; idx < group; ++idx) {
        if (idx >= (group - saveToWsNum) && needTransWorkspace) {
            // 只有一个核或者最后一个核处理时，直接覆盖后面的地址即可
            if (this->m_tilingData.realCoreNum == 1 || m_isLastCore) {
                DataCopy(
                    outGm[curLoopDstAddr + idx * this->m_tilingData.splitSize], outLocal[idx * m_splitSizeAlign],
                    m_splitSizeAlign);
                PipeBarrier<PIPE_MTE3>();
            } else if (this->m_tilingData.splitSize < this->m_tilingData.blockSize) {
                if (idx != group - 1) {
                    continue;
                } // 最后一行统一处理
                uint64_t numOneBlock =
                    (this->m_tilingData.blockSize + this->m_tilingData.splitSize - 1) / this->m_tilingData.splitSize;
                tailBlock = numOneBlock * this->m_tilingData.splitSize % this->m_tilingData.blockSize;
                for (int64_t i = 0; i < numOneBlock; i++) {
                    DataCopy(
                        tailTmpWsSelf[flag * this->m_tilingData.blockSize * 2 + i * this->m_tilingData.splitSize],
                        outLocal[(idx - numOneBlock + 1 + i) * m_splitSizeAlign], m_splitSizeAlign);
                    PipeBarrier<PIPE_MTE3>();
                }
            } else {
                DataCopy(
                    outGm[curLoopDstAddr + idx * this->m_tilingData.splitSize], outLocal[idx * m_splitSizeAlign],
                    alignLength); // 向下对齐拷贝
                DataCopy(
                    tailTmpWsSelf[flag * this->m_tilingData.blockSize * 2],
                    outLocal[idx * m_splitSizeAlign + alignLength - this->m_tilingData.blockSize],
                    this->m_tilingData.blockSize * 2); // 向下对齐拷贝
            }
        } else {
            DataCopy(
                outGm[curLoopDstAddr + idx * this->m_tilingData.splitSize], outLocal[idx * m_splitSizeAlign],
                m_splitSizeAlign);
            PipeBarrier<PIPE_MTE3>();
        }
    }
}

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::CopyOutBase(
    const int64_t& index, const int64_t& length, const int64_t& group, LocalTensor<T>& outLocal, GlobalTensor<T>& outGm,
    LocalTensor<T>& tmpLocal, int64_t flag)
{
    curLoopDstAddr = this->gmDYOffset + index * this->one_process_out_stride;
    if (m_isSplitSizeAlign) {
        DataCopy(outGm[curLoopDstAddr], outLocal, length);
    } else {
        tailBlock = this->m_tilingData.splitSize % this->m_tilingData.blockSize;
        alignLength = this->m_tilingData.splitSize - tailBlock;
        int64_t saveToWsNum = 1;
        if (this->m_tilingData.splitSize < this->m_tilingData.blockSize) {
            saveToWsNum = this->m_tilingData.blockSize / this->m_tilingData.splitSize;
        }
        if (saveToWsNum > group) {
            saveToWsNum = group;
        }
        bool needTransWorkspace = false;
        if ((index + 1) * this->m_tilingData.group >= this->m_tilingData.numPerCore) {
            needTransWorkspace = true;
        }
        CopyNew(saveToWsNum, group, outLocal, outGm, flag, needTransWorkspace);
        if (this->m_tilingData.realCoreNum > 1 && !m_isLastCore && needTransWorkspace) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);

            DataCopy(
                tmpLocal, tailTmpWsSelf[flag * this->m_tilingData.blockSize * 2 + tailBlock],
                this->m_tilingData.blockSize);

            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID3);

            DataCopy(
                outGm[curLoopDstAddr + group * this->m_tilingData.splitSize - this->m_tilingData.blockSize], tmpLocal,
                this->m_tilingData.blockSize);
        }
    }
}

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::CopyInXAlignLastBigWithoutPad(
    const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& ubX1, LocalTensor<T>& ubX2,
    const int64_t& delta)
{
    DataCopyParams copyParams{};
    copyParams.blockCount = 1;
    copyParams.blockLen = blockLen;

    int64_t offset = idxX * this->one_process_in_stride + idxY * this->m_tilingData.splitSize - delta;
    int64_t offsetA = offset + this->x1_stride;
    int64_t offsetB = offset + this->x2_stride;

    DataCopy(ubX1, this->xGm[offsetA], copyParams);
    DataCopy(ubX2, this->xGm[offsetB], copyParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::CopyOutMulBaseLastBigWithoutPad(
    const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& outLocal, const int64_t& delta)
{
    DataCopyParams copyParams{};
    copyParams.blockCount = 1;
    copyParams.blockLen = blockLen;

    int64_t offset = idxX * this->one_process_out_stride + idxY * this->m_tilingData.splitSize - delta;
    DataCopy(this->yMulGm[offset], outLocal, copyParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::CopyOutGeluBaseLastBigWithoutPad(
    const int64_t& idxX, const int64_t& idxY, const int64_t& blockLen, LocalTensor<T>& outLocal, const int64_t& delta)
{
    DataCopyParams copyParams{};
    copyParams.blockCount = 1;
    copyParams.blockLen = blockLen;

    int64_t offset = idxX * this->one_process_out_stride + idxY * this->m_tilingData.splitSize - delta;
    DataCopy(this->yGeluGm[offset], outLocal, copyParams);
}

template <typename T>
__aicore__ inline void GeGluV2Base310P<T>::ComputeGeluBaseErf(
    LocalTensor<float>& ubx2_fp32, LocalTensor<float>& computeOut, LocalTensor<float>& x1, LocalTensor<float>& xPow,
    const int64_t& length)
{
    Mins(x1, ubx2_fp32, ERF_THRESHOLD, length);

    Mul(xPow, x1, x1, length);
    Muls(computeOut, xPow, ERF_PARAM1, length);

    Adds(computeOut, computeOut, ERF_PARAM2, length);
    Mul(computeOut, computeOut, xPow, length);

    Adds(computeOut, computeOut, ERF_PARAM3, length);
    Mul(computeOut, computeOut, xPow, length);

    Adds(computeOut, computeOut, ERF_PARAM4, length);
    Mul(computeOut, computeOut, xPow, length);

    Adds(computeOut, computeOut, ERF_PARAM5, length);
    Mul(computeOut, computeOut, xPow, length);

    Adds(computeOut, computeOut, ERF_PARAM6, length);
    Mul(computeOut, computeOut, xPow, length);

    Adds(computeOut, computeOut, ERF_PARAM7, length);
    Mul(computeOut, computeOut, x1, length);

    Exp(computeOut, computeOut, length);

    Adds(computeOut, computeOut, 1.0f, length);
    Div(computeOut, ubx2_fp32, computeOut, length);
}
} // namespace GeGluV2
#endif // GeGluV2_BASE_310P_H
