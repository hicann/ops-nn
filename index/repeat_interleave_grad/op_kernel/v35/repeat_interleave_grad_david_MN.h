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
 * \file repeat_interleave_grad_david_MN.h
 * \brief
 */

#ifndef _REPEAT_INTERLEAVE_GRAD_DAVID_MN_H_
#define _REPEAT_INTERLEAVE_GRAD_DAVID_MN_H_
#include "kernel_operator.h"
#include "platform.h"
#include "reduce_buf_pool.h"
#include "repeat_interleave_grad_base.h"
#include "repeat_interleave_grad_david_tiling_data.h"
namespace RepeatInterleaveGrad {
using namespace AscendC;

template <typename DataT, typename PromoteDataT, typename IndexT>
class RepeatInterleaveGradDavidMN
{
public:
    constexpr static int32_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t BLOCK_SIZE_BYTE = platform::GetUbBlockSize();
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = platform::GetVRegSize() / sizeof(PromoteDataT);
    constexpr static uint64_t CACHE_BUF_SIZE = 16 * 1024;
    constexpr static uint64_t RES_BUF_SIZE = 8 * 1024;
    constexpr static uint64_t MAX_NUM_PER_RES = 2 * 1024;
    constexpr static float INIT_FLOAT_VALUE = 0.0;

public:
    __aicore__ inline RepeatInterleaveGradDavidMN(TPipe& pipe) : pipe_(pipe)
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace,
        const RepeatInterleaveGradDavidTilingData* __restrict tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessRepeatBlock(int32_t repeatFactor, LocalTensor<IndexT>& curRepeat);
    __aicore__ inline void ProcessZeroR(int64_t outputDataOffset, int32_t dim);
    __aicore__ inline void BinaryRedeuceSum(
        int64_t inputDataOffset, int64_t outputDataOffset, int32_t dim0, int32_t dim1);
    __aicore__ inline void SingleRedeuceSum(
        int64_t inputDataOffset, int64_t outputDataOffset, int32_t dimR, int32_t dimA);
    __aicore__ inline void CopyOut(const GlobalTensor<DataT>& dst, LocalTensor<PromoteDataT>& ubRes, int32_t dimA);
    template <bool isRight>
    __aicore__ inline void CopyInData(
        int64_t index, LocalTensor<DataT>& ubTensor, LocalTensor<PromoteDataT>& computeTensor, int64_t inputDataOffset,
        int32_t dim0, int32_t dim1);
    __aicore__ inline void UpdateCacheAux(const int64_t cacheID, const int64_t stride, const int64_t count);

private:
    const RepeatInterleaveGradDavidTilingData* tiling_;
    TPipe& pipe_;
    int32_t blockIdx_ = 0;
    GlobalTensor<DataT> xGm_;
    GlobalTensor<DataT> yGm_;
    GlobalTensor<IndexT> repeatGm_;
    ReduceBufPool bufPool_;
    TBuf<> tempResQue_;
    TBuf<> tempBufQue_;
    TBuf<> repeatInUbBuf_;
    LocalTensor<PromoteDataT> tempBuf_;
    LocalTensor<PromoteDataT> computeRes_;
    LocalTensor<IndexT> repeatInUb_;

    int64_t repeatOffset_ = 0;
    int64_t cacheStride_ = 0;
    IndexT lastRepeatCumsumVal_ = 0;
    int64_t lastRepeatCount_ = 0;
    int64_t blockInputOffset_ = 0;
    int64_t blockOutputOffset_ = 0;

    int32_t rCount_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t cacheCount_ = 0;
    int64_t bisectionTail_ = 0;

    event_t eventId_;

    bool isTailBlock_ = false;
    UbParaUnit nCoreUbPara_;
};

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::Init(
    GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace,
    const RepeatInterleaveGradDavidTilingData* __restrict tiling)
{
    tiling_ = tiling;
    int32_t computeBufferNum = 0;
    if constexpr (!IsSameType<DataT, PromoteDataT>::value) {
        computeBufferNum = DOUBLE_BUFFER;
    }
    bufPool_.template Init<DataT, PromoteDataT>(&pipe_, DOUBLE_BUFFER, computeBufferNum, tiling_->basicBlockSize);
    pipe_.InitBuffer(tempResQue_, RES_BUF_SIZE);
    computeRes_ = tempResQue_.Get<PromoteDataT>(); // ub内A轴大小

    pipe_.InitBuffer(tempBufQue_, CACHE_BUF_SIZE); // ub内二分需要的buf块
    tempBuf_ = tempBufQue_.template Get<PromoteDataT>();

    pipe_.InitBuffer(repeatInUbBuf_, tiling_->repeatBufferSize * 2); // 每次搬入的repeat的大小
    repeatInUb_ = repeatInUbBuf_.template Get<IndexT>();

    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ DataT*)x);
    repeatGm_.SetGlobalBuffer((__gm__ IndexT*)repeats);
    yGm_.SetGlobalBuffer((__gm__ DataT*)y);
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::Process()
{
    if (this->blockIdx_ >= tiling_->realCoreNum) {
        return;
    }

    int32_t mIdx = this->blockIdx_ / tiling_->coreNumPerM;
    int32_t nIdx = this->blockIdx_ % tiling_->coreNumPerM;

    if (nIdx == tiling_->coreNumPerM - 1) { // 尾核
        isTailBlock_ = true;
        nCoreUbPara_ = tiling_->nUbPara.tailCoreUbPara;
    } else {
        isTailBlock_ = false;
        nCoreUbPara_ = tiling_->nUbPara.mainCoreUbPara;
    }

    // M + N.o分核 表示dim!=None
    blockInputOffset_ = mIdx * tiling_->lenP * tiling_->lenN + nIdx * tiling_->nBlockPara.blockFactor;
    blockOutputOffset_ = mIdx * tiling_->lenR * tiling_->lenN + nIdx * tiling_->nBlockPara.blockFactor;

    lastRepeatCumsumVal_ = 0;
    lastRepeatCount_ = 0;
    LocalTensor<IndexT> curRepeat = repeatInUb_[0];

    repeatOffset_ = 0;
    int32_t repeatFactor = tiling_->repeatLoopPara.tailCoreUbPara.ubFactor;
    for (int32_t j = 0; j < tiling_->repeatLoopPara.tailCoreUbPara.ubCount - 1; j++) { // repeat.o
        ProcessRepeatBlock(repeatFactor, curRepeat);
        repeatOffset_ += repeatFactor;
    } // 完成repeat 切 128B之后整块的 repeat.i 的处理

    // repeat.i 尾块
    repeatFactor = tiling_->repeatLoopPara.tailCoreUbPara.ubTailFactor;
    ProcessRepeatBlock(repeatFactor, curRepeat);
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::ProcessRepeatBlock(
    int32_t repeatFactor, LocalTensor<IndexT>& curRepeat)
{
    // MTE2 搬入repeat, 计算前缀和得到索引并缓存 -> rBuf
    DataCopyPadExtParams<IndexT> copyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams repeatCopyParams{0, 0, 0, 0, 0};
    repeatCopyParams.blockCount = 1;
    repeatCopyParams.blockLen = repeatFactor * sizeof(IndexT);
    repeatCopyParams.srcStride = 0;
    repeatCopyParams.dstStride = 0;
    DataCopyPad(curRepeat, this->repeatGm_[repeatOffset_], repeatCopyParams, copyPadExtParams);
    eventId_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId_);
    WaitFlag<HardEvent::MTE2_S>(eventId_);

    // 遍历搬入的repeat -> r
    IndexT r = 0;
    for (int32_t k = 0; k < repeatFactor; k++) {
        lastRepeatCount_ = repeatOffset_ + k;
        int64_t pOffset = lastRepeatCumsumVal_ * tiling_->lenN;
        int64_t rOffset = lastRepeatCount_ * tiling_->lenN;
        r = curRepeat.GetValue(k);

        // r * lenN_ <= tiling_->basicBlockSize  -> 正常计算，实际调用的也是AscendC的接口，不作区分
        if (r == 0) {
            int32_t nFactor = nCoreUbPara_.ubFactor;
            int64_t nOffset = 0;
            int64_t outputDataOffset = 0;
            for (int32_t n = 0; n < nCoreUbPara_.ubCount - 1; n++) {
                nOffset = n * nFactor;
                outputDataOffset = blockOutputOffset_ + rOffset + nOffset;
                ProcessZeroR(outputDataOffset, nFactor);
            }
            int32_t nTailFactor = nCoreUbPara_.ubTailFactor;
            nOffset = (nCoreUbPara_.ubCount - 1) * nFactor;
            outputDataOffset = blockOutputOffset_ + rOffset + nOffset;
            ProcessZeroR(outputDataOffset, nTailFactor);
        } else if (r < tiling_->rFactor) {
            // 如果r很小，需要调整N.i.i的切分
            int32_t nBaseFactor = tiling_->basicBlockSize / r / BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE / sizeof(DataT);
            if (nBaseFactor > MAX_NUM_PER_RES) {
                nBaseFactor = MAX_NUM_PER_RES;
            }
            UbParaUnit newSplitN;
            int64_t blockLen = tiling_->nBlockPara.blockFactor;
            if (isTailBlock_) {
                blockLen = tiling_->nBlockPara.blockTailFactor;
            }
            __RIGUtil::DoUbSplit(blockLen, nBaseFactor, newSplitN);
            int64_t nOffset = 0;
            int64_t inputDataOffset = 0;
            int64_t outputDataOffset = 0;
            int32_t nFactor = newSplitN.ubFactor;
            for (int32_t n = 0; n < newSplitN.ubCount - 1; n++) { // 重新切分的整块
                nOffset = n * nFactor;
                inputDataOffset = blockInputOffset_ + pOffset + nOffset;
                outputDataOffset = blockOutputOffset_ + rOffset + nOffset;
                SingleRedeuceSum(inputDataOffset, outputDataOffset, r, nFactor);
            }

            // 重新切分的尾块
            nOffset = (newSplitN.ubCount - 1) * nFactor;
            inputDataOffset = blockInputOffset_ + pOffset + nOffset;
            outputDataOffset = blockOutputOffset_ + rOffset + nOffset;
            SingleRedeuceSum(inputDataOffset, outputDataOffset, r, newSplitN.ubTailFactor);
        } else {
            // 计算二分参数
            rCount_ = ops::CeilDiv(static_cast<int32_t>(r), tiling_->rFactor);
            bisectionPos_ = __RIGUtil::FindNearestPower2(rCount_);
            cacheCount_ = __RIGUtil::CalLog2(bisectionPos_) + 1;
            bisectionTail_ = rCount_ - bisectionPos_;

            int32_t nFactor = nCoreUbPara_.ubFactor;
            int64_t nOffset = 0;
            int64_t inputDataOffset = 0;
            int64_t outputDataOffset = 0;
            for (int32_t n = 0; n < nCoreUbPara_.ubCount - 1; n++) { // 二分累加的N轴，N.i.i
                nOffset = n * nFactor;
                inputDataOffset = blockInputOffset_ + pOffset + nOffset;
                outputDataOffset = blockOutputOffset_ + rOffset + nOffset;
                // 二分累加
                BinaryRedeuceSum(inputDataOffset, outputDataOffset, r, nFactor);
            }

            // N.i.i 尾块
            int32_t nTailFactor = nCoreUbPara_.ubTailFactor;
            nOffset = (nCoreUbPara_.ubCount - 1) * nFactor;
            inputDataOffset = blockInputOffset_ + pOffset + nOffset;
            outputDataOffset = blockOutputOffset_ + rOffset + nOffset;
            BinaryRedeuceSum(inputDataOffset, outputDataOffset, r, nTailFactor);
        }
        lastRepeatCumsumVal_ += r;
    } // 完成一个repeatFactor个repeat值的处理
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::ProcessZeroR(
    int64_t outputDataOffset, int32_t dim)
{
    Duplicate<PromoteDataT>(computeRes_, INIT_FLOAT_VALUE, dim);
    CopyOut(yGm_[outputDataOffset], computeRes_, dim);
    SetFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    WaitFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::SingleRedeuceSum(
    int64_t inputDataOffset, int64_t outputDataOffset, int32_t dimR, int32_t dimA)
{
    LocalTensor<DataT> tensor;
    bufPool_.template AllocTensor<true, DataT>(tensor);
    LocalTensor<PromoteDataT> computeLeft;

    CopyInData<false>(0, tensor, computeLeft, inputDataOffset, dimR, dimA);
    int32_t dimAAlign = ops::CeilAlign(dimA, static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(DataT)));
    uint32_t srcShape[2] = {static_cast<uint32_t>(dimR), static_cast<uint32_t>(dimAAlign)};
    ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(computeRes_, computeLeft, srcShape, false);
    bufPool_.FreeTensor(computeLeft);
    CopyOut(yGm_[outputDataOffset], computeRes_, dimA);
    eventId_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId_);
    WaitFlag<HardEvent::MTE3_V>(eventId_);
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::BinaryRedeuceSum(
    int64_t inputDataOffset, int64_t outputDataOffset, int32_t dim0, int32_t dim1)
{
    int32_t rMainFactor = tiling_->rFactor;
    int32_t rTailFactor = dim0 - rMainFactor * (rCount_ - 1);
    int64_t tailIndex = rCount_ - 1;

    int32_t dimR = rMainFactor;
    int32_t dimA = dim1;
    int32_t dimAAlign = ops::CeilAlign(dimA, static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(DataT)));
    for (int64_t i = 0; i < bisectionTail_; i++) {
        LocalTensor<DataT> tensorLeft;
        bufPool_.template AllocTensor<true, DataT>(tensorLeft);
        LocalTensor<PromoteDataT> computeLeft;
        // 从gm搬入ub内的tensor自动补pad到block对齐
        CopyInData<false>(i, tensorLeft, computeLeft, inputDataOffset, dimR, dimA);

        LocalTensor<DataT> tensorRight;
        bufPool_.template AllocTensor<true, DataT>(tensorRight);
        LocalTensor<PromoteDataT> computeRight;
        if (i + bisectionPos_ == tailIndex && rTailFactor < rMainFactor) {
            // CopyInData内已经保证了MTE2_V的同步
            CopyInData<true>(i, tensorRight, computeRight, inputDataOffset, rTailFactor, dimA);
            Duplicate<PromoteDataT>(computeRight[rTailFactor * dimAAlign], 0, (rMainFactor - rTailFactor) * dimAAlign);
        } else {
            CopyInData<true>(i, tensorRight, computeRight, inputDataOffset, dimR, dimA);
        }

        Add(computeLeft, computeLeft, computeRight, dimR * dimAAlign);
        uint32_t srcShape[2] = {static_cast<uint32_t>(dimR), static_cast<uint32_t>(dimAAlign)};
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(computeRes_, computeLeft, srcShape, false);

        // fp32 tensorLeft和computeLeft是同一个tensor，fp16 computeLeft在free时不用index
        bufPool_.FreeTensor(computeLeft);
        bufPool_.FreeTensor(computeRight);
        int64_t cacheID = __RIGUtil::GetCacheID(i);
        cacheStride_ = ops::CeilDiv(static_cast<int64_t>(dimA), static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                       ELEMENT_ONE_REPEAT_COMPUTE;
        UpdateCacheAux(cacheID, cacheStride_, dimA);
    }

    for (int64_t i = bisectionTail_; i < bisectionPos_; i++) {
        LocalTensor<DataT> tensor;
        bufPool_.template AllocTensor<true, DataT>(tensor);
        LocalTensor<PromoteDataT> computeLeft;

        CopyInData<false>(i, tensor, computeLeft, inputDataOffset, dimR, dimA);
        uint32_t srcShape[2] = {static_cast<uint32_t>(dimR), static_cast<uint32_t>(dimAAlign)};
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(computeRes_, computeLeft, srcShape, false);

        bufPool_.FreeTensor(computeLeft);
        int64_t cacheID = __RIGUtil::GetCacheID(i);
        cacheStride_ = ops::CeilDiv(static_cast<int64_t>(dimA), static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                       ELEMENT_ONE_REPEAT_COMPUTE;
        UpdateCacheAux(cacheID, cacheStride_, dimA);
    }
    int64_t tmpBufOffest = (cacheCount_ - 1) * cacheStride_;

    // 搬出 CopyOut内保证了V_MTE2
    LocalTensor<PromoteDataT> ubRes = tempBuf_[tmpBufOffest];
    CopyOut(yGm_[outputDataOffset], ubRes, dim1);
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::CopyOut(
    const GlobalTensor<DataT>& dst, LocalTensor<PromoteDataT>& ubRes, int32_t dimA)
{
    // copyOutParams
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
    copyOutParams.blockLen = dimA * sizeof(DataT);
    // DataCopy
    if constexpr (IsSameType<PromoteDataT, DataT>::value) {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        DataCopyPad(dst, ubRes, copyOutParams);
    } else {
        LocalTensor<DataT> outputLocal = ubRes.template ReinterpretCast<DataT>();
        Cast(outputLocal, ubRes, RoundMode::CAST_RINT, dimA);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        DataCopyPad(dst, outputLocal, copyOutParams);
    }
}

template <typename DataT, typename PromoteDataT, typename IndexT>
template <bool isRight>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::CopyInData(
    int64_t index, LocalTensor<DataT>& ubTensor, LocalTensor<PromoteDataT>& computeTensor, int64_t inputDataOffset,
    int32_t dim0, int32_t dim1)
{
    if constexpr (isRight) {
        index = index + bisectionPos_;
    }

    int64_t offset = inputDataOffset + index * tiling_->rFactor * tiling_->lenN;
    DataCopyPadExtParams<DataT> copyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCopyParams{0, 0, 0, 0, 0};
    dataCopyParams.blockCount = dim0;
    dataCopyParams.blockLen = dim1 * sizeof(DataT);
    dataCopyParams.srcStride = (tiling_->lenN - dim1) * sizeof(DataT);
    dataCopyParams.dstStride = 0;
    DataCopyPad(ubTensor, this->xGm_[offset], dataCopyParams, copyPadExtParams);
    if constexpr (IsSameType<PromoteDataT, DataT>::value) {
        eventId_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        computeTensor = ubTensor;
    } else {
        bufPool_.template AllocTensor<false, PromoteDataT>(computeTensor);
        eventId_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        int32_t dim1Align = ops::CeilAlign(dim1, static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(DataT)));
        Cast(computeTensor, ubTensor, RoundMode::CAST_NONE, dim0 * dim1Align);
        bufPool_.FreeTensor(ubTensor);
    }
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradDavidMN<DataT, PromoteDataT, IndexT>::UpdateCacheAux(
    const int64_t cacheID, const int64_t stride, const int64_t count)
{
    // count A轴的大小 * VL
    uint16_t outerLoopTimes =
        ops::CeilDiv(static_cast<int64_t>(count * sizeof(PromoteDataT)), static_cast<int64_t>(platform::GetVRegSize()));
    uint16_t innerLoopTimes = cacheID;
    uint32_t outerLoopStride = ELEMENT_ONE_REPEAT_COMPUTE;
    uint32_t innerLoopStride = stride; // cacahe的每一个idex的块的大小， A轴的大小
    LocalTensor<PromoteDataT> dstTensor = tempBuf_;
    LocalTensor<PromoteDataT> srcTensor = computeRes_;

    __VEC_SCOPE__
    {
        __local_mem__ PromoteDataT* dst = (__local_mem__ PromoteDataT*)dstTensor.GetPhyAddr();
        __local_mem__ PromoteDataT* cah = (__local_mem__ PromoteDataT*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ PromoteDataT* src = (__local_mem__ PromoteDataT*)srcTensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::MicroAPI::RegTensor<PromoteDataT> aReg, bReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) { // outerLoopTimes是dimA的大小
            pMask = AscendC::MicroAPI::UpdateMask<PromoteDataT>(sreg);
            DataCopy(aReg, (__local_mem__ PromoteDataT*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                DataCopy(bReg, (__local_mem__ PromoteDataT*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<PromoteDataT, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            DataCopy((__local_mem__ PromoteDataT*)cah + i * outerLoopStride, aReg, pMask);
        }
    }
}

} // namespace RepeatInterleaveGrad

#endif