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
 * \file repeat_interleave_grad_n_1.h
 * \brief
 */
#ifndef _REPEAT_INTERLEAVE_GRAD_N_ONE_H_
#define _REPEAT_INTERLEAVE_GRAD_N_ONE_H_
#include "kernel_operator.h"
#include "reduce_buf_pool.h"
#include "repeat_interleave_grad_base.h"
#include "repeat_interleave_grad_david_tiling_data.h"

namespace RepeatInterleaveGrad {
using namespace AscendC;

template <typename DataT, typename PromoteDataT, typename IndexT>
class RepeatInterleaveGradDimNOneDavid
{
private:
    const RepeatInterleaveGradDavidTilingData* tiling_;
    TPipe& pipe_;
    int32_t blockIdx_ = 0;
    GlobalTensor<DataT> xGm_;
    GlobalTensor<DataT> yGm_;
    GlobalTensor<IndexT> repeatGm_;
    TBuf<> inputBuf_;
    ReduceBufPool bufPool_;
    TBuf<> tempResQue_;
    TBuf<> tempScatterQue_;
    TBuf<> tempScatterOffsetQue_;
    TBuf<> tempBufQue_;
    TBuf<> tempUbQue_;
    TBuf<> repeatInUbBuf_;
    LocalTensor<PromoteDataT> tempUb_;
    LocalTensor<PromoteDataT> tempBuf_;
    LocalTensor<PromoteDataT> computeRes_;
    LocalTensor<PromoteDataT> scatterRes_; // 存放scatter后的结果
    LocalTensor<uint32_t> scatterOffset_;  // 存放scatter的offset
    LocalTensor<IndexT> repeatInUb_;

    int64_t repeatOffset_ = 0;
    int64_t cacheStride_ = 0;
    IndexT lastRepeatCumsumVal_ = 0;
    int64_t lastRepeatCount_ = 0;
    int64_t mInputOffset_ = 0;
    int64_t mOutputOffset_ = 0;

    int32_t rCount_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t cacheCount_ = 0;
    int64_t bisectionTail_ = 0;

public:
    constexpr static int32_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t CACHE_BUF_SIZE = 16 * 1024;
    constexpr static uint64_t RES_BUF_SIZE = 2 * 1024;
    constexpr static uint64_t SCATTER_OFFSET_SIZE = 2 * 1024;
    constexpr static uint64_t SCATTER_BUF_SIZE = 4 * 1024;
    constexpr static int32_t BASE_SPLIT_LENGTH = 128;
    constexpr static int32_t ELEMENT_ONE_REPEAT_ORI = VL_LENGTH_B / sizeof(DataT);
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = VL_LENGTH_B / sizeof(PromoteDataT);
    constexpr static int32_t ELEMENT_ONE_BLOCK_SIZE = BLOCK_SIZE_BYTE / sizeof(PromoteDataT);

public:
    __aicore__ inline RepeatInterleaveGradDimNOneDavid(TPipe& pipe) : pipe_(pipe)
    {}

    __aicore__ inline void Init(
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

        pipe_.InitBuffer(tempScatterQue_, SCATTER_BUF_SIZE);
        scatterRes_ = tempScatterQue_.Get<PromoteDataT>(); // ub内A轴大小

        pipe_.InitBuffer(tempScatterOffsetQue_, SCATTER_OFFSET_SIZE);
        scatterOffset_ = tempScatterOffsetQue_.Get<uint32_t>(); // ub内A轴大小

        pipe_.InitBuffer(tempBufQue_, CACHE_BUF_SIZE); // ub内二分需要的buf块
        tempBuf_ = tempBufQue_.template Get<PromoteDataT>();

        pipe_.InitBuffer(tempUbQue_, BLOCK_SIZE_BYTE);
        tempUb_ = tempUbQue_.template Get<PromoteDataT>();

        pipe_.InitBuffer(repeatInUbBuf_, tiling_->repeatBufferSize * DOUBLE_BUFFER); // 每次搬入的repeat的大小
        repeatInUb_ = repeatInUbBuf_.template Get<IndexT>();

        blockIdx_ = GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ DataT*)x);
        repeatGm_.SetGlobalBuffer((__gm__ IndexT*)repeats);
        yGm_.SetGlobalBuffer((__gm__ DataT*)y);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->realCoreNum) {
            return;
        }

        // M分核 表示dim!=None
        mInputOffset_ = blockIdx_ * tiling_->mBlockPara.blockFactor * tiling_->lenP * tiling_->lenN;
        mOutputOffset_ = blockIdx_ * tiling_->mBlockPara.blockFactor * tiling_->lenR * tiling_->lenN;

        // scatter 到 block对齐
        LocalTensor<int32_t> scatterOffsetInt = scatterOffset_.template ReinterpretCast<int32_t>();
        CreateVecIndex(scatterOffsetInt, 0, BASE_SPLIT_LENGTH);
        Muls(scatterOffsetInt, scatterOffsetInt, static_cast<int32_t>(BLOCK_SIZE_BYTE), BASE_SPLIT_LENGTH);

        lastRepeatCumsumVal_ = 0;
        lastRepeatCount_ = 0;

        LocalTensor<IndexT> repeatTensor = repeatInUb_[0];
        // repeat.i 整块
        // 500 -> 4 128 116
        repeatOffset_ = 0;
        int32_t repeatFactor = tiling_->repeatLoopPara.tailCoreUbPara.ubFactor;
        for (int32_t i = 0; i < tiling_->repeatLoopPara.tailCoreUbPara.ubCount - 1; i++) { // repeat.o
            ProcessRepeatBlock(repeatFactor, repeatTensor);
            repeatOffset_ += repeatFactor;
        } // 完成repeat 切 128B之后整块的 repeat.i 的处理

        // repeat.i 尾块
        repeatFactor = tiling_->repeatLoopPara.tailCoreUbPara.ubTailFactor;
        ProcessRepeatBlock(repeatFactor, repeatTensor);
    }

private:
    __aicore__ inline void ProcessRepeatBlock(int32_t repeatFactor, LocalTensor<IndexT>& repeatTensor)
    {
        int32_t mCount = 0;
        int32_t mFactor = 0;
        int32_t mTailFactor = 0;

        if (blockIdx_ == tiling_->mBlockPara.blockCount - 1) {
            mCount = tiling_->mUbPara.tailCoreUbPara.ubCount;
            mFactor = tiling_->mUbPara.tailCoreUbPara.ubFactor;
            mTailFactor = tiling_->mUbPara.tailCoreUbPara.ubTailFactor;
        } else {
            mCount = tiling_->mUbPara.mainCoreUbPara.ubCount;
            mFactor = tiling_->mUbPara.mainCoreUbPara.ubFactor;
            mTailFactor = tiling_->mUbPara.mainCoreUbPara.ubTailFactor;
        }
        // MTE2 搬入repeat, 计算前缀和得到索引并缓存 -> rBuf
        DataCopyPadExtParams<IndexT> copyPadExtParams = {false, 0, 0, 0};
        DataCopyExtParams repeatCopyParams{0, 0, 0, 0, 0};
        repeatCopyParams.blockCount = 1;
        repeatCopyParams.blockLen = repeatFactor * sizeof(IndexT);
        repeatCopyParams.srcStride = 0;
        repeatCopyParams.dstStride = 0;
        DataCopyPad(repeatTensor, repeatGm_[repeatOffset_], repeatCopyParams, copyPadExtParams);
        __RIGUtil::SetEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        // 遍历搬入的repeat -> r
        IndexT r = 0;
        for (int32_t k = 0; k < repeatFactor; k++) {
            lastRepeatCount_ = repeatOffset_ + k;
            int64_t pOffset = lastRepeatCumsumVal_; // tiling_->lenN = 1，所以没写乘法
            int64_t rOffset = lastRepeatCount_;
            int64_t mOffset = 0;
            int64_t inputDataOffset = 0;
            int64_t outputDataOffset = 0;
            r = repeatTensor.GetValue(k);
            if (r == 1) {
                for (int32_t m = 0; m < mCount - 1; m++) {
                    mOffset = m * mFactor;
                    inputDataOffset = mInputOffset_ + pOffset + mOffset * tiling_->lenP;
                    outputDataOffset = mOutputOffset_ + rOffset + mOffset * tiling_->lenR;
                    PureMove(inputDataOffset, outputDataOffset, mFactor);
                }
                mOffset = (mCount - 1) * mFactor;
                inputDataOffset = mInputOffset_ + pOffset + mOffset * tiling_->lenP;
                outputDataOffset = mOutputOffset_ + rOffset + mOffset * tiling_->lenR;
                PureMove(inputDataOffset, outputDataOffset, mTailFactor);
            } else if (r <= tiling_->rFactor) {
                for (int32_t m = 0; m < mCount - 1; m++) {
                    mOffset = m * mFactor;
                    inputDataOffset = mInputOffset_ + pOffset + mOffset * tiling_->lenP;
                    outputDataOffset = mOutputOffset_ + rOffset + mOffset * tiling_->lenR;
                    SingleRedeuceSum(inputDataOffset, outputDataOffset, r, mFactor);
                }

                mOffset = (mCount - 1) * mFactor;
                inputDataOffset = mInputOffset_ + pOffset + mOffset * tiling_->lenP;
                outputDataOffset = mOutputOffset_ + rOffset + mOffset * tiling_->lenR;
                SingleRedeuceSum(inputDataOffset, outputDataOffset, r, mTailFactor);
            } else {
                // 计算二分参数
                rCount_ = ops::CeilDiv(static_cast<int32_t>(r), tiling_->rFactor);
                bisectionPos_ = __RIGUtil::FindNearestPower2(rCount_);
                cacheCount_ = __RIGUtil::CalLog2(bisectionPos_) + 1;
                bisectionTail_ = rCount_ - bisectionPos_;
                for (int32_t m = 0; m < mCount - 1; m++) { // 二分累加的N轴，M.i 整块
                    mOffset = m * mFactor;
                    inputDataOffset = mInputOffset_ + pOffset + mOffset * tiling_->lenP;
                    outputDataOffset = mOutputOffset_ + rOffset + mOffset * tiling_->lenR;
                    // 二分累加
                    BinaryRedeuceSum(inputDataOffset, outputDataOffset, r, mFactor);
                }

                // M.i尾块
                mOffset = (mCount - 1) * mFactor;
                inputDataOffset = mInputOffset_ + pOffset + mOffset * tiling_->lenP;
                outputDataOffset = mOutputOffset_ + rOffset + mOffset * tiling_->lenR;
                BinaryRedeuceSum(inputDataOffset, outputDataOffset, r, mTailFactor);
            }
            lastRepeatCumsumVal_ += r;
        } // 完成一个repeatFactor个repeat值的处理
    }

    __aicore__ inline void BinaryRedeuceSum(
        int64_t inputDataOffset, int64_t outputDataOffset, int32_t dim0, int32_t dim1)
    {
        int32_t rMainFactor = tiling_->rFactor;
        int32_t rTailFactor = dim0 - rMainFactor * (rCount_ - 1);
        int64_t tailIndex = rCount_ - 1;

        int32_t dimR = rMainFactor;
        int32_t dimA = dim1;
        int32_t dimRAlign = ops::CeilAlign(dimR, static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(DataT)));
        for (int64_t i = 0; i < bisectionTail_; i++) {
            LocalTensor<DataT> tensorLeft;
            bufPool_.template AllocTensor<true, DataT>(tensorLeft);
            LocalTensor<PromoteDataT> computeLeft;
            // 从gm搬入ub内的tensor自动补pad到block对齐
            CopyInData<false>(i, tensorLeft, computeLeft, inputDataOffset, dimA, dimR, dimA, dimRAlign);

            LocalTensor<DataT> tensorRight;
            bufPool_.template AllocTensor<true, DataT>(tensorRight);
            LocalTensor<PromoteDataT> computeRight;
            if (i + bisectionPos_ == tailIndex) {
                // CopyInData内已经保证了MTE2_V的同步
                if (rTailFactor < dimR - ELEMENT_ONE_BLOCK_SIZE) {
                    // 不能随路补0的时候
                    Duplicate<DataT>(tensorRight, 0, dimA * dimRAlign);
                    __RIGUtil::SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                }
                CopyInData<true>(i, tensorRight, computeRight, inputDataOffset, dimA, rTailFactor, dimA, dimRAlign);
            } else {
                CopyInData<true>(i, tensorRight, computeRight, inputDataOffset, dimA, dimR, dimA, dimRAlign);
            }

            Add(computeLeft, computeLeft, computeRight, dimA * dimRAlign);
            bufPool_.FreeTensor(computeRight);
            uint32_t srcShape[2] = {static_cast<uint32_t>(dimA), static_cast<uint32_t>(dimRAlign)};
            ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::AR, true>(computeRes_, computeLeft, srcShape, false);

            // fp32 tensorLeft和computeLeft是同一个tensor，fp16 computeLeft在free时不用index
            bufPool_.FreeTensor(computeLeft);
            int64_t cacheID = __RIGUtil::GetCacheID(i);
            cacheStride_ = ops::CeilDiv(static_cast<int64_t>(dimA), static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                           ELEMENT_ONE_REPEAT_COMPUTE;
            UpdateCacheAux(cacheID, cacheStride_, dimA);
        }

        for (int64_t i = bisectionTail_; i < bisectionPos_; i++) {
            LocalTensor<DataT> tensor;
            bufPool_.template AllocTensor<true, DataT>(tensor);
            LocalTensor<PromoteDataT> computeLeft;

            CopyInData<false>(i, tensor, computeLeft, inputDataOffset, dimA, dimR, dimA, dimRAlign);
            uint32_t srcShape[2] = {static_cast<uint32_t>(dimA), static_cast<uint32_t>(dimRAlign)};
            ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::AR, true>(computeRes_, computeLeft, srcShape, false);

            bufPool_.FreeTensor(computeLeft);
            int64_t cacheID = __RIGUtil::GetCacheID(i);
            cacheStride_ = ops::CeilDiv(static_cast<int64_t>(dimA), static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                           ELEMENT_ONE_REPEAT_COMPUTE;
            UpdateCacheAux(cacheID, cacheStride_, dimA);
        }
        int64_t tmpBufOffest = (cacheCount_ - 1) * cacheStride_;

        // 搬出 CopyOut内保证了V_MTE2
        LocalTensor<PromoteDataT> ubRes = tempBuf_[tmpBufOffest];
        CopyOut(yGm_[outputDataOffset], ubRes, dimA);
        __RIGUtil::SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void SingleRedeuceSum(
        int64_t inputDataOffset, int64_t outputDataOffset, int32_t dimR, int32_t dimA)
    {
        LocalTensor<DataT> tensor;
        bufPool_.template AllocTensor<true, DataT>(tensor);
        LocalTensor<PromoteDataT> computeTensor;
        CopyInData<false>(0, tensor, computeTensor, inputDataOffset, dimA, dimR, dimA, dimR);
        int32_t dimRAlign = ops::CeilAlign(dimR, static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(DataT)));
        uint32_t srcShape[2] = {static_cast<uint32_t>(dimA), static_cast<uint32_t>(dimRAlign)};
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::AR, true>(computeRes_, computeTensor, srcShape, false);
        bufPool_.FreeTensor(computeTensor);
        CopyOut(yGm_[outputDataOffset], computeRes_, dimA);
        __RIGUtil::SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void PureMove(int64_t inputDataOffset, int64_t outputDataOffset, int32_t dimA)
    {
        LocalTensor<DataT> tensor;
        bufPool_.template AllocTensor<true, DataT>(tensor);
        DataCopyPadExtParams<DataT> copyPadExtParams = {true, 0, 0, 0};
        DataCopyExtParams dataCopyParams{0, 0, 0, 0, 0};
        dataCopyParams.blockCount = dimA;
        dataCopyParams.blockLen = sizeof(DataT);
        dataCopyParams.srcStride = (tiling_->lenP - 1) * sizeof(DataT);
        dataCopyParams.dstStride = 0;
        DataCopyPad(tensor, xGm_[inputDataOffset], dataCopyParams, copyPadExtParams);

        __RIGUtil::SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

        DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
        copyOutParams.blockLen = sizeof(DataT);
        copyOutParams.blockCount = dimA;
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (tiling_->lenR - 1) * sizeof(DataT);
        DataCopyPad(yGm_[outputDataOffset], tensor, copyOutParams);
        __RIGUtil::SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        bufPool_.FreeTensor(tensor);
    }

    __aicore__ inline void CopyOut(const GlobalTensor<DataT>& dst, LocalTensor<PromoteDataT>& ubRes, int32_t dimA)
    {
        LocalTensor<PromoteDataT> outTensor = ubRes;
        // 一次只能搬出一个数，多个repeat
        DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
        copyOutParams.blockLen = sizeof(DataT);
        copyOutParams.blockCount = dimA;
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (tiling_->lenR - 1) * sizeof(DataT);
        // DataCopy
        if constexpr (IsSameType<PromoteDataT, DataT>::value) {
            if (dimA != 1) {
                Scatter(scatterRes_, outTensor, scatterOffset_, 0, ops::CeilAlign(dimA, ELEMENT_ONE_BLOCK_SIZE));
                __RIGUtil::SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(dst, scatterRes_, copyOutParams);
            } else {
                __RIGUtil::SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(dst, outTensor, copyOutParams);
            }
        } else {
            LocalTensor<DataT> outputLocal = outTensor.template ReinterpretCast<DataT>();
            Cast(outputLocal, outTensor, RoundMode::CAST_RINT, ops::CeilAlign(dimA, ELEMENT_ONE_BLOCK_SIZE));
            if (dimA != 1) {
                // each element scatter as block
                LocalTensor<DataT> scatterTensor = scatterRes_.template ReinterpretCast<DataT>();
                Scatter(scatterTensor, outputLocal, scatterOffset_, 0, ops::CeilAlign(dimA, ELEMENT_ONE_BLOCK_SIZE));
                __RIGUtil::SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(dst, scatterTensor, copyOutParams);
            } else {
                __RIGUtil::SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(dst, outputLocal, copyOutParams);
            }
        }
    }

    template <bool isRight>
    __aicore__ inline void CopyInData(
        int64_t index, LocalTensor<DataT>& ubTensor, LocalTensor<PromoteDataT>& computeTensor, int64_t inputDataOffset,
        int32_t dim0, int32_t dim1, int32_t dst0, int32_t dst1)
    {
        if constexpr (isRight) {
            index = index + bisectionPos_;
        }

        int64_t offset = inputDataOffset + index * tiling_->rFactor * tiling_->lenN;
        DataCopyPadExtParams<DataT> copyPadExtParams = {true, 0, 0, 0};
        DataCopyExtParams dataCopyParams{0, 0, 0, 0, 0};
        dataCopyParams.blockCount = dim0;
        dataCopyParams.blockLen = dim1 * sizeof(DataT);
        dataCopyParams.srcStride = (tiling_->lenP - dim1) * sizeof(DataT);
        dataCopyParams.dstStride = (dst1 - dim1) * sizeof(DataT) / BLOCK_SIZE_BYTE;
        DataCopyPad(ubTensor, xGm_[offset], dataCopyParams, copyPadExtParams);
        if constexpr (IsSameType<PromoteDataT, DataT>::value) {
            __RIGUtil::SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            computeTensor = ubTensor;
        } else {
            bufPool_.template AllocTensor<false, PromoteDataT>(computeTensor);
            __RIGUtil::SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            int32_t dst1Align = ops::CeilAlign(dst1, static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(DataT)));
            Cast(computeTensor, ubTensor, RoundMode::CAST_NONE, dst0 * dst1Align);
            bufPool_.FreeTensor(ubTensor);
        }
    }

    __aicore__ inline void UpdateCacheAux(const int64_t cacheID, const int64_t stride, const int64_t count)
    {
        // count A轴的大小 * VL
        uint16_t outerLoopTimes = ops::CeilDiv(
            static_cast<int64_t>(count * sizeof(PromoteDataT)), static_cast<int64_t>(platform::GetVRegSize()));
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
};
} // namespace RepeatInterleaveGrad

#endif // _REPEAT_INTERLEAVE_GRAD_H_