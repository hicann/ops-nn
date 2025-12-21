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
 * \file repeat_interleave_grad_block_split_r.h
 * \brief repeat interleave grad block split R template
 */

#ifndef OPS_MATH_REPEAT_INTERLEAVE_GRAD_OP_KERNEL_V35_REPEAT_INTERLEAVE_GRAD_BLOCK_SPLIT_R_H
#define OPS_MATH_REPEAT_INTERLEAVE_GRAD_OP_KERNEL_V35_REPEAT_INTERLEAVE_GRAD_BLOCK_SPLIT_R_H

#include "kernel_operator.h"
#include "platform.h"
#include "reduce_buf_pool.h"
#include "repeat_interleave_grad_base.h"
#include "repeat_interleave_grad_david_tiling_data.h"

namespace RepeatInterleaveGrad {
using namespace AscendC;

template <typename DataT, typename PromoteDataT, typename IndexT>
class RepeatInterleaveGradBlockSplitR
{
public:
    constexpr static int32_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t RES_BUF_SIZE = 8 * 1024;
    constexpr static uint64_t CACHE_BUF_SIZE = 16 * 1024;
    constexpr static uint64_t WORKSPACE_SIZE_EXT = 64 * 8;
    constexpr static uint64_t BLOCK_SIZE_BYTE = platform::GetUbBlockSize();
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = platform::GetVRegSize() / sizeof(PromoteDataT);
    constexpr static int64_t INIT_VALUE = 0;
    constexpr static int32_t INIT_LENGTH = 64;
    constexpr static float INIT_FLOAT_VALUE = 0.0;

public:
    __aicore__ inline RepeatInterleaveGradBlockSplitR(TPipe& pipe) : pipe_(pipe)
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace,
        const RepeatInterleaveGradDavidTilingData* __restrict tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessRepeatSum();
    __aicore__ inline void CalcUbLoopInfo(int32_t& ubLoop, int32_t& ubFactor, int32_t& ubTailFactor);
    __aicore__ inline void SetCopyParam(
        bool isUbTail, int32_t ubFactor, int32_t ubTailFactor, DataCopyExtParams& copyParam);
    __aicore__ inline void Mte2Repeat(int64_t repeatStart, DataCopyExtParams& copyParam);
    __aicore__ inline void DoDtypeCastRepeatIfNeed(DataCopyExtParams& copyParam);
    __aicore__ inline void DoReduceSum(DataCopyExtParams& copyParam);
    __aicore__ inline void Mte3Repeat();
    __aicore__ inline void ProcessRepeatInterleaveGrad();
    __aicore__ inline void ProcessRepeatInterleaveGradPreProcess();
    __aicore__ inline void Mte2Workspace();
    __aicore__ inline void CalcCoreOffsetR();
    __aicore__ inline void ProcessRepeatInterleaveGradProcess();
    __aicore__ inline void ProcessRepeatBlock(int32_t repeatFactor, LocalTensor<IndexT>& curRepeat);
    __aicore__ inline void ProcessZeroR(int64_t outputDataOffset, int32_t dimA);
    __aicore__ inline void SingleRedeuceSum(
        int64_t inputDataOffset, int64_t outputDataOffset, int32_t dimR, int32_t dimA);
    __aicore__ inline void BinaryRedeuceSum(
        int64_t inputDataOffset, int64_t outputDataOffset, int32_t dim0, int32_t dim1);
    template <bool isRight>
    __aicore__ inline void CopyInData(
        int64_t index, LocalTensor<DataT>& ubTensor, LocalTensor<PromoteDataT>& computeTensor, int64_t inputDataOffset,
        int32_t dim0, int32_t dim1);
    __aicore__ inline void CopyOut(const GlobalTensor<DataT>& dst, LocalTensor<PromoteDataT>& ubRes, int32_t dimA);
    __aicore__ inline void UpdateCacheAux(const int64_t cacheID, const int64_t stride, const int64_t count);

private:
    int32_t blockIdx_ = 0;
    int64_t coreOffsetP_ = 0; // 累计repeat，结果为p的offset

    ReduceBufPool bufPool_;

    GlobalTensor<DataT> xGm_;
    GlobalTensor<DataT> yGm_;
    GlobalTensor<IndexT> repeatGm_;
    GlobalTensor<int64_t> workspaceGm_;
    TBuf<> repeatBuf_;
    TBuf<> repeatCastBuf_;
    TBuf<> workspaceBuf_;
    TBuf<> tempResQue_;
    TBuf<> tempBufQue_;
    TBuf<> tempUbQue_;
    TBuf<> repeatInUbBuf_;
    TBuf<> initBuf_;

    LocalTensor<PromoteDataT> computeRes_;
    LocalTensor<PromoteDataT> tempBuf_;
    LocalTensor<PromoteDataT> tempUb_;
    LocalTensor<IndexT> repeatInUb_;

    const RepeatInterleaveGradDavidTilingData* tiling_;
    TPipe& pipe_;

    int64_t lastRepeatCumsumVal_ = 0; // p offset
    int64_t lastRepeatCount_ = 0;     // r offset
    int64_t mInputOffset_ = 0;
    int64_t mOutputOffset_ = 0;
    int32_t rCount_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t cacheCount_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheStride_ = 0;
    event_t eventId_;
};

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::Init(
    GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace,
    const RepeatInterleaveGradDavidTilingData* __restrict tiling)
{
    tiling_ = tiling;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    xGm_.SetGlobalBuffer((__gm__ DataT*)x);
    yGm_.SetGlobalBuffer((__gm__ DataT*)y);
    repeatGm_.SetGlobalBuffer((__gm__ IndexT*)repeats);
    workspaceGm_.SetGlobalBuffer((__gm__ int64_t*)workspace);
    if (blockIdx_ == 0) {
        pipe_.InitBuffer(initBuf_, WORKSPACE_SIZE_EXT);
        LocalTensor<int64_t> initBufTensor = initBuf_.Get<int64_t>();
        Duplicate<int64_t>(initBufTensor, INIT_VALUE, INIT_LENGTH);
        SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        DataCopy(workspaceGm_, initBufTensor, INIT_LENGTH);
    }
    SyncAll();
    // 先初始化一阶段buff：计算各个核上repeat的sum，搬出到workspace
    pipe_.InitBuffer(repeatBuf_, tiling_->repeatSumBufferSize);
    if constexpr (!IsSameType<IndexT, int64_t>::value) {
        this->pipe_.InitBuffer(repeatCastBuf_, tiling_->repeatSumBufferSize * DOUBLE_BUFFER);
    }
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }

    // step1: 先搬入repeat，计算当前核上所有repeat的sum，搬出到workspace，供后续计算核上repeat的偏移量使用
    ProcessRepeatSum();

    // step2: 全核同步，保证各个核计算repeat的sum都全部完成
    SyncAll();

    // step3: 各个核上进行repeatInterleaveGrad的逻辑运算
    ProcessRepeatInterleaveGrad();
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::ProcessRepeatSum()
{
    // 因为M也参与分核，只需要一组M(即repeat的分核数)来计算repeat的sum即可，其他的核直接不处理返回
    if (blockIdx_ >= tiling_->repeatBlockPara.blockCount) {
        return;
    }

    /* 计算UB轮询的信息 */
    int32_t ubLoop = 0;
    int32_t ubFactor = 0;
    int32_t ubTailFactor = 0;
    CalcUbLoopInfo(ubLoop, ubFactor, ubTailFactor);

    /* UB上的轮询 */
    for (int32_t i = 0; i < ubLoop; i++) {
        // 计算repeat搬入的起始偏移量
        int64_t repeatStart = blockIdx_ * tiling_->repeatBlockPara.blockFactor + i * ubFactor;

        // 是否是最后一次UB轮询
        bool isTailUbLoop = (i == ubLoop - 1);

        // 计算copy参数
        DataCopyExtParams copyParam;
        SetCopyParam(isTailUbLoop, ubFactor, ubTailFactor, copyParam);

        // 搬入Repeat，GM->UB
        Mte2Repeat(repeatStart, copyParam);

        // 如果是int32需要转换成int64，因为reduceSum只支持int64
        DoDtypeCastRepeatIfNeed(copyParam);

        // 执行reduceSum求和
        DoReduceSum(copyParam);

        // 搬出Repeat，UB->Workspace，这里使用atomidAdd
        Mte3Repeat();
    }
    int64_t offset = blockIdx_ / 8;
    DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        workspaceGm_[offset]);
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::CalcUbLoopInfo(
    int32_t& ubLoop, int32_t& ubFactor, int32_t& ubTailFactor)
{
    if (blockIdx_ < tiling_->repeatBlockPara.blockCount - 1) { // 主核
        ubLoop = tiling_->repeatSumUbPara.mainCoreUbPara.ubCount;
        ubFactor = tiling_->repeatSumUbPara.mainCoreUbPara.ubFactor;
        ubTailFactor = tiling_->repeatSumUbPara.mainCoreUbPara.ubTailFactor;
    } else { // 尾核
        ubLoop = tiling_->repeatSumUbPara.tailCoreUbPara.ubCount;
        ubFactor = tiling_->repeatSumUbPara.tailCoreUbPara.ubFactor;
        ubTailFactor = tiling_->repeatSumUbPara.tailCoreUbPara.ubTailFactor;
    }
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::SetCopyParam(
    bool isUbTail, int32_t ubFactor, int32_t ubTailFactor, DataCopyExtParams& copyParam)
{
    copyParam.blockCount = 1;
    copyParam.blockLen = ubFactor * sizeof(IndexT);
    if (isUbTail) { // UB尾块
        copyParam.blockLen = ubTailFactor * sizeof(IndexT);
    }
    copyParam.srcStride = 0;
    copyParam.dstStride = 0;
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::Mte2Repeat(
    int64_t repeatStart, DataCopyExtParams& copyParam)
{
    DataCopyPadExtParams<IndexT> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyPad<IndexT>(repeatBuf_.Get<IndexT>(), repeatGm_[repeatStart], copyParam, dataCopyPadExtParams);

    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::DoDtypeCastRepeatIfNeed(
    DataCopyExtParams& copyParam)
{
    // 如果不需要转换int64，直接返回
    if constexpr (IsSameType<IndexT, int64_t>::value) {
        return;
    }

    Cast(
        repeatCastBuf_.Get<int64_t>(), repeatBuf_.Get<IndexT>(), RoundMode::CAST_NONE,
        copyParam.blockCount * copyParam.blockLen / sizeof(IndexT));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::DoReduceSum(
    DataCopyExtParams& copyParam)
{
    LocalTensor<int64_t> repeatTensor;
    if constexpr (IsSameType<IndexT, int64_t>::value) {
        repeatTensor = repeatBuf_.Get<IndexT>();
    } else {
        repeatTensor = repeatCastBuf_.Get<int64_t>();
    }

    ReduceSum<int64_t>(repeatTensor, repeatTensor, repeatTensor, copyParam.blockLen / sizeof(IndexT));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::Mte3Repeat()
{
    LocalTensor<int64_t> repeatTensor;
    if constexpr (IsSameType<IndexT, int64_t>::value) {
        repeatTensor = repeatBuf_.Get<IndexT>();
    } else {
        repeatTensor = repeatCastBuf_.Get<int64_t>();
    }
    __gm__ int64_t* workspaceAddr = workspaceGm_.GetPhyAddr(blockIdx_);

    SetFlag<HardEvent::V_S>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    WaitFlag<HardEvent::V_S>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    AtomicAdd<int64_t>(workspaceAddr, repeatTensor.GetValue(0));
    SetFlag<HardEvent::S_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    WaitFlag<HardEvent::S_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::ProcessRepeatInterleaveGrad()
{
    // step1: 重置阶段一buff，为阶段二重新分配buffer
    ProcessRepeatInterleaveGradPreProcess();

    // step2: 搬入workspace，用于计算各个核repeat的起始偏移
    Mte2Workspace();

    // step3: 计算当前核处理的R轴的起始偏移量
    CalcCoreOffsetR();

    // step4: 按照R的偏移量开始执行RepeatInterleaveGrad
    ProcessRepeatInterleaveGradProcess();
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void
RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::ProcessRepeatInterleaveGradPreProcess()
{
    pipe_.Reset();

    int32_t computeBufferNum = 0;
    if constexpr (!IsSameType<DataT, PromoteDataT>::value) {
        computeBufferNum = DOUBLE_BUFFER;
    }
    bufPool_.template Init<DataT, PromoteDataT>(&pipe_, DOUBLE_BUFFER, computeBufferNum, tiling_->basicBlockSize);
    pipe_.InitBuffer(tempResQue_, RES_BUF_SIZE);
    computeRes_ = tempResQue_.template Get<PromoteDataT>(); // ub内A轴大小

    pipe_.InitBuffer(tempBufQue_, CACHE_BUF_SIZE); // ub内二分需要的buf块
    tempBuf_ = tempBufQue_.template Get<PromoteDataT>();

    pipe_.InitBuffer(tempUbQue_, BLOCK_SIZE_BYTE);
    tempUb_ = tempUbQue_.template Get<PromoteDataT>();

    pipe_.InitBuffer(repeatInUbBuf_, tiling_->repeatBufferSize * DOUBLE_BUFFER); // 每次搬入的repeat的大小
    repeatInUb_ = repeatInUbBuf_.template Get<IndexT>();

    pipe_.InitBuffer(workspaceBuf_, WORKSPACE_SIZE_EXT);
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::Mte2Workspace()
{
    DataCopyPadExtParams<int64_t> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams copyParam;
    copyParam.blockCount = 1;
    copyParam.blockLen = WORKSPACE_SIZE_EXT;
    copyParam.srcStride = 0;
    copyParam.dstStride = 0;
    DataCopyPad<int64_t>(workspaceBuf_.Get<int64_t>(), workspaceGm_, copyParam, dataCopyPadExtParams);

    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::CalcCoreOffsetR()
{
    coreOffsetP_ = blockIdx_ % tiling_->repeatBlockPara.blockCount;
    if (coreOffsetP_ != 0) {
        ReduceSum<int64_t>(
            workspaceBuf_.Get<int64_t>(), workspaceBuf_.Get<int64_t>(), workspaceBuf_.Get<int64_t>(), coreOffsetP_);
        SetFlag<HardEvent::V_S>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        WaitFlag<HardEvent::V_S>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        coreOffsetP_ = workspaceBuf_.Get<int64_t>().GetValue(0);
    }
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void
RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::ProcessRepeatInterleaveGradProcess()
{
    // 循环读 repeat 128B
    LocalTensor<IndexT> curRepeat = repeatInUb_[0];

    // 确定第几个m,算出m的起始偏移
    int32_t mIndex = blockIdx_ / tiling_->repeatBlockPara.blockCount;
    int32_t rIndexInM = blockIdx_ % tiling_->repeatBlockPara.blockCount;
    mInputOffset_ = mIndex * tiling_->lenP * tiling_->lenN;
    mOutputOffset_ = mIndex * tiling_->lenR * tiling_->lenN;
    // p和r的起始offset, coreOffsetP_为p的起始offset
    lastRepeatCumsumVal_ = coreOffsetP_;
    lastRepeatCount_ = rIndexInM * tiling_->repeatBlockPara.blockFactor;

    UbParaUnit repeatLoopRIGUbPara = rIndexInM == tiling_->repeatBlockPara.blockCount - 1 ?
                                         tiling_->repeatLoopPara.tailCoreUbPara :
                                         tiling_->repeatLoopPara.mainCoreUbPara;
    int32_t repeatFactor = repeatLoopRIGUbPara.ubFactor;
    for (int32_t j = 0; j < repeatLoopRIGUbPara.ubCount - 1; j++) { // repeat.o
        ProcessRepeatBlock(repeatFactor, curRepeat);
    } // 完成repeat 切 128B之后整块的 repeat.i 的处理

    // repeat.i 尾块
    repeatFactor = repeatLoopRIGUbPara.ubTailFactor;
    ProcessRepeatBlock(repeatFactor, curRepeat);
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::ProcessRepeatBlock(
    int32_t repeatFactor, LocalTensor<IndexT>& curRepeat)
{
    // MTE2 搬入repeat, 计算前缀和得到索引并缓存 -> rBuf
    DataCopyPadExtParams<IndexT> copyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams repeatCopyParams{0, 0, 0, 0, 0};
    repeatCopyParams.blockCount = 1;
    repeatCopyParams.blockLen = repeatFactor * sizeof(IndexT);
    repeatCopyParams.srcStride = 0;
    repeatCopyParams.dstStride = 0;
    DataCopyPad(curRepeat, this->repeatGm_[lastRepeatCount_], repeatCopyParams, copyPadExtParams);
    eventId_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId_);
    WaitFlag<HardEvent::MTE2_S>(eventId_);

    // 遍历搬入的repeat -> r
    IndexT r = 0;
    for (int32_t k = 0; k < repeatFactor; k++) {
        int64_t pOffset = lastRepeatCumsumVal_ * tiling_->lenN;
        int64_t rOffset = lastRepeatCount_ * tiling_->lenN;
        r = curRepeat.GetValue(k);
        // 不切分n
        int32_t nFactor = tiling_->lenN;

        int64_t inputDataOffset = mInputOffset_ + pOffset;
        int64_t outputDataOffset = mOutputOffset_ + rOffset;
        if (r == 0) {
            ProcessZeroR(outputDataOffset, nFactor);
        } else if (r < tiling_->rFactor) {
            SingleRedeuceSum(inputDataOffset, outputDataOffset, r, nFactor);
        } else { // 正常切分r和N的流程
            // 计算二分参数
            rCount_ = ops::CeilDiv(static_cast<int32_t>(r), tiling_->rFactor);
            bisectionPos_ = __RIGUtil::FindNearestPower2(rCount_);
            cacheCount_ = __RIGUtil::CalLog2(bisectionPos_) + 1;
            bisectionTail_ = rCount_ - bisectionPos_;
            BinaryRedeuceSum(inputDataOffset, outputDataOffset, r, nFactor);
        }
        lastRepeatCumsumVal_ += r;
        lastRepeatCount_ += 1;
    } // 完成一个repeatFactor个repeat值的处理
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::ProcessZeroR(
    int64_t outputDataOffset, int32_t dimA)
{
    Duplicate<PromoteDataT>(computeRes_, INIT_FLOAT_VALUE, dimA);
    CopyOut(yGm_[outputDataOffset], computeRes_, dimA);
    SetFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    WaitFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::SingleRedeuceSum(
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
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::BinaryRedeuceSum(
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
    SetFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    WaitFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
}

template <typename DataT, typename PromoteDataT, typename IndexT>
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::CopyOut(
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
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::CopyInData(
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
__aicore__ inline void RepeatInterleaveGradBlockSplitR<DataT, PromoteDataT, IndexT>::UpdateCacheAux(
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

#endif // OPS_MATH_REPEAT_INTERLEAVE_GRAD_OP_KERNEL_V35_REPEAT_INTERLEAVE_GRAD_BLOCK_SPLIT_R_H