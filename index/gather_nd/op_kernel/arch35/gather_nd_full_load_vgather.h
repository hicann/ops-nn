/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_nd_all_load.h
 * \brief
 */
#ifndef GATHER_ND_GA_FULL_LOAD_H
#define GATHER_ND_GA_FULL_LOAD_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace GatherNd {
using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER_VG = 2;
constexpr int32_t HELP_BUFFER_SIZE_VG = 256;
constexpr int32_t HELP_RANK_SIZE = 32;

template <typename INDICES_T, const bool NIS>
class GatherNdAllLoadV
{
public:
    __aicore__ inline GatherNdAllLoadV(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR indices, GM_ADDR y, const GatherNdGaAllLoadTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void GenIndexBuf();
    __aicore__ inline void IndicesColBuf();
    __aicore__ inline void GenRankBuf();
    __aicore__ inline void Process();
    __aicore__ inline void InitXbuf();
    __aicore__ inline void IndicesProcess(int32_t indicesNum, int32_t indicesNumOffset);
    __aicore__ inline void CopyInX();
    __aicore__ inline void yProcess(int32_t indicesNum, int32_t indicesOffset);
    __aicore__ inline void DealIndicesBound(__local_mem__ INDICES_T* indicesAddr, int32_t indicesNum);
    __aicore__ inline void CopyInIndices(LocalTensor<INDICES_T>& indicesLocal, int32_t burstLen, int32_t coreOffset);
    __aicore__ inline void CopyOutY(int32_t nBurst, int32_t indicesCoreOffset);
    __aicore__ inline void DealByVgather(
        int32_t indicesNumCurPro, __local_mem__ int32_t* curIndicesAddr, __local_mem__ int8_t* xAddr,
        __local_mem__ int8_t* yAddr);

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe *pipe_;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> tmpIndexBuf_;
    TBuf<QuePosition::VECCALC> tmpIndicesColBuf_;
    TBuf<QuePosition::VECCALC> tmpRankBuf_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_VG> yQueue_;
    const GatherNdGaAllLoadTilingData* tilingData_;

    int32_t blockIdx_;
    int64_t curCoreIndicesNum_;
    int64_t indicesGmOffset_;
};

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR y, const GatherNdGaAllLoadTilingData* tilingData, TPipe* pipeIn)
{
    pipe_ = pipeIn;

    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    // 先分核再分UB
    int64_t indicesIndex = blockIdx_ % tilingData_->indicesOuter;
    indicesGmOffset_ = indicesIndex * tilingData_->normalCoreIndicesNum;
    curCoreIndicesNum_ = (indicesIndex + 1 == tilingData_->indicesOuter) ? tilingData_->tailCoreIndicesNum : tilingData_->normalCoreIndicesNum;

    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    xGm_.SetGlobalBuffer((__gm__ int8_t*)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y);

    pipe_->InitBuffer(xBuf_, tilingData_->xBufferSize);
    pipe_->InitBuffer(indicesBuf_, tilingData_->indicesBufferSize);
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER_VG, tilingData_->yBufferSize);
    pipe_->InitBuffer(tmpIndexBuf_, HELP_BUFFER_SIZE_VG);
    pipe_->InitBuffer(tmpIndicesColBuf_, HELP_BUFFER_SIZE_VG);
    pipe_->InitBuffer(tmpRankBuf_, HELP_RANK_SIZE);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::GenIndexBuf()
{
    LocalTensor<int32_t> helpTensor = tmpIndexBuf_.Get<int32_t>();
    __local_mem__ int32_t* helpAddr = (__local_mem__ int32_t*)helpTensor.GetPhyAddr();
    int32_t colFactor = platform::GetUbBlockSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> v0;
        AscendC::MicroAPI::RegTensor<int32_t> v1;
        AscendC::MicroAPI::RegTensor<int32_t> vd1;
        AscendC::MicroAPI::RegTensor<int32_t> vd2;
        AscendC::MicroAPI::RegTensor<int32_t> vd3;

        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Duplicate(v1, colFactor, preg);
        AscendC::MicroAPI::Arange(v0, 0);
        AscendC::MicroAPI::Div(vd1, v0, v1, preg);
        AscendC::MicroAPI::Mul(vd2, vd1, v1, preg);
        AscendC::MicroAPI::Sub(vd3, v0, vd2, preg);
        AscendC::MicroAPI::DataCopy(helpAddr, vd3, preg);
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::IndicesColBuf()
{
    LocalTensor<int32_t> helpTensor = tmpIndicesColBuf_.Get<int32_t>();
    __local_mem__ int32_t* helpAddr = (__local_mem__ int32_t*)helpTensor.GetPhyAddr();
    int32_t colFactor = platform::GetUbBlockSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> v0;
        AscendC::MicroAPI::RegTensor<int32_t> vd1;
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Arange(v0, 0);
        AscendC::MicroAPI::Muls(vd1, v0, colFactor, preg);
        AscendC::MicroAPI::DataCopy(helpAddr, vd1, preg);
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::IndicesProcess(
    int32_t indicesNum, int32_t indicesNumOffset)
{
    event_t eventIdMTE3toMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3toMTE2);

    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3toMTE2);
    LocalTensor<INDICES_T> indicesTensor = indicesBuf_.Get<INDICES_T>();
    CopyInIndices(indicesTensor, indicesNum, indicesNumOffset);
    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    __local_mem__ INDICES_T* indicesAddr = (__local_mem__ INDICES_T*)indicesTensor.GetPhyAddr();
    DealIndicesBound(indicesAddr, indicesNum);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::InitXbuf()
{
    LocalTensor<int8_t> xLocal = xBuf_.Get<int8_t>();
    __local_mem__ int8_t* xAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
    uint32_t aSizeAligned = tilingData_->aSizeAligned;
    uint16_t computeSize = platform::GetVRegSize();
    uint16_t repeatimes = (aSizeAligned + computeSize - 1) / computeSize;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> zeroConstReg;
        AscendC::MicroAPI::Duplicate(zeroConstReg, int8_t(0));
        MicroAPI::MaskReg preg;
        uint32_t sreg = aSizeAligned;
        for (uint16_t r = 0; r < repeatimes; r++) {
            preg = MicroAPI::UpdateMask<int8_t>(sreg);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int8_t>(r, computeSize);
            MicroAPI::DataCopy(xAddr, zeroConstReg, offset, preg);
        }
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::yProcess(int32_t indicesNum, int32_t indicesOffset)
{
    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    LocalTensor<INDICES_T> indicesTensor = indicesBuf_.Get<INDICES_T>();
    __local_mem__ INDICES_T* indicesAddr = (__local_mem__ INDICES_T*)indicesTensor.GetPhyAddr();
    LocalTensor<int8_t> xTensor = xBuf_.Get<int8_t>();
    __local_mem__ int8_t* xAddr = (__local_mem__ int8_t*)xTensor.GetPhyAddr();

    int32_t indicesNumByVGa = tilingData_->yBufferSize / tilingData_->aSizeAligned;
    uint16_t vRegBlockNum = platform::GetVRegSize() / platform::GetUbBlockSize();
    indicesNumByVGa = indicesNumByVGa / vRegBlockNum * vRegBlockNum;

    int32_t yloopCountWithGa = (indicesNum + indicesNumByVGa - 1) / indicesNumByVGa;

    for (int32_t y = 0; y < yloopCountWithGa; y++) {
        LocalTensor<int8_t> yLocal = yQueue_.AllocTensor<int8_t>();
        __local_mem__ int8_t* yAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
        int32_t curIndicesNum = y == (yloopCountWithGa - 1) ?
                                        indicesNum - (yloopCountWithGa - 1) * indicesNumByVGa :
                                        indicesNumByVGa;
        int32_t indicesUbOffset = y * indicesNumByVGa;
        __local_mem__ int32_t* curIndicesAddr = (__local_mem__ int32_t*)indicesAddr + indicesUbOffset;
        DealByVgather(curIndicesNum, (__local_mem__ int32_t*)curIndicesAddr, xAddr, yAddr);
        
        yQueue_.EnQue<int8_t>(yLocal);
        CopyOutY(curIndicesNum, indicesOffset + y * indicesNumByVGa);
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::DealByVgather(
    int32_t indicesNumCurPro, __local_mem__ int32_t* curIndicesAddr, __local_mem__ int8_t* xAddr,
    __local_mem__ int8_t* yAddr)
{
    constexpr uint16_t b32DtypeSize = 4;
    uint32_t ubBlockSize = platform::GetUbBlockSize();
    uint32_t aSizeAligned = tilingData_->aSizeAligned;
    uint32_t gaOffset = (tilingData_->gSize + 1) * aSizeAligned / b32DtypeSize;
    uint32_t yOffset = indicesNumCurPro * aSizeAligned / b32DtypeSize;

    uint16_t vRegBlockNum = platform::GetVRegSize() / ubBlockSize; // 单次加载8个索引
    uint16_t indicesLoopNum = (indicesNumCurPro + vRegBlockNum - 1) / vRegBlockNum;
    uint16_t tailIndices = indicesNumCurPro - (indicesLoopNum - 1) * vRegBlockNum;
    indicesLoopNum -= 1;
    uint16_t aAlignedLoopNum = aSizeAligned / ubBlockSize;
    uint16_t aNumPerLoop = ubBlockSize / b32DtypeSize;

    LocalTensor<uint32_t> helpTensor = tmpIndexBuf_.Get<uint32_t>();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();

    uint32_t blockStride = aSizeAligned / ubBlockSize;
    int32_t yInnerOffset = vRegBlockNum * aSizeAligned / b32DtypeSize;
    uint32_t tailANum = tailIndices * aNumPerLoop;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> indicesReg;
        MicroAPI::RegTensor<uint32_t> upIndex;
        MicroAPI::RegTensor<uint32_t> curUpIndex;

        MicroAPI::RegTensor<int32_t> vd0;

        MicroAPI::MaskReg preg = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<int32_t>(tailANum);

        __local_mem__ int32_t* curXAddr = (__local_mem__ int32_t*)xAddr;
        __local_mem__ int32_t* pYAddr = (__local_mem__ int32_t*)yAddr;
        MicroAPI::DataCopy<uint32_t>(upIndex, helpAddr);

        MicroAPI::Copy(curUpIndex, upIndex, preg);
        __local_mem__ int32_t* aYAddr = pYAddr;
        for (uint16_t r = 0; r < aAlignedLoopNum; r++) {
            __local_mem__ uint32_t* indicesAddr = (__local_mem__ uint32_t*)curIndicesAddr;
            __local_mem__ int32_t* curYAddr = aYAddr;
            for (uint16_t indices = 0; indices < indicesLoopNum; indices++) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_E2B_B32>(indicesReg, indicesAddr);

                MicroAPI::Add(indicesReg, indicesReg, curUpIndex, preg);
                MicroAPI::DataCopyGather(vd0, curXAddr, indicesReg, preg);
                MicroAPI::DataCopy<int32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    curYAddr, vd0, blockStride, preg);
                indicesAddr += vRegBlockNum;
                curYAddr += yInnerOffset;
            }
            MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_E2B_B32>(indicesReg, indicesAddr);
            MicroAPI::Add(indicesReg, indicesReg, curUpIndex, pTail);
            MicroAPI::DataCopyGather(vd0, curXAddr, indicesReg, pTail);
            MicroAPI::DataCopy<int32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(curYAddr, vd0, blockStride, pTail);
            MicroAPI::Adds(curUpIndex, curUpIndex, aNumPerLoop, preg);
            aYAddr += aNumPerLoop;
        }
        curXAddr += gaOffset;
        pYAddr += yOffset;
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::DealIndicesBound(
    __local_mem__ INDICES_T* indicesAddr, int32_t indicesNum)
{
    uint16_t rank = tilingData_->rank;
    int32_t aSizeAligned = tilingData_->aSizeAligned;
    constexpr uint16_t b32DtypeSize = 4;
    aSizeAligned = aSizeAligned / b32DtypeSize;

    int32_t indicesStride = platform::GetUbBlockSize() / sizeof(int32_t);

    uint16_t computeSizeT = platform::GetVRegSize() / sizeof(int32_t);
    uint16_t repeatimes = (indicesNum + computeSizeT - 1) / computeSizeT;

    __local_mem__ int32_t* curIndicesAddr = (__local_mem__ int32_t*)indicesAddr;

    LocalTensor<uint32_t> helpTensor = tmpIndicesColBuf_.Get<uint32_t>();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();

    LocalTensor<int32_t> rankTensor = tmpRankBuf_.Get<int32_t>();
    __local_mem__ int32_t* rankAddr = (__local_mem__ int32_t*)rankTensor.GetPhyAddr();
    for (int16_t x=0; x < tilingData_->rank; x++) {
        rankAddr[x] = tilingData_->xShape[x];
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> upIndex;
        MicroAPI::RegTensor<uint32_t> curUpIndex;

        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));

        AscendC::MicroAPI::RegTensor<int32_t> indicesReg;
        AscendC::MicroAPI::RegTensor<int32_t> tmpReg;

        AscendC::MicroAPI::RegTensor<uint32_t> indexIndicesReg;
        AscendC::MicroAPI::RegTensor<int32_t> tmpIndicesReg;

        uint32_t indicesMask = indicesNum;
        
        MicroAPI::DataCopy<uint32_t>(upIndex, helpAddr);
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<int32_t>(indicesMask);

        for (uint16_t i = 0; i < repeatimes; i++) {
            AscendC::MicroAPI::RegTensor<uint32_t> indicesOffset;
            AscendC::MicroAPI::Duplicate(indicesOffset, uint32_t(i * computeSizeT * indicesStride));
            AscendC::MicroAPI::MaskReg fixMask = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALLF>(); // 当前repeat的最终mask，初始为全0
            AscendC::MicroAPI::Duplicate(indicesReg, int32_t(0));

            for (uint16_t k = 0; k < rank; k++) {
                MicroAPI::Copy(curUpIndex, upIndex, preg0);
                AscendC::MicroAPI::MaskReg tmpFixMask;

                int32_t tmpGatherDimSize = rankAddr[k];
                AscendC::MicroAPI::RegTensor<int32_t> tmpLimitConstReg;
                AscendC::MicroAPI::Duplicate(tmpLimitConstReg, int32_t(tmpGatherDimSize));

                MicroAPI::Adds(curUpIndex, curUpIndex, k, preg0);
                
                MicroAPI::Add(indexIndicesReg, indicesOffset, curUpIndex, preg);
                MicroAPI::DataCopyGather(tmpIndicesReg, curIndicesAddr, indexIndicesReg, preg);

                if constexpr (NIS) {
                    AscendC::MicroAPI::Compare<int32_t, CMPMODE::LT>(tmpFixMask, tmpIndicesReg, zeroConstReg, preg);
                    AscendC::MicroAPI::Adds(tmpReg, tmpIndicesReg, tmpGatherDimSize, tmpFixMask); // 补偿负索引场景
                    Copy<int32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(tmpIndicesReg, tmpReg, tmpFixMask);
                }

                AscendC::MicroAPI::Compare<int32_t, CMPMODE::LT>(tmpFixMask, tmpIndicesReg, zeroConstReg, preg); // 负越界
                AscendC::MicroAPI::MaskOr(fixMask, fixMask, tmpFixMask, preg);

                AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(tmpFixMask, tmpIndicesReg, tmpLimitConstReg, preg); // 正越界
                AscendC::MicroAPI::MaskOr(fixMask, fixMask, tmpFixMask, preg);

                AscendC::MicroAPI::Muls(indicesReg, indicesReg, tmpGatherDimSize, preg);
                AscendC::MicroAPI::Add(indicesReg, indicesReg, tmpIndicesReg, preg);
            }
            AscendC::MicroAPI::Duplicate(tmpReg, int32_t(-1), fixMask); // 补偿越界
            Copy<int32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(indicesReg, tmpReg, fixMask);

            AscendC::MicroAPI::Adds(indicesReg, indicesReg, 1, preg);
            AscendC::MicroAPI::Muls(indicesReg, indicesReg, aSizeAligned, preg);
            AscendC::MicroAPI::DataCopy((__local_mem__ int32_t*)indicesAddr + i * computeSizeT, indicesReg, preg); // k合1的indices拷出到indicesBuf
        }
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::CopyInIndices(
    LocalTensor<INDICES_T>& indicesLocal, int32_t burstLen, int32_t coreOffset)
{
    DataCopyPadExtParams<INDICES_T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = burstLen;
    dataCopyExtParams.blockLen =  tilingData_->rank * sizeof(INDICES_T);
    dataCopyExtParams.srcStride = 0;
    dataCopyExtParams.dstStride = 0;
    DataCopyPad(indicesLocal, indicesGm_[(indicesGmOffset_ + coreOffset) * tilingData_->rank], dataCopyExtParams, dataCopyPadExtParams);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::CopyInX()
{
    LocalTensor<int8_t> xLocal = xBuf_.Get<int8_t>();
    DataCopyPadExtParams<int8_t> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = tilingData_->gSize;
    dataCopyExtParams.blockLen = tilingData_->aSize;
    dataCopyExtParams.srcStride = 0;
    dataCopyExtParams.dstStride = (tilingData_->aSizeAligned - tilingData_->aSize) / platform::GetUbBlockSize();

    DataCopyPad(xLocal[tilingData_->aSizeAligned], xGm_, dataCopyExtParams, dataCopyPadExtParams);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::CopyOutY(int32_t nBurst, int32_t indicesCoreOffset)
{
    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = nBurst;
    dataCopyExtParams.blockLen = tilingData_->aSize;
    dataCopyExtParams.srcStride = (tilingData_->aSizeAligned - tilingData_->aSize) / platform::GetUbBlockSize();
    dataCopyExtParams.dstStride = 0;

    LocalTensor<int8_t> yLocal = yQueue_.DeQue<int8_t>();
    DataCopyPad(yGm_[(indicesGmOffset_ + indicesCoreOffset) * tilingData_->aSize], yLocal, dataCopyExtParams);
    yQueue_.FreeTensor(yLocal);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void GatherNdAllLoadV<INDICES_T, NIS>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    int32_t indicesBufEleNum = tilingData_->indicesBufferSize / platform::GetUbBlockSize();
    int32_t indicesLoopCount = (curCoreIndicesNum_ + indicesBufEleNum - 1) / indicesBufEleNum;

    GenIndexBuf();
    IndicesColBuf();
    InitXbuf();
    CopyInX();
    for (int32_t i = 0; i < indicesLoopCount; i++) {
        int32_t indicesNums = i == (indicesLoopCount - 1) ?
                                    curCoreIndicesNum_ - (indicesLoopCount - 1) * indicesBufEleNum :
                                    indicesBufEleNum;
        int32_t indicesNumOffset = i * indicesBufEleNum;
        IndicesProcess(indicesNums, indicesNumOffset);

        yProcess(indicesNums, indicesNumOffset);
    }
}
} // namespace gatherNd
#endif // GATHER_ND_GA_FULL_LOAD_H
