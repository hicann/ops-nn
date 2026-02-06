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
 * \file gather_v2_ga_all_load.h
 * \brief
 */
#ifndef GATHER_V2_GA_ALL_LOAD_H
#define GATHER_V2_GA_ALL_LOAD_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "basic_api/kernel_vec_intf.h"
#include "micro_api/kernel_micro_intf.h"
#include "op_kernel/platform_util.h"

namespace gatherv2 {
using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t HELP_BUFFER_SIZE = 256;
constexpr int32_t GATHER_ENABLE_SIZE = 512;

template <typename INDICES_T, const bool NIS>
class Gatherv2GaAllLoad
{
public:
    __aicore__ inline Gatherv2GaAllLoad(TPipe *pipe): pipe_(pipe){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, const GatherV2GaAllLoadTilingData* tilingData);
    __aicore__ inline void GenIndexBuf();
    __aicore__ inline void Process();
    __aicore__ inline void IndicesProcess(int32_t indicesNumPro, int32_t indicesNumOffset);
    __aicore__ inline void xProcess(int32_t gaCoreOffset, int32_t gaNumPro);
    __aicore__ inline void yProcess(
        int32_t indicesNumPro, int32_t indicesNumOffset, int32_t gaNumPro, int32_t gaCoreOffset);
    __aicore__ inline void GatherProcessVf(
        int32_t indicesUbOffset, int32_t indicesNumCurPro, __local_mem__ int32_t* indicesAddr,
        __local_mem__ int8_t* xAddr, __local_mem__ int8_t* yAddr);
    __aicore__ inline void FixIndicesVf(__local_mem__ INDICES_T* indicesAddr, int32_t indicesNumPro);
    __aicore__ inline void CopyInIndices(LocalTensor<INDICES_T>& indicesLocal, int32_t burstLen, int32_t coreOffset);
    __aicore__ inline void CopyInX(LocalTensor<int8_t>& xLocal, int32_t gaCoreOffset, int32_t gaNumPro);
    __aicore__ inline void CopyOutY(int32_t nBurst, int32_t indicesCoreOffset, int32_t pCoreOffset, int32_t gaNumPro);
    __aicore__ inline void GatherProcessVfWithGA(
        int32_t indicesNumCurPro, uint16_t gaNumPro, __local_mem__ int32_t* curIndicesAddr, __local_mem__ int8_t* xAddr,
        __local_mem__ int8_t* yAddr);
    __aicore__ inline void GatherProcessVfWithGA2(
        int32_t indicesNumCurPro, uint16_t gaNumPro, __local_mem__ int32_t* curIndicesAddr, __local_mem__ int8_t* xAddr,
        __local_mem__ int8_t* yAddr);
    __aicore__ inline void InitializationX(int32_t xBufGaNum);

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe *pipe_;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> tmpIndexBuf_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue_;
    const GatherV2GaAllLoadTilingData* tilingData_;

    int32_t blockIdx_;
    bool enableGather_;

    int64_t curCoreIndicesNum_;
    int64_t curCoreGaNum_;

    int64_t xGmOffset_;
    int64_t indicesGmOffset_;
};

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, const GatherV2GaAllLoadTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    // 先分核在分UB
    int64_t pIndex = blockIdx_ / tilingData_->indicesOuter;
    int64_t indicesIndex = blockIdx_ % tilingData_->indicesOuter;

    indicesGmOffset_ = indicesIndex * tilingData_->normalCoreIndicesNum;
    xGmOffset_ = pIndex * tilingData_->normalCoreGaNum;

    curCoreIndicesNum_ = (indicesIndex + 1 == tilingData_->indicesOuter) ? tilingData_->tailCoreIndicesNum :
                                                                           tilingData_->normalCoreIndicesNum;
    curCoreGaNum_ = (pIndex + 1 == tilingData_->pOuter) ? tilingData_->tailCoreGaNum : tilingData_->normalCoreGaNum;

    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    xGm_.SetGlobalBuffer((__gm__ int8_t*)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y);

    pipe_->InitBuffer(xBuf_, tilingData_->xBufferSize);
    pipe_->InitBuffer(indicesBuf_, tilingData_->indicesBufferSize);
    pipe_->InitBuffer(tmpIndexBuf_, HELP_BUFFER_SIZE);
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tilingData_->yBufferSize);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::GenIndexBuf()
{
    LocalTensor<int32_t> helpTensor = tmpIndexBuf_.Get<int32_t>();
    __local_mem__ int32_t* helpAddr = (__local_mem__ int32_t*)helpTensor.GetPhyAddr();
    int32_t colFactor = Ops::Base::GetUbBlockSize() / sizeof(int32_t);
    int32_t colAlign = tilingData_->aSizeAligned / sizeof(int32_t);

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
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::IndicesProcess(
    int32_t indicesNumPro, int32_t indicesNumOffset)
{
    event_t eventIdMTE3toMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3toMTE2);

    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3toMTE2);
    LocalTensor<INDICES_T> indicesTensor = indicesBuf_.Get<INDICES_T>();
    CopyInIndices(indicesTensor, indicesNumPro, indicesNumOffset);
    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    __local_mem__ INDICES_T* indicesAddr = (__local_mem__ INDICES_T*)indicesTensor.GetPhyAddr();
    FixIndicesVf(indicesAddr, indicesNumPro);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::xProcess(int32_t gaCoreOffset, int32_t gaNumPro)
{
    event_t eventIdMTE3toMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3toMTE2);

    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3toMTE2);
    LocalTensor<int8_t> xLocal = xBuf_.Get<int8_t>();
    CopyInX(xLocal, gaCoreOffset, gaNumPro);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::InitializationX(int32_t xBufGaNum)
{
    LocalTensor<int8_t> xLocal = xBuf_.Get<int8_t>();
    __local_mem__ int8_t* xAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
    uint32_t aSizeAligned = tilingData_->aSizeAligned;
    uint32_t gaOffset = (tilingData_->gSize + 1) * aSizeAligned;
    uint16_t computeSize = Ops::Base::GetVRegSize();
    uint16_t repeatimes = (aSizeAligned + computeSize - 1) / computeSize;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> zeroConstReg;
        AscendC::MicroAPI::Duplicate(zeroConstReg, int8_t(0));
        MicroAPI::MaskReg preg;
        for (uint16_t g = 0; g < static_cast<uint16_t>(xBufGaNum); g++) {
            uint32_t sreg = aSizeAligned;
            for (uint16_t r = 0; r < repeatimes; r++) {
                preg = MicroAPI::UpdateMask<int8_t>(sreg);
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int8_t>(r, computeSize);
                MicroAPI::DataCopy(xAddr, zeroConstReg, offset, preg);
            }
            xAddr += gaOffset;
        }
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::yProcess(
    int32_t indicesNumPro, int32_t indicesNumOffset, int32_t gaNumPro, int32_t gaCoreOffset)
{
    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    LocalTensor<INDICES_T> indicesTensor = indicesBuf_.Get<INDICES_T>();
    __local_mem__ INDICES_T* indicesAddr = (__local_mem__ INDICES_T*)indicesTensor.GetPhyAddr();
    LocalTensor<int8_t> xTensor = xBuf_.Get<int8_t>();
    __local_mem__ int8_t* xAddr = (__local_mem__ int8_t*)xTensor.GetPhyAddr();

    int32_t indicesNumToDealWithGa = tilingData_->yBufferSize / (tilingData_->aSizeAligned * gaNumPro);
    if (enableGather_) {
        uint16_t vRegBlockNum = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
        indicesNumToDealWithGa = indicesNumToDealWithGa / vRegBlockNum * vRegBlockNum;
    }

    if (indicesNumToDealWithGa != 0) {
        int32_t yloopCountWithGa = (indicesNumPro + indicesNumToDealWithGa - 1) / indicesNumToDealWithGa;

        for (int32_t y = 0; y < yloopCountWithGa; y++) {
            LocalTensor<int8_t> yLocal = yQueue_.AllocTensor<int8_t>();
            __local_mem__ int8_t* yAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
            int32_t indicesNumCurPro = y == (yloopCountWithGa - 1) ?
                                           indicesNumPro - (yloopCountWithGa - 1) * indicesNumToDealWithGa :
                                           indicesNumToDealWithGa;
            int32_t indicesUbOffset = y * indicesNumToDealWithGa;
            __local_mem__ int32_t* curIndicesAddr = (__local_mem__ int32_t*)indicesAddr + indicesUbOffset;
            if (enableGather_) {
                GatherProcessVfWithGA2(
                    indicesNumCurPro, gaNumPro, (__local_mem__ int32_t*)curIndicesAddr, xAddr, yAddr);
            } else {
                GatherProcessVfWithGA(indicesNumCurPro, gaNumPro, (__local_mem__ int32_t*)curIndicesAddr, xAddr, yAddr);
            }
            yQueue_.EnQue<int8_t>(yLocal);
            CopyOutY(indicesNumCurPro, indicesNumOffset + y * indicesNumToDealWithGa, gaCoreOffset, gaNumPro);
        }
    } else {
        int32_t yBufANum = tilingData_->yBufferSize / tilingData_->aSizeAligned;
        int32_t yloopCount = (indicesNumPro + yBufANum - 1) / yBufANum;

        for (int32_t gaIndex = 0; gaIndex < gaNumPro; gaIndex++) {
            for (int32_t y = 0; y < yloopCount; y++) {
                LocalTensor<int8_t> yLocal = yQueue_.AllocTensor<int8_t>();
                __local_mem__ int8_t* yAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
                int32_t indicesNumCurPro =
                    y == (yloopCount - 1) ? indicesNumPro - (yloopCount - 1) * yBufANum : yBufANum;
                int32_t indicesUbOffset = y * yBufANum;
                __local_mem__ int8_t* xAddrWithOffset =
                    xAddr + gaIndex * (tilingData_->gSize + 1) * tilingData_->aSizeAligned;
                GatherProcessVf(
                    indicesUbOffset, indicesNumCurPro, (__local_mem__ int32_t*)indicesAddr, xAddrWithOffset, yAddr);
                yQueue_.EnQue<int8_t>(yLocal);
                CopyOutY(indicesNumCurPro, indicesNumOffset + y * yBufANum, gaCoreOffset + gaIndex, 1);
            }
        }
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::GatherProcessVf(
    int32_t indicesUbOffset, int32_t indicesNumCurPro, __local_mem__ int32_t* indicesAddr, __local_mem__ int8_t* xAddr,
    __local_mem__ int8_t* yAddr)
{
    uint16_t computeSize = Ops::Base::GetVRegSize();
    uint16_t repeatimes = (tilingData_->aSize + computeSize - 1) / computeSize;
    uint32_t aSize = tilingData_->aSize;
    uint32_t aSizeAligned = tilingData_->aSizeAligned;
    __local_mem__ int8_t* curYAddr = yAddr;
    __local_mem__ int32_t* curIndicesAddr = indicesAddr + indicesUbOffset;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int8_t> vregTemp;
        MicroAPI::MaskReg preg;

        for (uint16_t indices = 0; indices < static_cast<uint16_t>(indicesNumCurPro); indices++) {
            uint32_t indicesValue = (curIndicesAddr[indices]);
            uint32_t sreg = aSize;
            __local_mem__ int8_t* curXAddr = xAddr + indicesValue;
            for (uint16_t r = 0; r < repeatimes; r++) {
                preg = MicroAPI::UpdateMask<int8_t>(sreg);
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int8_t>(r, computeSize);
                MicroAPI::DataCopy(vregTemp, curXAddr, offset);
                MicroAPI::DataCopy(curYAddr, vregTemp, offset, preg);
            }
            curYAddr += aSizeAligned;
        }
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::GatherProcessVfWithGA(
    int32_t indicesNumCurPro, uint16_t gaNumPro, __local_mem__ int32_t* curIndicesAddr, __local_mem__ int8_t* xAddr,
    __local_mem__ int8_t* yAddr)
{
    uint16_t computeSize = Ops::Base::GetVRegSize();
    uint16_t repeatimes = (tilingData_->aSize + computeSize - 1) / computeSize;
    uint32_t aSize = tilingData_->aSize;
    uint32_t aSizeAligned = tilingData_->aSizeAligned;
    uint32_t gaOffset = (tilingData_->gSize + 1) * aSizeAligned;
    uint32_t yOffset = indicesNumCurPro * aSizeAligned;

    if (repeatimes == 1) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int8_t> vregTemp;
            MicroAPI::MaskReg preg;

            for (uint16_t indices = 0; indices < static_cast<uint16_t>(indicesNumCurPro); indices++) {
                uint32_t indicesValue = (curIndicesAddr[indices]);
                __local_mem__ int8_t* curYAddr = yAddr;
                __local_mem__ int8_t* curXAddr = xAddr + indicesValue;
                for (uint16_t ga = 0; ga < gaNumPro; ga++) {
                    uint32_t sreg = aSize;
                    preg = MicroAPI::UpdateMask<int8_t>(sreg);
                    MicroAPI::DataCopy(vregTemp, curXAddr);
                    MicroAPI::DataCopy(curYAddr, vregTemp, preg);
                    curXAddr += gaOffset;
                    curYAddr += yOffset;
                }
                yAddr += aSizeAligned;
            }
        }
    } else {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int8_t> vregTemp;
            MicroAPI::MaskReg preg;

            for (uint16_t indices = 0; indices < static_cast<uint16_t>(indicesNumCurPro); indices++) {
                uint32_t indicesValue = (curIndicesAddr[indices]);
                __local_mem__ int8_t* curYAddr = yAddr;
                __local_mem__ int8_t* curXAddr = xAddr + indicesValue;
                for (uint16_t ga = 0; ga < gaNumPro; ga++) {
                    uint32_t sreg = aSize;
                    for (uint16_t r = 0; r < repeatimes; r++) {
                        preg = MicroAPI::UpdateMask<int8_t>(sreg);
                        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int8_t>(r, computeSize);
                        MicroAPI::DataCopy(vregTemp, curXAddr, offset);
                        MicroAPI::DataCopy(curYAddr, vregTemp, offset, preg);
                    }
                    curXAddr += gaOffset;
                    curYAddr += yOffset;
                }
                yAddr += aSizeAligned;
            }
        }
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::GatherProcessVfWithGA2(
    int32_t indicesNumCurPro, uint16_t gaNumPro, __local_mem__ int32_t* curIndicesAddr, __local_mem__ int8_t* xAddr,
    __local_mem__ int8_t* yAddr)
{
    constexpr uint16_t b32DtypeSize = 4;
    uint32_t ubBlockSize = Ops::Base::GetUbBlockSize();
    uint32_t aSizeAligned = tilingData_->aSizeAligned;
    uint32_t gaOffset = (tilingData_->gSize + 1) * aSizeAligned / b32DtypeSize;
    uint32_t yOffset = indicesNumCurPro * aSizeAligned / b32DtypeSize;

    uint16_t vRegBlockNum = Ops::Base::GetVRegSize() / ubBlockSize; // 单次加载8个索引
    uint16_t indicesLoopNum = (indicesNumCurPro + vRegBlockNum - 1) / vRegBlockNum;
    uint16_t tailIndices = indicesNumCurPro - (indicesLoopNum - 1) * vRegBlockNum;
    indicesLoopNum -= 1;
    uint16_t aAlignedLoopNum = aSizeAligned / ubBlockSize;
    uint16_t aNumPerLoop = ubBlockSize / b32DtypeSize;
    uint16_t pLoopNum = gaNumPro;

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
        for (uint16_t ga = 0; ga < pLoopNum; ga++) {
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
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::FixIndicesVf(
    __local_mem__ INDICES_T* indicesAddr, int32_t indicesNumPro)
{
    int32_t aSizeAligned = tilingData_->aSizeAligned;
    if (enableGather_) {
        constexpr uint16_t b32DtypeSize = 4;
        aSizeAligned = aSizeAligned / b32DtypeSize;
    }

    int32_t gatherDimSize = tilingData_->gSize;
    uint16_t computeSizeT = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t repeatimes = (indicesNumPro + computeSizeT - 1) / computeSizeT;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));

        AscendC::MicroAPI::RegTensor<int32_t> limitConstReg;
        AscendC::MicroAPI::Duplicate(limitConstReg, int32_t(gatherDimSize));

        AscendC::MicroAPI::RegTensor<int32_t> indicesReg;
        AscendC::MicroAPI::RegTensor<int32_t> tmpReg;

        uint32_t indicesMask = indicesNumPro;
        AscendC::MicroAPI::MaskReg fixMask;
        for (uint16_t i = 0; i < repeatimes; i++) {
            AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<int32_t>(indicesMask);
            if constexpr (std::is_same<INDICES_T, int32_t>::value) {
                AscendC::MicroAPI::DataCopy(indicesReg, indicesAddr + i * computeSizeT);
            } else {
                AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo> indicesRegTwo;
                AscendC::MicroAPI::DataCopy(indicesRegTwo, indicesAddr + i * computeSizeT);
                indicesReg = (AscendC::MicroAPI::RegTensor<int32_t>&)indicesRegTwo.reg[0];
            }

            if constexpr (NIS) {
                AscendC::MicroAPI::Compare<int32_t, CMPMODE::LT>(fixMask, indicesReg, zeroConstReg, preg);
                AscendC::MicroAPI::Adds(tmpReg, indicesReg, gatherDimSize, fixMask); // 补偿负索引场景
                Copy<int32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(indicesReg, tmpReg, fixMask);
            }

            AscendC::MicroAPI::Compare<int32_t, CMPMODE::LT>(fixMask, indicesReg, zeroConstReg, preg);
            AscendC::MicroAPI::Duplicate(tmpReg, int32_t(-1), fixMask); // 补偿负越界场景
            Copy<int32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(indicesReg, tmpReg, fixMask);

            AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(fixMask, indicesReg, limitConstReg, preg);
            AscendC::MicroAPI::Duplicate(tmpReg, int32_t(-1), fixMask); // 补偿正越界场景
            Copy<int32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(indicesReg, tmpReg, fixMask);

            AscendC::MicroAPI::Adds(indicesReg, indicesReg, 1, preg);
            AscendC::MicroAPI::Muls(indicesReg, indicesReg, aSizeAligned, preg);
            AscendC::MicroAPI::DataCopy((__local_mem__ int32_t*)indicesAddr + i * computeSizeT, indicesReg, preg);
        }
    }
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::CopyInIndices(
    LocalTensor<INDICES_T>& indicesLocal, int32_t burstLen, int32_t coreOffset)
{
    DataCopyPadExtParams<INDICES_T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = burstLen * sizeof(INDICES_T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(indicesLocal, indicesGm_[indicesGmOffset_ + coreOffset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::CopyInX(
    LocalTensor<int8_t>& xLocal, int32_t gaCoreOffset, int32_t gaNumPro)
{
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = gaNumPro;
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop1SrcStride = tilingData_->gSize * tilingData_->aSize;
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop1DstStride = (tilingData_->gSize + 1) * tilingData_->aSizeAligned;
    loopModeParamsT1.loop2DstStride = 0;

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);

    DataCopyPadExtParams<int8_t> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = tilingData_->gSize;
    dataCoptExtParams.blockLen = tilingData_->aSize;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = (tilingData_->aSizeAligned - tilingData_->aSize) / Ops::Base::GetUbBlockSize();

    DataCopyPad(
        xLocal[tilingData_->aSizeAligned], xGm_[(xGmOffset_ + gaCoreOffset) * tilingData_->gSize * tilingData_->aSize],
        dataCoptExtParams, dataCopyPadExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::CopyOutY(
    int32_t nBurst, int32_t indicesCoreOffset, int32_t pCoreOffset, int32_t gaNumPro)
{
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = gaNumPro;
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop1SrcStride = nBurst * tilingData_->aSizeAligned;
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop1DstStride = tilingData_->indicesSize * tilingData_->aSize;
    loopModeParamsT1.loop2DstStride = 0;

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = tilingData_->aSize;
    dataCoptExtParams.srcStride = (tilingData_->aSizeAligned - tilingData_->aSize) / Ops::Base::GetUbBlockSize();
    dataCoptExtParams.dstStride = 0;

    LocalTensor<int8_t> yLocal = yQueue_.DeQue<int8_t>();
    DataCopyPad(
        yGm_
            [((xGmOffset_ + pCoreOffset) * tilingData_->indicesSize + indicesGmOffset_ + indicesCoreOffset) *
             tilingData_->aSize],
        yLocal, dataCoptExtParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    yQueue_.FreeTensor(yLocal);
}

template <typename INDICES_T, const bool NIS>
__aicore__ inline void Gatherv2GaAllLoad<INDICES_T, NIS>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    int32_t indicesBufEleNum = tilingData_->indicesBufferSize / sizeof(INDICES_T);
    int32_t indicesLoopCount = (curCoreIndicesNum_ + indicesBufEleNum - 1) / indicesBufEleNum;

    int32_t xBufGaNum = tilingData_->xBufferSize / (tilingData_->aSizeAligned * (tilingData_->gSize + 1));
    int32_t xLoopCount = (curCoreGaNum_ + xBufGaNum - 1) / xBufGaNum;

    // 避免选择gatherVF 但是 indices下取整后为零情况
    int32_t indicesNumToDealWithGa = tilingData_->yBufferSize / (tilingData_->aSizeAligned * xBufGaNum);
    uint16_t vRegBlockNum = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    indicesNumToDealWithGa = indicesNumToDealWithGa / vRegBlockNum * vRegBlockNum;

    if (xBufGaNum * tilingData_->aSizeAligned <= GATHER_ENABLE_SIZE && indicesNumToDealWithGa != 0) {
        enableGather_ = true;
        GenIndexBuf();
    } else {
        enableGather_ = false;
    }

    InitializationX(xBufGaNum);
    for (int32_t x = 0; x < xLoopCount; x++) {
        int32_t gaNumPro = x == (xLoopCount - 1) ? curCoreGaNum_ - (xLoopCount - 1) * xBufGaNum : xBufGaNum;
        int32_t gaCoreOffset = x * xBufGaNum;
        xProcess(gaCoreOffset, gaNumPro);

        for (int32_t i = 0; i < indicesLoopCount; i++) {
            int32_t indicesNumPro = i == (indicesLoopCount - 1) ?
                                        curCoreIndicesNum_ - (indicesLoopCount - 1) * indicesBufEleNum :
                                        indicesBufEleNum;
            int32_t indicesNumOffset = i * indicesBufEleNum;
            IndicesProcess(indicesNumPro, indicesNumOffset);

            yProcess(indicesNumPro, indicesNumOffset, gaNumPro, gaCoreOffset);
        }
    }
}
} // namespace gatherv2
#endif // GATHER_V2_GA_ALL_LOAD_H
