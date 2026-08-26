/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool2d_split_w.h
 * \brief SplitW template: optimized for downsampling with large kernel windows
 *        (wOut<=wIn && hOut<=hIn, kernelH*kernelW>=128). wIn is small enough that
 *        whole input H-rows are loaded; the W-window per output is reduced by the
 *        inner k<kernelW accumulate loop, so arbitrarily large kernels are handled
 *        without the SmallKernel kernel<128 restriction (which would otherwise fall
 *        through to the slow scalar BigKernel path).
 *
 * \arch Ascend950 / A5 / DAV_3510 only, RegBase (Reg) main path, VL = 256 Byte.
 *       [RegBase-native] Host-side gate: AdaptivePool2dBaseTiling::GetShapeAttrsInfo
 *       rejects GetCurNpuArch() != NpuArch::DAV_3510.
 */

#ifndef ADAPTIVE_AVG_POOL2D_SPLIT_W_H_
#define ADAPTIVE_AVG_POOL2D_SPLIT_W_H_

#include "adaptive_avg_pool2d_pooling_base.h"

namespace AdaptivePool2dSplitWNamespace {
using namespace AscendC;
using namespace ops;
using namespace AdaptiveAvgPool2dOp;
using namespace AdaptiveAvgPool2dPoolingBaseNs;

// VF functions must be defined before the class that calls them. Class members the loop needs
// (vlNum_, tilingData_->wOut, ...) are passed in as plain scalars since VF cannot touch `this`.
template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitWAccumulateWVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                            __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                            uint32_t rowBase, uint32_t outBase, uint16_t woNum, uint32_t vlNum,
                                            uint32_t vfLenFp32)
{
    Reg::RegTensor<float> inputReg;
    Reg::RegTensor<float> sumRegtensor;
    Reg::MaskReg preg = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    for (uint16_t w_o = 0; w_o < woNum; w_o++) {
        uint32_t baseOffset = (rowBase + static_cast<uint32_t>(wStartAddr[w_o])) * vlNum;
        uint16_t kernelW = static_cast<uint16_t>(wKerSizeAddr[w_o]);
        uint32_t sumOffset = outBase + static_cast<uint32_t>(w_o) * vlNum;

        Reg::LoadAlign(sumRegtensor, outAddr + sumOffset);
        for (uint16_t k = 0; k < kernelW; k++) {
            uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
            Reg::Add(sumRegtensor, sumRegtensor, inputReg, preg);
        }
        Reg::StoreAlign(outAddr + sumOffset, sumRegtensor, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            Reg::LoadAlign(sumRegtensor, outAddr + sumOffset + vfLenFp32);
            for (uint16_t k = 0; k < kernelW; k++) {
                uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
                Reg::Add(sumRegtensor, sumRegtensor, inputReg, preg);
            }
            Reg::StoreAlign(outAddr + sumOffset + vfLenFp32, sumRegtensor, preg);
        }
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
class AdaptiveAvgPool2dSplitW
    : protected AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitWTilingData> {
    using Base = AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitWTilingData>;

public:
    __aicore__ inline AdaptiveAvgPool2dSplitW(const AdaptivePool2dSplitWTilingData* tilingData, TPipe* pipe)
        : Base(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AccumulateW(int64_t rowOffset, int64_t hoLocal);
    __aicore__ inline void ProcessOneBlock(const BlockParam& blockPara);
};

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitW<T, ID_T, NC_FACTOR>::Init(GM_ADDR x, GM_ADDR y)
{
    if (!this->InitCommon(x, y)) {
        return;
    }
    uint64_t dataBlock = AAP_UB_BLOCK_SIZE;
    uint64_t wBufSize = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->wOut) * sizeof(int32_t), dataBlock);
    this->InitBuffers(wBufSize, this->wInAlign_);
}

// Scalar W-kernel indices are read from UB pointers inside the vector scope
// (stack-array dynamic indexing would trigger "Unsupported Inst must be hoisted").
// [RegBase-native] Compiler limitation seen with CANN 9.0.0 (V100R001C10SPC001B250).
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitW<T, ID_T, NC_FACTOR>::AccumulateW(int64_t rowOffset, int64_t hoLocal)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();

    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t rowBase = static_cast<uint32_t>(rowOffset);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * this->wOutAlign_ * this->vlNum_;
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);

    SplitWAccumulateWVf<T, NC_FACTOR>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, rowBase, outBase, woNum,
                                      this->vlNum_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitW<T, ID_T, NC_FACTOR>::ProcessOneBlock(const BlockParam& blockPara)
{
    int64_t ncIdx = blockPara.ncIdx;
    int64_t ncNum = blockPara.ncNum;
    int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;
    int64_t hoNum = blockPara.hoNum;
    int64_t hIn = this->tilingData_->hIn;
    int64_t hOut = this->tilingData_->hOut;

    this->CalWKernelInfo();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    this->ClearOutBuf();

    int64_t hiMin = 0;
    int64_t hiMax = 0; // exclusive
    this->CalHiRange(hoGlobalStart, hoNum, hiMin, hiMax);

    int64_t hoCursor = 0;

    for (int64_t hiBase = hiMin; hiBase < hiMax; hiBase += this->tilingData_->hiFactor) {
        int64_t hiBatch = this->tilingData_->hiFactor;
        if (hiBase + hiBatch > hiMax) {
            hiBatch = hiMax - hiBase;
        }
        this->CopyInputBatch(ncIdx, ncNum, hiBase, hiBatch);
        this->TransInputBatch(hiBatch);
        for (int64_t hiOffset = 0; hiOffset < hiBatch; hiOffset++) {
            int64_t hi = hiBase + hiOffset;
            while (hoCursor < hoNum && this->CalHoEnd(hoGlobalStart + hoCursor) <= hi) {
                hoCursor++;
            }
            for (int64_t hoLocal = hoCursor; hoLocal < hoNum; hoLocal++) {
                if (this->CalHoStart(hoGlobalStart + hoLocal) <= hi) {
                    AccumulateW(hiOffset * this->wInAlign_, hoLocal);
                } else {
                    break;
                }
            }
        }
    }

    for (int64_t hoLocal = 0; hoLocal < hoNum; hoLocal++) {
        int64_t hoGlobal = hoGlobalStart + hoLocal;
        int64_t hStart = (hoGlobal * hIn) / hOut;
        int64_t hEnd = ((hoGlobal + 1) * hIn + hOut - 1) / hOut;
        int64_t kernelH = hEnd - hStart;
        this->CalAvgOneHo(kernelH, hoLocal);
    }

    this->TransOut(hoNum);
    this->CopyOut(ncIdx, ncNum, hoGlobalStart, hoNum);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitW<T, ID_T, NC_FACTOR>::Process()
{
    if (GetBlockIdx() >= this->tilingData_->useCoreNum) {
        return;
    }

    BlockParam blockPara;
    for (int64_t curIdx = this->startBlockIdx_; curIdx < this->endBlockIdx_; curIdx++) {
        this->CalBlockPara(curIdx, blockPara);
        ProcessOneBlock(blockPara);
    }
}

} // namespace AdaptivePool2dSplitWNamespace
#endif // ADAPTIVE_AVG_POOL2D_SPLIT_W_H_
