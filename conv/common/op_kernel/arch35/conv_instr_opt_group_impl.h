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
 * \file conv_instr_opt_group_impl.h
 * \brief
 */

#ifndef CONV_INSTR_OPT_GROUP_IMPL_H
#define CONV_INSTR_OPT_GROUP_IMPL_H

#include "conv_config.h"
#include "conv_util.h"
#include "conv_instr_nd2nz_vf.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class OptGroupLoadGM2UBTools {
public:
    __aicore__ inline OptGroupLoadGM2UBTools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void LoadGM2UB()
    {
        if (unlikely(self_->ctx.groupOptIter == self_->ctx.vecId)) {
            if constexpr (Intf::isQuantScene) {
                LocalTensor<int8_t> zeroTensor = self_->ctx.ndUbBuf.template Get<int8_t>();
                Duplicate<int8_t>(zeroTensor, 0, self_->ctx.ubBufSize);
            } else {
                LocalTensor<typename Intf::WeightT> zeroTensor = self_->ctx.ndUbBuf
                                                                     .template Get<typename Intf::WeightT>();
                Duplicate<typename Intf::WeightT>(zeroTensor, 0, self_->ctx.ubBufSize);
            }

            // For nddma wait vdup
            event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);

            if constexpr (Intf::formatOutput == ConvFormat::NDHWC || Intf::formatOutput == ConvFormat::NHWC) {
                NDDMAFirstSetCopyParamsHWC();
            } else {
                NDDMAFirstSetCopyParamsCHW();
            }
            ndTensor = self_->ctx.ndUbBuf.template Get<typename Intf::WeightT>();
        }
        uint64_t gmOffset = 0;
        if constexpr (Intf::groupOptPreloadFlag) {
            uint64_t weightOneGroupSize = 0;
            if constexpr (Intf::formatWeight == ConvFormat::NCHW) {
                weightOneGroupSize = self_->ctx.coPerGroup * self_->ctx.ciPerGroup * self_->ctx.enlarge *
                                     self_->ctx.convTilingData->kernelHxkernelWxkernelD;
            } else if constexpr (Intf::formatWeight == ConvFormat::HWCN) {
                weightOneGroupSize = self_->ctx.coPerGroup * self_->ctx.enlarge;
            }
            gmOffset = weightOneGroupSize * self_->ctx.groupOptIter;
        }
        if constexpr (Intf::formatOutput == ConvFormat::NDHWC || Intf::formatOutput == ConvFormat::NHWC) {
            uint64_t curSrcCoOpt = self_->ctx.convTilingData->orgCo;
            copyParamsHWC.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = curSrcCoOpt;
            copyParamsHWC.loopInfo.loopSrcStride[NDDMA_LOOP3_INDEX] = curSrcCoOpt * self_->ctx.ciPerGroup;
            copyParamsHWC.loopInfo.loopSize[NDDMA_LOOP2_INDEX] = self_->ctx.singleGroups;
            DataCopy<typename Intf::WeightT, NDDMA_HWC_DIMS, kDefaultMultiCopyConfig>(
                ndTensor, self_->ctx.bgm[gmOffset], copyParamsHWC);
        } else {
            copyParams.loopInfo.loopSize[NDDMA_LOOP2_INDEX] = self_->ctx.singleGroups;
            DataCopy<typename Intf::WeightT, NDDMA_DIMS, kDefaultMultiCopyConfig>(ndTensor, self_->ctx.bgm[gmOffset],
                                                                                  copyParams);
        }
    }

private:
    __aicore__ inline void NDDMAFirstSetCopyParamsCHW()
    {
        uint64_t srcKSize = self_->ctx.ciPerGroup * self_->ctx.convTilingData->kernelHxkernelWxkernelD;
        // NDDMA Loop0 params
        copyParams.loopInfo.loopSize[NDDMA_LOOP0_INDEX] = srcKSize;
        copyParams.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = 1;
        copyParams.loopInfo.loopDstStride[NDDMA_LOOP0_INDEX] = 1;
        // NDDMA Loop1 params
        copyParams.loopInfo.loopSize[NDDMA_LOOP1_INDEX] = self_->ctx.coPerGroup;
        copyParams.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = srcKSize;
        copyParams.loopInfo.loopDstStride[NDDMA_LOOP1_INDEX] = self_->ctx.kUbSize;
        // NDDMA Loop2 params
        copyParams.loopInfo.loopSrcStride[NDDMA_LOOP2_INDEX] = self_->ctx.coPerGroup * srcKSize;
        copyParams.loopInfo.loopDstStride[NDDMA_LOOP2_INDEX] = self_->ctx.coPerGroup * self_->ctx.kUbSize + srcKSize;
    }

    __aicore__ inline void NDDMAFirstSetCopyParamsHWC()
    {
        // NDDMA Loop0 params
        copyParamsHWC.loopInfo.loopSize[NDDMA_LOOP0_INDEX] = self_->ctx.coPerGroup;
        copyParamsHWC.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = 1;
        copyParamsHWC.loopInfo.loopDstStride[NDDMA_LOOP0_INDEX] = 1;
        // NDDMA Loop1 params
        copyParamsHWC.loopInfo.loopSize[NDDMA_LOOP1_INDEX] = self_->ctx.ciPerGroup;
        copyParamsHWC.loopInfo.loopDstStride[NDDMA_LOOP1_INDEX] = self_->ctx.coOptAlign;
        // NDDMA Loop2 params
        copyParamsHWC.loopInfo.loopSrcStride[NDDMA_LOOP2_INDEX] = self_->ctx.coPerGroup;
        copyParamsHWC.loopInfo.loopDstStride[NDDMA_LOOP2_INDEX] = self_->ctx.coOptAlign * self_->ctx.ciPerGroup +
                                                                  self_->ctx.coPerGroup;
        // NDDMA Loop3 params
        copyParamsHWC.loopInfo.loopSize[NDDMA_LOOP3_INDEX] = self_->ctx.convTilingData->kernelHxkernelWxkernelD;
        copyParamsHWC.loopInfo.loopDstStride[NDDMA_LOOP3_INDEX] = self_->ctx.coOptAlign * self_->ctx.ciOptAlign;
    }

private:
    Intf* self_ = nullptr;
    NdDmaParams<typename Intf::WeightT, NDDMA_DIMS> copyParams;
    NdDmaParams<typename Intf::WeightT, NDDMA_HWC_DIMS> copyParamsHWC;
    LocalTensor<typename Intf::WeightT> ndTensor;
};

template <class Intf>
class OptGroupTransND2NZTools {
public:
    __aicore__ inline OptGroupTransND2NZTools() {}

    __aicore__ inline void SetParams(Intf* self)
    {
        self_ = self;

        indexTensor = self_->ctx.indexUbBuf.template Get<IndexT>();
        ndTensor = self_->ctx.ndUbBuf.template Get<SrcT>();

        if constexpr (Intf::isQuantScene) {
            coOptLoopTimes = self_->ctx.co1Opt * B8_CO0_LOOP_TIMES;
            coPerReg = BLOCK_L0_N / B8_CO0_LOOP_TIMES;
        } else {
            coOptLoopTimes = self_->ctx.co1Opt * CO0_LOOP_TIMES;
            coPerReg = BLOCK_L0_N / CO0_LOOP_TIMES;
        }
    }

    __aicore__ inline void TransND2NZ()
    {
        if (unlikely(self_->ctx.groupOptIter == self_->ctx.vecId)) {
            SetIndex();
        }

        // Due to the VF compile, reusing code it's bound to make scalar increase.
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            TransNCDHW2NZ();
        } else if constexpr (Intf::formatOutput == ConvFormat::NCHW) {
            TransNCHW2NZ();
        } else if constexpr (Intf::formatOutput == ConvFormat::NDHWC) {
            TransNDHWC2NZ();
        } else if constexpr (Intf::formatOutput == ConvFormat::NHWC) {
            TransNHWC2NZ();
        }
    }

private:
    __aicore__ inline void SetIndex()
    {
        IndexT curValue = 0;
        for (uint8_t idx = 0; idx < Intf::k0; ++idx) {
            indexTensor.SetValue(idx, curValue);
            if constexpr (Intf::formatOutput == ConvFormat::NCDHW || Intf::formatOutput == ConvFormat::NCHW) {
                curValue += self_->ctx.convTilingData->kernelHxkernelWxkernelD;
            } else {
                curValue += self_->ctx.coOptAlign;
            }
        }
        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);

        __ubuf__ IndexT* indexAddr = (__ubuf__ IndexT*)indexTensor.GetPhyAddr();
        uint16_t repeatTimes = static_cast<uint16_t>(REG_SIZE / sizeof(IndexT) / Intf::k0 - 1);
        IndexT nStride;
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW || Intf::formatOutput == ConvFormat::NCHW) {
            nStride = static_cast<IndexT>(self_->ctx.kUbSize);
        } else {
            nStride = static_cast<IndexT>(1);
        }
        SetIndexVf(indexAddr, repeatTimes, nStride, static_cast<uint8_t>(Intf::k0));
    }

    __aicore__ inline void TransNCHW2NZ()
    {
        TransND2NZVfParams<SrcT, DstT, IndexT> params;
        FillCommonNd2NzParams(params);
        params.srcCiStride = self_->ctx.convTilingData->kernelHxkernelW * Intf::k0;
        params.srcKhKwStride = 1;
        params.srcCoStride = coPerReg * self_->ctx.kUbSize;
        TransND2NZVf<SrcT, DstT, RegT, IndexT, Intf::isQuantScene>(params);
    }

    __aicore__ inline void TransNCDHW2NZ()
    {
        TransND2NZVfParams<SrcT, DstT, IndexT> params;
        FillCommonNd2NzParams(params);
        params.srcCiStride = self_->ctx.convTilingData->kernelHxkernelWxkernelD * Intf::k0;
        params.srcKhKwStride = 1;
        params.srcCoStride = coPerReg * self_->ctx.kUbSize;

        TransND2NZKdVfParams kdParams;
        kdParams.kdLoopTimes = self_->ctx.convTilingData->kernelD;
        kdParams.srcKdStride = self_->ctx.convTilingData->kernelHxkernelW;
        kdParams.dstKdStride = self_->ctx.ci1Opt * params.dstCiStride;
        TransND2NZKdVf<SrcT, DstT, RegT, IndexT, Intf::isQuantScene>(params, kdParams);
    }

    __aicore__ inline void TransNDHWC2NZ()
    {
        TransND2NZVfParams<SrcT, DstT, IndexT> params;
        FillCommonNd2NzParams(params);
        uint32_t srcGroupOptSize = self_->ctx.coOptAlign * self_->ctx.ciOptAlign;
        params.srcCiStride = self_->ctx.coOptAlign * Intf::k0;
        params.srcKhKwStride = srcGroupOptSize;
        params.srcCoStride = coPerReg;

        TransND2NZKdVfParams kdParams;
        kdParams.kdLoopTimes = self_->ctx.convTilingData->kernelD;
        kdParams.srcKdStride = self_->ctx.convTilingData->kernelHxkernelW * srcGroupOptSize;
        kdParams.dstKdStride = self_->ctx.convTilingData->kernelHxkernelW * self_->ctx.coOptAlign * self_->ctx.ci1Opt *
                               Intf::k0; // ci1Opt has updated in groupOptTail
        TransND2NZKdVf<SrcT, DstT, RegT, IndexT, Intf::isQuantScene>(params, kdParams);
    }

    __aicore__ inline void TransNHWC2NZ()
    {
        TransND2NZVfParams<SrcT, DstT, IndexT> params;
        FillCommonNd2NzParams(params);
        params.srcCiStride = self_->ctx.coOptAlign * Intf::k0;
        params.srcKhKwStride = self_->ctx.coOptAlign * self_->ctx.ciOptAlign;
        params.srcCoStride = coPerReg;
        TransND2NZVf<SrcT, DstT, RegT, IndexT, Intf::isQuantScene>(params);
    }

private:
    Intf* self_ = nullptr;

    using SrcT = typename Conditional<Intf::isQuantScene, int8_t, typename Intf::WeightT>::type;
    using DstT = typename Conditional<Intf::isQuantScene, int8_t, typename Intf::WeightT>::type;
    using RegT = typename Conditional<Intf::isQuantScene, int16_t, typename Intf::WeightT>::type;
    using IndexT = typename Conditional<AscendC::IsSameType<typename Intf::WeightT, float>::value, uint32_t,
                                        uint16_t>::type;

    LocalTensor<SrcT> ndTensor;
    LocalTensor<IndexT> indexTensor;

    uint16_t coOptLoopTimes = 0;
    uint16_t coPerReg = 0;

    __aicore__ inline void FillCommonNd2NzParams(TransND2NZVfParams<SrcT, DstT, IndexT>& params)
    {
        params.ciLoopTimes = self_->ctx.ci1Opt;
        params.coLoopTimes = coOptLoopTimes;
        params.khkwLoopTimes = self_->ctx.convTilingData->kernelHxkernelW;
        params.dstCiStride = self_->ctx.convTilingData->kernelHxkernelW * Intf::k0 * self_->ctx.coOptAlign;
        params.dstKhKwStride = Intf::k0 * self_->ctx.coOptAlign;
        params.dstCoStride = coPerReg * Intf::k0;
        params.srcAddr = (__ubuf__ SrcT*)ndTensor.GetPhyAddr();
        params.dstAddr = (__ubuf__ DstT*)self_->ctx.nzTensor.GetPhyAddr();
        params.indexAddr = (__ubuf__ IndexT*)indexTensor.GetPhyAddr();
    }
};

template <class Intf>
class OptGroupLoadUB2L1Tools {
public:
    __aicore__ inline OptGroupLoadUB2L1Tools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void LoadUB2L1()
    {
        if constexpr (Intf::isConv3D) {
            SetCopyParams3D();
        } else {
            SetCopyParams2D();
        }

        if constexpr (!Intf::bL1DBFlag) {
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.nzTensor[srcOffset], copyParams);
        } else {
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1[self_->ctx.pingPongFlag * self_->ctx.bL1SpaceSize],
                                             self_->ctx.nzTensor[srcOffset], copyParams);
            self_->ctx.pingPongFlag ^= 1;
        }
    }

private:
    __aicore__ inline void SetCopyParams2D()
    {
        if (unlikely(self_->ctx.groupOptIter == self_->ctx.vecId)) {
            copyParams.blockCount = self_->ctx.convTilingData->kBL1 / Intf::k0;
            copyParams.blockLen = self_->ctx.convTilingData->nBL1;
            copyParams.srcStride = self_->ctx.co1Opt * BLOCK_L0_N - self_->ctx.convTilingData->nBL1;
        }

        if constexpr (Intf::isKL1NL0FullLoad) {
            srcOffset = self_->ctx.coStartPos * Intf::k0 +
                        self_->ctx.kBL1Iter * self_->ctx.convTilingData->kBL1 * self_->ctx.coOptAlign;
        } else {
            srcOffset = (self_->ctx.coStartPos + self_->ctx.nBL1Iter * self_->ctx.convTilingData->nBL1) * Intf::k0 +
                        self_->ctx.kBL1Iter * self_->ctx.convTilingData->kBL1 * self_->ctx.coOptAlign;
        }
    }

    __aicore__ inline void SetCopyParams3D()
    {
        if (unlikely(self_->ctx.loadUB2L1Iter == 0)) {
            copyParams.blockLen = self_->ctx.convTilingData->nBL1;
            copyParams.srcStride = self_->ctx.co1Opt * BLOCK_L0_N - self_->ctx.convTilingData->nBL1;
            kOffset = self_->ctx.bL1Dk * self_->ctx.bL1Cin * self_->ctx.convTilingData->kernelHxkernelW;
        }

        uint64_t currentBL1Dk = IsKBL1Tail() ? self_->ctx.bL1DkTail : self_->ctx.bL1Dk;
        uint64_t currentBL1Cin1 = IsKBL1Tail() ? self_->ctx.bL1CinTail : self_->ctx.bL1Cin;
        copyParams.blockCount = currentBL1Dk * CeilDiv(currentBL1Cin1, Intf::k0) *
                                self_->ctx.convTilingData->kernelHxkernelW;

        if constexpr (Intf::isKL1NL0FullLoad) {
            srcOffset = self_->ctx.coStartPos * Intf::k0 + self_->ctx.kBL1Iter * kOffset * self_->ctx.coOptAlign;
        } else {
            srcOffset = (self_->ctx.coStartPos + self_->ctx.nBL1Iter * self_->ctx.convTilingData->nBL1) * Intf::k0 +
                        self_->ctx.kBL1Iter * kOffset * self_->ctx.coOptAlign;
        }
    }

    __aicore__ inline bool IsKBL1Tail()
    {
        if (self_->ctx.bL1CinLoadNum == 1) {
            return self_->ctx.kBL1Iter == self_->ctx.maxKBL1Iter;
        } else {
            return (self_->ctx.kBL1Iter + 1) % self_->ctx.bL1CinLoadNum == 0;
        }
    }

private:
    Intf* self_ = nullptr;

    DataCopyParams copyParams;
    uint64_t kOffset = 0;
    uint64_t srcOffset = 0;
};

}; // namespace ConvFunc

#endif // CONV_INSTR_OPT_GROUP_IMPL_H
