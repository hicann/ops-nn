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

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class OptGroupLoadGM2UBTools {
public:
    __aicore__ inline OptGroupLoadGM2UBTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void LoadGM2UB()
    {
        if (unlikely(self_->ctx.groupOptIter == self_->ctx.vecId)) {
            if constexpr (Intf::isQuantScene) {
                LocalTensor<int8_t> zeroTensor = self_->ctx.ndUbBuf.template Get<int8_t>();
                Duplicate<int8_t>(zeroTensor, 0, self_->ctx.ubBufSize);
            } else {
                LocalTensor<typename Intf::WeightT> zeroTensor =
                    self_->ctx.ndUbBuf.template Get<typename Intf::WeightT>();
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
        if constexpr (Intf::formatOutput == ConvFormat::NDHWC || Intf::formatOutput == ConvFormat::NHWC) {
            uint64_t curSrcCoOpt = self_->ctx.convTiling->orgCo;
            copyParamsHWC.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = curSrcCoOpt;
            copyParamsHWC.loopInfo.loopSrcStride[NDDMA_LOOP3_INDEX] = curSrcCoOpt * self_->ctx.ciPerGroup;
            copyParamsHWC.loopInfo.loopSize[NDDMA_LOOP2_INDEX] = self_->ctx.singleGroups;
            DataCopy<typename Intf::WeightT, NDDMA_HWC_DIMS, kDefaultMultiCopyConfig>(ndTensor, self_->ctx.bgm,
                copyParamsHWC);
        } else {
            copyParams.loopInfo.loopSize[NDDMA_LOOP2_INDEX] = self_->ctx.singleGroups;
            DataCopy<typename Intf::WeightT, NDDMA_DIMS, kDefaultMultiCopyConfig>(ndTensor, self_->ctx.bgm, copyParams);
        }
    }

private:
__aicore__ inline void NDDMAFirstSetCopyParamsCHW()
    {
        uint64_t srcKSize = self_->ctx.ciPerGroup * self_->ctx.convTiling->kernelHxkernelWxkernelD;
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
        copyParams.loopInfo.loopDstStride[NDDMA_LOOP2_INDEX] = self_->ctx.coPerGroup * self_->ctx.kUbSize +
                                                               srcKSize;
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
        copyParamsHWC.loopInfo.loopSize[NDDMA_LOOP3_INDEX] = self_->ctx.convTiling->kernelHxkernelWxkernelD;
        copyParamsHWC.loopInfo.loopDstStride[NDDMA_LOOP3_INDEX] = self_->ctx.coOptAlign * self_->ctx.ciOptAlign;
    }

private:
    Intf *self_ = nullptr;
    MultiCopyParams<typename Intf::WeightT, NDDMA_DIMS> copyParams;
    MultiCopyParams<typename Intf::WeightT, NDDMA_HWC_DIMS> copyParamsHWC;
    LocalTensor<typename Intf::WeightT> ndTensor;
};

template <class Intf>
class OptGroupTransND2NZTools {
public:
    __aicore__ inline OptGroupTransND2NZTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
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
                curValue += self_->ctx.convTiling->kernelHxkernelWxkernelD;
            } else {
                curValue += self_->ctx.coOptAlign;
            }
        }
        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);

        __local_mem__ IndexT *indexAddr = (__local_mem__ IndexT *)indexTensor.GetPhyAddr();
        uint16_t repeatTimes = static_cast<uint16_t>(REG_SIZE / sizeof(IndexT) / Intf::k0 - 1);
        uint8_t dstOffset = Intf::k0;
        uint8_t elesPerRepeat = Intf::k0;
        uint32_t maskL = Intf::k0;
        IndexT nStride;
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW || Intf::formatOutput == ConvFormat::NCHW) {
            nStride = static_cast<IndexT>(self_->ctx.kUbSize);
        } else {
            nStride = static_cast<IndexT>(1);
        }

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<IndexT> indexReg;
            MicroAPI::DataCopy<IndexT>(indexReg, indexAddr);
            MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<IndexT>(maskL);

            for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
                MicroAPI::Adds<IndexT, IndexT>(indexReg, indexReg, nStride, maskReg);
                MicroAPI::DataCopy<IndexT>(indexAddr + dstOffset, indexReg, maskReg);
                dstOffset += elesPerRepeat;
            }
        }
    }

    __aicore__ inline void TransNCHW2NZ()
    {
        uint16_t ciLoopTimes = self_->ctx.ci1Opt;
        uint16_t coLoopTimes = coOptLoopTimes;
        uint16_t khkwLoopTimes = self_->ctx.convTiling->kernelHxkernelW;
        uint32_t srcCiStride = self_->ctx.convTiling->kernelHxkernelW * Intf::k0;
        uint32_t srcCoStride = coPerReg * self_->ctx.kUbSize;
        uint32_t dstCiStride = self_->ctx.convTiling->kernelHxkernelW * Intf::k0 * self_->ctx.coOptAlign;
        uint32_t dstKhKwStride = Intf::k0 * self_->ctx.coOptAlign;
        uint32_t dstCoStride = coPerReg * Intf::k0;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<RegT> gatherReg;
            MicroAPI::RegTensor<IndexT> indexReg;
            MicroAPI::MaskReg gatherMaskReg = MicroAPI::CreateMask<RegT, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg vstsMaskReg;
            if constexpr (Intf::isQuantScene) {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::H>();
            } else {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::ALL>();
            }

            __local_mem__ SrcT *srcAddr = (__local_mem__ SrcT *)ndTensor.GetPhyAddr();
            __local_mem__ DstT *dstAddr = (__local_mem__ DstT *)self_->ctx.nzTensor.GetPhyAddr();
            __local_mem__ IndexT *indexAddr = (__local_mem__ IndexT *)indexTensor.GetPhyAddr();

            MicroAPI::DataCopy<IndexT>(indexReg, indexAddr);

            for (uint16_t ci1OptIndex = 0; ci1OptIndex < ciLoopTimes; ++ci1OptIndex) {
                for (uint16_t khkwIndex = 0; khkwIndex < khkwLoopTimes; ++khkwIndex) {
                    for (uint16_t coOptIndex = 0; coOptIndex < coLoopTimes; ++coOptIndex) {
                        uint32_t srcOffset = ci1OptIndex * srcCiStride + khkwIndex + coOptIndex * srcCoStride;
                        uint32_t dstOffset = ci1OptIndex * dstCiStride + khkwIndex * dstKhKwStride +
                                             coOptIndex * dstCoStride;

                        MicroAPI::DataCopyGather<RegT, SrcT, IndexT>(
                            gatherReg, srcAddr + srcOffset, indexReg, gatherMaskReg);

                        if constexpr (Intf::isQuantScene) {
                            // Remove the higher zeros of the int16_t data gathered by the Micro Gather instr
                            MicroAPI::Pack<uint8_t, RegT, MicroAPI::HighLowPart::LOWEST>(
                                (MicroAPI::RegTensor<uint8_t> &)gatherReg, gatherReg);
                            MicroAPI::DataCopy<DstT>(
                                dstAddr + dstOffset, (MicroAPI::RegTensor<DstT> &)gatherReg, vstsMaskReg);
                        } else {
                            MicroAPI::DataCopy<DstT>(
                                dstAddr + dstOffset, (MicroAPI::RegTensor<DstT> &)gatherReg, vstsMaskReg);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void TransNCDHW2NZ()
    {
        uint16_t kdLoopTimes = self_->ctx.convTiling->kernelD;
        uint16_t ciLoopTimes = self_->ctx.ci1Opt;
        uint16_t coLoopTimes = coOptLoopTimes;
        uint16_t khkwLoopTimes = self_->ctx.convTiling->kernelHxkernelW;
        uint32_t srcCiStride = self_->ctx.convTiling->kernelHxkernelWxkernelD * Intf::k0;
        uint32_t srcCoStride = coPerReg * self_->ctx.kUbSize;
        uint32_t dstCiStride = self_->ctx.convTiling->kernelHxkernelW * Intf::k0 * self_->ctx.coOptAlign;
        uint32_t dstKhKwStride = Intf::k0 * self_->ctx.coOptAlign;
        uint32_t dstCoStride = coPerReg * Intf::k0;
        uint32_t srcKdStride = self_->ctx.convTiling->kernelHxkernelW;
        uint32_t dstKdStride = self_->ctx.ci1Opt * dstCiStride;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<RegT> gatherReg;
            MicroAPI::RegTensor<IndexT> indexReg;
            MicroAPI::MaskReg gatherMaskReg = MicroAPI::CreateMask<RegT, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg vstsMaskReg;
            if constexpr (Intf::isQuantScene) {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::H>();
            } else {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::ALL>();
            }

            __local_mem__ SrcT *srcAddr = (__local_mem__ SrcT *)ndTensor.GetPhyAddr();
            __local_mem__ DstT *dstAddr = (__local_mem__ DstT *)self_->ctx.nzTensor.GetPhyAddr();
            __local_mem__ IndexT *indexAddr = (__local_mem__ IndexT *)indexTensor.GetPhyAddr();

            MicroAPI::DataCopy<IndexT>(indexReg, indexAddr);

            for (uint16_t kdIndex = 0; kdIndex < kdLoopTimes; ++kdIndex) {
                for (uint16_t ci1OptIndex = 0; ci1OptIndex < ciLoopTimes; ++ci1OptIndex) {
                    for (uint16_t khkwIndex = 0; khkwIndex < khkwLoopTimes; ++khkwIndex) {
                        for (uint16_t coOptIndex = 0; coOptIndex < coLoopTimes; ++coOptIndex) {
                            uint32_t srcOffset = kdIndex * srcKdStride + ci1OptIndex * srcCiStride + khkwIndex +
                                                 coOptIndex * srcCoStride;
                            uint32_t dstOffset = kdIndex * dstKdStride + ci1OptIndex * dstCiStride +
                                                 khkwIndex * dstKhKwStride + coOptIndex * dstCoStride;

                            MicroAPI::DataCopyGather<RegT, SrcT, IndexT>(
                                gatherReg, srcAddr + srcOffset, indexReg, gatherMaskReg);

                            if constexpr (Intf::isQuantScene) {
                                MicroAPI::Pack<uint8_t, RegT, MicroAPI::HighLowPart::LOWEST>(
                                    (MicroAPI::RegTensor<uint8_t> &)gatherReg, gatherReg);
                                MicroAPI::DataCopy<DstT>(
                                    dstAddr + dstOffset, (MicroAPI::RegTensor<DstT> &)gatherReg, vstsMaskReg);
                            } else {
                                MicroAPI::DataCopy<DstT>(dstAddr + dstOffset, gatherReg, vstsMaskReg);
                            }
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void TransNDHWC2NZ()
    {
        uint16_t kdLoopTimes = self_->ctx.convTiling->kernelD;
        uint16_t ciLoopTimes = self_->ctx.ci1Opt;
        uint16_t coLoopTimes = coOptLoopTimes;
        uint16_t khkwLoopTimes = self_->ctx.convTiling->kernelHxkernelW;
        uint32_t srcGroupOptSize = self_->ctx.coOptAlign * self_->ctx.ciOptAlign;
        uint32_t srcCiStride = self_->ctx.coOptAlign * Intf::k0;
        uint32_t srcCoStride = coPerReg;
        uint32_t dstCiStride = self_->ctx.convTiling->kernelHxkernelW * Intf::k0 * self_->ctx.coOptAlign;
        uint32_t dstKhKwStride = Intf::k0 * self_->ctx.coOptAlign;
        uint32_t srcKhKwStride = srcGroupOptSize;
        uint32_t dstCoStride = coPerReg * Intf::k0;
        uint32_t srcKdStride = self_->ctx.convTiling->kernelHxkernelW * srcGroupOptSize;
        uint32_t dstKdStride = self_->ctx.convTiling->kernelHxkernelW * self_->ctx.coOptAlign *
                               self_->ctx.ci1Opt * Intf::k0; // ci1Opt has updated in groupOptTail
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<RegT> gatherReg;
            MicroAPI::RegTensor<IndexT> indexReg;
            MicroAPI::MaskReg gatherMaskReg = MicroAPI::CreateMask<RegT, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg vstsMaskReg;
            if constexpr (Intf::isQuantScene) {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::H>();
            } else {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::ALL>();
            }

            __local_mem__ SrcT *srcAddr = (__local_mem__ SrcT *)ndTensor.GetPhyAddr();
            __local_mem__ DstT *dstAddr = (__local_mem__ DstT *)self_->ctx.nzTensor.GetPhyAddr();
            __local_mem__ IndexT *indexAddr = (__local_mem__ IndexT *)indexTensor.GetPhyAddr();

            MicroAPI::DataCopy<IndexT>(indexReg, indexAddr);

            for (uint16_t kdIndex = 0; kdIndex < kdLoopTimes; ++kdIndex) {
                for (uint16_t ci1OptIndex = 0; ci1OptIndex < ciLoopTimes; ++ci1OptIndex) {
                    for (uint16_t khkwIndex = 0; khkwIndex < khkwLoopTimes; ++khkwIndex) {
                        for (uint16_t coOptIndex = 0; coOptIndex < coLoopTimes; ++coOptIndex) {
                            uint32_t srcOffset = kdIndex * srcKdStride + ci1OptIndex * srcCiStride +
                                                 khkwIndex * srcKhKwStride + coOptIndex * srcCoStride;
                            uint32_t dstOffset = kdIndex * dstKdStride + ci1OptIndex * dstCiStride +
                                                 khkwIndex * dstKhKwStride + coOptIndex * dstCoStride;

                            MicroAPI::DataCopyGather<RegT, SrcT, IndexT>(
                                gatherReg, srcAddr + srcOffset, indexReg, gatherMaskReg);
                            if constexpr (Intf::isQuantScene) {
                                // Remove the higher zeros of the int16_t data gathered by the Micro Gather instr
                                MicroAPI::Pack<uint8_t, RegT, MicroAPI::HighLowPart::LOWEST>(
                                    (MicroAPI::RegTensor<uint8_t> &)gatherReg, gatherReg);
                                MicroAPI::DataCopy<DstT>(
                                    dstAddr + dstOffset, (MicroAPI::RegTensor<DstT> &)gatherReg, vstsMaskReg);
                            } else {
                                MicroAPI::DataCopy<DstT>(dstAddr + dstOffset, gatherReg, vstsMaskReg);
                            }
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void TransNHWC2NZ()
    {
        uint16_t ciLoopTimes = self_->ctx.ci1Opt;
        uint16_t khkwLoopTimes = self_->ctx.convTiling->kernelHxkernelW;
        uint16_t coLoopTimes = coOptLoopTimes;
        uint32_t srcCiStride = self_->ctx.coOptAlign * Intf::k0;
        uint32_t srcKhKwStride = self_->ctx.coOptAlign * self_->ctx.ciOptAlign; 
        uint32_t srcCoStride = coPerReg;
        uint32_t dstCiStride = self_->ctx.convTiling->kernelHxkernelW * Intf::k0 * self_->ctx.coOptAlign;
        uint32_t dstKhKwStride = Intf::k0 * self_->ctx.coOptAlign;
        uint32_t dstCoStride = coPerReg * Intf::k0;
 
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<RegT> gatherReg;
            MicroAPI::RegTensor<IndexT> indexReg;
            MicroAPI::MaskReg gatherMaskReg = MicroAPI::CreateMask<RegT, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg vstsMaskReg;
            if constexpr (Intf::isQuantScene) {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::H>();
            } else {
                vstsMaskReg = MicroAPI::CreateMask<DstT, MicroAPI::MaskPattern::ALL>();
            }
 
            __local_mem__ SrcT *srcAddr = (__local_mem__ SrcT *)ndTensor.GetPhyAddr();
            __local_mem__ DstT *dstAddr = (__local_mem__ DstT *)self_->ctx.nzTensor.GetPhyAddr();
            __local_mem__ IndexT *indexAddr = (__local_mem__ IndexT *)indexTensor.GetPhyAddr();
 
            MicroAPI::DataCopy<IndexT>(indexReg, indexAddr);
 
            for (uint16_t ci1OptIndex = 0; ci1OptIndex < ciLoopTimes; ++ci1OptIndex) {
                for (uint16_t khkwIndex = 0; khkwIndex < khkwLoopTimes; ++khkwIndex) {
                    for (uint16_t coOptIndex = 0; coOptIndex < coLoopTimes; ++coOptIndex) {
                        uint32_t srcOffset = ci1OptIndex * srcCiStride + khkwIndex * srcKhKwStride +
                                             coOptIndex * srcCoStride;
                        uint32_t dstOffset = ci1OptIndex * dstCiStride + khkwIndex * dstKhKwStride +
                                             coOptIndex * dstCoStride;
                        MicroAPI::DataCopyGather<RegT, SrcT, IndexT>(
                            gatherReg, srcAddr + srcOffset, indexReg, gatherMaskReg);
                        if constexpr (Intf::isQuantScene) {
                            // Remove the higher zeros of the int16_t data gathered by the Micro Gather instr
                            MicroAPI::Pack<uint8_t, RegT, MicroAPI::HighLowPart::LOWEST>(
                                (MicroAPI::RegTensor<uint8_t> &)gatherReg, gatherReg);
                            MicroAPI::DataCopy<DstT>(
                                dstAddr + dstOffset, (MicroAPI::RegTensor<DstT> &)gatherReg, vstsMaskReg);
                        } else {
                            MicroAPI::DataCopy<DstT>(dstAddr + dstOffset, gatherReg, vstsMaskReg);
                        }
                    }
                }
            }
        }
    }

private:
    Intf *self_ = nullptr;

    using SrcT = typename Conditional<Intf::isQuantScene, int8_t, typename Intf::WeightT>::type;
    using DstT = typename Conditional<Intf::isQuantScene, int8_t, typename Intf::WeightT>::type;
    using RegT = typename Conditional<Intf::isQuantScene, int16_t, typename Intf::WeightT>::type;
    using IndexT = typename Conditional<
        AscendC::IsSameType<typename Intf::WeightT, float>::value, uint32_t, uint16_t>::type;

    LocalTensor<SrcT> ndTensor;
    LocalTensor<IndexT> indexTensor;

    uint16_t coOptLoopTimes = 0;
    uint16_t coPerReg = 0;
};

template <class Intf>
class OptGroupLoadUB2L1Tools {
public:
    __aicore__ inline OptGroupLoadUB2L1Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

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
            copyParams.blockCount = self_->ctx.convTiling->kBL1 / Intf::k0;
            copyParams.blockLen = self_->ctx.convTiling->nBL1;
            copyParams.srcStride = self_->ctx.co1Opt * BLOCK_L0_N - self_->ctx.convTiling->nBL1;
        }

        srcOffset = (self_->ctx.coStartPos + self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1) * Intf::k0 +
                    self_->ctx.kBL1Iter * self_->ctx.convTiling->kBL1 * self_->ctx.coOptAlign;
    }

    __aicore__ inline void SetCopyParams3D()
    {
        if (unlikely(self_->ctx.loadUB2L1Iter == 0)) {
            copyParams.blockLen = self_->ctx.convTiling->nBL1;
            copyParams.srcStride = self_->ctx.co1Opt * BLOCK_L0_N - self_->ctx.convTiling->nBL1;
            kOffset = self_->ctx.bL1Dk * self_->ctx.bL1Cin * self_->ctx.convTiling->kernelHxkernelW;
        }

        uint64_t currentBL1Dk = IsKBL1Tail() ? self_->ctx.bL1DkTail : self_->ctx.bL1Dk;
        uint64_t currentBL1Cin1 = IsKBL1Tail() ? self_->ctx.bL1CinTail : self_->ctx.bL1Cin;
        copyParams.blockCount = currentBL1Dk * CeilDiv(currentBL1Cin1, Intf::k0) *
                                self_->ctx.convTiling->kernelHxkernelW;

        srcOffset = (self_->ctx.coStartPos + self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1) * Intf::k0 +
                    self_->ctx.kBL1Iter * kOffset * self_->ctx.coOptAlign;
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
    Intf *self_ = nullptr;

    DataCopyParams copyParams;
    uint64_t kOffset = 0;
    uint64_t srcOffset = 0;
};

};

#endif // CONV_INSTR_OPT_GROUP_IMPL_H