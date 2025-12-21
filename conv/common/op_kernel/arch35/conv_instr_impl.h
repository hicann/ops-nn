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
 * \file conv_instr_impl.h
 * \brief
 */

#ifndef CONV_INSTR_IMPL_H
#define CONV_INSTR_IMPL_H

#include "conv_config.h"
#include "conv_util.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf, typename ChannelWiseT>
class LoadChannelWiseL1Tools {
public:
    __aicore__ inline LoadChannelWiseL1Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        currentNL0_ = n;
    }

    __aicore__ inline void LoadChannelWiseL1FullLoad(const LocalTensor<ChannelWiseT> &tensorL1,
        const GlobalTensor<ChannelWiseT> &tensorGm, uint64_t loadNum, uint64_t gmStartAddr)
    {
        uint64_t byteNum = sizeof(ChannelWiseT);
        DataCopyParams dataCopyParams(1, loadNum * byteNum, 0, 0);
        uint8_t rightPadding = (uint8_t)(AlignB(loadNum * byteNum, PADDING_ALIGN_SIZE) / byteNum - loadNum);
        DataCopyPadParams padParams(true, 0, rightPadding, 0);
        DataCopyPad<ChannelWiseT>(tensorL1, tensorGm[gmStartAddr], dataCopyParams, padParams);
    }

    __aicore__ inline void LoadChannelWiseL1(const LocalTensor<ChannelWiseT> &tensorL1,
                                             const GlobalTensor<ChannelWiseT> &tensorGm)
    {
        uint64_t tensorGmOffset = self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
            self_->ctx.nL0Iter * self_->ctx.convTiling->nL0;

        LoadChannelWiseL1FullLoad(tensorL1, tensorGm, currentNL0_, tensorGmOffset);
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentNL0_ = 0;
};

template <class Intf, typename BiasL1T, typename BiasBtT>
class LoadBiasBtTools {
public:
    __aicore__ inline LoadBiasBtTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        currentNL0_ = n;
    }

    __aicore__ inline void LoadBiasBt(const LocalTensor<BiasBtT> &biasBt,
                                      const LocalTensor<BiasL1T> &biasL1)
    {
        if ASCEND_IS_AIV {
            return;
        }
        uint32_t offset = 0;
        if (self_->ctx.convTiling->biasFullLoadFlag) {
            offset = self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
                     self_->ctx.nL0Iter * self_->ctx.convTiling->nL0;
        }

        if constexpr (AscendC::IsSameType<typename Intf::BiasT, half>::value && 
                      AscendC::IsSameType<typename Intf::L0cT, int32_t>::value) {
            // fixed-point multiplication should set cvt_mode = 2 and fix_val, which is encapsulated by basic api.
            DataCopyParams biasBtCopyParams(1, currentNL0_ * Intf::sizeOfBias / BT_BLOCK_SIZE, 0, 0);
            DataCopy(biasBt, biasL1[offset], biasBtCopyParams);
        } else if constexpr (AscendC::IsSameType<typename Intf::BiasT, half>::value) {
            copy_cbuf_to_bt((uint64_t)0, (__cbuf__ typename Intf::BiasT*)self_->ctx.biasL1[offset].GetPhyAddr(),
                (bool)1, 1, currentNL0_ * Intf::sizeOfBias / BT_BLOCK_SIZE, 0, 0);
        } else {
            #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
            copy_cbuf_to_bt((uint64_t)0, (__cbuf__ typename Intf::BiasT*)self_->ctx.biasL1[offset].GetPhyAddr(),
                (bool)0, 1, currentNL0_ * Intf::sizeOfBias / BT_BLOCK_SIZE, 0, 0, 0);
            #else
            copy_cbuf_to_bt((uint64_t)0, (__cbuf__ typename Intf::BiasT*)self_->ctx.biasL1[offset].GetPhyAddr(),
                (bool)0, 1, currentNL0_ * Intf::sizeOfBias / BT_BLOCK_SIZE, 0, 0);
            #endif
        }
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentNL0_ = 0;
};

template <class Intf>
class LoadBL0Tools {
public:
    __aicore__ inline LoadBL0Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        nStep_ = self_->ctx.convTiling->nL0 / BLOCK_L0_N;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        ratioOfNToN0 = (self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter && self_->ctx.nL0Iter == self_->ctx.maxNL0Iter) ?
            n / BLOCK_L0_N : self_->ctx.convTiling->nStep;
    }

    __aicore__ inline void LoadBL0(bool isFirst)
    {
        if (unlikely(isFirst)) {
            uint64_t mStartPosition = self_->ctx.nL0Iter * nStep_;
            xt_ = (((self_->ctx.convTiling->nL1DivBlockSize & MASK_16) << 0) | ((ratioOfNToN0 & MASK_16) << DST_STRIDE_OFFSET));
            xmtmp_ = (((mStartPosition & MASK_16) << 0) | ((ratioOfNToN0 & MASK_8) << M_STEP_OFFSET));
        }
        uint64_t kStep = (self_->ctx.kIter != self_->ctx.maxKL0Iter) ?
                          self_->ctx.convTiling->kStep : (self_->ctx.kL0Tail / Intf::k0);
        xm_ = (((self_->ctx.kBL0Iter * self_->ctx.convTiling->kStep) & MASK_16) << K_START_OFFSET) |
        ((kStep & MASK_8) << K_STEP_OFFSET) | xmtmp_;
        load_cbuf_to_cb((__cb__ typename Intf::WeightT *)self_->ctx.bl0.GetPhyAddr(),
                        (__cbuf__ typename Intf::WeightT *)self_->ctx.bl1.GetPhyAddr(), xm_, xt_, false);
    }

private:
    Intf *self_ = nullptr;
    uint64_t ratioOfNToN0 = 0;
    uint64_t xm_ = 0;
    uint64_t xt_ = 0;
    uint64_t xmtmp_ = 0;
    uint64_t nStep_ = 0;
};

template <class Intf>
class MMadTools {
public:
    __aicore__ inline MMadTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void Mad()
    {
        MmadParams mmadParams;
        SetMmadParams(mmadParams);

        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::MULTI_BATCH)) {
            uint32_t srcOffset = 0;
            uint32_t dstOffset = 0;
            uint32_t srcBatchStride = self_->ctx.currentML0Align * self_->ctx.convTiling->kL0;
            uint32_t dstBatchStride = self_->ctx.currentML0Align * self_->ctx.currentNL0Align;
            for (uint32_t batchIdx = 0; batchIdx < self_->ctx.innerBatch; batchIdx++) {
                Mmad<typename Intf::L0cT, typename Intf::FmapT, typename Intf::WeightT>(
                    self_->ctx.cl0[dstOffset], self_->ctx.al0[srcOffset], self_->ctx.bl0, mmadParams);
                    srcOffset += srcBatchStride;
                    dstOffset += dstBatchStride;
            }
        } else {
            Mmad<typename Intf::L0cT, typename Intf::FmapT, typename Intf::WeightT>(
                self_->ctx.cl0, self_->ctx.al0, self_->ctx.bl0, mmadParams);
        }
    }

private:
    __aicore__ inline void SetMmadParams(MmadParams &mmadParams)
    {
        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            mmadParams.m = self_->ctx.innerBatch * currentML0_;
        } else {
            mmadParams.m = currentML0_;
        }
        mmadParams.n = currentNL0_;
        if constexpr (Intf::k0 == Intf::k0FmapTail) {
            mmadParams.k = IsKL0Tail() ? self_->ctx.kL0Tail : self_->ctx.convTiling->kL0;
        } else {
            mmadParams.k = IsKL0Tail() ? self_->ctx.kAL0Tail : self_->ctx.convTiling->kL0;
        }

        if (!self_->ctx.enableBias) {
            mmadParams.cmatrixInitVal = self_->ctx.kIter == 0;
            mmadParams.cmatrixSource = false;
        } else {
            mmadParams.cmatrixInitVal = false;
            mmadParams.cmatrixSource = self_->ctx.kIter == 0;
        }

        if constexpr (!Intf::isInnerBatchFlag) {
            mmadParams.unitFlag = IsKL0Tail() ? UNIT_FLAG_ENABLE_WITH_FLIP : UNIT_FLAG_ENABLE_ONLY;
        }
    }

    __aicore__ inline bool IsKL0Tail()
    {
        return self_->ctx.kIter == self_->ctx.maxKL0Iter;
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;
};

template <class Intf, typename OutputT, uint64_t FixpipeIdx = 0>
__aicore__ inline QuantMode_t GetQuantPre(Intf *self)
{
    if constexpr (AscendC::IsSameType<typename Intf::FmapT, hifloat8_t>::value ||
                  AscendC::IsSameType<typename Intf::FmapT, fp8_e4m3fn_t>::value) {
        if constexpr (AscendC::IsSameType<OutputT, float>::value) {
            return QuantMode_t::VQF322F32_PRE;
        }

        if constexpr (AscendC::IsSameType<OutputT, half>::value) {
            return QuantMode_t::VQF322F16_PRE;
        }

        if constexpr (AscendC::IsSameType<OutputT, bfloat16_t>::value) {
            return QuantMode_t::VQF322BF16_PRE;
        }

        if constexpr (AscendC::IsSameType<OutputT, hifloat8_t>::value) {
            if (self->ctx.convTiling->hasScale == 0) {
                return QuantMode_t::QF322HIF8_PRE;
            } else if (self->ctx.convTiling->roundMode == ROUND_MODE_ROUND) {
                return QuantMode_t::VQF322HIF8_PRE;
            }
        }

        if constexpr (AscendC::IsSameType<OutputT, fp8_e4m3fn_t>::value) {
            return QuantMode_t::VQF322FP8_PRE;
        }
    }

    if constexpr (AscendC::IsSameType<typename Intf::L0cT, int32_t>::value) {
        if constexpr (AscendC::IsSameType<OutputT, half>::value) {
            if constexpr (Intf::isExtendConv2d) {
                if constexpr (FixpipeIdx == 0) {
                    if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        return QuantMode_t::VDEQF16;
                    } else {
                        return QuantMode_t::DEQF16;
                    }
                } else {
                    if (self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        return QuantMode_t::VDEQF16;
                    } else {
                        return QuantMode_t::DEQF16;
                    }
                }
            } else if constexpr (Intf::isFixedPoint) {
                return QuantMode_t::DEQF16;
            } else {
                // current quant_conv2d/quant_conv3d are both vector quant.
                return QuantMode_t::VDEQF16;
            }
        } else if constexpr (AscendC::IsSameType<OutputT, int8_t>::value) {
            if constexpr (Intf::isExtendConv2d) {
                if constexpr (FixpipeIdx == 0) {
                    if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        return QuantMode_t::VREQ8;
                    } else {
                        return QuantMode_t::REQ8;
                    }
                } else {
                    if (self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        return QuantMode_t::VREQ8;
                    } else {
                        return QuantMode_t::REQ8;
                    }
                }
            } else if constexpr (Intf::isFixedPoint) {
                return QuantMode_t::REQ8;
            } else {
                // current quant_conv2d/quant_conv3d are both vector quant.
                return QuantMode_t::VREQ8;
            }
        }
    }

    if constexpr (AscendC::IsSameType<typename Intf::L0cT, float>::value) {
        if constexpr (AscendC::IsSameType<OutputT, float>::value) {
            return QuantMode_t::NoQuant;
        } else if constexpr (AscendC::IsSameType<OutputT, bfloat16_t>::value) {
            return QuantMode_t::F322BF16;
        } else if constexpr (AscendC::IsSameType<OutputT, half>::value) {
            return QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<OutputT, int8_t>::value) {
            if constexpr (Intf::isExtendConv2d) {
                if constexpr (FixpipeIdx == 0) {
                    if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        return QuantMode_t::VQF322B8_PRE;
                    } else {
                        return QuantMode_t::QF322B8_PRE;
                    }
                } else {
                    if (self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                        return QuantMode_t::VQF322B8_PRE;
                    } else {
                        return QuantMode_t::QF322B8_PRE;
                    }
                }
            }
        }
    }

    return QuantMode_t::F322F16;
}

};

#endif // CONV_INSTR_IMPL_H