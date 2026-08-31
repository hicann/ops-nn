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
 * \file conv3d_bp_intf_base.h
 * \brief
 */

#ifndef CONV3D_BP_INTF_ADVANCE_H
#define CONV3D_BP_INTF_ADVANCE_H

#include "conv3d_bp_config_base.h"
#include "conv3d_bp_func_arch35.h"
#include "conv3d_bp_util_arch35.h"
#include "kernel_tiling/kernel_tiling.h"

namespace Convolution3DBackprop {
// 用户可见的api原型集合
template <class Config_, template <typename, class> class Impl>
struct ConvBpIntf {
    using Config = Config_;
    using Ext = Impl<ConvBpIntf, Config>;
    // 透传 Config 的 IsSecondOutput（基类 ConvBpContext 为 false；Output1Intf 覆盖为 true）。
    constexpr static bool IsSecondOutput = Config::IsSecondOutput;
    using SrcT = typename Config::SrcT;
    using SrcBT = typename Config::SrcBT;
    using SrcAT = typename Config::SrcAT;
    using DstT = typename Config::DstT;
    using Dst1T = typename Config::Dst1T;
    using L0cT = typename Config::L0cT;
    using BiasT = typename Config::BiasT;
    using ScaleT0 = typename Config::ScaleT0;
    using Scale1T = typename Config::Scale1T;
    using IndexT = typename AscendC::Conditional<AscendC::IsSameType<SrcBT, float>::value, uint32_t, uint16_t>::type;
    using ContextData = typename Ext::ContextData;

public:
    ContextData ctx;
    constexpr static Conv3dConfig conv3dConfig = Config::conv3dConfig_;

public:
    __aicore__ inline ConvBpIntf() {}

    __aicore__ inline void Init(const Conv3DBackpropInputArch35TilingData& tiling, const bool hasBias = false,
                                const bool hasSecondOutput = false)
    {
        using Local = typename Ext::Init;
        // CheckFun检查impl是否实现了Init的call函数
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, tiling, hasBias, hasSecondOutput)) {
            Local::call(this, tiling, hasBias, hasSecondOutput);
        }
    }

    __aicore__ inline void SetFmap(const GlobalTensor<SrcAT>& fmap)
    {
        using Local = typename Ext::SetFmap;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, fmap)) {
            Local::call(this, fmap);
        }
    }

    __aicore__ inline void SetWeight(const GlobalTensor<SrcBT>& weight)
    {
        using Local = typename Ext::SetWeight;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, weight)) {
            Local::call(this, weight);
        }
    }

    __aicore__ inline void SetOutBackprop(const GlobalTensor<SrcAT>& outBackprop)
    {
        using Local = typename Ext::SetOutBackprop;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, outBackprop)) {
            Local::call(this, outBackprop);
        }
    }

    __aicore__ inline void SetBias(const GlobalTensor<BiasT>& bias)
    {
        using Local = typename Ext::SetBias;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, bias)) {
            Local::call(this, bias);
        }
    }

    __aicore__ inline void SetScale(const GlobalTensor<ScaleT0>& scale)
    {
        using Local = typename Ext::SetScale;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, scale)) {
            Local::call(this, scale);
        }
    }

    __aicore__ inline void SetScale1(const GlobalTensor<Scale1T>& scale)
    {
#ifdef DTYPE_Y1
        ctx.scale1Global_ = scale;
        ctx.hasSecondOutput_ = true;
#endif
    }

    __aicore__ inline void SetKernelSplitParams(uint32_t kSCoutFullLoad, uint32_t kSUseWorkSpace)
    {
        using Local = typename Ext::SetKernelSplitParams;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, kSCoutFullLoad, kSUseWorkSpace)) {
            Local::call(this, kSCoutFullLoad, kSUseWorkSpace);
        }
    }

    __aicore__ inline void SetSingleShapeParams(uint32_t curSplitHk, int32_t curBackpropPadUp)
    {
        using Local = typename Ext::SetSingleShapeParams;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, curSplitHk, curBackpropPadUp)) {
            Local::call(this, curSplitHk, curBackpropPadUp);
        }
    }

    __aicore__ inline void SetSingleShape(uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN,
                                          uint32_t singleShapeD)
    {
        using Local = typename Ext::SetSingleShape;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, singleShapeM, singleShapeK, singleShapeN,
                                singleShapeD)) {
            Local::call(this, singleShapeM, singleShapeK, singleShapeN, singleShapeD);
        }
    }

    __aicore__ inline void SetStartIdx(uint32_t curDinStartIdx, int64_t curMStartIdx, int32_t curCinStartIdx,
                                       int32_t curCoutStartIdx)
    {
        using Local = typename Ext::SetStartIdx;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, curDinStartIdx, curMStartIdx, curCinStartIdx,
                                curCoutStartIdx)) {
            Local::call(this, curDinStartIdx, curMStartIdx, curCinStartIdx, curCoutStartIdx);
        }
    }

    __aicore__ inline void SetBatchCoreIdx(uint32_t batchCoreIdx)
    {
        using Local = typename Ext::SetBatchCoreIdx;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, batchCoreIdx)) {
            Local::call(this, batchCoreIdx);
        }
    }

    __aicore__ inline void FreeB1Tensor()
    {
        using Local = typename Ext::FreeB1Tensor;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this)) {
            Local::call(this);
        }
    }

    __aicore__ inline void FreeBiasTensor()
    {
        using Local = typename Ext::FreeBiasTensor;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this)) {
            Local::call(this);
        }
    }

    template <bool sync = true>
    __aicore__ inline bool Iterate(bool enPartialSum = false, bool hasBias = false)
    {
        using Local = typename Ext::template Iterate<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, enPartialSum, hasBias)) {
            return Local::call(this, enPartialSum, hasBias);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& output, uint8_t enAtomic = 0,
                                      bool fullLoadBiasFlag_ = false, bool freeBiasFlag_ = false)
    {
        using Local = typename Ext::template IterateAll<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, output, enAtomic, fullLoadBiasFlag_,
                                freeBiasFlag_)) {
            Local::call(this, output, enAtomic, fullLoadBiasFlag_, freeBiasFlag_);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAllForKernelSplit(const GlobalTensor<DstT>& output, uint8_t enAtomic = 0)
    {
        using Local = typename Ext::template IterateAllForKernelSplit<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, output, enAtomic)) {
            Local::call(this, output, enAtomic);
        }
    }

    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& output, uint8_t enAtomic = 0,
                                      bool enSequentialWrite = false)
    {
        using Local = typename Ext::template GetTensorC<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, output, enAtomic, enSequentialWrite)) {
            Local::call(this, output, enAtomic, enSequentialWrite);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& output0, const GlobalTensor<Dst1T>& output1,
                                      uint8_t enAtomic = 0, bool fullLoadBiasFlag_ = false, bool freeBiasFlag_ = false)
    {
        bool hasBias = ctx.hasBias_;
        if (unlikely(hasBias && ctx.tiling_->isBiasFullLoad)) {
            if (freeBiasFlag_) {
                FreeBiasTensor();
            }
            if (fullLoadBiasFlag_) {
                Convolution3DBackpropFunc::FullLoadBias<ConvBpIntf<Config_, Impl>>(this);
            }
        }
        Convolution3DBackpropFunc::SetDequantScale<ConvBpIntf<Config_, Impl>>(this);
        if (ctx.enableSplitK_) {
            Convolution3DBackpropFunc::CalcSplitK_<ConvBpIntf<Config_, Impl>, sync>(this, enAtomic, output0, hasBias);
            if (ctx.useUbAccumForSplitK_ && ctx.needComputeFlag_) {
                using Intf1 = Convolution3DBackprop::Output1Intf<ConvBpIntf<Config_, Impl>>;
                auto* self1 = reinterpret_cast<Intf1*>(this);
                Convolution3DBackpropFunc::AccumulateSegmentOnWorkspace<Intf1>(self1, output1);
            }
        } else {
            while (Iterate<sync>(false, hasBias)) {
                VecPreProcess<sync>(output0, enAtomic);
                GetTensorC<sync>(output0, output1, enAtomic);
                VecPostProcess<sync>(output0, enAtomic);
            }
        }
        if ASCEND_IS_AIC_SCALAR {
            if constexpr (Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
                if (ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                    ctx.scale0L1Que_.FreeTensor(ctx.scale0L1Buf_);
                }
            }
            using Intf1 = Convolution3DBackprop::Output1Intf<ConvBpIntf<Config_, Impl>>;
            if constexpr (Intf1::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
                if (ctx.hasSecondOutput_ &&
                    ctx.tiling_->quantMode1 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                    ctx.scale1L1Que_.FreeTensor(ctx.scale1L1Buf_);
                }
            }
        }
        ctx.isFirstIter_ = true;
    }

    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& output0, const GlobalTensor<Dst1T>& output1,
                                      uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        Convolution3DBackpropFunc::LoadL0c2GmDual(this, output0, output1, enAtomic, enSequentialWrite);
    }

    template <bool sync = true>
    __aicore__ inline void VecPreProcess(const GlobalTensor<DstT>& output, uint8_t enAtomic = 0,
                                         bool enSequentialWrite = false)
    {
        using Local = typename Ext::template VecPreProcess<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, output, enAtomic, enSequentialWrite)) {
            Local::call(this, output, enAtomic, enSequentialWrite);
        }
    }

    template <bool sync = true>
    __aicore__ inline void VecPostProcess(const GlobalTensor<DstT>& output, uint8_t enAtomic = 0,
                                          bool enSequentialWrite = false)
    {
        using Local = typename Ext::template VecPostProcess<sync>;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, output, enAtomic, enSequentialWrite)) {
            Local::call(this, output, enAtomic, enSequentialWrite);
        }
    }

    __aicore__ inline void End()
    {
        using Local = typename Ext::End;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this)) {
            Local::call(this);
        }
    }
};

} // namespace Convolution3DBackprop

#endif
