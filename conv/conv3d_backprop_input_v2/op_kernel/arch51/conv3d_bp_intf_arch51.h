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
 * \file conv3d_bp_intf_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INTF_ARCH51_H
#define CONV3D_BP_INTF_ARCH51_H

#include "conv3d_bp_config_arch51.h"
#include "conv3d_bp_func_arch51.h"
#include "conv3d_bp_util_arch51.h"
#include "kernel_tiling/kernel_tiling.h"

namespace Convolution3DBackprop {
template <class Config_, template <typename, class> class Impl>
struct ConvBpIntf {
    using Config = Config_;
    using Ext = Impl<ConvBpIntf, Config>;
    using SrcT = typename Config::SrcT;
    using SrcBT = typename Config::SrcBT;
    using SrcAT = typename Config::SrcAT;
    using DstT = typename Config::DstT;
    using L0cT = typename Config::L0cT;
    using BiasT = typename Config::BiasT;
    using ScaleT0 = typename Config::ScaleT0;
    using ContextData = typename Ext::ContextData;

public:
    ContextData ctx;

public:
    __aicore__ inline ConvBpIntf() {}

    __aicore__ inline void Init(const Conv3DBackpropInputArch35TilingData& tiling, const bool hasBias = false)
    {
        using Local = typename Ext::Init;
        // CheckFun检查impl是否实现了Init的call函数
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, tiling, hasBias)) {
            Local::call(this, tiling, hasBias);
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

    __aicore__ inline void SetSingleShape(uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN,
                                          uint32_t singleShapeD)
    {
        using Local = typename Ext::SetSingleShape;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, singleShapeM, singleShapeK, singleShapeN,
                                singleShapeD)) {
            Local::call(this, singleShapeM, singleShapeK, singleShapeN, singleShapeD);
        }
    }

    __aicore__ inline void SetStartIdx(uint32_t curDinStartIdx, int64_t curMStartIdx, int32_t curCinStartIdx)
    {
        using Local = typename Ext::SetStartIdx;
        if constexpr (CHECK_FUN(Local, Convolution3DBackpropFunc, this, curDinStartIdx, curMStartIdx, curCinStartIdx)) {
            Local::call(this, curDinStartIdx, curMStartIdx, curCinStartIdx);
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
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& output, uint8_t enAtomic = 0,
                                      bool enSequentialWrite = false)
    {
        using Local = typename Ext::template GetTensorC<sync>;
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

template <class Config_, template <typename, class> class Impl>
struct Conv3DBpInputIntf : public ConvBpIntf<Config_, Impl> {
public:
    __aicore__ inline Conv3DBpInputIntf() {}
};

// 注册，通过别名定义用户接口（5162 单输出：无 y1/scale1 类型槽）
#define REGISTER_DX_IMPL(name, context, impl, intf)                                          \
    template <class X_T, class W_TYPE, class DEDY_T, class Y_T, class BIAS_T, class SCALE_T> \
    using name = intf<context<X_T, W_TYPE, DEDY_T, Y_T, BIAS_T, SCALE_T>, impl>

} // namespace Convolution3DBackprop

#endif
