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
 * \file conv_bp_intf_base.h
 * \brief
 */

#ifndef CONV_BP_INTF_H
#define CONV_BP_INTF_H

#include "conv_bp_func.h"
#include "conv_bp_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"

namespace ConvolutionBackprop {
// 用户可见的api原型集合
template <class Config_, template <typename, class> class Impl>
struct ConvBpIntf {
    using Config = Config_;
    using Ext = Impl<ConvBpIntf, Config>;
    using SrcT = typename Config::SrcT;
    using DstT = typename Config::DstT;
    using L0cT = typename Config::L0cT;
    using ContextData = typename Ext::ContextData;

public:
    ContextData ctx;
    constexpr static Conv3ddwConfig conv3ddwConfig = Config::conv3ddwConfig_;

public:
    __aicore__ inline ConvBpIntf()
    {}

    __aicore__ inline void Init(const AscendC::conv_bp_v2_kernel::TConv3DDwTiling *__restrict tiling, bool seperateDk = true)
    {
        using Local = typename Ext::Init;
        // CheckFun检查impl是否实现了Init的call函数
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, tiling, seperateDk)) {
            Local::call(this, tiling, seperateDk);
        }
    }

    __aicore__ inline void SetFmap(const GlobalTensor<SrcT> &fmap)
    {
        using Local = typename Ext::SetFmap;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, fmap)) {
            Local::call(this, fmap);
        }
    }

    __aicore__ inline void SetOutBackprop(const GlobalTensor<SrcT> &outBackprop)
    {
        using Local = typename Ext::SetOutBackprop;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, outBackprop)) {
            Local::call(this, outBackprop);
        }
    }

    __aicore__ inline void SetSingleShape(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t singleCoreK,
                                          uint32_t singleShapeCin, uint32_t singleShapeBatch)
    {
        using Local = typename Ext::SetSingleShape;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, singleCoreM, singleCoreN, singleCoreK,
                                                                      singleShapeCin, singleShapeBatch)) {
            Local::call(this, singleCoreM, singleCoreN, singleCoreK, singleShapeCin, singleShapeBatch);
        }
    }

    __aicore__ inline void SetStartIdx(uint64_t batchDoutIdx, uint32_t hoStartIdx, int32_t dkIdx)
    {
        using Local = typename Ext::SetStartIdx;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, batchDoutIdx, hoStartIdx, dkIdx)) {
            Local::call(this, batchDoutIdx, hoStartIdx, dkIdx);
        }
    }

    template <bool sync = true>
    __aicore__ inline bool UpdateMNIdx(ConvolutionBackpropFunc::Out2L1ScalarParams& out2L1Params)
    {
        using Local = typename Ext::template UpdateMNIdx<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, out2L1Params)) {
            return Local::call(this, out2L1Params);
        }
    }

    template <bool sync = true>
    __aicore__ inline void Compute(ConvolutionBackpropFunc::Out2L1ScalarParams& out2L1Params)
    {
        using Local = typename Ext::template Compute<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, out2L1Params)) {
            Local::call(this, out2L1Params);
        }
    }

    template <bool sync = true>
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        using Local = typename Ext::template Iterate<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, enPartialSum)) {
            return Local::call(this, enPartialSum);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &output, uint8_t enAtomic = 0)
    {
        using Local = typename Ext::template IterateAll<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, output, enAtomic)) {
            Local::call(this, output, enAtomic);
        }
    }

    template <bool sync = true>
    __aicore__ inline void GetTensorC(
        const GlobalTensor<DstT> &output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        using Local = typename Ext::template GetTensorC<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, output, enAtomic, enSequentialWrite)) {
            Local::call(this, output, enAtomic, enSequentialWrite);
        }
    }

    template <bool sync = true>
    __aicore__ inline void VecPostProcess(
        const GlobalTensor<DstT> &output, uint8_t enAtomic = 0)
    {
        using Local = typename Ext::template VecPostProcess<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, output, enAtomic)) {
            Local::call(this, output, enAtomic);
        }
    }

    template <bool sync = true>
    __aicore__ inline void DeterministicReduceKInUb(const GlobalTensor<DstT> &output,
        const GlobalTensor<DstT> &vecGm)
    {
        using Local = typename Ext::template DeterministicReduceKInUb<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, output, vecGm)) {
            Local::call(this, output, vecGm);
        }
    }

    __aicore__ inline void SetDeterministicCoreInfo(uint32_t addCoreNum, uint32_t addCoreIndex,
        uint32_t relatedCoreIndexTotal, bool isNoDeter)
    {
        using Local = typename Ext::SetDeterministicCoreInfo;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, addCoreNum, addCoreIndex,
            relatedCoreIndexTotal, isNoDeter)) {
            Local::call(this, addCoreNum, addCoreIndex, relatedCoreIndexTotal, isNoDeter);
        }
    }

    __aicore__ inline void End()
    {
        using Local = typename Ext::End;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this)) {
            Local::call(this);
        }
    }
};

}  // namespace ConvolutionBackprop

#endif
