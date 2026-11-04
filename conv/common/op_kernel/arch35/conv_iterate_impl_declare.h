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
 * \file conv_iterate_impl_declare.h
 * \brief
 */

#ifndef CONV_ITERATE_IMPL_DECLARE_H
#define CONV_ITERATE_IMPL_DECLARE_H

#include "conv_config.h"
#include "conv_framework_util_arch35.h"
#include "conv_iterate_base_impl.h"
#include "conv_iterate_hw_mode_impl.h"
#include "conv_iterate_m_mode_impl.h"
#include "conv_util.h"
#if defined(FORMAT_X) && (FORMAT_X == FORMAT_NCDHW || FORMAT_X == FORMAT_NDHWC)
#include "../../conv3d_v2/arch35/conv3d_v2_dequant_impl.h"
#endif

namespace ConvFunc {

CONV_DECLARE_REG_IMPL(Iterate);

template <class Intf, uint32_t ImplType>
struct Iterate {
    template <bool sync = true>
    static __aicore__ inline bool call(Intf* self, bool enPartialSum = false, bool* updateL1Flag = nullptr);
    static __aicore__ inline bool IterateImplHWMode(Intf* self, bool enPartialSum);
    static __aicore__ inline bool IterateImplMMode(Intf* self, bool enPartialSum);
    static __aicore__ inline void IterateK(Intf* self);
    static __aicore__ inline void IterateBiasScale(Intf* self);
    static __aicore__ inline void SetInitialMmadParams(Intf* self, MmadParams& mmadParams, const uint64_t& currentKL0);
    static __aicore__ inline void SetFinalMmadParams(Intf* self, MmadParams& mmadParams, const uint64_t& currentKL0);
    static __aicore__ inline void ReduceK(Intf* self, MmadParams& mmadParams);
    static __aicore__ inline void ReduceOneK(Intf* self, MmadParams& mmadParams);
    static __aicore__ inline void ReduceKPreloadFmapIter(Intf* self, TempIters& tempIters, bool updateIterByFmapTag,
                                                         const uint64_t& kIter, const uint64_t& multiKAL1);
    static __aicore__ inline void ReduceKPreloadWeightIter(Intf* self, TempIters& tempIters, bool updateIterByFmapTag,
                                                           const uint64_t& kIter, const uint64_t& multiKBL1);
    static __aicore__ inline void ReduceKPreloadFmapIter(Intf* self, TempIters& tempIters, bool updateIterByFmapTag,
                                                         bool ddr2l0LoopKExactDivideMultiKAL1, const uint64_t& kIter,
                                                         const uint64_t& multiKAL1);
    static __aicore__ inline void ReduceKPreloadWeightIter(Intf* self, TempIters& tempIters, bool updateIterByFmapTag,
                                                           bool ddr2l0LoopKExactDivideMultiKBL1, const uint64_t& kIter,
                                                           const uint64_t& multiKBL1);
    static __aicore__ inline void ReduceKPreload(Intf* self, MmadParams& mmadParams);
    static __aicore__ inline void ReduceKFmapPreload(Intf* self, MmadParams& mmadParams);
    static __aicore__ inline void ReduceGroupOptFmapPreload(Intf* self, MmadParams& mmadParams);
    static __aicore__ inline void UpdateNextGroupOptIters(Intf* self, TempIters& tempIters);
    static __aicore__ inline void ReduceKPreloadWithWeightFullloadL0(Intf* self, MmadParams& mmadParams);
    static __aicore__ inline bool UpdateItersByFmap(Intf* self, TempIters& tempIters, bool updateIterByFmapTag,
                                                    const uint64_t& kIter, const uint64_t& multiKAL1);
    static __aicore__ inline bool UpdateItersByWeight(Intf* self, TempIters& tempIters, bool updateIterByFmapTag,
                                                      const uint64_t& kIter, const uint64_t& multiKBL1);
    static __aicore__ inline bool UpdateCommonIters(Intf* self, TempIters& tempIters);
    static __aicore__ inline void ConfigUpdateL1Flag(Intf* self);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    template <bool sync = true>
    static __aicore__ inline bool call(Intf* self, IterateParam& iterParam, bool enPartialSum = false,
                                       bool* updateL1Flag = nullptr);
    static __aicore__ inline bool IterateImplHWMode(Intf* self, IterateParam& iterParam, bool enPartialSum);
    static __aicore__ inline void IterateK(Intf* self, IterateParam& iterParam);
    static __aicore__ inline void IterateBiasScale(Intf* self, IterateParam& iterParam);
    static __aicore__ inline void ReduceK(Intf* self, IterateParam& iterParam);
    static __aicore__ inline void ReduceOneK(Intf* self, IterateParam& iterParam);
    static __aicore__ inline void InitIterParam(Intf* self, IterateParam& iterParam);
    static __aicore__ inline void SetPreQuantAndReluMode(Intf* self, ConvFixBitModeParams& convFixParams);
    static __aicore__ inline void SetParamsFor5162(Intf* self, ConvKernelBitModeParams& kernelParams,
                                                   ConvFMatrixStepPosBitModeParams& convFMatrixParams,
                                                   ConvStrideBitModeParams& strideParams, IterateParam& iterParam);
    static __aicore__ inline void ProcessL0ModuleFor5162(Intf* self, IterateParam& iterParam,
                                                         const uint64_t kStartPosition, const uint64_t kStep,
                                                         uint16_t hStartPosIter, ConvKernelBitModeParams& kernelParams,
                                                         ConvFMatrixStepPosBitModeParams& convFMatrixParams,
                                                         ConvStrideBitModeParams& strideParams,
                                                         ConvFixBitModeParams& convFixParams);
#endif
};

} // namespace ConvFunc

#endif // CONV_ITERATE_IMPL_DECLARE_H
