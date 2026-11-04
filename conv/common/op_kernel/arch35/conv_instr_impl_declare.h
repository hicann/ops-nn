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
 * \file conv_instr_impl_declare.h
 * \brief
 */

#ifndef CONV_INSTR_IMPL_DECLARE_H
#define CONV_INSTR_IMPL_DECLARE_H

#include "conv_config.h"
#include "conv_util.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf, typename OutputT, uint64_t FixpipeIdx = 0>
class CopyOutToolsHWMode {
public:
    __aicore__ inline CopyOutToolsHWMode() {}
    __aicore__ inline void SetParams(Intf* self)
    {
        self_ = self;
        valueHoWo_ = self_->ctx.convTilingData->orgHo * self_->ctx.convTilingData->orgWo;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void SetFixpipeIntriParams(FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR>& intriParams);

    __aicore__ inline void SetFixpipeIntriParamsHWC(FixpipeParamsArch3510<CO2Layout::ROW_MAJOR>& intriParams);

    template <CO2Layout format>
    __aicore__ inline void SetFixpipeIntriParamsUb(FixpipeParamsArch3510<format>& intriParams,
                                                   CopyUbInfo* ubInfo = nullptr);

    template <CO2Layout format>
    __aicore__ inline void SetBaseParams(FixpipeParamsArch3510<format>& intriParams);

    __aicore__ inline uint64_t CalcFixpipeOffset();

    template <template <typename> class TensorTypeT, const FixpipeConfig& config>
    __aicore__ inline void ExtendConv2DFixpipe(const TensorTypeT<OutputT>& output,
                                               FixpipeParamsArch3510<config.format>& intriParams, uint64_t offset);

    __aicore__ inline uint64_t GetExtendConv2dScaleL1Addr();

    template <template <typename> class TensorTypeT, const FixpipeConfig& config>
    __aicore__ inline void CopyOut(const TensorTypeT<OutputT>& output, CopyUbInfo* ubInfo = nullptr);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    __aicore__ inline uint64_t CalcFixpipeOffset(IterateParam& iterParam);

    __aicore__ inline void SetFixpipeIntriParamsConvToL1(FixpipeParamsV516& intriParams, IterateParam& iterParam);

    template <template <typename> class TensorTypeT, const FixpipeConfig& config>
    __aicore__ inline void CopyOut(const TensorTypeT<OutputT>& output, IterateParam& iterParam,
                                   CopyUbInfo* ubInfo = nullptr);
#endif

private:
    Intf* self_ = nullptr;
    uint64_t valueHoWo_ = 0;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;

    __aicore__ inline uint64_t GetScaleL1Addr()
    {
        if constexpr (Intf::isKL1NL0FullLoad) {
            return 0;
        }
        if constexpr (Intf::isQuantScene) {
            if (self_->ctx.convTilingData->hasScale != 0 && self_->ctx.convTilingData->fixpParamsFullLoadFlag) {
                return self_->ctx.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                       self_->ctx.nL0Iter * self_->ctx.convTilingData->nL0;
            }
        }
        return 0;
    }
};

template <class Intf, typename ChannelWiseT>
class LoadChannelWiseL1Tools {
public:
    __aicore__ inline LoadChannelWiseL1Tools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void SetN(uint64_t n) { currentNL0_ = n; }

    __aicore__ inline void LoadChannelWiseL1FullLoad(const LocalTensor<ChannelWiseT>& tensorL1,
                                                     const GlobalTensor<ChannelWiseT>& tensorGm, uint64_t loadNum,
                                                     uint64_t gmStartAddr);

    __aicore__ inline void LoadChannelWiseL1(const LocalTensor<ChannelWiseT>& tensorL1,
                                             const GlobalTensor<ChannelWiseT>& tensorGm);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    __aicore__ inline void LoadChannelWiseL1(const LocalTensor<ChannelWiseT>& tensorL1,
                                             const GlobalTensor<ChannelWiseT>& tensorGm, IterateParam& iterParam);
#endif

private:
    Intf* self_ = nullptr;
    uint64_t currentNL0_ = 0;
};

template <class Intf, typename BiasL1T, typename BiasBtT>
class LoadBiasBtTools {
public:
    __aicore__ inline LoadBiasBtTools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void SetN(uint64_t n) { currentNL0_ = n; }

    __aicore__ inline void LoadBiasBt(const LocalTensor<BiasBtT>& biasBt, const LocalTensor<BiasL1T>& biasL1);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    __aicore__ inline void LoadBiasBt(const LocalTensor<BiasBtT>& biasBt, const LocalTensor<BiasL1T>& biasL1,
                                      IterateParam& iterParam);
#endif

private:
    Intf* self_ = nullptr;
    uint64_t currentNL0_ = 0;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
template <class Intf, typename ChannelWiseT>
class LoadChannelWiseFBTools {
public:
    __aicore__ inline LoadChannelWiseFBTools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }
    __aicore__ inline void LoadChannelWiseFB(const LocalTensor<ChannelWiseT>& tensorFB,
                                             const LocalTensor<ChannelWiseT>& tensorL1, IterateParam& iterParam,
                                             const PostProcBufBlockMode& postProcBufBlockMode);

private:
    Intf* self_ = nullptr;
};
#endif

template <class Intf>
class LoadBL0Tools {
public:
    __aicore__ inline LoadBL0Tools() {}

    __aicore__ inline void SetParams(Intf* self)
    {
        self_ = self;
        nStep_ = self_->ctx.convTilingData->nL0 / BLOCK_L0_N;
    }

    __aicore__ inline void SetN(uint64_t n);

    __aicore__ inline void SetFirst();

    __aicore__ inline void LoadBL0(const uint64_t& KStartPosition, const uint64_t& kStep,
                                   const LocalTensor<typename Intf::WeightT>& bl0);

    __aicore__ inline void FullLoadBL0(const LocalTensor<typename Intf::WeightT>& bl0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    __aicore__ inline void SetN(IterateParam& iterParam);

    __aicore__ inline void SetFirst(IterateParam& iterParam);

    __aicore__ inline uint64_t CalcOffsetForWeightByPass(IterateParam& iterParam);

    __aicore__ inline void LoadBL0(const uint64_t& kStartPosition, const uint64_t& kStep,
                                   const LocalTensor<typename Intf::WeightT>& bl0, uint64_t bgmOffset);
#endif

private:
    Intf* self_ = nullptr;
    uint64_t ratioOfNToN0 = 0;
    uint64_t nStep_ = 0;
    Load2DBitModeParam param_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    LoadData2DParamsV2 weightByPassParam_;
#endif
};

template <class Intf>
class MMadTools {
public:
    __aicore__ inline MMadTools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void Mad(const uint64_t& kIter, const LocalTensor<typename Intf::FmapT>& al0,
                               const LocalTensor<typename Intf::WeightT>& bl0, const MmadParams& mmadParams);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    __aicore__ inline void Mad(const LocalTensor<typename Intf::FmapT>& al0,
                               const LocalTensor<typename Intf::WeightT>& bl0,
                               const ConvFixBitModeParams& convFixParams, IterateParam& iterParam);
#endif

private:
    __aicore__ inline bool IsKL0Tail(const uint64_t& kIter) { return kIter == self_->ctx.maxKL0Iter; }

private:
    Intf* self_ = nullptr;
};

} // namespace ConvFunc

#endif // CONV_INSTR_IMPL_DECLARE_H
