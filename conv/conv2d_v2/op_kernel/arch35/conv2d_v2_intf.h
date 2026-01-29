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
 * \file conv2d_v2_intf.h
 * \brief
 */

#ifndef CONV2D_V2_INTF_H
#define CONV2D_V2_INTF_H

#include "../../common/arch35/conv_common_func.h"
#include "../../common/arch35/conv_framework_util.h"
#include "conv2d_v2_common_func.h"
#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"

namespace conv2d {
using namespace conv;
using namespace ConvFunc;

template<class Config, template<typename, class>class Impl>
struct Conv2dIntf {
    using Ext = Impl<Conv2dIntf, Config>;
    using FmapT = typename Config::FmapT;
    using WeightT = typename Config::WeightT;
    using OutputT = typename Config::OutputT;
    using Output1T = typename Config::Output1T;
    using BiasT = typename Config::BiasT;
    using ScaleT = typename Config::ScaleT;
    using ReluWeightT = typename Config::ReluWeightT;
    using ClipValue0T = typename Config::ClipValue0T;
    using ClipValue1T = typename Config::ClipValue1T;
    using L0cT = typename Config::L0cT;
    using ContextType = typename Ext::ContextData;
    using ImplDataType = typename Ext::ImplDataType;
    using ConvParam = typename Config::ConvParam;

    constexpr static bool isExtendConv2d = ConvParam::isExtendConv2d;
    constexpr static int8_t outputOrder = ConvParam::outputOrder;
    constexpr static int8_t groupType = ConvParam::groupType;
    constexpr static int8_t l0PingPong = ConvParam::l0PingPong;
    constexpr static int8_t l1PingPong = ConvParam::l1PingPong;
    constexpr static bool bL1DBFlag = ConvParam::l1PingPong == static_cast<int8_t>(ConvL1PingPong::BL1_OPEN) ||
                                      ConvParam::l1PingPong == static_cast<int8_t>(ConvL1PingPong::ALL_OPEN);
    constexpr static bool hasHL1IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::fmapTiling == 0 ||
        ConvParam::fmapTiling == 1) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool hasWL1IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::fmapTiling == 0 ||
        ConvParam::fmapTiling == 1) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool hasHL0IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::fmapTiling == 1) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool hasWL0IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::fmapTiling == 1) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool hasNL0IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::weightTiling == 1) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool hasNL1IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::weightTiling == 1 ||
        ConvParam::weightTiling == 0) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool hasML1IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::fmapTiling == 0 ||
        ConvParam::fmapTiling == 1) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool hasML0IterFlag = !(ConvParam::l0PingPong == 0 || ConvParam::fmapTiling == 1) ||
        !(ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV));
    constexpr static bool kl0FullLoadFlag = (ConvParam::l0PingPong == 1 || ConvParam::l0PingPong == 2) &&
                                            ConvParam::groupType != static_cast<int8_t>(ConvGroupType::ORI_GROUP_CONV);

    constexpr static bool kPreLoadAFlag =
        ConvParam::l1PingPong == static_cast<int8_t>(ConvL1PingPong::AL1_OPEN) && ConvParam::weightTiling == 0;
    constexpr static bool kPreLoadBFlag =
        ConvParam::l1PingPong == static_cast<int8_t>(ConvL1PingPong::BL1_OPEN) && ConvParam::fmapTiling == 0;
    constexpr static bool kPreLoadABFlag =
        ConvParam::l1PingPong == static_cast<int8_t>(ConvL1PingPong::ALL_OPEN) &&
        ConvParam::groupType == static_cast<int8_t>(ConvGroupType::NORMAL_CONV);
    constexpr static bool kPreLoadFlag =
        kPreLoadAFlag || kPreLoadBFlag || kPreLoadABFlag;
    constexpr static bool iterateMFirstFlag = ConvParam::iterOrder == 0;
    constexpr static bool iterateNFirstFlag = ConvParam::iterOrder == 1;

    constexpr static auto formatFmap = Config::formatFmap;
    constexpr static auto formatWeight = Config::formatWeight;
    constexpr static auto formatOutput = Config::formatOutput;
    constexpr static auto posOutput = Config::posOutput;
    constexpr static bool WEIGHT_NZ_FLAG = Config::WEIGHT_NZ_FLAG;
    constexpr static uint64_t k0 = C0_SIZE / sizeof(WeightT);
    constexpr static uint64_t k0FmapTail = C0_SIZE / sizeof(FmapT);
    constexpr static bool isFixedPoint = (AscendC::IsSameType<WeightT, half>::value &&
                                          AscendC::IsSameType<L0cT, int32_t>::value);
    constexpr static bool isQuantScene =
        !isFixedPoint &&
        ((AscendC::IsSameType<L0cT, int32_t>::value && AscendC::IsSameType<OutputT, half>::value) ||
        (AscendC::IsSameType<L0cT, int32_t>::value && AscendC::IsSameType<OutputT, int8_t>::value) ||
        AscendC::IsSameType<FmapT, hifloat8_t>::value ||
        AscendC::IsSameType<FmapT, fp8_e4m3fn_t>::value);

    constexpr static bool c04Flag =
        ConvParam::enableSmallChannel == static_cast<int8_t>(ConvEnableSmallChannel::OPEN);
    constexpr static bool c04NDFlag = formatWeight != ConvFormat::FRACTAL_Z_C04 &&
        ConvParam::enableSmallChannel == static_cast<int8_t>(ConvEnableSmallChannel::OPEN);

    constexpr static bool weightUbTrans =
        ConvParam::weightUbTrans == static_cast<int8_t>(ConvWeightUbTrans::OPEN);

    constexpr static bool isDmaFlag =
        ConvParam::fmapCopyMode == static_cast<int8_t>(ConvFmapCopyMode::DMA);

    constexpr static bool isInnerBatchFlag =
        ConvParam::innerBatch != static_cast<int8_t>(ConvInnerBatch::SINGLE_BATCH);

    constexpr static bool groupOptFlag =
        ConvParam::groupType == static_cast<int8_t>(ConvGroupType::OPT_GROUP_CONV);
    constexpr static bool groupOptNDFlag = ConvParam::groupType == static_cast<int8_t>(ConvGroupType::OPT_GROUP_CONV) &&
                                           formatWeight != ConvFormat::FRACTAL_Z;
    constexpr static bool groupOptNZFlag = ConvParam::groupType == static_cast<int8_t>(ConvGroupType::OPT_GROUP_CONV) &&
                                           formatWeight == ConvFormat::FRACTAL_Z;
    constexpr static bool isConv3D = false;
    constexpr static bool isDeQuantFlag = false;
    constexpr static bool disContinuousFlag =
        ConvParam::disContinuous == static_cast<int8_t>(ConvDisContinuous::INPUT_HWNC);

    constexpr static uint8_t sizeOfFmap = sizeof(FmapT);
    constexpr static uint8_t sizeOfWeight = sizeof(WeightT);
    constexpr static uint8_t sizeOfBias = sizeof(BiasT);
    constexpr static uint8_t sizeOfScale = sizeof(ScaleT);
    constexpr static uint8_t sizeOfL0c = sizeof(L0cT);

public:
    ContextType ctx;
    ImplDataType impl;

    __aicore__ inline Conv2dIntf() {}

    __aicore__ inline void Init(const void* __restrict convTiling)
    {
        using local = typename Ext::Init;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, convTiling)) {
            local::call(this, convTiling);
        }
    }

    __aicore__ inline void SetFmap(const GlobalTensor<FmapT>& fmap)
    {
        using local = typename Ext::SetFmap;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, fmap)) {
            local::call(this, fmap);
        }
    }

    __aicore__ inline void SetWeight(const GlobalTensor<WeightT>& weight)
    {
        using local = typename Ext::SetWeight;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, weight)) {
            local::call(this, weight);
        }
    }

    __aicore__ inline void SetBias(const GlobalTensor<BiasT>& bias)
    {
        using local = typename Ext::SetBias;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, bias)) {
            local::call(this, bias);
        }
    }

    __aicore__ inline void SetScale(const GlobalTensor<ScaleT>& scale)
    {
        using local = typename Ext::SetScale;
        if constexpr(CONV_CHECK_FUN(local, ConvFunc, this, scale)) {
            local::call(this, scale);
        }
    }

    __aicore__ inline void SetFixpipeParams(
        const Extendconv2dFixpipeParams<ScaleT, ReluWeightT, ClipValue0T, ClipValue1T>& fixpipeParams)
    {
        using local = typename Ext::SetFixpipeParams;
        if constexpr(CONV_CHECK_FUN(local, ConvFunc, this, fixpipeParams)) {
            local::call(this, fixpipeParams);
        }
    }

    __aicore__ inline void SetOrgBatch(uint64_t orgBatch)
    {
        using local = typename Ext::SetOrgBatch;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, orgBatch)) {
            local::call(this, orgBatch);
        }
    }

    __aicore__ inline void SetOrgFmapShape(uint64_t orgCi, uint64_t orgHi, uint64_t orgWi)
    {
        using local = typename Ext::SetOrgFmapShape;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, orgCi, orgHi, orgWi)) {
            local::call(this, orgCi, orgHi, orgWi);
        }
    }

    __aicore__ inline void SetOrgWeightShape(uint64_t orgCo, uint64_t orgCi, uint64_t orgKh, uint64_t orgKw)
    {
        using local = typename Ext::SetOrgWeightShape;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, orgCo, orgCi, orgKh, orgKw)) {
            local::call(this, orgCo, orgCi, orgKh, orgKw);
        }
    }

    __aicore__ inline void SetOrgOutputShape(uint64_t orgCo, uint64_t orgHo, uint64_t orgWo)
    {
        using local = typename Ext::SetOrgOutputShape;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, orgCo, orgHo, orgWo)) {
            local::call(this, orgCo, orgHo, orgWo);
        }
    }

    __aicore__ inline void SetSingleFmapShape(uint64_t singleCi, uint64_t singleHi, uint64_t singleWi)
    {
        using local = typename Ext::SetSingleFmapShape;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, singleCi, singleHi, singleWi)) {
            local::call(this, singleCi, singleHi, singleWi);
        }
    }

    __aicore__ inline void SetSingleOutputShape(uint64_t singleCo, uint64_t singleHo, uint64_t singleWo,
                                                uint64_t singleCoreBatch)
    {
        using local = typename Ext::SetSingleOutputShape;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, singleCo, singleHo, singleWo,
                                     singleCoreBatch)) {
            local::call(this, singleCo, singleHo, singleWo, singleCoreBatch);
        }
    }

    __aicore__ inline void SetSingleOutputShape(uint64_t singleCo, uint64_t singleM, uint64_t singleCoreBatch)
    {
        using local = typename Ext::SetSingleOutputShape;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, singleCo, singleM, singleCoreBatch)) {
            local::call(this, singleCo, singleM, singleCoreBatch);
        }
    }

    __aicore__ inline void SetOptGroupParams(uint64_t singleGroups, uint64_t singleGroupOpt)
    {
        using local = typename Ext::SetOptGroupParams;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, singleGroups, singleGroupOpt)) {
            local::call(this, singleGroups, singleGroupOpt);
        }
    }

    __aicore__ inline void SetFmapStartPosition(int64_t hiStartPos, int64_t wiStartPos)
    {
        using local = typename Ext::SetFmapStartPosition;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, hiStartPos, wiStartPos)) {
            local::call(this, hiStartPos, wiStartPos);
        }
    }

    __aicore__ inline void SetFmapStartPosition(int64_t mStartPos)
    {
        using local = typename Ext::SetFmapStartPosition;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, mStartPos)) {
            local::call(this, mStartPos);
        }
    }

    __aicore__ inline void SetWeightStartPosition(int64_t coStartPos, int64_t ciStartPos = 0)
    {
        using local = typename Ext::SetWeightStartPosition;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, coStartPos, ciStartPos)) {
            local::call(this, coStartPos, ciStartPos);
        }
    }

    __aicore__ inline void SetIterIndex(uint64_t groupOptIter)
    {
        using local = typename Ext::SetIterIndex;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this, groupOptIter)) {
            local::call(this, groupOptIter);
        }
    }

    __aicore__ inline void ConvPreProcess()
    {
        using local = typename Ext::ConvPreProcess;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this)) {
            local::call(this);
        }
    }

    __aicore__ inline void ConvPostProcess()
    {
        using local = typename Ext::ConvPostProcess;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this)) {
            local::call(this);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<OutputT>& output, bool enPartialSum = false)
    {
        using local = typename Ext::IterateAll;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, Conv2dFunc, sync, this, output, enPartialSum)) {
            local::template call<sync>(this, output, enPartialSum);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<OutputT>& output0,
                                      const GlobalTensor<Output1T>& output1, bool enPartialSum = false)
    {
        using local = typename Ext::IterateAll;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, Conv2dFunc, sync, this, output0, output1, enPartialSum)) {
            local::template call<sync>(this, output0, output1, enPartialSum);
        }
    }

    __aicore__ inline void End()
    {
        using local = typename Ext::End;
        if constexpr (CONV_CHECK_FUN(local, Conv2dFunc, this)) {
            local::call(this);
        }
    }

    template <bool sync = true>
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        using local = typename Ext::Iterate;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, Conv2dFunc, sync, this, enPartialSum)) {
            return local::template call<sync>(this, enPartialSum);
        }
        return false;
    }

    template <template <typename> class TensorTypeT, bool sync = true>
    __aicore__ inline void GetTensorC(const TensorTypeT<OutputT>& output, CopyUbInfo* ubInfo = nullptr, bool enSequentialWrite = false)
    {
        using local = typename Ext::GetTensorC;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, Conv2dFunc, TensorTypeT, this, output, ubInfo, enSequentialWrite)) {
            local::template call<TensorTypeT, sync>(this, output, ubInfo, enSequentialWrite);
        }
    }

    template <template <typename> class TensorTypeT, bool sync = true>
    __aicore__ inline void GetTensorC(const TensorTypeT<OutputT>& output0, const GlobalTensor<Output1T>& output1,
                                      CopyUbInfo* ubInfo = nullptr, bool enSequentialWrite = false)
    {
        using local = typename Ext::GetTensorC;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, Conv2dFunc, TensorTypeT, this, output0, output1, ubInfo, enSequentialWrite)) {
            local::template call<TensorTypeT, sync>(this, output0, output1, ubInfo, enSequentialWrite);
        }
    }
};

}  // namespace conv2d

#endif // CONV2D_V2_INTF_H