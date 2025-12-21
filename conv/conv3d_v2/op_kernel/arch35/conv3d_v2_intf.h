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
 * \file conv3d_v2_intf.h
 * \brief
 */

#ifndef CONV3D_V2_INTF_H
#define CONV3D_V2_INTF_H

#include "../../common/arch35/conv_common_func.h"
#include "../../common/arch35/conv_framework_util.h"
#include "conv3d_v2_common_func.h"
#include "conv3d_v2_config.h"
#include "conv3d_v2_util.h"

namespace conv3d {
using namespace conv;

template <class Config, template <typename, class> class Impl>
struct Conv3dIntf {
    using Ext = Impl<Conv3dIntf, Config>;
    using FmapT = typename Config::FmapT;
    using WeightT = typename Config::WeightT;
    using OutputT = typename Config::OutputT;
    using Output1T = half;
    using BiasT = typename Config::BiasT;
    using ScaleT = typename Config::ScaleT;
    using L0cT = typename Config::L0cT;
    using ContextType = typename Ext::ContextData;
    using ImplDataType = typename Ext::ImplDataType;
    using ConvParam = typename Config::ConvParam;

    constexpr static bool isExtendConv2d = false;
    constexpr static int8_t outputOrder = ConvParam::outputOrder;
    constexpr static int8_t groupType = ConvParam::groupType;
    constexpr static int8_t l0PingPong = ConvParam::l0PingPong;
    constexpr static int8_t l1PingPong = ConvParam::l1PingPong;
    constexpr static bool bL1DBFlag = ConvParam::l1PingPong == static_cast<int8_t>(ConvL1PingPong::BL1_OPEN) ||
                                      ConvParam::l1PingPong == static_cast<int8_t>(ConvL1PingPong::ALL_OPEN);
    constexpr static bool hasHL1IterFlag = true;
    constexpr static bool hasWL1IterFlag = true;
    constexpr static bool hasHL0IterFlag = true;
    constexpr static bool hasWL0IterFlag = true;
    constexpr static bool hasNL0IterFlag = true;
    constexpr static bool hasNL1IterFlag = true;
    constexpr static bool hasML1IterFlag = true;
    constexpr static bool hasML0IterFlag = true;
    constexpr static bool kl0FullLoadFlag = (ConvParam::l0PingPong == 1 || ConvParam::l0PingPong == 2) &&
                                            ConvParam::groupType != static_cast<int8_t>(ConvGroupType::ORI_GROUP_CONV);

    constexpr static bool kPreLoadAFlag = false;
    constexpr static bool kPreLoadBFlag = false;
    constexpr static bool kPreLoadABFlag = false;
    constexpr static bool kPreLoadFlag =
        kPreLoadAFlag || kPreLoadBFlag || kPreLoadABFlag;
    constexpr static bool iterateMFirstFlag = ConvParam::iterOrder == 0;
    constexpr static bool iterateNFirstFlag = ConvParam::iterOrder == 1;

    constexpr static auto formatWeight = Config::formatWeight;
    constexpr static auto formatOutput = Config::formatOutput;
    constexpr static auto posOutput = Config::posOutput;
    constexpr static bool WEIGHT_NZ_FLAG = false;
    constexpr static bool isFixedPoint = false;
    constexpr static uint64_t k0 = C0_SIZE / sizeof(WeightT);
    constexpr static uint64_t k0FmapTail = C0_SIZE / sizeof(FmapT);
    constexpr static bool isQuantScene = (AscendC::IsSameType<L0cT, int32_t>::value &&
                                          AscendC::IsSameType<OutputT, half>::value) ||
                                         AscendC::IsSameType<FmapT, hifloat8_t>::value ||
                                         AscendC::IsSameType<FmapT, fp8_e4m3fn_t>::value;
    constexpr static bool c04Flag = false;
    constexpr static bool c04NDFlag = false;
    constexpr static bool weightUbTrans = false;
    constexpr static bool isDmaFlag = false;
    constexpr static bool isInnerBatchFlag = false;
    constexpr static bool groupOptFlag =
        ConvParam::groupType == static_cast<int8_t>(ConvGroupType::OPT_GROUP_CONV);
    constexpr static bool groupOptNDFlag = ConvParam::groupType == static_cast<int8_t>(ConvGroupType::OPT_GROUP_CONV) &&
                                           formatWeight != ConvFormat::FRACTAL_Z;
    constexpr static bool groupOptNZFlag = false;
    constexpr static bool isConv3D = true;

    constexpr static uint8_t sizeOfFmap = sizeof(FmapT);
    constexpr static uint8_t sizeOfWeight = sizeof(WeightT);
    constexpr static uint8_t sizeOfBias = sizeof(BiasT);
    constexpr static uint8_t sizeOfScale = sizeof(ScaleT);
    constexpr static uint8_t sizeOfL0c = sizeof(L0cT);

public:
    ContextType ctx;
    ImplDataType impl;

    __aicore__ inline Conv3dIntf() {}

    __aicore__ inline void Init(const void *__restrict cubeTiling)
    {
        using local = typename Ext::Init;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, cubeTiling)) {
            local::call(this, cubeTiling);
        }
    }

    __aicore__ inline void SetFmap(const GlobalTensor<FmapT> &fmap)
    {
        using local = typename Ext::SetFmap;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, fmap)) {
            local::call(this, fmap);
        }
    }

    __aicore__ inline void SetWeight(const GlobalTensor<WeightT> &weight)
    {
        using local = typename Ext::SetWeight;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, weight)) {
            local::call(this, weight);
        }
    }

    __aicore__ inline void SetBias(const GlobalTensor<BiasT> &bias)
    {
        using local = typename Ext::SetBias;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, bias)) {
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

    __aicore__ inline void SetOrgFmapShape(uint64_t orgCi, uint64_t orgDi, uint64_t orgHi, uint64_t orgWi)
    {
        using local = typename Ext::SetOrgFmapShape;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, orgCi, orgDi, orgHi, orgWi)) {
            local::call(this, orgCi, orgDi, orgHi, orgWi);
        }
    }

    __aicore__ inline void SetOrgWeightShape(
        uint64_t orgCo, uint64_t orgCi, uint64_t orgKd, uint64_t orgKh, uint64_t orgKw)
    {
        using local = typename Ext::SetOrgWeightShape;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, orgCo, orgCi, orgKd, orgKh, orgKw)) {
            local::call(this, orgCo, orgCi, orgKd, orgKh, orgKw);
        }
    }

    __aicore__ inline void SetOrgOutputShape(uint64_t orgCo, uint64_t orgDo, uint64_t orgHo, uint64_t orgWo)
    {
        using local = typename Ext::SetOrgOutputShape;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, orgCo, orgDo, orgHo, orgWo)) {
            local::call(this, orgCo, orgDo, orgHo, orgWo);
        }
    }

    __aicore__ inline void SetSingleFmapShape(
        uint64_t singleCi, uint64_t singleDi, uint64_t singleHi, uint64_t singleWi)
    {
        using local = typename Ext::SetSingleFmapShape;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, singleCi, singleDi, singleHi, singleWi)) {
            local::call(this, singleCi, singleDi, singleHi, singleWi);
        }
    }

    __aicore__ inline void SetSingleOutputShape(
        uint64_t singleCo, uint64_t singleDo, uint64_t singleHo, uint64_t singleWo, uint64_t singleCoreBatch)
    {
        using local = typename Ext::SetSingleOutputShape;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, singleCo, singleDo, singleHo, singleWo, singleCoreBatch)) {
            local::call(this, singleCo, singleDo, singleHo, singleWo, singleCoreBatch);
        }
    }
    __aicore__ inline void SetSingleOutputShape(uint64_t singleCo, uint64_t singleDo, uint64_t singleM,
                                                uint64_t singleCoreBatch)
    {
        using local = typename Ext::SetSingleOutputShape;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, singleCo, singleDo, singleM, singleCoreBatch)) {
            local::call(this, singleCo, singleDo, singleM, singleCoreBatch);
        }
    }

    __aicore__ inline void SetOptGroupParams(uint64_t singleGroups, uint64_t singleGroupOpt)
    {
        using local = typename Ext::SetOptGroupParams;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, singleGroups, singleGroupOpt)) {
            local::call(this, singleGroups, singleGroupOpt);
        }
    }

    __aicore__ inline void SetFmapStartPosition(
        int64_t diStartPos, int64_t hiStartPos, int64_t wiStartPos, int64_t ciStartPos)
    {
        using local = typename Ext::SetFmapStartPosition;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, diStartPos, hiStartPos, wiStartPos, ciStartPos)) {
            local::call(this, diStartPos, hiStartPos, wiStartPos, ciStartPos);
        }
    }

    __aicore__ inline void SetFmapStartPosition(int64_t diStartPos, int64_t mStartPos, int64_t ciStartPos)
    {
        using local = typename Ext::SetFmapStartPosition;
        if constexpr (CONV_CHECK_FUN(local, Conv3dFunc, this, diStartPos, mStartPos, ciStartPos)) {
            local::call(this, diStartPos, mStartPos, ciStartPos);
        }
    }

    __aicore__ inline void SetWeightStartPosition(int64_t coStartPos, int64_t ciStartPos = 0)
    {
        using local = typename Ext::SetWeightStartPosition;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, coStartPos, ciStartPos)) {
            local::call(this, coStartPos, ciStartPos);
        }
    }

    __aicore__ inline void SetIterIndex(uint64_t groupOptIter)
    {
        using local = typename Ext::SetIterIndex;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, groupOptIter)) {
            local::call(this, groupOptIter);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<OutputT> &output, bool enPartialSum = false)
    {
        using local = typename Ext::IterateAll;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, ConvFunc, sync, this, output, enPartialSum)) {
            local::template call<sync>(this, output, enPartialSum);
        }
    }

    __aicore__ inline void End()
    {
        using local = typename Ext::End;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this)) {
            local::call(this);
        }
    }

private:
    template <bool sync = true>
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        using local = typename Ext::Iterate;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, ConvFunc, sync, this, enPartialSum)) {
            return local::template call<sync>(this, enPartialSum);
        }
        return false;
    }

    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<OutputT> &output, bool enSequentialWrite = false)
    {
        using local = typename Ext::GetTensorC;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, ConvFunc, sync, this, output, enSequentialWrite)) {
            local::template call<sync>(this, output, enSequentialWrite);
        }
    }
};

}  // namespace conv3d

#endif // CONV3D_V2_INTF_H