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
 * \file conv2d_v2_api_impl.h
 * \brief conv2d api impl class
 */

#ifndef CONV2D_V2_API_IMPL_H
#define CONV2D_V2_API_IMPL_H

#include "../../common/arch35/conv_framework_util.h"
#include "conv2d_v2_common_func.h"
#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"

namespace conv2d {
using namespace AscendC;
using namespace conv;

template<typename Intf, class Config>
struct Conv2dApiImpl {
public:
    using ConvParam = typename Config::ConvParam;
    constexpr static uint32_t ImplType = Config::implType;

public:
    __aicore__ inline Conv2dApiImpl() {}

    CONV_REG_IMPL(Config, Conv2dFunc, Init);
    CONV_REG_IMPL(Config, ConvFunc, SetFmap);
    CONV_REG_IMPL(Config, ConvFunc, SetWeight);
    CONV_REG_IMPL(Config, ConvFunc, SetBias);
    CONV_REG_IMPL(Config, ConvFunc, SetScale);
    CONV_REG_IMPL(Config, ConvFunc, SetFixpipeParams);
    CONV_REG_IMPL(Config, Conv2dFunc, SetOrgFmapShape);
    CONV_REG_IMPL(Config, Conv2dFunc, SetOrgWeightShape);
    CONV_REG_IMPL(Config, Conv2dFunc, SetOrgOutputShape);
    CONV_REG_IMPL(Config, Conv2dFunc, SetSingleFmapShape);
    CONV_REG_IMPL(Config, Conv2dFunc, SetSingleOutputShape);
    CONV_REG_IMPL(Config, Conv2dFunc, SetOptGroupParams);
    CONV_REG_IMPL(Config, Conv2dFunc, SetFmapStartPosition);
    CONV_REG_IMPL(Config, ConvFunc, SetWeightStartPosition);
    CONV_REG_IMPL(Config, ConvFunc, SetIterIndex);
    CONV_REG_IMPL(Config, ConvFunc, Iterate);
    CONV_REG_IMPL(Config, ConvFunc, IterateAll);
    CONV_REG_IMPL(Config, ConvFunc, GetTensorC);
    CONV_REG_IMPL(Config, ConvFunc, End);

    struct ContextData: public Config::ContextData {
        __aicore__ inline ContextData(){};

        const struct TConv2DTiling* __restrict convTiling;

        // Using Conditional<flag, type1, type2>::type to select M or HW, if flag=true, type=type1, else type=type2
        using LoadAL1MModeTools = typename Conditional<Intf::isInnerBatchFlag,
            Conv2dFunc::LoadAL1ToolsInnerBatch<Intf>, Conv2dFunc::LoadAL1ToolsMMode<Intf>>::type;
        using LoadAL1Tools = typename Conditional<
            ConvParam::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE),
            LoadAL1MModeTools, Conv2dFunc::LoadAL1ToolsHWMode<Intf>>::type;
        using LoadAL0Tools = typename Conditional<
            ConvParam::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE),
            ConvFunc::LoadAL0ToolsMMode<Intf>, ConvFunc::LoadAL0ToolsHWMode<Intf>>::type;
        using CopyOutTools = typename Conditional<
            ConvParam::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE),
            ConvFunc::CopyOutToolsMMode<Intf, typename Intf::OutputT>,
            ConvFunc::CopyOutToolsHWMode<Intf, typename Intf::OutputT>>::type;
        // extend conv2d exist dual ouputs
        using CopyOutTools1 = typename Conditional<
            ConvParam::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE),
            ConvFunc::CopyOutToolsMMode<Intf, typename Intf::Output1T, 1>,
            ConvFunc::CopyOutToolsHWMode<Intf, typename Intf::Output1T, 1>>::type;
        using LoadBL1Tools = typename Conditional<
            Config::WEIGHT_NZ_FLAG, Conv2dFunc::LoadBL1FZTools<Intf>, Conv2dFunc::LoadBL1Tools<Intf>>::type;

        LoadAL1Tools loadAl1Ins;
        LoadBL1Tools loadBL1Ins;
        ConvFunc::LoadChannelWiseL1Tools<Intf, typename Config::BiasT> loadBiasL1Ins;
        ConvFunc::LoadChannelWiseL1Tools<Intf, typename Config::ScaleT> loadScaleL1Ins;
        LoadAL0Tools loadAL0Ins;
        ConvFunc::LoadBL0Tools<Intf> loadBL0Ins;
        ConvFunc::LoadBiasBtTools<Intf, typename Config::BiasT, typename Config::L0cT> loadBiasBTIns;
        ConvFunc::MMadTools<Intf> madIns;
        CopyOutTools copyOutIns;
        CopyOutTools1 copyOutIns1;

        // normal: <A1, 1>, <B1, 1>; preload: <A1, 2>, <B1, 2>
        TQue<QuePosition::A1, static_cast<int8_t>(Intf::kPreLoadAFlag || Intf::kPreLoadABFlag) + 1> queueAL1; // AL1
        TQue<QuePosition::B1, static_cast<int8_t>(Intf::kPreLoadBFlag || Intf::kPreLoadABFlag) + 1> queueBL1; // BL1

        // Used in opt group mode(groups > 1)
        ConvFunc::OptGroupLoadGM2UBTools<Intf> optGroupLoadGm2UBTools;
        ConvFunc::OptGroupTransND2NZTools<Intf> optGroupTransND2NZTools;
        ConvFunc::OptGroupLoadUB2L1Tools<Intf> optGroupLoadUB2L1Tools;

        // Used in c04 mode
        Conv2dFunc::C04DupTools<Intf> c04DupTools;
        Conv2dFunc::C04LoadGM2UBTools<Intf> c04LoadGm2UBTools;
        Conv2dFunc::C04TransFractalZC04Tools<Intf> c04TransFractalZC04Tools;
        Conv2dFunc::C04LoadUB2L1Tools<Intf> c04LoadUB2L1Tools;

        // Used in weight ub trans mode
        Conv2dFunc::WeightLoadGM2UBTools<Intf> weightUbLoadGM2UBTools;
        Conv2dFunc::WeightND2NZTools<Intf> weightUbTransND2NZTools;
        Conv2dFunc::WeightUB2L1Tools<Intf> weightUbLoadUB2L1Tools;

        // Used in dma mode
        Conv2dFunc::DmaLoadGM2UBTools<Intf> dmaLoadGM2UBTools;
        Conv2dFunc::DmaUB2L1Tools<Intf> dmaLoadUB2L1Tools;
        TBuf<TPosition::A1> aL1TBuf;
        TBuf<TPosition::VECIN> ubBuf;
        LocalTensor<typename Intf::FmapT> img2ColTensor;
 
        uint16_t aL1LoadTimes = 0;
        uint16_t vecCi1Iter = 0;
        uint16_t maxVecCi1Iter = 0;
        uint16_t vecKhIter = 0;
        uint16_t maxVecKhIter = 0;
        uint16_t vecKhLoopTimes = 0;
        uint16_t vecKwIter = 0;
        uint16_t maxVecKwIter = 0;
        uint16_t vecKwLoopTimes = 0;
        uint16_t ddr2l1LoopKA = 0;
 
        uint32_t ciTail = 0;
        uint32_t currentHoL1xWoL1Align = 0;
        uint32_t currentCi1Ub = 0;
        uint32_t currentVec0Ci = 0;
        uint32_t vec0TotalSize = 0;

        uint16_t kwAL1Iter = 0;
        uint16_t khAL1Iter = 0;
        uint16_t kwBL1Iter = 0;
        uint16_t khBL1Iter = 0;
        uint16_t maxKwAL1Iter = 0;
        uint16_t maxKhAL1Iter = 0;
        uint16_t ddr2L1LoopKw = 0;
        uint16_t ddr2L1LoopKh = 0;
        uint16_t cinAL1 = 0;
        uint16_t cinAL1LoopTimes = 0;
        uint16_t cinAL1Iter = 0;
        uint16_t maxCinAL1Iter = 0;
        uint16_t cinBL1 = 0;
        uint16_t cinBL1LoopTimes = 0;
        uint16_t cinBL1Iter = 0;
        uint16_t maxCinBL1Iter = 0;

        LocalTensor<typename Intf::WeightT> ndTensor;

        uint32_t kSizeC04 = 0;
        uint32_t bUbNTailStep = 0;
        uint32_t currentUbNStep = 0;
        uint32_t currentUbNStepAilgn = 0;
        uint32_t currentNLoopRpSize = 0;
        uint32_t currentUbKStep = 0;
        uint32_t currentUbKStepAilgn = 0;
        uint32_t currentKLoopRpSize = 0;
        uint32_t nBL1Vec0 = 0;

        uint16_t vecNIter = 0;
        uint16_t maxVecNIter = 0;
        uint16_t vecNLoopTimes = 0;
        uint16_t vecKIter = 0;
        uint16_t maxVecKIter = 0;
        uint16_t vecKLoopTimes = 0;

        uint32_t nUbTail = 0;
        uint32_t kUbTail = 0;

        LocalTensor<typename Intf::WeightT> bL1Tensor;
        LocalTensor<typename Intf::FmapT> aL1Tensor;
    };

    struct ImplDataType {
        __aicore__ inline ImplDataType() {};
    };
};
}  // namespace conv2d

#endif // CONV2D_V2_API_IMPL_H