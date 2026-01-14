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
 * \file conv3d_v2_api_impl.h
 * \brief
 */

#ifndef CONV3D_V2_API_IMPL_H
#define CONV3D_V2_API_IMPL_H

#include "../../common/arch35/conv_framework_util.h"
#include "conv3d_v2_common_func.h"
#include "conv3d_v2_config.h"
#include "conv3d_v2_util.h"
#include "conv3d_v2_instr_impl.h"
#include "../../common/arch35/conv_instr_m_mode_impl.h"
#include "../../common/arch35/conv_instr_hw_mode_impl.h"
#include "../../common/arch35/conv_instr_impl.h"
#include "../../common/arch35/conv_instr_opt_group_impl.h"

namespace conv3d {
using namespace AscendC;
using namespace conv;

template <typename Intf, class Config>
struct Conv3dApiImpl {
public:
    using ConvParam = typename Config::ConvParam;
    constexpr static uint32_t ImplType = Config::implType;

public:
    __aicore__ inline Conv3dApiImpl(){};

    CONV_REG_IMPL(Config, Conv3dFunc, Init);
    CONV_REG_IMPL(Config, ConvFunc, SetFmap);
    CONV_REG_IMPL(Config, ConvFunc, SetWeight);
    CONV_REG_IMPL(Config, ConvFunc, SetBias);
    CONV_REG_IMPL(Config, ConvFunc, SetScale);
    CONV_REG_IMPL(Config, Conv3dFunc, SetOrgFmapShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetOrgWeightShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetOrgOutputShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetSingleFmapShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetSingleOutputShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetOptGroupParams);
    CONV_REG_IMPL(Config, Conv3dFunc, SetFmapStartPosition);
    CONV_REG_IMPL(Config, ConvFunc, SetWeightStartPosition);
    CONV_REG_IMPL(Config, ConvFunc, SetIterIndex);
    CONV_REG_IMPL(Config, ConvFunc, Iterate);
    CONV_REG_IMPL(Config, ConvFunc, ConvPreProcess);
    CONV_REG_IMPL(Config, ConvFunc, ConvPostProcess);
    CONV_REG_IMPL(Config, ConvFunc, IterateAll);
    CONV_REG_IMPL(Config, ConvFunc, GetTensorC);
    CONV_REG_IMPL(Config, ConvFunc, End);

    struct ContextData : public Config::ContextData {
        __aicore__ inline ContextData(){};

        const struct Ops::NN::Conv3dV2::TConv3DTiling *__restrict convTiling;

        // Using Conditional<flag, type1, type2>::type to select M or HW, if flag=true, type=type1, else type=type2
        using LoadAL1Tools = typename Conditional<
            ConvParam::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE),
            Conv3dFunc::LoadAL1ToolsMMode<Intf>, Conv3dFunc::LoadAL1ToolsHWmode<Intf>>::type;
        using LoadAL0Tools = typename Conditional<
            ConvParam::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE),
            ConvFunc::LoadAL0ToolsMMode<Intf>, ConvFunc::LoadAL0ToolsHWMode<Intf>>::type;
        using CopyOutTools = typename Conditional<
            ConvParam::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE),
            ConvFunc::CopyOutToolsMMode<Intf, typename Intf::OutputT>,
            ConvFunc::CopyOutToolsHWMode<Intf, typename Intf::OutputT>>::type;

        LoadAL1Tools loadAl1Ins;
        Conv3dFunc::LoadBL1Tools<Intf> loadBL1Ins;
        ConvFunc::LoadChannelWiseL1Tools<Intf, typename Config::BiasT> loadBiasL1Ins;
        ConvFunc::LoadChannelWiseL1Tools<Intf, typename Config::ScaleT> loadScaleL1Ins;
        LoadAL0Tools loadAL0Ins;
        ConvFunc::LoadBL0Tools<Intf> loadBL0Ins;
        ConvFunc::LoadBiasBtTools<Intf, typename Config::BiasT, typename Config::L0cT> loadBiasBTIns;
        ConvFunc::MMadTools<Intf> madIns;
        CopyOutTools copyOutIns;

        // Used in opt group mode(groups > 1)
        ConvFunc::OptGroupLoadGM2UBTools<Intf> optGroupLoadGm2UBTools;
        ConvFunc::OptGroupTransND2NZTools<Intf> optGroupTransND2NZTools;
        ConvFunc::OptGroupLoadUB2L1Tools<Intf> optGroupLoadUB2L1Tools;

        // normal: <A1, 1>, <B1, 1>; preload: <A1, 2>, <B1, 2>
        TQue<QuePosition::A1, static_cast<int8_t>(Intf::kPreLoadAFlag || Intf::kPreLoadABFlag) + 1> queueAL1; // AL1
        TQue<QuePosition::B1, static_cast<int8_t>(Intf::kPreLoadBFlag || Intf::kPreLoadABFlag) + 1> queueBL1; // BL1

        uint64_t orgDi = 0;
        uint64_t orgDo = 0;
        uint64_t kernelD = 0;
        uint64_t dilatedKernelD = 0;

        int64_t diStartPos = 0;
        uint64_t singleCoreDo = 0;

        uint64_t dOutIter = 0;
        uint64_t maxDOutIter = 0;
        uint64_t ddr2l1LoopD = 0;

        uint64_t cin1xcin0 = 0;
        uint64_t aL1Dk = 0;
        uint64_t aL1DkTail = 0;
        uint64_t aL1Cin = 0;
        uint64_t aL1CinTail = 0;
        uint64_t aL1CinLoadNum = 0;
        uint64_t bL1Dk = 0;
        uint64_t bL1DkTail = 0;
        uint64_t bL1Cin = 0;
        uint64_t bL1CinTail = 0;
        uint64_t bL1CinLoadNum = 0;

        Conv3dFunc::DeQuantL0C2UBTools<Intf> dequantL0C2UBTools;
        Conv3dFunc::DeQuantLoadParamsTools<Intf> dequantLoadParamsTools;
        Conv3dFunc::DeQuantCalcTools<Intf> dequantCalcTools;
        Conv3dFunc::DeQuantUB2GMTools<Intf> dequantUB2GmTools;

        TBuf<TPosition::VECIN> scaleUbBuf;
        TBuf<TPosition::VECIN> biasB16UbBuf;
        TBuf<TPosition::VECIN> biasB32UbBuf;
        TBuf<TPosition::VECIN> mmadResUbBuf;

        LocalTensor<typename Intf::ScaleT> scaleTensor;
        LocalTensor<typename Intf::BiasT> biasB16Tensor;
        LocalTensor<float> biasB32Tensor;
        LocalTensor<typename Intf::L0cT> mmadResUbTensor;
        LocalTensor<typename Intf::OutputT> outputResUbTensor;

        uint32_t mUbIter = 0;
        uint32_t maxMUbIter = 0;
        uint32_t l0C2UbLoopM = 0;
        uint32_t l0C2UbLoopWo = 0;
        uint32_t currentMUb = 0;
        uint32_t currentNUb = 0;
        uint32_t fixpTimes = 0;
        uint32_t fixpTimesTotal = 0;
    };

    struct ImplDataType {
        __aicore__ inline ImplDataType(){};
    };
};

}  // namespace conv3d

#endif // CONV3D_V2_API_IMPL_H