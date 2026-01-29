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
 * \file conv_config.h
 * \brief
 */

#ifndef CONV_CONFIG_H
#define CONV_CONFIG_H

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "conv_util.h"

namespace conv {
using namespace AscendC;

enum class ConvCfgTypeID : std::uint8_t {
    CONV_ID_Unknown = 0,
    CONV_ID_Normal,
    CONV_ID_Test,
    CONV_ID_END
};

enum class ConvFormat : std::uint8_t {
    ND = 0,
    NCHW,
    NHWC,
    HWCN,
    NC1HWC0,
    FRACTAL_Z,
    FRACTAL_Z_C04,
    NDC1HWC0,
    FRACTAL_Z_3D,
    NCDHW,
    NDHWC,
    DHWCN
};

template <typename ScaleT, typename ReluWeightT, typename ClipValue0T, typename ClipValue1T>
struct Extendconv2dFixpipeParams {
    __aicore__ inline Extendconv2dFixpipeParams(){};
    GlobalTensor<ScaleT> scale0;
    GlobalTensor<ReluWeightT> reluWeight0;
    GlobalTensor<ClipValue0T> clipValue0;
    GlobalTensor<ScaleT> scale1;
    GlobalTensor<ReluWeightT> reluWeight1;
    GlobalTensor<ClipValue1T> clipValue1;
};

struct CopyUbInfo {
    // only support batchUb = 1 in tiling hw mode
    // when batchUb > 1, mUb should equal to convTiling->hoL0
    uint64_t batchUb = 0;
    // nUb should be aligned to BLOCK_L0_N
    uint64_t nUb = 0;
    // mUb should be aligned to BLOCK_L0_M
    // In tiling m mode, mUb is part of convTiling->hoL0
    // In tiling hw mode, mUb is part of convTiling->hoL0*convTiling->woL0,
    // additionally, when convTiling->hoL0 != 1, mUb must be >= convTiling->woL0.
    uint64_t mUb = 0;
    uint64_t batchLoopIdx = 0;
    uint64_t nLoopIdx = 0;
    uint64_t mLoopIdx = 0;
    uint64_t realBatchUb = 0;
    uint64_t realNUb = 0;
    uint64_t realHUb = 0;
    uint64_t realWUb = 0;
    uint64_t outBatchIdx = 0;
    uint64_t outCIdx = 0;
    uint64_t outHIdx = 0;
    uint64_t outWIdx = 0;
};

enum class ConvFmapTiling : std::uint8_t {
    FULLLOAD_AL1 = 0,
    ONLY_M_FULLLOAD_AL1_AL0,
    OTHER
};

enum class ConvWeightTiling : std::uint8_t {
    FULLLOAD_BL1 = 0,
    ONLY_N_FULLLOAD_BL1_BL0,
    OTHER
};

enum class ConvL1PingPong : std::uint8_t {
    ALL_CLOSE = 0,
    AL1_OPEN,
    BL1_OPEN,
    ALL_OPEN
};

enum class ConvL0PingPong : std::uint8_t {
    ALL_CLOSE = 0,
    AL0_OPEN,
    BL0_OPEN,
    ALL_OPEN
};

enum class ConvOutputOrder : std::uint8_t {
    HW_MODE = 0,
    M_MODE
};

enum class ConvIterOrder : std::uint8_t {
    MITER_FIRST = 0,
    NITER_FIRST
};

enum class ConvGroupType : std::uint8_t {
    NORMAL_CONV = 0,
    ORI_GROUP_CONV,
    OPT_GROUP_CONV
};

enum class ConvEnableSmallChannel : std::uint8_t {
    CLOSE = 0,
    OPEN
};

enum class ConvWeightUbTrans : std::uint8_t {
    CLOSE = 0,
    OPEN
};

enum class ConvFmapCopyMode : std::uint8_t {
    LOAD3D = 0,
    DMA
};

enum class ConvInnerBatch : std::uint8_t {
    SINGLE_BATCH = 0,
    KERNEL_1X1_MULTI_BATCH,
    MULTI_BATCH
};

enum class IsExtendConv2D : std::uint8_t {
    FALSE = 0,
    TRUE
};

enum class ConvDisContinuous : std::uint8_t {
    CLOSE = 0,
    INPUT_HWNC
};

struct ConvParam {
    __aicore__ inline ConvParam(){};
    constexpr static int8_t fmapTiling = 0;
    constexpr static int8_t weightTiling = 0;
    constexpr static int8_t l1PingPong = 0;
    constexpr static int8_t l0PingPong = 0;
    constexpr static int8_t outputOrder = static_cast<int8_t>(ConvOutputOrder::M_MODE);
    constexpr static int8_t iterOrder = 0;
    constexpr static int8_t groupType = 0;
    constexpr static int8_t innerBatch = static_cast<int8_t>(ConvInnerBatch::SINGLE_BATCH);
};

template <typename T>
struct GetDstType {
    using Type = T;
};

template <>
struct GetDstType<float> {
    using Type = float;
};

template <>
struct GetDstType<half> {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    using Type = int32_t;
#else
    using Type = float;
#endif
};

template <>
struct GetDstType<bfloat16_t> {
    using Type = float;
};

template <>
struct GetDstType<int8_t> {
    using Type = int32_t;
};

template <>
struct GetDstType<hifloat8_t> {
    using Type = float;
};

template <>
struct GetDstType<fp8_e4m3fn_t> {
    using Type = float;
};

template <TPosition POSITION, ConvFormat FORMAT, typename TYPE>
struct ConvType {
    constexpr static TPosition pos = POSITION;
    constexpr static ConvFormat format = FORMAT;
    using T = TYPE;
};

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
struct ConvDataType {
    using ConvParam = CONV_CFG;
    using FmapT = typename FMAP_TYPE::T;
    using WeightT = typename WEIGHT_TYPE::T;
    using OutputT  = typename OUTPUT_TYPE::T;
    using Output1T = typename CONV_CFG::Output1Dtype;
    using BiasT = typename BIAS_TYPE::T;
    using ScaleT = typename SCALE_TYPE::T;
    using ReluWeightT = float;
    using ClipValue0T = typename OUTPUT_TYPE::T;
    using ClipValue1T = typename CONV_CFG::Output1Dtype;
    using L0cT = typename GetDstType<FmapT>::Type;

    constexpr static uint32_t configID = (uint32_t)ConvCfgTypeID::CONV_ID_Normal;
    constexpr static uint32_t implType = (uint32_t)ConvCfgTypeID::CONV_ID_Normal;
    constexpr static uint32_t intfType = (uint32_t)ConvCfgTypeID::CONV_ID_Normal;

    constexpr static auto formatFmap = FMAP_TYPE::format;
    constexpr static auto formatWeight = WEIGHT_TYPE::format;
    constexpr static auto formatOutput = OUTPUT_TYPE::format;
    constexpr static auto formatBias = BIAS_TYPE::format;
    constexpr static bool WEIGHT_NZ_FLAG = WEIGHT_TYPE::format == ConvFormat::FRACTAL_Z ||
        WEIGHT_TYPE::format == ConvFormat::FRACTAL_Z_C04;

    constexpr static auto posFmap = FMAP_TYPE::pos;
    constexpr static auto posWeight = WEIGHT_TYPE::pos;
    constexpr static auto posOutput = OUTPUT_TYPE::pos;
    constexpr static auto posBias = BIAS_TYPE::pos;

    using ContextData = struct _ {
        __aicore__ inline _() {}
    };
};

template <class ConvDataType>
struct ConvConfig : public ConvDataType {
public:
    __aicore__ inline ConvConfig() {}

    using ContextData = struct _ : public ConvDataType::ContextData {
        __aicore__ inline _() {}

        TPipe pipe;

        // GM Tensor
        GlobalTensor<typename ConvDataType::FmapT> agm;
        GlobalTensor<typename ConvDataType::WeightT> bgm;
        GlobalTensor<typename ConvDataType::BiasT> biasgm;
        GlobalTensor<typename ConvDataType::ScaleT> scalegm;
        GlobalTensor<typename ConvDataType::ScaleT> scale1gm;
        TQue<QuePosition::A1, 1> queueBiasL1; // BiasL1
        TQue<QuePosition::A1, 1> queueScaleL1; // ScaleL1
        TQue<TPosition::C2, 1> queueBiasBT; // BT
        TQue<QuePosition::CO1, 1> queueCL0;
        TBuf<TPosition::A2> al0Buf;
        TBuf<TPosition::B2> bl0Buf;
        TBuf<TPosition::CO1> l0cBuf;

        LocalTensor<typename ConvDataType::FmapT> al1;
        LocalTensor<typename ConvDataType::WeightT> bl1;
        LocalTensor<typename ConvDataType::BiasT> biasL1;
        LocalTensor<typename ConvDataType::ScaleT> scaleL1;
        LocalTensor<typename ConvDataType::FmapT> wholeAl0Tensor;
        LocalTensor<typename ConvDataType::FmapT> al0;
        LocalTensor<typename ConvDataType::WeightT> wholeBl0Tensor;
        LocalTensor<typename ConvDataType::WeightT> bl0;
        LocalTensor<typename ConvDataType::L0cT> biasBT;
        LocalTensor<typename ConvDataType::L0cT> wholeCl0Tensor =
            LocalTensor<typename ConvDataType::L0cT>(TPosition::CO1, 0, 0);
        LocalTensor<typename ConvDataType::L0cT> cl0 =
            LocalTensor<typename ConvDataType::L0cT>(TPosition::CO1, 0, 0);

        uint8_t enableBias = false;  // 是否有bias
        uint8_t enableVectorQuant = false;  // 是否有vector类型scale，双输出场景下任一scale为vector类型，即为true
        uint64_t scale1L1offset = 0; // 双输出时scale1在L1上的偏移
        uint64_t deqScalar0 = DEQ_SCALAR_ONE;  // m1 = 1.0, other bits are 0.
        uint64_t deqScalar1 = DEQ_SCALAR_ONE;  // m1 = 1.0, other bits are 0.
        uint64_t preReluScalar0 = 0;
        uint64_t preReluScalar1 = 0;

        uint64_t fmapOneBatchSize = 0;
        uint64_t outputOneBatchSize = 0;
        uint64_t dilatedKernelH = 0;
        uint64_t dilatedKernelW = 0;

        // Used in m_mode
        uint64_t mStartPos = 0;
        uint64_t singleCoreM = 0;
        uint64_t mAL1Iter = 0;
        uint64_t mL0Iter = 0;
        uint64_t maxMAL1Iter = 0;
        uint64_t maxML0Iter = 0;
        uint64_t ddr2l1LoopM = 0;
        uint64_t l12l0LoopM = 0;
        uint64_t mAL1 = 0;
        uint64_t mAL1Tail = 0;
        uint64_t mL0 = 0;
        uint64_t mAL0Tail = 0;
        uint8_t mAL1UpdateFlag = true;

        // Used in hw_mode
        int64_t hiStartPos = -1;
        int64_t wiStartPos = 0;
        uint64_t singleCoreHo = 0; // 单核上处理的Ho大小
        uint64_t singleCoreWo = 0; // 单核上处理的Wo大小
        uint64_t hoAL1Iter = 0;  // AL1上ho方向迭代器
        uint64_t woAL1Iter = 0;  // AL1上wo方向迭代器
        uint64_t hoL0Iter = 0; // L0A上ho方向迭代器
        uint64_t woL0Iter = 0; // L0A上wo方向迭代器
        uint64_t maxHoL1Iter = 0;
        uint64_t maxWoL1Iter = 0;
        uint64_t maxHoL0Iter = 0;
        uint64_t maxWoL0Iter = 0;

        uint64_t ddr2l1LoopH = 0;
        uint64_t ddr2l1LoopW = 0;
        uint64_t l12l0LoopH = 0;
        uint64_t l12l0LoopW = 0;

        uint64_t hoAL1Tail = 0; // AL1上ho方向的尾块大小
        uint64_t woAL1Tail = 0; // AL1上wo方向的尾块大小
        uint64_t woL1SmallTail = 0; // wo方向16不对齐场景会存在小尾块
        uint64_t hoL0Tail = 0;
        uint64_t woL0Tail = 0;

        uint64_t currentHoL1 = 0;
        uint64_t currentWoL1 = 0;
        uint64_t currentHoL0 = 0;
        uint64_t currentWoL0 = 0;

        // Used in opt group mode(groups > 1) and c04 mode
        TBuf<TPosition::VECIN> ndUbBuf;
        TBuf<TPosition::VECOUT> nzUbBuf;
        TBuf<TPosition::VECCALC> indexUbBuf;
        TBuf<TPosition::B1> bL1TBuf;

        LocalTensor<typename ConvDataType::WeightT> nzTensor;

        uint16_t vecId = 0;              // V侧: vec id(0/1) C侧: 本次groupOpt对应的vec同步id(0/16)
        uint32_t ubBufSize = 0;          // vec一次处理的矩阵大小
        uint32_t bL1SpaceSize = 0;       // Weight载入L1的大小

        uint8_t pingPongFlag = 0;

        // Used in opt group mode(groups > 1)
        uint16_t ciPerGroup = 0;
        uint16_t coPerGroup = 0;
        uint16_t ciOpt = 0;
        uint16_t ci1Opt = 0;
        uint16_t ciOptAlign = 0;
        uint16_t kUbSize = 0;
        uint16_t coOpt = 0;
        uint16_t co1Opt = 0;
        uint16_t coOptAlign = 0;
        uint16_t singleGroups = 0;       // 单核上处理的groups大小
        uint16_t singleGroupOpt = 0;     // 单核上处理的groupOpt大小
        uint16_t groupOptIter = 0;
        uint16_t ddr2l1LoopKB = 0;
        uint16_t loadUB2L1Iter = 0;
        uint16_t ddr2l1LoopInner = 0;
        uint16_t ddr2l1LoopTmp = 0;
        uint16_t innerIter = 0;
        uint16_t ddr2l1LoopOuter = 0;
        uint16_t outerIter = 0;
        uint16_t bL1LoadTimes = 0;

        int64_t ciStartPos = 0;
        int64_t coStartPos = 0;
        uint64_t singleCoreCi = 0; // 单核上处理的Cin大小
        uint64_t singleCoreCo = 0; // 单核上处理的Co大小

        // Iterate Var
        uint8_t isFirstIterate = true; // 是否第一次Iterate
        uint8_t kAL1fullload = false;
        uint8_t kBL1fullload = false;
        uint8_t loadAL1Flag = true; // 是否载入AL1的标志
        uint8_t loadBL1Flag = true; // 是否载入BL1的标志
        uint8_t loadAL0Flag = true; // 是否载入AL0的标志
        uint8_t loadBL0Flag = true; // 是否载入BL0的标志

        uint64_t kIter = 0; // k方向迭代器，从DDR到L0
        uint64_t batchIter = 0; // batch方向迭代器
        uint64_t kAL1Iter = 0;    // AL1上k方向迭代器
        uint64_t kBL1Iter = 0;    // BL1上k方向迭代器
        uint64_t nBL1Iter = 0;   // BL1上n方向迭代器
        uint64_t kAL0Iter = 0; // L1A 到L0方向的迭代器
        uint64_t kBL0Iter = 0; // L1B 到L0方向的迭代器
        uint64_t nL0Iter = 0; // BL0上n方向迭代器

        uint64_t maxBatchIter = 0;
        uint64_t multiKAL1 = 1;
        uint64_t multiKBL1 = 1;
        uint64_t maxKAL1Iter = 0;
        uint64_t maxKBL1Iter = 0;
        uint64_t maxNBL1Iter = 0;
        uint64_t maxKL0Iter = 0;
        uint64_t maxNL0Iter = 0;

        uint64_t ddr2l1LoopBatch = 0;
        uint64_t ddr2l1LoopN = 0;
        uint64_t ddr2l0LoopK = 0;
        uint64_t l12l0LoopN = 0;

        uint64_t singleCoreBatch = 0;
        uint64_t innerBatch = 0;
        uint64_t innerBatchTail = 0;
        uint64_t kAL1Tail = 0;  // AL1上k方向的尾块大小
        uint64_t kAL1AlignK0Tail = 0; // AL1对齐k0后上k方向的尾块大小
        uint64_t kBL1Tail = 0; // BL1上k方向的尾块大小
        uint64_t kBL1AlignK0Tail = 0; // BL1上对齐k0后上k方向的尾块大小
        uint64_t nBL1Tail = 0; // BL1上N方向的尾块大小
        uint64_t kL0Tail = 0;
        uint64_t nL0Tail = 0; // BL0上n方向的对齐后尾块大小
        uint64_t kAL0Tail = 0; // AL0上k方向的尾块大小

        uint64_t currentNBL1 = 0;
        uint64_t currentNBL1Align = 0;
        uint64_t currentNL0Align = 0;
        uint64_t currentML0Align = 0;

        uint64_t lastLoopKAL1StartPos = 0;
        uint64_t lastLoopKBL1StartPos = 0;
        uint64_t lastLoopKAL1StartPosTail = 0;
        uint64_t lastLoopKBL1StartPosTail = 0;
        uint16_t kL0FullLoadAl0PingPongFlag = 0;
        uint16_t kL0FullLoadBl0PingPongFlag = 0;
        uint16_t cl0PingPongFlag = 0;

        // Input shape info
        uint64_t orgBatch = 0;  // fmap上batch大小
        uint64_t orgCi = 0;     // fmap上cin大小
        uint64_t orgCo = 0;     // weight上cout大小
        uint64_t orgHi = 0;     // fmap上h大小
        uint64_t orgWi = 0;     // fmap上w大小
        uint64_t orgHo = 0;     // output上h大小
        uint64_t orgWo = 0;     // output上w大小
        uint64_t kernelH = 0;   // weight上h大小
        uint64_t kernelW = 0;   // weight上w大小
    };
};

}  // namespace conv
#endif