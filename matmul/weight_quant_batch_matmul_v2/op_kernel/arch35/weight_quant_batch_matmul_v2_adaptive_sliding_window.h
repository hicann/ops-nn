/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window.h
 * \brief ASW kernel（内联手写指令版mmad）。
 *        数据流：GM -> L1(Nd2Nz) -> L0(LoadData2D) -> Mmad -> L0C -> Fixpipe(随路反量化) -> GM。
 *        仅覆盖 ASW tiling 的使用场景：stepM/stepN = 1，单核处理 baseM x baseN 的子块，K 方向循环累加。
 */

#pragma once

#include "weight_quant_batch_matmul_v2_asw_block.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_v2_kernel_util.h"
#include "../tool.h"

using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
namespace WeightQuantBatchMatmulV2::Arch35 {

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                        \
    template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans, \
              QuantType antiQuantType, bool hasAntiQuantOffset, QuantType quantType>
#define LOCAL_TEMPLATE_FUNC_PARAMS \
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType

LOCAL_TEMPLATE_CLASS_PARAMS
class WeightQuantBatchMatmulV2ASWKernel {
    constexpr static uint64_t C0_SIZE_A = AscendC::AuxGetC0Size<xType>();
    constexpr static uint64_t C0_SIZE_W = AscendC::AuxGetC0Size<wType>();
    constexpr static uint64_t C0_SIZE_BIAS = AscendC::AuxGetC0Size<biasType>();
    constexpr static uint64_t BLOCK_CUBE_SIZE = AscendC::BLOCK_CUBE;
    // L0A/L0B开启双缓冲，L0C按tiling l0cDB使能乒乓双缓冲
    constexpr static uint64_t HALF_L0A_ELEMS = 64 * 1024 / 2 / sizeof(xType);
    constexpr static uint64_t HALF_L0B_ELEMS = 64 * 1024 / 2 / sizeof(wType);
    // L0C乒乓时单个半区的元素数（int32累加）
    constexpr static uint64_t HALF_L0C_ELEMS = AscendC::TOTAL_L0C_SIZE / 2 / sizeof(int32_t);
    // 硬事件flag id
    constexpr static uint16_t L1_MTE1_MTE2_FLAG = 0;   // 0~3: A/B的L1多buffer
    constexpr static uint16_t L1_MTE2_MTE1_FLAG = 0;   // 0~3: A/B的L1多buffer
    constexpr static uint16_t BIAS_MTE1_MTE2_FLAG = 4; // 避让L1 buffer的0~3
    constexpr static uint16_t L0_M_MTE1_FLAG = 3;      // 2/3: L0双缓冲
    constexpr static uint16_t L0_MTE1_M_FLAG = 2;      // 2/3: L0双缓冲
    constexpr static uint16_t L0C_FIX_M_FLAG = 0;      // 0/1: L0C乒乓双缓冲
    constexpr static uint16_t L0C_M_FIX_FLAG = 0;      // 0/1: L0C乒乓双缓冲
    constexpr static uint16_t SCALE_FIX_MTE2_FLAG = 0; // 0~3: scale L1缓冲，级数同l1BufferNum
    constexpr static uint16_t SCALE_MTE2_FIX_FLAG = 0; // 0~3: scale L1缓冲，级数同l1BufferNum
    // bias BT(C2)单缓冲复用，等上一tile读取bias的Mmad完成后再覆写，避让L0的3/4号flag
    constexpr static uint16_t BIAS_BT_M_MTE1_FLAG = 5;
    // 多buffer flag序号（0~3级buffer对应的偏移）
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SECOND_FLAG = 2;
    constexpr static uint16_t THIRD_FLAG = 3;
    // unitflag状态：3=最后一次累加，2=非最后一次累加，硬件自动完成M与FIX的同步
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
#if __NPU_ARCH__ == 5102
    constexpr static uint8_t FIX_SHIFT_VAL_LEN_A16W16 = 58;
    // 本kernel固定为A16W8路径（x为fp16、weight为int8），bias搬运定点位宽取29
    constexpr static uint8_t FIX_SHIFT_VAL_LEN_A16W8 = 29;
#endif

public:
    __aicore__ inline WeightQuantBatchMatmulV2ASWKernel()
    {
        // 预设"资源空闲"flag，使首个WaitFlag直接通过；L1最多4buffer，flag 0~3全部预设
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + SECOND_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + THIRD_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + 1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + 1);
        // scale flag 0~3全部预设，l1BufNum为2时多余的2/3号由析构配平
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + SECOND_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + THIRD_FLAG);
    }

    __aicore__ inline ~WeightQuantBatchMatmulV2ASWKernel()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + SECOND_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + THIRD_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + 1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + 1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + SECOND_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + THIRD_FLAG);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void SetL2CacheHint();

protected:
    __aicore__ inline void ComputeBasicOptiLoop();
    __aicore__ inline void ComputeBmmOptiLoop();
    __aicore__ inline void SetMMParaAndCompute();

private:
    __aicore__ inline void InitMmad();
    __aicore__ inline void SetSingleShape(int32_t singleM, int32_t singleN, int32_t singleK);
    __aicore__ inline void CopyInA1(uint64_t kGmOffset, uint64_t curK, uint64_t l1BufId);
    __aicore__ inline void CopyInB1(uint64_t kGmOffset, uint64_t curK, uint64_t l1BufId);
    __aicore__ inline void CopyInBiasL1();
    __aicore__ inline void CopyInA2(uint64_t kL1Offset, uint64_t curK, uint64_t curK0, uint64_t l0BufId);
    __aicore__ inline void CopyInB2(uint64_t kL1Offset, uint64_t curK, uint64_t curK0, uint64_t l0BufId);
    __aicore__ inline void CopyInBiasBt();
    __aicore__ inline void MmadCompute(uint64_t curK0, uint64_t l0BufId, uint64_t l0cBufId, bool needBias,
                                       bool needCInit, bool isFinalAccum);
    __aicore__ inline void CopyScaleL1(uint64_t scaleBufId);
    __aicore__ inline void FixpipeOut(const AscendC::GlobalTensor<yType>& gm, uint64_t l0cBufId, uint64_t scaleBufId);
    __aicore__ inline void Iterate();
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const AscendC::GlobalTensor<yType>& gm);

    uint32_t blockIdx_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWCustomTilingDataParams* tiling_ = nullptr;
    WeightQuantBmmAswBlock block_;
    // GM基址
    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<uint64_t> scaleGlobal_;
    // 当前tile偏移后的GM地址
    GlobalTensor<xType> aTileGlobal_;
    GlobalTensor<wType> bTileGlobal_;
    GlobalTensor<biasType> biasTileGlobal_;
    GlobalTensor<uint64_t> scaleTileGlobal_;

    uint64_t singleM_ = 1;
    uint64_t singleN_ = 1;
    uint64_t singleK_ = 1;
    uint64_t baseM_ = 1;
    uint64_t baseN_ = 1;
    uint64_t baseK_ = 1;
    uint64_t kL1_ = 1;           // A/B在L1上单buffer的K方向长度
    uint64_t l1BufNum_ = 2;      // L1 buffer数量（2或4），来自tiling l1BufferNum
    uint64_t aL1BufElems_ = 0;   // A单个L1 buffer的元素数
    uint64_t bL1BufElems_ = 0;   // B单个L1 buffer的元素数
    uint64_t bL1ByteOffset_ = 0; // B在L1上的字节偏移
    uint64_t scaleL1ByteOffset_ = 0;
    uint64_t biasL1ByteOffset_ = 0;
    uint64_t l1LoopCnt_ = 0;
    uint64_t l0LoopCnt_ = 0;
    uint64_t l0cPingPong_ = 0;  // L0C乒乓计数，每完成一个tile的搬出自增
    uint64_t scaleLoopCnt_ = 0; // scale L1 buffer轮转计数，每完成一个tile的搬出自增
    bool isFirstTile_ = true;   // 当前核首个tile标记，用于首轮K减半流水预热
    bool enableL0cDb_ = false;  // L0C双缓冲使能，来自tiling l0cDB
    uint64_t quantScalar_ = 0;
    bool hasBias_ = false;
#if __NPU_ARCH__ == 5102
    uint8_t shiftValue_ = 42;
#endif

    AscendC::LocalTensor<xType> aL1_;
    AscendC::LocalTensor<wType> bL1_;
    AscendC::LocalTensor<uint64_t> scaleL1_;
    AscendC::LocalTensor<biasType> biasL1_;
    AscendC::LocalTensor<xType> aL0_;
    AscendC::LocalTensor<wType> bL0_;
    AscendC::LocalTensor<int32_t> cL0_;
    AscendC::LocalTensor<biasType> biasBt_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData)
{
    if ASCEND_IS_AIV {
        return;
    }
    tiling_ = static_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWCustomTilingDataParams*>(tilingData);
    blockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    InitMmad();
}

// mmad 的 buffer 布局与 tiling 参数初始化
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitMmad()
{
    baseM_ = tiling_->baseM;
    baseN_ = tiling_->baseN;
    baseK_ = tiling_->baseK;
    singleK_ = tiling_->k;
    // ASW tiling保证stepKa/stepKb互为整数倍，取较小者作为A/B统一的L1 K向切分，保证两侧K进度一致
    uint64_t stepKa = static_cast<uint64_t>(tiling_->stepKa);
    uint64_t stepKb = static_cast<uint64_t>(tiling_->stepKb);
    kL1_ = (stepKa < stepKb ? stepKa : stepKb) * baseK_;
    aL1BufElems_ = baseM_ * kL1_;
    // B 为 NZ 格式，C0 内轴需按 C0 对齐占用 L1：transB 时 K 为内轴，否则 N 为内轴
    bL1BufElems_ = bTrans ? baseN_ * CeilAlign(kL1_, C0_SIZE_W) : CeilAlign(baseN_, C0_SIZE_W) * kL1_;

    // L1布局：A多buffer | B多buffer | scale多buffer(仅per-channel，级数同l1BufferNum) | bias(仅有bias，单buffer)
    // l1BufferNum仅支持2/4（2的幂，& (l1BufNum_-1)取模），非法值回退2
    l1BufNum_ = tiling_->l1BufferNum == QUADRUPLE_BUFFER_NUM ? QUADRUPLE_BUFFER_NUM : DOUBLE_BUFFER_NUM;
    uint64_t aL1Bytes = l1BufNum_ * aL1BufElems_ * sizeof(xType);
    uint64_t bL1Bytes = l1BufNum_ * bL1BufElems_ * sizeof(wType);
    bL1ByteOffset_ = aL1Bytes;
    scaleL1ByteOffset_ = aL1Bytes + bL1Bytes;
    // scale单buffer为baseN_个uint64_t，总区域随l1BufferNum翻倍
    uint64_t scaleL1Bytes = (antiQuantType == QuantType::PER_TENSOR) ?
                                0 :
                                l1BufNum_ * static_cast<uint64_t>(baseN_) * sizeof(uint64_t);
    biasL1ByteOffset_ = scaleL1ByteOffset_ + scaleL1Bytes;

    aL1_ = AscendC::LocalTensor<xType>(AscendC::TPosition::A1, 0, aL1Bytes);
    bL1_ = AscendC::LocalTensor<wType>(AscendC::TPosition::A1, bL1ByteOffset_, bL1Bytes);
    if constexpr (antiQuantType != QuantType::PER_TENSOR) {
        scaleL1_ = AscendC::LocalTensor<uint64_t>(AscendC::TPosition::A1, scaleL1ByteOffset_, scaleL1Bytes);
    }
    if (static_cast<bool>(tiling_->isBias)) {
        biasL1_ = AscendC::LocalTensor<biasType>(AscendC::TPosition::A1, biasL1ByteOffset_, baseN_ * sizeof(biasType));
    }
    aL0_ = AscendC::LocalTensor<xType>(AscendC::TPosition::A2, 0, 64 * 1024);
    bL0_ = AscendC::LocalTensor<wType>(AscendC::TPosition::B2, 0, 64 * 1024);
    cL0_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::CO1, 0, 4 * 1024 * 1024);
    biasBt_ = AscendC::LocalTensor<biasType>(AscendC::TPosition::C2, 0, 4 * 1024);
    l1LoopCnt_ = 0;
    l0LoopCnt_ = 0;
    l0cPingPong_ = 0;
    scaleLoopCnt_ = 0;
    isFirstTile_ = true;
    hasBias_ = false;
    // L0C双缓冲：tiling保证l0cDB==2时baseM*baseN*4B*2不超过L0C，此处再防御性校验半区容量
    enableL0cDb_ = tiling_->l0cDB >= DOUBLE_BUFFER_NUM && baseM_ * baseN_ <= HALF_L0C_ELEMS;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    UpdateGlobalAddrHelper<xType, wType, biasType, yType, antiQuantType>(
        x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, block_, tiling_,
        blockIdx_, aGlobal_, bGlobal_, cGlobal_, biasGlobal_, scaleGlobal_, tiling_->l2CacheDisable);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetL2CacheHint()
{
    SetL2CacheHintHelper<xType, wType, biasType, yType>(tiling_->l2CacheDisable, aGlobal_, bGlobal_);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    if (tiling_->batchDimAll == 1UL) {
        block_.offset_.batchCOffset = 0;
        block_.offset_.batchAOffset = 0;
        block_.offset_.batchBOffset = 0;
        ComputeBasicOptiLoop();
    } else {
        ComputeBmmOptiLoop();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicOptiLoop()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);
        // 1. Set single core param
        block_.UpdateBlockParams(roundIndex);
        if (block_.params_.singleCoreM == 0 || block_.params_.singleCoreN == 0) {
            continue;
        }
        SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, tiling_->k);
        // 2. compute offset
        block_.CalcGMOffset<aTrans, bTrans, CubeFormat::ND>();
        // 3. set offset and compute
        SetMMParaAndCompute();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBmmOptiLoop()
{
    uint64_t batchC3C4 = static_cast<uint64_t>(tiling_->batchC3) * tiling_->batchC4;
    uint64_t batchC2C3C4 = tiling_->batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(tiling_->batchB3) * tiling_->batchB4;
    uint64_t batchB2B3B4 = tiling_->batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(tiling_->batchA3) * tiling_->batchA4;
    uint64_t batchA2A3A4 = tiling_->batchA2 * batchA3A4;
    uint32_t multiA1C1 = tiling_->batchA1 / tiling_->batchC1;
    uint32_t multiA2C2 = tiling_->batchA2 / tiling_->batchC2;
    uint32_t multiA3C3 = tiling_->batchA3 / tiling_->batchC3;
    uint32_t multiA4C4 = tiling_->batchA4 / tiling_->batchC4;
    uint32_t multiB1C1 = tiling_->batchB1 / tiling_->batchC1;
    uint32_t multiB2C2 = tiling_->batchB2 / tiling_->batchC2;
    uint32_t multiB3C3 = tiling_->batchB3 / tiling_->batchC3;
    uint32_t multiB4C4 = tiling_->batchB4 / tiling_->batchC4;
    for (uint64_t b1Index = 0; b1Index < tiling_->batchC1; ++b1Index) {
        uint64_t batchC1Offset = b1Index * batchC2C3C4;
        uint64_t batchA1Offset = b1Index * batchA2A3A4 * multiA1C1;
        uint64_t batchB1Offset = b1Index * batchB2B3B4 * multiB1C1;
        for (uint64_t b2Index = 0; b2Index < tiling_->batchC2; ++b2Index) {
            uint64_t batchC2Offset = b2Index * batchC3C4 + batchC1Offset;
            uint64_t batchA2Offset = b2Index * batchA3A4 * multiA2C2 + batchA1Offset;
            uint64_t batchB2Offset = b2Index * batchB3B4 * multiB2C2 + batchB1Offset;
            for (uint64_t b3Index = 0; b3Index < tiling_->batchC3; ++b3Index) {
                uint64_t batchC3Offset = b3Index * tiling_->batchC4 + batchC2Offset;
                uint64_t batchA3Offset = b3Index * batchA3A4 * multiA3C3 + batchA2Offset;
                uint64_t batchB3Offset = b3Index * batchB3B4 * multiB3C3 + batchB2Offset;
                for (uint64_t b4Index = 0; b4Index < tiling_->batchC4; ++b4Index) {
                    block_.offset_.batchCOffset = batchC3Offset + b4Index;
                    block_.offset_.batchAOffset = batchA3Offset + b4Index * multiA4C4;
                    block_.offset_.batchBOffset = batchB3Offset + b4Index * multiB4C4;
                    block_.ResetAddressOffsets();
                    ComputeBasicOptiLoop();
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMMParaAndCompute()
{
    if constexpr (antiQuantType == QuantType::PER_TENSOR) { // pertensor
        quantScalar_ = block_.offset_.scaleScalar;
    } else {
        scaleTileGlobal_ = scaleGlobal_[block_.offset_.offsetScale];
    }

    if (tiling_->isBias) {
        biasTileGlobal_ = biasGlobal_[block_.offset_.offsetBias];
        hasBias_ = true;
    }
#if __NPU_ARCH__ == 5102
    shiftValue_ = static_cast<uint8_t>(tiling_->mmadParam);
#endif
    aTileGlobal_ = aGlobal_[block_.offset_.offsetA];
    bTileGlobal_ = bGlobal_[block_.offset_.offsetB];
    Iterate();
    GetTensorC(cGlobal_[block_.offset_.offsetC]);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetSingleShape(int32_t singleM,
                                                                                                     int32_t singleN,
                                                                                                     int32_t singleK)
{
    singleM_ = static_cast<uint64_t>(singleM);
    singleN_ = static_cast<uint64_t>(singleN);
    singleK_ = static_cast<uint64_t>(singleK);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInA1(uint64_t kGmOffset,
                                                                                               uint64_t curK,
                                                                                               uint64_t l1BufId)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = aTrans ? curK : singleM_;
    uint64_t dDim = aTrans ? singleM_ : curK;
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = aTrans ? tiling_->m : tiling_->k;
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, BLOCK_CUBE_SIZE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    uint64_t gmOffset = aTrans ? kGmOffset * static_cast<uint64_t>(tiling_->m) : kGmOffset;
    AscendC::DataCopy(aL1_[l1BufId * aL1BufElems_], aTileGlobal_[gmOffset], nd2nzParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInB1(uint64_t kGmOffset,
                                                                                               uint64_t curK,
                                                                                               uint64_t l1BufId)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = bTrans ? singleN_ : curK;
    uint64_t dDim = bTrans ? curK : singleN_;
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = bTrans ? tiling_->k : tiling_->n;
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, BLOCK_CUBE_SIZE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    uint64_t gmOffset = bTrans ? kGmOffset : kGmOffset * static_cast<uint64_t>(tiling_->n);
    AscendC::DataCopy(bL1_[l1BufId * bL1BufElems_], bTileGlobal_[gmOffset], nd2nzParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInBiasL1()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
    AscendC::DataCopyPadParams padParams;
    AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(singleN_ * sizeof(biasType)), 0, 0};
    AscendC::DataCopyPad(biasL1_, biasTileGlobal_, biasParam, padParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInA2(uint64_t kL1Offset,
                                                                                               uint64_t curK,
                                                                                               uint64_t curK0,
                                                                                               uint64_t l0BufId)
{
    uint64_t mL1Align = CeilAlign(singleM_, BLOCK_CUBE_SIZE);
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    if constexpr (!aTrans) {
        // (M, K)
        loadDataParams.mStep = CeilDiv(singleM_, BLOCK_CUBE_SIZE);
        loadDataParams.kStep = CeilDiv(curK0, C0_SIZE_A);
        loadDataParams.srcStride = CeilDiv(singleM_, BLOCK_CUBE_SIZE);
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;
    } else {
        // (K, M)
        loadDataParams.mStep = CeilDiv(curK0, BLOCK_CUBE_SIZE);
        loadDataParams.kStep = CeilDiv(singleM_, C0_SIZE_A);
        loadDataParams.srcStride = CeilDiv(curK, BLOCK_CUBE_SIZE);
        loadDataParams.dstStride = loadDataParams.kStep;
        loadDataParams.ifTranspose = true;
    }
    uint64_t l1BufId = l1LoopCnt_ & (l1BufNum_ - 1);
    uint64_t aL1Offset = l1BufId * aL1BufElems_ + (aTrans ? (kL1Offset * C0_SIZE_A) : (kL1Offset * mL1Align));
    AscendC::LoadData<xType>(aL0_[l0BufId * HALF_L0A_ELEMS], aL1_[aL1Offset], loadDataParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInB2(uint64_t kL1Offset,
                                                                                               uint64_t curK,
                                                                                               uint64_t curK0,
                                                                                               uint64_t l0BufId)
{
    uint64_t l1BufId = l1LoopCnt_ & (l1BufNum_ - 1);
    uint64_t bL1Offset = l1BufId * bL1BufElems_;
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    if constexpr (bTrans) {
        // (N, K)
        uint64_t nL1Align = CeilAlign(singleN_, BLOCK_CUBE_SIZE);
        loadDataParams.mStep = CeilDiv(singleN_, BLOCK_CUBE_SIZE);
        loadDataParams.kStep = CeilDiv(curK0, C0_SIZE_W);
        loadDataParams.srcStride = CeilDiv(singleN_, BLOCK_CUBE_SIZE);
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;
        bL1Offset += kL1Offset * nL1Align;
        AscendC::LoadData<wType>(bL0_[l0BufId * HALF_L0B_ELEMS], bL1_[bL1Offset], loadDataParams);
    } else {
        // (K, N)
        loadDataParams.kStep = CeilDiv(singleN_, C0_SIZE_W);
        loadDataParams.srcStride = CeilDiv(curK, BLOCK_CUBE_SIZE);
        loadDataParams.dstStride = CeilDiv(singleN_, BLOCK_CUBE_SIZE);
        loadDataParams.ifTranspose = true;
        uint16_t fullMStep = static_cast<uint16_t>(CeilDiv(curK0, BLOCK_CUBE_SIZE));
        // int8 转置加载 mStep 最小为 2，按 2 个分型为步长循环加载
        constexpr uint16_t M_STEP_MIN_VAL_B8 = 2;
        uint16_t l0BLoop = static_cast<uint16_t>(
            CeilDiv(static_cast<uint64_t>(fullMStep), static_cast<uint64_t>(M_STEP_MIN_VAL_B8)));
        loadDataParams.mStep = M_STEP_MIN_VAL_B8;
        uint64_t dstOffset = 0;
        // dst 每次步进 Align(n, 16) 个 32B 块
        uint64_t dstAddrStride = CeilAlign(singleN_, BLOCK_CUBE_SIZE) * ONE_BLK_SIZE / sizeof(wType);
        uint16_t oriMstartPos = static_cast<uint16_t>(CeilDiv(kL1Offset, BLOCK_CUBE_SIZE));
        for (uint16_t idx = 0; idx < l0BLoop; ++idx) {
            loadDataParams.mStartPosition = oriMstartPos + M_STEP_MIN_VAL_B8 * idx;
            AscendC::LoadData<wType>(bL0_[l0BufId * HALF_L0B_ELEMS + dstOffset], bL1_[bL1Offset], loadDataParams);
            dstOffset += dstAddrStride;
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInBiasBt()
{
    // s32场景要对齐到2，因此是align(n / biasC0, 16 / biasC0)
    uint64_t nL1Align = CeilAlign(singleN_, BLOCK_CUBE_SIZE);
    uint64_t btAlign = BLOCK_CUBE_SIZE / C0_SIZE_BIAS;
    uint16_t burstLen = CeilAlign(nL1Align / C0_SIZE_BIAS, btAlign);
    AscendC::DataCopyParams biasParam{1, burstLen, 0, 0};
    // biasBt_为单缓冲，须等上一tile读取bias的Mmad完成后再覆写
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
#if __NPU_ARCH__ == 5102
    biasParam.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W8 - shiftValue_;
#endif
    AscendC::DataCopy(biasBt_, biasL1_, biasParam);
    // bias在L1上的空间随MTE1读完成即可释放
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::MmadCompute(
    uint64_t curK0, uint64_t l0BufId, uint64_t l0cBufId, bool needBias, bool needCInit, bool isFinalAccum)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = singleM_;
    mmadParams.n = singleN_;
    mmadParams.k = curK0;
    mmadParams.disableGemv = true;
    // L0C乒乓使能时用显式flag同步（unitFlag=0）；未使能时开启unitflag：末次累加置3，硬件自动同步M->FIX
    mmadParams.unitFlag = enableL0cDb_ ? 0 : (isFinalAccum ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION);
    // 每个tile的首个Mmad重新初始化L0C（半区），避免跨tile累加
    mmadParams.cmatrixInitVal = (needCInit && !needBias);
#if __NPU_ARCH__ == 5102
    mmadParams.fixShiftVal = shiftValue_;
#endif
    if (needBias) {
        mmadParams.cmatrixSource = true;
        AscendC::Mmad(cL0_[l0cBufId * HALF_L0C_ELEMS], aL0_[l0BufId * HALF_L0A_ELEMS], bL0_[l0BufId * HALF_L0B_ELEMS],
                      biasBt_, mmadParams);
    } else {
        mmadParams.cmatrixSource = false;
        AscendC::Mmad(cL0_[l0cBufId * HALF_L0C_ELEMS], aL0_[l0BufId * HALF_L0A_ELEMS], bL0_[l0BufId * HALF_L0B_ELEMS],
                      mmadParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyScaleL1(uint64_t scaleBufId)
{
    // 等待l1BufNum_级前的同序号scale buffer被fixpipe读完，scale多buffer与A/B L1 buffer同级
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + scaleBufId);
    AscendC::DataCopyPadParams padParams;
    AscendC::DataCopyParams scaleParam{1, static_cast<uint16_t>(singleN_ * sizeof(uint64_t)), 0, 0};
    AscendC::DataCopyPad(scaleL1_[scaleBufId * baseN_], scaleTileGlobal_, scaleParam, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(SCALE_MTE2_FIX_FLAG + scaleBufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(SCALE_MTE2_FIX_FLAG + scaleBufId);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::FixpipeOut(
    const AscendC::GlobalTensor<yType>& gm, uint64_t l0cBufId, uint64_t scaleBufId)
{
    AscendC::FixpipeParamsC310 fixpipeParams;
    fixpipeParams.nSize = singleN_;
    fixpipeParams.mSize = singleM_;
    fixpipeParams.dstStride = tiling_->n;
    fixpipeParams.srcStride = CeilAlign(singleM_, BLOCK_CUBE_SIZE);
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        if constexpr (AscendC::IsSameType<yType, int8_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<yType, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<yType, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::QS322BF16_PRE;
        } else {
            fixpipeParams.quantPre = QuantMode_t::NoQuant;
        }
        fixpipeParams.deqScalar = quantScalar_;
    } else {
        if constexpr (AscendC::IsSameType<yType, int8_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::VREQ8;
        } else if constexpr (AscendC::IsSameType<yType, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<yType, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::VQS322BF16_PRE;
        } else {
            fixpipeParams.quantPre = QuantMode_t::NoQuant;
        }
    }
    // L0C乒乓使能时用显式flag同步（unitFlag=0）；未使能时开启unitflag，硬件自动同步FIX->M
    fixpipeParams.unitFlag = enableL0cDb_ ? 0 : FINAL_ACCUMULATION;
    fixpipeParams.params = {1, 1, 1};
#if __NPU_ARCH__ == 5102
    if constexpr (AscendC::IsSameType<xType, half>::value && AscendC::IsSameType<wType, half>::value) {
        fixpipeParams.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - shiftValue_;
    }
#endif
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        AscendC::Fixpipe(gm, cL0_[l0cBufId * HALF_L0C_ELEMS], fixpipeParams);
    } else {
        AscendC::Fixpipe(gm, cL0_[l0cBufId * HALF_L0C_ELEMS], scaleL1_[scaleBufId * baseN_], fixpipeParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Iterate()
{
    uint64_t l0cBufId = enableL0cDb_ ? (l0cPingPong_ & 1) : 0;
    if (enableL0cDb_) {
        // 等待两轮前GetTensorC对该半区L0C的fixpipe完成，L0C半区可复用
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + l0cBufId);
    }
    // 未使能L0C乒乓时，L0C复用由unitflag硬件自动同步（末次累加M->FIX、fixpipe FIX->M）
    uint64_t kL1Iter = CeilDiv(singleK_, kL1_);
    uint64_t kGmOffset = 0;
    bool biasLoaded = false;
    bool needCInit = true;
    // 首个tile流水线预热：前两轮L1搬运量减半，提前首条Mmad发射。
    // 要求K跨越>=2个L1 chunk（ASW的kL1_未按singleK_截断，防止减半值越过singleK_），
    // 且单chunk含>=2个baseK（保证半块也凑出完整L0迭代，与后续MTE2搬运重叠）
    bool firstKL1Half = isFirstTile_ && kL1Iter >= 2 && kL1_ / baseK_ >= 2;
    if (firstKL1Half) {
        kL1Iter++;
    }
    for (uint64_t kIter = 0; kIter < kL1Iter; ++kIter) {
        uint64_t curK = (kIter + 1 == kL1Iter) ? (singleK_ - kGmOffset) : kL1_;
        // 前两轮搬运量减半：首轮取kL1_的一半（16对齐），次轮补齐该完整chunk的余量
        if (firstKL1Half) {
            if (kIter == 0) {
                curK = CeilAlign(kL1_ / 2, BLOCK_CUBE_SIZE);
            } else if (kIter == 1) {
                curK = kL1_ - kGmOffset;
            }
        }
        uint64_t l1BufId = l1LoopCnt_ & (l1BufNum_ - 1);
        // 1. GM -> L1，多buffer轮转
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + l1BufId);
        CopyInA1(kGmOffset, curK, l1BufId);
        CopyInB1(kGmOffset, curK, l1BufId);
        bool needBias = hasBias_ && !biasLoaded;
        if (needBias) {
            CopyInBiasL1();
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_MTE2_MTE1_FLAG + l1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_MTE2_MTE1_FLAG + l1BufId);
        // 2. L1 -> L0 -> Mmad，L0双缓冲
        uint64_t kL0Iter = CeilDiv(curK, baseK_);
        uint64_t kL1Offset = 0;
        for (uint64_t k0Iter = 0; k0Iter < kL0Iter; ++k0Iter) {
            uint64_t curK0 = (k0Iter + 1 == kL0Iter) ? (curK - kL1Offset) : baseK_;
            uint64_t l0BufId = l0LoopCnt_ & 1;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + l0BufId);
            CopyInA2(kL1Offset, curK, curK0, l0BufId);
            CopyInB2(kL1Offset, curK, curK0, l0BufId);
            if (needBias && k0Iter == 0) {
                CopyInBiasBt();
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_MTE1_M_FLAG + l0BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_MTE1_M_FLAG + l0BufId);
            bool isFinalAccum = (kIter + 1 == kL1Iter) && (k0Iter + 1 == kL0Iter);
            MmadCompute(curK0, l0BufId, l0cBufId, needBias && k0Iter == 0, needCInit, isFinalAccum);
            if (needBias && k0Iter == 0) {
                // bias已随本Mmad读入，允许下一tile覆写biasBt_
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
            }
            needCInit = false;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + l0BufId);
            l0LoopCnt_++;
            kL1Offset += curK0;
        }
        // L1 buffer数据已全部读入L0，释放给下一轮MTE2
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + l1BufId);
        l1LoopCnt_++;
        kGmOffset += curK;
        biasLoaded = biasLoaded || needBias;
    }
    isFirstTile_ = false;
    if (enableL0cDb_) {
        // 3. 等待该半区全部Mmad完成
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0C_M_FIX_FLAG + l0cBufId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0C_M_FIX_FLAG + l0cBufId);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
template <bool sync>
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetTensorC(
    const AscendC::GlobalTensor<yType>& gm)
{
    uint64_t l0cBufId = enableL0cDb_ ? (l0cPingPong_ & 1) : 0;
    uint64_t scaleBufId = scaleLoopCnt_ & (l1BufNum_ - 1);
    if constexpr (antiQuantType != QuantType::PER_TENSOR) {
        CopyScaleL1(scaleBufId);
    }
    FixpipeOut(gm, l0cBufId, scaleBufId);
    if constexpr (antiQuantType != QuantType::PER_TENSOR) {
        // scale已被fixpipe读取，释放该级L1空间
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + scaleBufId);
        scaleLoopCnt_++;
    }
    if (enableL0cDb_) {
        // 标记该半区L0C空闲，供两轮后的Iterate复用
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + l0cBufId);
        l0cPingPong_++;
    }
    // 未使能时L0C复用由fixpipe unitflag硬件自动同步，无需显式SetFlag
}

} // namespace WeightQuantBatchMatmulV2::Arch35
