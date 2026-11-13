/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_mx.h
 * \brief
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "../../../common/arch35/weight_quant_batch_matmul_cube_compute_tools.h"
#include "../tool_arch35.h"
#include "../../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_9201_asw_block.h"
#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_common.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                  \
    template <typename xType, typename wType, typename biasType, typename scaleType, typename yType, \
              QuantType antiQuantType>
#define LOCAL_TEMPLATE_FUNC_PARAMS xType, wType, biasType, scaleType, yType, antiQuantType

using AscendC::BLOCK_CUBE;
using AscendC::Dn2NzParams;

namespace WeightQuantBatchMatmulV2::Arch35 {

LOCAL_TEMPLATE_CLASS_PARAMS
class WeightQuantBatchMatmulV2ASWMxKernel {
public:
    __aicore__ inline WeightQuantBatchMatmulV2ASWMxKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeBasicOptiLoop();
    __aicore__ inline void EndSync();
    __aicore__ inline void InitCubeBuffs();
    __aicore__ inline void InitEventIds();
    __aicore__ inline void CopyScaleToL1(uint64_t kOutIdx);
    __aicore__ inline void CopyBias2L1();
    __aicore__ inline void GetABToL1(uint64_t kOutIdx);
    __aicore__ inline void CopyXNd2Nz(uint64_t kOutIdx);
    __aicore__ inline void CopyWeightGm2L1(uint64_t kOutIdx);
    __aicore__ inline void GetTensorC(const GlobalTensor<yType>& outTensor);
    __aicore__ inline void IterateMatmul(uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t kOffset);
    __aicore__ inline void MatmulCompute(uint64_t roundIndex);

    uint32_t blockIdx_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData* tiling_;

    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<scaleType> scaleGlobal_;
    GlobalTensor<uint8_t> wGlobalU8_;

    uint64_t roundDbFlag_ = 0;
    uint64_t l1DbFlag_ = 0;
    uint64_t l0DbFlag_ = 0;
    uint64_t kBlockSize_ = 0;
    uint64_t totalKBlocks_ = 0;
    uint64_t scaleKbSize_ = 1;

    LocalTensor<xType> aL1PingLocalTensor_;
    LocalTensor<xType> aL1PongLocalTensor_;
    LocalTensor<wType> bL1PingLocalTensor_;
    LocalTensor<wType> bL1PongLocalTensor_;
    LocalTensor<scaleType> scalePingLocalTensor_;
    LocalTensor<scaleType> scalePongLocalTensor_;
    LocalTensor<biasType> biasPingLocalTensor_;
    LocalTensor<biasType> biasPongLocalTensor_;

    LocalTensor<float> cL0Tensor_;
    LocalTensor<xType> aL0Tensor_;
    LocalTensor<wType> bL0Tensor_;
    LocalTensor<float> biasBtTensor_;

    WeightQuantBatchMatmulV2::Arch35::WeightQuantBmmAswBlock block_;

    static constexpr uint64_t MX_GROUP_SIZE = GetMxGroupSize<wType>();
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData)
{
    if ASCEND_IS_AIV {
        return;
    }

    tiling_ = static_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData*>(tilingData);
    blockIdx_ = AscendC::GetBlockIdx();
    UpdateGlobalAddr(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    wGlobalU8_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(weight));
    kBlockSize_ = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
    totalKBlocks_ = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, kBlockSize_);
    InitCubeBuffs();
    InitEventIds();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    block_.Init(tiling_, blockIdx_);

    // update global buffer
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x));
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType*>(weight));
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));
    scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType*>(antiquantScale));
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias));
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitCubeBuffs()
{
    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
    uint64_t bL1DataSizes = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseK *
                            tiling_->matmulTiling.stepKb;
    uint64_t bL1DataBytes = bL1DataSizes;
    if constexpr (AscendC::IsSupportB4<wType>()) {
        bL1DataBytes = bL1DataSizes >> 1;
    }
    uint64_t aL1DataSize = tiling_->matmulTiling.baseM * tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseK *
                           tiling_->matmulTiling.stepKa;
    uint64_t scaleKbFactor = (tiling_->matmulTiling.mxTypePara >> 8) & 0xff; // 高8位表示scaleKbFactor
    scaleKbSize_ = scaleKbFactor * kBlockSize_ / MX_GROUP_SIZE;
    uint64_t scaleDataSize = scaleKbSize_ * tiling_->matmulTiling.baseN;
    uint64_t scaleBytes = scaleDataSize * sizeof(scaleType);

    bL1PingLocalTensor_ = AscendC::LocalTensor<wType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, bL1DataSizes);
    l1BufOffsetPart1 += bL1DataBytes;
    bL1PongLocalTensor_ = AscendC::LocalTensor<wType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, bL1DataSizes);
    l1BufOffsetPart2 += bL1DataBytes;

    aL1PingLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, aL1DataSize);
    l1BufOffsetPart1 += aL1DataSize * sizeof(xType);
    aL1PongLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, aL1DataSize);
    l1BufOffsetPart2 += aL1DataSize * sizeof(xType);

    scalePingLocalTensor_ = AscendC::LocalTensor<scaleType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1,
                                                            scaleDataSize);
    l1BufOffsetPart1 += scaleBytes;
    scalePongLocalTensor_ = AscendC::LocalTensor<scaleType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2,
                                                            scaleDataSize);
    l1BufOffsetPart2 += scaleBytes;
    if (tiling_->matmulTiling.isBias) {
        uint64_t biasBytes = tiling_->matmulTiling.baseN * sizeof(biasType);
        biasPingLocalTensor_ = AscendC::LocalTensor<biasType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1,
                                                              tiling_->matmulTiling.baseN);
        l1BufOffsetPart1 += biasBytes;
        biasPongLocalTensor_ = AscendC::LocalTensor<biasType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2,
                                                              tiling_->matmulTiling.baseN);
        l1BufOffsetPart2 += biasBytes;
        biasBtTensor_ = AscendC::LocalTensor<float>(TPosition::C2, 0, BIAS_TABLE_SIZE / sizeof(float));
    }

    cL0Tensor_ = AscendC::LocalTensor<float>(TPosition::CO1, 0, L0C_BUFFER_SIZE / sizeof(float));
    aL0Tensor_ = AscendC::LocalTensor<xType>(TPosition::A2, 0, L0A_BUFFER_SIZE / sizeof(xType));
    bL0Tensor_ = AscendC::LocalTensor<wType>(TPosition::B2, 0, L0B_BUFFER_SIZE << 1);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitEventIds()
{
    for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_2[idx]);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::EndSync()
{
    if ASCEND_IS_AIC {
        for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
            WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_2[idx]);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    ComputeBasicOptiLoop();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicOptiLoop()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);

        if (block_.evadeDuplicateBlock_ && block_.params_.index >= block_.params_.nCnt) {
            continue;
        }
        // 1. Set single core param
        block_.UpdateBlockParams(roundIndex);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }

        // 2. compute offset
        block_.template CalcGMOffset<false, true, wType, CubeFormat::NZ, true, antiQuantType>();
        // 3. compute
        MatmulCompute(roundIndex);
        roundDbFlag_ ^= 1;
    }
    EndSync();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::MatmulCompute(
    uint64_t roundIndex)
{
    uint64_t kOffset = 0;
    if (tiling_->matmulTiling.isBias) {
        WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]); // bias的反向等待ID和AB矩阵区分开
        CopyBias2L1();
        uint64_t nL0Size = Min<uint64_t>(tiling_->matmulTiling.N - block_.offset_.offsetN, block_.params_.singleCoreN);
        SetFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_FLAG_ID_6);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_FLAG_ID_6);
        WeightQuantBatchMatmul::Arch35::CopyBiasL1ToBT(
            biasBtTensor_[roundDbFlag_ << 9], (roundDbFlag_ == IDX_0) ? biasPingLocalTensor_ : biasPongLocalTensor_,
            nL0Size); // 左移9位，biastable的一半
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]);
    }
    for (uint64_t kOutIdx = 0; kOutIdx < totalKBlocks_; kOutIdx++) {
        WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);

        GetABToL1(kOutIdx);
        CopyScaleToL1(kOutIdx);

        uint64_t curKSize = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
        uint64_t kInnerBlockCount = CeilDiv<uint64_t>(curKSize, tiling_->matmulTiling.baseK);

        SetFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);

        for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
            IterateMatmul(kOutIdx, kInnerIdx, kOffset);
            kOffset += tiling_->matmulTiling.baseK;
        }
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);

        l1DbFlag_ = l1DbFlag_ ^ 1;
    }
    GetTensorC(cGlobal_[block_.offset_.offsetC]);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IterateMatmul(
    uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t kOffset)
{
    if ASCEND_IS_AIC {
        WeightQuantBatchMatmul::Arch35::L0CopyAndCalcParams l0CopyAndCalcParams;
        l0CopyAndCalcParams.mL0Size = Min<uint64_t>(tiling_->matmulTiling.M - block_.offset_.offsetM,
                                                    block_.params_.singleCoreM);
        l0CopyAndCalcParams.nL0Size = Min<uint64_t>(tiling_->matmulTiling.N - block_.offset_.offsetN,
                                                    block_.params_.singleCoreN);
        l0CopyAndCalcParams.nL1Size = l0CopyAndCalcParams.nL0Size;
        l0CopyAndCalcParams.mL1Size = l0CopyAndCalcParams.mL0Size;
        l0CopyAndCalcParams.kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
        l0CopyAndCalcParams.kL0Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, tiling_->matmulTiling.baseK);
        l0CopyAndCalcParams.isFirstKLoop = kOutIdx == 0 && kInnerIdx == 0;
        l0CopyAndCalcParams.isLastKLoop = tiling_->matmulTiling.Ka - kOffset <= tiling_->matmulTiling.baseK;

        uint64_t aL1Offset = kInnerIdx * tiling_->matmulTiling.baseK *
                             CeilAlign<uint64_t>(l0CopyAndCalcParams.mL0Size, BLOCK_CUBE);
        uint64_t bL1Offset = kInnerIdx * tiling_->matmulTiling.baseK *
                             CeilAlign<uint64_t>(l0CopyAndCalcParams.nL0Size, BLOCK_CUBE);
        WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);
        // 14: 根据xType取L0A的一半位置
        LocalTensor<xType> aL0Tensor = aL0Tensor_[l0DbFlag_ << 14];
        WeightQuantBatchMatmul::Arch35::LoadAL1ToL0<xType, xType>(
            aL0Tensor, (l1DbFlag_ == IDX_0) ? aL1PingLocalTensor_[aL1Offset] : aL1PongLocalTensor_[aL1Offset],
            l0CopyAndCalcParams);
        // 16: 根据wType取L0B的一半位置
        LocalTensor<wType> bL0Tensor = bL0Tensor_[l0DbFlag_ << 16];

        l0CopyAndCalcParams.scaleKL1Size = scaleKbSize_;
        uint64_t scaleL1Offset;
        if constexpr (IsSameType<wType, hifloat4x2_t>::value) {
            scaleL1Offset = kInnerIdx * tiling_->matmulTiling.baseK / 4;
        } else {
            scaleL1Offset = kInnerIdx * tiling_->matmulTiling.baseK / 2;
        }
        WeightQuantBatchMatmul::Arch35::LoadBAndScaleL1ToL0<wType, wType, scaleType>(
            bL0Tensor, (l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_[bL1Offset] : bL1PongLocalTensor_[bL1Offset],
            (l1DbFlag_ == IDX_0) ? scalePingLocalTensor_[scaleL1Offset] : scalePongLocalTensor_[scaleL1Offset],
            l0CopyAndCalcParams);

        uint64_t tailK = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, tiling_->matmulTiling.baseK);
        // 15: 根据cType取L0C的一半位置
        LocalTensor<float> cL0Tensor = cL0Tensor_[roundDbFlag_ << 15];
        SetFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        WaitFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        if (l0CopyAndCalcParams.isFirstKLoop && tiling_->matmulTiling.isBias) {
            // 9: baisTable一半位置
            WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, wType, float>(
                cL0Tensor, aL0Tensor, bL0Tensor, biasBtTensor_[roundDbFlag_ << 9], l0CopyAndCalcParams);
        } else {
            WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, wType>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                             l0CopyAndCalcParams);
        }

        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);

        l0DbFlag_ ^= 1;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetABToL1(uint64_t kOutIdx)
{
    CopyXNd2Nz(kOutIdx);
    CopyWeightGm2L1(kOutIdx);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyScaleToL1(uint64_t kOutIdx)
{
    uint64_t kOffset = kOutIdx * kBlockSize_;
    uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Kb - kOffset, kBlockSize_);
    uint64_t nL1Size = Min<uint64_t>(tiling_->matmulTiling.N - block_.offset_.offsetN, block_.params_.singleCoreN);
    uint64_t scaleGmOffset = (block_.offset_.offsetN * tiling_->matmulTiling.Kb + kOffset) / MX_GROUP_SIZE;

    kL1Size = CeilAlign<uint64_t>(kL1Size, C0<wType>::value);
    uint64_t kAlignSize = CeilAlign<uint64_t>(tiling_->matmulTiling.Kb, C0<wType>::value);
    scaleGmOffset = (block_.offset_.offsetN * kAlignSize + kOffset) / MX_GROUP_SIZE;

    uint64_t scaleGmSize;
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = nL1Size;
    dn2NzParams.srcDnMatrixStride = 0;
    if constexpr (IsSameType<wType, hifloat4x2_t>::value) {
        dn2NzParams.nValue = CeilDiv<uint64_t>(kL1Size, MX_GROUP_SIZE) * SCALE_COPY_GROUP_SIZE;

        dn2NzParams.srcDValue = CeilDiv<uint64_t>(tiling_->matmulTiling.Kb, MX_GROUP_SIZE) * SCALE_COPY_GROUP_SIZE;
        dn2NzParams.dstNzC0Stride = scaleKbSize_ * SCALE_COPY_GROUP_SIZE;
        scaleGmSize = nL1Size * kL1Size / MX_GROUP_SIZE * SCALE_COPY_GROUP_SIZE;
    } else {
        dn2NzParams.nValue = CeilDiv<uint64_t>(CeilDiv<uint64_t>(kL1Size, MX_GROUP_SIZE),
                                               SCALE_COPY_GROUP_SIZE); // 将两个e8m0合成一个half类型，一起搬运

        dn2NzParams.srcDValue = CeilDiv<uint64_t>(CeilDiv<uint64_t>(tiling_->matmulTiling.Kb, MX_GROUP_SIZE),
                                                  SCALE_COPY_GROUP_SIZE);
        dn2NzParams
            .dstNzC0Stride = scaleKbSize_ /
                             SCALE_COPY_GROUP_SIZE; // stride始终按照tiling预分配的空间跳，便于MTE1统一设置src_stride
        scaleGmSize = nL1Size * kL1Size / MX_GROUP_SIZE / SCALE_COPY_GROUP_SIZE;
    }

    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = 0;
    GlobalTensor<half> b16ScaleGlobal;
    b16ScaleGlobal.SetGlobalBuffer((__gm__ half*)(scaleGlobal_[scaleGmOffset].GetPhyAddr()), scaleGmSize);
    auto b16ScaleLocal = (l1DbFlag_ == IDX_0) ? scalePingLocalTensor_.template ReinterpretCast<half>() :
                                                scalePongLocalTensor_.template ReinterpretCast<half>();
    DataCopy(b16ScaleLocal, b16ScaleGlobal, dn2NzParams, tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyXNd2Nz(uint64_t kOutIdx)
{
    uint64_t kOffset = kOutIdx * kBlockSize_;
    uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
    uint64_t mL1Size = Min<uint64_t>(tiling_->matmulTiling.M - block_.offset_.offsetM, block_.params_.singleCoreM);
    uint64_t aGmOffset = (block_.offset_.offsetM * tiling_->matmulTiling.Ka + kOffset);

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = mL1Size;
    nd2nzParams.dValue = kL1Size;
    nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;
    nd2nzParams.srcNdMatrixStride = 0;

    nd2nzParams.dstNzC0Stride = CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy((l1DbFlag_ == IDX_0) ? aL1PingLocalTensor_ : aL1PongLocalTensor_, aGlobal_[aGmOffset], nd2nzParams,
             tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyWeightGm2L1(
    uint64_t kOutIdx)
{
    uint64_t kOffset = kOutIdx * kBlockSize_;
    uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
    uint64_t nL1Size = Min<uint64_t>(tiling_->matmulTiling.N - block_.offset_.offsetN, block_.params_.singleCoreN);
    // 转置的 B 矩阵在 GM 上搬运的偏移计算 (w4 需要除以 2)：kOffset * problem_n + nOffset * K0_INT4
    uint64_t weightGmOffset = (kOffset * tiling_->matmulTiling.N + block_.offset_.offsetN * C0<wType>::value) >> 1;

    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = CeilDiv<uint64_t>(kL1Size, C0<wType>::value);
    dataCopyParams.blockLen = nL1Size;
    dataCopyParams.srcStride = tiling_->matmulTiling.N - nL1Size;
    dataCopyParams.dstStride = CeilAlign<uint64_t>(nL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE)) - nL1Size;

    // weight shape:(k1,n1,(n0=16,k0=64))
    uint64_t nAlignSize = CeilAlign<uint64_t>(tiling_->matmulTiling.N, AscendC::BLOCK_CUBE);
    weightGmOffset = (kOffset * nAlignSize + block_.offset_.offsetN * C0<wType>::value) >> 1;

    uint64_t nL1AlignSize = CeilAlign<uint64_t>(nL1Size, AscendC::BLOCK_CUBE);
    dataCopyParams.srcStride = nAlignSize - nL1Size;

    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    AscendC::DataCopy((l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_.template ReinterpretCast<uint8_t>() :
                                             bL1PongLocalTensor_.template ReinterpretCast<uint8_t>(),
                      wGlobalU8_[weightGmOffset], dataCopyParams, tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyBias2L1()
{
    AscendC::DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = block_.params_.singleCoreN * sizeof(biasType);
    params.srcStride = (tiling_->matmulTiling.N - block_.params_.singleCoreN) * sizeof(biasType);
    params.dstStride = 0;
    AscendC::DataCopyPadParams padParams;
    AscendC::DataCopyPad((roundDbFlag_ == IDX_0) ? biasPingLocalTensor_ : biasPongLocalTensor_,
                         biasGlobal_[block_.offset_.offsetBias], params, padParams, tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetTensorC(
    const GlobalTensor<yType>& outTensor)
{
    LocalTensor<float> cL0Tensor = cL0Tensor_[(roundDbFlag_) << 15];
    WeightQuantBatchMatmul::Arch35::FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = block_.params_.singleCoreM;
    fixL0CToDstParams.nL0Size = block_.params_.singleCoreN;
    fixL0CToDstParams.outNSize = tiling_->matmulTiling.N;
    WeightQuantBatchMatmul::Arch35::FixL0CToDst<yType, float>(outTensor, cL0Tensor, fixL0CToDstParams);
}

} // namespace WeightQuantBatchMatmulV2::Arch35
