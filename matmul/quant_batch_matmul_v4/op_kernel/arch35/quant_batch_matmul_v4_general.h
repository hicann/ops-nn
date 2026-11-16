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
 * \file quant_batch_matmul_v4_general.h
 * \brief
 */

#pragma once

#include "lib/matmul_intf.h"
#include "../../common/arch35/weight_quant_batch_matmul_cube_compute_tools.h"
#include "tool.h"

#ifndef LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
#define LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL \
    template <typename xType, typename wType, typename yType, bool bTrans, typename scaleType>
#endif
#ifndef LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL
#define LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL xType, wType, yType, bTrans, scaleType
#endif

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BLOCK_CUBE;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace MicroAPI = AscendC::MicroAPI;

namespace QuantBatchMatmulV4 {
template <typename xType, typename wType, typename scaleType>
struct ParamsGroupKN {
    uint32_t groupNumUb;       // UB上一次计算多少个group
    uint32_t vLLoopNumInGroup; // 每个group中需要处理几次256个数据
    uint32_t n1LoopNum;        // UB上一次计算多少次n1
    uint32_t scaleN1Stride;    // scale addrReg n1轴stride
    uint32_t bubNLen;          // scale addrReg group轴stride
    uint32_t weightInGroupIdStride;
    uint32_t weighInN1Stride;
    uint32_t weightOutN1Stride;
    uint32_t weightOutGroupIdStride;
    uint32_t weightOutVlStride;
    __local_mem__ scaleType* scaleBaseAddr0;
    __local_mem__ scaleType* scaleBaseAddr1;
    __local_mem__ uint8_t* scaleMaskBaseAddr;
    __local_mem__ int8_t* weightInBaseAddr0;
    __local_mem__ int8_t* weightInBaseAddr1;
    __local_mem__ xType* weightOutBaseAddr;
};

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
class QuantBatchMatmulV4GeneralKernel {
public:
    __aicore__ inline QuantBatchMatmulV4GeneralKernel(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x1, GM_ADDR x2, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitUbBuffs();
    __aicore__ inline void EndSync();
    __aicore__ inline void InitCubeBuffs();

    uint32_t curBlockIdx_;
    const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tiling_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<scaleType> weightScaleGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;

private:
    static constexpr uint8_t MTE_SYNC_FLAG_ID_4 = 4;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_5 = 5;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_6 = 6;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_7 = 7;
    static constexpr uint8_t MTE_SYNC_FLAG_ID_8 = 8;
    static constexpr uint8_t MTE_SYNC_DB_FLAG_IDS_1[DOUBLE_BUFFER_NUM] = {MTE_SYNC_FLAG_ID_4, MTE_SYNC_FLAG_ID_5};
    static constexpr uint8_t MTE_SYNC_DB_FLAG_IDS_2[DOUBLE_BUFFER_NUM] = {MTE_SYNC_FLAG_ID_7, MTE_SYNC_FLAG_ID_8};
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L0A_BUFFER_SIZE = 64 * 1024;
    static constexpr uint64_t L0B_BUFFER_SIZE = 64 * 1024;
    static constexpr uint64_t L0C_BUFFER_SIZE = 256 * 1024;
    static constexpr int32_t C0_SIZE_B8 = 32;
    static constexpr int32_t SCALE_UB_MASK_LEN = 4;
    static constexpr uint64_t SCALE_UB_MASK_VALUE = 0x00000000ffffffff;
    static constexpr int32_t OFFSET_64 = 64;
    static constexpr int32_t VEC_MAX_ELEM_B8 = VECTOR_REG_WIDTH / sizeof(xType);
    // 15:L0A、L0B大小为64K，B8类型的一半需要用1偏移15位
    static constexpr int64_t L0A_BUFFER_HALF_SHL_B8 = 15;
    static constexpr int64_t L0B_BUFFER_HALF_SHL_B8 = 15;

    // 15:L0C大小为256K，输出类型float32的一半需要用1偏移15位
    static constexpr int64_t L0C_BUFFER_HALF_SHL_B32 = 15;

    uint64_t sumLoopNum_ = 0;
    uint64_t pipeCToVSyncCount_ = 0;
    uint64_t totalLoop_ = 0;
    uint64_t roundDbFlag_ = 0;
    uint64_t l1DbFlag_ = 0;
    uint64_t l0DbFlag_ = 0;
    uint64_t kBlockSize_ = 0;
    uint64_t totalKBlocks_ = 0;
    uint64_t weightUbInElements_ = 0;
    uint64_t scaleElements_ = 0;
    uint64_t weightUbOutElements_ = 0;
    uint64_t kUbSize_ = 0;
    uint64_t nUbSize_ = 0;

    // L1 buffers
    LocalTensor<xType> aL1PingLocalTensor_;
    LocalTensor<xType> aL1PongLocalTensor_;
    LocalTensor<xType> bL1PingLocalTensor_;
    LocalTensor<xType> bL1PongLocalTensor_;
    LocalTensor<uint64_t> quantScalePingLocalTensor_;
    LocalTensor<uint64_t> quantScalePongLocalTensor_;

    // UB buffers
    LocalTensor<int8_t> weightUbInTensor_;
    LocalTensor<xType> weightUbOutTensor_;
    LocalTensor<scaleType> weightScaleLocalTensor_;
    LocalTensor<uint64_t> scaleMaskUb_;

    // L0C buffers
    LocalTensor<float> cL0Tensor_;
    LocalTensor<xType> aL0Tensor_;
    LocalTensor<xType> bL0Tensor_;

    __aicore__ inline void AntiquantCompute();
    __aicore__ inline void GetAToL1(uint64_t offsetM, uint64_t kOutIdx);
    __aicore__ inline void CopyQuantScaleToL1(uint64_t offsetN);

    __aicore__ inline void CopyWeightGm2UB(uint64_t offsetN, uint64_t kOutIdx);
    __aicore__ inline void CopyWeightUB2L1();
    __aicore__ inline void IterateMatmul(uint64_t offsetM, uint64_t offsetN, uint64_t kOutIdx, uint64_t kInnerIdx,
                                         uint64_t kOffset);
    __aicore__ inline void GetTensorC(uint64_t offsetM, uint64_t offsetN, const GlobalTensor<yType>& outTensor);
    __aicore__ inline void CopyWeightScaleToUB(uint64_t offsetN, uint64_t kOutIdx);
    __aicore__ inline void MatmulCompute(uint64_t mOffset, uint64_t nOffset);
    __aicore__ inline void InitEventIds();
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();
    __aicore__ inline void WaitAicToAiv();
};

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y, GM_ADDR workspace, const void* tilingData)
{
    tiling_ = static_cast<const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams*>(tilingData);
    curBlockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(x1, x2, x2_scale, y_scale, y, workspace);
    kBlockSize_ = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
    totalKBlocks_ = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, kBlockSize_);

    uint64_t mLoopNum = CeilDiv<uint64_t>(CeilDiv<uint64_t>(tiling_->matmulTiling.M, tiling_->cubeBlockDimM),
                                          tiling_->matmulTiling.baseM);
    uint64_t nLoopNum = CeilDiv<uint64_t>(CeilDiv<uint64_t>(tiling_->matmulTiling.N, tiling_->cubeBlockDimN),
                                          tiling_->matmulTiling.baseN);
    sumLoopNum_ = totalKBlocks_ * mLoopNum * nLoopNum;
    pipeCToVSyncCount_ = sumLoopNum_ > 2 ? sumLoopNum_ - 2 : 0; // 2为double buff，最后两次不用设置同步

    InitCubeBuffs();
    InitEventIds();
    if ASCEND_IS_AIV {
        InitUbBuffs();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::InitEventIds()
{
    if ASCEND_IS_AIV {
        for (int idx = 0; idx < DOUBLE_BUFFER_NUM; ++idx) {
            SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            SetFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        }
    }
    if ASCEND_IS_AIC {
        for (int idx = 0; idx < DOUBLE_BUFFER_NUM; ++idx) {
            SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_2[idx]);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::InitCubeBuffs()
{
    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
    uint64_t bL1DataSizes = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseK *
                            tiling_->matmulTiling.stepKb;
    uint64_t bL1DataBytes = bL1DataSizes * sizeof(xType);

    bL1PingLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, bL1DataSizes);
    l1BufOffsetPart1 += bL1DataBytes;
    bL1PongLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, bL1DataSizes);
    l1BufOffsetPart2 += bL1DataBytes;

    if ASCEND_IS_AIC {
        uint64_t aL1DataSize = tiling_->matmulTiling.baseM * tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseK *
                               tiling_->matmulTiling.stepKa;

        aL1PingLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, aL1DataSize);
        l1BufOffsetPart1 += aL1DataSize * sizeof(xType);
        aL1PongLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, aL1DataSize);
        l1BufOffsetPart2 += aL1DataSize * sizeof(xType);

        uint64_t quantScaleBytes = sizeof(uint64_t) * tiling_->matmulTiling.baseN;
        quantScalePingLocalTensor_ = AscendC::LocalTensor<uint64_t>(AscendC::QuePosition::TSCM, l1BufOffsetPart1,
                                                                    tiling_->matmulTiling.baseN);
        l1BufOffsetPart1 += quantScaleBytes;
        quantScalePongLocalTensor_ = AscendC::LocalTensor<uint64_t>(AscendC::QuePosition::TSCM, l1BufOffsetPart2,
                                                                    tiling_->matmulTiling.baseN);
        l1BufOffsetPart2 += quantScaleBytes;

        // Init L0 Buffers
        cL0Tensor_ = AscendC::LocalTensor<float>(TPosition::CO1, 0, L0C_BUFFER_SIZE / sizeof(float));
        aL0Tensor_ = AscendC::LocalTensor<xType>(TPosition::A2, 0, L0A_BUFFER_SIZE / sizeof(xType));
        // mmad运算之前均转为X的类型
        bL0Tensor_ = AscendC::LocalTensor<xType>(TPosition::B2, 0, L0B_BUFFER_SIZE / sizeof(xType));
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::InitUbBuffs()
{
    // QuantType::PER_GROUP
    uint64_t ubBuffOffset = 0;

    uint64_t singleNSize = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN;
    uint64_t singleKSize = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;
    scaleElements_ = singleNSize * CeilDiv<uint64_t>(singleKSize, tiling_->groupSize);
    weightUbOutElements_ = singleNSize * singleKSize;

    // weightUbOut buff -> xType
    weightUbOutTensor_ = AscendC::LocalTensor<xType>(AscendC::TPosition::VECIN, ubBuffOffset,
                                                     weightUbOutElements_ * DOUBLE_BUFFER_NUM);
    uint64_t weightOutSize = weightUbOutElements_ * sizeof(xType);
    ubBuffOffset += (weightOutSize * DOUBLE_BUFFER_NUM);

    // weightUbScale buff -> scaleType
    weightScaleLocalTensor_ = AscendC::LocalTensor<scaleType>(AscendC::TPosition::VECIN, ubBuffOffset,
                                                              scaleElements_ * DOUBLE_BUFFER_NUM);
    uint64_t scaleSize = scaleElements_ * sizeof(scaleType);
    ubBuffOffset += (scaleSize * DOUBLE_BUFFER_NUM);

    // weightUbIn buff分配：weight dtype是b4
    weightUbInElements_ = (singleNSize * singleKSize);
    if constexpr (AscendC::IsSupportB4<wType>()) {
        weightUbInElements_ = weightUbInElements_ >> 1;
    }
    weightUbInTensor_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECIN, ubBuffOffset,
                                                     weightUbInElements_ * DOUBLE_BUFFER_NUM);
    ubBuffOffset += (weightUbInElements_ * DOUBLE_BUFFER_NUM);

    scaleMaskUb_ = AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECIN, ubBuffOffset, SCALE_UB_MASK_LEN);
    scaleMaskUb_.SetValue(0, SCALE_UB_MASK_VALUE);
    scaleMaskUb_.SetValue(1, SCALE_UB_MASK_VALUE);
    scaleMaskUb_.SetValue(2, SCALE_UB_MASK_VALUE);
    scaleMaskUb_.SetValue(3, SCALE_UB_MASK_VALUE);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::UpdateGlobalAddr(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR y, GM_ADDR workspace)
{
    // 申请 Global Buffer
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x1));
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType*>(x2));
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));
    weightScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType*>(x2_scale));
    quantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(y_scale));
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::Process()
{
    uint64_t usedCoreNum = tiling_->cubeBlockDimM * tiling_->cubeBlockDimN;
    if (curBlockIdx_ >= usedCoreNum) {
        return;
    }

    // 计算总的块数量
    uint64_t totalMBlocks = CeilDiv<uint64_t>(tiling_->matmulTiling.M, tiling_->matmulTiling.baseM);
    uint64_t totalNBlocks = CeilDiv<uint64_t>(tiling_->matmulTiling.N, tiling_->matmulTiling.baseN);
    uint64_t totalBlocks = totalMBlocks * totalNBlocks;

    uint64_t totalKBlocks = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, kBlockSize_);
    // 当前核负责的块索引
    for (uint64_t blockIdx = curBlockIdx_; blockIdx < totalBlocks; blockIdx += usedCoreNum) {
        // 计算对应的 mOffset 和 nOffset
        uint64_t mOffset = (blockIdx / totalNBlocks) * tiling_->matmulTiling.baseM;
        uint64_t nOffset = (blockIdx % totalNBlocks) * tiling_->matmulTiling.baseN;

        // 处理该块
        MatmulCompute(mOffset, nOffset);

        roundDbFlag_ ^= 1;
    }

    EndSync();
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::MatmulCompute(
    uint64_t mOffset, uint64_t nOffset)
{
    uint64_t kOffset = 0;
    for (uint64_t kOutIdx = 0; kOutIdx < totalKBlocks_; kOutIdx++) {
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            CopyWeightGm2UB(nOffset, kOutIdx);
            CopyWeightScaleToUB(nOffset, kOutIdx);
            if (totalLoop_ > 1) {
                WaitAicToAiv();
            }
            AntiquantCompute();
            SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]); // VF结束 可以再搬
            SetFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_6);
            WaitFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_6); // 等VF计算完成 开始拷贝
            CopyWeightUB2L1();
            SetFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]); // 告诉VF前序拷贝完成 可以算别的了
            SetAivToAic();                                                 // 拷贝完成 核间同步
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            uint64_t curKSize = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
            uint64_t kInnerBlockCount = CeilDiv<uint64_t>(curKSize, tiling_->matmulTiling.baseK);
            GetAToL1(mOffset, kOutIdx);
            WaitAivToAic();
            SetFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
                IterateMatmul(mOffset, nOffset, kOutIdx, kInnerIdx, kOffset);
                kOffset += tiling_->matmulTiling.baseK;
            }
            SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            if (totalLoop_ < pipeCToVSyncCount_) {
                SetAicToAiv();
            }
        }
        l1DbFlag_ = l1DbFlag_ ^ 1;
        totalLoop_++;
    }

    if ASCEND_IS_AIC {
        CopyQuantScaleToL1(nOffset);
        uint64_t outOffset = mOffset * tiling_->matmulTiling.N + nOffset;
        GetTensorC(mOffset, nOffset, yGlobal_[outOffset]);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyWeightScaleToUB(
    uint64_t offsetN, uint64_t kOutIdx)
{
    // QuantType::PER_GROUP
    AscendC::DataCopyExtParams dataCopyParams;
    AscendC::DataCopyPadExtParams<scaleType> padParams;

    dataCopyParams.blockCount = CeilDiv<uint64_t>(kUbSize_, tiling_->groupSize);
    // 32B对齐
    dataCopyParams.blockLen = sizeof(scaleType) * nUbSize_;
    dataCopyParams.srcStride = (tiling_->matmulTiling.N - nUbSize_) * sizeof(scaleType);
    dataCopyParams.dstStride = 0;
    uint64_t gmScaleOffset = (kOutIdx * kBlockSize_ / tiling_->groupSize) * tiling_->matmulTiling.N + offsetN;

    DataCopyPad(weightScaleLocalTensor_[scaleElements_ * l1DbFlag_], weightScaleGlobal_[gmScaleOffset], dataCopyParams,
                padParams, true);
}

template <typename xType, typename wType, typename scaleType>
__aicore__ inline void AntiquantW4PergroupKN(ParamsGroupKN<xType, wType, scaleType>& param)
{
    constexpr int32_t OFFSET_FOR_4BITS = 128;
    static constexpr MicroAPI::CastTrait castF162F32Trait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait castF162F32Trait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait castF322F8Trait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::CAST_RINT};
    static constexpr MicroAPI::CastTrait castF322F8Trait2 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::CAST_RINT};
    static constexpr MicroAPI::CastTrait castF42F16Trait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<scaleType> scaleRegCompute, scaleRegAssist, weightF16Reg0, weightF16Reg1;
    MicroAPI::RegTensor<wType> weightInReg0, weightInReg1;
    MicroAPI::RegTensor<float> weightF32Reg0, weightF32Reg1, weightF32Reg2, weightF32Reg3;
    MicroAPI::RegTensor<xType> weightF8Reg0, weightF8Reg1, weightF8Reg2, weightF8Reg3, weightF8SelReg0, weightF8SelReg1;
    MicroAPI::MaskReg scaleMaskReg = MicroAPI::CreateMask<uint8_t>();
    MicroAPI::MaskReg maskRegALL = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRegVsel = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::M4>();

    MicroAPI::DataCopy(scaleMaskReg, param.scaleMaskBaseAddr);
    for (uint16_t n1Idx = 0; n1Idx < param.n1LoopNum; n1Idx++) {
        for (uint16_t groupIdx = 0; groupIdx < param.groupNumUb; groupIdx++) {
            MicroAPI::AddrReg scaleAddrReg = MicroAPI::CreateAddrReg<scaleType>(n1Idx, param.scaleN1Stride, groupIdx,
                                                                                param.bubNLen);
            MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_BLK>(scaleRegCompute, param.scaleBaseAddr0,
                                                                        scaleAddrReg);
            MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_BLK>(scaleRegAssist, param.scaleBaseAddr1,
                                                                        scaleAddrReg);
            MicroAPI::Select(scaleRegCompute, scaleRegCompute, scaleRegAssist, scaleMaskReg);
            for (uint16_t vLIdx = 0; vLIdx < param.vLLoopNumInGroup; vLIdx++) {
                MicroAPI::AddrReg weightInAddrReg = MicroAPI::CreateAddrReg<uint8_t>(
                    n1Idx, param.weighInN1Stride, groupIdx, param.weightInGroupIdStride, vLIdx, OFFSET_FOR_4BITS);
                MicroAPI::AddrReg weightOutAddrReg = MicroAPI::CreateAddrReg<uint8_t>(
                    n1Idx, param.weightOutN1Stride, groupIdx, param.weightOutGroupIdStride, vLIdx,
                    param.weightOutVlStride);
                MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    (MicroAPI::RegTensor<uint8_t>&)weightInReg0, (__local_mem__ uint8_t*)param.weightInBaseAddr0,
                    weightInAddrReg);
                MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    (MicroAPI::RegTensor<uint8_t>&)weightInReg1, (__local_mem__ uint8_t*)param.weightInBaseAddr1,
                    weightInAddrReg);
                MicroAPI::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg0, weightInReg0, maskRegALL);
                MicroAPI::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg1, weightInReg1, maskRegALL);
                MicroAPI::Mul(weightF16Reg0, weightF16Reg0, scaleRegCompute, maskRegALL);
                MicroAPI::Mul(weightF16Reg1, weightF16Reg1, scaleRegCompute, maskRegALL);

                MicroAPI::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg0, weightF16Reg0, maskRegALL);
                MicroAPI::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg1, weightF16Reg0, maskRegALL);
                MicroAPI::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg2, weightF16Reg1, maskRegALL);
                MicroAPI::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg3, weightF16Reg1, maskRegALL);
                MicroAPI::Cast<xType, float, castF322F8Trait0>(weightF8Reg0, weightF32Reg0, maskRegALL);
                MicroAPI::Cast<xType, float, castF322F8Trait2>(weightF8Reg1, weightF32Reg1, maskRegALL);
                MicroAPI::Cast<xType, float, castF322F8Trait0>(weightF8Reg2, weightF32Reg2, maskRegALL);
                MicroAPI::Cast<xType, float, castF322F8Trait2>(weightF8Reg3, weightF32Reg3, maskRegALL);
                MicroAPI::Select(weightF8SelReg0, weightF8Reg0, weightF8Reg1, maskRegVsel);
                MicroAPI::Select(weightF8SelReg1, weightF8Reg2, weightF8Reg3, maskRegVsel);

                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_PACK_B16>(param.weightOutBaseAddr, weightF8SelReg0,
                                                                              weightOutAddrReg, maskRegALL);
                // 第二次数据搬运，UB偏置为128个fp8
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_PACK_B16>(
                    param.weightOutBaseAddr + 128, weightF8SelReg1, weightOutAddrReg, maskRegALL);
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::AntiquantCompute()
{
    // 处理一块 (kBub, nBub), 数据格式为 (nBub1, kBub1, kBub0, nBub0)
    // 每次处理 256 个数，按照 256 个数 (256B) 输出, 对于每个 256B, 间隔 512B 存放, 避免 bank 冲突
    ParamsGroupKN<xType, wType, scaleType> params;
    params.scaleMaskBaseAddr = (__local_mem__ uint8_t*)scaleMaskUb_.GetPhyAddr();
    params.groupNumUb = kUbSize_ / tiling_->groupSize;
    params.vLLoopNumInGroup = tiling_->groupSize * C0_SIZE_B8 / VECTOR_REG_WIDTH;
    params.n1LoopNum = CeilDiv<uint32_t>(nUbSize_, C0_SIZE_B8);
    params.bubNLen = nUbSize_;
    params.scaleN1Stride = C0_SIZE_B8;
    params.weightInGroupIdStride = tiling_->groupSize * C0_SIZE_B8 / 2;
    params.weighInN1Stride = params.weightInGroupIdStride * params.groupNumUb;

    uint64_t scaleUbOffset = scaleElements_ * l1DbFlag_;
    params.scaleBaseAddr0 = (__local_mem__ scaleType*)weightScaleLocalTensor_[scaleUbOffset].GetPhyAddr();
    params.scaleBaseAddr1 = params.scaleBaseAddr0 + BLOCK_CUBE;

    params.weightInBaseAddr0 = (__local_mem__ int8_t*)weightUbInTensor_[weightUbInElements_ * l1DbFlag_].GetPhyAddr();
    params.weightInBaseAddr1 = params.weightInBaseAddr0 + OFFSET_64;

    uint64_t weightOutUbOffset = VEC_MAX_ELEM_B8 * l1DbFlag_;
    params.weightOutBaseAddr = (__local_mem__ xType*)weightUbOutTensor_[weightOutUbOffset].GetPhyAddr();

    params.weightOutVlStride = VECTOR_REG_WIDTH * DOUBLE_BUFFER_NUM;
    params.weightOutGroupIdStride = params.vLLoopNumInGroup * params.weightOutVlStride;
    params.weightOutN1Stride = params.groupNumUb * params.weightOutGroupIdStride;

    SetFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_6);
    WaitFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_6); // 拷贝 等待

    WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);

    AscendC::VF_CALL<AntiquantW4PergroupKN<xType, wType, scaleType>>(params);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyWeightGm2UB(
    uint64_t offsetN, uint64_t kOutIdx)
{
    uint64_t singleNSize = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
    uint64_t kOffset = kOutIdx * kBlockSize_;
    kUbSize_ = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
    nUbSize_ = Min<uint64_t>(tiling_->matmulTiling.N - offsetN, singleNSize);
    AscendC::DataCopyExtParams dataCopyParams;
    AscendC::DataCopyPadExtParams<wType> padParams;

    // 以下bTrans=false且weightNZ
    dataCopyParams.blockCount = CeilDiv<uint16_t>(nUbSize_, C0_SIZE_B8);
    // weight类型为B4:(k1 * k0 * n0)/2 -> (kUbSize_/ BLOCK_CUBE) * (BLOCK_CUBE * C0_SIZE_B8) / 2
    dataCopyParams.blockLen = kUbSize_ * BLOCK_CUBE;
    // (k1 * k0 * n0)/2 -> ((kSize - kUbSize_) / BLOCK_CUBE) * (BLOCK_CUBE * C0_SIZE_B8) / 2
    dataCopyParams.srcStride = (CeilAlign(tiling_->matmulTiling.Kb, BLOCK_CUBE) - kUbSize_) * BLOCK_CUBE;
    dataCopyParams.dstStride = 0;

    uint64_t weightGmOffset = offsetN * CeilAlign(tiling_->matmulTiling.Kb, BLOCK_CUBE) + kOffset * C0_SIZE_B8;

    DataCopyPad(weightUbInTensor_[weightUbInElements_ * l1DbFlag_].template ReinterpretCast<wType>(),
                wGlobal_[weightGmOffset], dataCopyParams, padParams, true);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyWeightUB2L1()
{
    AscendC::DataCopyParams dataCopyParams;

    // weightNZ，非转置(n1,k1,(k0=16,n0=32))
    dataCopyParams.blockLen = VECTOR_REG_WIDTH / C0_SIZE_B8;
    dataCopyParams.blockCount = CeilDiv<uint64_t>(kUbSize_, BLOCK_CUBE) * CeilDiv<uint64_t>(nUbSize_, BLOCK_CUBE);

    // BLOCK_CUBE*BLOCK_CUBE*sizeof(xType)/32B
    dataCopyParams.srcStride = BLOCK_CUBE * BLOCK_CUBE * sizeof(xType) / 32;
    dataCopyParams.dstStride = 0;
    AscendC::DataCopy((l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_ : bL1PongLocalTensor_,
                      weightUbOutTensor_[VEC_MAX_ELEM_B8 * l1DbFlag_], dataCopyParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::IterateMatmul(
    uint64_t offsetM, uint64_t offsetN, uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t kOffset)
{
    if ASCEND_IS_AIC {
        uint64_t singleMSize = tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseM;
        uint64_t singleNSize = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
        WeightQuantBatchMatmul::Arch35::L0CopyAndCalcParams l0CopyAndCalcParams;
        l0CopyAndCalcParams.mL0Size = Min<uint64_t>(tiling_->matmulTiling.M - offsetM, singleMSize);
        l0CopyAndCalcParams.nL0Size = Min<uint64_t>(tiling_->matmulTiling.N - offsetN, singleNSize);
        l0CopyAndCalcParams.nL1Size = l0CopyAndCalcParams.nL0Size;
        l0CopyAndCalcParams.mL1Size = l0CopyAndCalcParams.mL0Size;
        l0CopyAndCalcParams.kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
        l0CopyAndCalcParams.kL0Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, tiling_->matmulTiling.baseK);
        l0CopyAndCalcParams.isFirstKLoop = kOutIdx == 0 && kInnerIdx == 0;
        l0CopyAndCalcParams.isLastKLoop = tiling_->matmulTiling.Ka - kOffset <= tiling_->matmulTiling.baseK;

        uint64_t aL1Offset = kInnerIdx * tiling_->matmulTiling.baseK *
                             CeilAlign<uint64_t>(l0CopyAndCalcParams.mL0Size, BLOCK_CUBE);
        // bTrans:false
        uint64_t bL1Offset = kInnerIdx * tiling_->matmulTiling.baseK * C0_SIZE_B8;

        WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);
        // 取L0A的一半位置
        LocalTensor<xType> aL0Tensor = aL0Tensor_[l0DbFlag_ << L0A_BUFFER_HALF_SHL_B8];
        WeightQuantBatchMatmul::Arch35::LoadAL1ToL0<xType, xType>(
            aL0Tensor, (l1DbFlag_ == IDX_0) ? aL1PingLocalTensor_[aL1Offset] : aL1PongLocalTensor_[aL1Offset],
            l0CopyAndCalcParams);
        // weight在VF量化后变成xType
        LocalTensor<xType> bL0Tensor = bL0Tensor_[l0DbFlag_ << L0B_BUFFER_HALF_SHL_B8];
        WeightQuantBatchMatmul::Arch35::LoadBL1ToL0<xType, xType, bTrans>(
            bL0Tensor, (l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_[bL1Offset] : bL1PongLocalTensor_[bL1Offset],
            l0CopyAndCalcParams);
        SetFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        WaitFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        // 根据cType取L0C的一半位置
        LocalTensor<float> cL0Tensor = cL0Tensor_[roundDbFlag_ << L0C_BUFFER_HALF_SHL_B32];

        WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, xType>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                         l0CopyAndCalcParams);

        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);
        l0DbFlag_ ^= 1;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::GetAToL1(uint64_t offsetM,
                                                                                                     uint64_t kOutIdx)
{
    // A 矩阵 K 方向搬入 L1 Size 计算
    uint64_t kOffset = kOutIdx * kBlockSize_;
    uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
    // A矩阵 M 方向搬入L1 Size计算
    uint64_t singleMSize = tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseM;
    uint64_t mL1Size = Min<uint64_t>(tiling_->matmulTiling.M - offsetM, singleMSize);
    // A 矩阵在 GM 上搬运的偏移计算：mOffset * K + kOffset
    uint64_t aGmOffset = (offsetM * tiling_->matmulTiling.Ka + kOffset);

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = mL1Size;
    nd2nzParams.dValue = kL1Size;
    nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;
    nd2nzParams.srcNdMatrixStride = 0;

    nd2nzParams.dstNzC0Stride = CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    DataCopy((l1DbFlag_ == IDX_0) ? aL1PingLocalTensor_ : aL1PongLocalTensor_, xGlobal_[aGmOffset], nd2nzParams, true);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyQuantScaleToL1(
    uint64_t offsetN)
{
    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    // 32B对齐
    uint64_t singleCoreN = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
    uint64_t curNSize = Min<uint64_t>(tiling_->matmulTiling.N - offsetN, singleCoreN);
    dataCopyParams.blockLen = CeilDiv<uint64_t>(sizeof(uint64_t) * curNSize, 32);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    LocalTensor<uint64_t> scaleLocalTensor = (roundDbFlag_ == 0) ? quantScalePingLocalTensor_ :
                                                                   quantScalePongLocalTensor_;

    SetFlag<HardEvent::FIX_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
    WaitFlag<HardEvent::FIX_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
    DataCopy(scaleLocalTensor, quantScaleGlobal_[offsetN], dataCopyParams, true);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::GetTensorC(
    uint64_t offsetM, uint64_t offsetN, const GlobalTensor<yType>& outTensor)
{
    LocalTensor<float> cL0Tensor = cL0Tensor_[(roundDbFlag_) << L0C_BUFFER_HALF_SHL_B32];
    WeightQuantBatchMatmul::Arch35::FixL0CToDstParams fixL0CToDstParams;
    uint64_t singleMSize = tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseM;
    uint64_t singleNSize = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;

    fixL0CToDstParams.mL0Size = Min<uint64_t>(tiling_->matmulTiling.M - offsetM, singleMSize);
    fixL0CToDstParams.nL0Size = Min<uint64_t>(tiling_->matmulTiling.N - offsetN, singleNSize);
    fixL0CToDstParams.outNSize = tiling_->matmulTiling.N;

    SetFlag<HardEvent::MTE2_FIX>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]);
    WaitFlag<HardEvent::MTE2_FIX>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]);

    WeightQuantBatchMatmul::Arch35::FixL0CToDst<yType, float, true>(
        outTensor, cL0Tensor, roundDbFlag_ == 0 ? quantScalePingLocalTensor_ : quantScalePongLocalTensor_,
        fixL0CToDstParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::EndSync()
{
    if ASCEND_IS_AIV {
        for (int idx = 0; idx < DOUBLE_BUFFER_NUM; ++idx) {
            WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        }
    }

    if ASCEND_IS_AIC {
        for (int idx = 0; idx < DOUBLE_BUFFER_NUM; ++idx) {
            WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_2[idx]);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::SetAivToAic()
{
    AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::WaitAivToAic()
{
    AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}
LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::SetAicToAiv()
{
    AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void QuantBatchMatmulV4GeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::WaitAicToAiv()
{
    AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
}

} // namespace QuantBatchMatmulV4
