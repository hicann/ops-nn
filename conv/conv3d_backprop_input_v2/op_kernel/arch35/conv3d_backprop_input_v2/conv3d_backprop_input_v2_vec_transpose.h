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
 * \file conv3d_backprop_input_v2_vec_transpose.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_INPUT_V2_VEC_TRANSPOSE_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_V2_VEC_TRANSPOSE_ADVANCE_H

#include "conv3d_backprop_input_v2_tiling_data.h"

namespace AscendC {
namespace DxVecTranspose{
static constexpr uint8_t SYNC_MODE0 = 0;
static constexpr uint8_t SYNC_MODE2 = 2;
static constexpr uint16_t SYNC_AIV_ONLY_ALL_DET_FLAG = 2;
static constexpr uint16_t SYNC_AIV_AIC_DET_FLAG = 6;
static constexpr uint32_t MULTIPLE_AIV_TO_AIC = 2;
static constexpr uint32_t DATA_BLOCK_SIZE = 32;
static constexpr uint32_t POWER_5 = 5;

template <typename filterType>
class Conv3dDxVecTranspose {
protected:
    GlobalTensor<filterType> filterGm_;
    GlobalTensor<filterType> ubOutGm_; // tmp workspace to store filter matrix with format NDHWC1C0
    TPipe tPipe_;
    TQue<QuePosition::VECIN, 1> vecInQueue_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue_;
    uint32_t cout_ = 0;
    uint32_t cin_ = 0;
    uint32_t cin0_ = 0;
    uint32_t cin0Tail_ = 0;
    uint32_t cin1_ = 0;
    uint32_t dkhkwk_ = 0;
    uint32_t dkhkwkAlign_ = 0;
    uint32_t dkhkwkSize_ = 0;
    uint32_t dkhkwkSizeAlign_ = 0;
    uint64_t totalCnt_ = 0; // cout * cin1
    uint32_t tailBlockCnt_ = 0;
    uint64_t calBlockRound_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t curCoutCnt_ = 0;
    uint32_t curCin1Cnt_ = 0;

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData)
    {
        cout_ = tilingData->conv3DDxTiling.cout;
        cin_ = tilingData->conv3DDxTiling.cin;
        cin0_ = 16; // cin0固定是16
        cin1_ = (cin_ + cin0_ - 1) >> 4; // 4: 16是2的4次方
        cin0Tail_ = cin_ - (cin1_ - 1) * cin0_;
        // 暂不支持切dkhkwk_，dkhkwk_上限为halfUBSize/sizeofFp32/cin0=1984
        dkhkwk_ = tilingData->conv3DDxTiling.dk * tilingData->conv3DDxTiling.hk * tilingData->conv3DDxTiling.wk;
        dkhkwkSize_ = dkhkwk_ * sizeof(filterType);
        dkhkwkSizeAlign_ = ((dkhkwkSize_ + DATA_BLOCK_SIZE - 1) >> POWER_5) * DATA_BLOCK_SIZE; // 向上对其到32B
        dkhkwkAlign_ = dkhkwkSizeAlign_;
        if constexpr (std::is_same<filterType, float>::value) {
            dkhkwkAlign_ = dkhkwkSizeAlign_ >> 2; // 2：filterType is 32bit, sizeof(32bit)=4=2^2
        } else if constexpr (std::is_same<filterType, bfloat16_t>::value || std::is_same<filterType, half>::value) {
            dkhkwkAlign_ = dkhkwkSizeAlign_ >> 1; // 1：filterType is 16bit, sizeof(16bit)=4=2^1
        }
        totalCnt_ = static_cast<uint64_t>(cout_) * static_cast<uint64_t>(cin1_); // 一共处理多少个cin0_*dkhkwk_
        if (totalCnt_ < GetBlockNum() * MULTIPLE_AIV_TO_AIC) { // vec视角
            usedCoreNum_ = (totalCnt_ + 1) >> 1; // cube视角
        } else {
            usedCoreNum_ = GetBlockNum();
        }
        calBlockRound_ = totalCnt_ / (usedCoreNum_ * MULTIPLE_AIV_TO_AIC);
        tailBlockCnt_ = this->totalCnt_ - this->calBlockRound_ * (this->usedCoreNum_ * MULTIPLE_AIV_TO_AIC);
    }

    __aicore__ inline void VecTransposeProcess()
    {
        CopyInToUB();
        TransFormat();
        CopyOutToGm();
    }
    __aicore__ inline void CopyInToUB()
    {
        LocalTensor<filterType> vecInBuf_ = vecInQueue_.template AllocTensor<filterType>();
        uint64_t srcGmOffset = static_cast<uint64_t>(curCoutCnt_) * cin_ * dkhkwk_ + static_cast<uint64_t>(curCin1Cnt_) * cin0_ * dkhkwk_;

        DataCopyExtParams loadGm2UbParams;
        loadGm2UbParams.srcStride = 0; // GM 单位为Byte，变量的数据类型int64_t
        loadGm2UbParams.dstStride = 0; // UB 单位为32Byte，变量的数据类型int64_t
        loadGm2UbParams.blockLen = dkhkwkSize_; // 单位为Byte，变量的数据类型uint32_t
        loadGm2UbParams.blockCount = cin0_;
        if (curCin1Cnt_ == cin1_ - 1) {
            loadGm2UbParams.blockCount = cin0Tail_;
        }
        uint8_t rightPadPoint = static_cast<uint8_t>(dkhkwkAlign_ - dkhkwk_);
        if constexpr (std::is_same<filterType, hifloat8_t>::value || std::is_same<filterType, fp8_e4m3fn_t>::value) {
            // 指令不支持hifloat8_t,fp8_e4m3fn_t，用uint8_t伪装
            DataCopyPadExtParams<uint8_t> padExtParams{true, 0, rightPadPoint, 0};
            DataCopyPad<uint8_t, PaddingMode::Normal>(vecInBuf_.template ReinterpretCast<uint8_t>(),
                filterGm_.template ReinterpretCast<uint8_t>()[srcGmOffset], loadGm2UbParams, padExtParams);
        } else {
            DataCopyPadExtParams<filterType> padExtParams{true, 0, rightPadPoint, 0};
            DataCopyPad<filterType, PaddingMode::Normal>(vecInBuf_, filterGm_[srcGmOffset], loadGm2UbParams, padExtParams);
        }
        vecInQueue_.EnQue(vecInBuf_);
    }

    __aicore__ inline void InitTransDataTo5DParams(TransDataTo5HDParams &transDataParams)
    {
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = static_cast<uint16_t>(dkhkwkSizeAlign_ >> POWER_5); // 只针对16和32bit
        if (transDataParams.repeatTimes == 1) { // 参考AscendC API的使用说明，当repeatTimes为1时，repStride需要设置为0
            transDataParams.dstRepStride = 0; // 单位 dataBlock
            transDataParams.srcRepStride = 0; // 单位 dataBlock
        } else {
            if constexpr (std::is_same<filterType, float>::value) { // 单位 dataBlock
                transDataParams.dstRepStride = (cin0_ * 8 * sizeof(filterType)) >> POWER_5; // 8: 32bit数据VNCHWCONV一个dataBlock是8个point
            } else if constexpr (std::is_same<filterType, bfloat16_t>::value || std::is_same<filterType, half>::value) {
                transDataParams.dstRepStride = (cin0_ * 16 * sizeof(filterType)) >> POWER_5; // 16: 16bit数据VNCHWCONV一个dataBlock是16个point
            } else if constexpr (std::is_same<filterType, hifloat8_t>::value || std::is_same<filterType, fp8_e4m3fn_t>::value) {
                transDataParams.dstRepStride = (cin0_ * 64 * sizeof(filterType)) >> POWER_5; // 64: 8bit时需要用到高低位，一次高低位完成2个dataBlock
            }
            transDataParams.srcRepStride = 1; // 单位 dataBlock
        }
    }

    __aicore__ inline void TransFormat()
    {
        LocalTensor<filterType> vecInBuf_ = vecInQueue_.template DeQue<filterType>();
        LocalTensor<filterType> vecOutBuf_ = vecOutQueue_.template AllocTensor<filterType>();
        TransDataTo5HDParams transDataParams;
        InitTransDataTo5DParams(transDataParams);
        uint64_t dstLocalList[16]; // 16: VNCHWCONV一次处理16个dataBlock，此处必须是立即数
        uint64_t srcLocalList[16]; // 16: VNCHWCONV一次处理16个dataBlock，此处必须是立即数
        uint64_t dstCount = 0;
        if constexpr(std::is_same<filterType, float>::value) {
            for (int i = 0; i < cin0_; i++) {
                dstCount = (i >> 1) * cin0_ + (i & 1) * 8; // 8: 32bit数据VNCHWCONV一个dataBlock是8个point
                dstLocalList[i] = reinterpret_cast<uint64_t>(vecOutBuf_[dstCount].GetPhyAddr());
            }
        } else if constexpr (std::is_same<filterType, bfloat16_t>::value || std::is_same<filterType, half>::value) {
            for (int i = 0; i < cin0_; i++) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(vecOutBuf_[dstCount].GetPhyAddr());
                dstCount += cin0_;
            }
        } else if constexpr (std::is_same<filterType, hifloat8_t>::value || std::is_same<filterType, fp8_e4m3fn_t>::value) {
            for (int i = 0; i < cin0_; i++) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(vecOutBuf_[dstCount].GetPhyAddr());
                dstCount += DATA_BLOCK_SIZE;    // 8bit 需要数据VNCHWCONV一个dataBlock是32个point
            }
        }

        uint64_t srcCount = 0;
        for (int i = 0; i < cin0_; i++) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(vecInBuf_[srcCount].GetPhyAddr());
            srcCount += dkhkwkAlign_;
        }

        if constexpr (std::is_same<filterType, hifloat8_t>::value || std::is_same<filterType, fp8_e4m3fn_t>::value) {
            TransDataTo5HD<uint8_t>(dstLocalList, srcLocalList, transDataParams);
            transDataParams.srcHighHalf = true;
            for (int i = 0; i < cin0_; i++) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(vecOutBuf_[dstCount].GetPhyAddr());
                dstCount += DATA_BLOCK_SIZE;    // 8bit 需要数据VNCHWCONV一个dataBlock是32个point
            }
            TransDataTo5HD<uint8_t>(dstLocalList, srcLocalList, transDataParams);
        } else {
            TransDataTo5HD<filterType>(dstLocalList, srcLocalList, transDataParams);
        }

        vecOutQueue_.EnQue(vecOutBuf_);
        vecInQueue_.FreeTensor(vecInBuf_);
    }
    __aicore__ inline void CopyOutToGm()
    {
        LocalTensor<filterType> vecOutBuf_ = vecOutQueue_.template DeQue<filterType>();
        uint64_t dstGmOffset = static_cast<uint64_t>(curCoutCnt_) * dkhkwk_ * cin1_ * cin0_ + static_cast<uint64_t>(curCin1Cnt_) * cin0_;
        if constexpr (std::is_same<filterType, hifloat8_t>::value || std::is_same<filterType, fp8_e4m3fn_t>::value) {
            // 8bit时，转置完后最内轴c0为32，而真实有效数据量为16，因此需排除无效部分，只写出有效部分
            DataCopyExtParams loadUb2GmParams;
            loadUb2GmParams.srcStride = (DATA_BLOCK_SIZE - cin0_) >> POWER_5; // 单位为 32B
            loadUb2GmParams.dstStride = (cin1_ * cin0_ - cin0_) * sizeof(filterType) ;
            loadUb2GmParams.blockLen = cin0_ * sizeof(filterType);
            loadUb2GmParams.blockCount = dkhkwk_;
            DataCopyPad<filterType>(ubOutGm_[dstGmOffset], vecOutBuf_, loadUb2GmParams);
        } else {
            DataCopyParams loadUb2GmParams;
            loadUb2GmParams.srcStride = 0; // 单位为 32B
            loadUb2GmParams.dstStride = ((cin1_ * cin0_ - cin0_) * sizeof(filterType)) >> POWER_5; // 单位为 32B
            loadUb2GmParams.blockLen = (cin0_ * sizeof(filterType)) >> POWER_5; // 单位为 32B
            loadUb2GmParams.blockCount = dkhkwk_;
            DataCopy<filterType>(ubOutGm_[dstGmOffset], vecOutBuf_, loadUb2GmParams);
        }

        vecOutQueue_.FreeTensor(vecOutBuf_);
    }
    __aicore__ inline void VecNotifyCube()
    {
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL_DET_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL_DET_FLAG);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_DET_FLAG);
    }
public:
    __aicore__ inline Conv3dDxVecTranspose() {}
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR workSpace, const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData)
    {
        InitTilingData(tilingData);
        filterGm_.SetGlobalBuffer((__gm__ filterType *)filter); // ub输入数据
        ubOutGm_.SetGlobalBuffer((__gm__ filterType *)workSpace); // ub输出数据
        uint32_t halfUbSize = TOTAL_UB_SIZE / HALF_FACTOR;
        tPipe_.InitBuffer(vecInQueue_, 1, halfUbSize);
        tPipe_.InitBuffer(vecOutQueue_, 1, halfUbSize);
    }
    __aicore__ inline void Process()
    {
        // here is AIV
        if (GetBlockIdx() >= usedCoreNum_ * MULTIPLE_AIV_TO_AIC) {
            VecNotifyCube();
            return;
        }

        if (GetBlockIdx() < tailBlockCnt_) {
            ++calBlockRound_;
        }
        for (uint64_t i = 0; i < calBlockRound_; i++) { // 一次只处理一个Cin0*Dk*Hk*Wk
            uint64_t curCin0DHWCnt = static_cast<uint64_t>(i) * static_cast<uint64_t>(usedCoreNum_ * MULTIPLE_AIV_TO_AIC) +
                GetBlockIdx();
            curCoutCnt_ = curCin0DHWCnt / cin1_;
            curCin1Cnt_ = curCin0DHWCnt % cin1_;
            VecTransposeProcess(); // 指令的实施
        }
        VecNotifyCube();
    }
    __aicore__ inline void Destroy()
    {
        vecInQueue_.FreeAllEvent();
        vecOutQueue_.FreeAllEvent();
        tPipe_.Destroy();
    }
};
} // namespace DxVecTranspose
} // namespace AscendC

#endif  // CONV3D_BACKPROP_INPUT_V2_VEC_TRANSPOSE_ADVANCE_H