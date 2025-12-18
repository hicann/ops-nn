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
 * \file sparse4to2quant_matmul.h
 * \brief
 */
#ifndef SPARSE4TO2_QUANT_MATMUL_H
#define SPARSE4TO2_QUANT_MATMUL_H

#include "sparse4to2quant_matmul_block.h"
#include "sparse4to2quant_matmul_update.h"

namespace AscendC {
template <
    typename x1Type, typename x2Type, typename yType, CubeFormat x1Format, CubeFormat x2Format, bool aTrans = false,
    bool bTrans = true, class UPDATE_TYPE = Sparse4to2QuantMatmulUpdate>
class Sparse4to2QuantMatmul {
public:
    __aicore__ inline Sparse4to2QuantMatmul()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR sparseWeight, GM_ADDR index, GM_ADDR bias, GM_ADDR xScale, GM_ADDR sparseWeightScale,
        GM_ADDR y, GM_ADDR workSpace, const SparseQmm::Sparse4to2QuantMatmulTilingData* __restrict tilingData,
        TPipe* tPipe)
    {
        blockIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            blockIdx_ /= GetTaskRation();
        }
        subBlockIdx_ = GetSubBlockIdx();
        usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        pipe_ = tPipe;
        mm_.Init(&(tilingData->matmulTiling), pipe_);
        InitTilingData(tilingData);
        InitGlobalBuffers(x, sparseWeight, index, bias, xScale, sparseWeightScale, y, workSpace);
        if ASCEND_IS_AIV {
            InitLocalBuffers();
        }
        offsetWorkspaceC_ = BUFFER_NUM * blockIdx_ * baseM_ * baseN_;
        block_.Init(tilingData);
        update_.template Init<x1Format, x2Format, aTrans, bTrans>(&tilingData->matmulTiling, block_.params_);
        loop_ = 0; // all_gather_quant_batch_mat_mul.h循环调Init和Process，管理CV同步的计数器每次都要清零
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        bool reverse = true;
        bool pongSwitch = false;
        // 每个batch重新初始化offsetWorkspaceC_和控制同步的loop
        offsetWorkspaceC_ = BUFFER_NUM * blockIdx_ * baseM_ * baseN_;
        loop_ = 0;
        uint64_t pingOffsetC = offsetWorkspaceC_;
        // 首块计算，兼容无L2cache切分场景，减少scalar计算
        block_.InitFirstTileBlockIndex();
        OneTileCompute(0, 0, pingOffsetC, pongSwitch);

        for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
            reverse = !reverse;
            for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
                uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
                if (mTileIndex > 0 || nTileIndex > 0) { // 跳过首块
                    block_.UpdateBlockCnt(mTileIndex, nTileIndex);
                    block_.InitBlockIndex();
                    OneTileCompute(mTileIndex, nTileIndex, pingOffsetC, pongSwitch);
                }
            }
        }
        End();
    }

private:
    __aicore__ inline void InitTilingData(const SparseQmm::Sparse4to2QuantMatmulTilingData* tilingData)
    {
        isPerTensor_ = tilingData->params.weightScaleDim == 1;
        m_ = tilingData->matmulTiling.M;
        n_ = tilingData->matmulTiling.N;
        ka_ = tilingData->matmulTiling.Ka;

        baseM_ = tilingData->matmulTiling.baseM;
        baseN_ = tilingData->matmulTiling.baseN;
        hasBias_ = tilingData->matmulTiling.isBias;
        biasDtypeSize_ = sizeof(half);
        ubCalcM_ = tilingData->params.ubCalcM;
        ubCalcN_ = tilingData->params.ubCalcN;
        ubTmpBuffer_ = tilingData->params.needUbBuffer;
    }

    __aicore__ inline void InitGlobalBuffers(
        GM_ADDR x, GM_ADDR sparseWeight, GM_ADDR index, GM_ADDR bias, GM_ADDR xScale, GM_ADDR sparseWeightScale,
        GM_ADDR y, GM_ADDR workSpace)
    {
        if (isPerTensor_) {
            scaleScalar_ = *((__gm__ float*)sparseWeightScale);
        }
        xGm_.SetGlobalBuffer((__gm__ x1Type*)x);
        weightGm_.SetGlobalBuffer((__gm__ x2Type*)sparseWeight);
        indexGm_.SetGlobalBuffer((__gm__ uint8_t*)index);
        if (m_ <= baseM_) {
            weightGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (hasBias_ != 0U) {
            biasGmBf16_.SetGlobalBuffer((__gm__ bfloat16_t*)bias);
        }
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        scaleGm_.SetGlobalBuffer((__gm__ float*)sparseWeightScale);
        xScaleGm_.SetGlobalBuffer((__gm__ float*)xScale);
        mmOutGm_.SetGlobalBuffer((__gm__ int32_t*)workSpace, BUFFER_NUM * usedCoreNum_ * baseM_ * baseN_);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
        pipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(yType));
        if (hasBias_ != 0U) {
            pipe_->InitBuffer(biasFp32Tmp_, ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(vecQueBias_, BUFFER_NUM, ubCalcN_ * biasDtypeSize_);
        }
        if (!isPerTensor_) {
            pipe_->InitBuffer(vecQueWightScale_, BUFFER_NUM, ubCalcN_ * sizeof(float));
        }
        pipe_->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
        pipe_->InitBuffer(vecQueXScale_, BUFFER_NUM, SparseQmm::Align(ubCalcM_, 8U) * sizeof(float));
        pipe_->InitBuffer(broadcastFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
    }

    __aicore__ inline void OneTileCompute(
        uint64_t mTileIndex, uint64_t nTileIndex, uint64_t pingOffsetC, bool& pongSwitch)
    {
        for (uint64_t j = 0; j < block_.realRound_; j++) {
            // 更新此次基本块的大小和输入输出地址
            update_.template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans>(
                block_.params_, offset_, mTileIndex, nTileIndex);
            offsetWorkspaceC_ = pingOffsetC + pongSwitch * baseM_ * baseN_;
            BasicMMDequantCompute(
                block_.params_.singleCoreM, block_.params_.singleCoreN, C2V_PING_FLAG | pongSwitch,
                V2C_PING_FLAG | pongSwitch);
            pongSwitch = !pongSwitch;
            block_.UpdateBlockIndex();
        }
    }

    __aicore__ inline void BasicMMDequantCompute(
        uint32_t CurAicM, uint32_t CurAicN, uint16_t v2cSyncFlag, uint16_t c2vSyncFlag)
    {
        if ASCEND_IS_AIC {
            if (++loop_ > 2) { // 2表示跳过第一次ping和第一次pong
                WaitEvent(v2cSyncFlag);
            }
            BasicMMCompute(CurAicM, CurAicN);
            NotifyEvent<PIPE_FIX>(c2vSyncFlag);
        }

        if ASCEND_IS_AIV {
            WaitEvent(c2vSyncFlag);
            BasicDequantCompute(mmOutGm_, CurAicM, CurAicN);
            NotifyEvent<PIPE_MTE2>(v2cSyncFlag);
        }
    }

    __aicore__ inline void BasicMMCompute(uint32_t baseM, uint32_t baseN)
    {
        mm_.SetSingleShape(baseM, baseN, ka_);
        mm_.SetTensorA(xGm_[offset_.offsetA], aTrans);
        mm_.SetTensorB(weightGm_[offset_.offsetB], bTrans);
        mm_.SetSparseIndex(indexGm_[offset_.offsetSparseIndex]);
        mm_.Iterate();
        mm_.GetTensorC(mmOutGm_[offsetWorkspaceC_], 0, true);
    }

    __aicore__ inline void PertokenCalculate(
        uint32_t basicBlockComputeInfo[], uint32_t mUbLoopIdx, DataCopyPadParams& padParams,
        LocalTensor<float>& dstLocalFp32, LocalTensor<float>& tmpdstLocal)

    {
        uint32_t curAivN = basicBlockComputeInfo[0];
        uint32_t curAivM = basicBlockComputeInfo[1];
        uint32_t ubResAlignedN = basicBlockComputeInfo[2];
        uint32_t subBlockoffset = basicBlockComputeInfo[3]; // 数组下标3

        DataCopyParams scale2UbParams{1, 0, 0, 0};
        scale2UbParams.blockLen = curAivM * sizeof(float);
        uint64_t offsetPertoken = offset_.offsetPertoken + mUbLoopIdx * ubCalcM_ + subBlockoffset;
        uint32_t computedAivN = SparseQmm::Align(curAivN, 8U); // 8: 32B aligned for float

        const uint32_t broadCastDst[M_N_TWO_DIMS] = {curAivM, computedAivN};
        const uint32_t broadCastSrc[M_N_TWO_DIMS] = {curAivM, 1};

        LocalTensor<float> broadcastFp32 = broadcastFp32Tmp_.Get<float>();
        LocalTensor<float> xScaleLocal = vecQueXScale_.AllocTensor<float>();

        DataCopyPad(xScaleLocal, xScaleGm_[offsetPertoken], scale2UbParams, padParams);
        vecQueXScale_.EnQue<float>(xScaleLocal);
        xScaleLocal = vecQueXScale_.DeQue<float>();

        BroadCast<float, M_N_TWO_DIMS, 1>(broadcastFp32, xScaleLocal, broadCastDst, broadCastSrc);

        AscendC::PipeBarrier<PIPE_V>();

        if (computedAivN == ubResAlignedN) {
            Mul(tmpdstLocal, broadcastFp32, dstLocalFp32, computedAivN * curAivM);
        } else {
            for (auto i = 0; i < curAivM; i++) {
                Mul(tmpdstLocal[ubResAlignedN * i], broadcastFp32[computedAivN * i], dstLocalFp32[computedAivN * i],
                    computedAivN);
            }
        }
        vecQueXScale_.FreeTensor(xScaleLocal);
    }

    __aicore__ inline void BasicDequantCompute(GlobalTensor<int32_t>& curMmOutGm, uint32_t curAicM, uint32_t curAicN)
    {
        LocalTensor<float> dstLocalFp32 = outFp32Tmp_.Get<float>();
        LocalTensor<float> biasFp32;
        LocalTensor<bfloat16_t> oriBiasBf16;
        LocalTensor<half> oriBiasFp16;
        LocalTensor<float> oriBiasFp32;
        // aic:aiv 1:2 spilt m
        uint32_t subBlockoffset = 0;
        // 获取cube/vector配比 1:2  在vector会返回2
        int64_t vecNum = GetTaskRation();
        vecNum = (vecNum == 0) ? 1 : vecNum;
        subBlockoffset = subBlockIdx_ * curAicM / vecNum;
        curAicM = curAicM / vecNum + subBlockIdx_ * (curAicM % vecNum);
        uint32_t curAivM = ubCalcM_;
        // calcN in ub is equal to aicN
        uint32_t curAivN = curAicN;
        uint32_t mUbLoops = SparseQmm::CeilDiv(curAicM, ubCalcM_);
        DataCopyParams gm2UbParams{1, 0, 0, 0};
        DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        DataCopyPadParams padParams;
        DequantParams dequantParams;
        SparseQmm::CalcDequantParams(mUbLoops == 1 ? curAicM : ubCalcM_, curAicN, dequantParams);
        for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
            if (mUbLoopIdx == mUbLoops - 1) {
                curAivM = curAicM - ubCalcM_ * (mUbLoops - 1);
                SparseQmm::CalcDequantParams(curAivM, curAicN, dequantParams, mUbLoops != 1 && curAivM != ubCalcM_);
            }
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            LocalTensor<yType> dstLocal = vecQueOut_.AllocTensor<yType>();
            LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();
            // datacopypad 32B aligned
            SparseQmm::SetGm2UbParams(gm2UbParams, curAivM, curAivN);
            SparseQmm::CopyMmOutToLocal(
                srcLocal, curMmOutGm, gm2UbParams, padParams,
                offsetWorkspaceC_ + mUbLoopIdx * ubCalcM_ * curAicN + subBlockoffset * curAicN);
            if (hasBias_ != 0) {
                BiasTensorInit(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32);
                BiasGm2Ub(oriBiasBf16, oriBiasFp16, oriBiasFp32, padParams, curAicN);
            }
            if (isPerTensor_) {
                AscendDequant(dstLocalFp32, srcLocal, scaleScalar_, tmpLocal, dequantParams);
            } else {
                LocalTensor<float> scaleLocal = vecQueWightScale_.AllocTensor<float>();
                SparseQmm::Bf16ScaleGm2Ub<float>(scaleLocal, scaleGm_, padParams, curAicN, offset_.offsetScale);
                AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal, dequantParams);
                vecQueWightScale_.FreeTensor(scaleLocal);
            }
            uint32_t ubResAlignedN = SparseQmm::Align(curAivN); // 16: sizeof(yType) is 2, 32B / 2
            LocalTensor<float> tmpdstLocal = vecQueTmp_.Get<float>();
            uint32_t basicBlockComputeInfo[4] = {curAivN, curAivM, ubResAlignedN, subBlockoffset};
            PertokenCalculate(basicBlockComputeInfo, mUbLoopIdx, padParams, dstLocalFp32, tmpdstLocal);

            if (hasBias_ != 0) {
                CalBiasAdd(tmpdstLocal, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32, curAivN, curAivM);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Cast(dstLocal, tmpdstLocal, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            vecQueSrc_.FreeTensor(srcLocal);
            // dst from ub -> gm
            SparseQmm::SetUb2GmParams<yType>(ub2GmParams, curAivM, curAivN, n_);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            SparseQmm::CopyUbToGm<yType>(
                offset_.offsetC + mUbLoopIdx * ubCalcM_ * n_ + subBlockoffset * n_, ub2GmParams, dstLocal, yGm_,
                vecQueOut_);
        }
    }

    __aicore__ inline void BiasTensorInit(
        LocalTensor<float>& /* dstLocalFp32 */, LocalTensor<float>& biasFp32, LocalTensor<bfloat16_t>& oriBiasBf16,
        LocalTensor<half>& oriBiasFp16, LocalTensor<float>& oriBiasFp32)
    {
        biasFp32 = biasFp32Tmp_.Get<float>();
        oriBiasBf16 = vecQueBias_.AllocTensor<bfloat16_t>(); // free in CalBiasAdd
    }

    __aicore__ inline void BiasGm2Ub(
        LocalTensor<bfloat16_t>& oriBiasBf16, LocalTensor<half>& oriBiasFp16, LocalTensor<float>& oriBiasFp32,
        DataCopyPadParams padParams, uint32_t curAivN)
    {
        DataCopyParams bias2UbParams{1, 0, 0, 0};
        bias2UbParams.blockLen = curAivN * biasDtypeSize_;
        DataCopyPad(oriBiasBf16, biasGmBf16_[offset_.offsetBias], bias2UbParams, padParams);
    }

    __aicore__ inline void CalBiasAdd(
        LocalTensor<float>& dstLocalFp32, LocalTensor<float>& biasFp32, LocalTensor<bfloat16_t>& oriBiasBf16,
        LocalTensor<half>& oriBiasFp16, LocalTensor<float>& oriBiasFp32, uint32_t curAivN, uint32_t curAivM)
    {
        uint32_t computedAivN = SparseQmm::Align(curAivN, 8U); // 8: 32B aligened for int32_t
        uint32_t ubResAlignedN = SparseQmm::Align(curAivN);    // 16: sizeof(ytype) is 2 , 32B / 2
        AscendC::PipeBarrier<PIPE_V>();
        Cast(biasFp32, oriBiasBf16, RoundMode::CAST_NONE, ubResAlignedN);
        AscendC::PipeBarrier<PIPE_V>();
        vecQueBias_.FreeTensor(oriBiasBf16);
        for (int32_t mIdx = 0; mIdx < curAivM; ++mIdx) {
            Add(dstLocalFp32[mIdx * ubResAlignedN], dstLocalFp32[mIdx * ubResAlignedN], biasFp32, ubResAlignedN);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            // AIC跳过前两次Wait，也就是一次ping一次pong，这里补上
            if (loop_ > 2) { // 大于2表示需要补上开头跳过的ping
                WaitEvent(C2V_PING_FLAG);
            }

            if (loop_ > 3) { // 大于3表示需要补上开头跳过的pong
                WaitEvent(C2V_PONG_FLAG);
            }
            mm_.End();
        }
    }

private:
    GlobalTensor<x1Type> xGm_;
    GlobalTensor<x2Type> weightGm_;
    GlobalTensor<uint8_t> indexGm_;
    GlobalTensor<bfloat16_t> biasGmBf16_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> xScaleGm_;
    GlobalTensor<int32_t> mmOutGm_;

    TPipe* pipe_;
    // define the que
    TQue<QuePosition::VECIN, 1> vecQueSrc_;
    TQue<QuePosition::VECIN, 1> vecQueWightScale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    // used when bias type is bf16/fp16/fp32, dequant result should be fp32
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;
    TQue<QuePosition::VECIN, 1> vecQueXScale_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;

    float scaleScalar_;

    // tilling data
    bool isPerTensor_;
    uint32_t usedCoreNum_;
    uint32_t m_;
    uint32_t n_;
    uint32_t ka_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t hasBias_;
    uint32_t biasDtypeSize_ = 0;
    // vector
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;

    uint32_t blockIdx_;
    uint32_t subBlockIdx_ = 0;
    uint64_t offsetWorkspaceC_ = 0;
    uint64_t loop_ = 0;

    Sparse4to2QuantMatmulBlock block_;
    UPDATE_TYPE update_; // 量化mm或mc2的更新计算大小和地址的接口
    SparseBlockOffset offset_;

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, x1Type, aTrans>;
    using BMatmulType = matmul::SparseMatmulType<TPosition::GM, TPosition::GM, x2Format, x2Type, bTrans>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    // notice: the TPos of ctype must be ub given by mm api when iterate<false>,
    // but actually we can move data to gm then to ub.
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_DEFAULT_MDL_CFG> mm_;
};
} // namespace AscendC

#endif // SPARSE4TO2_QUANT_MATMUL_H