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
 * \file quant_batch_matmul_v4_perblock.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_PERBLOCK_H
#define QUANT_BATCH_MATMUL_V4_PERBLOCK_H

#include "quant_batch_matmul_v4_common.h"

namespace AscendC {

template <
    typename xType, typename wType, typename biasType, typename x1scaleType, typename x2scaleType, typename yType,
    bool weightNz = false>
class QuantBatchMatmulV4Perblock : public QuantBatchMatmulV4Common
{
public:
    __aicore__ inline QuantBatchMatmulV4Perblock(){};
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
        GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y, GM_ADDR workspace,
        const QuantBatchMatmulV4PerblockTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void InitLocalBuffers();

private:
    uint32_t nNgroups_;
    uint32_t loopk_;
    uint32_t singleCoreK_;
    bool transA_ = false;
    bool transB_ = true;

    bool aL1PingFlag = true;
    bool bL1PingFlag = true;
    bool wsPingFlag = true;
    bool ubPingFlag = true;

    GlobalTensor<xType> x1Global_;
    GlobalTensor<wType> x2Global_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<int32_t> mmOutGlobal_;
    GlobalTensor<x1scaleType> x1ScaleGlobal_;
    GlobalTensor<x2scaleType> x2ScaleGlobal_;
    GlobalTensor<yType> yGlobal_;
    // define pingpong buf
    TBuf<TPosition::A1> L1Buf_;
    LocalTensor<xType> aL1Ping;
    LocalTensor<xType> aL1Pong;
    LocalTensor<wType> bL1Ping;
    LocalTensor<wType> bL1Pong;
    LocalTensor<xType> aL1;
    LocalTensor<wType> bL1;

    // Vec buffer
    // 不切N ubCalcN_ = BaseN
    // mmOut, mmOutFp32, dequantOut可共享内存
    TBuf<> ubBuf_;
    LocalTensor<int32_t> mmOut;     // [ubCalcM_, ubCalcN_]
    LocalTensor<float> mmOutFp32;   // [ubCalcM_, ubCalcN_]
    LocalTensor<float> dequantAccu; // [ubCalcM_, ubCalcN_]
    LocalTensor<yType> dequantOut;  // [ubCalcM_, ubCalcN_]
    LocalTensor<float> X1Scale;     // [ubCalcM_, 8] // 8个K方向scale保证对齐
    LocalTensor<float> MulScale;    // [ubCalcN_ / 128, ubCalcM_, 8]
    LocalTensor<float> biasAccu;        // [ubCalcN_]
    LocalTensor<float> BrcbScale;
    LocalTensor<float> x2Scale;

    TEventID eVecPing = EVENT_ID0;
    TEventID eVecPong = EVENT_ID1;

    uint32_t mStart_;
    uint32_t mEnd_;
    uint32_t ubMTail_;
    uint64_t loop_ = 0;

    __aicore__ inline uint32_t GetCurAivM() { // 计算m轴尾块
        if ASCEND_IS_AIC {
            return 0;
        }
        uint32_t curAivM = ubCalcM_;
        if (block_.params_.singleCoreM == baseM_) {
            curAivM = ubCalcM_; // ubCalcM_
        } else { // m轴尾块
            if (subBlockIdx_ == 0) {
                curAivM = block_.params_.singleCoreM > curAivM ? curAivM : block_.params_.singleCoreM;
            } else {
                curAivM = block_.params_.singleCoreM > curAivM ? block_.params_.singleCoreM - ubCalcM_ : 0UL;
            }
        }
        return curAivM;
    }

    // tilingData
    using inputX1Type = MatmulType<TPosition::A1, CubeFormat::NZ, int8_t, false>;
    using inputX2Type = MatmulType<TPosition::B1, CubeFormat::NZ, int8_t, true>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
    MatmulImpl<inputX1Type, inputX2Type, outputYType, inputBiasType, CFG_MDL> mmObj_;

    __aicore__ inline void CopyInAL1(uint32_t kidx, uint32_t realM, uint32_t realK)
    {
        // [m, k]
        aL1 = aL1PingFlag ? aL1Ping : aL1Pong;
        TEventID e12 = aL1PingFlag ? eAL1Ping12_ : eAL1Pong12_;
        TEventID e21 = aL1PingFlag ? eAL1Ping21_ : eAL1Pong21_;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.srcDValue = block_.matmulTilingData_->Ka;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        nd2nzParams.dstNzC0Stride = CeilAlign(realM, ALIGN_UNIT_16);

        uint64_t Offset = offset_.offsetA;
        Offset += kidx * groupSizeK_;
        nd2nzParams.nValue = realM;
        nd2nzParams.dValue = realK;
        WaitFlag<HardEvent::MTE1_MTE2>(e12);
        DataCopy(aL1, x1Global_[Offset], nd2nzParams);
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
    }

    __aicore__ inline void CopyInBL1(uint32_t kidx, uint32_t realN, uint32_t realK)
    {
        // [n, k]
        bL1 = bL1PingFlag ? bL1Ping : bL1Pong;
        TEventID e12 = bL1PingFlag ? eBL1Ping12_ : eBL1Pong12_;
        TEventID e21 = bL1PingFlag ? eBL1Ping21_ : eBL1Pong21_;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.srcDValue = block_.matmulTilingData_->Ka;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        nd2nzParams.dstNzC0Stride = CeilAlign(realN, ALIGN_UNIT_16);

        uint64_t Offset = offset_.offsetB;
        Offset += kidx * groupSizeK_;
        nd2nzParams.nValue = realN;
        nd2nzParams.dValue = realK;
        WaitFlag<HardEvent::MTE1_MTE2>(e12);
        DataCopy(bL1, x2Global_[Offset], nd2nzParams);
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
    }

    __aicore__ inline void PrepareDequantParam(uint32_t kidx, uint32_t curAivM)
    {
        // x1_scale: [AivM, 8]
        // [M, K//128] -> [AivM, 8] (每行后7位为脏数据)
        uint32_t singleCoreM = block_.params_.singleCoreM;
        uint64_t offsetX1Scale = (offset_.offsetPertoken + subBlockIdx_ * ubCalcM_) * nKgroups_ + kidx;
        DataCopyExtParams copyParams{
            uint16_t(curAivM),  // block count
            4,                  // block len
            (nKgroups_ - 1) * 4, // src stride
            0,                  // dst stride
            0                   // rsv
        };

        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(MulScale, x1ScaleGlobal_[offsetX1Scale], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // [AivM, 8] * 1 = [AivM, 8] 1 N方向两次， 总计[2, AivM, 8]
        // 先乘N方向第二个scale
        // AivM * 8 / 256
        uint64_t nidx = offset_.offsetBias;
        // uint64_t offsetX2Scale = kidx * nNgroups_ + nidx / groupSizeN_; 非转置index
        uint64_t offsetX2Scale = nidx / groupSizeN_ * nKgroups_ + kidx;
        uint32_t numCols = 8;
        if (baseN_ == BASE_N_256) {
            Muls(MulScale[curAivM * numCols], MulScale, x2ScaleGlobal_.GetValue(offsetX2Scale + nKgroups_),
                curAivM * numCols);
            PipeBarrier<PIPE_V>();
        }
        Muls(MulScale, MulScale, x2ScaleGlobal_.GetValue(offsetX2Scale), curAivM * numCols);
        PipeBarrier<PIPE_V>();
        // [2, AivM, 8] -> [2, AivM, 64] 每8个datablock有7个datablock是脏数据
        Brcb(BrcbScale, MulScale, (baseN_ / BASE_N_128) * curAivM, {1, 8});
        PipeBarrier<PIPE_V>();
        return;
    }

    __aicore__ inline void DequantAndAccu(uint32_t curAivM, uint32_t curAivN)
    {
        // Cast INT32 -> Float
        // curAivM 64 curAivN 256
        PipeBarrier<PIPE_V>();
        Cast(mmOutFp32, mmOut, RoundMode::CAST_NONE, curAivM * curAivN);
        PipeBarrier<PIPE_ALL>();
        // Mul Scale
        // mmOutFp32[AivM, 256]  BrcbScale[2, AivM, 64]
        // BrcbScale[:, :8] 为有效数据，每行8个元素相同，BrcbScale[:, 8:]为脏数据
        uint8_t repeatStride = curAivN / 8;
        uint8_t src0RepStride = 8;
        // 完成mmOutFp32[：, 0:64]
        Mul(mmOutFp32, BrcbScale, mmOutFp32, NUM_ELEMENTS_PER_ITER, curAivM,
            {1, 0, 1, repeatStride, src0RepStride, repeatStride});
        // 完成mmOutFp32[：, 64:128]
        Mul(mmOutFp32[NUM_ELEMENTS_PER_ITER], BrcbScale, mmOutFp32[NUM_ELEMENTS_PER_ITER], NUM_ELEMENTS_PER_ITER, curAivM,
            {1, 0, 1, repeatStride, src0RepStride, repeatStride});
        if (baseN_ == BASE_N_256) {
            // 完成mmOutFp32[：, 128:192]
            Mul(mmOutFp32[NUM_ELEMENTS_PER_ITER * FOLD2], BrcbScale[curAivM * NUM_ELEMENTS_PER_ITER],
                mmOutFp32[NUM_ELEMENTS_PER_ITER * FOLD2], NUM_ELEMENTS_PER_ITER, curAivM,
                {1, 0, 1, repeatStride, src0RepStride, repeatStride});
            // 完成mmOutFp32[：, 192:256]
            Mul(mmOutFp32[NUM_ELEMENTS_PER_ITER * FOLD3], BrcbScale[curAivM * NUM_ELEMENTS_PER_ITER],
                mmOutFp32[NUM_ELEMENTS_PER_ITER * FOLD3], NUM_ELEMENTS_PER_ITER, curAivM,
                {1, 0, 1, repeatStride, src0RepStride, repeatStride});
        }
        // Accu
        PipeBarrier<PIPE_ALL>();
        Add(dequantAccu, mmOutFp32, dequantAccu, curAivM * curAivN);
        PipeBarrier<PIPE_V>();
        return;
    }

    __aicore__ inline void CopyInMMOut(GlobalTensor<int32_t> mmOutGM, uint32_t curAivM, uint32_t curAivN)
    {
        // share 输入输出buffer需要MTE2等MTE3 不share的话不需要等
        // curAivM * curAivN 需32B对齐
        DataCopy(mmOut, mmOutGM[subBlockIdx_ * ubCalcM_ * curAivN], curAivM * curAivN);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        return;
    }

    __aicore__ inline void CastCopyOut(uint32_t curAivM, uint32_t curAivN)
    {
        Cast(dequantOut, dequantAccu, RoundMode::CAST_RINT, curAivM * curAivN);
        PipeBarrier<PIPE_V>();
        // HardCode curAivN 256, N 32B对齐, dtypesize==2
        DataCopyParams copyParams{uint16_t(curAivM), uint16_t(curAivN / 16), // * sizeof(bfloat16) / 32B
            0, uint16_t((block_.matmulTilingData_->N - curAivN) / 16)};
        uint64_t offset = offset_.offsetC;
        offset += subBlockIdx_ * ubCalcM_ * block_.matmulTilingData_->N;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(yGlobal_[offset], dequantOut, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        return;
    }

    __aicore__ inline void MMCompute(uint32_t curAicM, uint32_t curAicN, uint32_t kidx)
    {
        // k轴尾块处理
        if (kidx % stepka_ == 0) {
            uint32_t curKa = stepka_ * groupSizeK_;
            if (kidx == loopk_ - 1) { curKa = singleCoreK_ - kidx * groupSizeK_; } // k tail
            CopyInAL1(kidx, curAicM, curKa);
        }
        if (kidx % stepkb_ == 0) {
            uint32_t curKb = stepkb_ * groupSizeK_;
            if (kidx == loopk_ - 1) { curKb = singleCoreK_ - kidx * groupSizeK_; } // k tail
            CopyInBL1(kidx, curAicN, curKb);
        }
        mmObj_.SetTensorA(aL1[kidx % stepka_ * groupSizeK_ * CeilAlign(curAicM, ALIGN_UNIT_16)], /*transa*/ transA_);
        mmObj_.SetTensorB(bL1[kidx % stepkb_ * groupSizeK_ * baseN_], /*transb*/ transB_);
        mmObj_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, groupSizeK_);
        mmObj_.Iterate(false);
        if ((kidx + 1) % stepka_ == 0) {
            SetFlag<HardEvent::MTE1_MTE2>(aL1PingFlag ? eAL1Ping12_ : eAL1Pong12_);
            aL1PingFlag = !aL1PingFlag;
        }
        if ((kidx + 1) % stepkb_ == 0) {
            SetFlag<HardEvent::MTE1_MTE2>(bL1PingFlag ? eBL1Ping21_ : eBL1Pong21_);
            bL1PingFlag = !bL1PingFlag;
        }
        CrossCoreWaitFlag(wsPingFlag ? V2C_PING_FLAG : V2C_PONG_FLAG);
        mmObj_.GetTensorC(mmOutGlobal_[wsPingFlag ? offsetWorkspaceC_ : offsetWorkspacePong_], 0);
        CrossCoreSetFlag<CV_SYNC_FLAG, PIPE_FIX>(wsPingFlag ? C2V_PING_FLAG : C2V_PONG_FLAG);
        wsPingFlag = !wsPingFlag;
    }

    __aicore__ inline void BasicMMDequantCompute(uint32_t curAicM, uint32_t curAicN, uint32_t kidx)
    {
        if ASCEND_IS_AIC { MMCompute(curAicM, curAicN, kidx); }

        if ASCEND_IS_AIV {
            // m方向不对齐的尾块处理
            // 划分给两个vector的方式
            uint32_t curAivM = GetCurAivM();
            if (curAivM > 0) {
                PrepareDequantParam(kidx, curAivM);
                CrossCoreWaitFlag(wsPingFlag ? C2V_PING_FLAG : C2V_PONG_FLAG);
                CopyInMMOut(mmOutGlobal_[wsPingFlag ? offsetWorkspaceC_ : offsetWorkspacePong_], curAivM, curAicN);
                CrossCoreSetFlag<CV_SYNC_FLAG, PIPE_MTE2>(wsPingFlag ? V2C_PING_FLAG : V2C_PONG_FLAG);
                DequantAndAccu(curAivM, curAicN);
            } else {
                CrossCoreWaitFlag(wsPingFlag ? C2V_PING_FLAG : C2V_PONG_FLAG);
                CrossCoreSetFlag<CV_SYNC_FLAG, PIPE_MTE2>(wsPingFlag ? V2C_PING_FLAG : V2C_PONG_FLAG);
            }
            wsPingFlag = !wsPingFlag;
        }
    }

    __aicore__ inline void AddBias(uint32_t curAivM, uint32_t curAivN)
    {
        uint64_t offsetBias = offset_.offsetBias;
        DataCopy(biasAccu, biasGlobal_[offsetBias], curAivN);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // dequantAccu[:, 0:64]
        uint8_t repeatStride = curAivN / 8;
        Add(dequantAccu, biasAccu, dequantAccu, NUM_ELEMENTS_PER_ITER, curAivM, {1, 1, 1, repeatStride, 0, repeatStride});
        // dequantAccu[:, 64:128]
        Add(dequantAccu[NUM_ELEMENTS_PER_ITER], biasAccu[NUM_ELEMENTS_PER_ITER], dequantAccu[NUM_ELEMENTS_PER_ITER],
            NUM_ELEMENTS_PER_ITER, curAivM, {1, 1, 1, repeatStride, 0, repeatStride});
        if (baseN_ == BASE_N_256) {
            // dequantAccu[:, 128:192]
            Add(dequantAccu[NUM_ELEMENTS_PER_ITER * FOLD2], biasAccu[NUM_ELEMENTS_PER_ITER * FOLD2],
                dequantAccu[NUM_ELEMENTS_PER_ITER * FOLD2], NUM_ELEMENTS_PER_ITER, curAivM,
                {1, 1, 1, repeatStride, 0, repeatStride});
            // dequantAccu[:, 192:256]
            Add(dequantAccu[NUM_ELEMENTS_PER_ITER * FOLD3], biasAccu[NUM_ELEMENTS_PER_ITER * FOLD3],
                dequantAccu[NUM_ELEMENTS_PER_ITER * FOLD3], NUM_ELEMENTS_PER_ITER, curAivM,
                {1, 1, 1, repeatStride, 0, repeatStride});
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    __aicore__ inline void OneTileCompute(uint64_t mTileIndex, uint64_t nTileIndex)
    {
        for (uint64_t j = 0; j < block_.realRound_; j++) {
            // 更新此次基本块的大小和输入输出地址
            update_.template UpdateBlockParamsAndCalcGmOffset<X1_FORMAT, X2_FORMAT, /*atrans*/ false, /*btrans*/ true>(
                block_.params_, offset_, mTileIndex, nTileIndex);
            if ASCEND_IS_AIV {
                uint32_t curAivM = GetCurAivM();
                if (curAivM > 0) {
                    Duplicate(dequantAccu, float(0), curAivM * ubCalcN_);
                    PipeBarrier<PIPE_V>();
                    AddBias(curAivM, ubCalcN_);
                }
            }

            for (uint64_t k = 0; k < loopk_; k++) {
                BasicMMDequantCompute(block_.params_.singleCoreM, block_.params_.singleCoreN, k);
            }

            block_.UpdateBlockIndex();
            if ASCEND_IS_AIV {
                uint32_t curAivM = GetCurAivM();
                if (curAivM > 0) { CastCopyOut(curAivM, baseN_); }
            }
        }
    }
};

template <
    typename xType, typename wType, typename biasType, typename x1scaleType, typename x2scaleType, typename yType,
    bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4Perblock<xType, wType, biasType, x1scaleType, x2scaleType, yType, weightNz>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
    GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y, GM_ADDR workspace,
    const QuantBatchMatmulV4PerblockTilingData* tilingData, TPipe* tPipe)
{
    commonInit(tPipe, &(tilingData->matmulTiling));
    groupSizeM_ = tilingData->groupSizeM;
    groupSizeN_ = tilingData->groupSizeN;
    groupSizeK_ = tilingData->groupSizeK;

    transA_ = tilingData->transA;
    transB_ = tilingData->transB;

    loopk_ = CeilDiv(tilingData->matmulTiling.singleCoreK, groupSizeK_);
    singleCoreK_ = tilingData->matmulTiling.singleCoreK;

    x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x1));
    x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ wType*>(x2));
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias));
    mmOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));
    x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x1scaleType*>(x1_scale));
    x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x2scaleType*>(x2_scale));
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));

    ubCalcM_ = tilingData->ubCalcM;
    ubCalcN_ = tilingData->ubCalcN;

    InitLocalBuffers();

    block_.Init(&(tilingData->tileL2cacheTiling), &(tilingData->matmulTiling));
    update_.template Init<X1_FORMAT, X2_FORMAT, false, true>(&tilingData->matmulTiling, block_.params_);
    nNgroups_ = CeilDiv(block_.matmulTilingData_->N, groupSizeK_);
    nKgroups_ = CeilDiv(block_.matmulTilingData_->Ka, groupSizeK_);
    mmObj_.SetSubBlockIdx(0);
    mmObj_.Init(&(tilingData->matmulTiling), pipe_);
    mmObj_.SetOrgShape(baseM_, baseN_, block_.matmulTilingData_->Ka);
}

template <
    typename xType, typename wType, typename biasType, typename x1scaleType, typename x2scaleType, typename yType,
    bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4Perblock<xType, wType, biasType, x1scaleType, x2scaleType, yType, weightNz>::InitLocalBuffers()
{
    if ASCEND_IS_AIC {
        pipe_->InitBuffer(L1Buf_, L1_MAX_SIZE_910B);
        uint32_t a_size = baseM_ * groupSizeK_ * stepka_;
        uint32_t b_size = baseN_ * groupSizeK_ * stepkb_;
        aL1Ping = L1Buf_.Get<xType>()[0];
        aL1Pong = L1Buf_.Get<xType>()[a_size];
        bL1Ping = L1Buf_.Get<xType>()[a_size * 2];
        bL1Pong = L1Buf_.Get<xType>()[a_size * 2 + b_size];
    }

    if ASCEND_IS_AIV {
        uint32_t curAivM = ubCalcM_;
        pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE); // 192KB
        mmOut = ubBuf_.Get<int32_t>()[0];         // 0-64KB
        mmOutFp32 = ubBuf_.Get<float>()[0];
        dequantOut = ubBuf_.Get<yType>()[0];
        uint64_t offset = curAivM * ubCalcN_;
        dequantAccu = ubBuf_.Get<float>()[offset]; // 64KB-128KB
        offset += curAivM * ubCalcN_;
        BrcbScale = ubBuf_.Get<float>()[offset]; // 128KB - 160KB
        offset += ubCalcN_ / groupSizeN_ * curAivM * 64;
        // BrcbScale: [2, AivM, 64] 每8个datablock有7个datablock是脏数据, 2 == BaseN//128
        MulScale = ubBuf_.Get<float>()[offset]; // 160KB - 164KB
        offset += ubCalcN_ / groupSizeN_ * curAivM * 8;
        biasAccu = ubBuf_.Get<float>()[offset]; // 164KB - 165KB
        offset += ubCalcN_;
    }
}

template <
    typename xType, typename wType, typename biasType, typename x1scaleType, typename x2scaleType, typename yType,
    bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4Perblock<xType, wType, biasType, x1scaleType, x2scaleType, yType, weightNz>::Process()
{
    if (blockIdx_ >= usedCoreNum_) { return; }
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<CV_SYNC_FLAG, PIPE_MTE2>(V2C_PING_FLAG);
        CrossCoreSetFlag<CV_SYNC_FLAG, PIPE_MTE2>(V2C_PONG_FLAG);
    }
    // 首块计算，兼容无L2cache切分场景，减少scalar计算
    // 反向flag
    if ASCEND_IS_AIC {
        initEventID();
        SetFlag<HardEvent::MTE1_MTE2>(eAL1Ping12_);
        SetFlag<HardEvent::MTE1_MTE2>(eAL1Pong12_);
        SetFlag<HardEvent::MTE1_MTE2>(eBL1Ping12_);
        SetFlag<HardEvent::MTE1_MTE2>(eBL1Pong12_);
    }

    bool reverse = true;
    block_.InitFirstTileBlockIndex();
    OneTileCompute(0, 0);

    for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            if (mTileIndex > 0 || nTileIndex > 0) { // 跳过首块
                block_.UpdateBlockCnt(mTileIndex, nTileIndex);
                block_.InitBlockIndex();
                OneTileCompute(mTileIndex, nTileIndex);
            }
        }
    }

    // 反向flag
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(eAL1Ping12_);
        WaitFlag<HardEvent::MTE1_MTE2>(eAL1Pong12_);
        WaitFlag<HardEvent::MTE1_MTE2>(eBL1Ping12_);
        WaitFlag<HardEvent::MTE1_MTE2>(eBL1Pong12_);
        CrossCoreWaitFlag(V2C_PING_FLAG);
        CrossCoreWaitFlag(V2C_PONG_FLAG);
        mmObj_.End();
        releaseEventID();
    }
}

} // namespace AscendC
#endif