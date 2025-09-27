/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file quant_batch_matmul_v4_pergroup.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_PERGROUP_H
#define QUANT_BATCH_MATMUL_V4_PERGROUP_H

#include "quant_batch_matmul_v4_common.h"

namespace AscendC {

template <typename xType, typename wType, typename biasType, typename scaleType, typename yType>
class QuantBatchMatmulV4Pergroup : public QuantBatchMatmulV4Common
{
public:
    __aicore__ inline QuantBatchMatmulV4Pergroup(){};
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
        GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y, GM_ADDR workspace,
        const QuantBatchMatmulV3TilingData* tilingData, TPipe* tPipe)
    {
        commonInit(tPipe, &(tilingData->matmulTiling));
        groupSizeK_ = tilingData->params.groupSizeK;

        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2));
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias));
        mmOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));
        x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType*>(x1_scale));
        x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType*>(x2_scale));
        x2OffsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2_offset));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));

        ubCalcM_ = tilingData->params.ubCalcM;
        ubCalcN_ = tilingData->params.ubCalcN;
        maxAivM_ = CeilDiv(baseM_, VECCORE_NUM);

        block_.Init(tilingData);
        update_.template Init<FORMAT_X1, FORMAT_X2, false, true>(&tilingData->matmulTiling, block_.params_);
        nKgroups_ = CeilDiv(block_.matmulTilingData_->Ka, groupSizeK_);
        mmObj_.SetSubBlockIdx(0);
        mmObj_.Init(&(tilingData->matmulTiling), pipe_);
        mmObj_.SetOrgShape(baseM_, baseN_, block_.matmulTilingData_->Ka);
        InitLocalBuffers();
    };

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) { return; }
        if ASCEND_IS_AIV {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            CrossCoreSetFlag<2, PIPE_MTE2>(V2C_PING_FLAG);
            CrossCoreSetFlag<2, PIPE_MTE2>(V2C_PONG_FLAG);
        }
        // 首块计算，兼容无L2cache切分场景，减少scalar计算
        // 反向flag
        if ASCEND_IS_AIC {
            initEventID();
            eAL1_12_ = eAL1Ping12_;
            eAL1_21_ = eAL1Ping21_;
            eBL1_12_ = eBL1Ping12_;
            eBL1_21_ = eBL1Ping21_;
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
        if ASCEND_IS_AIV { WaitFlag<HardEvent::MTE3_V>(EVENT_ID7); }
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            // Note: assume xtype == wtype
            pipe_->InitBuffer(L1Buf_, L1_MAX_SIZE_910B);
            uint32_t a_size = baseM_ * groupSizeK_ * stepka_;
            uint32_t b_size = baseN_ * groupSizeK_ * stepkb_;
            aL1Ping_ = L1Buf_.Get<xType>()[0];
            aL1Pong_ = aL1Ping_[a_size];
            bL1Ping_ = aL1Pong_[a_size];
            bL1Pong_ = bL1Ping_[b_size];
            aL1_ = aL1Ping_;
            bL1_ = bL1Ping_;
        }

        if ASCEND_IS_AIV {
            // mmOutFp32 和 x2scale需要避免bank冲突
            // mmOutFp32 和 dequantAccu需要避免bank冲突
            pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE); // 192KB
            uint64_t offset = 0;
            dequantAccu = ubBuf_.Get<float>()[0];
            dequantOut = ubBuf_.Get<yType>()[0];
            offset += maxAivM_ * ubCalcN_;
            mmOut = ubBuf_.Get<int32_t>()[offset]; // [ubCalcM_, ubCalcN_]
            offset += ubCalcM_ * ubCalcN_;
            // 解bank冲突： mmOutFp32 和 dequantAccu 需要隔开256B
            // offset += 64;
            mmOutFp32 = ubBuf_.Get<float>()[offset];
            offset += ubCalcM_ * ubCalcN_;
            // ubCalcN_ 32B对齐
            X2Scale = ubBuf_.Get<float>()[offset];
            offset += ubCalcN_;
            bias = ubBuf_.Get<float>()[offset];
            offset += ubCalcN_;
            X1Scale = ubBuf_.Get<float>()[offset];
            offset += CeilAlign(maxAivM_, ALIGN_UNIT_8);
            X1BrcbScale = ubBuf_.Get<float>()[offset];
            offset += (CeilAlign(maxAivM_, ALIGN_UNIT_8) * ALIGN_UNIT_8);

            X1Fp32 = ubBuf_.Get<float>()[offset];
            X1INT8 = X1Fp32.ReinterpretCast<int8_t>();
            X1INT4 = X1Fp32.ReinterpretCast<int4b_t>();
            offset += ubCalcM_ * groupSizeK_;
            X1Half = ubBuf_.Get<float>()[offset].ReinterpretCast<half>();
            offset += ubCalcM_ * groupSizeK_ >> 1;
            X1RowSum = ubBuf_.Get<float>()[offset];
            offset += CeilAlign(maxAivM_, 8);
            X1BCast = ubBuf_.Get<float>()[offset];
            offset += ubCalcM_ * ubCalcN_;
            X2Offset = ubBuf_.Get<float>()[offset];
            offset += ubCalcN_;
            X2OffsetFP16 = ubBuf_.Get<float>()[offset].ReinterpretCast<half>();
            offset += ubCalcN_ >> 1;
            X1BCastMulX2Offset = ubBuf_.Get<float>()[offset];
            offset += ubCalcM_ * ubCalcN_;
        }
    };

private:
    QBmmBlockOffset preload_offset_;

    TEventID eAL1_12_;
    TEventID eAL1_21_;
    TEventID eBL1_12_;
    TEventID eBL1_21_;

    bool aL1PingFlag_ = true;
    bool bL1PingFlag_ = true;
    bool wsPingFlag_ = true;
    bool ubPingFlag_ = true;
    bool x2ScalePingFlag_ = true;

    GlobalTensor<int8_t> x1Global_;
    GlobalTensor<int8_t> x2Global_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<int32_t> mmOutGlobal_;
    // GlobalTensor<float16_t> mmOutGlobal_;
    GlobalTensor<scaleType> x1ScaleGlobal_;
    GlobalTensor<scaleType> x2ScaleGlobal_;
    GlobalTensor<half> x2OffsetGlobal_;
    GlobalTensor<yType> yGlobal_;
    // define pingpong buf
    TBuf<TPosition::A1> L1Buf_;
    LocalTensor<xType> aL1_;
    LocalTensor<wType> bL1_;
    LocalTensor<xType> aL1Ping_;
    LocalTensor<wType> aL1Pong_;
    LocalTensor<xType> bL1Ping_;
    LocalTensor<wType> bL1Pong_;

    // Vec buffer
    // 不切N ubCalcN_ = BaseN
    // mmOut, mmOutFp32, dequantOut可共享内存
    TBuf<> ubBuf_;
    LocalTensor<int32_t> mmOut;     // [ubCalcM_, ubCalcN_]
    LocalTensor<float> mmOutFp32;   // [ubCalcM_, ubCalcN_]
    LocalTensor<float> dequantAccu; // [AivM_, ubCalcN_]
    LocalTensor<yType> dequantOut;  // [AivM_, ubCalcN_]
    LocalTensor<float> X1BrcbScale; // [AivM_, 8]
    LocalTensor<float> X1Scale;     // [AivM_]
    LocalTensor<float> X2Scale;     // [ubCalcN_] // 32B align
    LocalTensor<float> bias;        // [ubCalcN_]

    LocalTensor<int8_t> X1INT8;            // [ubCalcM_, groupsizeK]
    LocalTensor<int4b_t> X1INT4;           // [ubCalcM_, groupsizeK]
    LocalTensor<float> X2Offset;           // [ubCalcN_]
    LocalTensor<half> X2OffsetFP16;        // [ubCalcN_]
    LocalTensor<float> X1Fp32;             // [ubCalcM_, groupsizeK]
    LocalTensor<half> X1Half;              // [ubCalcM_, groupsizeK]
    LocalTensor<float> X1RowSum;           // [ubCalcM_]
    LocalTensor<float> X1BCast;            // [ubCalcM_, ubCalcN_]
    LocalTensor<float> X1BCastMulX2Offset; // [ubCalcM_, ubCalcN_]

    TEventID eX1Scale_ = EVENT_ID0;
    TEventID eX2Scale_ = EVENT_ID1;
    TEventID eMmOut_ = EVENT_ID2;
    TEventID eX2Offset_ = EVENT_ID3;
    TEventID eX1_ = EVENT_ID4;

    uint32_t maxAivM_;

    // tilingData
    using inputX1Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, xType, false>;
    using inputX2Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, wType, true>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, biasType, false>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
    MatmulImpl<inputX1Type, inputX2Type, outputYType, inputBiasType, CFG_MDL> mmObj_;

    __aicore__ inline void CopyInX1Scale(uint32_t curAivM)
    {
        // X1Scale [AivM_] // 32B align
        uint32_t offset = offset_.offsetPertoken + subBlockIdx_ * maxAivM_;
        DataCopyPad(
            X1Scale, x1ScaleGlobal_[offset],
            {/*blk_count*/ 1, /*blk_len*/ uint32_t(CeilAlign(curAivM, ALIGN_UNIT_8) * sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eX1Scale_);
    }

    __aicore__ inline void CopyInX2Scale(uint32_t kidx, uint32_t curAivN)
    {
        // X2Scale; // [ubCalcN_] // 32B align
        // x2ScaleGlobal_ [nKgroups_, n]
        uint32_t offset = kidx * block_.matmulTilingData_->N + offset_.offsetScale;
        DataCopyPad(
            X2Scale, x2ScaleGlobal_[offset],
            {/*blk_count*/ 1, /*blk_len*/ uint32_t(CeilAlign(curAivN, ALIGN_UNIT_8) * sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eX2Scale_);
    }

    __aicore__ inline void CopyInX2Offset(uint32_t kidx, uint32_t curAivN)
    {
        // X2Offset; // [ubCalcN_] // 32B align
        // x2SOffset_ [nKgroups_, n]
        uint32_t offset = kidx * block_.matmulTilingData_->N + offset_.offsetScale;
        DataCopyPad(
            X2OffsetFP16, x2OffsetGlobal_[offset],
            {/*blk_count*/ 1, /*blk_len*/ uint32_t(CeilAlign(curAivN, ALIGN_UNIT_16) * sizeof(half)), 0, 0, 0},
            {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eX2Offset_);
    }

    __aicore__ inline void CopyInX1INT8(uint32_t kidx, uint32_t midx) {
        // CopyIn X1
        // Copy INT4 as INT8
        // [m, k]
        uint32_t X1_offset = offset_.offsetA + kidx * groupSizeK_ +
                             (subBlockIdx_ * maxAivM_ + ubCalcM_ * midx) * block_.matmulTilingData_->Ka;
        uint16_t blockCount = ubCalcM_;
        uint16_t blockLen = groupSizeK_ / 2 / DATA_BLOCK_LEN;
        uint16_t srcGap = (block_.matmulTilingData_->Ka - groupSizeK_) / 2 / DATA_BLOCK_LEN;
        uint16_t dstGap = 0;
        DataCopy(X1INT8, x1Global_[X1_offset >> 1], {blockCount, blockLen, srcGap, dstGap});
        SetFlag<HardEvent::MTE2_V>(eX1_);
        WaitFlag<HardEvent::MTE2_V>(eX1_);
    }

    __aicore__ inline void X2OffsetProcess(uint32_t curAivM, uint32_t curAivN, uint32_t kidx, uint32_t midx)
    {
        CopyInX1INT8(kidx, midx);
        uint8_t dstRepeatStride = 8UL;
        uint8_t srcRepeatStride1 = 2UL;
        Cast<half, int4b_t, false>(
            X1Half, X1INT4, RoundMode::CAST_NONE, /*mask*/ 0UL, /*repeatTime*/ srcRepeatStride1 * curAivM,
            {1, 1, dstRepeatStride, srcRepeatStride1});
        PipeBarrier<PIPE_V>();
        uint8_t srcRepeatStride2 = 4UL;
        Cast<float, half, false>(
            X1Fp32, X1Half, RoundMode::CAST_NONE, /*mask*/ 0UL, /*repeatTime*/ srcRepeatStride2 * curAivM,
            {1, 1, dstRepeatStride, srcRepeatStride2});
        PipeBarrier<PIPE_V>();
        // // use X1BCastMulX2Offset as sharedTmpBuffer
        // // 文档未说明此处tmpbuffer会用多大，但已X1BCastMulX2Offset为起点的ub空间不会踩踏前面有用的空间
        // // 后面再用到X1BCastMulX2Offset时通过pipe_V隔离
        uint32_t shape[] = {curAivM, groupSizeK_};
        ReduceSum<float, Pattern::Reduce::AR, false>(
            X1RowSum, X1Fp32, /*tmp*/ X1BCastMulX2Offset.ReinterpretCast<uint8_t>(), shape, true);
        PipeBarrier<PIPE_V>();
        uint32_t dstShape_[] = {curAivM, curAivN};
        uint32_t srcShape_[] = {curAivM, 1};
        LocalTensor<uint8_t> sharedTmpBuffer = X1BCastMulX2Offset.ReinterpretCast<uint8_t>();
        constexpr int32_t dim = 2;
        constexpr int32_t axis = 1;
        Broadcast<float, dim, axis, false>(X1BCast, X1RowSum, dstShape_, srcShape_, sharedTmpBuffer);
        PipeBarrier<PIPE_V>();
        int32_t resN = curAivN;
        uint8_t repeatStride = CeilDiv(curAivN, ALIGN_UNIT_8);
        uint32_t repeatTime = 4;
        if (midx == 0) { WaitFlag<HardEvent::MTE2_V>(eX2Offset_);
            Cast<float, half, false>(X2Offset, X2OffsetFP16, RoundMode::CAST_NONE, /*mask*/ 0UL, /*repeatTime*/ repeatTime,
                {1, 1, dstRepeatStride, srcRepeatStride2});
            PipeBarrier<PIPE_V>();
        }
        while (resN >= NUM_ELEMENTS_PER_ITER) {
            Mul<float, false>(X1BCastMulX2Offset[curAivN - resN], X2Offset[curAivN - resN], X1BCast[curAivN - resN], 0UL, curAivM,
                {1, 1, 1, repeatStride, 0, repeatStride});
            resN = resN - NUM_ELEMENTS_PER_ITER;
        }
        if (resN > 0) {
            Mul<float, true>(X1BCastMulX2Offset[curAivN - resN], X2Offset[curAivN - resN], X1BCast[curAivN - resN], resN, curAivM,
                {1, 1, 1, repeatStride, 0, repeatStride});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DequantAndAccu(uint32_t curAivM, uint32_t curAivN, uint32_t kidx, uint32_t midx)
    {
        WaitFlag<HardEvent::MTE2_V>(eMmOut_);
        uint8_t dstRepeatStride = 8UL;
        uint8_t srcRepeatStride = 8UL;

        Cast<float, int32_t, false>(
            mmOutFp32, mmOut, RoundMode::CAST_NONE, /*mask*/ 0UL, /*repeatTime*/ FOLD4 * curAivM,
            {1, 1, dstRepeatStride, srcRepeatStride});
        PipeBarrier<PIPE_V>();
        X2OffsetProcess(curAivM, curAivN, kidx, midx);
        uint8_t src0RepeatStride = 8UL;
        uint8_t src1RepeatStride = 8UL;
        Sub<float, false>(
            mmOutFp32, mmOutFp32, X1BCastMulX2Offset, 0UL, FOLD4 * curAivM,
            {1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride});
        PipeBarrier<PIPE_V>();
        int32_t resN = curAivN;
        uint8_t repeatStride = CeilDiv(curAivN, ALIGN_UNIT_8);

        if (midx == 0) { WaitFlag<HardEvent::MTE2_V>(eX2Scale_); }
        while (resN >= NUM_ELEMENTS_PER_ITER) {
            Mul<float, false>(mmOutFp32[curAivN - resN], X2Scale[curAivN - resN], mmOutFp32[curAivN - resN], 0UL, curAivM,
                {1, 1, 1, repeatStride, 0, repeatStride});
            resN = resN - NUM_ELEMENTS_PER_ITER;
        }
        if (resN > 0) {
            Mul<float, true>(mmOutFp32[curAivN - resN], X2Scale[curAivN - resN], mmOutFp32[curAivN - resN], resN, curAivM,
                {1, 1, 1, repeatStride, 0, repeatStride});
        }
        PipeBarrier<PIPE_V>();
        uint32_t moffset = ubCalcM_ * midx * curAivN;
        Add<float, false>(dequantAccu[moffset], mmOutFp32, dequantAccu[moffset], 0UL, FOLD4 * curAivM,
            {1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride});
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInMMOut(uint32_t midx, uint32_t realM, uint32_t CurAicN)
    {
        GlobalTensor<int32_t> src = mmOutGlobal_[wsPingFlag_ ? offsetWorkspaceC_ : offsetWorkspacePong_];
        uint32_t moffset = (subBlockIdx_ * maxAivM_ + ubCalcM_ * midx) * CurAicN;
        src = src[moffset];
        DataCopy(mmOut, src, realM * CurAicN);
        SetFlag<HardEvent::MTE2_V>(eMmOut_);
    }

    __aicore__ inline void CastCopyOut(uint32_t curAivM, uint32_t curAivN)
    {
        WaitFlag<HardEvent::MTE2_V>(eX1Scale_);
        uint8_t dstRepeatStride = 8UL;
        Brcb(X1BrcbScale, X1Scale, CeilDiv(curAivM, ALIGN_UNIT_8), {1, dstRepeatStride});
        PipeBarrier<PIPE_V>();
        int32_t resN = curAivN;
        uint8_t repeatStride = CeilDiv(curAivN, ALIGN_UNIT_8);
        while (resN >= NUM_ELEMENTS_PER_ITER) {
            Mul(dequantAccu[curAivN - resN], X1BrcbScale, dequantAccu[curAivN - resN], NUM_ELEMENTS_PER_ITER, curAivM,
                {1, 0, 1, repeatStride, 1, repeatStride});
            resN = resN - NUM_ELEMENTS_PER_ITER;
        }
        if (resN > 0) {
            Mul(dequantAccu[curAivN - resN], X1BrcbScale, dequantAccu[curAivN - resN], resN, curAivM,
                {1, 0, 1, repeatStride, 1, repeatStride});
        }
        PipeBarrier<PIPE_V>();
        Cast(dequantOut, dequantAccu, RoundMode::CAST_RINT, curAivM * curAivN);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        uint64_t stride = DATA_BLOCK / sizeof(yType);
        DataCopyParams copyParams{uint16_t(curAivM), uint16_t(curAivN / stride), // * sizeof(bfloat16) / 32B
            0, uint16_t((block_.matmulTilingData_->N - curAivN) / stride)};
        uint64_t offset = offset_.offsetC;
        offset += subBlockIdx_ * maxAivM_ * block_.matmulTilingData_->N;
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
        DataCopy(yGlobal_[offset], dequantOut, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        return;
    }

    __aicore__ inline void shiftFlagAL1()
    {
        aL1PingFlag_ = !aL1PingFlag_;
        aL1_ = aL1PingFlag_ ? aL1Ping_ : aL1Pong_;
        eAL1_12_ = aL1PingFlag_ ? eAL1Ping12_ : eAL1Pong12_;
        eAL1_21_ = aL1PingFlag_ ? eAL1Ping21_ : eAL1Pong21_;
    }

    __aicore__ inline void shiftFlagBL1()
    {
        bL1PingFlag_ = !bL1PingFlag_;
        bL1_ = bL1PingFlag_ ? bL1Ping_ : bL1Pong_;
        eBL1_12_ = bL1PingFlag_ ? eBL1Ping12_ : eBL1Pong12_;
        eBL1_21_ = bL1PingFlag_ ? eBL1Ping21_ : eBL1Pong21_;
    }

    __aicore__ inline void CubeProcess(uint32_t CurAicM, uint32_t CurAicN, uint32_t kidx)
    {
        if (kidx % stepka_ == 0) {
            // INT4也要按照INT8搬
            WaitFlag<HardEvent::MTE1_MTE2>(eAL1_12_);
            Nd2NzParams nd2nzParamsA;
            nd2nzParamsA.ndNum = 1;
            nd2nzParamsA.srcDValue = block_.matmulTilingData_->Ka >> 1;
            nd2nzParamsA.srcNdMatrixStride = 0;
            nd2nzParamsA.dstNzNStride = 1;
            nd2nzParamsA.dstNzMatrixStride = 0;
            nd2nzParamsA.dstNzC0Stride = CeilAlign(CurAicM, ALIGN_UNIT_16);
            nd2nzParamsA.nValue = CurAicM;
            nd2nzParamsA.dValue = (stepka_ * groupSizeK_) >> 1;
            DataCopy(aL1_.template ReinterpretCast<int8_t>(), x1Global_[(offset_.offsetA + kidx * groupSizeK_) >> 1], nd2nzParamsA);
            SetFlag<HardEvent::MTE2_MTE1>(eAL1_21_);
            WaitFlag<HardEvent::MTE2_MTE1>(eAL1_21_);
        }
        if (kidx % stepkb_ == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(eBL1_12_);
            Nd2NzParams nd2nzParamsB;
            nd2nzParamsB.ndNum = 1;
            nd2nzParamsB.srcDValue = block_.matmulTilingData_->Ka >> 1;
            nd2nzParamsB.srcNdMatrixStride = 0;
            nd2nzParamsB.dstNzNStride = 1;
            nd2nzParamsB.dstNzMatrixStride = 0;
            nd2nzParamsB.dstNzC0Stride = CeilAlign(CurAicN, ALIGN_UNIT_16);
            nd2nzParamsB.nValue = CurAicN;
            nd2nzParamsB.dValue = (stepkb_ * groupSizeK_) >> 1;
            DataCopy(bL1_.template ReinterpretCast<int8_t>(), x2Global_[(offset_.offsetB + kidx * groupSizeK_) >> 1], nd2nzParamsB);
            SetFlag<HardEvent::MTE2_MTE1>(eBL1_21_);
            WaitFlag<HardEvent::MTE2_MTE1>(eBL1_21_);
        }

        mmObj_.SetTensorA(aL1_[kidx % stepka_ * groupSizeK_ * CeilAlign(CurAicM, ALIGN_UNIT_16)], /*transa*/ false);
        mmObj_.SetTensorB(bL1_[kidx % stepkb_ * groupSizeK_ * baseN_], /*transb*/ true);

        mmObj_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, groupSizeK_);
        mmObj_.Iterate(false);
        if ((kidx + 1) % stepka_ == 0) {
            SetFlag<HardEvent::MTE1_MTE2>(eAL1_12_);
            shiftFlagAL1();
        }
        if ((kidx + 1) % stepkb_ == 0) {
            SetFlag<HardEvent::MTE1_MTE2>(eBL1_12_);
            shiftFlagBL1();
        }
        CrossCoreWaitFlag(V2C_PING_FLAG | wsPingFlag_);
        mmObj_.GetTensorC(mmOutGlobal_[wsPingFlag_ ? offsetWorkspaceC_ : offsetWorkspacePong_], 0, true);
        CrossCoreSetFlag<CV_SYNC_FLAG, PIPE_FIX>(C2V_PING_FLAG | wsPingFlag_);
    }

    __aicore__ inline void BasicMMDequantCompute(uint32_t CurAicM, uint32_t CurAicN, uint32_t kidx, int32_t CurAivM)
    {
        if ASCEND_IS_AIC { CubeProcess(CurAicM, CurAicN, kidx); }
        if ASCEND_IS_AIV {
            if (CurAivM > 0) {
                uint64_t mloop = CeilDiv(CurAivM, ubCalcM_);
                uint32_t tailM = CurAivM - ubCalcM_ * (mloop - 1);
                for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mloop; ++mUbLoopIdx) {
                    if (mUbLoopIdx == 0) {
                        CrossCoreWaitFlag(C2V_PING_FLAG | wsPingFlag_);
                    }
                    CopyInMMOut(mUbLoopIdx, mloop == 1 ? tailM : ubCalcM_, CurAicN);
                    if (mUbLoopIdx == mloop - 1) {
                        CrossCoreSetFlag<2, PIPE_MTE2>(V2C_PING_FLAG | wsPingFlag_);
                    }
                    DequantAndAccu(mUbLoopIdx == mloop - 1 ? tailM : ubCalcM_, CurAicN, kidx, mUbLoopIdx);
                }
            } else {
                CrossCoreWaitFlag(C2V_PING_FLAG | wsPingFlag_);
                CrossCoreSetFlag<2, PIPE_MTE2>(V2C_PING_FLAG | wsPingFlag_);
            }
        }
        wsPingFlag_ = !wsPingFlag_;
    }

    __aicore__ inline uint32_t GetCurAivM()
    { // 计算m轴尾块
        uint32_t curAivM = maxAivM_;
        if (block_.params_.singleCoreM != baseM_) {
            if (subBlockIdx_ == 0) {
                curAivM = block_.params_.singleCoreM > curAivM ? curAivM : block_.params_.singleCoreM;
            } else {
                curAivM = block_.params_.singleCoreM > curAivM ? block_.params_.singleCoreM - curAivM : 0UL;
            }
        }
        return curAivM;
    }

    __aicore__ inline void OneTileCompute(uint64_t mTileIndex, uint64_t nTileIndex)
    {
        for (uint64_t j = 0; j < block_.realRound_; j++) {
            // 更新此次基本块的大小和输入输出地址
            update_.template UpdateBlockParamsAndCalcGmOffset<FORMAT_X1, FORMAT_X2, /*atrans*/ false, /*btrans*/ true>(
                block_.params_, offset_, mTileIndex, nTileIndex);
            uint32_t aivM = GetCurAivM();
            if ASCEND_IS_AIV {
                if (aivM > 0) {
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
                    Duplicate(dequantAccu, float(0), aivM * ubCalcN_);
                    PipeBarrier<PIPE_V>();
                    CopyInX1Scale(aivM);
                }
            }
            for (uint64_t k = 0; k < nKgroups_; k++) {
                if (aivM > 0) {
                    CopyInX2Scale(k, block_.params_.singleCoreN);
                    CopyInX2Offset(k, block_.params_.singleCoreN);
                }
                BasicMMDequantCompute(block_.params_.singleCoreM, block_.params_.singleCoreN, k, aivM);
            }
            block_.UpdateBlockIndex();
            if ASCEND_IS_AIV {
                if (aivM > 0) { CastCopyOut(aivM, block_.params_.singleCoreN); }
            }
        }
    }
};

} // namespace AscendC
#endif