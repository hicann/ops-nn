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
 * \file flat_quant.h
 * \brief
 */
#ifndef FLAT_QUANT_H
#define FLAT_QUANT_H

#include <cmath>
#include "tensor_utils.h"
#include "lib/matmul_intf.h"

namespace FlatQuantNS {

template <typename T>
class TestVec {
public:
    aifunc TestVec(){}
    aifunc void Init(GM_ADDR xmtx_, GM_ADDR p1mtx_, GM_ADDR p2mtx_, GM_ADDR out_, GM_ADDR qscale_, GM_ADDR workspace_, const FlatQuantTilingData* tilingData){
        // load tiling info and do in-core tiling
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        shape.K = tilingData->K;
        shape.clipRatio = tilingData->clipRatio;
        tiling();

        // assign x and out address to tensors
        xGM.SetGlobalBuffer((__gm__ T *)xmtx_);
        p1GM.SetGlobalBuffer((__gm__ T *)p1mtx_);
        p2GM.SetGlobalBuffer((__gm__ T *)p2mtx_);
        outGM.SetGlobalBuffer((__gm__ int4b_t *)out_);
        qscaleGM.SetGlobalBuffer((__gm__ float *)qscale_);

        // assign workspace
        int offset = 0;
        xnzGM.SetGlobalBuffer((__gm__ T *)workspace_ + offset);
        offset += (shape.Mceil * shape.Nceil * shape.K);
        outnzGM.SetGlobalBuffer((__gm__ half *)workspace_ + offset);
        offset += (shape.Mceil * shape.Nceil * shape.K);
        p1nzGM.SetGlobalBuffer((__gm__ T *)workspace_ + offset);
        offset += (shape.Mceil * shape.Mceil);
        p2nzGM.SetGlobalBuffer((__gm__ T *)workspace_ + offset);

        // we assume maximum shape 128 128
        // 0~64kb, input area
        pipe.InitBuffer(bufQueue, NUM_ONE_NINE_TWO * NUM_ONE_ZERO_TWO_FOUR);
        LocalTensor<uint8_t> tmp = bufQueue.Get<uint8_t>();
        xTensor = tmp.ReinterpretCast<T>();
        x2Tensor = tmp[shape.Mceil * shape.Nceil * sizeof(T)].ReinterpretCast<T>();
        p1Tensor = tmp.ReinterpretCast<T>();

        // 64~128kb, output area
        offset = NUM_SIX_FOUR * NUM_ONE_ZERO_TWO_FOUR;
        xnzTensor = tmp[offset].ReinterpretCast<T>();
        xnz2Tensor = tmp[offset + shape.Mceil * shape.Nceil * sizeof(T)].ReinterpretCast<T>();
        offset = NUM_SIX_FOUR * NUM_ONE_ZERO_TWO_FOUR;
        p1nzTensor = tmp[offset].ReinterpretCast<T>();

        // >128kb: other buffers
        offset = NUM_ONE_TWO_EIGHT * NUM_ONE_ZERO_TWO_FOUR;
        qscaleTensor = tmp[offset].ReinterpretCast<half>();
        offset += NUM_FOUR * NUM_FIVE_ONE_TWO * sizeof(half);
        maxTensor = tmp[offset].ReinterpretCast<half>();
        offset = NUM_ONE_FOUR_FOUR * NUM_ONE_ZERO_TWO_FOUR;
        absTensor = tmp[offset].ReinterpretCast<half>();
        floatTensor = tmp[offset].ReinterpretCast<float>();

        eventIdVS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        // pre-set all buffers to zero
        Duplicate<T>(xTensor, (T)0, shape.Mceil * shape.Nceil);
        Duplicate<T>(x2Tensor, (T)0, shape.Mceil * shape.Nceil);
        Duplicate<half>(absTensor, (half)0, shape.Mceil * shape.Nceil);
        Duplicate<half>(maxTensor, (half)0, NUM_ONE_TWO_EIGHT);
        Duplicate<half>(qscaleTensor, (half)0, NUM_ONE_TWO_EIGHT);
    }

    aifunc void tiling(){
        // split K
        int k_per_core = ((shape.K + GetBlockNum() - 1) / GetBlockNum() + K_PER_VEC - 1) / (K_PER_VEC) * (K_PER_VEC);
        shape.K1 = k_per_core * (GetBlockIdx() / NUM_TWO);  // vector blk idx is 0~40
        shape.K2 = k_per_core + shape.K1;
        if (shape.K2 > shape.K) {
            shape.K2 = shape.K;
        }
        shape.Mceil = (shape.M + NUM_ONE_SIX - 1) / NUM_ONE_SIX * NUM_ONE_SIX;
        shape.Nceil = (shape.N + NUM_ONE_SIX - 1) / NUM_ONE_SIX * NUM_ONE_SIX;
        // We will distribute P1 and P2 matrix pre-process to all vectors
        if (NUM_ONE_SIX * GetBlockIdx() >= (shape.Mceil + shape.Nceil)) {
            shape.procP1 = -1;
            shape.pstart = 0;
        } else if (NUM_ONE_SIX * GetBlockIdx() >= shape.Mceil) {
            shape.procP1 = 0;
            shape.pstart = NUM_ONE_SIX * GetBlockIdx() - shape.Mceil;
            shape.prows = (shape.pstart > shape.N - NUM_ONE_SIX) ? (shape.N - shape.pstart) : NUM_ONE_SIX;
        } else {
            shape.procP1 = 1;
            shape.pstart = NUM_ONE_SIX * GetBlockIdx();
            shape.prows = (shape.pstart > shape.M - NUM_ONE_SIX) ? (shape.M - shape.pstart) : NUM_ONE_SIX;
        }
    }

    aifunc void Process(){
        ProcessP1P2();
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x0, PIPE_MTE3>(NUM_FIVE);
        CrossCoreWaitFlag(NUM_FIVE);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(NUM_FOUR);

        ProcessXQuant();
    }

    aifunc void ProcessP1P2(){
        // process P1 or P2 according to tiling
        if (shape.procP1 == 0) {
            ProcessP2();
        } else if (shape.procP1 == 1) {
            ProcessP1();
        }
    }

    aifunc void ProcessP1(){
        if (shape.M == shape.Mceil) {
            DataCopy(p1Tensor, p1GM[shape.pstart * shape.M], shape.prows * shape.M);
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(shape.prows), static_cast<uint32_t>(shape.M * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(shape.Mceil - shape.M), 0};
            DataCopyPad(p1Tensor, p1GM[shape.pstart * shape.M], copyParams, padParams);
        }

        SetFlag<HardEvent::MTE2_V>((event_t)NUM_FIVE);
        WaitFlag<HardEvent::MTE2_V>((event_t)NUM_FIVE);
        // nd2zz
        for (int j = 0; j < shape.Mceil / NUM_ONE_SIX; ++j) {
            Copy(p1nzTensor[j * NUM_ONE_SIX * NUM_ONE_SIX], p1Tensor[j * NUM_ONE_SIX], NUM_ONE_SIX, NUM_ONE_SIX,
                 {1, 1, 1, static_cast<uint16_t>(shape.Mceil / NUM_ONE_SIX)});
        }
        SetFlag<HardEvent::V_MTE3>((event_t)NUM_FIVE);
        WaitFlag<HardEvent::V_MTE3>((event_t)NUM_FIVE);
        DataCopy(p1nzGM[shape.pstart * shape.Mceil], p1nzTensor, NUM_ONE_SIX * shape.Mceil);
    }

    aifunc void ProcessP2(){
        if (shape.N == shape.Nceil) {
            DataCopy(p1Tensor, p2GM[shape.pstart * shape.N], shape.prows * shape.N);
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(shape.prows), static_cast<uint32_t>(shape.N * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(shape.Nceil - shape.N), 0};
            DataCopyPad(p1Tensor, p2GM[shape.pstart * shape.N], copyParams, padParams);
        }

        SetFlag<HardEvent::MTE2_V>((event_t)NUM_FIVE);
        WaitFlag<HardEvent::MTE2_V>((event_t)NUM_FIVE);
        // nd2zz
        for (int j = 0; j < shape.Nceil / NUM_ONE_SIX; ++j) {
            Copy(p1nzTensor[j * NUM_ONE_SIX * NUM_ONE_SIX], p1Tensor[j * NUM_ONE_SIX], NUM_ONE_SIX, NUM_ONE_SIX,
                 {1, 1, 1, static_cast<uint16_t>(shape.Nceil / NUM_ONE_SIX)});
        }
        SetFlag<HardEvent::V_MTE3>((event_t)NUM_FIVE);
        WaitFlag<HardEvent::V_MTE3>((event_t)NUM_FIVE);
        DataCopy(p2nzGM[shape.pstart * shape.Nceil], p1nzTensor, NUM_ONE_SIX * shape.Nceil);
    }

    aifunc void ProcessXQuant(){
        in_empty.setall();
        out_empty.setall();

        int delay_cnt = 0;
        int kk = shape.K1;
        int k_tail_start = shape.K1;  // process output should delay several iterations

        // process X (nd2zn) and quant output
        for (; kk < shape.K2; kk += K_PER_VEC) {
            int end_k = ((kk + K_PER_VEC) > shape.K2) ? shape.K2 : (kk + K_PER_VEC);

            for (int k = kk; k < end_k; ++k) {
                if (GetSubBlockIdx() == 0) {
                    ProcessX(k);
                }
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(NUM_FOUR);

            // delay N_PRELOAD iterations
            if (delay_cnt < N_PRELOAD) {
                delay_cnt++;
            } else {
                // OUTNZ_WAIT for vec1 is inside Quant function, to ensure synchronization
                if (GetSubBlockIdx() == 0){
                    CrossCoreWaitFlag(0);
                }
                for (int k = k_tail_start; k < kk - (N_PRELOAD - 1) * K_PER_VEC; ++k) {
                    if (GetSubBlockIdx() == 1)
                        Quant(k);
                }
                k_tail_start += K_PER_VEC;
            }
        }

        // Quant the delayed batches
        for (int i = k_tail_start; i < shape.K2; i += K_PER_VEC) {
            int end_k = ((i + K_PER_VEC) > shape.K2) ? shape.K2 : (i + K_PER_VEC);
            if (GetSubBlockIdx() == 0){
                CrossCoreWaitFlag(0);
            }
            for (int k = i; k < end_k; ++k) {
                if (GetSubBlockIdx() == 1)
                    Quant(k);
            }
        }

        in_empty.release();
        out_empty.release();
        // write quant scales to GM after finishing computation
        CopyOutQuant();
    }

    aifunc void CopyOutQuant(){
        if ((shape.K1 < shape.K2) && (GetSubBlockIdx() == 1)) {
            Cast(floatTensor, qscaleTensor, RoundMode::CAST_NONE, shape.K2 - shape.K1);
            PipeBarrier<PIPE_V>();
            Muls(floatTensor, floatTensor, shape.clipRatio / NUM_FLOAT_SEVEN, shape.K2 - shape.K1);
            SetFlag<HardEvent::V_MTE3>((event_t)0);
            WaitFlag<HardEvent::V_MTE3>((event_t)0);
            if ((shape.K2 - shape.K1) % NUM_EIGHT == 0) {
                DataCopy(qscaleGM[shape.K1], floatTensor, shape.K2 - shape.K1);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>((shape.K2 - shape.K1) * sizeof(float)), 0, 0, 0};
                DataCopyPad(qscaleGM[shape.K1], floatTensor, copyParams);
            }
        }
    }

    aifunc void ProcessX(int k){
        in_empty.wait();
        if (shape.N == shape.Nceil) {
            DataCopy(GetXTensor(k), xGM[k * shape.M * shape.N], shape.M * shape.N);
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(shape.M), static_cast<uint32_t>(shape.N * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(shape.Nceil - shape.N), 0};
            DataCopyPad(GetXTensor(k), xGM[k * shape.M * shape.N], copyParams, padParams);
        }
        in_ready.set();

        out_empty.wait();
        in_ready.wait();
        // x should be ZN (mnmn)
        for (int i = 0; i < shape.Mceil / NUM_ONE_SIX; ++i) {
            for (int j = 0; j < shape.Nceil / NUM_ONE_SIX; ++j) {
                Copy(GetXNZTensor(k)[(i * (shape.Nceil / NUM_ONE_SIX) + j) * NUM_ONE_SIX * NUM_ONE_SIX],
                     GetXTensor(k)[i * NUM_ONE_SIX * shape.Nceil + j * NUM_ONE_SIX], NUM_ONE_SIX, NUM_ONE_SIX,
                     {1, 1, 1, static_cast<uint16_t>(shape.Nceil / NUM_ONE_SIX)});
            }
        }
        in_empty.set();
        out_ready.set();

        out_ready.wait();
        DataCopy(xnzGM[k * shape.Mceil * shape.Nceil], GetXNZTensor(k), shape.Mceil * shape.Nceil);
        out_empty.set();
    }

    aifunc void Quant(int k){
        if (k == shape.K1) {
            CrossCoreWaitFlag(0);
            in_empty.wait();
            DataCopy(GetXTensor(k).template ReinterpretCast<half>(), outnzGM[k * shape.Mceil * shape.Nceil], shape.Mceil * shape.Nceil);
            in_ready.set();
        }
        if (k < shape.K2 - 1) {
            if ((k + 1) % K_PER_VEC == 0){
                CrossCoreWaitFlag(0);
            }
            in_empty.wait();
            DataCopy(GetXTensor(k + 1).template ReinterpretCast<half>(), outnzGM[(k + 1) * shape.Mceil * shape.Nceil], shape.Mceil * shape.Nceil);
            in_ready.set();
        }

        // Quant  (abs -> max -> 7/max -> x*(7/max))
        out_empty.wait();
        in_ready.wait();

        int64_t realCount = static_cast<int64_t>(shape.M) * static_cast<int64_t>(shape.N);
        Abs(absTensor, GetXTensor(k).template ReinterpretCast<half>(), realCount);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(maxTensor, absTensor, NUM_ONE_TWO_EIGHT, shape.Mceil * shape.Nceil/NUM_ONE_TWO_EIGHT, 1, 1, NUM_EIGHT, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(qscaleTensor[k - shape.K1], maxTensor, NUM_ONE_TWO_EIGHT, 1, 1, 1, NUM_EIGHT, ReduceOrder::ORDER_ONLY_VALUE);
        SetFlag<HardEvent::V_S>(eventIdVS);
        WaitFlag<HardEvent::V_S>(eventIdVS);
        float maxValue = static_cast<float>(qscaleTensor.GetValue(k - shape.K1));

        float maxValueFloat = maxValue != 0 ? ((NUM_FLOAT_SEVEN / shape.clipRatio) / maxValue) : NUM_FLOAT_SEVEN;
        LocalTensor<int4b_t> outTensor = GetXNZTensor(k).template ReinterpretCast<int4b_t>();
        LocalTensor<half> outTensorHalf = GetXNZTensor(k).template ReinterpretCast<half>();
        Muls(outTensorHalf, GetXTensor(k).template ReinterpretCast<half>(), static_cast<half>(maxValueFloat), realCount);
        PipeBarrier<PIPE_V>();
        Cast(outTensor, outTensorHalf, RoundMode::CAST_NONE, realCount);

        out_ready.set();
        in_empty.set();

        out_ready.wait();
        DataCopyExtParams copyParams{1, (uint32_t)realCount / NUM_TWO, 0, 0, 0};
        DataCopyPad(outGM[k * shape.M * shape.N], outTensor, copyParams);
        out_empty.set();
    }

    __aicore__ inline LocalTensor<T> GetXTensor(int64_t k){
        return (k % NUM_TWO == 0) ? xTensor : x2Tensor;
    };

    __aicore__ inline LocalTensor<T> GetXNZTensor(int64_t k){
        return (k % NUM_TWO == 0) ? xnzTensor : xnz2Tensor;
    };

private:
    TPipe pipe;
    FlatQuantShapeInfo shape;
    GlobalTensor<T> xGM;
    GlobalTensor<T> p1GM;
    GlobalTensor<T> p2GM;
    GlobalTensor<T> xnzGM;
    GlobalTensor<T> p1nzGM;
    GlobalTensor<T> p2nzGM;
    GlobalTensor<half> outnzGM;
    GlobalTensor<float> qscaleGM;
    GlobalTensor<int4b_t> outGM;

    TBuf<QuePosition::VECCALC> bufQueue;
    LocalTensor<T> xTensor;
    LocalTensor<T> x2Tensor;
    LocalTensor<T> p1Tensor;
    LocalTensor<T> xnzTensor;
    LocalTensor<T> xnz2Tensor;
    LocalTensor<T> p1nzTensor;

    LocalTensor<half> maxTensor;
    LocalTensor<half> qscaleTensor;
    LocalTensor<half> absTensor;

    LocalTensor<float> floatTensor;

    event_t eventIdVS;

    DEvent<PIPE_MTE2, PIPE_V> in_ready{NUM_FOUR, NUM_FIVE};
    DEvent<PIPE_V, PIPE_MTE2> in_empty{NUM_FOUR, NUM_FIVE};
    DEvent<PIPE_V, PIPE_MTE3> out_ready{NUM_FOUR, NUM_FIVE};
    DEvent<PIPE_MTE3, PIPE_V> out_empty{NUM_FOUR, NUM_FIVE};
};

template <typename T>
class TestCube {
public:
    aifunc TestCube(){}
    aifunc void Init(GM_ADDR workspace_, const FlatQuantTilingData* tilingData){
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        shape.K = tilingData->K;
        tiling();

        int offset = 0;
        xnzGM.SetGlobalBuffer((__gm__ T *)workspace_ + offset);
        offset += shape.Mceil * shape.Nceil * shape.K;
        outnzGM.SetGlobalBuffer((__gm__ half *)workspace_ + offset);
        offset += shape.Mceil * shape.Nceil * shape.K;
        p1nzGM.SetGlobalBuffer((__gm__ T *)workspace_ + offset);
        offset += shape.Mceil * shape.Mceil;
        p2nzGM.SetGlobalBuffer((__gm__ T *)workspace_ + offset);

        pipe.InitBuffer(l1Buf, NUM_TWO_ZERO_EIGHT * NUM_ONE_ZERO_TWO_FOUR * sizeof(T));
        LocalTensor<T> tmpL1 = l1Buf.Get<T>();
        offset = 0;
        l1xpongTensor = tmpL1[offset];
        l1x2pongTensor = tmpL1[offset + shape.Mceil * shape.Nceil];

        offset = NUM_SIX_FOUR * NUM_ONE_ZERO_TWO_FOUR;
        l1p1Tensor = tmpL1[offset];
        l1p2Tensor = tmpL1[offset + shape.Mceil * shape.Mceil];

        offset = NUM_ONE_FOUR_FOUR * NUM_ONE_ZERO_TWO_FOUR;
        l1xpingTensor = tmpL1[offset];
        l1x2pingTensor = tmpL1[offset + shape.Mceil * shape.Nceil];

        pipe.InitBuffer(l0aBuf, NUM_TWO * NUM_THREE_TWO * NUM_FIVE_ONE_TWO * sizeof(T));
        l0apongTensor = l0aBuf.Get<T>();
        l0apingTensor = l0apongTensor[NUM_THREE_TWO * NUM_FIVE_ONE_TWO];

        pipe.InitBuffer(l0bBuf, NUM_TWO * NUM_THREE_TWO * NUM_FIVE_ONE_TWO * sizeof(T));
        l0bpongTensor = l0bBuf.Get<T>();
        l0bpingTensor = l0bpongTensor[NUM_THREE_TWO * NUM_FIVE_ONE_TWO];

        pipe.InitBuffer(l0cBuf, NUM_TWO * NUM_THREE_TWO * NUM_FIVE_ONE_TWO * sizeof(float));
        l0cpongTensor = l0cBuf.Get<float>();
        l0cpingTensor = l0cpongTensor[NUM_THREE_TWO * NUM_FIVE_ONE_TWO];
    }

    aifunc void tiling(){
        int k_per_core = ((shape.K + GetBlockNum() - 1) / GetBlockNum() + K_PER_VEC - 1) / (K_PER_VEC) * (K_PER_VEC);
        shape.K1 = k_per_core * (GetBlockIdx());  // cube blk idx is 0~20
        shape.K2 = ((k_per_core + shape.K1) > shape.K) ? shape.K : (k_per_core + shape.K1);
        shape.Mceil = (shape.M + NUM_ONE_SIX - 1) / NUM_ONE_SIX * NUM_ONE_SIX;
        shape.Nceil = (shape.N + NUM_ONE_SIX - 1) / NUM_ONE_SIX * NUM_ONE_SIX;
    }

    aifunc void Process(){
        l1empty.setall();
        l0empty.setall();
        outempty.setall();

        // pre-load p1 and p2
        CrossCoreWaitFlag(NUM_FOUR);
        DataCopy(l1p1Tensor, p1nzGM, DataCopyParams(1, shape.Mceil * shape.Mceil / NUM_ONE_SIX, 0, 0));
        DataCopy(l1p2Tensor, p2nzGM, DataCopyParams(1, shape.Nceil * shape.Nceil / NUM_ONE_SIX, 0, 0));

        // process K_PER_VEC in each iter
        for (int kk = shape.K1; kk < shape.K2; kk += K_PER_VEC) {
            int endk = ((kk + K_PER_VEC) > shape.K2) ? shape.K2 : (kk + K_PER_VEC);

            for (int k = kk; k < endk; ++k) {
                // the following is the safe version of CrossCoreFlag, but slightly slower
                if (((k < shape.K2 - 1) && (k + 1) % K_PER_VEC == 0) || (k == shape.K1)){
                    CrossCoreWaitFlag(NUM_FOUR);
                }
                ProcessK(k);
            }

            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        }

        l1empty.release();
        l0empty.release();
        outempty.release();
    }

    aifunc void ProcessK(int k){
        if (k == shape.K1) {
            ProcessXP2(k);
        }
        if (k < shape.K2 - 1) {
            ProcessXP2(k + 1);
        }
        ProcessP1X(k);
    }

    aifunc void ProcessXP2(int k){
        l1empty.wait();
        DataCopy(GetL1XTensor(k), xnzGM[k * shape.Mceil * shape.Nceil], DataCopyParams(1, shape.Mceil * shape.Nceil / NUM_ONE_SIX, 0, 0));
        l1ready.set();

        l0empty.wait();
        l1ready.wait();
        LoadData(GetL0aTensor(k), GetL1XTensor(k), LoadData2DParams(0, shape.Mceil * shape.Nceil / NUM_ONE_SIX / NUM_ONE_SIX, 1, 0, 0, false, 0));
        LoadData(GetL0bTensor(k), l1p2Tensor, LoadData2DParams(0, shape.Nceil * shape.Nceil / NUM_ONE_SIX / NUM_ONE_SIX, 1, 0, 0, true, 0));
        l0ready.set();
        l1empty.set();

        outempty.wait();
        l0ready.wait();
        CalMatrix(GetL0cTensor(k), GetL0aTensor(k), GetL0bTensor(k), shape.Mceil, shape.Nceil, shape.Nceil, UFMode3, false, false, true);

        QuantMode_t quantMode = F322F16;
        if constexpr(std::is_same<T, bfloat16_t>::value) {
            quantMode = F322BF16;
        }
        DataCopyCO12DstParams dataCopyParams(shape.Mceil, shape.Nceil, shape.Nceil, shape.Nceil, quantMode, 0, false, false);
        dataCopyParams.unitFlag = UFMode3;
        DataCopy(GetL1X2Tensor(k), GetL0cTensor(k), dataCopyParams);
        x2ready.set();
    }

    aifunc void ProcessP1X(int k){
        x2ready.wait();
        LoadData(GetL0aTensor(k), l1p1Tensor, LoadData2DParams(0, shape.Mceil * shape.Mceil / NUM_ONE_SIX / NUM_ONE_SIX, 1, 0, 0, false, 0));
        // NZ2ZN
        for (int mblk = 0; mblk < shape.Mceil / NUM_ONE_SIX; ++mblk) {
            LoadData(GetL0bTensor(k)[mblk * shape.Nceil * NUM_ONE_SIX], GetL1X2Tensor(k)[mblk * NUM_ONE_SIX * NUM_ONE_SIX], LoadData2DParams(0, shape.Nceil / NUM_ONE_SIX, shape.Mceil / NUM_ONE_SIX, 0, 0, true, 0));
        }
        x2p1ready.set();

        x2p1ready.wait();
        CalMatrix(GetL0cTensor(k), GetL0aTensor(k), GetL0bTensor(k), shape.M, shape.M, shape.N, UFMode3, false, false, true);
        l0empty.set();

        DataCopyCO12DstParams dataCopyParams(shape.N, shape.Mceil, shape.N, shape.Mceil, F322F16, 0, false, true);
        dataCopyParams.unitFlag = UFMode3;
        DataCopy(outnzGM[k * shape.Mceil * shape.Nceil], GetL0cTensor(k), dataCopyParams);
        outempty.set();
    }

    __aicore__ inline LocalTensor<T> GetL1XTensor(int64_t k){
        return (k % NUM_TWO == 0) ? l1xpongTensor : l1xpingTensor;
    };

    __aicore__ inline LocalTensor<T> GetL1X2Tensor(int64_t k){
        return (k % NUM_TWO == 0) ? l1x2pongTensor : l1x2pingTensor;
    };

    __aicore__ inline LocalTensor<T> GetL0aTensor(int64_t k){
        return (k % NUM_TWO == 0) ? l0apongTensor : l0apingTensor;
    };

    __aicore__ inline LocalTensor<T> GetL0bTensor(int64_t k){
        return (k % NUM_TWO == 0) ? l0bpongTensor : l0bpingTensor;
    };

    __aicore__ inline LocalTensor<float> GetL0cTensor(int64_t k){
        return (k % NUM_TWO == 0) ? l0cpongTensor : l0cpingTensor;
    };

private:
    TPipe pipe;
    FlatQuantShapeInfo shape;
    GlobalTensor<T> xnzGM;
    GlobalTensor<T> p1nzGM;
    GlobalTensor<T> p2nzGM;
    GlobalTensor<half> outnzGM;
    TBuf<TPosition::A1> l1Buf;
    TBuf<TPosition::A2> l0aBuf;
    TBuf<TPosition::B2> l0bBuf;
    TBuf<TPosition::C2> l0cBuf;
    LocalTensor<T> l1xpongTensor;
    LocalTensor<T> l1xpingTensor;
    LocalTensor<T> l1x2pongTensor;
    LocalTensor<T> l1x2pingTensor;
    LocalTensor<T> l1p1Tensor;
    LocalTensor<T> l1p2Tensor;
    LocalTensor<T> l0apongTensor;
    LocalTensor<T> l0apingTensor;
    LocalTensor<T> l0bpongTensor;
    LocalTensor<T> l0bpingTensor;
    LocalTensor<float> l0cpongTensor;
    LocalTensor<float> l0cpingTensor;

    DEvent<PIPE_MTE2, PIPE_MTE1> l1ready{NUM_FOUR, NUM_FIVE};
    DEvent<PIPE_MTE1, PIPE_MTE2> l1empty{NUM_FOUR, NUM_FIVE};
    DEvent<PIPE_MTE1, PIPE_M> l0ready{NUM_FOUR, NUM_FIVE};
    DEvent<PIPE_M, PIPE_MTE1> l0empty{NUM_FOUR, NUM_FIVE};

    DEvent<(pipe_t)NUM_TEN, PIPE_MTE1> x2ready{NUM_FOUR, NUM_FIVE};
    DEvent<PIPE_MTE1, PIPE_M> x2p1ready{NUM_TWO, NUM_THREE};
    DEvent<(pipe_t)NUM_TEN, PIPE_M> outempty{NUM_FOUR, NUM_FIVE};
};
}  // namespace FlatQuantNS

#endif  // FLAT_QUANT_H