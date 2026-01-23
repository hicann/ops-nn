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
 * \file flat_quant_cube_one.h
 * \brief
 */
#ifndef FLAT_QUANT_CUBE_ONE_H
#define FLAT_QUANT_CUBE_ONE_H

#include <cmath>
#include "tensor_utils.h"

namespace FlatQuantNS {
template <typename T, uint8_t MM_MODE>
class FlatQuantCubeOne {
public:
    aifunc FlatQuantCubeOne(){}
    aifunc void Init(GM_ADDR xmtx_, GM_ADDR p1mtx_, GM_ADDR p2mtx_, GM_ADDR workspace_, const FlatQuantTilingData* tilingData){
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        shape.K = tilingData->K;
        tiling();

        xGM.SetGlobalBuffer((__gm__ T *)xmtx_);
        p1GM.SetGlobalBuffer((__gm__ T *)p1mtx_);
        p2GM.SetGlobalBuffer((__gm__ T *)p2mtx_);
        outnzGM.SetGlobalBuffer((__gm__ T *)workspace_);
        doubleP1GM.SetGlobalBuffer((__gm__ T *)(workspace_ + (shape.K + shape.K % 2) * shape.Mceil * shape.N * sizeof(T)));

        SetFixpipeNz2ndFlag(1, 1, shape.Nceil);
        pipe.InitBuffer(l1Buf, L1_SIZE);
        xTensor = l1Buf.Get<T>();
        x2Tensor = xTensor[shape.calM * shape.Nceil];
        yTensor = x2Tensor[shape.calM * shape.Nceil];
        y2Tensor = yTensor[shape.calM * shape.Nceil];
        p1Tensor = y2Tensor[shape.calM * shape.Nceil];
        p2Tensor = p1Tensor[shape.calM * shape.calM];

        pipe.InitBuffer(l0aBuf, DOUBLE * DATA_COUNT * sizeof(T));
        aTensor = l0aBuf.Get<T>();
        a2Tensor = aTensor[DATA_COUNT];

        pipe.InitBuffer(l0bBuf, DOUBLE * DATA_COUNT * sizeof(T));
        bTensor = l0bBuf.Get<T>();
        b2Tensor = bTensor[DATA_COUNT];

        pipe.InitBuffer(l0cBuf, DOUBLE * DATA_COUNT * sizeof(float));
        cTensor = l0cBuf.Get<float>();
        c2Tensor = cTensor[DATA_COUNT];
    }

    aifunc void tiling(){
        int allTimes = GetBlockNum() * BATCH_SIZE; // 每个核处理16个批次, 控制同步计数器不会超过16 
        shape.M = (((shape.K + GetBlockNum() - 1) / GetBlockNum()) + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE; //reshape K、M
        if (shape.M > BASE_SIZE) {
            shape.M = BASE_SIZE;
        }
        tailK = shape.K % shape.M == 0 ? shape.M : shape.K % shape.M;
        shape.K = (shape.K + shape.M - 1) / shape.M;
        shape.perK = (shape.K + allTimes - 1) / allTimes; // 每个批次处理多少个K
        shape.perK = (shape.perK + K_PER_VEC_ONE - 1) / (K_PER_VEC_ONE) * (K_PER_VEC_ONE); // 和4对齐

        int k_per_core = ((shape.K + GetBlockNum() - 1) / GetBlockNum() + shape.perK - 1) / shape.perK * shape.perK;
        shape.K1 = k_per_core * (GetBlockIdx());  // cube blk idx is 0~20
        shape.K2 = ((k_per_core + shape.K1) > shape.K) ? shape.K : (k_per_core + shape.K1);
        shape.fractalM = (shape.M + CEIL_SIZE - 1) / CEIL_SIZE;
        shape.fractalN = (shape.N + CEIL_SIZE - 1) / CEIL_SIZE;
        shape.Mceil = shape.fractalM * CEIL_SIZE;
        shape.Nceil = shape.fractalN * CEIL_SIZE;
        shape.calM = shape.Mceil;

        if constexpr (MM_MODE == MM_SPLIT_MODE) {
            int64_t splitM = (shape.Mceil + BASE_SIZE - 1) / BASE_SIZE;
            int64_t splitN = (shape.Nceil + BASE_SIZE - 1) / BASE_SIZE;
            matmulInfo.splitCount = splitM * splitN;
            matmulInfo.splitCount2 = matmulInfo.splitCount * splitN;
            matmulInfo.splitCount1 = matmulInfo.splitCount * splitM;
        }
        invalidK = static_cast<int64_t>(static_cast<float>(shape.K * shape.M - shape.Mceil) / static_cast<float>(shape.M));
    }

    aifunc void Process(){
        l1empty.setall();
        l0empty.setall();
        outempty.setall();
        // set zero for L1
        int dataCount = shape.calM * shape.calM + shape.Nceil * shape.Nceil + DOUBLE * DOUBLE * shape.calM * shape.Nceil;
        InitConstValue(xTensor, {1, static_cast<uint16_t>(dataCount / CEIL_SIZE), 0, (T)0});
        AscendC::PipeBarrier<PIPE_MTE2>();
        // Preload P1,P2
        CrossCoreWaitFlag(VEC_CUBE_SYNC_ID);
        CopyGmToL1(p2Tensor, p2GM, shape.N, shape.N, shape.Nceil);

        for (int64_t startK = shape.K1; startK < shape.K2; startK += shape.perK) {
            int64_t endK = (startK + shape.perK > shape.K2) ? shape.K2 : (startK + shape.perK);
            bool isLast = startK + shape.perK > shape.K2;
            if constexpr (MM_MODE == MM_SPLIT_MODE) {
                for (int64_t k = startK; k < endK; ++k) {
                    ProcessSplitK(k, isLast);
                }
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(CUBE_VEC_SYNC_ID);
        }

        l1empty.release();
        l0empty.release();
        outempty.release();
    }

    aifunc void ProcessSplitK(int64_t k, bool isLast){
        if (k == shape.K1) {
            l1empty.wait();
            if (k == shape.K2 - 1 && isLast) {
                int64_t oriM = shape.M;
                shape.M = tailK;
                CopyXToL1(GetXTensor(k), xGM[k * oriM * shape.N], tailK != 0, shape);
                shape.M = oriM;
            } else {
                CopyXToL1(GetXTensor(k), xGM[k * shape.M * shape.N], k > invalidK, shape);
            }
            l1ready.set();
        }
        int64_t nextK = k + 1;
        if (nextK < shape.K2 - 1) {
            l1empty.wait();
            CopyXToL1(GetXTensor(nextK), xGM[nextK * shape.M * shape.N], nextK > invalidK, shape);
            l1ready.set();
        } else if (nextK < shape.K2 && !isLast) {
            l1empty.wait();
            CopyXToL1(GetXTensor(nextK), xGM[nextK * shape.M * shape.N], nextK > invalidK, shape);
            l1ready.set();
        } else if (nextK < shape.K2 && isLast) {
            int64_t oriM = shape.M;
            shape.M = tailK;
            l1empty.wait();
            CopyXToL1(GetXTensor(nextK), xGM[nextK * oriM * shape.N], tailK != 0, shape);
            shape.M = oriM;
            l1ready.set();
        }
        if (isLast && k == shape.K2 - 1) {
            ProcessSplitXP2(k, true);
            return;
        }
        ProcessSplitXP2(k, false);
    }

    aifunc void ProcessSplitXP2(int64_t k, bool isTail){  // 20,32,16
        int64_t c = (k - shape.K1) * (matmulInfo.splitCount);
        int64_t p = (k - shape.K1) * (matmulInfo.splitCount2);
        l1ready.wait();
        int32_t numM = 0;
        if (isTail) {
            numM = (tailK + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        } else {
            numM = shape.Mceil;
        }
        for (int32_t nIdx = 0; nIdx < shape.Nceil; nIdx += BASE_SIZE) {
            int32_t numN = (shape.Nceil - nIdx < BASE_SIZE) ? shape.Nceil - nIdx : BASE_SIZE;
            int32_t realN = (shape.N - nIdx < BASE_SIZE) ? shape.N - nIdx : BASE_SIZE;
            for (int32_t kIdx = 0; kIdx < shape.Nceil; kIdx += BASE_SIZE) {
                int32_t numK = (shape.Nceil - kIdx < BASE_SIZE) ? shape.Nceil - kIdx : BASE_SIZE;
                l0empty.wait();
                uint16_t baseIdx = kIdx / CEIL_SIZE;
                for (int32_t sIdx = 0; sIdx < numM / CEIL_SIZE; sIdx ++) {
                    LoadData(GetATensor(p)[sIdx * numK * CEIL_SIZE], GetXTensor(k), LoadData2DParams(baseIdx, numK / CEIL_SIZE, 1, 0, 0, false, 0));
                    baseIdx += shape.fractalN;
                }
                baseIdx = kIdx * shape.fractalN / CEIL_SIZE + nIdx / CEIL_SIZE;
                for (int32_t sIdx = 0; sIdx < numK / CEIL_SIZE; sIdx ++) {
                    LoadData(GetBTensor(p)[sIdx * numN * CEIL_SIZE], p2Tensor, LoadData2DParams(baseIdx, numN / CEIL_SIZE, 1, 0, 0, true, 0));
                    baseIdx += shape.fractalN;
                }
                l0ready.set();
                if (nIdx + numN == shape.Nceil && kIdx + numK == shape.Nceil) {
                    l1empty.set();
                }

                if (kIdx == 0) {
                    outempty.wait();
                }
                l0ready.wait();
                CalMatrix(GetCTensor(c), GetATensor(p), GetBTensor(p), numM, numK, numN, kIdx + numK == shape.Nceil ? UFMode3 : UFMode2, false, false, kIdx == 0);
                PipeBarrier<PIPE_M>();
                l0empty.set();
                p ++;
            }
            QuantMode_t quantMode = F322F16;
            if constexpr(std::is_same<T, bfloat16_t>::value) {
                quantMode = F322BF16;
            }
            DataCopyCO12DstParams dataCopyParams(realN, numM, shape.N, numM, quantMode, 0, false, true);
            dataCopyParams.unitFlag = UFMode3;
            DataCopy(outnzGM[k * shape.Mceil * shape.N + nIdx], GetCTensor(c), dataCopyParams);

            outempty.set();
            c ++;
        }
    }

    __aicore__ inline LocalTensor<T> GetXTensor(int64_t k){
        return ((k & 1) == 0) ? xTensor : x2Tensor;
    };

    __aicore__ inline LocalTensor<T> GetYTensor(int64_t k){
        return ((k & 1) == 0) ? yTensor : y2Tensor;
    };

    __aicore__ inline LocalTensor<T> GetATensor(int64_t k){
        return ((k & 1) == 0) ? aTensor : a2Tensor;
    };

    __aicore__ inline LocalTensor<T> GetBTensor(int64_t k){
        return ((k & 1) == 0) ? bTensor : b2Tensor;
    };

    __aicore__ inline LocalTensor<float> GetCTensor(int64_t k){
        return ((k & 1) == 0) ? cTensor : c2Tensor;
    };

private:
    TPipe pipe;
    FlatQuantShapeInfo shape;
    MatmulInfo matmulInfo;
    GlobalTensor<T> xGM;
    GlobalTensor<T> p1GM;
    GlobalTensor<T> p2GM;
    GlobalTensor<T> outnzGM;
    GlobalTensor<T> doubleP1GM;

    TBuf<TPosition::A1> l1Buf;
    TBuf<TPosition::A2> l0aBuf;
    TBuf<TPosition::B2> l0bBuf;
    TBuf<TPosition::CO1> l0cBuf;

    LocalTensor<T> xTensor;
    LocalTensor<T> x2Tensor;
    LocalTensor<T> yTensor;
    LocalTensor<T> y2Tensor;
    LocalTensor<T> p1Tensor;
    LocalTensor<T> p2Tensor;
    LocalTensor<T> aTensor;
    LocalTensor<T> a2Tensor;
    LocalTensor<T> bTensor;
    LocalTensor<T> b2Tensor;
    LocalTensor<float> cTensor;
    LocalTensor<float> c2Tensor;

    DEvent<PIPE_MTE2, PIPE_MTE1> l1ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_MTE1, PIPE_MTE2> l1empty{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_MTE1, PIPE_M> l0ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_M, PIPE_MTE1> l0empty{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_FIX, PIPE_M> outempty{EVENT_ID4, EVENT_ID5};

    int64_t invalidK = 0; // 从invalidK开始，需要区分尾块/非尾块搬入x
    uint32_t tailK = 0;
};
}  // namespace FlatQuantNS

#endif  // FLAT_QUANT_CUBE_H