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
 * \file tensor_utils.h
 * \brief
 */
#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace FlatQuantNS {
constexpr MatmulConfig MDL_CFG = GetMDLConfig(false, false, 0, false, false, false, true);
constexpr uint8_t MM_BASE_MODE = 1;
constexpr uint8_t MM_DOUBLE_MODE = 2;
constexpr uint8_t MM_SPLIT_MODE = 3;
constexpr uint8_t MM_HIGH_MODE = 4;

constexpr uint8_t SYNC_MODE0 = 0;
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint8_t CUBE_VEC_SYNC_ID = 0;
constexpr uint8_t VEC_CUBE_SYNC_ID = 4;
constexpr uint8_t VEC_SYNC_ID = 5;

constexpr int32_t DOUBLE = 2;
constexpr int32_t CEIL_SIZE = 16;
constexpr int32_t UB_SIZE = 192 * 1024;
constexpr int32_t HIGH_UB_SIZE = 148 * 1024;
constexpr int32_t L1_SIZE = 512 * 1024;
constexpr int32_t DATA_COUNT = 16384;
constexpr int32_t CAST_COUNT = 8192;
constexpr int32_t SCALE_COUNT = 2048;
constexpr int32_t BASE_SIZE = 128;
constexpr float NUM_FLOAT_SEVEN = 7.0f;
constexpr int32_t K_PER_VEC = 4; // batch number. Each loop processes K_PER_VEC*M*N
constexpr int32_t K_DOUBLE_VEC = DOUBLE * K_PER_VEC;

struct FlatQuantShapeInfo {
    int64_t K;
    int64_t M;
    int64_t N; // basic shape
    int64_t perK;
    int64_t K1;
    int64_t K2; // loop start and loop end
    int64_t Mceil;
    int64_t Nceil; // ceil shape
    int64_t fractalM;
    int64_t fractalN;
    int64_t calM;
    int64_t realM;
};

struct MatmulInfo {
    int64_t splitCount;
    int64_t splitCount2;
    int64_t splitCount1;
};

#define aifunc __aicore__ inline

/* ------------- Events ------------- */

template <pipe_t p1, pipe_t p2>
class DEvent {
public:
    aifunc DEvent(int e_id1, int e_id2)
    {
        id1 = (event_t)e_id1;
        id2 = (event_t)e_id2;
    }
    aifunc DEvent(event_t e_id1, event_t e_id2)
    {
        id1 = e_id1;
        id2 = e_id2;
    }
    aifunc void wait()
    {
        if ((wait_cnt & 1) == 0) {
            sync.WaitFlag(id1);
        } else {
            sync.WaitFlag(id2);
        }
        wait_cnt++;
    }
    aifunc void set()
    {
        if ((set_cnt & 1) == 0) {
            sync.SetFlag(id1);
        } else {
            sync.SetFlag(id2);
        }
        set_cnt++;
    }

    aifunc void setall()
    {
        set();
        set();
    }
    aifunc void release()
    {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    TQueSync<p1, p2> sync;
    event_t id1 = (event_t)0;
    event_t id2 = (event_t)1;
    int wait_cnt = 0;
    int set_cnt = 0;
};

template <typename CType, typename DType>	
__aicore__ inline void CalMatrix(LocalTensor<CType> c, LocalTensor<DType> a, LocalTensor<DType> b, uint16_t m, uint16_t k,
    uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal)
{
    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    mmadParams.cmatrixSource = cmatrixSource;
    mmadParams.unitFlag = unitFlag;
    Mmad(c, a, b, mmadParams);
}

template <typename T>
__aicore__ inline void CopyGmToL1(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t realN, uint32_t realD, uint32_t ceilD)
{
    // nd2zz
    uint32_t tailN = realN % CEIL_SIZE;
    if (tailN < realN) {
        DataCopy(dst, src, Nd2NzParams(realN / CEIL_SIZE, CEIL_SIZE, realD, CEIL_SIZE * realD, realD, CEIL_SIZE, 1, CEIL_SIZE * ceilD));
    }
    if (tailN != 0) {
        int offsetN = realN / CEIL_SIZE * CEIL_SIZE;
        DataCopy(dst[offsetN * ceilD], src[offsetN * realD], Nd2NzParams(1, tailN, realD, 0, realD, CEIL_SIZE, 1, 0));
    }
}

template <typename T>
__aicore__ inline void CopyXToL1(LocalTensor<T> dst, GlobalTensor<T> src, bool useSlowCopy, FlatQuantShapeInfo shape)
{
    if (useSlowCopy) {
        CopyGmToL1(dst, src, shape.M, shape.N, shape.Nceil);
    } else {
        DataCopy(dst, src, Nd2NzParams(shape.Mceil / CEIL_SIZE, CEIL_SIZE, shape.N, CEIL_SIZE * shape.N, shape.N, CEIL_SIZE, 1, CEIL_SIZE * shape.Nceil));
    }
}

template <typename T>
__aicore__ inline void CalReduceMax(LocalTensor<T> srcTensor, LocalTensor<T> maxTensor, int32_t len, event_t eventIdVToS)
{
    WholeReduceMax(maxTensor, srcTensor, BASE_SIZE, len/BASE_SIZE, 1, 1, DEFAULT_REPEAT_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(srcTensor, maxTensor, BASE_SIZE, 1, 1, 1, DEFAULT_REPEAT_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}
} // namespace FlatQuantNS

#endif // TENSOR_UTILS_H