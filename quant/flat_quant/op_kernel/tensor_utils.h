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

using namespace AscendC;

namespace FlatQuantNS {
constexpr int NUM_TWO = 2;
constexpr int NUM_THREE = 3;
constexpr int NUM_FOUR = 4;
constexpr int NUM_FIVE = 5;
constexpr int NUM_EIGHT = 8;
constexpr int NUM_TEN = 10;
constexpr int NUM_ONE_SIX = 16;
constexpr int NUM_SIX_FOUR = 64;
constexpr int NUM_ONE_TWO_EIGHT = 128;
constexpr int NUM_ONE_FOUR_FOUR = 144;
constexpr int NUM_ONE_NINE_TWO = 192;
constexpr int NUM_TWO_ZERO_EIGHT = 208;
constexpr int NUM_FIVE_ONE_TWO = 512;
constexpr int NUM_ONE_ZERO_TWO_FOUR = 1024;
constexpr int NUM_THREE_TWO = 32;
constexpr float NUM_FLOAT_SEVEN = 7.0f;

constexpr int N_PRELOAD = NUM_TWO;   // pre-process n batches (n*K_PER_VEC) before running cube
constexpr int K_PER_VEC = NUM_FOUR;  // batch number. Each loop processes K_PER_VEC*M*N

struct FlatQuantShapeInfo {
    int32_t K;
    int32_t M;
    int32_t N;        // basic shape
    int32_t K1;
    int32_t K2;         // loop start and loop end
    int32_t Mceil;
    int32_t Nceil;   // ceil shape
    int32_t procP1;         // pre-process p1 or p2
    int32_t pstart;
    int32_t prows;  // start row number of p1 or p2 matrix
    float clipRatio;
};

#define aifunc __aicore__ inline

/* ------------- Events ------------- */

template <pipe_t p1, pipe_t p2>
class DEvent {
public:
    aifunc DEvent()
    {}
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
        if (wait_cnt % NUM_TWO == 0) {
            sync.WaitFlag(id1);
        } else {
            sync.WaitFlag(id2);
        }
        wait_cnt++;
    }
    aifunc void set()
    {
        if (set_cnt % NUM_TWO == 0) {
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
    mmadParams.cmatrixInitVal = true;
    mmadParams.unitFlag = unitFlag;
    Mmad(c, a, b, mmadParams);
}
}  // namespace FlatQuantNS

#endif  // TENSOR_UTILS_H