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
 * \file ascend_quant_v2_nz.h
 * \brief
 */

#ifndef ASCEND_QUANT_V2_NZ_H
#define ASCEND_QUANT_V2_NZ_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace AscendQuantV2 {

using namespace AscendC;

constexpr int32_t MAX_UB_SIZE = 192 * 1024;

template <typename T>
class AscendQuantV2NZFP32 {
public:
    TPipe pipe;
    __aicore__ inline AscendQuantV2NZFP32(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                              const AscendQuantV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t offset_in, DataCopyExtParams dataCopyParams);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t offset_out, DataCopyExtParams dataCopyParams);

private:

    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<T> x1Tmp;  //16*8*4*8=4096

    GlobalTensor<T> inputGm;
    GlobalTensor<int8_t> outputGm;

    int64_t E;
    int64_t K;
    int64_t N;
    int64_t blockIdx;
    int64_t needCoreNum;
};

template <typename T>
__aicore__ inline void AscendQuantV2NZFP32<T>::Init(GM_ADDR x, GM_ADDR y,
                              const AscendQuantV2TilingData* tilingData) {
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));;

    E = tilingData->E;
    K = tilingData->K;
    N = tilingData->N;  //tiling测获取输入的专家数以及长和宽
    blockIdx = GetBlockIdx();
    needCoreNum = tilingData->needCoreNum;  //一个核一次最多处理16*64个数    K/16/8为所需核数
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void AscendQuantV2NZFP32<T>::Process() {
    if (blockIdx >= needCoreNum) {
        return;
    }
    int64_t kloop = K / 16 / needCoreNum;  //K维度上循环次数，对应多少组核
    int64_t tail = K - 16 * needCoreNum * kloop;
    int64_t tail_core = tail / 16;
    if(tail > 0){
        kloop++;
    }
    int64_t nloop = E * N / 64; //最外层循环，
    int64_t dataCount = 16 * 8 * 8;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int64_t i = 0; i < nloop; i++){
        int64_t offset_n = K * i * 64;
        for (int64_t j = 0; j < kloop; j++){
            if(tail > 0 && blockIdx >= tail_core && j == kloop - 1){
                continue;
            }
            int64_t offset_k =16 * 16 * blockIdx + 16 * 16 * needCoreNum * j;
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            for(int64_t k = 0; k < 4; k++){
                int64_t offset = K * 16 * k;//
                DataCopyExtParams dataCopyParams{16, static_cast<uint32_t>(16 * sizeof(T)), 0, 6, 0};  //repeat次数，一次repeat的长度
                CopyIn(offset, offset + offset_k + offset_n, dataCopyParams);
            }
            Compute(dataCount);
            DataCopyExtParams outParams {1, static_cast<uint32_t>(64 * 16/2), 0, 0, 0};
            CopyOut(offset_k * 4 + offset_n, outParams);
    }
}
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void AscendQuantV2NZFP32<T>::CopyIn(int64_t offset, int64_t offset_in, DataCopyExtParams dataCopyParams){
    x1Tmp = tmpTensor[0].ReinterpretCast<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(x1Tmp[offset/K], inputGm[offset_in], dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline void AscendQuantV2NZFP32<T>::Compute(int64_t dataCount){

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(x1Tmp.template ReinterpretCast<int32_t>(), x1Tmp, RoundMode::CAST_RINT, dataCount);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(x1Tmp.template ReinterpretCast<half>(), x1Tmp.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
#ifdef __CCE_UT_TEST__
#else
    Cast(x1Tmp.template ReinterpretCast<int4b_t>(), x1Tmp.template ReinterpretCast<half>(), RoundMode::CAST_RINT, dataCount);
#endif
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AscendQuantV2NZFP32<T>::CopyOut(int64_t offset_out, DataCopyExtParams outParams){
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(outputGm[offset_out/2], x1Tmp.template ReinterpretCast<int8_t>(), outParams);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
}
#endif