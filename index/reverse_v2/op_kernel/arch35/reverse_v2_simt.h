/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
* \file reverse_v2_simt.h
* \brief reverse_v2
*/

#ifndef ASCENDC_REVERSE_V2_SIMT_H
#define ASCENDC_REVERSE_V2_SIMT_H

#include "kernel_operator.h"

namespace ReverseV2 {
using namespace AscendC;

const uint32_t LAUNCH_THREAD_NUM = 1024;
const uint32_t SIMT_PARAM_BUFFER = 512;
const uint32_t SIMT_PARAM_NUM = 31;

const uint32_t DIM1 = 1;
const uint32_t DIM2 = 2;
const uint32_t DIM3 = 3;
const uint32_t DIM4 = 4;
const uint32_t DIM5 = 5;
const uint32_t DIM6 = 6;
const uint32_t DIM7 = 7;
const uint32_t DIM8 = 8;

enum class SIMT_PARAM_INDEX : uint32_t {
    SIMT_PARAM_INDEX0 = 0,
    SIMT_PARAM_INDEX1,
    SIMT_PARAM_INDEX2,
    SIMT_PARAM_INDEX3,
    SIMT_PARAM_INDEX4,
    SIMT_PARAM_INDEX5,
    SIMT_PARAM_INDEX6,
    SIMT_PARAM_INDEX7,
    SIMT_PARAM_INDEX8,
    SIMT_PARAM_INDEX9,
    SIMT_PARAM_INDEX10,
    SIMT_PARAM_INDEX11,
    SIMT_PARAM_INDEX12,
    SIMT_PARAM_INDEX13,
    SIMT_PARAM_INDEX14,
    SIMT_PARAM_INDEX15,
    SIMT_PARAM_INDEX16,
    SIMT_PARAM_INDEX17,
    SIMT_PARAM_INDEX18,
    SIMT_PARAM_INDEX19,
    SIMT_PARAM_INDEX20,
    SIMT_PARAM_INDEX21,
    SIMT_PARAM_INDEX22,
    SIMT_PARAM_INDEX23,
    SIMT_PARAM_INDEX24,
    SIMT_PARAM_INDEX25,
    SIMT_PARAM_INDEX26,
    SIMT_PARAM_INDEX27,
    SIMT_PARAM_INDEX28,
    SIMT_PARAM_INDEX29,
    SIMT_PARAM_INDEX30
};

template <typename T>
struct FastDivParam {
    T magic0 = 0;
    T magic1 = 0;
    T magic2 = 0;
    T magic3 = 0;
    T magic4 = 0;
    T magic5 = 0;
    T magic6 = 0;
    T shift0 = 0;
    T shift1 = 0;
    T shift2 = 0;
    T shift3 = 0;
    T shift4 = 0;
    T shift5 = 0;
    T shift6 = 0;
};

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum> 
class ReverseV2Simt {
public:
    __aicore__ inline ReverseV2Simt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReverseV2TilingData4AscendC* __restrict tilingData);
    __aicore__ inline void Process();

private:
    static __simt_vf__ __aicore__ inline void SimtCompute(__gm__ T2* input, __gm__ T2* output, T1 blockIdx,
                                                          __local_mem__ T1* simtParam);

    static __simt_callee__ __aicore__ inline void ProcDim8(__gm__ T2* input, __gm__ T2* output, T1 &param0, T1 &param1,
                                           T1 &param2, T1 &param3,  T1 &param4, T1 &param5, T1 &param6, T1 &dim0, T1 &dim1,
                                           T1 &dim2, T1 &dim3, T1 &dim4, T1 &dim5, T1 &dim6, T1 &dim7, 
                                           const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1, T1 &idx2,
                                           T1 &idx3, T1 &idx4, T1 &idx5, T1 &idx6, T1 &idx7, T1 &y0, T1 &y1, T1 &y2, T1 &y3,
                                           T1 &y4, T1 &y5, T1 &y6, T1 &idx, T1 &inputIdx);

    static __simt_callee__ __aicore__ inline void ProcDim7(__gm__ T2* input, __gm__ T2* output, T1 &param0, T1 &param1,
                                           T1 &param2, T1 &param3,  T1 &param4, T1 &param5, T1 &param6, 
                                           T1 &dim0, T1 &dim1, T1 &dim2, T1 &dim3, T1 &dim4, T1 &dim5, T1 &dim6,
                                           const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1, T1 &idx2,
                                           T1 &idx3, T1 &idx4, T1 &idx5, T1 &idx6, T1 &y0, T1 &y1, T1 &y2, T1 &y3,
                                           T1 &y4, T1 &y5, T1 &idx, T1 &inputIdx);

    static __simt_callee__ __aicore__ inline void ProcDim6(__gm__ T2* input, __gm__ T2* output, T1 &param0, T1 &param1,
                                           T1 &param2, T1 &param3,  T1 &param4, T1 &param5,
                                           T1 &dim0, T1 &dim1, T1 &dim2, T1 &dim3, T1 &dim4, T1 &dim5,
                                           const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1, T1 &idx2,
                                           T1 &idx3, T1 &idx4, T1 &idx5, T1 &y0, T1 &y1, T1 &y2, T1 &y3,
                                           T1 &y4, T1 &idx, T1 &inputIdx);

    static __simt_callee__ __aicore__ inline void ProcDim5(__gm__ T2* input, __gm__ T2* output, T1 &param0, T1 &param1,
                                           T1 &param2, T1 &param3, T1 &param4, T1 &dim0, T1 &dim1, T1 &dim2, 
                                           T1 &dim3, T1 &dim4, const FastDivParam<T1> &fastDivParam, T1 &idx0, 
                                           T1 &idx1, T1 &idx2, T1 &idx3, T1 &idx4, T1 &y0, T1 &y1, 
                                           T1 &y2, T1 &y3, T1 &idx, T1 &inputIdx);

    static __simt_callee__ __aicore__ inline void ProcDim4(__gm__ T2* input, __gm__ T2* output, T1 &param0, T1 &param1,
                                           T1 &param2, T1 &param3, T1 &dim0, T1 &dim1, T1 &dim2, 
                                           T1 &dim3, T1 &dim4, const FastDivParam<T1> &fastDivParam, T1 &idx0, 
                                           T1 &idx1, T1 &idx2, T1 &idx3, T1 &y0, T1 &y1, 
                                           T1 &y2, T1 &y3, T1 &idx, T1 &inputIdx);

    static __simt_callee__ __aicore__ inline void ProcDim3(__gm__ T2* input, __gm__ T2* output, T1 &param0, T1 &param1,
                                           T1 &param2, T1 &dim0, T1 &dim1, T1 &dim2, 
                                           const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1, T1 &idx2,
                                           T1 &y0, T1 &y1, T1 &idx, T1 &inputIdx);

    static __simt_callee__ __aicore__ inline void ProcDim2(__gm__ T2* input, __gm__ T2* output, T1 &param0, T1 &param1,
                                           T1 &dim0, T1 &dim1, const FastDivParam<T1> &fastDivParam,
                                           T1 &idx0, T1 &idx1, T1 &y0, T1 &idx, T1 &inputIdx);

    static __simt_callee__ __aicore__ inline void ProcDim1(__gm__ T2* input, __gm__ T2* output, 
                                           T1 &param0, T1 &idx0,T1 &dim0, T1 &idx, T1 &inputIdx);

private:
    TPipe pipe_;
    GlobalTensor<T2> inputGm_;
    GlobalTensor<T2> outputGm_;
    T1 blockIdx_;
    const ReverseV2TilingData4AscendC* tilingData_;
    TBuf<TPosition::VECCALC> simtBuf_;
};

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::Init(GM_ADDR x, GM_ADDR y, 
    const ReverseV2TilingData4AscendC* __restrict tilingData)
{
    inputGm_.SetGlobalBuffer((__gm__ T2*)(x));
    outputGm_.SetGlobalBuffer((__gm__ T2*)(y));
    blockIdx_ = AscendC::GetBlockIdx();
    tilingData_ = tilingData;
    pipe_.InitBuffer(simtBuf_, SIMT_PARAM_BUFFER);
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim8(__gm__ T2* input, __gm__ T2* output,
    T1 &param0, T1 &param1, T1 &param2, T1 &param3,  T1 &param4, T1 &param5, T1 &param6, T1 &dim0, T1 &dim1,
    T1 &dim2, T1 &dim3, T1 &dim4, T1 &dim5, T1 &dim6, T1 &dim7, const FastDivParam<T1> &fastDivParam, T1 &idx0,
    T1 &idx1, T1 &idx2, T1 &idx3, T1 &idx4, T1 &idx5, T1 &idx6, T1 &idx7, T1 &y0, T1 &y1, T1 &y2, T1 &y3, T1 &y4,
    T1 &y5, T1 &y6, T1 &idx, T1 &inputIdx)
{
    idx0 = Simt::UintDiv(idx, fastDivParam.magic0, fastDivParam.shift0);
    y1 = idx - idx0 * param0;
    idx1 = Simt::UintDiv(y1, fastDivParam.magic1, fastDivParam.shift1);
    y2 = y1 - idx1 * param1;

    idx2 = Simt::UintDiv(y2, fastDivParam.magic2, fastDivParam.shift2);
    y3 = y2 - idx2 * param2;
    idx3 = Simt::UintDiv(y3, fastDivParam.magic3, fastDivParam.shift3);
    y4 = y3 - idx3 * param3;
    idx4 = Simt::UintDiv(y4, fastDivParam.magic4, fastDivParam.shift4);
    y5 = y4 - idx4 * param4;
    idx5 = Simt::UintDiv(y5, fastDivParam.magic5, fastDivParam.shift5);
    y6 = y5 - idx5 * param5;
    idx6 = Simt::UintDiv(y6, fastDivParam.magic6, fastDivParam.shift6);
    idx7 = y6 - idx6 * param6;

    if constexpr (isReverse) {
        idx0 = dim0 - idx0 - 1;
        idx2 = dim2 - idx2 - 1;
        idx4 = dim4 - idx4 - 1;
        idx6 = dim6 - idx6 - 1;
    } else {
        idx1 = dim1 - idx1 - 1;
        idx3 = dim3 - idx3 - 1;
        idx5 = dim5 - idx5 - 1;
        idx7 = dim7 - idx7 - 1;
    }
    inputIdx = idx0 * param0 + idx1 * param1 + idx2 * param2 + idx3 * param3 + idx4 * param4 + idx5 * param5 + idx6 * param6 + idx7;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim7(__gm__ T2* input, __gm__ T2* output,
    T1 &param0, T1 &param1, T1 &param2, T1 &param3,  T1 &param4, T1 &param5, T1 &param6, T1 &dim0, T1 &dim1,
    T1 &dim2, T1 &dim3, T1 &dim4, T1 &dim5, T1 &dim6, const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1,
    T1 &idx2, T1 &idx3, T1 &idx4, T1 &idx5, T1 &idx6, T1 &y0, T1 &y1, T1 &y2, T1 &y3, T1 &y4, T1 &y5, T1 &idx, T1 &inputIdx)
{
    idx0 = Simt::UintDiv(idx, fastDivParam.magic0, fastDivParam.shift0);
    y1 = idx - idx0 * param0;
    idx1 = Simt::UintDiv(y1, fastDivParam.magic1, fastDivParam.shift1);
    y2 = y1 - idx1 * param1;
    idx2 = Simt::UintDiv(y2, fastDivParam.magic2, fastDivParam.shift2);
    y3 = y2 - idx2 * param2;
    idx3 = Simt::UintDiv(y3, fastDivParam.magic3, fastDivParam.shift3);
    y4 = y3 - idx3 * param3;
    idx4 = Simt::UintDiv(y4, fastDivParam.magic4, fastDivParam.shift4);
    y5 = y4 - idx4 * param4;
    idx5 = Simt::UintDiv(y5, fastDivParam.magic5, fastDivParam.shift5);
    idx6 = y5 - idx5 * param5;

    if constexpr (isReverse) {
        idx0 = dim0 - idx0 - 1;
        idx2 = dim2 - idx2 - 1;
        idx4 = dim4 - idx4 - 1;
        idx6 = dim6 - idx6 - 1;
    } else {
        idx1 = dim1 - idx1 - 1;
        idx3 = dim3 - idx3 - 1;
        idx5 = dim5 - idx5 - 1;
    }
    inputIdx = idx0 * param0 + idx1 * param1 + idx2 * param2 + idx3 * param3 + idx4 * param4 + idx5 * param5 + idx6;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim6(__gm__ T2* input, __gm__ T2* output,
    T1 &param0, T1 &param1, T1 &param2, T1 &param3,  T1 &param4, T1 &param5, T1 &dim0, T1 &dim1, T1 &dim2, T1 &dim3,
    T1 &dim4, T1 &dim5, const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1, T1 &idx2, T1 &idx3, T1 &idx4,
    T1 &idx5, T1 &y0, T1 &y1, T1 &y2, T1 &y3, T1 &y4, T1 &idx, T1 &inputIdx)
{
    idx0 = Simt::UintDiv(idx, fastDivParam.magic0, fastDivParam.shift0);
    y1 = idx - idx0 * param0;
    idx1 = Simt::UintDiv(y1, fastDivParam.magic1, fastDivParam.shift1);
    y2 = y1 - idx1 * param1;
    idx2 = Simt::UintDiv(y2, fastDivParam.magic2, fastDivParam.shift2);
    y3 = y2 - idx2 * param2;
    idx3 = Simt::UintDiv(y3, fastDivParam.magic3, fastDivParam.shift3);
    y4 = y3 - idx3 * param3;
    idx4 = Simt::UintDiv(y4, fastDivParam.magic4, fastDivParam.shift4);
    idx5 = y4 - idx4 * param4;

    if constexpr (isReverse) {
        idx0 = dim0 - idx0 - 1;
        idx2 = dim2 - idx2 - 1;
        idx4 = dim4 - idx4 - 1;
    } else {
        idx1 = dim1 - idx1 - 1;
        idx3 = dim3 - idx3 - 1;
        idx5 = dim5 - idx5 - 1;
    }
    inputIdx = idx0 * param0 + idx1 * param1 + idx2 * param2 + idx3 * param3 + idx4 * param4 + idx5;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim5(__gm__ T2* input, __gm__ T2* output,
    T1 &param0, T1 &param1, T1 &param2, T1 &param3, T1 &param4, T1 &dim0, T1 &dim1, T1 &dim2, T1 &dim3, T1 &dim4,
    const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1, T1 &idx2, T1 &idx3, T1 &idx4, T1 &y0, T1 &y1, 
    T1 &y2, T1 &y3, T1 &idx, T1 &inputIdx)
{
    idx0 = Simt::UintDiv(idx, fastDivParam.magic0, fastDivParam.shift0);
    y1 = idx - idx0 * param0;
    idx1 = Simt::UintDiv(y1, fastDivParam.magic1, fastDivParam.shift1);
    y2 = y1 - idx1 * param1;
    idx2 = Simt::UintDiv(y2, fastDivParam.magic2, fastDivParam.shift2);
    y3 = y2 - idx2 * param2;
    idx3 = Simt::UintDiv(y3, fastDivParam.magic3, fastDivParam.shift3);
    idx4 = y3 - idx3 * param3;

    if constexpr (isReverse) {
        idx0 = dim0 - idx0 - 1;
        idx2 = dim2 - idx2 - 1;
        idx4 = dim4 - idx4 - 1;
    } else {
        idx1 = dim1 - idx1 - 1;
        idx3 = dim3 - idx3 - 1;
    }
    inputIdx = idx0 * param0 + idx1 * param1 + idx2 * param2 + idx3 * param3 + idx4;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim4(__gm__ T2* input, __gm__ T2* output,
    T1 &param0, T1 &param1, T1 &param2, T1 &param3, T1 &dim0, T1 &dim1, T1 &dim2, T1 &dim3, T1 &dim4,
    const FastDivParam<T1> &fastDivParam, T1 &idx0, T1 &idx1, T1 &idx2, T1 &idx3, T1 &y0, T1 &y1, T1 &y2, 
    T1 &y3, T1 &idx, T1 &inputIdx)
{
    idx0 = Simt::UintDiv(idx, fastDivParam.magic0, fastDivParam.shift0);
    y1 = idx - idx0 * param0;
    idx1 = Simt::UintDiv(y1, fastDivParam.magic1, fastDivParam.shift1);
    y2 = y1 - idx1 * param1;
    idx2 = Simt::UintDiv(y2, fastDivParam.magic2, fastDivParam.shift2);
    idx3 = y2 - idx2 * param2;

    if constexpr (isReverse) {
        idx0 = dim0 - idx0 - 1;
        idx2 = dim2 - idx2 - 1;
    } else {
        idx1 = dim1 - idx1 - 1;
        idx3 = dim3 - idx3 - 1;
    }
    inputIdx = idx0 * param0 + idx1 * param1 + idx2 * param2 + idx3;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim3(__gm__ T2* input, __gm__ T2* output,
    T1 &param0, T1 &param1, T1 &param2, T1 &dim0, T1 &dim1, T1 &dim2, const FastDivParam<T1> &fastDivParam,
    T1 &idx0, T1 &idx1, T1 &idx2, T1 &y0, T1 &y1, T1 &idx, T1 &inputIdx)
{
    idx0 = Simt::UintDiv(idx, fastDivParam.magic0, fastDivParam.shift0);
    y1 = idx - idx0 * param0;
    idx1 = Simt::UintDiv(y1, fastDivParam.magic1, fastDivParam.shift1);
    idx2 = y1 - idx1 * param1;

    if constexpr (isReverse) {
        idx0 = dim0 - idx0 - 1;
        idx2 = dim2 - idx2 - 1;
    } else {
        idx1 = dim1 - idx1 - 1;
    }
    inputIdx = idx0 * param0 + idx1 * param1 + idx2;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim2(__gm__ T2* input, __gm__ T2* output,
    T1 &param0, T1 &param1, T1 &dim0, T1 &dim1, const FastDivParam<T1> &fastDivParam,
    T1 &idx0, T1 &idx1, T1 &y0, T1 &idx, T1 &inputIdx)
{
    idx0 = Simt::UintDiv(idx, fastDivParam.magic0, fastDivParam.shift0);
    idx1 = idx - idx0 * param0;
    
    if constexpr (isReverse) {
        idx0 = dim0 - idx0 - 1;
    } else {
        idx1 = dim1 - idx1 - 1;
    }
    inputIdx = idx0 * param0 + idx1;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_callee__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::ProcDim1(
    __gm__ T2* input, __gm__ T2* output, 
    T1 &param0, T1 &idx0,T1 &dim0, T1 &idx, T1 &inputIdx)
{
    idx0 = idx;
    idx0 = dim0 - idx0 - 1;
    inputIdx = idx0 * param0;
    output[idx] = input[inputIdx];
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__simt_vf__ __aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::SimtCompute(__gm__ T2* input, __gm__ T2* output, T1 blockIdx,
                                                                                         __local_mem__ T1* simtParam)

{
    T1 param0 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX0)];
    T1 param1 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX1)];
    T1 param2 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX2)];
    T1 param3 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX3)];
    T1 param4 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX4)];
    T1 param5 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX5)];
    T1 param6 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX6)];
    T1 dim0 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX7)];
    T1 dim1 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX8)];
    T1 dim2 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX9)];
    T1 dim3 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX10)];
    T1 dim4 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX11)];
    T1 dim5 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX12)];
    T1 dim6 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX13)];
    T1 dim7 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX14)];
    T1 blockFactor = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX15)];
    T1 inputSize = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX16)];
    
    FastDivParam<T1> fastDivParam;
    fastDivParam.magic0 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX17)];
    fastDivParam.magic1 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX18)];
    fastDivParam.magic2 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX19)];
    fastDivParam.magic3 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX20)];
    fastDivParam.magic4 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX21)];
    fastDivParam.magic5 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX22)];
    fastDivParam.magic6 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX23)];
    fastDivParam.shift0 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX24)];
    fastDivParam.shift1 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX25)];
    fastDivParam.shift2 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX26)];
    fastDivParam.shift3 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX27)];
    fastDivParam.shift4 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX28)];
    fastDivParam.shift5 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX29)];
    fastDivParam.shift6 = simtParam[static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX30)];

    T1 curCoreEndIdx = (blockIdx + 1) * blockFactor;
    curCoreEndIdx =  curCoreEndIdx > inputSize ? inputSize : curCoreEndIdx;
    T1 idx0 = 0;
    T1 idx1 = 0;
    T1 idx2 = 0;
    T1 idx3 = 0;
    T1 idx4 = 0;
    T1 idx5 = 0;
    T1 idx6 = 0;
    T1 idx7 = 0;
    T1 y0 = 0;
    T1 y1 = 0;
    T1 y2 = 0;
    T1 y3 = 0;
    T1 y4 = 0;
    T1 y5 = 0;
    T1 y6 = 0;
    T1 inputIdx = 0;
    for (T1 idx = blockIdx * blockFactor + Simt::GetThreadIdx(); idx < curCoreEndIdx; idx += Simt::GetThreadNum()) {
        if constexpr (dimNum == DIM8) {
            ProcDim8(input, output, param0, param1, param2, param3, param4, param5, param6, dim0, dim1,
                     dim2, dim3, dim4, dim5, dim6, dim7, fastDivParam, idx0, idx1, idx2,
                     idx3, idx4, idx5, idx6, idx7, y0, y1, y2, y3, y4, y5, y6, idx,inputIdx);
        } else if constexpr (dimNum == DIM7) {
            ProcDim7(input, output, param0, param1, param2, param3,  param4, param5, param6, dim0, dim1,
                     dim2, dim3, dim4, dim5, dim6, fastDivParam, idx0, idx1, idx2, idx3, idx4, idx5,
                     idx6, y0, y1, y2, y3, y4, y5, idx,inputIdx);
        } else if constexpr (dimNum == DIM6) {
            ProcDim6(input, output, param0, param1, param2, param3,  param4, param5, dim0, dim1, dim2, dim3,
                     dim4, dim5, fastDivParam, idx0, idx1, idx2, idx3, idx4, idx5, y0, y1, y2, y3, y4, idx,inputIdx);
        } else if constexpr (dimNum == DIM5) {
            ProcDim5(input, output, param0, param1, param2, param3, param4, dim0, dim1, dim2, dim3, dim4,
                     fastDivParam, idx0, idx1, idx2, idx3, idx4, y0, y1, y2, y3, idx,inputIdx);
        } else if constexpr (dimNum == DIM4) {
            ProcDim4(input, output, param0, param1, param2, param3, dim0, dim1, dim2, dim3, dim4, fastDivParam,
                     idx0, idx1, idx2, idx3, y0, y1, y2, y3, idx,inputIdx);
        } else if constexpr (dimNum == DIM3) {
            ProcDim3(input, output, param0, param1, param2, dim0, dim1, dim2, fastDivParam, idx0, idx1, idx2,
                     y0, y1, idx,inputIdx);
        } else if constexpr (dimNum == DIM2) {
            ProcDim2(input, output, param0, param1, dim0, dim1, fastDivParam, idx0, idx1, y0, idx, inputIdx);
        } else if constexpr (dimNum == DIM1){
            ProcDim1(input, output, param0, idx0,dim0, idx, inputIdx);
        }
    }
}

template <typename T1, typename T2, const bool isReverse, const uint32_t dimNum>
__aicore__ inline void ReverseV2Simt<T1, T2, isReverse, dimNum>::Process()
{
    FastDivParam<T1> fastDivParam;
    if constexpr (dimNum == DIM8) {
        GetUintDivMagicAndShift(fastDivParam.magic0, fastDivParam.shift0, static_cast<T1>(tilingData_->param0));
        GetUintDivMagicAndShift(fastDivParam.magic1, fastDivParam.shift1, static_cast<T1>(tilingData_->param1));
        GetUintDivMagicAndShift(fastDivParam.magic2, fastDivParam.shift2, static_cast<T1>(tilingData_->param2));
        GetUintDivMagicAndShift(fastDivParam.magic3, fastDivParam.shift3, static_cast<T1>(tilingData_->param3));
        GetUintDivMagicAndShift(fastDivParam.magic4, fastDivParam.shift4, static_cast<T1>(tilingData_->param4));
        GetUintDivMagicAndShift(fastDivParam.magic5, fastDivParam.shift5, static_cast<T1>(tilingData_->param5));
        GetUintDivMagicAndShift(fastDivParam.magic6, fastDivParam.shift6, static_cast<T1>(tilingData_->param6));
    } else if constexpr (dimNum == DIM7) {
        GetUintDivMagicAndShift(fastDivParam.magic0, fastDivParam.shift0, static_cast<T1>(tilingData_->param0));
        GetUintDivMagicAndShift(fastDivParam.magic1, fastDivParam.shift1, static_cast<T1>(tilingData_->param1));
        GetUintDivMagicAndShift(fastDivParam.magic2, fastDivParam.shift2, static_cast<T1>(tilingData_->param2));
        GetUintDivMagicAndShift(fastDivParam.magic3, fastDivParam.shift3, static_cast<T1>(tilingData_->param3));
        GetUintDivMagicAndShift(fastDivParam.magic4, fastDivParam.shift4, static_cast<T1>(tilingData_->param4));
        GetUintDivMagicAndShift(fastDivParam.magic5, fastDivParam.shift5, static_cast<T1>(tilingData_->param5));
    } else if constexpr (dimNum == DIM6) {
        GetUintDivMagicAndShift(fastDivParam.magic0, fastDivParam.shift0, static_cast<T1>(tilingData_->param0));
        GetUintDivMagicAndShift(fastDivParam.magic1, fastDivParam.shift1, static_cast<T1>(tilingData_->param1));
        GetUintDivMagicAndShift(fastDivParam.magic2, fastDivParam.shift2, static_cast<T1>(tilingData_->param2));
        GetUintDivMagicAndShift(fastDivParam.magic3, fastDivParam.shift3, static_cast<T1>(tilingData_->param3));
        GetUintDivMagicAndShift(fastDivParam.magic4, fastDivParam.shift4, static_cast<T1>(tilingData_->param4));
    } else if constexpr (dimNum == DIM5) {
        GetUintDivMagicAndShift(fastDivParam.magic0, fastDivParam.shift0, static_cast<T1>(tilingData_->param0));
        GetUintDivMagicAndShift(fastDivParam.magic1, fastDivParam.shift1, static_cast<T1>(tilingData_->param1));
        GetUintDivMagicAndShift(fastDivParam.magic2, fastDivParam.shift2, static_cast<T1>(tilingData_->param2));
        GetUintDivMagicAndShift(fastDivParam.magic3, fastDivParam.shift3, static_cast<T1>(tilingData_->param3));
    } else if constexpr (dimNum == DIM4) {
        GetUintDivMagicAndShift(fastDivParam.magic0, fastDivParam.shift0, static_cast<T1>(tilingData_->param0));
        GetUintDivMagicAndShift(fastDivParam.magic1, fastDivParam.shift1, static_cast<T1>(tilingData_->param1));
        GetUintDivMagicAndShift(fastDivParam.magic2, fastDivParam.shift2, static_cast<T1>(tilingData_->param2));
    } else if constexpr (dimNum == DIM3) {
        GetUintDivMagicAndShift(fastDivParam.magic0, fastDivParam.shift0, static_cast<T1>(tilingData_->param0));
        GetUintDivMagicAndShift(fastDivParam.magic1, fastDivParam.shift1, static_cast<T1>(tilingData_->param1));
    } else if constexpr (dimNum == DIM2) {
        GetUintDivMagicAndShift(fastDivParam.magic0, fastDivParam.shift0, static_cast<T1>(tilingData_->param0));
    }

    LocalTensor<T1> simtParamLocal = simtBuf_.Get<T1>(SIMT_PARAM_NUM);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX0)) = static_cast<T1>(tilingData_->param0);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX1)) = static_cast<T1>(tilingData_->param1);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX2)) = static_cast<T1>(tilingData_->param2);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX3)) = static_cast<T1>(tilingData_->param3);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX4)) = static_cast<T1>(tilingData_->param4);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX5)) = static_cast<T1>(tilingData_->param5);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX6)) = static_cast<T1>(tilingData_->param6);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX7)) = static_cast<T1>(tilingData_->dim0);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX8)) = static_cast<T1>(tilingData_->dim1);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX9)) = static_cast<T1>(tilingData_->dim2);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX10)) = static_cast<T1>(tilingData_->dim3);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX11)) = static_cast<T1>(tilingData_->dim4);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX12)) = static_cast<T1>(tilingData_->dim5);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX13)) = static_cast<T1>(tilingData_->dim6);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX14)) = static_cast<T1>(tilingData_->dim7);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX15)) = static_cast<T1>(tilingData_->blockFactor);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX16)) = static_cast<T1>(tilingData_->inputSize);
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX17)) = fastDivParam.magic0;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX18)) = fastDivParam.magic1;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX19)) = fastDivParam.magic2;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX20)) = fastDivParam.magic3;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX21)) = fastDivParam.magic4;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX22)) = fastDivParam.magic5;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX23)) = fastDivParam.magic6;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX24)) = fastDivParam.shift0;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX25)) = fastDivParam.shift1;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX26)) = fastDivParam.shift2;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX27)) = fastDivParam.shift3;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX28)) = fastDivParam.shift4;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX29)) = fastDivParam.shift5;
    simtParamLocal(static_cast<uint32_t>(SIMT_PARAM_INDEX::SIMT_PARAM_INDEX30)) = fastDivParam.shift6;

    DataSyncBarrier<MemDsbT::UB>();
    Simt::VF_CALL<SimtCompute>(Simt::Dim3(LAUNCH_THREAD_NUM), (__gm__ T2*)(inputGm_.GetPhyAddr()), (__gm__ T2*)(outputGm_.GetPhyAddr()),
            blockIdx_, (__local_mem__ T1*)(simtParamLocal.GetPhyAddr()));
}
} // namespace ReverseV2

#endif // ASCENDC_REVERSE_V2_SIMT_H