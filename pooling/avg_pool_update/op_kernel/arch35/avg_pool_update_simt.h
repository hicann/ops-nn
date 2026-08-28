/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_update_simt.h
 * \brief avg_pool_update 算子 SIMT kernel
 *
 * Grid-Stride 模式：每个线程按步长迭代输出元素，将线性索引分解为 (h, w) 坐标，
 * 动态计算 mean_matrix，执行逐元素除法 y = x1 / mean。
 *
 * UB 传参方案：所有 int64_t/int32_t 标量参数（含 magic/shift）通过 UB 传递（uint64_t 存储）。
 * ⚠️ UB 指针必须放在第 2 个参数位置（紧跟 totalNum），其后只允许 GM 指针，
 * 否则 UB 指针地址会被损坏（参考 crop_and_resize/split_v/broadcast_add 实现）。
 *
 * 性能优化：UintDiv 快速除法替代硬件 % 和 /；IDX_T 模板（uint32_t/uint64_t）
 * 按 totalNum 范围选择 32/64 位路径减少寄存器压力。
 */

#ifndef AVG_POOL_UPDATE_SIMT_H_
#define AVG_POOL_UPDATE_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "avg_pool_update_tiling_data.h"
#include "avg_pool_update_tiling_key.h"

namespace NsAvgPoolUpdate {
using namespace AscendC;

// uint32_t 路径 1024 线程，uint64_t 路径 512 线程（减少寄存器压力）
static constexpr uint32_t THREADS_U32 = 1024;
static constexpr uint32_t THREADS_U64 = 512;
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == 4) ? THREADS_U32 : THREADS_U64;

// UB 偏移常量：18 个参数各占 1 个 uint64_t slot，对齐到 20 slot = 160B（32B 对齐）
static constexpr uint32_t OFF_OUT_H = 0;
static constexpr uint32_t OFF_OUT_W = 1;
static constexpr uint32_t OFF_OUT_C = 2;
static constexpr uint32_t OFF_INPUT_H = 3;
static constexpr uint32_t OFF_INPUT_W = 4;
static constexpr uint32_t OFF_KH = 5;
static constexpr uint32_t OFF_KW = 6;
static constexpr uint32_t OFF_STRIDE_H = 7;
static constexpr uint32_t OFF_STRIDE_W = 8;
static constexpr uint32_t OFF_PAD_T = 9;
static constexpr uint32_t OFF_PAD_B = 10;
static constexpr uint32_t OFF_PAD_L = 11;
static constexpr uint32_t OFF_PAD_R = 12;
static constexpr uint32_t OFF_IS_NHWC = 13;
static constexpr uint32_t OFF_MAGIC_W = 14;
static constexpr uint32_t OFF_SHIFT_W = 15;
static constexpr uint32_t OFF_MAGIC_H = 16;
static constexpr uint32_t OFF_SHIFT_H = 17;
static constexpr uint32_t UB_PARAM_COUNT = 20; // 18 个参数，对齐到 20（160B）

// mean_dim = min(min(idx*stride - padBefore + kDim, (outDim-1-idx)*stride - padAfter + kDim), min(kDim, inputDim))
// 保证 mean_dim >= 1（防止 CALCULATED padding 模式下除零）
template <typename IDX_T>
__simt_callee__ inline int64_t ComputeMeanDim(IDX_T idx, int64_t outDim, int64_t stride, int64_t kDim, int64_t inputDim,
                                              int64_t padBefore, int64_t padAfter)
{
    int64_t frontOverlap = static_cast<int64_t>(idx) * stride - padBefore + kDim;
    int64_t backOverlap = (outDim - 1 - static_cast<int64_t>(idx)) * stride - padAfter + kDim;
    int64_t mean = frontOverlap < backOverlap ? frontOverlap : backOverlap;
    int64_t kInput = kDim < inputDim ? kDim : inputDim;
    mean = mean < kInput ? mean : kInput;
    mean = mean < 1 ? 1 : mean;
    return mean;
}

// 坐标分解：将线性 index 分解为 (h, w) 坐标
// NHWC: index = n*(outH*outW*C) + h*(outW*C) + w*C + c
// NCHW: index = n*(C*outH*outW) + c*(outH*outW) + h*outW + w
template <typename IDX_T>
__simt_callee__ inline void DecomposeHwCoords(IDX_T index, IDX_T outWIdx, IDX_T outHIdx, IDX_T outCIdx, IDX_T magicW,
                                              IDX_T shiftW, IDX_T magicH, IDX_T shiftH, int64_t isNhwc, IDX_T& h,
                                              IDX_T& w)
{
    if (isNhwc) {
        // outC 未预计算 magic/shift（UB slot 预算限制），使用硬件除法
        IDX_T hwIndex = index / outCIdx;
        IDX_T hwQuotient = Simt::UintDiv<IDX_T>(hwIndex, magicW, shiftW);
        w = hwIndex - hwQuotient * outWIdx;
        IDX_T nc = Simt::UintDiv<IDX_T>(hwQuotient, magicH, shiftH);
        h = hwQuotient - nc * outHIdx;
    } else {
        IDX_T hwQuotient = Simt::UintDiv<IDX_T>(index, magicW, shiftW);
        w = index - hwQuotient * outWIdx;
        IDX_T nc = Simt::UintDiv<IDX_T>(hwQuotient, magicH, shiftH);
        h = hwQuotient - nc * outHIdx;
    }
}

// VF kernel：⚠️ UB 指针必须放在第 2 个参数位置（紧跟 totalNum），其后只允许 GM 指针
template <typename T, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpAvgPoolUpdateSimt(IDX_T totalNum,
                                                                                         __ubuf__ uint64_t* ub,
                                                                                         __gm__ T* x1Gm, __gm__ T* yGm)
{
    // 循环外一次性从 UB 读取所有标量参数
    int64_t outH = static_cast<int64_t>(ub[OFF_OUT_H]);
    int64_t outW = static_cast<int64_t>(ub[OFF_OUT_W]);
    int64_t outC = static_cast<int64_t>(ub[OFF_OUT_C]);
    int64_t inputH = static_cast<int64_t>(ub[OFF_INPUT_H]);
    int64_t inputW = static_cast<int64_t>(ub[OFF_INPUT_W]);
    int64_t kH = static_cast<int64_t>(ub[OFF_KH]);
    int64_t kW = static_cast<int64_t>(ub[OFF_KW]);
    int64_t strideH = static_cast<int64_t>(ub[OFF_STRIDE_H]);
    int64_t strideW = static_cast<int64_t>(ub[OFF_STRIDE_W]);
    int64_t padT = static_cast<int64_t>(ub[OFF_PAD_T]);
    int64_t padB = static_cast<int64_t>(ub[OFF_PAD_B]);
    int64_t padL = static_cast<int64_t>(ub[OFF_PAD_L]);
    int64_t padR = static_cast<int64_t>(ub[OFF_PAD_R]);
    int64_t isNhwc = static_cast<int64_t>(ub[OFF_IS_NHWC]);
    IDX_T magicW = static_cast<IDX_T>(ub[OFF_MAGIC_W]);
    IDX_T shiftW = static_cast<IDX_T>(ub[OFF_SHIFT_W]);
    IDX_T magicH = static_cast<IDX_T>(ub[OFF_MAGIC_H]);
    IDX_T shiftH = static_cast<IDX_T>(ub[OFF_SHIFT_H]);

    IDX_T outWIdx = static_cast<IDX_T>(outW);
    IDX_T outHIdx = static_cast<IDX_T>(outH);
    IDX_T outCIdx = static_cast<IDX_T>(outC);

    for (IDX_T index = static_cast<IDX_T>(blockIdx.x) * blockDim.x + threadIdx.x; index < totalNum;
         index += static_cast<IDX_T>(blockDim.x) * gridDim.x) {
        // 坐标分解（mean_matrix 仅依赖 (h, w)，与 n/c 无关）
        IDX_T h, w;
        DecomposeHwCoords<IDX_T>(index, outWIdx, outHIdx, outCIdx, magicW, shiftW, magicH, shiftH, isNhwc, h, w);

        // mean_h/mean_w 计算（int64_t 中间变量保证计数精度），防止除零 max(mean, 1)
        int64_t meanH = ComputeMeanDim<IDX_T>(h, outH, strideH, kH, inputH, padT, padB);
        int64_t meanW = ComputeMeanDim<IDX_T>(w, outW, strideW, kW, inputW, padL, padR);

        // mean = mean_h * mean_w，cast 为输入 dtype
        int64_t mean = meanH * meanW;
        T meanVal = static_cast<T>(static_cast<float>(mean));

        // 逐元素除法 y = x1 / mean
        T x1Val = x1Gm[index];
        yGm[index] = x1Val / meanVal;
    }
}

// magic/shift 计算写入 UB（GetUintDivMagicAndShift 是 device 端 API）
template <typename IDX_T>
__aicore__ inline void FillMagicShift(LocalTensor<uint64_t> ubTensor, int64_t outW, int64_t outH)
{
    IDX_T magicW = 0, shiftW = 0, magicH = 0, shiftH = 0;
    GetUintDivMagicAndShift<IDX_T>(magicW, shiftW, static_cast<IDX_T>(outW));
    GetUintDivMagicAndShift<IDX_T>(magicH, shiftH, static_cast<IDX_T>(outH));
    ubTensor.SetValue(OFF_MAGIC_W, static_cast<uint64_t>(magicW));
    ubTensor.SetValue(OFF_SHIFT_W, static_cast<uint64_t>(shiftW));
    ubTensor.SetValue(OFF_MAGIC_H, static_cast<uint64_t>(magicH));
    ubTensor.SetValue(OFF_SHIFT_H, static_cast<uint64_t>(shiftH));
}

// asc_vf_call 启动逻辑
template <typename T, typename IDX_T>
__aicore__ inline void LaunchAvgPoolUpdateKernel(int64_t totalNum, __gm__ T* x1Gm, __gm__ T* yGm, __ubuf__ uint64_t* ub)
{
    asc_vf_call<OpAvgPoolUpdateSimt<T, IDX_T>>(dim3(THREADS<IDX_T>), static_cast<IDX_T>(totalNum), ub, x1Gm, yGm);
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(AvgPoolUpdateTilingData, tilingData, tiling);
    __gm__ T* x1Gm = (__gm__ T*)x1;
    __gm__ T* yGm = (__gm__ T*)y;
    // x2 不参与 kernel 计算（仅 Tiling 阶段用其 shape）

    int64_t totalNum = tilingData.totalNum;

    // tiling 阶段已拒绝空 shape（totalNum==0 报错），kernel 无需空保护
    // 标量参数通过 UB 传递到 VF
    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<uint64_t> ubTensor = ubAlloc.Alloc<uint64_t>(UB_PARAM_COUNT);
    ubTensor.SetValue(OFF_OUT_H, static_cast<uint64_t>(tilingData.outH));
    ubTensor.SetValue(OFF_OUT_W, static_cast<uint64_t>(tilingData.outW));
    ubTensor.SetValue(OFF_OUT_C, static_cast<uint64_t>(tilingData.outC));
    ubTensor.SetValue(OFF_INPUT_H, static_cast<uint64_t>(tilingData.inputH));
    ubTensor.SetValue(OFF_INPUT_W, static_cast<uint64_t>(tilingData.inputW));
    ubTensor.SetValue(OFF_KH, static_cast<uint64_t>(tilingData.kH));
    ubTensor.SetValue(OFF_KW, static_cast<uint64_t>(tilingData.kW));
    ubTensor.SetValue(OFF_STRIDE_H, static_cast<uint64_t>(tilingData.strideH));
    ubTensor.SetValue(OFF_STRIDE_W, static_cast<uint64_t>(tilingData.strideW));
    ubTensor.SetValue(OFF_PAD_T, static_cast<uint64_t>(tilingData.padT));
    ubTensor.SetValue(OFF_PAD_B, static_cast<uint64_t>(tilingData.padB));
    ubTensor.SetValue(OFF_PAD_L, static_cast<uint64_t>(tilingData.padL));
    ubTensor.SetValue(OFF_PAD_R, static_cast<uint64_t>(tilingData.padR));
    ubTensor.SetValue(OFF_IS_NHWC, static_cast<uint64_t>(tilingData.isNhwc));

    // 计算 magic/shift 并写入 UB
    if (totalNum <= static_cast<int64_t>(INT32_MAX)) {
        FillMagicShift<uint32_t>(ubTensor, tilingData.outW, tilingData.outH);
    } else {
        FillMagicShift<uint64_t>(ubTensor, tilingData.outW, tilingData.outH);
    }

    DataSyncBarrier<MemDsbT::UB>();

    __ubuf__ uint64_t* ubParams = (__ubuf__ uint64_t*)ubTensor.GetPhyAddr();

    // uint32/uint64 路径选择
    if (totalNum <= static_cast<int64_t>(INT32_MAX)) {
        LaunchAvgPoolUpdateKernel<T, uint32_t>(totalNum, x1Gm, yGm, ubParams);
    } else {
        LaunchAvgPoolUpdateKernel<T, uint64_t>(totalNum, x1Gm, yGm, ubParams);
    }
}
} // namespace NsAvgPoolUpdate
#endif // AVG_POOL_UPDATE_SIMT_H_
