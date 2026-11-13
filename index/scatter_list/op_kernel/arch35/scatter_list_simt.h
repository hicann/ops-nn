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
 * \file scatter_list_simt.h
 * \brief SIMT kernel implementation for scatter_list (arch35 / A5)
 *
 * ScatterList 是纯搬运算子(reduce 仅支持 update)。本实现以元素为线程粒度, 单一场景覆盖
 * 全部形态 —— 更新轴位置、indice 维度、mask 有无、各 dtype 均为运行时参数或模板参数,
 * 不再按 UB 容量档与对齐性分裂实现:
 *
 *   Process()                — scalar 域入口, 解析张量列表并启动 VF
 *   OpScatterListSimtKernel()— VF kernel, 对 updates 全元素做 Grid-Stride 循环
 *   ProcessOneElement()      — 单元素: 索引分解 -> mask/indice 过滤 -> 边界判定 -> 搬运
 *
 * 线程取元素粒度而非段粒度的理由: 相邻线程处理相邻尾维元素, 访存天然合并; 并行度等于
 * updates 元素总量, 不受段长与段数分布影响(段长 1 元素的极端形态同样满载)。相邻线程
 * 相邻元素是访存合并的前提, 因此不改为"每线程一段连续元素"。
 */

#ifndef SCATTER_LIST_ARCH35_SIMT_H
#define SCATTER_LIST_ARCH35_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "scatter_list_tiling_data.h"

namespace NsScatterListArch35 {
using namespace AscendC;

// VF 线程数, 编译期常量
static constexpr uint32_t THREAD_NUM = 1024;
// indice 为 2 维时每个张量占两个元素: [轴上偏移, 实际更新长度]
static constexpr int64_t INDICE_PAIR = 2;
// 元素总量不超过该值时用 32 位做索引分解(整除序列更短), 否则退 64 位
static constexpr int64_t IDX32_LIMIT = 2147483647;

/*!
 * \brief 单元素搬运
 *
 * 线性索引按 updates 的存储序分解:
 *   idx = ((tensorIdx * leadNum + leadIdx) * updateAxisLen + axisIdx) * trailLen + trailIdx
 * 写入位置:
 *   var[tensorIdx][(leadIdx * varAxisLen + axisOffset + axisIdx) * trailLen + trailIdx]
 *
 * 该式对任意合法更新轴统一成立: 更新轴为末轴时 trailLen 退化为 1, varAxisLen 即末维长度,
 * 偏移自然落在行内, 无需另设坐标系。
 *
 * \tparam IdxT 索引分解用的整型。各维长度与元素总量由 tiling 保证在该类型范围内; 但
 *              indice 给出的轴上偏移是用户数据(可为任意值, 含极端负值), 故偏移与边界
 *              判定一律用 int64_t, 避免溢出使判定失效。
 */
template <typename T, typename TIdx, typename IdxT>
__simt_callee__ inline void ProcessOneElement(IdxT linearIdx, IdxT leadNum, IdxT updateAxisLen, IdxT varAxisLen,
                                              IdxT trailLen, int32_t indiceDims, int32_t hasMask,
                                              __gm__ uint64_t* varAddrArray, __gm__ TIdx* indiceGm, __gm__ T* updatesGm,
                                              __gm__ uint8_t* maskGm)
{
    const IdxT trailIdx = linearIdx % trailLen;
    const IdxT axisRest = linearIdx / trailLen;
    const IdxT axisIdx = axisRest % updateAxisLen;
    const IdxT leadRest = axisRest / updateAxisLen;
    const IdxT leadIdx = leadRest % leadNum;
    const IdxT tensorIdx = leadRest / leadNum;

    // mask 过滤: 未被选中的张量整体不更新
    if (hasMask != 0 && maskGm[tensorIdx] == 0) {
        return;
    }

    int64_t axisOffset = 0;
    if (indiceDims == static_cast<int32_t>(INDICE_PAIR)) {
        axisOffset = static_cast<int64_t>(indiceGm[tensorIdx * INDICE_PAIR]);
        // 2 维 indice 下各张量更新长度可不同, 超出有效长度的部分不写
        if (static_cast<int64_t>(axisIdx) >= static_cast<int64_t>(indiceGm[tensorIdx * INDICE_PAIR + 1])) {
            return;
        }
    } else {
        axisOffset = static_cast<int64_t>(indiceGm[tensorIdx]);
    }

    // 边界判定落到单个元素: 偏移越界时只跳过越界元素, 张量内的前缀照常写入。
    // 三个因子各自在界内, 故平坦写地址必然落在该张量元素总量之内。
    const int64_t varAxisIdx = axisOffset + static_cast<int64_t>(axisIdx);
    if (varAxisIdx < 0 || varAxisIdx >= static_cast<int64_t>(varAxisLen)) {
        return;
    }

    __gm__ T* varGm = reinterpret_cast<__gm__ T*>(*(varAddrArray + tensorIdx));
    const int64_t dstIdx = (static_cast<int64_t>(leadIdx) * static_cast<int64_t>(varAxisLen) + varAxisIdx) *
                               static_cast<int64_t>(trailLen) +
                           static_cast<int64_t>(trailIdx);
    varGm[dstIdx] = updatesGm[linearIdx];
}

/*!
 * \brief VF kernel — 对 updates 全元素做 Grid-Stride 循环, 线程间无协作
 */
template <typename T, typename TIdx, typename IdxT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpScatterListSimtKernel(
    IdxT totalElements, IdxT leadNum, IdxT updateAxisLen, IdxT varAxisLen, IdxT trailLen, int32_t indiceDims,
    int32_t hasMask, __gm__ uint64_t* varAddrArray, __gm__ TIdx* indiceGm, __gm__ T* updatesGm, __gm__ uint8_t* maskGm)
{
    // 循环变量一律用 int64: 元素总量可以贴着 IdxT 的上界(实测有 totalElements =
    // 2^31 - 34 的形态), 此时末轮 idx += stride 会越过 IdxT 上界而回绕成负数,
    // 循环条件 idx < totalElements 永远成立 -> 死循环, kernel 不返回。
    // 索引分解仍按 IdxT 做(idx < totalElements <= IdxT 上界, 窄化安全)。
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    const int64_t total = static_cast<int64_t>(totalElements);
    for (int64_t idx =
             static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
         idx < total; idx += stride) {
        ProcessOneElement<T, TIdx, IdxT>(static_cast<IdxT>(idx), leadNum, updateAxisLen, varAxisLen, trailLen,
                                         indiceDims, hasMask, varAddrArray, indiceGm, updatesGm, maskGm);
    }
}

/*!
 * \brief scalar 域入口 — 解析张量列表并启动 VF
 * \tparam T     var / updates 元素类型(DTYPE_VAR)
 * \tparam TIdx  indice 元素类型(DTYPE_INDICE)
 */
template <typename T, typename TIdx>
__aicore__ inline void Process(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
                               const ScatterListArch35TilingData* tilingData)
{
    const int64_t totalElements = tilingData->totalElements;
    const int64_t leadNum = tilingData->leadNum;
    const int64_t updateAxisLen = tilingData->updateAxisLen;
    const int64_t varAxisLen = tilingData->varAxisLen;
    const int64_t trailLen = tilingData->trailLen;

    // tiling 已保证各维为正; 此处兜底, 避免除零与空网格
    if (totalElements <= 0 || leadNum <= 0 || updateAxisLen <= 0 || trailLen <= 0) {
        return;
    }

    // 张量列表布局: 首 8 字节为地址数组相对列表首地址的字节偏移, 其后为各张量首地址。
    // 这一跳在标量域解一次即可; 放在 VF 里会让每个元素都背上"读偏移再读地址"的串行
    // 访存依赖链(后一次读的地址取决于前一次读的结果)。
    __gm__ uint64_t* listBase = reinterpret_cast<__gm__ uint64_t*>(var);
    // 右移 3 位即除以 sizeof(uint64_t)
    __gm__ uint64_t* varAddrArray = listBase + (*listBase >> 3);

    __gm__ TIdx* indiceGm = reinterpret_cast<__gm__ TIdx*>(indice);
    __gm__ T* updatesGm = reinterpret_cast<__gm__ T*>(updates);
    __gm__ uint8_t* maskGm = reinterpret_cast<__gm__ uint8_t*>(mask);

    if (totalElements <= IDX32_LIMIT) {
        asc_vf_call<OpScatterListSimtKernel<T, TIdx, int32_t>>(
            dim3(THREAD_NUM), static_cast<int32_t>(totalElements), static_cast<int32_t>(leadNum),
            static_cast<int32_t>(updateAxisLen), static_cast<int32_t>(varAxisLen), static_cast<int32_t>(trailLen),
            tilingData->indiceDims, tilingData->hasMask, varAddrArray, indiceGm, updatesGm, maskGm);
    } else {
        asc_vf_call<OpScatterListSimtKernel<T, TIdx, int64_t>>(
            dim3(THREAD_NUM), totalElements, leadNum, updateAxisLen, varAxisLen, trailLen, tilingData->indiceDims,
            tilingData->hasMask, varAddrArray, indiceGm, updatesGm, maskGm);
    }
}

} // namespace NsScatterListArch35
#endif // SCATTER_LIST_ARCH35_SIMT_H
