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
 * \file weight_quant_batch_matmul_v2_asw_cmct.h
 * \brief ASW CMCT kernel 入口：类型组装 + Params 填充，计算委托给 matmul/common/cmct 公共库的
 *        wqbmm_asw 分层组件。
 *        仅支持 ND、batchA==batchB==batchC，不支持 antiquantOffset；
 *        K 须按 B 侧 C0 对齐（int8 为 32 元素），由 host 侧保证。
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "cmct/utils/common_utils.h"
#include "cmct/utils/integral_constant.h"
#include "cmct/policy/dispatch_policy.h"
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_mmad_a16w8_fixpipe_antiquant.h"
#include "cmct/kernel/kernel_wqbmm_asw_without_que.h"

namespace WeightQuantBatchMatmulV2::Arch35 {

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                        \
    template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans, \
              QuantType antiQuantType, bool hasAntiQuantOffset, QuantType quantType>
#define LOCAL_TEMPLATE_FUNC_PARAMS \
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType

LOCAL_TEMPLATE_CLASS_PARAMS
class WeightQuantBatchMatmulV2AswCmctKernel {
public:
    __aicore__ inline WeightQuantBatchMatmulV2AswCmctKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData);
    __aicore__ inline void Process();

protected:
    const wqbmmv2_tiling::WqbmmV2AswTilingData* tiling_ = nullptr;
    GM_ADDR xGm_ = nullptr;
    GM_ADDR weightGm_ = nullptr;
    GM_ADDR antiquantScaleGm_ = nullptr;
    GM_ADDR biasGm_ = nullptr;
    GM_ADDR yGm_ = nullptr;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2AswCmctKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData)
{
    if ASCEND_IS_AIV {
        return;
    }
    tiling_ = static_cast<const wqbmmv2_tiling::WqbmmV2AswTilingData*>(tilingData);
    xGm_ = x;
    weightGm_ = weight;
    antiquantScaleGm_ = antiquantScale;
    biasGm_ = bias;
    yGm_ = y;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2AswCmctKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    // mmad 不实现 antiquantOffset 路径，host 侧保证 ASW 场景无 offset
    static_assert(!hasAntiQuantOffset, "WQBMMV2 ASW cmct kernel does not support antiquantOffset");
    // antiQuantType 映射 AntiQuantMode
    static constexpr Cmct::Gemm::WqbmmAntiQuantMode antiQuantMode = (antiQuantType == QuantType::PER_TENSOR) ?
                                                                        Cmct::Gemm::WqbmmAntiQuantMode::PER_TENSOR :
                                                                        Cmct::Gemm::WqbmmAntiQuantMode::PER_CHANNEL;
    // L1/L0 tile shape 由 tiling 动态给出
    using L1TileShape = AscendC::Shape<Cmct::Gemm::_0, Cmct::Gemm::_0, Cmct::Gemm::_0>;
    using L0TileShape = AscendC::Shape<Cmct::Gemm::_0, Cmct::Gemm::_0, Cmct::Gemm::_0>;
    // 调度器标签 + 编译期 policy
    using BlockScheduler = Cmct::Gemm::WqbmmAswtScheduler;
    using DispatchPolicy = Cmct::Gemm::WqbmmMatmulWithoutQuePolicy<
        AscendC::Shape<Cmct::Gemm::_0, Cmct::Gemm::_0, Cmct::Gemm::_0, Cmct::Gemm::_0>, antiQuantMode>;
    // block mmad 与 kernel 层直接组装（ASW 仅 ND 格式，转置标记由模板参数给出）
    using ATensorType = Cmct::Gemm::Block::WqbmmTensorType<xType, aTrans>;
    using BTensorType = Cmct::Gemm::Block::WqbmmTensorType<wType, bTrans>;
    using CTensorType = Cmct::Gemm::Block::WqbmmTensorType<yType, false>;
    using BiasTensorType = Cmct::Gemm::Block::WqbmmTensorType<biasType, false>;
    using BlockMmad = Cmct::Gemm::Block::WqbmmBlockMmad<DispatchPolicy, L1TileShape, L0TileShape, ATensorType,
                                                        BTensorType, CTensorType, BiasTensorType>;
    using MatmulKernel = Cmct::Gemm::Kernel::WqbmmKernelMatmul<Cmct::Gemm::MatmulShape, BlockMmad, BlockScheduler,
                                                               L1TileShape, L0TileShape>;
    using Params = typename MatmulKernel::template Params<wqbmmv2_tiling::WqbmmV2AswBasicTilingData>;
    const wqbmmv2_tiling::WqbmmV2AswBasicTilingData& basicTiling = tiling_->matMulTilingData;
    // L2 cache 关闭枚举 -> bool（比较逻辑留在算子侧，公共库不依赖本算子枚举类型）
    using L2Mode = wqbmmv2_tiling::L2CacheMode;
    const auto l2CacheDisable = basicTiling.l2CacheDisable;
    const bool aL2CacheDisable = (l2CacheDisable == L2Mode::ALL_L2_CACHE_DISABLE ||
                                  l2CacheDisable == L2Mode::A_L2_CACHE_DISABLE);
    const bool bL2CacheDisable = (l2CacheDisable == L2Mode::ALL_L2_CACHE_DISABLE ||
                                  l2CacheDisable == L2Mode::B_L2_CACHE_DISABLE);
    Params params = {{static_cast<int64_t>(basicTiling.m), static_cast<int64_t>(basicTiling.n),
                      static_cast<int64_t>(basicTiling.k), static_cast<int64_t>(tiling_->batchDimAll)}, // shape
                     {xGm_, weightGm_, yGm_, biasGm_, antiquantScaleGm_},                               // gm addr
                     {&basicTiling, aL2CacheDisable, bL2CacheDisable}}; // scheduler params
    MatmulKernel mm;
    mm(params);
}

} // namespace WeightQuantBatchMatmulV2::Arch35
