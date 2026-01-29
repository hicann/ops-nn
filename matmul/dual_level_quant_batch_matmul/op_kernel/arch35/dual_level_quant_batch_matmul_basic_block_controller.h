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
 * \file dual_level_quant_batch_matmul_basic_block_controller.h
 * \brief
 */

#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_CONTROLLER_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_CONTROLLER_H

#include "../dual_level_quant_batch_matmul_tiling_data.h"
#include "dual_level_quant_batch_matmul_basic_block.h"
#include "dual_level_quant_batch_matmul_block.h"
#include "dual_level_quant_batch_matmul_cube_compute_tools.h"
#include "tool_arch35.h"

using AscendC::GetBlockIdx;

namespace DualLevelQuantBatchMatmul::Arch35 {

static constexpr uint64_t L0_BASE_M = 128;
static constexpr uint64_t L0_BASE_N = 128;

LOCAL_TEMPLATE_CLASS_PARAMS
class DualLevelQuantBatchMatmulBasicBlockController {
public:
    __aicore__ inline DualLevelQuantBatchMatmulBasicBlockController() = default;

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Level0Scale, GM_ADDR x1Level1Scale, GM_ADDR x2Level0Scale,
        GM_ADDR x2Level1Scale, GM_ADDR bias, GM_ADDR y, const DualLevelQuantBatchMatmulBasicTilingData* tilingData);

    __aicore__ inline void Process();

private:
    DualLevelQuantBatchMatmulBaseBlock block_;
    DualLevelQuantMatmulBasicBlock<LOCAL_TEMPLATE_FUNC_PARAMS> basicBlock_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantBatchMatmulBasicBlockController<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Level0Scale, GM_ADDR x1Level1Scale, GM_ADDR x2Level0Scale, GM_ADDR x2Level1Scale,
    GM_ADDR bias, GM_ADDR y, const DualLevelQuantBatchMatmulBasicTilingData* tilingData)
{
    uint64_t blockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx = blockIdx / AscendC::GetTaskRation();
    }
    block_.Init(tilingData, blockIdx);
    basicBlock_.Init(
        reinterpret_cast<__gm__ x1Type*>(x1), reinterpret_cast<__gm__ x2Type*>(x2),
        reinterpret_cast<__gm__ x1Level0ScaleType*>(x1Level0Scale),
        reinterpret_cast<__gm__ x1Level1ScaleType*>(x1Level1Scale),
        reinterpret_cast<__gm__ x2Level0ScaleType*>(x2Level0Scale),
        reinterpret_cast<__gm__ x2Level1ScaleType*>(x2Level1Scale), reinterpret_cast<__gm__ biasType*>(bias),
        reinterpret_cast<__gm__ yType*>(y), tilingData);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantBatchMatmulBasicBlockController<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if (block_.blockIdx_ >= block_.tiling_->usedCoreNum) {
        return;
    }

    L0CopyAndCalcParams l0Params;
    // 这里需要验证：没有k轴累加，并且每次unit写回
    l0Params.isFirstKLoop = true;
    l0Params.isLastKLoop = true;

    basicBlock_.SetAivToAic();
    for (uint64_t roundIdx = 0; roundIdx < block_.params_.round; ++roundIdx) {
        block_.UpdateBasicIndex(roundIdx);
        block_.UpdateBlockParams(roundIdx);
        if (block_.offset_.singleCoreM <= 0 || block_.offset_.singleCoreN <= 0) {
            continue;
        }
        block_.CalcGMOffset();

        l0Params.mL1Size = block_.offset_.singleCoreM;
        l0Params.nL1Size = block_.offset_.singleCoreN;
        // 固定128的二级切分，后续考虑动态切分
        l0Params.mL0Size = DualLevelQuantBatchMatmul::Arch35::Min<uint64_t>(L0_BASE_M, block_.offset_.singleCoreM);
        l0Params.nL0Size = DualLevelQuantBatchMatmul::Arch35::Min<uint64_t>(L0_BASE_N, block_.offset_.singleCoreN);
        basicBlock_.ComputeBasicBlock(block_.offset_, l0Params);
    }
    basicBlock_.WaitAivToAic();
    basicBlock_.End();
}

} // namespace DualLevelQuantBatchMatmul::Arch35
#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_CONTROLLER_H