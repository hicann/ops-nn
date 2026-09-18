/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CONV_BP_DLOAD_H
#define CONV_BP_DLOAD_H

#include "conv_bp_dload_config.h"
#include "conv_bp_dload_compute.h"
#include "../util/conv_bp_common_data_blocks.h" // 蛇形分核 BlockIterator（原空壳转发头已删除）

namespace BpDLoad {

// DLoad 顶层串联类：DLoadCompute（单块 K 全载计算）+ 蛇形分核块走位（BpUtils::BlockIterator）。
// 调用契约（顺序不可变；三个 GM 地址均裸传，调用侧不构造 GlobalTensor）：
//   1. Init 一次：fmapGm/dyGm + config + blockNum（0 = GetBlockNum()；直调/UT 场景
//      <<<blockNum>>> 寄存器可能未按 aclnn 语义填充，显式传入更稳）
//   2. Process 一次：yGm（[cout][cin][dhwK] 全局 ND，恒 fp32），内部完成全部块循环
//   3. End 必须调用：消费背压残留 Set，否则残留 flag 污染同核后续 kernel
template <typename SrcT>
class ConvBackpropFilterDLoad {
public:
    inline __aicore__ void Init(GM_ADDR fmapGm, GM_ADDR dyGm, const DLoadConfig& config, uint32_t blockNum = 0)
    {
        config_ = config;
        blockNum_ = blockNum;
        // GM 地址即用即传（compute 类自持 GlobalTensor）
        AscendC::GlobalTensor<SrcT> fmap;
        fmap.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(fmapGm));
        AscendC::GlobalTensor<SrcT> dy;
        dy.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(dyGm));
        computer_.Init(fmap, dy, config_);
    }

    inline __aicore__ void Process(GM_ADDR yGm)
    {
        // 纯 Cube 模板：AIV 核直接跳出（块循环不得在 AIV 上执行）
        if ASCEND_IS_AIV {
            return;
        }
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yGm));
        // 蛇形分核（IterDir=CIN：cout 块沿拓扑 H、cin 块沿 W，L2 友好）；空块跳过
        // （valid=false 时 length=0，IterateK 不调用，事件链不受影响）
        auto blockIter = BpUtils::BlockIterator<BpUtils::BlockIterDirection::CIN>::Create(
            false, config_.shape.cout, config_.shape.cin, config_.tiling.singleShapeAligned16Cout,
            config_.tiling.singleShapeAligned16Cin, blockNum_);
        while (blockIter.More()) {
            BpUtils::CoutCinRange cRange;
            if (blockIter.GetLocalBlock(cRange)) {
                computer_.IterateK(config_, cRange, y_, CalcYBase(cRange));
            }
            blockIter.Next();
        }
    }

    inline __aicore__ void End() { computer_.End(); }

private:
    // 本块 y 基址 = coutIdx*cin*dhwK + cinIdx*dhwK（生产布局，与 host golden 同式）
    inline __aicore__ uint64_t CalcYBase(const BpUtils::CoutCinRange& cRange) const
    {
        const uint32_t dhwk = config_.shape.dk * config_.shape.hk * config_.shape.wk;
        return static_cast<uint64_t>(cRange.coutIdx) * config_.shape.cin * dhwk +
               static_cast<uint64_t>(cRange.cinIdx) * dhwk;
    }

    DLoadConfig config_ = {};
    uint32_t blockNum_ = 0;
    AscendC::GlobalTensor<float> y_; // y 输出恒 fp32
    DLoadCompute<SrcT> computer_;
};

} // namespace BpDLoad
#endif // CONV_BP_DLOAD_H
