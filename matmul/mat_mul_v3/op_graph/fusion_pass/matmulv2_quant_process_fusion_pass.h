/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATMULV2_QUANT_PROCESS_FUSION_PASS_H_
#define MATMULV2_QUANT_PROCESS_FUSION_PASS_H_

#include "fusion_pass/bias_optimize_quant_rollback_base.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
/*
 * 说明: CheckOpSupported 沿用基类现状——主线 c2d029f185 起 GE CheckNodeSupportOnAicore
 * 接口存在兼容性问题, 基类默认已改为恒返回 true（详见基类 UT 注释）。本算子实现齐全
 * （MatMulV2 量化组合平台可跑）, 与同家族 conv3d/conv2dt 一致不覆写; QUANT_ROLLBACK 分支
 * 保留语义完整性（UT 注入 Probe 覆写验证）, 生产环境当前不可达。dwconv2d 的覆写系
 * "950 包缺 depthwise TBE 实现"的特例兜底, 不适用本算子。
 * （fusion pattern 结构图见 .cpp 文件头, 对齐 canndev 原版位置）
 */
class __attribute__((visibility("default"))) MatmulV2QuantProcessFusionPass : public ge::fusion::FusionBasePass,
                                                                              public BiasOptimizeQuantRollbackBase {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    ge::Status GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co) override;

    ge::Status GetQuantProcessMode(const ge::GNodePtr& quantNode, const ge::GNodePtr& cubeNode,
                                   QuantProcessMode& quantProcessMode) override;
};
} // namespace ops

#endif // MATMULV2_QUANT_PROCESS_FUSION_PASS_H_
