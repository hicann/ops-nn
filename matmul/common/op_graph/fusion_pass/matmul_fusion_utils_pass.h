/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef NN_MATMUL_FUSION_UTILS_PASS_H
#define NN_MATMUL_FUSION_UTILS_PASS_H

#include <string>
#include <type_traits>
#include <vector>

#include "graph/gnode.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "platform/platform_info.h"
#include "es_tensor_holder.h"

#define FUSION_PASS_CHECK(condition, log_func, return_expr)                                                      \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
    do {                                                                                                         \
        if (condition) {                                                                                         \
            log_func;                                                                                            \
            return_expr;                                                                                         \
        }                                                                                                        \
    } while (0)

namespace ops {

constexpr int64_t kX1InputIdx = 0;
constexpr int64_t kX2InputIdx = 1;
constexpr int64_t kBiasInputIdx = 2;
constexpr int64_t kOffsetWInputIdx = 3;
constexpr int64_t kBaseNodeNum = 2;
constexpr int64_t kThreeInputNum = 3;
constexpr int64_t kFourInputNum = 4;
constexpr int64_t kCaptureTensorIdx = 0;
constexpr char kOpTypeMatMul[] = "MatMul";
constexpr char kOpTypeMatMulV2[] = "MatMulV2";
constexpr char kOpTypeBatchMatMul[] = "BatchMatMul";
constexpr char kOpTypeBatchMatMulV2[] = "BatchMatMulV2";
constexpr char kAttrAdjX1[] = "adj_x1";
constexpr char kAttrAdjX2[] = "adj_x2";
constexpr char kAttrTransposeX1[] = "transpose_x1";
constexpr char kAttrTransposeX2[] = "transpose_x2";
constexpr char kAttrOffsetX[] = "offset_x";
constexpr int32_t kTargetGeCompilerVersion = 90100000;

bool IsSupportL12BtBf16(const fe::PlatformInfo& platformInfo);

bool IsNpuArch3510Series();

int32_t GetGeCompilerVersionNum();

ge::CustomPassStage GetCompatPassStage();

bool CopyOtherAttrs(const ge::GNode& matchedNode, ge::GNode& v3Node, const std::string& passName);

void ReportFusion(const std::vector<ge::GNode>& nodesBeforeFuse, const std::vector<ge::GNode>& nodesAfterFuse,
                  ge::CustomPassContext& passContext, const std::string& passName);

// 上报一次结构匹配（记 match_times, 不改变 effect_times）。Run 式 pass 在匹配到目标结构
// 但未生效/主动让位（如 QUANT_UNDEFINED）的场景调用, 对齐旧框架 match_times 全托管的语义。
// ReportMatch 为 GE 9.2.0+ 接口, 弱符号探测, 旧运行时自动跳过。
void ReportMatch(const std::vector<ge::GNode>& matchedNodes, ge::CustomPassContext& passContext,
                 const std::string& passName);

ge::es::EsTensorHolder CreateMatMulLikeNode(ge::es::EsGraphBuilder& graphBuilder, const char* opType,
                                            const ge::es::EsTensorHolder& x1, const ge::es::EsTensorHolder& x2,
                                            const ge::es::EsTensorHolder& bias, const ge::es::EsTensorHolder& offsetW);

std::vector<ge::fusion::PatternUniqPtr> BuildMatMulPatterns(const std::string& prefix);
std::vector<ge::fusion::PatternUniqPtr> BuildMatMulV2Patterns(const std::string& prefix);
std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulPatterns(const std::string& prefix);
std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulV2Patterns(const std::string& prefix);

bool GetTransposePermFromConst(const ge::GNode& transposeNode, std::vector<int64_t>& permValue);

// The dtype override described by the compile option (aligned with the canndev OpCustomizeDtype).
struct CustomizeDtype {
    std::vector<ge::DataType> inputDtypes;
    std::vector<ge::DataType> outputDtypes;
};

// Read the compile-time precision mode (atc --precision_mode, e.g. "force_fp16") from the pass
// context. Returns an empty string when the option is absent (e.g. in the UT environment), the
// caller skips the precision-reduction logic accordingly.
std::string GetPrecisionModeStr(const ge::CustomPassContext& passContext);

// Check whether the compile option customize_dtypes overrides the given selector (the operator
// type or the operator instance name). Returns false when no override is specified.
bool GetCustomizeDtypeBySelector(const ge::CustomPassContext& passContext, const std::string& selector);

} // namespace ops

#endif // NN_MATMUL_FUSION_UTILS_PASS_H
