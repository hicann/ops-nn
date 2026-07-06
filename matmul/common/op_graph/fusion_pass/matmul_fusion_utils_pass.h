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

#define FUSION_PASS_CHECK(condition, log_func, return_expr)                                                            \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (condition) {                                                                                               \
            log_func;                                                                                                  \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

namespace ops {

constexpr int64_t kX1InputIdx = 0;
constexpr int64_t kX2InputIdx = 1;
constexpr int64_t kBiasInputIdx = 2;
constexpr int64_t kOffsetWInputIdx = 3;
constexpr int64_t kCaptureTensorIdx = 0;
constexpr char kOpTypeMatMul[] = "MatMul";
constexpr char kOpTypeMatMulV2[] = "MatMulV2";
constexpr char kOpTypeBatchMatMul[] = "BatchMatMul";
constexpr char kOpTypeBatchMatMulV2[] = "BatchMatMulV2";

bool IsSupportL12BtBf16(const fe::PlatformInfo& platformInfo);

bool IsDav3510Platform(const std::string& passName);

bool CopyOtherAttrs(const ge::GNode& matchedNode, ge::GNode& v3Node, const std::string& passName);

ge::es::EsTensorHolder CreateMatMulLikeNode(ge::es::EsGraphBuilder& graphBuilder, const char* opType,
                                            const ge::es::EsTensorHolder& x1, const ge::es::EsTensorHolder& x2,
                                            const ge::es::EsTensorHolder& bias, const ge::es::EsTensorHolder& offsetW);

std::vector<ge::fusion::PatternUniqPtr> BuildMatMulPatterns(const std::string& prefix);
std::vector<ge::fusion::PatternUniqPtr> BuildMatMulV2Patterns(const std::string& prefix);
std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulPatterns(const std::string& prefix);
std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulV2Patterns(const std::string& prefix);

} // namespace ops

#endif // NN_MATMUL_FUSION_UTILS_PASS_H
