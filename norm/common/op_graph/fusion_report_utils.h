/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NORM_FUSION_REPORT_UTILS_H_
#define OPS_NORM_FUSION_REPORT_UTILS_H_

#include <vector>
#include "graph/graph.h"
#include "ge/fusion/subgraph_boundary.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

// Keep these declarations aligned with GE. Do not include graph_rewriter.h or
// graph_fuse_inspector_utils.h in the same translation unit.
namespace ge {
namespace fusion {
class SubgraphRewriter {
public:
    static Status Replace(const SubgraphBoundary& boundary, const Graph& replacement);
    static Status Replace(const SubgraphBoundary& boundary, Graph&& replacement);
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90100000U)
    static Status Replace(const SubgraphBoundary& boundary, const Graph& replacement, CustomPassContext& context)
        __attribute__((weak));
    static Status Replace(const SubgraphBoundary& boundary, Graph&& replacement, CustomPassContext& context)
        __attribute__((weak));
#endif
};

#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90100000U)
class GraphFuseInspectorUtils {
public:
    static bool CanFuse(const std::vector<GNode>& nodes, AscendString& failedReason) __attribute__((weak));
    static Status ReportFuse(const std::vector<GNode>& before, const std::vector<GNode>& after,
                             CustomPassContext& context) __attribute__((weak));
#if GE_COMPILER_VERSION_NUM >= 90200000U
    static Status ReportMatch(const std::vector<GNode>& nodes, CustomPassContext& context) __attribute__((weak));
#endif
};
#endif
} // namespace fusion
} // namespace ge

namespace ops {
namespace fusion {
/**
 * @brief Replace a subgraph with automatic reporting when the runtime supports it.
 * @return Replacement status; fall back only when the reporting overload is unavailable.
 */
inline ge::Status ReplaceSubgraphWithReport(const ge::fusion::SubgraphBoundary& boundary, const ge::Graph& replacement,
                                            ge::CustomPassContext& context)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90100000U)
    using ReplaceWithContext = ge::Status (*)(const ge::fusion::SubgraphBoundary&, const ge::Graph&,
                                              ge::CustomPassContext&);
    const auto replace = static_cast<ReplaceWithContext>(&ge::fusion::SubgraphRewriter::Replace);
    if (replace != nullptr) {
        return replace(boundary, replacement, context);
    }
#else
    (void)context;
#endif
    return ge::fusion::SubgraphRewriter::Replace(boundary, replacement);
}

/**
 * @brief Report a matched node set when the runtime provides the reporting API.
 * @return Reporting status, or SUCCESS when reporting is unavailable.
 */
inline ge::Status ReportMatchNodes(const std::vector<ge::GNode>& nodes, ge::CustomPassContext& context)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90200000U)
    if (ge::fusion::GraphFuseInspectorUtils::ReportMatch != nullptr) {
        return ge::fusion::GraphFuseInspectorUtils::ReportMatch(nodes, context);
    }
#else
    (void)nodes;
    (void)context;
#endif
    return ge::SUCCESS;
}

/**
 * @brief Report fusion after rewiring and before removing the original nodes.
 * @return Reporting status, or SUCCESS when reporting is unavailable.
 */
inline ge::Status ReportFuseNodes(const std::vector<ge::GNode>& before, const std::vector<ge::GNode>& after,
                                  ge::CustomPassContext& context)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90100000U)
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse != nullptr) {
        return ge::fusion::GraphFuseInspectorUtils::ReportFuse(before, after, context);
    }
#else
    (void)before;
    (void)after;
    (void)context;
#endif
    return ge::SUCCESS;
}
} // namespace fusion
} // namespace ops

#endif // OPS_NORM_FUSION_REPORT_UTILS_H_
