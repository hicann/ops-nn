/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_WEIGHT_QUANT_FUSION_UTILS_H
#define NN_WEIGHT_QUANT_FUSION_UTILS_H

#include <algorithm>
#include <cstdint>
#include <vector>
#include "acl/acl_rt.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/subgraph_boundary.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"

namespace ge {
namespace fusion {
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
};

// 弱符号兼容声明：2 参数 Replace 自 GE 8.5.0 起存在；带 CustomPassContext 的 3 参数版本
// 仅在 GE 9.1.0+ 运行时中存在强符号。3 参数版本声明为弱符号，运行时无强定义时其函数
// 指针为空，调用方需回退到 2 参数版本。注意：包含本头文件的编译单元不可再包含
// ge/fusion/graph_rewriter.h（会导致类重定义）。
class SubgraphRewriter {
public:
    static Status Replace(const SubgraphBoundary& subgraph, const Graph& replacement);
#if GE_COMPILER_VERSION_NUM >= 90100000U
    static Status Replace(const SubgraphBoundary& subgraph, const Graph& replacement, CustomPassContext& ctx)
        __attribute__((weak));
#endif
};
} // namespace fusion
} // namespace ge

namespace ops {
namespace weight_quant {

inline bool IsGeVersionSupported()
{
    int32_t version = 0;
    if (aclsysGetVersionNum("ge-compiler", &version) != ACL_SUCCESS) {
        OP_LOGW("WeightQuantFusionUtils", "Failed to get ge-compiler version");
        return false;
    }
    constexpr int32_t kMinGeCompilerVersion = 90100000;
    return version >= kMinGeCompilerVersion;
}

inline ge::GNodePtr GetInputNode(const ge::GNode& n, int64_t i) { return n.GetInDataNodesAndPortIndexs(i).first; }

inline bool IsTrans(const ge::AscendString& nodeType) { return nodeType == "Transpose" || nodeType == "TransposeD"; }

inline bool HasOtherConsumers(const ge::GNodePtr& node) { return !node->GetOutDataNodesAndPortIndexs(0).empty(); }

inline void AddNodeToRemove(std::vector<ge::GNode>& nodesBeforeFuse, std::vector<ge::GNodePtr>& nodesToRemove,
                            const ge::GNodePtr& node)
{
    if (node == nullptr || std::find(nodesToRemove.begin(), nodesToRemove.end(), node) != nodesToRemove.end()) {
        return;
    }
    nodesToRemove.emplace_back(node);
    nodesBeforeFuse.emplace_back(*node);
}

inline void ReportTransposeFusion(const char* logTag, const std::vector<ge::GNode>& nodesBeforeFuse,
                                  const ge::GNode& fusedNode, ge::CustomPassContext& passContext)
{
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse == nullptr) {
        return;
    }
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, {fusedNode}, passContext) != ge::SUCCESS) {
        OP_LOGW(logTag, "Failed to report fusion result.");
    }
}

inline bool FlipTransposeAttr(const char* logTag, ge::GNode& n, const char* attrName)
{
    bool val = false;
    if (n.GetAttr(attrName, val) != ge::GRAPH_SUCCESS) {
        val = false;
    }
    val = !val;
    if (n.SetAttr(attrName, val) != ge::GRAPH_SUCCESS) {
        OP_LOGW(logTag, "Failed to set %s attribute", attrName);
        return false;
    }
    OP_LOGD(logTag, "Successfully set %s to %d", attrName, val);
    return true;
}

} // namespace weight_quant
} // namespace ops

#endif // NN_WEIGHT_QUANT_FUSION_UTILS_H
