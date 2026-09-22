/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_CONV_GRAPH_FUSE_INSPECTOR_UTILS_H
#define NN_CONV_GRAPH_FUSE_INSPECTOR_UTILS_H

#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90100000U
namespace ge {
namespace fusion {
// Weak mirror of ge/fusion/graph_fuse_inspector_utils.h, the single maintenance point of the conv domain:
// the fusion pass plugin still loads when the runtime libge does not export these symbols.
// Keep signatures in sync with the GE header and check the address against nullptr before calling.
class GraphFuseInspectorUtils {
public:
    static bool CanFuse(const std::vector<GNode>& nodesBeforeFuse, AscendString& failedReason) __attribute__((weak));
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
};
} // namespace fusion
} // namespace ge
#endif

#endif // NN_CONV_GRAPH_FUSE_INSPECTOR_UTILS_H
