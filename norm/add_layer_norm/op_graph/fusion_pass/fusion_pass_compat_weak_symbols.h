/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_PASS_COMPAT_WEAK_SYMBOLS_H_
#define FUSION_PASS_COMPAT_WEAK_SYMBOLS_H_

#include "graph/graph.h"
#include "ge/fusion/subgraph_boundary.h"
#include "ge/fusion/match_result.h"
#include "ge_common/ge_api_types.h"
#include "register/register_custom_pass.h"

namespace ge {
namespace fusion {

// InferShapeUtil (9.1.0) - 用于 Replacement 阶段对 replacement graph 做 shape/dtype 推导
// 注意：使用此头文件时，不要同时 include infer_shape_util.h
class InferShapeUtil {
 public:
  __attribute__((weak))
  static Status InferShape(const Graph &replacement_graph, const SubgraphBoundary &subgraph_boundary);
  __attribute__((weak))
  static Status InferShape(const Graph &replacement_graph, const MatchResult &match_result);
  __attribute__((weak))
  static Status InferShape(const Graph &replacement_graph, const GNode &matched_node);
};

}  // namespace fusion
}  // namespace ge

#endif  // FUSION_PASS_COMPAT_WEAK_SYMBOLS_H_
