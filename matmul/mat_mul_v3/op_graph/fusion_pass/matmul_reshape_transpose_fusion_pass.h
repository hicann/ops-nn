/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_MATMUL_RESHAPE_TRANSPOSE_FUSION_PASS_H
#define NN_MATMUL_RESHAPE_TRANSPOSE_FUSION_PASS_H

#include "ge/fusion/pass/fusion_base_pass.h"

namespace ge {
namespace fusion {
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
};
} // namespace fusion
} // namespace ge

namespace ops {

class __attribute__((visibility("default"))) MatmulReshapeTransposeFusionPass : public ge::fusion::FusionBasePass {
protected:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;
};

} // namespace ops

#endif // NN_MATMUL_RESHAPE_TRANSPOSE_FUSION_PASS_H
