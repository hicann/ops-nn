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
 * \file index_put_v2_fusion_pass.h
 * \brief IndexPutV2 fusion pass header
 */

#ifndef NN_INDEX_PUT_V2_FUSION_PASS_H
#define NN_INDEX_PUT_V2_FUSION_PASS_H

#include "ge/fusion/pass/fusion_base_pass.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ge::fusion {

using namespace ge;

struct IndexPutV2InputInfo {
    std::vector<int64_t> xDims;
    std::vector<int64_t> valueDims;
    std::vector<int64_t> indexedSizeDims;
    std::vector<int64_t> indexedStridesDims;
    std::vector<std::vector<int64_t>> indicesDims;
    
    DataType xDtype;
    DataType valueDtype;
    DataType indexedSizeDtype;
    DataType indexedStridesDtype;
    std::vector<DataType> indicesDtypes;
    
    Format xFmt;
    Format valueFmt;
    Format indexedSizeFmt;
    Format indexedStridesFmt;
    Format indicesFmt;

    std::vector<Shape> indicesShape;
    
    bool attrAccumulate = false;
    int64_t indicesNum = 0;
};

class __attribute__((visibility("default"))) IndexPutV2FusionPass : public FusionBasePass {
public:
    Status Run(GraphPtr &graph, CustomPassContext &passContext) override;

private:
    bool CheckPlatform() const;
    bool CheckDeterministic(ge::CustomPassContext& passContext) const;
    bool CheckDtypes(const GNode &node, int64_t indicesNum) const;
    bool CheckNode(const GNode &node);
    bool CheckDynamic(const GNode &node, int64_t indicesNum) const;
    IndexPutV2InputInfo GetInputInfo(const GNode &node) const;
    GraphUniqPtr CreateReplacement(const GNode &node);
    std::unique_ptr<SubgraphBoundary> ConstructBoundary(const GNode &node) const;
};

} // namespace ge::fusion
#endif // NN_INDEX_PUT_V2_FUSION_PASS_H