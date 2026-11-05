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
 * \file map_index_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;
namespace ops {

constexpr size_t X_INDEX = 0;
constexpr size_t DATA_SEQ_INDEX = 1;
constexpr size_t LEVEL_INDEX_INDEX = 2;
constexpr size_t Y_INDEX = 0;

static ge::graphStatus InferShape4MapIndex(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MapIndexInferShape");
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* dataSeqShape = context->GetInputShape(DATA_SEQ_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataSeqShape);
    const gert::Shape* levelIndexShape = context->GetOptionalInputShape(LEVEL_INDEX_INDEX);

    OP_CHECK_IF(xShape->GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(), "x shape is empty, or shape size is not equal to 1, but got %zu",
                        xShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataSeqShape->GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(), "data_seq shape is empty, or shape size is not equal to 1, but got %zu",
                        dataSeqShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    const int64_t xLength = xShape->GetDim(0);
    const int64_t dataSeqLength = dataSeqShape->GetDim(0);
    // ===== 不支持动态场景：x 与 data_seq 为未知维(-1)或未知秩(-2)时直接拒绝 =====
    OP_CHECK_IF(Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownRank(*xShape),
                OP_LOGE(context->GetNodeName(), "x does not support dynamic shape, dim is %ld", xLength),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Ops::Base::IsUnknownShape(*dataSeqShape) || Ops::Base::IsUnknownRank(*dataSeqShape),
                OP_LOGE(context->GetNodeName(), "data_seq does not support dynamic shape, dim is %ld", dataSeqLength),
                return ge::GRAPH_FAILED);
    // 与 map_index_tiling_arch35.cpp 的 X_MAX_DIM0/DATA_SEQ_MAX_DIM0 保持一致
    // 本算子仅交付 ascend950/ascend350/ascend960dt（均绑 arch35 tiling），infershape 与 tiling 统一使用同一组上限
    constexpr int64_t maxLen = 24000;
    constexpr int64_t maxGear = 256;

    OP_CHECK_IF(xLength == 0,
                OP_LOGE(context->GetNodeName(), "the length of x should not be 0, but it is currently %ld", xLength),
                return ge::GRAPH_FAILED);
    // 整除校验前置: 保证 level_index 分支与档位校验中的除法均为精确商, 避免不整除时先误报长度不符
    OP_CHECK_IF(dataSeqLength % xLength != 0,
                OP_LOGE(context->GetNodeName(),
                        "the length of data_seq must be multiple of the length of x, but data_seq length is %ld, "
                        "x length is %ld",
                        dataSeqLength, xLength),
                return ge::GRAPH_FAILED);
    if (levelIndexShape != nullptr && levelIndexShape->GetDimNum() != 0 &&
        !Ops::Base::IsUnknownShape(*levelIndexShape) && !Ops::Base::IsUnknownRank(*levelIndexShape)) {
        const int64_t levelIndexLength = levelIndexShape->GetDim(0);
        OP_CHECK_IF(levelIndexLength != (dataSeqLength / xLength),
                    OP_LOGE(context->GetNodeName(),
                            "the length of level_index must be equal to the length of data_seq divided by "
                            "the length of x, but got %ld, expected %ld",
                            levelIndexLength, dataSeqLength / xLength),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        xLength > maxLen,
        OP_LOGE(context->GetNodeName(), "the length of x %ld should be less than or equal to %ld", xLength, maxLen),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dataSeqLength / xLength > maxGear,
        OP_LOGE(context->GetNodeName(), "data_seq_length / x_length should be less than or equal to %ld, but got %ld",
                maxGear, dataSeqLength / xLength),
        return ge::GRAPH_FAILED);
    gert::Shape* outputShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShape->SetScalar();
    OP_LOGD(context->GetNodeName(), "End to do MapIndexInferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MapIndex).InferShape(InferShape4MapIndex);
} // namespace ops
