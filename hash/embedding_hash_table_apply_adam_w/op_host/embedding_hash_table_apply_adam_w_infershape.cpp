/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_hash_table_apply_adam_w_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t inputKeysIndex = 1;
static constexpr size_t inputMIndex = 2;
static constexpr size_t inputVIndex = 3;
static constexpr size_t inputBeta1PowerIndex = 4;
static constexpr size_t inputBeta2PowerIndex = 5;
static constexpr size_t inputGradIndex = 11;
static constexpr size_t inputMaxGradNormIndex = 12;

static constexpr size_t outputMIndex = 0;
static constexpr size_t outputVIndex = 1;
static constexpr size_t outputBeta1PowerIndex = 2;
static constexpr size_t outputBeta2PowerIndex = 3;
static constexpr size_t outputMaxGradNormIndex = 4;

static constexpr size_t attrRequiredNum = 2;
static constexpr size_t attrEmbeddingDimIndex = 0;
static constexpr size_t attrBucketSizeIndex = 1;

graphStatus EmbeddingHashTableApplyAdamWInferOneRefOutputShape(gert::InferShapeContext *context, size_t inputIndex, size_t outputIndex)
{
    const gert::Shape *inputShape = context->GetInputShape(inputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    gert::Shape *outputShape = context->GetOutputShape(outputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

graphStatus CheckForEmbeddingHashTableApplyAdamW(const gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    size_t attrNum = attrs->GetAttrNum();
    OP_CHECK_IF(attrNum < attrRequiredNum,
        OP_LOGE(context->GetNodeName(), "infer shape check attr num: %lu less than min: %lu, failed.", attrNum,
            attrRequiredNum),
        return ge::GRAPH_FAILED);

    auto embeddingDimPtr = attrs->GetInt(attrEmbeddingDimIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDimPtr);
    auto bucketSizePtr = attrs->GetInt(attrBucketSizeIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketSizePtr);

    int64_t embeddingDim = *embeddingDimPtr;
    int64_t bucketSize = *bucketSizePtr;

    OP_CHECK_IF(embeddingDim < 0, OP_LOGE(context->GetNodeName(), "infer shape check attr eDim: %ld failed.", embeddingDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(bucketSize < 0, OP_LOGE(context->GetNodeName(), "infer shape check attr bucketSize: %ld failed.", bucketSize),
        return ge::GRAPH_FAILED);

    auto keyShape = context->GetInputShape(inputKeysIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    int64_t keyNum = keyShape->GetShapeSize();
    if (keyNum <= 0 || Ops::Base::IsUnknownRank(*keyShape) || Ops::Base::IsUnknownShape(*keyShape)) {
        OP_LOGD(context->GetNodeName(), "infer check success. keyNum: %ld eDim: %ld bucketSize: %ld", keyNum, embeddingDim,
            bucketSize);
        return ge::GRAPH_SUCCESS;
    }

    int64_t shouldShapeSize = bucketSize * embeddingDim;

    // 算子不强依赖shape完全一致，可以宽松校验，相关输入的 shape size 相等即可
    auto mShape = context->GetInputShape(inputMIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, mShape);
    int64_t mShapeSize = mShape->GetShapeSize();
    OP_CHECK_IF(mShapeSize != shouldShapeSize && mShapeSize > 0 && !Ops::Base::IsUnknownRank(*mShape) &&
            !Ops::Base::IsUnknownShape(*mShape),
        OP_LOGE(context->GetNodeName(), "infer shape check m shape size: %ld fail. kNum: %ld eDim: %ld bSize: %ld",
            mShapeSize, keyNum, embeddingDim, bucketSize),
        return ge::GRAPH_FAILED);

    auto vShape = context->GetInputShape(inputVIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);
    int64_t vShapeSize = vShape->GetShapeSize();
    OP_CHECK_IF(vShapeSize != shouldShapeSize && vShapeSize > 0 && !Ops::Base::IsUnknownRank(*vShape) &&
            !Ops::Base::IsUnknownShape(*vShape),
        OP_LOGE(context->GetNodeName(), "infer shape check v shape size: %ld fail. kNum: %ld eDim: %ld bSize: %ld",
            vShapeSize, keyNum, embeddingDim, bucketSize),
        return ge::GRAPH_FAILED);

    int64_t gradShouldShapeSize = keyNum * embeddingDim;
    auto gradShape = context->GetInputShape(inputGradIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    int64_t gradShapeSize = gradShape->GetShapeSize();
    OP_CHECK_IF(
        gradShapeSize != gradShouldShapeSize && gradShapeSize > 0 && !Ops::Base::IsUnknownRank(*gradShape) &&
            !Ops::Base::IsUnknownShape(*gradShape),
        OP_LOGE(context->GetNodeName(), "infer shape check grad shape size: %ld fail. kNum: %ld eDim: %ld bSize: %ld",
            gradShapeSize, keyNum, embeddingDim, bucketSize),
        return ge::GRAPH_FAILED);

    auto maxGradNormShape = context->GetInputShape(inputMaxGradNormIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxGradNormShape);
    int64_t maxGradNormShapeSize = maxGradNormShape->GetShapeSize();
    OP_CHECK_IF(maxGradNormShapeSize != shouldShapeSize && maxGradNormShapeSize > 0 &&
            !Ops::Base::IsUnknownRank(*maxGradNormShape) && !Ops::Base::IsUnknownShape(*maxGradNormShape),
        OP_LOGE(context->GetNodeName(),
            "infer shape check maxGradNorm shape size: %ld fail. kNum: %ld eDim: %ld bSize: %ld", maxGradNormShapeSize,
            keyNum, embeddingDim, bucketSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "infer check success. keyNum: %ld eDim: %ld bucketSize: %ld", keyNum, embeddingDim, bucketSize);
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShapeForEmbeddingHashTableApplyAdamW(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForEmbeddingHashTableApplyAdamW");

    graphStatus checkParamRes = CheckForEmbeddingHashTableApplyAdamW(context);
    OP_CHECK_IF(checkParamRes != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "infer shape check params failed."),
        return checkParamRes);

    graphStatus res = EmbeddingHashTableApplyAdamWInferOneRefOutputShape(context, inputMIndex, outputMIndex);
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "infer shape for input: %lu output: %lu failed.",
            inputMIndex, outputMIndex),
        return res);

    res = EmbeddingHashTableApplyAdamWInferOneRefOutputShape(context, inputVIndex, outputVIndex);
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "infer shape for input: %lu output: %lu failed.",
            inputVIndex, outputVIndex),
        return res);

    res = EmbeddingHashTableApplyAdamWInferOneRefOutputShape(context, inputBeta1PowerIndex, outputBeta1PowerIndex);
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "infer shape for input: %lu output: %lu failed.",
            inputBeta1PowerIndex, outputBeta1PowerIndex),
        return res);

    res = EmbeddingHashTableApplyAdamWInferOneRefOutputShape(context, inputBeta2PowerIndex, outputBeta2PowerIndex);
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "infer shape for input: %lu output: %lu failed.",
            inputBeta2PowerIndex, outputBeta2PowerIndex),
        return res);

    res = EmbeddingHashTableApplyAdamWInferOneRefOutputShape(context, inputMaxGradNormIndex, outputMaxGradNormIndex);
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "infer shape for input: %lu output: %lu failed.",
            inputMaxGradNormIndex, outputMaxGradNormIndex),
        return res);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForEmbeddingHashTableApplyAdamW");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForEmbeddingHashTableApplyAdamW(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForEmbeddingHashTableApplyAdamW");
    context->SetOutputDataType(outputMIndex, context->GetInputDataType(inputMIndex));
    context->SetOutputDataType(outputVIndex, context->GetInputDataType(inputMIndex));
    context->SetOutputDataType(outputBeta1PowerIndex, context->GetInputDataType(inputMIndex));
    context->SetOutputDataType(outputBeta2PowerIndex, context->GetInputDataType(inputMIndex));
    context->SetOutputDataType(outputMaxGradNormIndex, context->GetInputDataType(inputMIndex));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForEmbeddingHashTableApplyAdamW");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingHashTableApplyAdamW).InferShape(InferShapeForEmbeddingHashTableApplyAdamW)
                .InferDataType(InferDataTypeForEmbeddingHashTableApplyAdamW);
}  // namespace ops