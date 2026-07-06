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
 * \file index_put_v2_fusion_pass.cpp
 * \brief IndexPutV2 fusion pass: IndexPutV2 -> LinearIndexV2 + Sort + IndexPutWithSortV2
 */

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "common/inc/error_util.h"
#include "index_put_v2_fusion_pass.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/fusion/graph_rewriter.h"
#include "log/log.h"
#include "register/register_custom_pass.h"
#include "index_put_v2_utils.h"

namespace ge::fusion {

using namespace ge;
using namespace fe;

namespace {
const std::string kPassName = "IndexPutV2FusionPass";

constexpr size_t kIdxX = 0;
constexpr size_t kIdxValue = 1;
constexpr size_t kIdxIndexedSizes = 2;
constexpr size_t kIdxIndexedStrides = 3;
constexpr size_t kFixedInputNum = 4;
constexpr int64_t DYNAMIC_SHAPE_UNKNOWN = -1;
constexpr int64_t DYNAMIC_SHAPE_RANGE = -2;
constexpr int32_t kPortSelfRef = 0;
constexpr int32_t kPortValues = 1;
constexpr int32_t kPortMasks = 2;
constexpr int32_t kPortIndicesBase = 3;
constexpr int32_t kSortPortSelf = 0;
constexpr int32_t kSortPortSortIdx = 1;
constexpr int32_t kSortPortPosIdx = 2;
constexpr int32_t kSortPortValue = 3;

bool CheckDtype(DataType dtype, const std::vector<DataType>& validTypes)
{
    return std::find(validTypes.begin(), validTypes.end(), dtype) != validTypes.end();
}

std::vector<int32_t> CalculateStride(const std::vector<int64_t>& xDims)
{
    std::vector<int32_t> stride;
    for (size_t i = 0; i < xDims.size(); ++i) {
        int32_t s = 1;
        for (size_t j = i + 1; j < xDims.size(); ++j) {
            s *= static_cast<int32_t>(xDims[j]);
        }
        stride.push_back(s);
    }
    return stride;
}

std::vector<int32_t> GetXShapeSize(const std::vector<int64_t>& xDims)
{
    std::vector<int32_t> xShapeSize;
    for (auto dim : xDims) {
        xShapeSize.push_back(static_cast<int32_t>(dim));
    }
    return xShapeSize;
}

bool ShapeIsDynamic(const TensorDesc& Desc)
{
    auto shapes = Desc.GetShape().GetDims();
    for (auto it : shapes) {
        if (it == DYNAMIC_SHAPE_UNKNOWN || it == DYNAMIC_SHAPE_RANGE) {
            return true;
        }
    }

    return false;
}

void UpdateTensorDescsSortV2(const IndexPutV2InputInfo& info, const es::EsTensorHolder& linearIndex,
                             const es::EsTensorHolder& sortIndex, const es::EsTensorHolder& posIdx,
                             const es::EsTensorHolder& posIdxCast, const es::EsTensorHolder& indexPutWithSort)
{
    // 计算相关 shape
    std::vector<int32_t> strideVec = CalculateStride(info.xDims);
    TensorDesc strideDesc(Shape({static_cast<int64_t>(strideVec.size())}), FORMAT_ND, DT_INT32);
    std::vector<int32_t> xShapeSizeVec = GetXShapeSize(info.xDims);
    TensorDesc xShapeSizeDesc(Shape({static_cast<int64_t>(xShapeSizeVec.size())}), FORMAT_ND, DT_INT32);
    int64_t indexShapeSize = 1;
    bool unKnowFlag = false;
    for (int32_t i = 0; i < info.indicesDims[0].size(); i++) {
        indexShapeSize *= info.indicesDims[0][i];
        if (info.indicesDims[0][i] == -1) {
            unKnowFlag = true;
        }
    }
    indexShapeSize = unKnowFlag ? -1 : indexShapeSize;
    std::vector<int64_t> linearIndexDims{indexShapeSize};

    // LinearIndexV2 输入输出描述
    TensorDesc linearIndexOutputDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    linearIndex.GetProducer()->UpdateOutputDesc(0, linearIndexOutputDesc);
    for (int64_t i = 0; i < info.indicesNum; ++i) {
        TensorDesc indicesDesc(Shape(info.indicesDims[i]), info.indicesFmt, info.indicesDtypes[i]);
        linearIndex.GetProducer()->UpdateInputDesc(static_cast<uint32_t>(i), indicesDesc);
    }
    linearIndex.GetProducer()->UpdateInputDesc(static_cast<uint32_t>(info.indicesNum), strideDesc);
    linearIndex.GetProducer()->UpdateInputDesc(static_cast<uint32_t>(info.indicesNum + 1), xShapeSizeDesc);

    // Sort 输入输出描述
    TensorDesc sortInputDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    sortIndex.GetProducer()->UpdateInputDesc(0, sortInputDesc);
    TensorDesc sortOutputDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    sortIndex.GetProducer()->UpdateOutputDesc(0, sortOutputDesc);
    TensorDesc posIdxOutputDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    posIdx.GetProducer()->UpdateOutputDesc(1, posIdxOutputDesc);

    // Cast 输入输出描述
    TensorDesc castInputDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    posIdxCast.GetProducer()->UpdateInputDesc(0, castInputDesc);
    TensorDesc castOutputDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    posIdxCast.GetProducer()->UpdateOutputDesc(0, castOutputDesc);

    // IndexPutWithSortV2 输入输出描述
    TensorDesc indexPutInputXDesc(Shape(info.xDims), info.xFmt, info.xDtype);
    indexPutWithSort.GetProducer()->UpdateInputDesc(kSortPortSelf, indexPutInputXDesc);
    TensorDesc indexPutInputSortIdxDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    indexPutWithSort.GetProducer()->UpdateInputDesc(kSortPortSortIdx, indexPutInputSortIdxDesc);
    TensorDesc indexPutInputPosDesc(Shape(linearIndexDims), FORMAT_ND, DT_INT32);
    indexPutWithSort.GetProducer()->UpdateInputDesc(kSortPortPosIdx, indexPutInputPosDesc);
    TensorDesc indexPutInputValueDesc(Shape(info.valueDims), info.valueFmt, info.valueDtype);
    indexPutWithSort.GetProducer()->UpdateInputDesc(kSortPortValue, indexPutInputValueDesc);
    TensorDesc indexPutOutputDesc(Shape(info.xDims), info.xFmt, info.xDtype);
    indexPutWithSort.GetProducer()->UpdateOutputDesc(0, indexPutOutputDesc);
}
} // namespace

bool IndexPutV2FusionPass::CheckPlatform() const
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "Get platform_info failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc != "Ascend950" && soc != "MC62") {
        OP_LOGI(kPassName.c_str(), "Platform %s is not supported, only Ascend950, MC62", soc.c_str());
        return false;
    }
    return true;
}

bool IndexPutV2FusionPass::CheckDeterministic(ge::CustomPassContext& passContext) const
{
    ge::AscendString optionValue;
    ge::AscendString optionKey = ge::configure_option::DETERMINISTIC;

    ge::graphStatus status = passContext.GetOptionValue(optionKey, optionValue);
    if (status == ge::GRAPH_SUCCESS) {
        std::string value(optionValue.GetString());
        OP_LOGD(kPassName.c_str(), "Deterministic option found, value: %s", value.c_str());
        bool isEnabled = (value == "1");
        OP_LOGD(kPassName.c_str(), "Deterministic mode: %s", isEnabled ? "ENABLED" : "DISABLED");
        return isEnabled;
    }
    OP_LOGD(kPassName.c_str(), "Deterministic option not found (status=%d), default: DISABLED",
            static_cast<int>(status));

    return false;
}

bool IndexPutV2FusionPass::CheckDtypes(const GNode& node, int64_t indicesNum) const
{
    std::vector<DataType> dtypes = {DT_BOOL, DT_INT8, DT_UINT8, DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT, DT_INT64};
    std::vector<DataType> deterministicDtypes = {DT_FLOAT16, DT_BF16, DT_FLOAT};
    TensorDesc xDesc;
    if (node.GetInputDesc(kIdxX, xDesc) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "GetInputDesc for x failed");
        return false;
    }
    DataType xDtype = xDesc.GetDataType();
    if (!CheckDtype(xDtype, dtypes)) {
        OP_LOGI(kPassName.c_str(), "x dtype not support, actual: %d", static_cast<int>(xDtype));
        return false;
    }

    TensorDesc valueDesc;
    if (node.GetInputDesc(kIdxValue, valueDesc) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "GetInputDesc for value failed");
        return false;
    }
    DataType valueDtype = valueDesc.GetDataType();
    if (!CheckDtype(valueDtype, dtypes)) {
        OP_LOGI(kPassName.c_str(), "value dtype not support, actual: %d", static_cast<int>(valueDtype));
        return false;
    }

    if (xDtype != valueDtype) {
        OP_LOGI(kPassName.c_str(), "x dtype should same with value dtype, x: %d, value: %d", static_cast<int>(xDtype),
                static_cast<int>(valueDtype));
        return false;
    }

    TensorDesc yDesc;
    if (node.GetOutputDesc(0, yDesc) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "GetOutputDesc for y failed");
        return false;
    }
    DataType yDtype = yDesc.GetDataType();
    if (!CheckDtype(yDtype, dtypes)) {
        OP_LOGI(kPassName.c_str(), "y dtype not support, actual: %d", static_cast<int>(yDtype));
        return false;
    }

    if (xDtype != yDtype) {
        OP_LOGI(kPassName.c_str(), "x dtype should same with y dtype, x: %d, y: %d", static_cast<int>(xDtype),
                static_cast<int>(yDtype));
        return false;
    }

    for (int64_t i = 0; i < indicesNum; ++i) {
        TensorDesc indicesDesc;
        if (node.GetInputDesc(kFixedInputNum + i, indicesDesc) != SUCCESS) {
            OP_LOGI(kPassName.c_str(), "GetInputDesc for indices[%ld] failed", i);
            return false;
        }
        DataType indicesDtype = indicesDesc.GetDataType();
        if (!CheckDtype(indicesDtype, {DT_INT32, DT_INT64})) {
            OP_LOGI(kPassName.c_str(), "indices[%ld] dtype only support int32/int64, actual: %d", i,
                    static_cast<int>(indicesDtype));
            return false;
        }
    }

    bool accumulate = false;
    node.GetAttr("accumulate", accumulate);
    if (accumulate) {
        if (!CheckDtype(xDtype, deterministicDtypes)) {
            OP_LOGI(kPassName.c_str(), "accumulate is true, x dtype do not support Deterministic , actual: %d",
                    static_cast<int>(xDtype));
            return false;
        }
    }

    return true;
}

bool IndexPutV2FusionPass::CheckDynamic(const GNode& node, int64_t indicesNum) const
{
    TensorDesc xDesc;
    if (node.GetInputDesc(kIdxX, xDesc) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "GetInputDesc for x failed");
        return true;
    }
    if (ShapeIsDynamic(xDesc)) {
        OP_LOGI(kPassName.c_str(), "X is Dynamic");
        return true;
    }

    TensorDesc valueDesc;
    if (node.GetInputDesc(kIdxValue, valueDesc) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "GetInputDesc for value failed");
        return true;
    }
    if (ShapeIsDynamic(valueDesc)) {
        OP_LOGI(kPassName.c_str(), "Value is Dynamic");
        return true;
    }

    for (int64_t i = 0; i < indicesNum; ++i) {
        TensorDesc indicesDesc;
        if (node.GetInputDesc(kFixedInputNum + i, indicesDesc) != SUCCESS) {
            OP_LOGI(kPassName.c_str(), "GetInputDesc for indices[%ld] failed", i);
            return true;
        }
        if (ShapeIsDynamic(indicesDesc)) {
            OP_LOGI(kPassName.c_str(), "Indices is Dynamic");
            return true;
        }
    }

    return false;
}

bool IndexPutV2FusionPass::CheckNode(const GNode& node)
{
    AscendString nodeType;
    if (node.GetType(nodeType) != SUCCESS) {
        return false;
    }
    if (std::string(nodeType.GetString()) != "IndexPutV2") {
        return false;
    }

    size_t inputSize = node.GetInputsSize();
    int64_t indicesNum = static_cast<int64_t>(inputSize) - kFixedInputNum;

    if (!CheckDtypes(node, indicesNum)) {
        return false;
    }

    return true;
}

IndexPutV2InputInfo IndexPutV2FusionPass::GetInputInfo(const GNode& node) const
{
    IndexPutV2InputInfo info;

    TensorDesc xDesc;
    node.GetInputDesc(kIdxX, xDesc);
    info.xDims = xDesc.GetShape().GetDims();
    info.xDtype = xDesc.GetDataType();
    info.xFmt = xDesc.GetFormat();

    TensorDesc valueDesc;
    node.GetInputDesc(kIdxValue, valueDesc);
    info.valueDims = valueDesc.GetShape().GetDims();
    info.valueDtype = valueDesc.GetDataType();
    info.valueFmt = valueDesc.GetFormat();

    TensorDesc indexedSizeDesc;
    node.GetInputDesc(kIdxIndexedSizes, indexedSizeDesc);
    info.indexedSizeDims = indexedSizeDesc.GetShape().GetDims();
    info.indexedSizeDtype = indexedSizeDesc.GetDataType();
    info.indexedSizeFmt = indexedSizeDesc.GetFormat();

    TensorDesc indexedStridesDesc;
    node.GetInputDesc(kIdxIndexedStrides, indexedStridesDesc);
    info.indexedStridesDims = indexedStridesDesc.GetShape().GetDims();
    info.indexedStridesDtype = indexedStridesDesc.GetDataType();
    info.indexedStridesFmt = indexedStridesDesc.GetFormat();

    info.indicesNum = static_cast<int64_t>(node.GetInputsSize()) - kFixedInputNum;
    for (int64_t i = 0; i < info.indicesNum; ++i) {
        TensorDesc indicesDesc;
        node.GetInputDesc(kFixedInputNum + i, indicesDesc);
        info.indicesDims.push_back(indicesDesc.GetShape().GetDims());
        info.indicesDtypes.push_back(indicesDesc.GetDataType());
        info.indicesFmt = indicesDesc.GetFormat();
        info.indicesShape.push_back(indicesDesc.GetShape());
    }

    info.attrAccumulate = false;
    node.GetAttr("accumulate", info.attrAccumulate);

    return info;
}

void UpdateTensorDescsIndexPutV3(const IndexPutV2InputInfo& info, const es::EsTensorHolder& indexPut)
{
    TensorDesc indexPutInputXDesc(Shape(info.xDims), info.xFmt, info.xDtype);
    indexPut.GetProducer()->UpdateInputDesc(kPortSelfRef, indexPutInputXDesc);
    TensorDesc indexPutInputValueDesc(Shape(info.valueDims), info.valueFmt, info.valueDtype);
    indexPut.GetProducer()->UpdateInputDesc(kPortValues, indexPutInputValueDesc);
    TensorDesc indexPutInputMasksDesc(Shape(info.indexedSizeDims), info.indexedSizeFmt, info.indexedSizeDtype);
    indexPut.GetProducer()->UpdateInputDesc(kPortMasks, indexPutInputMasksDesc);
    for (int64_t i = 0; i < info.indicesNum; ++i) {
        TensorDesc indicesDesc(Shape(info.indicesDims[i]), info.indicesFmt, info.indicesDtypes[i]);
        indexPut.GetProducer()->UpdateInputDesc(static_cast<uint32_t>(kPortIndicesBase + i), indicesDesc);
    }
    TensorDesc indexPutOutputDesc(Shape(info.xDims), info.xFmt, info.xDtype);
    indexPut.GetProducer()->UpdateOutputDesc(0, indexPutOutputDesc);
}

GraphUniqPtr IndexPutV2FusionPass::CreateReplacement(const GNode& node)
{
    IndexPutV2InputInfo info = GetInputInfo(node);
    bool isDynamic = CheckDynamic(node, info.indicesNum);
    auto builder = es::EsGraphBuilder("replacement");

    size_t newIdx = 0;
    std::vector<int64_t> indexedSizes(info.indicesNum, 0);
    auto rX = builder.CreateInput(newIdx++, "x", info.xDtype, info.xFmt, info.xDims);
    auto rValue = builder.CreateInput(newIdx++, "value", info.valueDtype, info.valueFmt, info.valueDims);
    auto rIndexedSizes = builder.CreateInput(newIdx++, "indexed_sizes", info.indexedSizeDtype, info.indexedSizeFmt,
                                             info.indexedSizeDims);
    auto rIndexedStrides = builder.CreateInput(newIdx++, "indexed_strides", info.indexedStridesDtype,
                                               info.indexedStridesFmt, info.indexedStridesDims);

    std::vector<es::EsTensorHolder> rIndices;
    for (int64_t i = 0; i < info.indicesNum; ++i) {
        std::string name = "indices" + std::to_string(i);
        rIndices.emplace_back(
            builder.CreateInput(newIdx++, name.c_str(), info.indicesDtypes[i], info.indicesFmt, info.indicesDims[i]));
        if (info.indicesShape[i].GetShapeSize() != 0) { //索引不为空，则置1
            indexedSizes[i] = 1;
        }
    }

    std::vector<es::EsTensorHolder> outputs;
    if (!isDynamic) {
        std::vector<int32_t> strideVec = CalculateStride(info.xDims);
        auto xStride = builder.CreateConst(strideVec, {static_cast<int64_t>(strideVec.size())}, DT_INT32, FORMAT_ND);
        std::vector<int32_t> xShapeSizeVec = GetXShapeSize(info.xDims);
        auto xShapeSize = builder.CreateConst(xShapeSizeVec, {static_cast<int64_t>(xShapeSizeVec.size())}, DT_INT32,
                                              FORMAT_ND);

        auto linearIndex = es::LinearIndexV2(rIndices, xStride, xShapeSize);
        auto [sortIndex, posIdx] = es::Sort(linearIndex, -1, false, true, DT_INT32);
        auto posIdxCast = es::Cast(posIdx, DT_INT32);
        auto indexPutWithSort = es::IndexPutWithSortV2(rX, sortIndex, posIdxCast, rValue, indexedSizes,
                                                       info.attrAccumulate);

        UpdateTensorDescsSortV2(info, linearIndex, sortIndex, posIdx, posIdxCast, indexPutWithSort);
        outputs.emplace_back(indexPutWithSort);
    } else {
        auto indexPut = es::IndexPutV3(rX, rValue, rIndexedSizes, rIndices, info.attrAccumulate, true);
        UpdateTensorDescsIndexPutV3(info, indexPut);
        outputs.emplace_back(indexPut);
    }

    return builder.BuildAndReset(outputs);
}

std::unique_ptr<SubgraphBoundary> IndexPutV2FusionPass::ConstructBoundary(const GNode& node) const
{
    auto boundary = std::make_unique<SubgraphBoundary>();

    for (size_t i = 0; i < node.GetInputsSize(); i++) {
        SubgraphInput subgraphInput;
        subgraphInput.AddInput({node, static_cast<int64_t>(i)});
        if (boundary->AddInput(i, std::move(subgraphInput)) != SUCCESS) {
            OP_LOGI(kPassName.c_str(), "AddInput failed for idx %zu", i);
            return nullptr;
        }
    }

    SubgraphOutput output({node, 0});
    if (boundary->AddOutput(0, std::move(output)) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "AddOutput failed");
        return nullptr;
    }

    return boundary;
}

Status IndexPutV2FusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OP_LOGI(kPassName.c_str(), "Enter IndexPutV2FusionPass");

    if (!CheckPlatform()) {
        return GRAPH_NOT_CHANGED;
    }

    if (!CheckDeterministic(passContext)) {
        OP_LOGI(kPassName.c_str(), "Deterministic check failed, skip fusion");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> indexPutNodes;
    for (auto& node : graph->GetDirectNode()) {
        if (CheckNode(node)) {
            indexPutNodes.emplace_back(node);
        }
    }

    if (indexPutNodes.empty()) {
        OP_LOGI(kPassName.c_str(), "No IndexPutV2 nodes to fuse");
        return GRAPH_NOT_CHANGED;
    }

    Graph originGraph = *graph;

    for (auto& node : indexPutNodes) {
        auto replacement = CreateReplacement(node);
        if (!replacement) {
            AscendString nodeName;
            node.GetName(nodeName);
            OP_LOGI(kPassName.c_str(), "CreateReplacement failed for %s", nodeName.GetString());
            *graph = originGraph;
            return GRAPH_NOT_CHANGED;
        }

        auto boundary = ConstructBoundary(node);
        if (!boundary) {
            AscendString nodeName;
            node.GetName(nodeName);
            OP_LOGI(kPassName.c_str(), "ConstructBoundary failed for %s", nodeName.GetString());
            *graph = originGraph;
            return GRAPH_NOT_CHANGED;
        }

        Status replaceStatus = SubgraphRewriter::Replace(*boundary, *replacement);
        if (replaceStatus != SUCCESS) {
            AscendString nodeName;
            node.GetName(nodeName);
            OP_LOGI(kPassName.c_str(), "Replace failed for %s, status=%d", nodeName.GetString(),
                    static_cast<int>(replaceStatus));
            *graph = originGraph;
            return GRAPH_NOT_CHANGED;
        }
    }

    OP_LOGI(kPassName.c_str(), "Fusion completed, %zu nodes fused", indexPutNodes.size());
    return SUCCESS;
}

REG_FUSION_PASS(IndexPutV2FusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ge::fusion