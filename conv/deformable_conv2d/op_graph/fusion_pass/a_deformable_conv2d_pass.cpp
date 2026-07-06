/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "a_deformable_conv2d_pass.h"

#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace DfmConv2dFusion;
using namespace ge;
using namespace fusion;
using namespace es;

namespace {
bool IsUnknownDim(int64_t dim)
{
    return dim < KNOWN_DIM_LOWER;
}

bool FindHwPos(const AscendString &fmtStr, size_t &posH, size_t &posW)
{
    std::string fmt = fmtStr.GetString();
    posH = fmt.find('H');
    posW = fmt.find('W');
    if (posH == std::string::npos || posW == std::string::npos || fmt.size() != DIM_SIZE) {
        return false;
    }
    return true;
}
} // namespace

void ADeformableConv2dPass::InitMember()
{
    convDescInfo = ConvDescInfo();
    baseAttrs = ConvBaseAttrs();
    offsetsDesc = TensorDesc();
    dfmOffsetOutDesc = TensorDesc();
    dfmPos = DfmPos();
    ksize = {};
    dfmGroups = DEFAULT_DFM_GROUPS;
    kn = DEFAULT_KN;
    modulated = true;
    isDav3510 = false;
    npuArch = NpuArch::DAV_RESV;
}

bool ADeformableConv2dPass::GetDfmDescInfo(const GNode &convNode)
{
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FMAP_INDEX, convDescInfo.fmapDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get %s x tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FILTER_INDEX, convDescInfo.filterDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get %s filter tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_OFFSETS_INDEX, offsetsDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get %s offsets tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_DFM_BIAS_INDEX, convDescInfo.biasDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSION_NAME, "Get %s bias tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    }
    FUSION_PASS_CHECK(convNode.GetOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get %s output tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    return true;
}

bool ADeformableConv2dPass::ParseFilter()
{
    AscendString filterFmtStr = TypeUtils::FormatToAscendString(convDescInfo.filterDesc.GetOriginFormat());
    FUSION_PASS_CHECK(FILTER_FORMATS.find(filterFmtStr) == FILTER_FORMATS.end(),
        OP_LOGD(FUSION_NAME, "%s filter format %s unsupported.", convDescInfo.nodeNameStr.c_str(),
            filterFmtStr.GetString()),
        return false);
    std::string fmt = filterFmtStr.GetString();
    size_t posH = fmt.find('H');
    size_t posW = fmt.find('W');
    size_t posN = fmt.find('N');
    std::vector<int64_t> filterShape = convDescInfo.filterDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(filterShape.size() != static_cast<size_t>(DIM_SIZE),
        OP_LOGD(FUSION_NAME, "%s filter origin shape dim %zu not equal to %ld.", convDescInfo.nodeNameStr.c_str(),
            filterShape.size(), DIM_SIZE),
        return false);
    ksize = {filterShape[posH], filterShape[posW]};
    kn = filterShape[posN];
    return true;
}

bool ADeformableConv2dPass::GetDfmAttrs(const GNode &convNode)
{
    FUSION_PASS_CHECK(convNode.GetAttr(STRIDES, baseAttrs.strides) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get strides from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetAttr(PADS, baseAttrs.pads) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get pads from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetAttr(DILATIONS, baseAttrs.dilations) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get dilations from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetAttr(GROUPS, baseAttrs.groups) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get groups from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetAttr(DATA_FORMAT, baseAttrs.dataFormat) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get data_format from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetAttr(DFM_GROUPS, dfmGroups) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get deformable_groups from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetAttr(MODULATED, modulated) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get modulated from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);
    return true;
}

bool ADeformableConv2dPass::ResolveDfmPos()
{
    AscendString xFmtStr = TypeUtils::FormatToAscendString(convDescInfo.fmapDesc.GetOriginFormat());
    AscendString outFmtStr = TypeUtils::FormatToAscendString(convDescInfo.outputDesc.GetOriginFormat());
    std::string xFmt = xFmtStr.GetString();
    dfmPos.xPosC = xFmt.find('C');
    FUSION_PASS_CHECK(dfmPos.xPosC == std::string::npos,
        OP_LOGD(FUSION_NAME, "%s x format %s has no C dim.", convDescInfo.nodeNameStr.c_str(), xFmtStr.GetString()),
        return false);
    FUSION_PASS_CHECK(!FindHwPos(xFmtStr, dfmPos.xPosH, dfmPos.xPosW),
        OP_LOGD(FUSION_NAME, "%s x format %s H/W pos not found.", convDescInfo.nodeNameStr.c_str(),
            xFmtStr.GetString()),
        return false);
    FUSION_PASS_CHECK(!FindHwPos(outFmtStr, dfmPos.outPosH, dfmPos.outPosW),
        OP_LOGD(FUSION_NAME, "%s output format %s H/W pos not found.", convDescInfo.nodeNameStr.c_str(),
            outFmtStr.GetString()),
        return false);
    return true;
}

bool ADeformableConv2dPass::CalcDfmOutShape()
{
    dfmOffsetOutDesc = convDescInfo.outputDesc;
    FUSION_PASS_CHECK_NOLOG(!ResolveDfmPos(), return false);

    std::vector<int64_t> shape = convDescInfo.fmapDesc.GetOriginShape().GetDims();
    const auto outShape = convDescInfo.outputDesc.GetOriginShape().GetDims();
    shape[dfmPos.xPosH] = outShape[dfmPos.outPosH] * ksize[KS_H_INDEX];
    shape[dfmPos.xPosW] = outShape[dfmPos.outPosW] * ksize[KS_W_INDEX];

    const bool outHUnknown = IsUnknownDim(outShape[dfmPos.outPosH]);
    const bool outWUnknown = IsUnknownDim(outShape[dfmPos.outPosW]);
    if (outHUnknown) {
        shape[dfmPos.xPosH] = UNKNOWN_DIM;
    }
    if (outWUnknown) {
        shape[dfmPos.xPosW] = UNKNOWN_DIM;
    }

    if (outHUnknown || outWUnknown) {
        std::vector<std::pair<int64_t, int64_t>> outRange;
        convDescInfo.outputDesc.GetShapeRange(outRange);
        if (!outRange.empty() && outRange[dfmPos.outPosH].second > INVALID_SHAPE_RANGE &&
            outRange[dfmPos.outPosW].second > INVALID_SHAPE_RANGE) {
            std::vector<std::pair<int64_t, int64_t>> range(outRange);
            range[dfmPos.xPosC] = {kn, kn};
            range[dfmPos.xPosH] = {shape[dfmPos.xPosH], shape[dfmPos.xPosH]};
            range[dfmPos.xPosW] = {shape[dfmPos.xPosW], shape[dfmPos.xPosW]};
            if (outHUnknown) {
                range[dfmPos.xPosH].first =
                    std::max(outRange[dfmPos.outPosH].first * ksize[KS_H_INDEX], DYNAMIC_RANGE_LOWER);
                range[dfmPos.xPosH].second =
                    std::min(outRange[dfmPos.outPosH].second * ksize[KS_H_INDEX], DYNAMIC_RANGE_UPPER);
            }
            if (outWUnknown) {
                range[dfmPos.xPosW].first =
                    std::max(outRange[dfmPos.outPosW].first * ksize[KS_W_INDEX], DYNAMIC_RANGE_LOWER);
                range[dfmPos.xPosW].second =
                    std::min(outRange[dfmPos.outPosW].second * ksize[KS_W_INDEX], DYNAMIC_RANGE_UPPER);
            }
            dfmOffsetOutDesc.SetShapeRange(range);
        }
    }

    dfmOffsetOutDesc.SetShape(Shape(shape));
    dfmOffsetOutDesc.SetOriginShape(Shape(shape));
    return true;
}

bool ADeformableConv2dPass::MeetRequirements(const GNode &convNode)
{
    InitMember();
    FUSION_PASS_CHECK(convNode.GetName(convDescInfo.nodeName) != GRAPH_SUCCESS,
        OP_LOGW(FUSION_NAME, "Get node name failed."), return false);
    convDescInfo.nodeNameStr = convDescInfo.nodeName.GetString();
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do ADeformableConv2dPass.");
    isDav3510 = ConvFusionUtilsPass::CheckSocList(ND_SOC_LIST, npuArch);

    size_t inSize = convNode.GetInputsSize();
    FUSION_PASS_CHECK(inSize != INPUT_COUNT_WITHOUT_BIAS && inSize != INPUT_COUNT_WITH_BIAS,
        OP_LOGD(FUSION_NAME, "%s in-size %zu unsupported.", convDescInfo.nodeNameStr.c_str(), inSize), return false);
    convDescInfo.hasBias = (inSize == INPUT_COUNT_WITH_BIAS);

    FUSION_PASS_CHECK_NOLOG(!GetDfmDescInfo(convNode), return false);

    FUSION_PASS_CHECK_NOLOG(!ParseFilter(), return false);
    FUSION_PASS_CHECK(IsUnknownDim(ksize[KS_H_INDEX]) || IsUnknownDim(ksize[KS_W_INDEX]),
        OP_LOGD(FUSION_NAME, "%s filter ksize unknown.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convDescInfo.fmapDesc.GetOriginShape().GetDims().size() != static_cast<size_t>(DIM_SIZE),
        OP_LOGD(FUSION_NAME, "%s fmap origin shape dim not equal to %ld.", convDescInfo.nodeNameStr.c_str(), DIM_SIZE),
        return false);
    FUSION_PASS_CHECK(convDescInfo.outputDesc.GetOriginShape().GetDims().size() != static_cast<size_t>(DIM_SIZE),
        OP_LOGD(FUSION_NAME, "%s output origin shape dim not equal to %ld.", convDescInfo.nodeNameStr.c_str(),
            DIM_SIZE),
        return false);
    FUSION_PASS_CHECK_NOLOG(!GetDfmAttrs(convNode), return false);
    FUSION_PASS_CHECK(baseAttrs.strides.size() != static_cast<size_t>(DIM_SIZE),
        OP_LOGD(FUSION_NAME, "%s strides dim invalid.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK_NOLOG(!CalcDfmOutShape(), return false);
    return true;
}

bool ADeformableConv2dPass::SetConv2dAttrs(GNode &conv2dNode)
{
    std::vector<int64_t> newStrides = baseAttrs.strides;
    AscendString xFmtStr = TypeUtils::FormatToAscendString(convDescInfo.fmapDesc.GetOriginFormat());
    size_t posH = 0;
    size_t posW = 0;
    if (FindHwPos(xFmtStr, posH, posW)) {
        newStrides[posH] = ksize[KS_H_INDEX];
        newStrides[posW] = ksize[KS_W_INDEX];
    }
    FUSION_PASS_CHECK(conv2dNode.SetAttr(STRIDES, newStrides) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set strides failed.", convDescInfo.nodeNameStr.c_str()), return false);
    std::vector<int64_t> zeroPads(DIM_SIZE, PAD_VALUE);
    std::vector<int64_t> oneDilations(DIM_SIZE, DILATION_VALUE);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(PADS, zeroPads) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set pads failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(DILATIONS, oneDilations) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set dilations failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(GROUPS, baseAttrs.groups) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set groups failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(DATA_FORMAT, baseAttrs.dataFormat) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set data_format failed.", convDescInfo.nodeNameStr.c_str()), return false);
    if (isDav3510) {
        int64_t offsetX = DEFAULT_OFFSET_X;
        FUSION_PASS_CHECK(conv2dNode.SetAttr(OFFSET_X, offsetX) != GRAPH_SUCCESS,
            OP_LOGE(FUSION_NAME, "%s set offset_x failed.", convDescInfo.nodeNameStr.c_str()), return false);
    }
    // DeformableConv2D does not support HI-Float-32; force Float32 execution on decomposed Conv2D.
    int64_t enableFloat32Execution = DfmConv2dFusion::ENABLE_FLOAT32_EXECUTION;
    FUSION_PASS_CHECK(conv2dNode.SetAttr(OP_IMPL_MODE_ENUM, enableFloat32Execution) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set _op_impl_mode_enum failed.", convDescInfo.nodeNameStr.c_str()), return false);
    return true;
}

bool ADeformableConv2dPass::UpdateDfmOffsetDesc(GNode &dfmOffsetNode)
{
    FUSION_PASS_CHECK(
        dfmOffsetNode.UpdateInputDesc(DFM_OFFSET_INPUT_X_INDEX, convDescInfo.fmapDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update DeformableOffsets x input desc failed.", convDescInfo.nodeNameStr.c_str()),
        return false);
    FUSION_PASS_CHECK(
        dfmOffsetNode.UpdateInputDesc(DFM_OFFSET_INPUT_OFFSETS_INDEX, offsetsDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update DeformableOffsets offsets input desc failed.",
            convDescInfo.nodeNameStr.c_str()),
        return false);
    FUSION_PASS_CHECK(dfmOffsetNode.UpdateOutputDesc(OUTPUT_INDEX, dfmOffsetOutDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update DeformableOffsets output desc failed.", convDescInfo.nodeNameStr.c_str()),
        return false);
    return true;
}

bool ADeformableConv2dPass::UpdateConv2dDesc(GNode &conv2dNode)
{
    convDescInfo.fmapDesc = dfmOffsetOutDesc;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(&conv2dNode, convDescInfo), return false);
    FUSION_PASS_CHECK(conv2dNode.UpdateOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update output tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    return true;
}

GraphUniqPtr ADeformableConv2dPass::Replacement(const GNode &convNode)
{
    auto graphBuilder = EsGraphBuilder("replacement");

    auto x = graphBuilder.CreateInput(static_cast<int64_t>(INPUT_FMAP_INDEX));
    auto filter = graphBuilder.CreateInput(static_cast<int64_t>(INPUT_FILTER_INDEX));
    auto offsets = graphBuilder.CreateInput(static_cast<int64_t>(INPUT_OFFSETS_INDEX));
    auto dfmOffset = DeformableOffsets(x, offsets, baseAttrs.strides, baseAttrs.pads, ksize, baseAttrs.dilations,
        baseAttrs.dataFormat.GetString(), dfmGroups, modulated);
    auto *dfmOffsetNode = dfmOffset.GetProducer();
    FUSION_PASS_CHECK(dfmOffsetNode == nullptr,
        OP_LOGE(FUSION_NAME, "%s get DeformableOffsets node failed.", convDescInfo.nodeNameStr.c_str()),
        return nullptr);
    FUSION_PASS_CHECK_NOLOG(!UpdateDfmOffsetDesc(*dfmOffsetNode), return nullptr);

    std::vector<EsTensorHolder> inputs = {dfmOffset, filter};
    if (convDescInfo.hasBias) {
        inputs.emplace_back(graphBuilder.CreateInput(static_cast<int64_t>(INPUT_DFM_BIAS_INDEX)));
    }

    auto *replaceGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode conv2dNode;
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::BuildConv2dNode(
        replaceGraph, convDescInfo.nodeNameStr + "_To_Conv2D", inputs, conv2dNode),
        OP_LOGE(FUSION_NAME, "%s build Conv2D node failed.", convDescInfo.nodeNameStr.c_str()), return nullptr);
    FUSION_PASS_CHECK_NOLOG(!SetConv2dAttrs(conv2dNode), return nullptr);
    FUSION_PASS_CHECK_NOLOG(!UpdateConv2dDesc(conv2dNode), return nullptr);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(conv2dNode, OUTPUT_INDEX);
    FUSION_PASS_CHECK(yHolder == nullptr,
        OP_LOGE(FUSION_NAME, "%s get Conv2D output tensor holder failed.", convDescInfo.nodeNameStr.c_str()),
        return nullptr);

    return graphBuilder.BuildAndReset({EsTensorHolder(yHolder)});
}

#if GE_COMPILER_VERSION_NUM >= 90000000U
REG_DECOMPOSE_PASS(ADeformableConv2dPass, {DfmConv2dFusion::DFM_CONV2D})
    .Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
