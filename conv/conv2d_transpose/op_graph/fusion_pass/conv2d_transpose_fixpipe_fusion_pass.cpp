/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv2d_transpose_fixpipe_fusion_pass.cpp
 * \brief conv2dtranspose to extendconvtranspose fusion pass
 * (conv2dtranspose -> postcube ===> extendConvTranspose, 2D input expand to 3D, 3D output squeeze to 2D,
 * A16W8 weight rearranged by the RearrangeA16W8WeightFzOp host node before ExtendConvTranspose)
 */
#include "conv2d_transpose_fixpipe_fusion_pass.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "conv/common/op_host/conv_npu_arch_resolver.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/graph_rewriter.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

namespace ops {
using namespace Conv2DTransposeFixpipeFusion;
using namespace ConvBackpropFusionUtils;
using namespace ge;
using namespace ge::es;
using namespace fusion;

// alias to avoid name conflicts between ConvFusionUtils and ConvBackpropFusionUtils/cube_utils
using ConvFusionUtilsPassAlias = Ops::NN::Conv::ConvFusionUtils::ConvFusionUtilsPass;
namespace ConvFusionUtilsAlias = Ops::NN::Conv::ConvFusionUtils;

namespace {
// SetM1Dequant node attribute names, keep consistent with cube_utils
constexpr char const* kAttrSingleOp = "_is_single_op";
constexpr char const* kAttrOutDtype = "dequantouttype";
constexpr char const* kAttrOffset = "offset";
constexpr char const* kSetQuantOffsetScaleOp = "SetQuantOffsetScale";

bool IsDav5162Arch()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }
    return platformInfo.str_info.short_soc_version == "Ascend5162";
}
// the SetM1Dequant host op packs the requant zero point into M1 bits[45:37] as a 9-bit
// two's-complement field: it reads the "offset" attr as a float, converts it to an int and
// shifts it left by 37, so a negative zero point must be pre-encoded with the 9-bit mask
// (e.g. zp -128 -> 384)
constexpr int32_t kRequantZeroPointMask = 0x1FF;
// symmetric int8 requant zero point, used when no requant zero point attr is available
constexpr int32_t kSymmetricInt8RequantZeroPoint = -128;

// read the requant zero point from the "offset" attr, the attr may be stored as float, int32
// or int64 by different graph sources
bool GetZeroPointAttr(const GNode& node, int32_t& zeroPoint)
{
    float floatOffset = 0.0f;
    if (node.GetAttr(kAttrOffset, floatOffset) == GRAPH_SUCCESS) {
        zeroPoint = static_cast<int32_t>(floatOffset);
        return true;
    }
    int32_t int32Offset = 0;
    if (node.GetAttr(kAttrOffset, int32Offset) == GRAPH_SUCCESS) {
        zeroPoint = int32Offset;
        return true;
    }
    int64_t int64Offset = 0;
    if (node.GetAttr(kAttrOffset, int64Offset) == GRAPH_SUCCESS) {
        zeroPoint = static_cast<int32_t>(int64Offset);
        return true;
    }
    return false;
}

// PostCube keeps the folded AscendQuant offset as input(0) of the
// SetQuantOffsetScale node feeding FixPipe quant_scale_1.
bool GetQuantOffsetFromFixPipe(const GNode& postCubeNode, int32_t& zeroPoint)
{
    auto quantScale1 = postCubeNode.GetInDataNodesAndPortIndexs(POST_CUBE_INPUT_QUANT_SCALE_1_INDEX);
    if (quantScale1.first == nullptr) {
        return false;
    }
    AscendString producerType;
    if (quantScale1.first->GetType(producerType) != GRAPH_SUCCESS || producerType != kSetQuantOffsetScaleOp) {
        return false;
    }
    TensorDesc offsetDesc;
    if (quantScale1.first->GetInputDesc(0, offsetDesc) != GRAPH_SUCCESS || offsetDesc.GetDataType() != DT_FLOAT) {
        return false;
    }
    Tensor offsetTensor;
    if (quantScale1.first->GetInputConstData(0, offsetTensor) != GRAPH_SUCCESS || offsetTensor.GetData() == nullptr) {
        auto offsetConst = quantScale1.first->GetInDataNodesAndPortIndexs(0);
        if (offsetConst.first == nullptr || offsetConst.first->GetAttr("value", offsetTensor) != GRAPH_SUCCESS ||
            offsetTensor.GetData() == nullptr) {
            return false;
        }
    }
    zeroPoint = static_cast<int32_t>(*static_cast<const float*>(static_cast<const void*>(offsetTensor.GetData())));
    return true;
}
} // namespace

void Conv2DTransposeFixpipeFusionPass::InitMember()
{
    ConvBackpropFusionBasePass::InitMember();
    postCubeNode = nullptr;
    postCubeFusionOps.clear();
    hasPostCube = false;
    hasScale = false;
    hasBias = false;
    hasScale1 = false;
    allPostCubeNodes.clear();
    allPostCubeFusionOps.clear();
    allOtherNodes.clear();
    outputCase = Conv2DTransposeFixpipeFusion::OutputCase::SINGLE;
    firstOutputNode = nullptr;
    convTransposeOutputDesc = TensorDesc();
    hasRearrangeWeight = false;
    biasDesc = TensorDesc();
    outputDesc1 = TensorDesc();
    outputPadding = {0, 0, 0, 0};
    offsetX = 0;
    fusionMode = DEFAULT_FUSION_MODE;
    requantOffset = 0.0f;
    requantOffset1 = 0.0f;
    quantOffsetsByPostCubeName.clear();
    transposeNode = nullptr;
    transposeOutputDesc = TensorDesc();
    hasFusedTranspose = false;
}

ge::AscendString Conv2DTransposeFixpipeFusionPass::GetNodeType() const
{
    static const AscendString kPassName(FUSION_NAME.c_str());
    return kPassName;
}

bool Conv2DTransposeFixpipeFusionPass::MeetRequirements(const GNode& convTransposeNode)
{
    OP_LOGD(GetNodeType().GetString(), "Enter MeetRequirements");

    // soc check: support DAV_3510 arch or cube_vector_combine = fuse
    OP_CHECK_IF(!ConvFusionUtilsPassAlias::CheckSocList(SUPPORT_SOC_LIST, npuArch, true),
                OP_LOGD(GetNodeType().GetString(), "Current soc not supported, no fusion."), return false);

    return true;
}

bool Conv2DTransposeFixpipeFusionPass::GetNodeDesc(const GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeDesc(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeDesc failed"), return false);

    // for 2D transpose, x and filter are both 4 dims input
    OP_CHECK_IF(!Update2DInputDesc(input1Desc), OP_LOGE(GetNodeType().GetString(), "Update input1Desc failed"),
                return false);
    OP_CHECK_IF(!Update2DInputDesc(input2Desc), OP_LOGE(GetNodeType().GetString(), "Update input2Desc failed"),
                return false);

    // fp32 x/filter is converted to the fp16 path of ExtendConvTranspose (migrated from the
    // original fixpipe pass), the A16W8 rearrange condition below reads the converted dtype
    if (input1Desc.GetDataType() == ge::DT_FLOAT) {
        input1Desc.SetDataType(ge::DT_FLOAT16);
    }
    if (input2Desc.GetDataType() == ge::DT_FLOAT) {
        input2Desc.SetDataType(ge::DT_FLOAT16);
    }

    hasBias = node.GetInputsSize() > static_cast<size_t>(BIAS_INDEX);
    if (hasBias) {
        OP_CHECK_IF(node.GetInputDesc(BIAS_INDEX, biasDesc) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Get bias desc failed"), return false);
        if (IsDav5162Arch() && biasDesc.GetDataType() == ge::DT_FLOAT) {
            biasDesc.SetDataType(ge::DT_FLOAT16);
        }
    }

    return true;
}

bool Conv2DTransposeFixpipeFusionPass::GetNodeAttrs(const GNode& node)
{
    // the base pass also reads hf32/_op_impl_mode_enum/padding, they are not propagated to
    // ExtendConvTranspose: only int8 x + int8 filter and fp16 x + int8 filter (A16W8) are fused
    // by this pass, so there is no fp32 or hf32 path
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeAttrs(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeAttrs failed"), return false);

    // output_padding/offset_x keep the default values reset by InitMember when the attrs are absent
    if (node.GetAttr("output_padding", outputPadding) != GRAPH_SUCCESS) {
        OP_LOGD(GetNodeType().GetString(), "Get output_padding attr failed, use default value.");
    }
    if (node.GetAttr("offset_x", offsetX) != GRAPH_SUCCESS) {
        OP_LOGD(GetNodeType().GetString(), "Get offset_x attr failed, use default value.");
    }

    // expand 2D attrs to 3D: strides/dilations/output_padding insert D dim, pads insert 2 pad values ahead
    ConvBackpropFusionUtilsPass::ExpandAttrs(convBpAttr.strides, convBpAttr.pads, convBpAttr.dilations,
                                             convBpAttr.dataFormat, &outputPadding);

    // ExtendConvTranspose needs 5 strides/dilations and 6 pads, invalid attrs would produce a wrong kernel
    if (convBpAttr.strides.size() != static_cast<size_t>(CONV_DIM_LENGTH)) {
        OP_LOGD(GetNodeType().GetString(), "The size of strides is %zu, it should be %d, no fusion.",
                convBpAttr.strides.size(), CONV_DIM_LENGTH);
        return false;
    }
    if (convBpAttr.dilations.size() != static_cast<size_t>(CONV_DIM_LENGTH)) {
        OP_LOGD(GetNodeType().GetString(), "The size of dilations is %zu, it should be %d, no fusion.",
                convBpAttr.dilations.size(), CONV_DIM_LENGTH);
        return false;
    }
    if (convBpAttr.pads.size() != PADS_3D_SIZE) {
        OP_LOGD(GetNodeType().GetString(), "The size of pads is %zu, it should be %zu, no fusion.",
                convBpAttr.pads.size(), PADS_3D_SIZE);
        return false;
    }

    return true;
}

void Conv2DTransposeFixpipeFusionPass::SelectPostCubePassByWhiteList(
    std::vector<ops::PostCubePassInfo>& matchLists) const
{
    std::vector<ops::PostCubePassInfo> tmpLists(matchLists);
    matchLists.clear();
    for (auto& postCubePass : tmpLists) {
        bool allNodesSupported = true;
        for (auto& opNode : postCubePass.m_opnodes) {
            auto curNode = opNode.GetNode();
            AscendString nodeType;
            if (curNode == nullptr || curNode->GetType(nodeType) != GRAPH_SUCCESS ||
                std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), nodeType) ==
                    SUPPORTED_NODE_TYPES.end()) {
                OP_LOGD(GetNodeType().GetString(), "Node type[%s] is not supported, removing cur list.",
                        curNode == nullptr ? "null" : nodeType.GetString());
                allNodesSupported = false;
                break;
            }
        }
        if (allNodesSupported) {
            matchLists.push_back(postCubePass);
        }
    }
}

bool Conv2DTransposeFixpipeFusionPass::HasScaleNode() const { return IsScaleEnable(postCubeFusionOps); }

bool Conv2DTransposeFixpipeFusionPass::HasReluNode() const { return IsReluEnable(postCubeFusionOps); }

bool Conv2DTransposeFixpipeFusionPass::IsScaleEnable(const std::vector<ge::AscendString>& fusionOps) const
{
    return std::find(fusionOps.begin(), fusionOps.end(), kAscendDequantOp) != fusionOps.end() ||
           std::find(fusionOps.begin(), fusionOps.end(), kAscendRequantOp) != fusionOps.end() ||
           std::find(fusionOps.begin(), fusionOps.end(), kAscendQuantOp) != fusionOps.end();
}

bool Conv2DTransposeFixpipeFusionPass::IsReluEnable(const std::vector<ge::AscendString>& fusionOps) const
{
    return std::find(fusionOps.begin(), fusionOps.end(), kReluOp) != fusionOps.end();
}

bool Conv2DTransposeFixpipeFusionPass::IsCubeVectorFusePlatform() const
{
    fe::PlatFormInfos platformInfo;
    fe::OptionalInfos optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        GRAPH_SUCCESS) {
        return false;
    }
    return conv_arch::IsCubeVectorFuseSoc(platformInfo);
}

void Conv2DTransposeFixpipeFusionPass::CheckOutputTranspose(const GNode& outputNode)
{
    hasFusedTranspose = false;
    transposeNode = nullptr;

    // the Transpose successor is removed only when it is the single consumer of the output path
    // and its output is 4D NHWC (migrated from the original fixpipe pass)
    auto outNodes = outputNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    if (outNodes.size() != 1 || outNodes[0].first == nullptr) {
        return;
    }
    AscendString nodeType;
    if (outNodes[0].first->GetType(nodeType) != GRAPH_SUCCESS || nodeType != TRANSPOSE_OP) {
        return;
    }
    TensorDesc outDesc;
    if (outNodes[0].first->GetOutputDesc(OUTPUT_INDEX, outDesc) != GRAPH_SUCCESS) {
        return;
    }
    if (outDesc.GetFormat() != ge::FORMAT_NHWC || outDesc.GetShape().GetDimNum() != TRANSPOSE_OUTPUT_DIM) {
        return;
    }

    hasFusedTranspose = true;
    transposeNode = outNodes[0].first;
    transposeOutputDesc = outDesc;
}

ge::GNodePtr Conv2DTransposeFixpipeFusionPass::GetScaleSetM1Node(const GNode& postCubeNode) const
{
    auto setM1Pair = postCubeNode.GetInDataNodesAndPortIndexs(POST_CUBE_INPUT_QUANT_SCALE_0_INDEX);
    FUSION_PASS_CHECK_NOLOG(setM1Pair.first == nullptr, return nullptr);
    AscendString setM1Type;
    FUSION_PASS_CHECK_NOLOG(setM1Pair.first->GetType(setM1Type) != GRAPH_SUCCESS || setM1Type != SET_M1_DEQUANT_OP,
                            return nullptr);
    return setM1Pair.first;
}

void Conv2DTransposeFixpipeFusionPass::CaptureRequantOffsets(const std::string& nodeName,
                                                             const std::vector<ops::PostCubePassInfo>& matchLists)
{
    for (const auto& pass : matchLists) {
        // The INT8 output can be an AscendQuant or an AscendDequant produced by
        // dequant/requant processing. Keep the offset of the final quantized op
        // on each branch before PostCube removes it.
        for (size_t index = pass.m_opnodes.size(); index > 1; --index) {
            const auto& quantNode = pass.m_opnodes[index - 1].GetNode();
            if (quantNode == nullptr) {
                continue;
            }
            AscendString nodeType;
            if (quantNode->GetType(nodeType) != GRAPH_SUCCESS ||
                (nodeType != kAscendQuantOp && nodeType != kAscendRequantOp && nodeType != kAscendDequantOp)) {
                continue;
            }
            TensorDesc quantOutputDesc;
            if (quantNode->GetOutputDesc(OUTPUT_INDEX, quantOutputDesc) != GRAPH_SUCCESS ||
                quantOutputDesc.GetDataType() != ge::DT_INT8) {
                continue;
            }
            int32_t zeroPoint = 0;
            if (GetZeroPointAttr(*quantNode, zeroPoint)) {
                quantOffsetsByPostCubeName[nodeName + "_pass." + std::to_string(pass.pass_index)] = zeroPoint;
                break;
            }
        }
    }
}

void Conv2DTransposeFixpipeFusionPass::ResolveRequantOffset()
{
    requantOffset = 0.0f;
    requantOffset1 = 0.0f;
    if (hasScale && outputDesc.GetDataType() == ge::DT_INT8) {
        auto setM1Node = GetScaleSetM1Node(*postCubeNode);
        if (setM1Node != nullptr) {
            requantOffset = ResolveRequantZeroPoint(*setM1Node, *postCubeNode);
        }
    }
    // The second requant path must be checked independently: y0 may be FP16 while y1 is INT8.
    if (hasScale1 && outputDesc1.GetDataType() == ge::DT_INT8 && !allPostCubeNodes.empty()) {
        auto setM1Node1 = GetScaleSetM1Node(*allPostCubeNodes.back());
        if (setM1Node1 != nullptr) {
            requantOffset1 = ResolveRequantZeroPoint(*setM1Node1, *allPostCubeNodes.back());
        }
    }
}

float Conv2DTransposeFixpipeFusionPass::ResolveRequantZeroPoint(const GNode& setM1Node, const GNode& postCubeNode) const
{
    AscendString postCubeName;
    if (postCubeNode.GetName(postCubeName) == GRAPH_SUCCESS) {
        const auto quantOffset = quantOffsetsByPostCubeName.find(postCubeName.GetString());
        if (quantOffset != quantOffsetsByPostCubeName.end()) {
            return static_cast<float>(quantOffset->second & kRequantZeroPointMask);
        }
    }
    int32_t requantZeroPoint = 0;
    if (GetQuantOffsetFromFixPipe(postCubeNode, requantZeroPoint)) {
        OP_LOGD(GetNodeType().GetString(), "Use FixPipe quant_scale_1 offset %d for requant M1.", requantZeroPoint);
        return static_cast<float>(requantZeroPoint & kRequantZeroPointMask);
    }
    // Existing graphs can provide a FixPipe without a selected quantized output match.
    if (!GetZeroPointAttr(setM1Node, requantZeroPoint) && !GetZeroPointAttr(postCubeNode, requantZeroPoint)) {
        requantZeroPoint = kSymmetricInt8RequantZeroPoint;
        std::string postCubeName;
        (void)ConvBackpropFusionUtilsPass::GetNodeName(postCubeNode, postCubeName);
        OP_LOGW(GetNodeType().GetString(),
                "No requant zero point attr on node[%s] or the postcube node[%s], use the "
                "symmetric int8 requant zero point %d.",
                SET_M1_DEQUANT_OP.GetString(), postCubeName.c_str(), kSymmetricInt8RequantZeroPoint);
    }
    // return the 9-bit encoded field value, Replacement sets it as the float "offset" attr
    return static_cast<float>(requantZeroPoint & kRequantZeroPointMask);
}

bool Conv2DTransposeFixpipeFusionPass::NeedRearrangeWeightNode() const
{
    // A16W8 weight trans (fp16/fp32 x + int8 filter, both group and non-group scenarios, migrated
    // from the original fixpipe pass): the int8 weight is rearranged by the
    // RearrangeA16W8WeightFzOp host node for the AIC; the fp32 x is already converted to fp16
    return input1Desc.GetDataType() == ge::DT_FLOAT16 && input2Desc.GetDataType() == SUPPORT_FILTER_DTYPE;
}

bool Conv2DTransposeFixpipeFusionPass::CreateRearrangeWeightNode(EsGraphBuilder& builder, EsTensorHolder& filterInput,
                                                                 const TensorDesc& filterInputDesc,
                                                                 const std::string& nodeNamePrefix,
                                                                 EsTensorHolder& rearrangeOutput)
{
    auto* innerGraph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(innerGraph == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get inner graph of replacement builder failed."), return false);

    // the rearrange op input/output desc follows the filter desc of ExtendConvTranspose (the
    // unsqueezed 3D desc or the kept fractal desc), the output dtype is int8 which the AIC
    // requires for the A16W8 weight
    TensorDesc rearrangeOutDesc = filterInputDesc;
    rearrangeOutDesc.SetDataType(SUPPORT_FILTER_DTYPE);

    std::string rearrangeName = nodeNamePrefix + "_BEFORE_rearrange_a16w8_weight";
    ge::GNode rearrangeNode = ge::es::CompliantNodeBuilder(innerGraph)
                                  .OpType(REARRANGE_A16W8_WEIGHT_FZ_OP.GetString())
                                  .Name(rearrangeName.c_str())
                                  .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                  .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                  .Build();

    // conv2dtranspose consumes the weight in reverse order, the reverse mode and groups attrs
    // keep consistent with the original fixpipe pass
    int64_t rearrangeGroups = convBpAttr.groups;
    int64_t reverseMode = REARRANGE_REVERSE_MODE;
    OP_CHECK_IF(rearrangeNode.SetAttr(kAttrRearrangeGroups, rearrangeGroups) != GRAPH_SUCCESS ||
                    rearrangeNode.SetAttr(kAttrReverseMode, reverseMode) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Set attrs for rearrange weight node failed."), return false);

    auto* filterProducer = filterInput.GetProducer();
    OP_CHECK_IF(filterProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Get filter input producer failed."),
                return false);
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*innerGraph, *filterProducer, filterInput.GetProducerOutIndex(),
                                                 rearrangeNode, REARRANGE_INPUT_X_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge from filter input to rearrange weight node failed."),
                return false);
    OP_CHECK_IF(rearrangeNode.UpdateInputDesc(REARRANGE_INPUT_X_INDEX, filterInputDesc) != GRAPH_SUCCESS ||
                    rearrangeNode.UpdateOutputDesc(OUTPUT_INDEX, rearrangeOutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update rearrange weight node desc failed."), return false);

    auto* rearrangeOutHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(rearrangeNode, OUTPUT_INDEX);
    OP_CHECK_IF(rearrangeOutHolder == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get rearrange weight output tensor holder failed."), return false);
    rearrangeOutput = EsTensorHolder(rearrangeOutHolder);
    return true;
}

bool Conv2DTransposeFixpipeFusionPass::ReadAndCheckPostCubeFusionOps(const GNode& postCubeNode,
                                                                     std::vector<AscendString>& fusionOps) const
{
    // read fusion op list from PostCube, all ops must be in white list
    FUSION_PASS_CHECK(postCubeNode.GetAttr(FUSION_OP_LIST, fusionOps) != GRAPH_SUCCESS,
                      OP_LOGE(GetNodeType().GetString(), "Get fusion op list from PostCube unsuccessfully."),
                      return false);

    AscendString supportedListStr = ConvFusionUtilsPassAlias::ListToAscendString(SUPPORTED_NODE_TYPES);
    for (auto& fusionOp : fusionOps) {
        if (std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), fusionOp) ==
            SUPPORTED_NODE_TYPES.end()) {
            std::stringstream reason;
            reason << "In the fusion pass " << FUSION_NAME
                   << ", the postcube fusion op list can only contain the following types: "
                   << supportedListStr.GetString() << ", but got " << fusionOp.GetString();
            OP_LOGD(GetNodeType().GetString(), "%s, skip this node.", reason.str().c_str());
            return false;
        }
    }

    if (IsScaleEnable(fusionOps)) {
        // check postcube unit input format: quant_scale_0
        TensorDesc quantScale0Desc;
        FUSION_PASS_CHECK(
            postCubeNode.GetInputDesc(POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, quantScale0Desc) != GRAPH_SUCCESS,
            OP_LOGE(GetNodeType().GetString(), "Get PostCube quant_scale_0 tensor desc failed."), return false);
        Format quantScale0Format = quantScale0Desc.GetFormat();
        if (quantScale0Format != FORMAT_ND && quantScale0Format != FORMAT_NCHW) {
            std::string checkFormatStr = ConvFusionUtilsAlias::GeFormatToString(quantScale0Format);
            std::string correctFormat = ConvFusionUtilsAlias::GeFormatToString(FORMAT_ND) + ", " +
                                        ConvFusionUtilsAlias::GeFormatToString(FORMAT_NCHW);
            OP_LOGD(GetNodeType().GetString(),
                    "The format %s of quant_scale_0 is not supported, it should be %s, skip this node.",
                    checkFormatStr.c_str(), correctFormat.c_str());
            return false;
        }

        // check postcube unit input dtype: quant_scale_0, ExtendConvTranspose only accepts uint64 scale
        DataType quantScale0Dtype = quantScale0Desc.GetDataType();
        if (quantScale0Dtype != DT_UINT64) {
            std::string checkDtypeStr = ConvFusionUtilsAlias::GeDtypeToString(quantScale0Dtype);
            std::string correctDtypeStr = ConvFusionUtilsAlias::GeDtypeToString(DT_UINT64);
            OP_LOGD(GetNodeType().GetString(),
                    "The dtype %s of quant_scale_0 is not supported, it should be %s, skip this node.",
                    checkDtypeStr.c_str(), correctDtypeStr.c_str());
            return false;
        }
    }

    return true;
}

void Conv2DTransposeFixpipeFusionPass::CollectOutputNodes(const GNode& convTransposeNode)
{
    auto convOutputNodes = convTransposeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& outputNode : convOutputNodes) {
        GNodePtr nodePtr = outputNode.first;
        if (nodePtr == nullptr) {
            continue;
        }
        AscendString nodeType;
        if (nodePtr->GetType(nodeType) != GRAPH_SUCCESS) {
            continue;
        }
        if (nodeType == POST_CUBE_OP) {
            allPostCubeNodes.emplace_back(nodePtr);
        } else {
            // passthrough consumer of the conv2dtranspose output (node + input index)
            allOtherNodes.emplace_back(nodePtr, static_cast<int32_t>(outputNode.second));
        }
    }
}

void Conv2DTransposeFixpipeFusionPass::SetOutputCase()
{
    // the dual output cases are enabled only on the cube vector fuse soc with exactly two
    // outputs (migrated from the updated fixpipe pass), all the other combinations keep the
    // single output behaviour
    if (!IsCubeVectorFusePlatform() || allPostCubeNodes.size() + allOtherNodes.size() != DUAL_OUTPUT_NUM) {
        outputCase = Conv2DTransposeFixpipeFusion::OutputCase::SINGLE;
        return;
    }
    if (allPostCubeNodes.size() == DUAL_OUTPUT_NUM) {
        outputCase = Conv2DTransposeFixpipeFusion::OutputCase::DUAL_FIXPIPE;
        return;
    }
    // The V2 INT8 secondary output is inaccurate on the cube vector fuse SoC when the
    // primary output is FP16. Place the quantized FixPipe path on y0 for mixed outputs;
    // ConstructBoundary maps each output back to its original consumers.
    TensorDesc fixPipeOutputDesc;
    if (allPostCubeNodes[0]->GetOutputDesc(OUTPUT_INDEX, fixPipeOutputDesc) == GRAPH_SUCCESS &&
        fixPipeOutputDesc.GetDataType() == ge::DT_INT8 &&
        (convTransposeOutputDesc.GetDataType() == ge::DT_FLOAT16 ||
         convTransposeOutputDesc.GetDataType() == ge::DT_FLOAT)) {
        outputCase = Conv2DTransposeFixpipeFusion::OutputCase::FIXPIPE_CONV2DT;
        return;
    }
    // For other mixed outputs, retain the original consumer order.
    bool isPostCubeFirst = firstOutputNode == allPostCubeNodes[0];
    outputCase = isPostCubeFirst ? Conv2DTransposeFixpipeFusion::OutputCase::FIXPIPE_CONV2DT :
                                   Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE;
}

PostCubeFusionResult Conv2DTransposeFixpipeFusionPass::GetPostCubeNodes(const GNode& convTransposeNode)
{
    auto convOutputNodes = convTransposeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    OP_CHECK_IF(convOutputNodes.empty(),
                OP_LOGD(GetNodeType().GetString(), "Conv2DTranspose has no out data nodes, no postcube fusion."),
                return PostCubeFusionResult::kNoPostCube);

    // remember the first consumer of the conv2dtranspose output for the output order of the
    // mixed dual output cases
    firstOutputNode = convOutputNodes[0].first;

    int32_t postCubeNum = 0;
    for (auto& outputNode : convOutputNodes) {
        GNodePtr nodePtr = outputNode.first;
        if (nodePtr == nullptr) {
            continue;
        }
        AscendString nodeType;
        OP_CHECK_IF(nodePtr->GetType(nodeType) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Get node type failed."),
                    return PostCubeFusionResult::kSkipNode);
        if (nodeType == POST_CUBE_OP) {
            postCubeNum += 1;
        }
    }

    if (postCubeNum == 0) {
        OP_LOGD(GetNodeType().GetString(), "Conv2DTranspose output does not have postcube node, no postcube fusion.");
        return PostCubeFusionResult::kNoPostCube;
    }

    // collect the FixPipe branches and the passthrough consumers (migrated from the updated
    // fixpipe pass), the dual output cases are decided by the total output count
    CollectOutputNodes(convTransposeNode);
    if (allPostCubeNodes.size() + allOtherNodes.size() > DUAL_OUTPUT_NUM) {
        std::stringstream reason;
        reason << "In the fusion pass " << FUSION_NAME << ", unsupported multiple outputs with "
               << allPostCubeNodes.size() << " postcube and " << allOtherNodes.size() << " other nodes";
        OP_LOGD(GetNodeType().GetString(), "%s, skip this node.", reason.str().c_str());
        return PostCubeFusionResult::kSkipNode;
    }

    // the conv2dtranspose original output desc is the passthrough path desc of the mixed cases
    (void)convTransposeNode.GetOutputDesc(OUTPUT_INDEX, convTransposeOutputDesc);

    // Keep INT8 on y0 when the two FixPipe branches have mixed output dtypes. The V2
    // secondary quantization path saturates for FP16 y0 + INT8 y1; the boundary uses the
    // original FixPipe nodes, so changing their order does not change graph output order.
    if (allPostCubeNodes.size() == DUAL_OUTPUT_NUM) {
        TensorDesc firstOutputDesc;
        TensorDesc secondOutputDesc;
        if (allPostCubeNodes[0]->GetOutputDesc(OUTPUT_INDEX, firstOutputDesc) == GRAPH_SUCCESS &&
            allPostCubeNodes[1]->GetOutputDesc(OUTPUT_INDEX, secondOutputDesc) == GRAPH_SUCCESS &&
            (firstOutputDesc.GetDataType() == ge::DT_FLOAT16 || firstOutputDesc.GetDataType() == ge::DT_FLOAT) &&
            secondOutputDesc.GetDataType() == ge::DT_INT8) {
            std::swap(allPostCubeNodes[0], allPostCubeNodes[1]);
        }
    }

    // the front members always alias the first FixPipe branch for the existing single path logic
    postCubeNode = allPostCubeNodes.front();
    allPostCubeFusionOps.resize(allPostCubeNodes.size());
    for (size_t i = 0; i < allPostCubeNodes.size(); ++i) {
        if (!ReadAndCheckPostCubeFusionOps(*allPostCubeNodes[i], allPostCubeFusionOps[i])) {
            OP_LOGW(GetNodeType().GetString(), "PostCube case not supported, skip this node.");
            return PostCubeFusionResult::kSkipNode;
        }
    }
    postCubeFusionOps = allPostCubeFusionOps.front();
    hasScale1 = allPostCubeNodes.size() > 1 && IsScaleEnable(allPostCubeFusionOps[1]);

    SetOutputCase();
    if (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE) {
        // The only FixPipe is on y1, so its quant_scale_0 feeds scale1 rather than scale0.
        hasScale1 = IsScaleEnable(postCubeFusionOps);
    }

    // more than one FixPipe branch is only handled by the dual output cases, the single output
    // path only adopts the first branch and the leftover FixPipe would break the graph; the
    // equivalent dual outputs (same dtype and same relu behaviour) also keep the dual output
    // form: the two V2 output paths carry the same value, which is functionally equivalent to
    // the single output downgrade of the original fixpipe pass
    if (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::SINGLE &&
        (allPostCubeNodes.size() > 1 || !allOtherNodes.empty())) {
        std::stringstream reason;
        reason << "In the fusion pass " << FUSION_NAME
               << ", the Conv2DTranspose output consumed by both postcube and other nodes is not supported";
        OP_LOGD(GetNodeType().GetString(), "%s, skip this node.", reason.str().c_str());
        return PostCubeFusionResult::kSkipNode;
    }

    return PostCubeFusionResult::kFused;
}

PostCubeFusionResult Conv2DTransposeFixpipeFusionPass::PostCubeFusionImpl(ge::GraphPtr& graph,
                                                                          const GNode& convTransposeNode,
                                                                          ge::CustomPassContext& passContext)
{
    // conv2dtranspose must have output nodes before postcube fusion
    auto convOutputNodes = convTransposeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    if (convOutputNodes.empty()) {
        OP_LOGD(GetNodeType().GetString(), "Conv2DTranspose out nodes is empty, skip postcube fusion.");
        return PostCubeFusionResult::kNoPostCube;
    }

    // the postcube node is created with the conv2dtranspose node name, same as the entity name
    // of the pass reports
    std::string nodeName;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::GetNodeName(convTransposeNode, nodeName) || nodeName.empty(),
                OP_LOGE(GetNodeType().GetString(), "Get node name failed."), return PostCubeFusionResult::kNoPostCube);
    GNodePtr nodePtr = graph->FindNodeByName(nodeName.c_str());
    OP_CHECK_IF(nodePtr == nullptr, OP_LOGE(GetNodeType().GetString(), "Find node by name failed."),
                return PostCubeFusionResult::kNoPostCube);

    // [Step 1] Determine which nodes in the subsequent nodes satisfy postcube hardware unit
    ops::PostCubeUtils postCubeUtils;
    if (postCubeUtils.GetPostCubeNodeList(nodePtr, passContext) != GRAPH_SUCCESS) {
        OP_LOGD(GetNodeType().GetString(), "GetPostCubeNodeList failed, skip postcube fusion.");
        return PostCubeFusionResult::kNoPostCube;
    }

    // [Step 2] To customize the selection of the postcube fusion range
    SelectPostCubePassByWhiteList(postCubeUtils.m_matchpasses_);

    // [Step 3] PostCube tool method selects 1~2 PostCube paths
    if (postCubeUtils.SelectPostCubeNodeList(true) != GRAPH_SUCCESS) {
        OP_LOGD(GetNodeType().GetString(), "SelectPostCubeNodeList failed, skip postcube fusion.");
        return PostCubeFusionResult::kNoPostCube;
    }

    // [Step 3.5] all skip decisions are made before the creation: CreatePostCubeNode deletes the
    // folded post op chain and the graph cannot be restored afterwards, a skip after it would
    // leave the engine-less FixPipe node in the graph and the graph compile fails with "cannot
    // find engine", so every post-creation check is derived from the selected passes and the
    // original graph before the graph is changed
    if (!CheckPostCubePassesPrepared(convTransposeNode, postCubeUtils.m_matchpasses_)) {
        OP_LOGW(GetNodeType().GetString(), "PostCube case not supported before creation, skip this node.");
        return PostCubeFusionResult::kSkipNode;
    }
    CaptureRequantOffsets(nodeName, postCubeUtils.m_matchpasses_);

    // [Step 4] Create the PostCube operator node and modify the graph, the graph may have been changed
    // even when it fails, so the created nodes are checked in both cases by GetPostCubeNodes
    std::vector<GNodePtr> newNodes;
    if (postCubeUtils.CreatePostCubeNode(nodeName, *graph, newNodes) != GRAPH_SUCCESS) {
        OP_LOGW(GetNodeType().GetString(), "CreatePostCubeNode failed, check the created postcube nodes.");
    }

    // [Step 5] Get the created PostCube node and check the support case
    return GetPostCubeNodes(convTransposeNode);
}

bool Conv2DTransposeFixpipeFusionPass::CheckPostCubePassesPrepared(
    const GNode& convTransposeNode, const std::vector<ops::PostCubePassInfo>& matchLists) const
{
    // no matched pass: the creation below builds nothing and GetPostCubeNodes only adopts a
    // pre-existing FixPipe node, the graph is not changed by this pass
    if (matchLists.empty()) {
        return true;
    }
    // each selected pass creates one FixPipe node consuming the conv2dtranspose output: at
    // most two postcube outputs are supported and the dual output additionally requires the
    // cube vector fuse soc (migrated from the updated fixpipe pass)
    bool fusePlatform = IsCubeVectorFusePlatform();
    if (matchLists.size() > DUAL_OUTPUT_NUM || (matchLists.size() == DUAL_OUTPUT_NUM && !fusePlatform)) {
        std::stringstream reason;
        reason << "In the fusion pass " << FUSION_NAME << ", at most " << DUAL_OUTPUT_NUM
               << " postcube outputs of Conv2DTranspose are supported on the current soc, "
               << "but got " << matchLists.size() << " postcube nodes";
        OP_LOGD(GetNodeType().GetString(), "%s, skip this node.", reason.str().c_str());
        return false;
    }

    // collect the folded post ops of the selected passes: the creation builds the FixPipe
    // fusion_op_list from their types, copies the FixPipe output desc from the last one and
    // deletes all of them from the graph; the scale source and the tail node are traced per
    // pass because every FixPipe branch carries its own quant_scale_0 and output desc
    std::vector<GNodePtr> foldedNodes;
    std::vector<AscendString> foldedNodeNames;
    std::vector<GNodePtr> scaleNodes;
    std::vector<bool> scaleNodesAreQuantOp;
    std::vector<GNodePtr> tailNodes;
    for (auto& postCubePass : matchLists) {
        GNodePtr passScaleNode = nullptr;
        bool passScaleNodeIsQuantOp = false;
        size_t passFoldedNum = 0;
        for (size_t index = 1; index < postCubePass.m_opnodes.size(); ++index) {
            auto foldedNode = postCubePass.m_opnodes[index].GetNode();
            FUSION_PASS_CHECK(foldedNode == nullptr, OP_LOGE(GetNodeType().GetString(), "Get folded post node failed."),
                              return false);
            AscendString foldedType;
            FUSION_PASS_CHECK(foldedNode->GetType(foldedType) != GRAPH_SUCCESS,
                              OP_LOGE(GetNodeType().GetString(), "Get folded post node type failed."), return false);
            AscendString foldedName;
            FUSION_PASS_CHECK(foldedNode->GetName(foldedName) != GRAPH_SUCCESS,
                              OP_LOGE(GetNodeType().GetString(), "Get folded post node name failed."), return false);
            foldedNodes.push_back(foldedNode);
            foldedNodeNames.push_back(foldedName);
            passFoldedNum += 1;
            // the first dequant/requant/quant op is the quant_scale_0 source of the FixPipe
            if (passScaleNode == nullptr &&
                (foldedType == kAscendDequantOp || foldedType == kAscendRequantOp || foldedType == kAscendQuantOp)) {
                passScaleNode = foldedNode;
                passScaleNodeIsQuantOp = foldedType == kAscendQuantOp;
            }
        }
        scaleNodes.push_back(passScaleNode);
        scaleNodesAreQuantOp.push_back(passScaleNodeIsQuantOp);
        // the tail node of the pass is its last folded op, nullptr when the pass folds nothing
        tailNodes.push_back(passFoldedNum > 0 ? foldedNodes.back() : nullptr);
    }

    // every consumer of the conv2dtranspose output must be folded into the FixPipe unless the
    // dual output is enabled on the fuse soc (the unfolded consumers become the passthrough
    // paths, pre-check of the other consumer case in GetPostCubeNodes)
    uint32_t passthroughNum = 0;
    auto convOutputNodes = convTransposeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& outputNode : convOutputNodes) {
        // GNode handles returned by different apis are distinct objects of the same node, the
        // absorption is decided by the node name instead of the pointer
        AscendString consumerName;
        if (outputNode.first == nullptr || outputNode.first->GetName(consumerName) != GRAPH_SUCCESS ||
            std::find(foldedNodeNames.begin(), foldedNodeNames.end(), consumerName) == foldedNodeNames.end()) {
            if (!fusePlatform) {
                std::stringstream reason;
                reason << "In the fusion pass " << FUSION_NAME
                       << ", the Conv2DTranspose output consumed by both postcube and other nodes is not supported";
                OP_LOGD(GetNodeType().GetString(), "%s, skip this node.", reason.str().c_str());
                return false;
            }
            passthroughNum += 1;
        }
    }
    // the FixPipe branches plus the passthrough consumers take at most the two output paths
    // of the dual output
    if (passthroughNum > 0 && matchLists.size() + passthroughNum > DUAL_OUTPUT_NUM) {
        std::stringstream reason;
        reason << "In the fusion pass " << FUSION_NAME << ", the dual output expects at most " << DUAL_OUTPUT_NUM
               << " output paths, but got " << matchLists.size() << " postcube and " << passthroughNum
               << " passthrough consumers";
        OP_LOGD(GetNodeType().GetString(), "%s, skip this node.", reason.str().c_str());
        return false;
    }
    // a selected pass without any folded op would create a meaningless FixPipe which folds
    // nothing, the conv2dtranspose output consumers above cannot be folded by such a pass either
    FUSION_PASS_CHECK(foldedNodes.empty(),
                      OP_LOGE(GetNodeType().GetString(), "No folded post node in the selected passes."), return false);

    // the FixPipe quant_scale_0 desc is copied from the scale input of the first dequant/requant
    // op with the dtype forced to uint64, its format is carried over, so the scale input must
    // have a producer and a supported format (pre-check of the quant_scale_0 case in
    // ReadAndCheckPostCubeFusionOps; the AscendQuant scale is generated from its attrs and has
    // no original graph source, so it is not checked here)
    for (size_t i = 0; i < scaleNodes.size(); ++i) {
        if (scaleNodes[i] == nullptr || scaleNodesAreQuantOp[i]) {
            continue;
        }
        auto scaleInput = scaleNodes[i]->GetInDataNodesAndPortIndexs(QUANT_OP_SCALE_INPUT_INDEX);
        FUSION_PASS_CHECK(scaleInput.first == nullptr,
                          OP_LOGE(GetNodeType().GetString(), "Get scale input producer failed."), return false);
        TensorDesc quantScale0Desc;
        FUSION_PASS_CHECK(scaleNodes[i]->GetInputDesc(QUANT_OP_SCALE_INPUT_INDEX, quantScale0Desc) != GRAPH_SUCCESS,
                          OP_LOGE(GetNodeType().GetString(), "Get PostCube quant_scale_0 tensor desc failed."),
                          return false);
        Format quantScale0Format = quantScale0Desc.GetFormat();
        if (quantScale0Format != FORMAT_ND && quantScale0Format != FORMAT_NCHW) {
            std::string checkFormatStr = ConvFusionUtilsAlias::GeFormatToString(quantScale0Format);
            std::string correctFormat = ConvFusionUtilsAlias::GeFormatToString(FORMAT_ND) + ", " +
                                        ConvFusionUtilsAlias::GeFormatToString(FORMAT_NCHW);
            OP_LOGD(GetNodeType().GetString(),
                    "The format %s of quant_scale_0 is not supported, it should be %s, skip this node.",
                    checkFormatStr.c_str(), correctFormat.c_str());
            return false;
        }
    }

    // the FixPipe output desc is copied from the output desc of the last folded op of every
    // pass, the final output dtype/format check of the postcube conversion is derived here from
    // the original graph before it is changed
    for (auto& tailNode : tailNodes) {
        if (tailNode == nullptr) {
            continue;
        }
        TensorDesc tailOutputDesc;
        FUSION_PASS_CHECK(tailNode->GetOutputDesc(OUTPUT_INDEX, tailOutputDesc) != GRAPH_SUCCESS,
                          OP_LOGE(GetNodeType().GetString(), "Get postcube output tensor desc failed."), return false);
        DataType outDtype = tailOutputDesc.GetDataType();
        if (std::find(SUPPORT_OUTPUT_DTYPES.begin(), SUPPORT_OUTPUT_DTYPES.end(), outDtype) ==
            SUPPORT_OUTPUT_DTYPES.end()) {
            std::string checkDtypeStr = ConvFusionUtilsAlias::GeDtypeToString(outDtype);
            std::string supportDtypesStr = ConvFusionUtilsAlias::VectorToString(SUPPORT_OUTPUT_DTYPES);
            OP_LOGD(GetNodeType().GetString(),
                    "The output dtype %s of y is not supported, it should be within the range %s, skip this node.",
                    checkDtypeStr.c_str(), supportDtypesStr.c_str());
            return false;
        }
        if (tailOutputDesc.GetFormat() != SUPPORT_FORMAT_2D) {
            std::string checkFormatStr = ConvFusionUtilsAlias::GeFormatToString(tailOutputDesc.GetFormat());
            std::string supportFormatStr = ConvFusionUtilsAlias::GeFormatToString(SUPPORT_FORMAT_2D);
            OP_LOGD(GetNodeType().GetString(),
                    "The output format %s of y is not supported, it should be %s, skip this "
                    "node.",
                    checkFormatStr.c_str(), supportFormatStr.c_str());
            return false;
        }
    }

    return true;
}

bool Conv2DTransposeFixpipeFusionPass::UpdateExtendConvTransposeDesc(GNode& extendConvTransposeNode,
                                                                     const TensorDesc& unsqueezeXOutDesc,
                                                                     const TensorDesc& unsqueezeFilterOutDesc,
                                                                     const TensorDesc& output3DDesc,
                                                                     const TensorDesc& output3DDesc1)
{
    // step1: update input_size, keep the 4 elements desc: ExtendConvTranspose inserts the D dim itself
    // for a 4 elements input_size (see InferShapeForConvBackpropExtend3D), the const data is not changed
    OP_CHECK_IF(extendConvTransposeNode.UpdateInputDesc(INPUT_SIZE_INDEX, input0Desc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update input_size tensor desc failed."), return false);

    // step2: update x, 2D desc expand to 3D desc by Unsqueeze
    OP_CHECK_IF(extendConvTransposeNode.UpdateInputDesc(X_INDEX, unsqueezeXOutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update x tensor desc failed."), return false);

    // step3: update filter, 2D desc expand to 3D desc by Unsqueeze; the A16W8 rearrange weight node
    // reuses the same desc with the int8 dtype, which is already int8 for both supported scenarios
    OP_CHECK_IF(extendConvTransposeNode.UpdateInputDesc(FILTER_INDEX, unsqueezeFilterOutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update filter tensor desc failed."), return false);

    // step4: update bias
    if (hasBias) {
        OP_CHECK_IF(extendConvTransposeNode.UpdateInputDesc(BIAS_INDEX, biasDesc) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Update bias tensor desc failed."), return false);
    }

    // step5: update scale, which comes from PostCube quant_scale_0
    if (hasScale) {
        TensorDesc quantScaleTensor;
        FUSION_PASS_CHECK(
            postCubeNode->GetInputDesc(POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, quantScaleTensor) != GRAPH_SUCCESS,
            OP_LOGE(GetNodeType().GetString(), "Get PostCube quant_scale_0 tensor desc failed."), return false);
        quantScaleTensor.SetFormat(FORMAT_ND);
        quantScaleTensor.SetOriginFormat(FORMAT_ND);
        OP_CHECK_IF(extendConvTransposeNode.UpdateInputDesc(SCALE_INDEX, quantScaleTensor) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Update scale tensor desc failed."), return false);
    }

    // step5.5: the dual output updates the scale1 input desc, which comes from the back FixPipe
    // branch quant_scale_0 (migrated from the updated fixpipe pass)
    if (outputCase != Conv2DTransposeFixpipeFusion::OutputCase::SINGLE && hasScale1) {
        TensorDesc quantScaleTensor1;
        FUSION_PASS_CHECK(allPostCubeNodes.back()->GetInputDesc(POST_CUBE_INPUT_QUANT_SCALE_0_INDEX,
                                                                quantScaleTensor1) != GRAPH_SUCCESS,
                          OP_LOGE(GetNodeType().GetString(), "Get PostCube quant_scale_0 tensor desc failed."),
                          return false);
        quantScaleTensor1.SetFormat(FORMAT_ND);
        quantScaleTensor1.SetOriginFormat(FORMAT_ND);
        OP_CHECK_IF(extendConvTransposeNode.UpdateInputDesc(SCALE1_INDEX, quantScaleTensor1) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Update scale1 tensor desc failed."), return false);
    }

    // step6: update output, 2D output desc expand to 3D desc; the y1 output of the dual output
    // expands its own 2D desc the same way
    OP_CHECK_IF(extendConvTransposeNode.UpdateOutputDesc(OUTPUT_INDEX, output3DDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update output tensor desc failed."), return false);
    if (outputCase != Conv2DTransposeFixpipeFusion::OutputCase::SINGLE) {
        OP_CHECK_IF(extendConvTransposeNode.UpdateOutputDesc(1, output3DDesc1) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Update y1 tensor desc failed."), return false);
    }

    return true;
}

int64_t Conv2DTransposeFixpipeFusionPass::GetScaleGraphInputIndex() const
{
    // the replacement graph input index must be continuous from 0, scale takes the bias index when
    // the optional bias input is absent, it must be consistent between boundary and Replacement
    return hasBias ? SCALE_INDEX : BIAS_INDEX;
}

int64_t Conv2DTransposeFixpipeFusionPass::GetScale1GraphInputIndex() const
{
    // The boundary inputs are packed without gaps even if y0 has no scale0.
    if (hasScale) {
        return hasBias ? SCALE1_INDEX : SCALE_INDEX;
    }
    return hasBias ? SCALE_INDEX : BIAS_INDEX;
}

bool Conv2DTransposeFixpipeFusionPass::BuildSetM1DequantNode(EsGraphBuilder& builder, EsTensorHolder& iScale,
                                                             const GNode& setM1Node, const std::string& setM1Name,
                                                             float requantOffsetValue, const TensorDesc& pathOutputDesc,
                                                             EsTensorHolder& scaleOutput)
{
    auto* innerGraph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(innerGraph == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get inner graph of replacement builder failed."), return false);

    TensorDesc setM1InDesc;
    TensorDesc setM1OutDesc;
    OP_CHECK_IF(setM1Node.GetInputDesc(SET_M1_INPUT_INDEX, setM1InDesc) != GRAPH_SUCCESS ||
                    setM1Node.GetOutputDesc(OUTPUT_INDEX, setM1OutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Get SetM1Dequant tensor desc failed."), return false);

    ge::GNode setM1NewNode = ge::es::CompliantNodeBuilder(innerGraph)
                                 .OpType(SET_M1_DEQUANT_OP.GetString())
                                 .Name(setM1Name.c_str())
                                 .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .Build();

    // copy host op attributes from the original SetM1Dequant node
    bool isSingleOp = true;
    if (setM1Node.GetAttr(kAttrSingleOp, isSingleOp) != GRAPH_SUCCESS) {
        OP_LOGD(GetNodeType().GetString(), "Get _is_single_op attr failed, use the default value true.");
    }
    setM1NewNode.SetAttr(kAttrSingleOp, isSingleOp);
    int64_t m1OutDtype = 0;
    if (setM1Node.GetAttr(kAttrOutDtype, m1OutDtype) != GRAPH_SUCCESS) {
        OP_LOGD(GetNodeType().GetString(), "Get dequantouttype attr failed, use the default value 0.");
    }
    if (pathOutputDesc.GetDataType() == ge::DT_INT8) {
        m1OutDtype = static_cast<int64_t>(ge::DT_INT8);
    } else if (pathOutputDesc.GetDataType() == ge::DT_FLOAT16) {
        // Older PostCube graphs may carry DT_FLOAT on the source SetM1Dequant.
        // The fused ExtendConvTranspose output and its M1 scale both use FP16.
        m1OutDtype = static_cast<int64_t>(ge::DT_FLOAT16);
    }
    setM1NewNode.SetAttr(kAttrOutDtype, m1OutDtype);
    // the host op selects the M1 packing path by the presence of the "offset" attr and
    // reads it as a float (converted to an int internally):
    // - requant scenario (int8 output): it packs the value as a 9-bit two's-complement
    //   requant zero point into the M1 high word (M1 bits[45:37]), which the device
    //   kernel requires for the int8 output, so the encoded zero point (resolved by
    //   ResolveRequantOffset) must be set as the float attr value; an int attr is
    //   invisible to the host op float read and leaves the zero point bits of the M1
    //   high word zero;
    // - float output scenario: the attr must NOT be set, the with-offset path returns
    //   a zero M1 for float output dtypes
    if (m1OutDtype == static_cast<int64_t>(ge::DT_INT8)) {
        float m1Offset = requantOffsetValue;
        setM1NewNode.SetAttr(kAttrOffset, m1Offset);
        OP_LOGD(GetNodeType().GetString(), "SetM1Dequant dequantouttype is %ld, offset is %f for the rebuilt node.",
                static_cast<long>(m1OutDtype), m1Offset);
    } else {
        OP_LOGD(GetNodeType().GetString(),
                "SetM1Dequant dequantouttype is %ld, offset attr is not set for the rebuilt node.",
                static_cast<long>(m1OutDtype));
    }

    OP_CHECK_IF(setM1NewNode.UpdateInputDesc(SET_M1_INPUT_INDEX, setM1InDesc) != GRAPH_SUCCESS ||
                    setM1NewNode.UpdateOutputDesc(OUTPUT_INDEX, setM1OutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update SetM1Dequant tensor desc failed."), return false);

    auto* scaleInputNode = iScale.GetProducer();
    OP_CHECK_IF(scaleInputNode == nullptr, OP_LOGE(GetNodeType().GetString(), "Get scale input node failed."),
                return false);
    OP_CHECK_IF(
        innerGraph->AddDataEdge(*scaleInputNode, OUTPUT_INDEX, setM1NewNode, SET_M1_INPUT_INDEX) != GRAPH_SUCCESS,
        OP_LOGE(GetNodeType().GetString(), "Add edge from scale input to SetM1Dequant failed."), return false);

    auto* setM1OutHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(setM1NewNode, OUTPUT_INDEX);
    OP_CHECK_IF(setM1OutHolder == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get SetM1Dequant output tensor holder failed."), return false);
    scaleOutput = EsTensorHolder(setM1OutHolder);
    return true;
}

std::unique_ptr<SubgraphBoundary> Conv2DTransposeFixpipeFusionPass::ConstructBoundary(const GNode& convTransposeNode)
{
    auto boundary = std::make_unique<SubgraphBoundary>();
    // add conv2dtranspose required inputs: input_size, x, filter
    for (int64_t index = 0; index < static_cast<int64_t>(CONV2D_TRANSPOSE_INPUT_NUM); ++index) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPassAlias::AddSubgraphInput(boundary, convTransposeNode, index, index),
                                return nullptr);
    }

    // add optional bias input
    if (hasBias) {
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPassAlias::AddSubgraphInput(boundary, convTransposeNode, BIAS_INDEX, BIAS_INDEX),
            return nullptr);
    }

    // add scale input, which comes from PostCube quant_scale_0
    if (hasScale) {
        // quant_scale_0 is produced by a SetM1Dequant host node which has no registered op_proto,
        // it gets pruned during subgraph replace and the scale would become zero, so the scale
        // boundary must be bound to the source of SetM1Dequant input(0), i.e. the dequant param
        // const; SetM1Dequant itself is rebuilt inside the replacement graph
        auto setM1Node = GetScaleSetM1Node(*postCubeNode);
        if (setM1Node != nullptr) {
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPassAlias::AddSubgraphInput(
                                        boundary, *setM1Node, SET_M1_INPUT_INDEX, GetScaleGraphInputIndex()),
                                    return nullptr);
        } else {
            FUSION_PASS_CHECK_NOLOG(
                !ConvFusionUtilsPassAlias::AddSubgraphInput(
                    boundary, *postCubeNode, POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, GetScaleGraphInputIndex()),
                return nullptr);
        }
    }

    // add the scale1 input of the dual output, which comes from the back FixPipe branch
    // quant_scale_0 (migrated from the updated fixpipe pass)
    if (outputCase != Conv2DTransposeFixpipeFusion::OutputCase::SINGLE && hasScale1) {
        auto setM1Node1 = GetScaleSetM1Node(*allPostCubeNodes.back());
        if (setM1Node1 != nullptr) {
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPassAlias::AddSubgraphInput(
                                        boundary, *setM1Node1, SET_M1_INPUT_INDEX, GetScale1GraphInputIndex()),
                                    return nullptr);
        } else {
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPassAlias::AddSubgraphInput(boundary, *allPostCubeNodes.back(),
                                                                                POST_CUBE_INPUT_QUANT_SCALE_0_INDEX,
                                                                                GetScale1GraphInputIndex()),
                                    return nullptr);
        }
    }

    // add graph output: the removed Transpose output when the transpose fusion happens,
    // postcube output when postcube fusion success, otherwise conv2dtranspose output
    if (hasFusedTranspose) {
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPassAlias::AddSubgraphOutput(boundary, *transposeNode, OUTPUT_INDEX, OUTPUT_INDEX),
            return nullptr);
    } else if (hasPostCube && outputCase != Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE) {
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPassAlias::AddSubgraphOutput(boundary, *postCubeNode, OUTPUT_INDEX, OUTPUT_INDEX),
            return nullptr);
    } else {
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPassAlias::AddSubgraphOutput(boundary, convTransposeNode, OUTPUT_INDEX, OUTPUT_INDEX),
            return nullptr);
    }

    // add the y1 graph output of the dual output: the back FixPipe branch output, the mixed
    // FIXPIPE_CONV2DT case takes the conv2dtranspose original output for the passthrough path
    // (migrated from the updated fixpipe pass); the boundary output index is continuous from
    // 0, the y1 path takes index 1 regardless of the node input slot of scale1
    if (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::DUAL_FIXPIPE ||
        outputCase == Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE) {
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPassAlias::AddSubgraphOutput(boundary, *allPostCubeNodes.back(), OUTPUT_INDEX, 1),
            return nullptr);
    } else if (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::FIXPIPE_CONV2DT) {
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPassAlias::AddSubgraphOutput(boundary, convTransposeNode, OUTPUT_INDEX, 1), return nullptr);
    }

    return boundary;
}

GraphUniqPtr Conv2DTransposeFixpipeFusionPass::Replacement(const GNode& convTransposeNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");

    std::string nodeNamePrefix;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::GetNodeName(convTransposeNode, nodeNamePrefix),
                OP_LOGE(GetNodeType().GetString(), "Get node name failed."), return nullptr);

    auto builder = EsGraphBuilder("replacement");
    auto [iInputSize, iX, iFilter] = builder.CreateInputs<CONV2D_TRANSPOSE_INPUT_NUM>();

    // input side: unsqueeze x and filter to expand 2D input to 3D
    UnsqueezeNodeInfo unsqueezeXInfo;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
                    builder, iX, input1Desc, nodeNamePrefix + "_Unsqueeze_0", unsqueezeXInfo, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create unsqueeze x node failed"), return nullptr);

    // input side: unsqueeze x and filter to expand 2D input to 3D. A fractal filter (FRACTAL_Z/
    // FRACTAL_Z_G) keeps its layout directly instead: the fractal layout of the D=1 3D filter is
    // identical to the 2D one, so only the origin format is forced to NCHW and the filter input
    // is connected to ExtendConvTranspose without the Unsqueeze node (migrated from the original
    // fixpipe pass)
    bool isFractalFilter = input2Desc.GetFormat() == ge::FORMAT_FRACTAL_Z ||
                           input2Desc.GetFormat() == ge::FORMAT_FRACTAL_Z_G;
    UnsqueezeNodeInfo unsqueezeFilterInfo;
    TensorDesc filterInputDesc;
    if (isFractalFilter) {
        filterInputDesc = input2Desc;
        filterInputDesc.SetOriginFormat(ge::FORMAT_NCHW);
    } else {
        OP_CHECK_IF(
            !ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
                builder, iFilter, input2Desc, nodeNamePrefix + "_Unsqueeze_1", unsqueezeFilterInfo, GetNodeType()),
            OP_LOGE(GetNodeType().GetString(), "Create unsqueeze filter node failed"), return nullptr);
        filterInputDesc = unsqueezeFilterInfo.outDesc;
    }
    EsTensorHolder filterInput = isFractalFilter ? iFilter :
                                                   EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(
                                                       unsqueezeFilterInfo.node, OUTPUT_INDEX));

    // optional inputs: bias and scale, the index must be consistent with ConstructBoundary
    auto iBias = hasBias ? builder.CreateInput(BIAS_INDEX) : nullptr;
    auto iScale = hasScale ? builder.CreateInput(GetScaleGraphInputIndex()) : nullptr;
    bool isDualOutput = outputCase != Conv2DTransposeFixpipeFusion::OutputCase::SINGLE;
    auto iScale1 = (isDualOutput && hasScale1) ? builder.CreateInput(GetScale1GraphInputIndex()) : nullptr;

    // rebuild SetM1Dequant inside replacement graph: the original one (source of postcube
    // quant_scale_0) has no registered op_proto and is pruned during subgraph replace, so the
    // M1 scale must be regenerated here from the dequant param const; the second scale path of
    // the dual output is rebuilt the same way from the back FixPipe branch
    EsTensorHolder scaleOutput;
    auto setM1Node = hasScale ? GetScaleSetM1Node(*postCubeNode) : nullptr;
    if (setM1Node != nullptr) {
        OP_CHECK_IF(!BuildSetM1DequantNode(builder, iScale, *setM1Node, nodeNamePrefix + "_SetM1Dequant", requantOffset,
                                           outputDesc, scaleOutput),
                    OP_LOGE(GetNodeType().GetString(), "Build SetM1Dequant node failed."), return nullptr);
    }
    EsTensorHolder scaleOutput1;
    auto setM1Node1 = (isDualOutput && hasScale1) ? GetScaleSetM1Node(*allPostCubeNodes.back()) : nullptr;
    if (setM1Node1 != nullptr) {
        OP_CHECK_IF(!BuildSetM1DequantNode(builder, iScale1, *setM1Node1, nodeNamePrefix + "_SetM1Dequant1",
                                           requantOffset1, outputDesc1, scaleOutput1),
                    OP_LOGE(GetNodeType().GetString(), "Build SetM1Dequant node for scale1 failed."), return nullptr);
    }

    auto unsqueezeXOutput = EsTensorHolder(
        builder.GetCGraphBuilder()->GetTensorHolderFromNode(unsqueezeXInfo.node, OUTPUT_INDEX));
    // A16W8 weight trans (fp16 x + int8 filter + groups == 1): the int8 weight is rearranged by
    // the RearrangeA16W8WeightFzOp host node before entering ExtendConvTranspose, the host op has
    // no registered op_proto so it is rebuilt inside the replacement graph; the int8 x scenario
    // and the multi-group A16W8 scenario keep the filter input directly
    EsTensorHolder rearrangeWeightOutput;
    if (hasRearrangeWeight) {
        OP_CHECK_IF(
            !CreateRearrangeWeightNode(builder, filterInput, filterInputDesc, nodeNamePrefix, rearrangeWeightOutput),
            OP_LOGE(GetNodeType().GetString(), "Create rearrange weight node failed."), return nullptr);
    }
    auto filterArg = hasRearrangeWeight ? rearrangeWeightOutput : filterInput;

    // The removed Transpose supplies the final 4D shape and NHWC format. Its original dtype
    // belongs to the graph boundary; the ExtendConvTranspose kernel must use the normalized
    // dtype from outputDesc (FP16 for an FP32 Conv2DTranspose output).
    TensorDesc output2DTensor = hasFusedTranspose ? transposeOutputDesc : outputDesc;
    TensorDesc extendOutput2DTensor = output2DTensor;
    if (hasFusedTranspose) {
        extendOutput2DTensor.SetDataType(outputDesc.GetDataType());
    }
    TensorDesc output3DTensor;
    ConvBackpropFusionUtilsPass::ExpandOutputDesc(extendOutput2DTensor, output3DTensor);
    TensorDesc output3DTensor1;
    if (isDualOutput) {
        ConvBackpropFusionUtilsPass::ExpandOutputDesc(outputDesc1, output3DTensor1);
    }

    // create ExtendConvTranspose (the dual output cases create ExtendConvTransposeV2 which
    // additionally contains the scale1 input and the y1 output, migrated from the updated
    // fixpipe pass). The node is built by CompliantNodeBuilder (same as the SetM1Dequant and
    // rearrange weight nodes above) instead of the generated es api: the generated
    // es::ExtendConvTranspose declaration only exists in es sdk versions that already contain
    // this op, the compliant builder only needs the op type string and keeps the pass
    // compilable on older environments
    auto* innerGraph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(innerGraph == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get inner graph of replacement builder failed."), return nullptr);

    std::string extendConvTransposeName = nodeNamePrefix + "_to_ExtendConvTranspose";
    std::vector<ge::es::CompliantNodeBuilder::IrInputDef> extendInputDefs = {
        {"input_size", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"filter", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        {isDualOutput ? "scale0" : "scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}};
    if (isDualOutput) {
        extendInputDefs.push_back({"scale1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""});
    }
    std::vector<ge::es::CompliantNodeBuilder::IrOutputDef> extendOutputDefs = {
        {isDualOutput ? "y0" : "y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    if (isDualOutput) {
        extendOutputDefs.push_back({"y1", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""});
    }
    ge::GNode extendConvTransposeNode = ge::es::CompliantNodeBuilder(innerGraph)
                                            .OpType(isDualOutput ? "ExtendConvTransposeV2" : "ExtendConvTranspose")
                                            .Name(extendConvTransposeName.c_str())
                                            .IrDefInputs(extendInputDefs)
                                            .IrDefOutputs(extendOutputDefs)
                                            .Build();

    // connect the producers: input_size, unsqueezed x, filter (unsqueezed or rearranged)
    auto* inputSizeProducer = iInputSize.GetProducer();
    OP_CHECK_IF(inputSizeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Get input_size producer failed."),
                return nullptr);
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*innerGraph, *inputSizeProducer, iInputSize.GetProducerOutIndex(),
                                                 extendConvTransposeNode, INPUT_SIZE_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for input_size failed."), return nullptr);

    auto* xProducer = unsqueezeXOutput.GetProducer();
    OP_CHECK_IF(xProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Get x producer failed."), return nullptr);
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*innerGraph, *xProducer, unsqueezeXOutput.GetProducerOutIndex(),
                                                 extendConvTransposeNode, X_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for x failed."), return nullptr);

    auto* filterProducer = filterArg.GetProducer();
    OP_CHECK_IF(filterProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Get filter producer failed."),
                return nullptr);
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*innerGraph, *filterProducer, filterArg.GetProducerOutIndex(),
                                                 extendConvTransposeNode, FILTER_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for filter failed."), return nullptr);

    // optional bias and scale producers: scale comes from the rebuilt SetM1Dequant or the boundary input
    if (hasBias) {
        auto* biasProducer = iBias.GetProducer();
        OP_CHECK_IF(biasProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Get bias producer failed."),
                    return nullptr);
        OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*innerGraph, *biasProducer, iBias.GetProducerOutIndex(),
                                                     extendConvTransposeNode, BIAS_INDEX) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Add edge for bias failed."), return nullptr);
    }
    if (hasScale) {
        auto scaleArg = setM1Node != nullptr ? scaleOutput : iScale;
        auto* scaleProducer = scaleArg.GetProducer();
        OP_CHECK_IF(scaleProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Get scale producer failed."),
                    return nullptr);
        OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*innerGraph, *scaleProducer, scaleArg.GetProducerOutIndex(),
                                                     extendConvTransposeNode, SCALE_INDEX) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Add edge for scale failed."), return nullptr);
    }
    // the second scale of the dual output comes from the rebuilt SetM1Dequant1 or the boundary
    // input, its graph input index keeps continuous with the scale index when bias is absent
    // (the node input slot is always SCALE1_INDEX no matter whether bias is present)
    if (isDualOutput && hasScale1) {
        auto scaleArg1 = setM1Node1 != nullptr ? scaleOutput1 : iScale1;
        auto* scaleProducer1 = scaleArg1.GetProducer();
        OP_CHECK_IF(scaleProducer1 == nullptr, OP_LOGE(GetNodeType().GetString(), "Get scale1 producer failed."),
                    return nullptr);
        OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*innerGraph, *scaleProducer1, scaleArg1.GetProducerOutIndex(),
                                                     extendConvTransposeNode, SCALE1_INDEX) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Add edge for scale1 failed."), return nullptr);
    }

    // the same attr set as the generated es api: the 3D attrs expanded by GetNodeAttrs + fusion attrs
    AscendString dataFormat = convBpAttr.dataFormat.c_str();
    int64_t yQuantMode = DEFAULT_Y_QUANT_MODE;
    // the fixed shift value is propagated from the conv2dtranspose node, or derived from the
    // original filter dtype when it is absent/zero (migrated from the original fixpipe pass)
    int64_t fixedShiftVal = 0;
    if (convTransposeNode.GetAttr("fixed_shift_value", fixedShiftVal) != GRAPH_SUCCESS || fixedShiftVal == 0) {
        TensorDesc originFilterDesc;
        OP_CHECK_IF(convTransposeNode.GetInputDesc(FILTER_INDEX, originFilterDesc) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Get original filter desc failed."), return nullptr);
        DataType originFilterDtype = originFilterDesc.GetDataType();
        fixedShiftVal = (originFilterDtype == ge::DT_FLOAT16 || originFilterDtype == ge::DT_FLOAT) ?
                            FIXED_SHIFT_VALUE_FP16 :
                            FIXED_SHIFT_VALUE_INT8;
    }
    // the fp32 bias keeps its dtype on the bias input of ExtendConvTranspose (migrated from the
    // original fixpipe pass)
    int64_t keepBiasDtype = (hasBias && biasDesc.GetDataType() == ge::DT_FLOAT) ? 1 : 0;
    // the dual output flag of ExtendConvTransposeV2 (migrated from the updated fixpipe pass)
    bool dualOutputAttr = true;
    OP_CHECK_IF(extendConvTransposeNode.SetAttr("strides", convBpAttr.strides) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("pads", convBpAttr.pads) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("dilations", convBpAttr.dilations) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("groups", convBpAttr.groups) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("data_format", dataFormat) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("output_padding", outputPadding) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("offset_x", offsetX) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("fusion_mode", fusionMode) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("y_quant_mode", yQuantMode) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("fixed_shift_value", fixedShiftVal) != GRAPH_SUCCESS ||
                    extendConvTransposeNode.SetAttr("_keep_dtype", keepBiasDtype) != GRAPH_SUCCESS ||
                    (isDualOutput && extendConvTransposeNode.SetAttr("dual_output", dualOutputAttr) != GRAPH_SUCCESS),
                OP_LOGE(GetNodeType().GetString(), "Set attrs for ExtendConvTranspose failed."), return nullptr);

    OP_CHECK_IF(!UpdateExtendConvTransposeDesc(extendConvTransposeNode, unsqueezeXInfo.outDesc, filterInputDesc,
                                               output3DTensor, output3DTensor1),
                OP_LOGE(GetNodeType().GetString(), "Update ExtendConvTranspose desc failed."), return nullptr);

    // Squeeze only changes rank. When the removed Transpose exposed FP32 but the Extend
    // kernel produces FP16, preserve the output contract with an explicit Cast afterwards.
    TensorDesc squeeze2DTensor = output2DTensor;
    if (hasFusedTranspose) {
        squeeze2DTensor.SetDataType(outputDesc.GetDataType());
    }
    GNode squeezeNode;
    OP_CHECK_IF(
        !ConvBackpropFusionUtilsPass::BuildSqueezeNode(builder, extendConvTransposeNode, output3DTensor,
                                                       squeeze2DTensor, nodeNamePrefix, squeezeNode, GetNodeType()),
        OP_LOGE(GetNodeType().GetString(), "Build Squeeze node failed"), return nullptr);

    auto squeezeOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(squeezeNode, OUTPUT_INDEX));
    EsTensorHolder y0Output = squeezeOutput;
    if (squeeze2DTensor.GetDataType() != output2DTensor.GetDataType()) {
        auto* castGraph = builder.GetCGraphBuilder()->GetGraph();
        OP_CHECK_IF(castGraph == nullptr, OP_LOGE(GetNodeType().GetString(), "Get graph for Cast failed."),
                    return nullptr);
        GNode castNode = ge::es::CompliantNodeBuilder(castGraph)
                             .OpType("Cast")
                             .Name((nodeNamePrefix + "_Cast_0").c_str())
                             .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
        int64_t castDstType = static_cast<int64_t>(output2DTensor.GetDataType());
        OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*castGraph, squeezeNode, OUTPUT_INDEX, castNode, OUTPUT_INDEX) !=
                            GRAPH_SUCCESS ||
                        castNode.UpdateInputDesc(OUTPUT_INDEX, squeeze2DTensor) != GRAPH_SUCCESS ||
                        castNode.UpdateOutputDesc(OUTPUT_INDEX, output2DTensor) != GRAPH_SUCCESS ||
                        castNode.SetAttr("dst_type", castDstType) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Build output Cast failed."), return nullptr);
        y0Output = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(castNode, OUTPUT_INDEX));
    }
    std::vector<EsTensorHolder> outputs = {y0Output};

    // the y1 path of the dual output squeezes the 3D output back to its own 2D desc, the node
    // is built manually because the util builder hardcodes the _Squeeze_0 name suffix
    // (migrated from the updated fixpipe pass)
    if (isDualOutput) {
        auto* squeezeGraph = builder.GetCGraphBuilder()->GetGraph();
        OP_CHECK_IF(squeezeGraph == nullptr,
                    OP_LOGE(GetNodeType().GetString(), "Get inner graph of replacement builder failed."),
                    return nullptr);
        int32_t squeeze1Axis = ConvBackpropFusionUtilsPass::GetExpandAxis(outputDesc1.GetFormat());
        GNode squeezeNode1 = ge::es::CompliantNodeBuilder(squeezeGraph)
                                 .OpType("Squeeze")
                                 .Name((nodeNamePrefix + "_Squeeze_1").c_str())
                                 .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .Build();
        OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*squeezeGraph, extendConvTransposeNode, 1, squeezeNode1,
                                                     OUTPUT_INDEX) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Add edge for Squeeze1 failed."), return nullptr);
        OP_CHECK_IF(squeezeNode1.UpdateInputDesc(OUTPUT_INDEX, output3DTensor1) != GRAPH_SUCCESS ||
                        squeezeNode1.UpdateOutputDesc(OUTPUT_INDEX, outputDesc1) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Update Squeeze1 tensor desc failed."), return nullptr);
        std::vector<int64_t> squeeze1Axes = {squeeze1Axis};
        OP_CHECK_IF(squeezeNode1.SetAttr("axis", squeeze1Axes) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Set axis for Squeeze1 failed."), return nullptr);
        outputs.push_back(
            EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(squeezeNode1, OUTPUT_INDEX)));
    }

    OP_LOGD(GetNodeType().GetString(), "Conv2DTranspose trans to ExtendConvTranspose fusion success!");
    return builder.BuildAndReset(outputs);
}

Status Conv2DTransposeFixpipeFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OP_LOGD(GetNodeType().GetString(), "Begin to do Conv2DTransposeFixpipeFusionPass.");

    std::vector<GNode> matchedNodes;
    OP_CHECK_IF(!ConvFusionUtilsPassAlias::GetMatchedNodes(graph, matchedNodes, {CONV2D_TRANSPOSE}),
                OP_LOGE(GetNodeType().GetString(), "Get matched nodes failed."), return FAILED);
    if (matchedNodes.empty()) {
        OP_LOGD(GetNodeType().GetString(), "No matched node, exit.");
        return GRAPH_NOT_CHANGED;
    }

    int32_t effectTimes = 0;
    for (auto& node : matchedNodes) {
        InitMember();

        if (!MeetRequirements(node)) {
            OP_LOGD(GetNodeType().GetString(), "MeetRequirements check failed, skip.");
            continue;
        }

        OP_CHECK_IF(!GetNodeDesc(node), OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed."), return FAILED);
        OP_CHECK_IF(!GetNodeAttrs(node), OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed."), return FAILED);

        // mark the multiple output ability on the conv2dtranspose node before the postcube
        // fusion so that PostCubeUtils creates one FixPipe branch per output path (migrated
        // from the updated fixpipe pass, the cube vector fuse soc provides the ability)
        if (IsCubeVectorFusePlatform()) {
            PostCubeComm::SetPostCubeAbilityAttr(std::make_shared<GNode>(node),
                                                 PostCubeAbilityType::SupportMultipleOutput);
        }

        // A16W8 weight trans scenario flag, the RearrangeA16W8WeightFzOp host node is rebuilt in
        // Replacement when the flag is set
        hasRearrangeWeight = NeedRearrangeWeightNode();

        // postcube fusion for the post nodes of conv2dtranspose, the graph may have been changed by
        // PostCubeUtils, so the conversion must be skipped when postcube fusion is not supported
        auto postCubeResult = PostCubeFusionImpl(graph, node, passContext);
        if (postCubeResult == PostCubeFusionResult::kSkipNode) {
            OP_LOGW(GetNodeType().GetString(), "PostCube fusion not supported, skip this node.");
            continue;
        }

        hasPostCube = postCubeResult == PostCubeFusionResult::kFused;
        if (hasPostCube) {
            hasScale = outputCase != Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE && HasScaleNode();
            // fusion_mode bits (migrated from the updated fixpipe pass): bit0 = relu of the
            // y0 path, bit1 = relu of the y1 path, the passthrough path never carries a relu
            bool relu0 = outputCase != Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE &&
                         IsReluEnable(allPostCubeFusionOps.front());
            bool relu1 = (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::DUAL_FIXPIPE ||
                          outputCase == Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE) &&
                         allPostCubeNodes.size() > 1 && IsReluEnable(allPostCubeFusionOps.back());
            fusionMode = (relu1 ? RELU1_FUSION_MODE : DEFAULT_FUSION_MODE) |
                         (relu0 ? RELU_FUSION_MODE : DEFAULT_FUSION_MODE);
            // the y0 output desc comes from the front FixPipe branch, the CONV2DT_FIXPIPE case
            // takes the conv2dtranspose original output instead (the passthrough path first)
            if (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE) {
                outputDesc = convTransposeOutputDesc;
            } else {
                OP_CHECK_IF(postCubeNode->GetOutputDesc(OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
                            OP_LOGE(GetNodeType().GetString(), "Get PostCube output desc failed."), return FAILED);
            }
            OP_LOGD(GetNodeType().GetString(), "Get postcube node success!");
        } else {
            OP_LOGD(GetNodeType().GetString(), "Get no postcube node, skip postcube fusion.");
        }

        // resolve the y1 output desc of the dual output cases: the back FixPipe branch or the
        // conv2dtranspose original output of the mixed case (migrated from the updated fixpipe
        // pass)
        outputDesc1 = TensorDesc();
        if (outputCase != Conv2DTransposeFixpipeFusion::OutputCase::SINGLE) {
            if (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::FIXPIPE_CONV2DT) {
                outputDesc1 = convTransposeOutputDesc;
            } else {
                OP_CHECK_IF(allPostCubeNodes.back()->GetOutputDesc(OUTPUT_INDEX, outputDesc1) != GRAPH_SUCCESS,
                            OP_LOGE(GetNodeType().GetString(), "Get PostCube output desc failed."), return FAILED);
            }
        }

        // fp32/int32 output is converted to the fp16/int8 path of ExtendConvTranspose (migrated
        // from the original fixpipe pass)
        if (outputDesc.GetDataType() == ge::DT_FLOAT) {
            outputDesc.SetDataType(ge::DT_FLOAT16);
        } else if (outputDesc.GetDataType() == ge::DT_INT32) {
            outputDesc.SetDataType(ge::DT_INT8);
        }
        if (outputDesc1.GetDataType() == ge::DT_FLOAT) {
            outputDesc1.SetDataType(ge::DT_FLOAT16);
        } else if (outputDesc1.GetDataType() == ge::DT_INT32) {
            outputDesc1.SetDataType(ge::DT_INT8);
        }

        // detect the removable Transpose successor of the output path (migrated from the
        // original fixpipe pass): the Transpose is absorbed by the subgraph replace, the
        // ExtendConvTranspose output becomes NDHWC and the Squeeze output takes its 4D desc
        // (only the y0 path is checked, aligned with the original fixpipe pass)
        const GNode& y0OutputNode = hasPostCube ?
                                        (outputCase == Conv2DTransposeFixpipeFusion::OutputCase::CONV2DT_FIXPIPE ?
                                             node :
                                             *postCubeNode) :
                                        node;
        CheckOutputTranspose(y0OutputNode);

        // resolve the requant offset for the rebuilt SetM1Dequant before the replacement changes
        // the graph, it falls back to the symmetric int8 requant zero point when neither the
        // SetM1Dequant node nor the fused postcube node carries an offset attr
        ResolveRequantOffset();

        auto boundary = ConstructBoundary(node);
        OP_CHECK_IF(boundary == nullptr, OP_LOGE(GetNodeType().GetString(), "Construct boundary failed."),
                    return FAILED);

        auto replacement = Replacement(node);
        OP_CHECK_IF(replacement == nullptr, OP_LOGE(GetNodeType().GetString(), "Construct replacement failed."),
                    return FAILED);

#if GE_COMPILER_VERSION_NUM >= 90100000U
        OP_CHECK_IF(SubgraphRewriter::Replace(*boundary, std::move(*replacement), passContext) != SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Replace for conv2dtranspose failed."), return FAILED);
#else
        OP_CHECK_IF(SubgraphRewriter::Replace(*boundary, std::move(*replacement)) != SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Replace for conv2dtranspose failed."), return FAILED);
#endif

        effectTimes++;
        OP_LOGD(GetNodeType().GetString(), "Conv2DTranspose to ExtendConvTranspose fusion success.");
    }

    OP_LOGD(GetNodeType().GetString(), "Conv2DTransposeFixpipeFusionPass completed.");
    return effectTimes != 0 ? SUCCESS : GRAPH_NOT_CHANGED;
}

REG_DECOMPOSE_PASS(Conv2DTransposeFixpipeFusionPass, {CONV2D_TRANSPOSE}).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
