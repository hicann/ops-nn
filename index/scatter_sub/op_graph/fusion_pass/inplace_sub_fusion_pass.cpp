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
 * \file inplace_sub_fusion_pass.cpp
 * \brief InplaceSub --> TensorMove + ScatterSub
 *
 *   x        indices    v(updates)        x       indices  updates
 *    \         |          /                |          |       /
 *     \        |         /                 |          |      /
 *      InplaceSub             -->     TensorMove      |     /
 *            |                            |           |    /
 *            |                            |           |   /
 *            y                        ScatterSub(use_locking=false)
 *                                         |
 *                                         y
 */

#include "inplace_sub_fusion_pass.h"
#include "es_nn_ops.h"
#include "compliant_node_builder.h"
#include "common/inc/error_util.h"
#include "ge/ge_utils.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"
#include "acl/acl_rt.h"
#include <algorithm>

using namespace ge;
using namespace fusion;

namespace OPS {
namespace NN {
namespace {

const std::string PASS_NAME = "AInplaceSubFusionPass";
const int64_t CAPTURE_IDX_OUTPUT = 0l;
const std::string MATCHED_OP_TYPE = "InplaceSub";
const std::string NOT_SUPPORT_SOC = "Ascend310";

// kCompatibleInherited 出于 9.0.0(GE_COMPILER_VERSION_NUM = 90000000)。
// 本 pass 内部全部使用 8.5.0 已有 API,唯一的新接口是注册 stage kCompatibleInherited,
// 属兼容场景 D1(仅适用整体静默):9.x 运行时正常融合,8.5.0 运行时空跑。
const int32_t kGeCompilerVersion900 = 90000000;

// 查询运行时 GE compiler 版本号;查询失败按最低版本(0)处理,走兜底空跑分支
static int32_t GetRuntimeGeCompilerVersion()
{
    int32_t version = 0;
    char moduleName[] = "ge_compiler";
    aclsysGetVersionNum(moduleName, &version);
    return version;
}

// InplaceSub 输入个数(x, indices, v)
const size_t kInplaceSubInputNum = 3;

// ScatterSub var/updates 支持的 dtype,与 ScatterSub IR 定义对齐
static bool IsSupportedDataDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedDataDtypes = {DT_FLOAT16, DT_FLOAT, DT_INT32,
                                                                         DT_INT8,    DT_UINT8, DT_BF16};
    return std::find(kSupportedDataDtypes.begin(), kSupportedDataDtypes.end(), dtype) != kSupportedDataDtypes.end();
}

// ScatterSub indices 支持的 dtype(IndexNumberType: int32/int64)
static bool IsSupportedIndicesDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedIndicesDtypes = {DT_INT32, DT_INT64};
    return std::find(kSupportedIndicesDtypes.begin(), kSupportedIndicesDtypes.end(), dtype) !=
           kSupportedIndicesDtypes.end();
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

static void UpdateInputFormat(GNode& node, uint32_t idx, Format format)
{
    TensorDesc desc;
    node.GetInputDesc(idx, desc);
    desc.SetFormat(format);
    node.UpdateInputDesc(idx, desc);
}

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

// 平台校验:原融合规则只排除 Ascend310,其余平台均融合
static bool IsSupportedPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    OP_LOGE_IF(
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        false, PASS_NAME.c_str(), "Get platform_info failed.");

    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc == NOT_SUPPORT_SOC) {
        OPS_LOG_D(PASS_NAME.c_str(), "Platform %s is not supported, skip.", soc.c_str());
        return false;
    }
    return true;
}

} // anonymous namespace

// ===========================================================================
// Patterns — InplaceSub 无 ES API,使用 CompliantNodeBuilder 构建 Pattern
// ===========================================================================
std::vector<PatternUniqPtr> AInplaceSubFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for AInplaceSubFusionPass");
    std::vector<PatternUniqPtr> patterns;

    auto graphBuilder = es::EsGraphBuilder((PASS_NAME + "_" + MATCHED_OP_TYPE).c_str());
    auto x = graphBuilder.CreateInput(0);
    auto indices = graphBuilder.CreateInput(1);
    auto v = graphBuilder.CreateInput(2);

    ge::Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();

    GNode opNode = es::CompliantNodeBuilder(graphPtr)
                       .OpType(MATCHED_OP_TYPE.c_str())
                       .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"v", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();

    GNode xNode = *x.GetProducer();
    GNode indicesNode = *indices.GetProducer();
    GNode vNode = *v.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, opNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, indicesNode, 0, opNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, vNode, 0, opNode, 2);

    es::EsTensorHolder output(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(opNode, 0));

    auto graph = graphBuilder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});
    patterns.emplace_back(std::move(pattern));
    return patterns;
}

// ===========================================================================
// MeetRequirements — 校验平台 + 匹配节点 OpType
// ===========================================================================
bool AInplaceSubFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for AInplaceSubFusionPass");

    // 运行时兼容(D1 整体静默):9.x 编译包跑在 8.5.0 运行时上时,
    // kCompatibleInherited 调度点不存在,直接空跑,不做融合
    if (GetRuntimeGeCompilerVersion() < kGeCompilerVersion900) {
        OPS_LOG_D(PASS_NAME.c_str(), "Runtime ge_compiler version < 9.0.0, skip fusion (compat silent).");
        return false;
    }

    if (!IsSupportedPlatform()) {
        return false;
    }

    NodeIo captured;
    OP_LOGE_IF(matchResult->GetCapturedTensor(CAPTURE_IDX_OUTPUT, captured) != SUCCESS, false, PASS_NAME.c_str(),
               "Failed to get captured tensor.");

    AscendString nodeType;
    captured.node.GetType(nodeType);
    std::string typeStr = nodeType.GetString();
    if (typeStr != MATCHED_OP_TYPE) {
        OPS_LOG_D(PASS_NAME.c_str(), "Matched node type %s is not supported, skip.", typeStr.c_str());
        return false;
    }

    // 融合后的 ScatterSub 必须支持 var(x)/indices/updates(v) 的 dtype
    GNode sourceNode = captured.node;
    if (sourceNode.GetInputsSize() != kInplaceSubInputNum) {
        OPS_LOG_D(PASS_NAME.c_str(), "InplaceSub input num %zu is not %zu, skip.", sourceNode.GetInputsSize(),
                  kInplaceSubInputNum);
        return false;
    }

    TensorDesc xDesc;
    OP_LOGE_IF(sourceNode.GetInputDesc(0, xDesc) != SUCCESS, false, PASS_NAME.c_str(), "Get input x desc failed.");
    TensorDesc indicesDesc;
    OP_LOGE_IF(sourceNode.GetInputDesc(1, indicesDesc) != SUCCESS, false, PASS_NAME.c_str(),
               "Get input indices desc failed.");
    TensorDesc vDesc;
    OP_LOGE_IF(sourceNode.GetInputDesc(2, vDesc) != SUCCESS, false, PASS_NAME.c_str(), "Get input v desc failed.");

    if (!IsSupportedDataDtype(xDesc.GetDataType()) || !IsSupportedDataDtype(vDesc.GetDataType()) ||
        !IsSupportedIndicesDtype(indicesDesc.GetDataType())) {
        OPS_LOG_D(PASS_NAME.c_str(), "ScatterSub does not support input dtype (x:%d, indices:%d, v:%d), skip.",
                  xDesc.GetDataType(), indicesDesc.GetDataType(), vDesc.GetDataType());
        return false;
    }

    return true;
}

// ===========================================================================
// Replacement — 构建替换图:TensorMove + ScatterSub
// ===========================================================================
GraphUniqPtr AInplaceSubFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement for AInplaceSubFusionPass");

    // 1. 获取子图边界的所有输入信息
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    // 2. 创建替换图输入
    auto builder = es::EsGraphBuilder("replacement");
    auto rX = builder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], inputShapes[0].GetDims());
    auto rIndices = builder.CreateInput(1, "indices", inputDtypes[1], inputFormats[1], inputShapes[1].GetDims());
    auto rUpdates = builder.CreateInput(2, "updates", inputDtypes[2], inputFormats[2], inputShapes[2].GetDims());

    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();

    // 3. TensorMove 节点(仓内无 ES API,使用 CompliantNodeBuilder)
    GNode tensorMoveNode = es::CompliantNodeBuilder(graphPtr)
                               .OpType("TensorMove")
                               .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                               .InstanceOutputShape("y", inputShapes[0].GetDims())
                               .InstanceOutputDataType("y", inputDtypes[0])
                               .InstanceOutputFormat("y", inputFormats[0])
                               .Build();

    // 4. 连边:x -> TensorMove
    GNode xNode = *rX.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, tensorMoveNode, 0);

    // 4.1 刷新 TensorMove 节点输入的 Format(InferShape 不推导 Format)
    UpdateInputFormat(tensorMoveNode, 0, inputFormats[0]);

    // 5. TensorMove 输出转为 EsTensorHolder,用于 ScatterSub ES API
    es::EsTensorHolder tensorMoveOutput(builder.GetCGraphBuilder()->GetTensorHolderFromNode(tensorMoveNode, 0));

    // 6. ScatterSub 节点(使用 ES API),use_locking=false 与原融合规则一致
    auto scatterSubOutput = es::ScatterSub(tensorMoveOutput, rIndices, rUpdates, false);

    // 7. 刷新 ScatterSub 节点所有输入的 Format(InferShape 不推导 Format),并保留 inplace 构建选项
    GNode scatterSubNode = *scatterSubOutput.GetProducer();
    UpdateInputFormat(scatterSubNode, 0, inputFormats[0]);
    UpdateInputFormat(scatterSubNode, 1, inputFormats[1]);
    UpdateInputFormat(scatterSubNode, 2, inputFormats[2]);
    AscendString inplaceBuildOptions("{\"is_inplace\",True}");
    scatterSubNode.SetAttr("fusion_op_build_options", inplaceBuildOptions);

    // 8. 构建替换图
    GraphUniqPtr replaceGraph = builder.BuildAndReset({scatterSubOutput});
    if (replaceGraph == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildAndReset returned nullptr.");
        return nullptr;
    }

    // 9. InferShape
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_W(PASS_NAME.c_str(), "InferShape failed, continue with manual desc.");
    }

    OPS_LOG_D(PASS_NAME.c_str(), "Replacement graph built successfully.");
    return replaceGraph;
}

// ===========================================================================
// 注册 —— 版本兼容(D1:仅使用新增 Stage 枚举 kCompatibleInherited @9.0.0)
// ===========================================================================
#if GE_COMPILER_VERSION_NUM >= 90000000
namespace {
// 9.x 编译产物的注册 stage 由运行时版本决定:
//   - 9.x 运行时:注册到期望的 kCompatibleInherited,正常融合
//   - 8.5.0 运行时:kCompatibleInherited 调度点不存在,降级注册到 8.5.0 已有的
//     kBeforeInferShape 占位,并由 MeetRequirements 返回 false 空跑(整体静默)
CustomPassStage GetRegisterStage()
{
    if (GetRuntimeGeCompilerVersion() >= kGeCompilerVersion900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}
} // namespace
REG_FUSION_PASS(AInplaceSubFusionPass).Stage(GetRegisterStage());
#else
// 8.5.0 编译:kCompatibleInherited 枚举不存在,注册到 8.5.0 已有的 kAfterInferShape。
// 本 pass 逻辑全部为 8.5.0 兼容接口,8.5.0 编译 + 8.5.0 运行可正常融合。
REG_FUSION_PASS(AInplaceSubFusionPass).Stage(CustomPassStage::kAfterInferShape);
#endif
} // namespace NN
} // namespace OPS
