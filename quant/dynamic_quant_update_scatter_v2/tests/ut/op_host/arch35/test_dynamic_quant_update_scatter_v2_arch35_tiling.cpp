/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/dynamic_quant_update_scatter_v2_tiling_data.h"

namespace optiling {
struct DynamicQuantUpdateScatterV2CompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling

namespace {
constexpr uint64_t UB_SIZE = 245760;
constexpr uint64_t TILING_KEY_REGBASE = 0;

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = UINT64_MAX;
    uint32_t blockDim = 0;
    DynamicQuantUpdateScatterV2RegbaseTilingData data{};
};

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableStorageShape().AppendDim(dim);
        shape.MutableOriginShape().AppendDim(dim);
    }
    return shape;
}

TilingResult RunTiling(const std::vector<int64_t>& xDims, const std::vector<int64_t>& indicesDims,
                       const std::vector<int64_t>& varDims, const std::vector<int64_t>& scaleDims,
                       const std::vector<int64_t>& outDims, const std::vector<int64_t>& paramOutDims,
                       ge::DataType xDtype)
{
    std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": )" + std::to_string(UB_SIZE) +
                                    R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}})";
    std::map<std::string, std::string> socInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::DynamicQuantUpdateScatterV2CompileInfo compileInfo;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatterV2");
    if (opImpl == nullptr || opImpl->tiling_parse == nullptr || opImpl->tiling == nullptr) {
        return {};
    }

    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs(
                               {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                           .Outputs({&compileInfo})
                           .Build();
    auto parseContext = parseHolder.GetContext<gert::TilingParseContext>();
    parseContext->GetPlatformInfo()->Init();
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (opImpl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return {};
    }

    auto xShape = MakeShape(xDims);
    auto indicesShape = MakeShape(indicesDims);
    auto varShape = MakeShape(varDims);
    auto scaleShape = MakeShape(scaleDims);
    auto offsetShape = MakeShape(scaleDims);
    auto outShape = MakeShape(outDims);
    auto paramOutShape = MakeShape(paramOutDims);
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || workspace == nullptr) {
        return {};
    }

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &indicesShape, &varShape, &scaleShape, &offsetShape})
                      .OutputShapes({&outShape, &paramOutShape, &paramOutShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    TilingResult result;
    result.status = opImpl->tiling(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.key = context->GetTilingKey();
        result.blockDim = context->GetBlockDim();
        result.data = *reinterpret_cast<const DynamicQuantUpdateScatterV2RegbaseTilingData*>(
            context->GetRawTilingData()->GetData());
    }
    return result;
}
} // namespace

TEST(DynamicQuantUpdateScatterV2TilingArch35, Bf16Rank3)
{
    auto result = RunTiling({192, 1, 128}, {192}, {192, 1075, 1, 128}, {192, 1075}, {192, 1075, 128}, {1, 192, 1075},
                            ge::DT_BF16);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, TILING_KEY_REGBASE);
    EXPECT_EQ(result.blockDim, 64);
    EXPECT_EQ(result.data.rowLen, 128);
    EXPECT_EQ(result.data.rowPerHeadCore, 3);
    EXPECT_EQ(result.data.rowPerTailCore, 3);
    EXPECT_EQ(result.data.batchSize, 192);
    EXPECT_EQ(result.data.dstSeqLen, 1075);
    EXPECT_EQ(result.data.outAlignLen, 64);
    EXPECT_EQ(result.data.varByteLen, 13209600);
}

TEST(DynamicQuantUpdateScatterV2TilingArch35, RejectFp16Rank2)
{
    auto result = RunTiling({8, 128}, {8}, {8, 4, 128}, {8, 4}, {8, 128}, {1, 8}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterV2TilingArch35, RejectOddRowLen)
{
    auto result = RunTiling({8, 1, 127}, {8}, {8, 4, 1, 127}, {8, 4}, {8, 4, 127}, {1, 8, 4}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterV2TilingArch35, RejectIndicesBatchMismatch)
{
    auto result = RunTiling({8, 1, 128}, {7}, {8, 4, 1, 128}, {8, 4}, {8, 4, 128}, {1, 8, 4}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterV2TilingArch35, RejectUnexpectedXMiddleDim)
{
    auto result = RunTiling({8, 2, 128}, {8}, {8, 4, 1, 128}, {8, 4}, {8, 4, 128}, {1, 8, 4}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterV2TilingArch35, RejectVarShapeMismatch)
{
    auto result = RunTiling({8, 1, 128}, {8}, {8, 4, 2, 128}, {8, 4}, {8, 4, 128}, {1, 8, 4}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterV2TilingArch35, RejectParamOutputShapeMismatch)
{
    auto result = RunTiling({8, 1, 128}, {8}, {8, 4, 1, 128}, {8, 4}, {8, 4, 128}, {8, 4}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterV2TilingArch35, RejectUbOverflow)
{
    auto result = RunTiling({8, 1, 100000}, {8}, {8, 4, 1, 100000}, {8, 4}, {8, 4, 100000}, {1, 8, 4}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
