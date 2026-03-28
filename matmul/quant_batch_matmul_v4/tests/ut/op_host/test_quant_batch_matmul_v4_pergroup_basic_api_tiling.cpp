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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ut_op_util.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_v4_compile_info.h"

using namespace ge;
using namespace ut_util;

namespace {
struct PergroupBasicApiShapeGuardCase {
    std::string caseName;
    std::string socVersion;
    std::vector<int64_t> x1Dims;
    std::vector<int64_t> x2Dims;
    ge::graphStatus expectedResult;
    uint64_t expectedTilingKey;
    uint32_t expectedNumBlocks;
    bool checkTilingMeta;
};

struct PergroupBasicApiCaseResult {
    ge::graphStatus result = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint64_t numBlocks = 0;
};

class TestQuantBatchMatmulV4PergroupBasicApiTiling
    : public testing::TestWithParam<PergroupBasicApiShapeGuardCase> {};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    auto &storageShape = shape.MutableStorageShape();
    for (const auto dim : dims) {
        storageShape.AppendDim(dim);
    }
    shape.MutableOriginShape() = shape.MutableStorageShape();
    return shape;
}

static PergroupBasicApiCaseResult RunPergroupBasicApiCase(const PergroupBasicApiShapeGuardCase &param)
{
    PergroupBasicApiCaseResult caseResult;
    const int64_t m = param.x1Dims[0];
    const int64_t k = param.x1Dims[1];
    const int64_t n = param.x2Dims[0];
    constexpr int64_t groupSize = 256;
    auto x1Shape = MakeStorageShape(param.x1Dims);
    auto x2Shape = MakeStorageShape(param.x2Dims);
    auto x1ScaleShape = MakeStorageShape({m, 1});
    auto x2ScaleShape = MakeStorageShape({k / groupSize, n});
    auto x2OffsetShape = MakeStorageShape({k / groupSize, n});
    gert::StorageShape outputShape({m, n}, {m, n});

    std::string compileInfoStr;
    if (param.socVersion == "Ascend910B") {
        compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 201326592, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
    } else {
        // Ascend950 platform setup. Keep Intrinsic_mmad as s8s4 to trigger supportMmadS8S4 branch.
        compileInfoStr = R"({
        "hardware_info" : {
            "BT_SIZE" : 4096,
            "load3d_constraints" : "unknown",
            "Intrinsic_fix_pipe_l0c2out" : true,
            "Intrinsic_data_move_l12ub" : false,
            "Intrinsic_data_move_l0c2ub" : false,
            "Intrinsic_data_move_l12bt" : true,
            "Intrinsic_data_move_out2l1_nd2nz" : true,
            "UB_SIZE" : 253952,
            "L2_SIZE" : 134217728,
            "L1_SIZE" : 524288,
            "L0A_SIZE" : 65536,
            "L0B_SIZE" : 65536,
            "L0C_SIZE" : 262144,
            "CORE_NUM" : 32,
            "cube_core_cnt": 32,
            "vector_core_cnt": 64,
            "core_type_list": "CubeCore,VectorCore",
            "socVersion" : "Ascend950",
            "NpuArch" : "3510"
        }})";
    }

    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> version;
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics, version);
    socInfos["socVersion"] = param.socVersion;
    aicoreSpec["cube_freq"] = "1800";
    if (param.socVersion == "Ascend910B") {
        version["Short_SoC_version"] = "Ascend910B";
    }

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::QuantBatchMatmulV4CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char *>(compileInfoStr.c_str()), reinterpret_cast<void *>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    const std::string opType("QuantBatchMatmulV4");
    auto *opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr) {
        return caseResult;
    }

    auto rawTilingData = gert::TilingData::CreateCap(4096);
    if (rawTilingData == nullptr) {
        return caseResult;
    }
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, nullptr, &x1ScaleShape, &x2ScaleShape, nullptr,
                                    nullptr, &x2OffsetShape, nullptr, nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)},
                                  {"compute_type", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    auto *tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return caseResult;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", version);

    if (opImpl->tiling_parse == nullptr || opImpl->tiling == nullptr) {
        return caseResult;
    }
    if (opImpl->tiling_parse(kernelHold.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return caseResult;
    }
    caseResult.result = opImpl->tiling(tilingContext);
    if (caseResult.result == ge::GRAPH_SUCCESS) {
        caseResult.tilingKey = tilingContext->GetTilingKey();
        caseResult.numBlocks = tilingContext->GetBlockDim();
    }
    return caseResult;
}

TEST_P(TestQuantBatchMatmulV4PergroupBasicApiTiling, ShapeGuard)
{
    const auto &param = GetParam();
    const auto result = RunPergroupBasicApiCase(param);
    ASSERT_EQ(result.result, param.expectedResult) << "case=" << param.caseName;
    if (param.checkTilingMeta) {
        ASSERT_EQ(result.tilingKey, param.expectedTilingKey) << "case=" << param.caseName;
        ASSERT_EQ(result.numBlocks, param.expectedNumBlocks) << "case=" << param.caseName;
    }
}

// Guard framework for Ascend950/Ascend910B:
// 1) Ascend950 keeps Intrinsic_mmad=s8s4.
// 2) Ascend910B maps Short_SoC_version=Ascend910B.
// 3) Verify by only changing x1/x2 dims below.
// 4) Add future guard cases in this table.
static PergroupBasicApiShapeGuardCase g_cases[] = {
    // Legal cases for pergroup basic api path:
    // m=128, k=1024 (k%1024==0), n=256 (n%256==0), all inputs are 2D.
    {"legal_rank2_should_succeed_950", "Ascend950", {1024, 1024}, {1024, 1024}, ge::GRAPH_SUCCESS, 537, 32, true},
    {"legal_rank2_should_succeed_910b", "Ascend910B", {128, 1024}, {256, 1024}, ge::GRAPH_SUCCESS, 1040, 1, true},
};

INSTANTIATE_TEST_CASE_P(MM, TestQuantBatchMatmulV4PergroupBasicApiTiling, testing::ValuesIn(g_cases));
} // namespace
