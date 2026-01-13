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
 * \file test_grouped_dynamic_mx_quant_tiling.cpp
 * \brief
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/grouped_dynamic_mx_quant_tiling_arch35.h"
#include "any_value.h"


using namespace std;

class GroupedDynamicMxQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "GroupedDynamicMxQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "GroupedDynamicMxQuantTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tilingData)
{
    auto data = tilingData->GetData();
    string result;
    for (size_t i = 0; i < tilingData->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tilingData->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void ExecuteTestCase(ge::DataType inDtype, ge::DataType outDtype, ge::DataType in2Dtype, ge::DataType out2Dtype, gert::StorageShape shape, gert::StorageShape groupIdxShape,
    gert::StorageShape outShape, gert::StorageShape scaleShape, int64_t blockSize, string expectTilingData,
    ge::graphStatus status = ge::GRAPH_SUCCESS) {
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::GroupedDynamicMxQuantCompileInfo compile_info;

    std::string op_type("GroupedDynamicMxQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                        .KernelIONum(2, 2)
                        .Inputs({const_cast<char *>("{}"), reinterpret_cast<void *>(&platform_info)})
                        .Outputs({&compile_info})
                        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                    .NodeIoNum(2, 2)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&shape, &groupIdxShape})
                    .OutputShapes({&outShape, &scaleShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, in2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, out2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom((int64_t)outDtype)},
                                {"blocksize", Ops::NN::AnyValue::CreateFrom(blockSize)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "64 1 1 1 1 128 128 128 32 32 128 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_float16_fp8e4m3fn) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "64 1 1 1 1 128 128 128 32 32 128 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "64 1 1 1 1 128 128 128 32 32 128 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_float16_fp8e5m2) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "64 1 1 1 1 128 128 128 32 32 128 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_inDtype) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_inDtype2) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT16, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_outDtype) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_outDtype2) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT16, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_blockSize) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 64;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_scale_shape) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 3}, {1, 128, 3}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}


TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_input0_dim) {
    gert::StorageShape shape = {{1, 32, 128}, {1, 32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_input1_dim) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1, 1}, {1, 1}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_output0_dim) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape outShape = {{1, 32, 128}, {1, 32, 128}};
    gert::StorageShape scaleShape = {{1, 128, 2}, {1, 128, 2}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, outShape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantTiling, GroupedDynamicMxQuant_tiling_ascendc_error_output1_dim) {
    gert::StorageShape shape = {{32, 128}, {32, 128}};
    gert::StorageShape groupIdxShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 256}, {1, 256}};
    int64_t blockSize = 32;
    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_INT32, ge::DT_FLOAT8_E8M0, shape, groupIdxShape, shape, scaleShape, blockSize, expectTilingData,
        ge::GRAPH_FAILED);
}