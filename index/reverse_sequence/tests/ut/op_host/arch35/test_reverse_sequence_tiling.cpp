/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_elements_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "graph/graph.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/reverse_sequence_tiling.h"
#include "../../../../op_host/arch35/reverse_sequence_simt_tiling.h"

using namespace std;
using namespace ge;

class ReverseSequenceTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReverseSequenceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReverseSequenceTiling TearDown" << std::endl;
    }
};

template <typename T>
static string to_string(void *buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

static void ExecuteTestCase(ge::DataType data_dtype, ge::DataType seq_dtype,
                            gert::StorageShape data_shape, gert::StorageShape seq_shape,
                            int64_t batchDim, int64_t seqDim, uint64_t tilingKeyValue,
                            string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::ReverseSequenceCompileInfo compile_info;

    std::string op_type("ReverseSequence");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&data_shape, &seq_shape})
                      .OutputShapes({&data_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, data_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, seq_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, data_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"seq_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(seqDim)},
                                  {"batch_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(batchDim)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    ASSERT_EQ(tiling_key, tilingKeyValue);
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(ReverseSequenceTiling, test_tiling_ascendc_bfloat16) // ASB
{
    gert::StorageShape shape1 = {{2, 3, 2, 2}, {2, 3, 2, 2}};
    gert::StorageShape shape2 = {{2}, {2}};
    string expectTilingData = "2 3 2 4 24 28608 1 1 24 1 24 ";
    uint64_t tilingKeyValue = 2;
    int64_t batchDim = 2;
    int64_t seqDim = 1;

    ExecuteTestCase(ge::DT_BF16, ge::DT_INT32, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}

TEST_F(ReverseSequenceTiling, test_tiling_ascendc_float16) // BS
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{3}, {3}};
    string expectTilingData = "3 5 5 1 15 28608 1 1 15 1 15 ";
    uint64_t tilingKeyValue = 2;
    int64_t batchDim = 0;
    int64_t seqDim = 1;

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_INT64, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}

TEST_F(ReverseSequenceTiling, test_tiling_ascendc_int16) // SB
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{5}, {5}};
    string expectTilingData = "5 3 1 5 15 28608 1 1 15 1 15 ";
    uint64_t tilingKeyValue = 2;
    int64_t batchDim = 1;
    int64_t seqDim = 0;

    ExecuteTestCase(ge::DT_INT16, ge::DT_INT64, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}

TEST_F(ReverseSequenceTiling, test_tiling_ascendc_float32) // A0SA1B -> ASB
{
    gert::StorageShape shape1 = {{2, 3, 2, 2}, {2, 3, 2, 2}};
    gert::StorageShape shape2 = {{2}, {2}};
    string expectTilingData = "2 3 1 4 24 14304 1 1 24 1 24 ";
    uint64_t tilingKeyValue = 4;
    int64_t batchDim = 3;
    int64_t seqDim = 1;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_INT64, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}


TEST_F(ReverseSequenceTiling, test_tiling_ascendc_int32) // ABS
{
    gert::StorageShape shape1 = {{2, 3, 2, 2}, {2, 3, 2, 2}};
    gert::StorageShape shape2 = {{3}, {3}};
    string expectTilingData = "3 2 4 1 24 14304 1 1 24 1 24 ";
    uint64_t tilingKeyValue = 4;
    int64_t batchDim = 1;
    int64_t seqDim = 3;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}

TEST_F(ReverseSequenceTiling, test_tiling_ascendc_uint16) // SBA
{
    gert::StorageShape shape1 = {{2, 3, 2, 2}, {2, 3, 2, 2}};
    gert::StorageShape shape2 = {{3}, {3}};
    string expectTilingData = "3 2 4 12 24 28608 1 1 24 1 24 ";
    uint64_t tilingKeyValue = 2;
    int64_t batchDim = 1;
    int64_t seqDim = 0;

    ExecuteTestCase(ge::DT_UINT16, ge::DT_INT64, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}

TEST_F(ReverseSequenceTiling, test_tiling_ascendc_uint8) // BAS
{
    gert::StorageShape shape1 = {{2, 3, 2, 2}, {2, 3, 2, 2}};
    gert::StorageShape shape2 = {{2}, {2}};
    string expectTilingData = "2 2 12 1 24 57216 1 1 24 1 24 ";
    uint64_t tilingKeyValue = 0;
    int64_t batchDim = 0;
    int64_t seqDim = 3;

    ExecuteTestCase(ge::DT_UINT8, ge::DT_INT64, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}

TEST_F(ReverseSequenceTiling, test_tiling_ascendc_int8) // SAB
{
    gert::StorageShape shape1 = {{2, 3, 2, 2}, {2, 3, 2, 2}};
    gert::StorageShape shape2 = {{2}, {2}};
    string expectTilingData = "2 2 2 12 24 57216 1 1 24 1 24 ";
    uint64_t tilingKeyValue = 0;
    int64_t batchDim = 2;
    int64_t seqDim = 0;

    ExecuteTestCase(ge::DT_INT8, ge::DT_INT64, shape1, shape2, batchDim, seqDim, tilingKeyValue, expectTilingData, 0);
}