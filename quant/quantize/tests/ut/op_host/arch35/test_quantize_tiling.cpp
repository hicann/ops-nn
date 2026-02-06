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
 * \file test_quantize_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../../../op_host/arch35/quantize_tiling_arch35.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace std;

class QuantizeTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantizeTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantizeTiling TearDown" << std::endl;
    }
};

template <typename T>
static string to_string(void* buf, size_t size)
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

static void ExecuteTestCase(ge::DataType xDtype, ge::DataType scalesDtype, ge::DataType zeroPointsDtype,
                            ge::DataType yDtype, gert::StorageShape xShape, gert::StorageShape scalesShape,
                            gert::StorageShape zeroPointsShape, gert::StorageShape yShape, string dtype, int64_t axis,
                            string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false, 
                           "Intrinsic_data_move_l12ub": true, 
                           "Intrinsic_data_move_l0c2ub": true, 
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64}
                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    Ops::Base::BroadcastCompileInfo compile_info;

    std::string op_type("Quantize");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder =
        gert::TilingContextFaker()
            .SetOpType(op_type)
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&xShape, &scalesShape, &zeroPointsShape})
            .OutputShapes({&yShape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, scalesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, zeroPointsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<string>(dtype)}, {"axis", Ops::NN::AnyValue::CreateFrom(axis)}})
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
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_TENSOR)
{
    gert::StorageShape xShape = {{2, 2, 2, 2}, {2, 2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2, 2}, {2, 2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    string expectTilingData = "1 1 1 1 16 2 1 32 16 1 32 1 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_TENSOR1)
{
    gert::StorageShape xShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape scalesShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape zeroPointsShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape yShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};

    string dtype = "torch.qint32";
    int64_t axis = 2;

    string expectTilingData = "2 1 1 1 120 4 1 64 56 1 64 1 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_TENSOR2)
{
    gert::StorageShape xShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape scalesShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape zeroPointsShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape yShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};

    string dtype = "torch.qint32";
    int64_t axis = -1;

    string expectTilingData = "2 1 1 1 120 4 1 64 56 1 64 1 3 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_TENSOR3)
{
    gert::StorageShape xShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape scalesShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape zeroPointsShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape yShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};

    string dtype = "torch.qint32";
    int64_t axis = -5;

    string expectTilingData = "";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_CHANNEL1)
{
    gert::StorageShape xShape = {{2, 3, 4, 4096}, {2, 3, 4, 4096}};
    gert::StorageShape scalesShape = {{4096}, {4096}};
    gert::StorageShape zeroPointsShape = {{4096}, {4096}};
    gert::StorageShape yShape = {{2, 3, 4, 4096}, {2, 3, 4, 4096}};

    string dtype = "torch.qint32";
    int64_t axis = -1;

    string expectTilingData = "64 1 0 24 4096 4 1 64 64 24 64 1 3 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_CHANNEL2)
{
    gert::StorageShape xShape = {{4096, 4, 3, 2}, {4096, 4, 3, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape yShape = {{4096, 4, 3, 2}, {4096, 4, 3, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 3;

    string expectTilingData = "64 0 0 49152 2 3 1 768 768 768 2 1 3 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_HEAD1)
{
    gert::StorageShape xShape = {{4096, 2, 2, 2}, {4096, 2, 2, 2}};
    gert::StorageShape scalesShape = {{4096}, {4096}};
    gert::StorageShape zeroPointsShape = {{4096}, {4096}};
    gert::StorageShape yShape = {{4096, 2, 2, 2}, {4096, 2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    string expectTilingData = "64 1 1 1 4096 8 64 64 64 64 32 1 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_HEAD2)
{
    gert::StorageShape xShape = {{4096, 2, 2, 2}, {4096, 2, 2, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape yShape = {{4096, 2, 2, 2}, {4096, 2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 1;

    string expectTilingData = "64 0 0 4096 2 4 1 64 64 2 64 1 1 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_PER_HEAD3)
{
    gert::StorageShape xShape = {{4096, 2, 2, 2}, {4096, 2, 2, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape yShape = {{4096, 2, 2, 2}, {4096, 2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = -2;

    string expectTilingData = "64 0 0 8192 2 2 1 128 128 2 64 1 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(QuantizeTiling, QuantizeTiling_X_UNSUPPORT_DTYPE)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_SCALES_UNSUPPORT_DTYPE)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_ZERO_POINTS_UNSUPPORT_DTYPE)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_Y_UNSUPPORT_DTYPE)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_BF16_NOT_SAME_CASE1)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, xShape, scalesShape, zeroPointsShape, yShape,
                    dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_BF16_NOT_SAME_CASE2)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape, yShape,
                    dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_BF16_NOT_SAME_CASE3)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, xShape, scalesShape, zeroPointsShape, yShape,
                    dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_BF16_NOT_SAME_CASE4)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, xShape, scalesShape, zeroPointsShape, yShape,
                    dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_AXIS_INVALID1)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 5;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_AXIS_INVALID2)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = -5;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_DTYPE_NOT_SUPPORT)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.float32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_INVALID_SCALES)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1, 2, 2}, {1, 2, 2}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_INVALID_SCALES2)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape zeroPointsShape = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 1;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_INVALID_ZERO_POINTS1)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1, 1, 2}, {1, 1, 2}};
    gert::StorageShape zeroPointsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_INVALID_ZERO_POINTS2)
{
    gert::StorageShape xShape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape scalesShape = {{1, 1, 2}, {1, 1, 2}};
    gert::StorageShape zeroPointsShape = {{1, 1, 3}, {1, 1, 3}};
    gert::StorageShape yShape = {{2, 2, 2}, {2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 0;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, "", ge::GRAPH_FAILED);
}

TEST_F(QuantizeTiling, QuantizeTiling_special_axis)
{
    gert::StorageShape xShape = {{2, 2, 2, 2}, {2, 2, 2, 2}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape zeroPointsShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 2, 2, 2}, {2, 2, 2, 2}};

    string dtype = "torch.qint32";
    int64_t axis = 5;

    string expectTilingData = "1 1 1 1 16 2 1 32 16 1 32 1 3 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, xShape, scalesShape, zeroPointsShape,
                    yShape, dtype, axis, expectTilingData, ge::GRAPH_SUCCESS);
}