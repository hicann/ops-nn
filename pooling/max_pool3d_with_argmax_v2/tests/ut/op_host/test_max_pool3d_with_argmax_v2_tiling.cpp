/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
struct MaxPool3DWithArgmaxV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
} // namespace optiling

class MaxPool3DWithArgmaxV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPool3DWithArgmaxV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPool3DWithArgmaxV2Tiling TearDown" << std::endl;
    }
};

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_001_float_correct_check)
{
    // //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 8, 6, 6, 6}, {1, 8, 6, 6, 6}};
    gert::StorageShape out_shape = {{1, 8, 1, 1, 1}, {1, 8, 1, 1, 1}};
    gert::StorageShape indices_shape = {{1, 8, 1, 1, 1}, {1, 8, 1, 1, 1}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {3, 3, 3};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_002_half_correct_check)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 16, 6, 6, 6}, {1, 16, 6, 6, 6}};
    gert::StorageShape out_shape = {{1, 16, 1, 1, 1}, {1, 16, 1, 1, 1}};
    gert::StorageShape indices_shape = {{1, 16, 1, 1, 1}, {1, 16, 1, 1, 1}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {2, 2, 2};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100001);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_003_dtype_check)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 8, 6, 6, 6}, {1, 8, 6, 6, 6}};
    gert::StorageShape out_shape = {{1, 8, 1, 1, 1}, {1, 8, 1, 1, 1}};
    gert::StorageShape indices_shape = {{1, 8, 1, 1, 1}, {1, 8, 1, 1, 1}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {3, 3, 3};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    auto tiling_key = tiling_context->GetTilingKey();
    // ASSERT_EQ(tiling_key, 0);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_004_format_check)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{6, 6, 6}, {6, 6, 6}};
    gert::StorageShape out_shape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape indices_shape = {{2, 2, 2}, {2, 2, 2}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {3, 3, 3};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCL)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCL)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCL)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    auto tiling_key = tiling_context->GetTilingKey();
    // ASSERT_EQ(tiling_key, 0);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_005_one_channel)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}};
    gert::StorageShape out_shape = {{1, 8, 2, 2, 2}, {1, 8, 2, 2, 2}};
    gert::StorageShape indices_shape = {{1, 8, 2, 2, 2}, {1, 8, 2, 2, 2}};

    std::vector<int64_t> kernelSize = {4, 4, 4};
    std::vector<int64_t> stride = {8, 8, 8};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110000);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_006_cut_data)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 8, 144, 144, 144}, {1, 8, 144, 144, 144}};
    gert::StorageShape out_shape = {{1, 8, 47, 47, 47}, {1, 8, 47, 47, 47}};
    gert::StorageShape indices_shape = {{1, 8, 47, 47, 47}, {1, 8, 47, 47, 47}};

    std::vector<int64_t> kernelSize = {1, 1, 1};
    std::vector<int64_t> stride = {3, 3, 3};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111000);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_007_one_channel_cut_data)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1, 144, 144, 144}, {1, 1, 144, 144, 144}};
    gert::StorageShape out_shape = {{1, 1, 17, 17, 17}, {1, 1, 17, 17, 17}};
    gert::StorageShape indices_shape = {{1, 1, 17, 17, 17}, {1, 1, 17, 17, 17}};

    std::vector<int64_t> kernelSize = {4, 4, 4};
    std::vector<int64_t> stride = {8, 8, 8};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111100);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_008_float_stride_less_than_kernel)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 8, 144, 144, 144}, {1, 8, 144, 144, 144}};
    gert::StorageShape out_shape = {{1, 8, 33, 33, 33}, {1, 8, 33, 33, 33}};
    gert::StorageShape indices_shape = {{1, 8, 33, 33, 33}, {1, 8, 33, 33, 33}};

    std::vector<int64_t> kernelSize = {4, 4, 4};
    std::vector<int64_t> stride = {4, 4, 4};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111100);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_009_large_kernel)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 8, 63, 63, 63}, {1, 8, 63, 63, 63}};
    gert::StorageShape out_shape = {{1, 8, 1, 1, 1}, {1, 8, 1, 1, 1}};
    gert::StorageShape indices_shape = {{1, 8, 1, 1, 1}, {1, 8, 1, 1, 1}};

    std::vector<int64_t> kernelSize = {32, 32, 32};
    std::vector<int64_t> stride = {32, 32, 32};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {2, 2, 2};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111110);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_010_no_expand_indices)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 100, 200, 300, 400}, {1, 100, 200, 300, 400}};
    gert::StorageShape out_shape = {{1, 100, 99, 149, 199}, {1, 100, 99, 149, 199}};
    gert::StorageShape indices_shape = {{1, 100, 99, 149, 199}, {1, 100, 99, 149, 199}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {2, 2, 2};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 300001);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_011_no_expand_indices_pad)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 100, 200, 300, 400}, {1, 100, 200, 300, 400}};
    gert::StorageShape out_shape = {{1, 100, 100, 150, 200}, {1, 100, 100, 150, 200}};
    gert::StorageShape indices_shape = {{1, 100, 100, 150, 200}, {1, 100, 100, 150, 200}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {2, 2, 2};
    std::vector<int64_t> padding = {1, 1, 1};
    std::vector<int64_t> dilation = {1, 1, 1};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 300002);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_012_no_expand_indices_ceilmode)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 100, 200, 300, 400}, {1, 100, 200, 300, 400}};
    gert::StorageShape out_shape = {{1, 100, 99, 149, 199}, {1, 100, 99, 149, 199}};
    gert::StorageShape indices_shape = {{1, 100, 99, 149, 199}, {1, 100, 99, 149, 199}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {2, 2, 2};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 300002);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_013_no_expand_indices_pad_ceilmode)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 100, 200, 300, 400}, {1, 100, 200, 300, 400}};
    gert::StorageShape out_shape = {{1, 100, 100, 150, 200}, {1, 100, 100, 150, 200}};
    gert::StorageShape indices_shape = {{1, 100, 100, 150, 200}, {1, 100, 100, 150, 200}};

    std::vector<int64_t> kernelSize = {3, 3, 3};
    std::vector<int64_t> stride = {2, 2, 2};
    std::vector<int64_t> padding = {1, 1, 1};
    std::vector<int64_t> dilation = {1, 1, 1};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 300002);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_014_large_kernel_for_fp32)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{10, 10, 64, 64, 64}, {10, 10, 64, 64, 64}};
    gert::StorageShape out_shape = {{10, 10, 1, 1, 1}, {10, 10, 1, 1, 1}};
    gert::StorageShape indices_shape = {{10, 10, 1, 1, 1}, {10, 10, 1, 1, 1}};

    std::vector<int64_t> kernelSize = {64, 64, 64};
    std::vector<int64_t> stride = {64, 64, 64};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 311110);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_015_large_kernel_for_fp16)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{10, 10, 64, 64, 64}, {10, 10, 64, 64, 64}};
    gert::StorageShape out_shape = {{10, 10, 1, 1, 1}, {10, 10, 1, 1, 1}};
    gert::StorageShape indices_shape = {{10, 10, 1, 1, 1}, {10, 10, 1, 1, 1}};

    std::vector<int64_t> kernelSize = {64, 64, 64};
    std::vector<int64_t> stride = {64, 64, 64};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 311111);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(MaxPool3DWithArgmaxV2Tiling, MaxPool3DWithArgmaxV2_tiling_016_large_kernel_for_bf16)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{5, 5, 64, 64, 64}, {5, 5, 64, 64, 64}};
    gert::StorageShape out_shape = {{5, 5, 1, 1, 1}, {5, 5, 1, 1, 1}};
    gert::StorageShape indices_shape = {{5, 5, 1, 1, 1}, {5, 5, 1, 1, 1}};

    std::vector<int64_t> kernelSize = {64, 64, 64};
    std::vector<int64_t> stride = {64, 64, 64};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1};

    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DWithArgmaxV2CompileInfo compile_info;

    std::string op_type("MaxPool3DWithArgmaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DWithArgmaxV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&out_shape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"kernelSize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(kernelSize)},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stride)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padding)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceilMode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 311112);
    //dlog_setlevel(static_cast<int>(OP), 0, 1);
}