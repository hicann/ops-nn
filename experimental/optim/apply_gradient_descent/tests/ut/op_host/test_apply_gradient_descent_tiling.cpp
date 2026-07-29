/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/apply_gradient_descent_tiling.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ApplyGradientDescentTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyGradientDescentTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyGradientDescentTiling TearDown" << std::endl; }
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape storageShape;
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        shape.SetDim(i, dims[i]);
    }
    storageShape.MutableOriginShape() = shape;
    storageShape.MutableStorageShape() = shape;
    return storageShape;
}

static void RunApplyGradientDescentTilingCase(const std::vector<int64_t>& varDims,
                                              const std::vector<int64_t>& alphaDims, ge::DataType dtype,
                                              uint64_t expectTilingKey)
{
    gert::StorageShape varShape = MakeStorageShape(varDims);
    gert::StorageShape alphaShape = MakeStorageShape(alphaDims);
    gert::StorageShape deltaShape = MakeStorageShape(varDims);
    gert::StorageShape outShape = MakeStorageShape(varDims);

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                        "Intrinsic_fix_pipe_l0c2out": false,
                                                        "Intrinsic_data_move_l12ub": true,
                                                        "Intrinsic_data_move_l0c2ub": true,
                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                        "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ApplyGradientDescentCompileInfo compile_info;

    std::string op_type("ApplyGradientDescent");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&varShape, &alphaShape, &deltaShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), expectTilingKey);
    EXPECT_GT(tiling_context->GetBlockDim(), 0U);
}

TEST_F(ApplyGradientDescentTiling, apply_gradient_descent_tiling_float32)
{
    RunApplyGradientDescentTilingCase({2048, 1, 48}, {1}, ge::DT_FLOAT, 2);
}

TEST_F(ApplyGradientDescentTiling, apply_gradient_descent_tiling_float16)
{
    RunApplyGradientDescentTilingCase({1024, 256}, {1}, ge::DT_FLOAT16, 1);
}

TEST_F(ApplyGradientDescentTiling, apply_gradient_descent_tiling_bfloat16)
{
    RunApplyGradientDescentTilingCase({128, 128}, {1}, ge::DT_BF16, 3);
}

TEST_F(ApplyGradientDescentTiling, apply_gradient_descent_tiling_small_shape)
{
    RunApplyGradientDescentTilingCase({10}, {1}, ge::DT_FLOAT, 2);
}
