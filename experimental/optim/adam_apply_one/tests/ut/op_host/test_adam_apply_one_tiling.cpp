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
#include "log/log.h"
#include "../../../op_kernel/adam_apply_one_tiling_key.h"
#include "../../../op_kernel/adam_apply_one_tiling_data.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AdamApplyOneTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdamApplyOneTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdamApplyOneTiling TearDown" << std::endl; }
};

TEST_F(AdamApplyOneTiling, AdamApplyOneTiling_01)
{
    // input
    gert::StorageShape input0_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape input1_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape input2_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape input3_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape input4_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};

    // mul0_x, mul1_x, mul2_x, mul3_x
    gert::StorageShape mul0_x_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape mul1_x_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape mul2_x_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape mul3_x_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};

    // add2_y
    gert::StorageShape add2_y_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};

    // output
    gert::StorageShape output0_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape output1_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape output2_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};

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
    struct AdamApplyOneTilingCompileInfo {};
    AdamApplyOneTilingCompileInfo compile_info;

    std::string op_type("AdamApplyOne");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(10, 3)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 1024 * 1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdamApplyOne")
                      .NodeIoNum(10, 3)
                      //   .IrInstanceNum({1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape,
                                    &mul0_x_shape, &mul1_x_shape, &mul2_x_shape, &mul3_x_shape, &add2_y_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}