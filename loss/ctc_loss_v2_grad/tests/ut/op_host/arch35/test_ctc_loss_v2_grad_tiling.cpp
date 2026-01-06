/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/ctc_loss_v2_grad_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class CTCLossV2GradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CTCLossV2Grad Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CTCLossV2Grad Tiling TearDown" << std::endl;
    }
};

TEST_F(CTCLossV2GradTiling, test_rt2_success)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 245760,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // 定义编译信息结构体
    struct CTCLossV2GradCompileInfo {
    } compile_info;

    // 获取操作符实现
    std::string op_type("CTCLossV2Grad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // // 定义操作符和输入性状
    gert::StorageShape gradOutShape = {{1}, {1}};
    gert::StorageShape logProbsShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape targetsShape = {{1, 32}, {1, 32}};
    gert::StorageShape inputLengthsShape = {{1}, {1}};
    gert::StorageShape targetLengthsShape = {{1}, {1}};
    gert::StorageShape lossShape = {{1}, {1}};
    gert::StorageShape logAlphaShape = {{1, 32, 65}, {1, 32, 65}};
    gert::StorageShape gradShape = {{32, 1, 1024}, {32, 1, 1024}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV2Grad")
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradOutShape, &logProbsShape, &targetsShape, &inputLengthsShape, &targetLengthsShape,
                           &lossShape, &logAlphaShape})
                      .OutputShapes({&gradShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                           {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

    std::cout << "test>> holder.GetContext" << std::endl;

    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    std::cout << "test>> holder.GetContext end" << std::endl;

    if (tiling_func == nullptr) {
        std::cout << "test>> tiling_func is nil" << std::endl;
    } else {
        std::cout << "test>> tiling_func is valid" << std::endl;
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

        auto realTilingKey = tiling_context->GetTilingKey();
        ASSERT_EQ(realTilingKey, 0);

        std::cout << "test>> holder.GetContext end" << std::endl;
    }
}