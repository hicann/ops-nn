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
#include <vector>
#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/max_pool_grad_grad_with_argmax_tiling_data.h"

using namespace std;
using namespace ge;
using namespace gert;

struct MaxPoolGradGradWithArgmaxCompileInfo {};

class MaxPoolGradGradWithArgmaxTilingRunTime : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolGradGradWithArgmaxTilingRunTime SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MaxPoolGradGradWithArgmaxTilingRunTime TearDown" << std::endl; }
};

static void RunTilingCase(const string& caseName, const string& opType, initializer_list<int64_t> xShape,
                          initializer_list<int64_t> gradShape, initializer_list<int64_t> argmaxShape,
                          initializer_list<int64_t> yShape, ge::DataType xDt, ge::DataType argmaxDt,
                          uint64_t expectTilingKey, bool expectSuccess)
{
    // dlog_setlevel(0, 0, 0);
    cout << "run case " << caseName << endl;

    string platform_info_str = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
        "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(platform_info_str.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    ASSERT_NE(OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    MaxPoolGradGradWithArgmaxCompileInfo opInfo;
    string compileInfo = "{}";
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(1, 1)
                            .Inputs({const_cast<char*>(compileInfo.c_str())})
                            .Outputs({&opInfo})
                            .Build();
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), GRAPH_SUCCESS);

    StorageShape xS = {xShape, xShape};
    StorageShape gradS = {gradShape, gradShape};
    StorageShape argmaxS = {argmaxShape, argmaxShape};
    StorageShape yS = {yShape, yShape};
    vector<pair<string, Ops::NN::AnyValue>> attrsPairs = {
        make_pair("ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})),
        make_pair("strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})),
        make_pair("padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")),
    };
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xS, &gradS, &argmaxS})
                      .OutputShapes({&yS})
                      .NodeAttrs(attrsPairs)
                      .NodeInputTd(0, xDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, xDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, argmaxDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&opInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tilingData.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                      .Build();

    TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    if (expectSuccess) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    } else {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
        return;
    }

    auto tilingKey = tilingContext->GetTilingKey();
    auto blockDim = tilingContext->GetBlockDim();
    cout << "tilingKey=" << tilingKey << " blockDim=" << blockDim << endl;
    EXPECT_EQ(tilingKey, expectTilingKey);
}

TEST_F(MaxPoolGradGradWithArgmaxTilingRunTime, local_mode_float32_int64)
{
    RunTilingCase("local_mode_float32_int64", "MaxPoolGradGradWithArgmax", {1, 4, 4, 1}, {1, 4, 4, 1}, {1, 2, 2, 1},
                  {1, 2, 2, 1}, ge::DT_FLOAT, ge::DT_INT64, 0, true);
}

TEST_F(MaxPoolGradGradWithArgmaxTilingRunTime, local_mode_float16_int32)
{
    RunTilingCase("local_mode_float16_int32", "MaxPoolGradGradWithArgmax", {2, 8, 8, 3}, {2, 8, 8, 3}, {2, 4, 4, 3},
                  {2, 4, 4, 3}, ge::DT_FLOAT16, ge::DT_INT32, 0, true);
}

TEST_F(MaxPoolGradGradWithArgmaxTilingRunTime, empty_tensor)
{
    RunTilingCase("empty_tensor", "MaxPoolGradGradWithArgmax", {1, 4, 4, 1}, {1, 4, 4, 1}, {1, 0, 0, 1}, {1, 0, 0, 1},
                  ge::DT_FLOAT, ge::DT_INT64, 0, true);
}
