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
 * \file test_adaptive_avg_pool2d_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

using namespace std;
using namespace ge;

struct AdaptiveAvgPool2dGradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

static const string TEST_OP_TYPE = "AdaptiveAvgPool2dGrad";

static const string COMPILE_INFO_STRING = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 40}
                                     })";

class AdaptiveAvgPool2dGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveAvgPool2dGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool2dGradTilingTest TearDown" << std::endl; }
};

// ============ Big Kernel Tiling Tests ============

TEST_F(AdaptiveAvgPool2dGradTilingTest, big_kernel_op_registered)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, big_kernel_tiling_parse_not_null)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, big_kernel_tiling_parse_success)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaptiveAvgPool2dGradCompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_EQ(opImpl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, big_kernel_compile_info_correct)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaptiveAvgPool2dGradCompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_EQ(opImpl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    EXPECT_GT(compile_info.coreNum, 0u);
    EXPECT_GT(compile_info.ubSize, 0u);
}

// ============ Simt Kernel Tiling Tests ============

TEST_F(AdaptiveAvgPool2dGradTilingTest, simt_kernel_op_registered)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, simt_kernel_tiling_parse_not_null)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, simt_kernel_tiling_parse_success)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaptiveAvgPool2dGradCompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_EQ(opImpl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, simt_kernel_compile_info_correct)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaptiveAvgPool2dGradCompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_EQ(opImpl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    EXPECT_GT(compile_info.coreNum, 0u);
    EXPECT_GT(compile_info.ubSize, 0u);
}
