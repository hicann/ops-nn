/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_embedding_dense_grad_v2_regbase_tiling.cpp
 * \brief UT for embedding_dense_grad_v2_regbase_tiling.cpp
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

static const string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64}
                                    })";

class EmbeddingDenseGradV2RegBaseTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EmbeddingDenseGradV2RegBaseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EmbeddingDenseGradV2RegBaseTiling TearDown" << std::endl;
    }
};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, const std::string& socVersion)
{
    fe::OptionalInfos optionInfo;
    if (fe::PlatformInfoManager::GeInstance().InitializePlatformInfo() != ge::GRAPH_SUCCESS) {
        return;
    }

    if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(socVersion, platFormInfo, optionInfo) !=
        ge::GRAPH_SUCCESS) {
        return;
    }
}

static void ExecuteRegBaseTilingTest(
    const gert::StorageShape& grad_shape,
    const gert::StorageShape& indices_shape,
    const gert::StorageShape& posidx_shape,
    const gert::StorageShape& output_shape,
    ge::DataType grad_dtype,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    ge::graphStatus expect_result)
{
    std::string op_type("EmbeddingDenseGradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {
        {"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    InitPlatForm(platform_info, "Ascend950");

    struct EmbeddingDenseGradV2CompileInfo {
        int32_t totalCoreNum = 0;
        uint64_t ubSizePlatForm = 0;
        bool isRegBase = true;
    } compile_info;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({const_cast<gert::StorageShape*>(&grad_shape),
                          const_cast<gert::StorageShape*>(&indices_shape),
                          const_cast<gert::StorageShape*>(&posidx_shape)})
            .OutputShapes({const_cast<gert::StorageShape*>(&output_shape)})
            .CompileInfo(&compile_info)
            .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(num_weights)},
                        {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(padding_idx)},
                        {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(scale_grad_by_freq)}})
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, grad_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, grad_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), expect_result);
}

// ======================== RegBase Success Cases ========================

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_basic_fp32)
{
    gert::StorageShape grad_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape indices_shape = {{1024}, {1024}};
    gert::StorageShape posidx_shape = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 4096}, {100, 4096}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 100, -1, false,
                             ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_scale_grad)
{
    gert::StorageShape grad_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape indices_shape = {{1024}, {1024}};
    gert::StorageShape posidx_shape = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 4096}, {100, 4096}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 100, -1, true,
                             ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_small_dim)
{
    gert::StorageShape grad_shape = {{1024, 256}, {1024, 256}};
    gert::StorageShape indices_shape = {{1024}, {1024}};
    gert::StorageShape posidx_shape = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 256}, {100, 256}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 100, -1, false,
                             ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_multi_dim_indices)
{
    gert::StorageShape grad_shape = {{400, 385}, {400, 385}};
    gert::StorageShape indices_shape = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
    gert::StorageShape posidx_shape = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
    gert::StorageShape output_shape = {{49, 385}, {49, 385}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 49, 6, false,
                             ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_large_elewise)
{
    gert::StorageShape grad_shape = {{141, 10000}, {141, 10000}};
    gert::StorageShape indices_shape = {{3, 47, 1, 1, 1, 1}, {3, 47, 1, 1, 1, 1}};
    gert::StorageShape posidx_shape = {{3, 47, 1, 1, 1, 1}, {3, 47, 1, 1, 1, 1}};
    gert::StorageShape output_shape = {{49, 10000}, {49, 10000}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 49, 6, false,
                             ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_fp16)
{
    gert::StorageShape grad_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape indices_shape = {{1024}, {1024}};
    gert::StorageShape posidx_shape = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 4096}, {100, 4096}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT16, 100, -1, false,
                             ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_small_scatter)
{
    gert::StorageShape grad_shape = {{100, 128}, {100, 128}};
    gert::StorageShape indices_shape = {{100}, {100}};
    gert::StorageShape posidx_shape = {{100}, {100}};
    gert::StorageShape output_shape = {{50, 128}, {50, 128}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 50, -1, false,
                             ge::GRAPH_SUCCESS);
}

// ======================== RegBase Failure Cases ========================

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_indices_pos_not_equal)
{
    gert::StorageShape grad_shape = {{400, 385}, {400, 385}};
    gert::StorageShape indices_shape = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
    gert::StorageShape posidx_shape = {{3, 47, 1, 1, 1, 1}, {3, 47, 1, 1, 1, 1}};
    gert::StorageShape output_shape = {{49, 385}, {49, 385}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 49, 6, false,
                             ge::GRAPH_FAILED);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_indices_grad_mismatch)
{
    gert::StorageShape grad_shape = {{386, 385}, {386, 385}};
    gert::StorageShape indices_shape = {{3, 50, 2, 1, 1, 1}, {3, 50, 2, 1, 1, 1}};
    gert::StorageShape posidx_shape = {{3, 50, 2, 1, 1, 1}, {3, 50, 2, 1, 1, 1}};
    gert::StorageShape output_shape = {{49, 385}, {49, 385}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 49, 6, false,
                             ge::GRAPH_FAILED);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_num_weights_mismatch)
{
    gert::StorageShape grad_shape = {{400, 385}, {400, 385}};
    gert::StorageShape indices_shape = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
    gert::StorageShape posidx_shape = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
    gert::StorageShape output_shape = {{49, 385}, {49, 385}};

    ExecuteRegBaseTilingTest(grad_shape, indices_shape, posidx_shape, output_shape, ge::DT_FLOAT, 40, 6, false,
                             ge::GRAPH_FAILED);
}

TEST_F(EmbeddingDenseGradV2RegBaseTiling, regbase_tiling_indices_int64)
{
    std::string op_type("EmbeddingDenseGradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {
        {"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    InitPlatForm(platform_info, "Ascend950");

    struct EmbeddingDenseGradV2CompileInfo {
        int32_t totalCoreNum = 0;
        uint64_t ubSizePlatForm = 0;
        bool isRegBase = true;
    } compile_info;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape grad_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape indices_shape = {{1024}, {1024}};
    gert::StorageShape posidx_shape = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 4096}, {100, 4096}};

    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&grad_shape, &indices_shape, &posidx_shape})
            .OutputShapes({&output_shape})
            .CompileInfo(&compile_info)
            .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(100)},
                        {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                        {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
