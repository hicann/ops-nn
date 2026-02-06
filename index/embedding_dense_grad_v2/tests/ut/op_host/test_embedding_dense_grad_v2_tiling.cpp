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
 * \file test_embedding_dense_grad_v2_tiling.cpp
 * \brief
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

class EmbeddingDenseGradV2Tiling : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "EmbeddingDenseGradV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "EmbeddingDenseGradV2Tiling TearDown" << std::endl;
    }
};

// static void InitPlatForm(fe::PlatFormInfos& platFormInfo, const std::string& socVersion)
// {
//     fe::OptionalInfos optionInfo;
//     if (fe::PlatformInfoManager::GeInstance().InitializePlatformInfo() != ge::GRAPH_SUCCESS) {
//         return;
//     }

//     if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(socVersion, platFormInfo, optionInfo) !=
//         ge::GRAPH_SUCCESS) {
//         return;
//     }
// }

TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_0) {
    std::string op_type("EmbeddingDenseGradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradV2CompileInfo {
      int32_t totalCoreNum = 0;
      uint64_t ubSizePlatForm = 0;
      bool isRegbase = false;
    } compile_info;
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape input_1 = {{1024}, {1024}};
    gert::StorageShape input_2 = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 4096}, {100, 4096}};

    // tilingParseFunc simulate
    auto holder =
        gert::TilingContextFaker()
          .NodeIoNum(3, 1)
          .IrInstanceNum({1, 1, 1})
          .InputShapes({&input_0, &input_1, &input_2})
          .OutputShapes({&output_shape})
          .CompileInfo(&compile_info)
          .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(100)},
                      {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
          .PlatformInfo(reinterpret_cast<char *>(&platform_info))
          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_1) {
    std::string op_type("EmbeddingDenseGradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradV2CompileInfo {
      int32_t totalCoreNum = 0;
      uint64_t ubSizePlatForm = 0;
      bool isRegbase = false;
    } compile_info;
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape input_1 = {{1024}, {1024}};
    gert::StorageShape input_2 = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 4096}, {100, 4096}};

    // tilingParseFunc simulate
    auto holder =
        gert::TilingContextFaker()
          .NodeIoNum(3, 1)
          .IrInstanceNum({1, 1, 1})
          .InputShapes({&input_0, &input_1, &input_2})
          .OutputShapes({&output_shape})
          .CompileInfo(&compile_info)
          .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(100)},
                      {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
          .PlatformInfo(reinterpret_cast<char *>(&platform_info))
          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1);
}

TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_2) {
    std::string op_type("EmbeddingDenseGradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradV2CompileInfo {
      int32_t totalCoreNum = 0;
      uint64_t ubSizePlatForm = 0;
      bool isRegbase = false;
    } compile_info;
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{1024, 256}, {1024, 256}};
    gert::StorageShape input_1 = {{1024}, {1024}};
    gert::StorageShape input_2 = {{1024}, {1024}};
    gert::StorageShape output_shape = {{100, 256}, {100, 256}};

    // tilingParseFunc simulate
    auto holder =
        gert::TilingContextFaker()
          .NodeIoNum(3, 1)
          .IrInstanceNum({1, 1, 1})
          .InputShapes({&input_0, &input_1, &input_2})
          .OutputShapes({&output_shape})
          .CompileInfo(&compile_info)
          .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(100)},
                      {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
          .PlatformInfo(reinterpret_cast<char *>(&platform_info))
          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 101);
}

// TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_split) {
//     std::string op_type("EmbeddingDenseGradV2");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                        "Intrinsic_fix_pipe_l0c2out": false,
//                                                        "Intrinsic_data_move_l12ub": true,
//                                                        "Intrinsic_data_move_l0c2ub": true,
//                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                        "CORE_NUM": 64}
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     InitPlatForm(platform_info, "Ascend950");
//     // compile info
//     struct EmbeddingDenseGradV2CompileInfo {
//       int32_t totalCoreNum = 0;
//       uint64_t ubSizePlatForm = 0;
//       bool isRegbase = true;
//     } compile_info;
//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                             intrinsics);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape input_0 = {{400, 385}, {400, 385}};
//     gert::StorageShape input_1 = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
//     gert::StorageShape input_2 = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
//     gert::StorageShape output_shape = {{49, 385}, {49, 385}};

//     // tilingParseFunc simulate
//     auto holder =
//         gert::TilingContextFaker()
//           .NodeIoNum(3, 1)
//           .IrInstanceNum({1, 1, 1})
//           .InputShapes({&input_0, &input_1, &input_2})
//           .OutputShapes({&output_shape})
//           .CompileInfo(&compile_info)
//           .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(49)},
//                       {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
//                       {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
//           .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//           .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .TilingData(param.get())
//           .Workspace(ws_size)
//           .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     // todo check tiling result
//     auto tiling_key = tiling_context->GetTilingKey();
//     auto raw_tiling_data = tiling_context->GetRawTilingData();
//     auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
//     uint64_t tilingKeyValue = 1000001;
//     ASSERT_EQ(tiling_key, tilingKeyValue);
//     std::string expectTilingData = "400 0 385 0 13 0 4 0 1 0 1 0 1 0 49 0 6 0 32 128 13 1 1216 385 1 0 1 0 ";
//     EXPECT_EQ(tiling_data_result, expectTilingData);
// }

// TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_no_split) {
//   std::string op_type("EmbeddingDenseGradV2");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                      "Intrinsic_fix_pipe_l0c2out": false,
//                                                      "Intrinsic_data_move_l12ub": true,
//                                                      "Intrinsic_data_move_l0c2ub": true,
//                                                      "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                      "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                      "CORE_NUM": 64}
//                                   })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//   InitPlatForm(platform_info, "Ascend950");
//   // compile info
//   struct EmbeddingDenseGradV2CompileInfo {
//     int32_t totalCoreNum = 0;
//     uint64_t ubSizePlatForm = 0;
//     bool isRegbase = true;
//   } compile_info;
//   // tilingParseFunc simulate
//   auto kernel_holder =
//       gert::KernelRunContextFaker()
//           .KernelIONum(2, 1)
//           .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//           .Outputs({&compile_info})
//           .Build();
//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                           intrinsics);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   ASSERT_NE(param, nullptr);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//   gert::StorageShape input_0 = {{282, 100}, {282, 100}};
//   gert::StorageShape input_1 = {{3, 47, 2, 1, 1, 1}, {3, 47, 2, 1, 1, 1}};
//   gert::StorageShape input_2 = {{3, 47, 2, 1, 1, 1}, {3, 47, 2, 1, 1, 1}};
//   gert::StorageShape output_shape = {{49, 100}, {49, 100}};

//   // tilingParseFunc simulate
//   auto holder =
//       gert::TilingContextFaker()
//         .NodeIoNum(3, 1)
//         .IrInstanceNum({1, 1, 1})
//         .InputShapes({&input_0, &input_1, &input_2})
//         .OutputShapes({&output_shape})
//         .CompileInfo(&compile_info)
//         .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(49)},
//                     {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
//                     {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
//         .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//         .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .TilingData(param.get())
//         .Workspace(ws_size)
//         .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//   // todo check tiling result
//   auto tiling_key = tiling_context->GetTilingKey();
//   auto raw_tiling_data = tiling_context->GetRawTilingData();
//   auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
//   uint64_t tilingKeyValue = 1000000;
//   ASSERT_EQ(tiling_key, tilingKeyValue);
//   std::string expectTilingData = "282 0 100 0 9 0 1 0 1 0 1 0 1 0 49 0 6 0 32 100 9 1 1792 100 1 0 1 0 ";
//   EXPECT_EQ(tiling_data_result, expectTilingData);
// }

// TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_scale_no_split) {
//   std::string op_type("EmbeddingDenseGradV2");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                      "Intrinsic_fix_pipe_l0c2out": false,
//                                                      "Intrinsic_data_move_l12ub": true,
//                                                      "Intrinsic_data_move_l0c2ub": true,
//                                                      "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                      "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                      "CORE_NUM": 64}
//                                   })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//   InitPlatForm(platform_info, "Ascend950");
//   // compile info
//   struct EmbeddingDenseGradV2CompileInfo {
//     int32_t totalCoreNum = 0;
//     uint64_t ubSizePlatForm = 0;
//     bool isRegbase = true;
//   } compile_info;
//   // tilingParseFunc simulate
//   auto kernel_holder =
//       gert::KernelRunContextFaker()
//           .KernelIONum(2, 1)
//           .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//           .Outputs({&compile_info})
//           .Build();
//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                           intrinsics);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   ASSERT_NE(param, nullptr);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//   gert::StorageShape input_0 = {{282, 100}, {282, 100}};
//   gert::StorageShape input_1 = {{3, 47, 2, 1, 1, 1}, {3, 47, 2, 1, 1, 1}};
//   gert::StorageShape input_2 = {{3, 47, 2, 1, 1, 1}, {3, 47, 2, 1, 1, 1}};
//   gert::StorageShape output_shape = {{49, 100}, {49, 100}};

//   // tilingParseFunc simulate
//   auto holder =
//       gert::TilingContextFaker()
//         .NodeIoNum(3, 1)
//         .IrInstanceNum({1, 1, 1})
//         .InputShapes({&input_0, &input_1, &input_2})
//         .OutputShapes({&output_shape})
//         .CompileInfo(&compile_info)
//         .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(49)},
//                     {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
//                     {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
//         .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//         .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .TilingData(param.get())
//         .Workspace(ws_size)
//         .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//   // todo check tiling result
//   auto tiling_key = tiling_context->GetTilingKey();
//   auto raw_tiling_data = tiling_context->GetRawTilingData();
//   auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
//   uint64_t tilingKeyValue = 1000010;
//   ASSERT_EQ(tiling_key, tilingKeyValue);
//   std::string expectTilingData = "282 0 100 0 9 0 1 0 1 0 1 0 1 0 49 0 6 0 32 100 9 1 1792 100 1 0 1 0 ";
//   EXPECT_EQ(tiling_data_result, expectTilingData);
// }

// TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_eltwise_too_large) {
//     std::string op_type("EmbeddingDenseGradV2");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                        "Intrinsic_fix_pipe_l0c2out": false,
//                                                        "Intrinsic_data_move_l12ub": true,
//                                                        "Intrinsic_data_move_l0c2ub": true,
//                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                        "CORE_NUM": 64}
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     InitPlatForm(platform_info, "Ascend950");
//     // compile info
//     struct EmbeddingDenseGradV2CompileInfo {
//       int32_t totalCoreNum = 0;
//       uint64_t ubSizePlatForm = 0;
//       bool isRegbase = true;
//     } compile_info;
//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                             intrinsics);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape input_0 = {{282, 5000}, {282, 5000}};
//     gert::StorageShape input_1 = {{3, 47, 1, 1, 1, 1}, {3, 47, 1, 1, 1, 1}};
//     gert::StorageShape input_2 = {{3, 47, 1, 1, 1, 1}, {3, 47, 1, 1, 1, 1}};
//     gert::StorageShape output_shape = {{49, 5000}, {49, 5000}};

//     // tilingParseFunc simulate
//     auto holder =
//         gert::TilingContextFaker()
//           .NodeIoNum(3, 1)
//           .IrInstanceNum({1, 1, 1})
//           .InputShapes({&input_0, &input_1, &input_2})
//           .OutputShapes({&output_shape})
//           .CompileInfo(&compile_info)
//           .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(49)},
//                       {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
//                       {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
//           .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//           .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .TilingData(param.get())
//           .Workspace(ws_size)
//           .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_indice_pos_not_equal) {
//     std::string op_type("EmbeddingDenseGradV2");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                        "Intrinsic_fix_pipe_l0c2out": false,
//                                                        "Intrinsic_data_move_l12ub": true,
//                                                        "Intrinsic_data_move_l0c2ub": true,
//                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                        "CORE_NUM": 64}
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     InitPlatForm(platform_info, "Ascend950");
//     // compile info
//     struct EmbeddingDenseGradV2CompileInfo {
//       int32_t totalCoreNum = 0;
//       uint64_t ubSizePlatForm = 0;
//       bool isRegbase = true;
//     } compile_info;
//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                             intrinsics);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape input_0 = {{400, 385}, {400, 385}};
//     gert::StorageShape input_1 = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
//     gert::StorageShape input_2 = {{3, 47, 1, 1, 1, 1}, {3, 47, 1, 1, 1, 1}};
//     gert::StorageShape output_shape = {{49, 385}, {49, 385}};

//     // tilingParseFunc simulate
//     auto holder =
//         gert::TilingContextFaker()
//           .NodeIoNum(3, 1)
//           .IrInstanceNum({1, 1, 1})
//           .InputShapes({&input_0, &input_1, &input_2})
//           .OutputShapes({&output_shape})
//           .CompileInfo(&compile_info)
//           .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(49)},
//                       {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
//                       {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
//           .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//           .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .TilingData(param.get())
//           .Workspace(ws_size)
//           .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_indice_grad_not_equal) {
//   std::string op_type("EmbeddingDenseGradV2");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                      "Intrinsic_fix_pipe_l0c2out": false,
//                                                      "Intrinsic_data_move_l12ub": true,
//                                                      "Intrinsic_data_move_l0c2ub": true,
//                                                      "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                      "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                      "CORE_NUM": 64}
//                                   })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//   InitPlatForm(platform_info, "Ascend950");
//   // compile info
//   struct EmbeddingDenseGradV2CompileInfo {
//     int32_t totalCoreNum = 0;
//     uint64_t ubSizePlatForm = 0;
//     bool isRegbase = true;
//   } compile_info;
//   // tilingParseFunc simulate
//   auto kernel_holder =
//       gert::KernelRunContextFaker()
//           .KernelIONum(2, 1)
//           .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//           .Outputs({&compile_info})
//           .Build();
//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                           intrinsics);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   ASSERT_NE(param, nullptr);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//   gert::StorageShape input_0 = {{386, 385}, {386, 385}};
//   gert::StorageShape input_1 = {{3, 50, 2, 1, 1, 1}, {3, 50, 2, 1, 1, 1}};
//   gert::StorageShape input_2 = {{3, 50, 2, 1, 1, 1}, {3, 50, 2, 1, 1, 1}};
//   gert::StorageShape output_shape = {{49, 385}, {49, 385}};

//   // tilingParseFunc simulate
//   auto holder =
//       gert::TilingContextFaker()
//         .NodeIoNum(3, 1)
//         .IrInstanceNum({1, 1, 1})
//         .InputShapes({&input_0, &input_1, &input_2})
//         .OutputShapes({&output_shape})
//         .CompileInfo(&compile_info)
//         .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(49)},
//                     {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
//                     {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
//         .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//         .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .TilingData(param.get())
//         .Workspace(ws_size)
//         .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(EmbeddingDenseGradV2Tiling, embedding_dense_grad_v2_tiling_out_not_equal_to_num_weight) {
//   std::string op_type("EmbeddingDenseGradV2");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                      "Intrinsic_fix_pipe_l0c2out": false,
//                                                      "Intrinsic_data_move_l12ub": true,
//                                                      "Intrinsic_data_move_l0c2ub": true,
//                                                      "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                      "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                      "CORE_NUM": 64}
//                                   })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//   InitPlatForm(platform_info, "Ascend950");
//   // compile info
//   struct EmbeddingDenseGradV2CompileInfo {
//     int32_t totalCoreNum = 0;
//     uint64_t ubSizePlatForm = 0;
//     bool isRegbase = true;
//   } compile_info;
//   // tilingParseFunc simulate
//   auto kernel_holder =
//       gert::KernelRunContextFaker()
//           .KernelIONum(2, 1)
//           .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//           .Outputs({&compile_info})
//           .Build();
//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                           intrinsics);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   ASSERT_NE(param, nullptr);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//   gert::StorageShape input_0 = {{400, 385}, {400, 385}};
//   gert::StorageShape input_1 = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
//   gert::StorageShape input_2 = {{4, 50, 2, 1, 1, 1}, {4, 50, 2, 1, 1, 1}};
//   gert::StorageShape output_shape = {{49, 385}, {49, 385}};

//   // tilingParseFunc simulate
//   auto holder =
//       gert::TilingContextFaker()
//         .NodeIoNum(3, 1)
//         .IrInstanceNum({1, 1, 1})
//         .InputShapes({&input_0, &input_1, &input_2})
//         .OutputShapes({&output_shape})
//         .CompileInfo(&compile_info)
//         .NodeAttrs({{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(40)},
//                     {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
//                     {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
//         .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//         .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//         .TilingData(param.get())
//         .Workspace(ws_size)
//         .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }
