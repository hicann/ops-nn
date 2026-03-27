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
 * \file test_sync_batch_norm_backward_reduce_tiling.cpp
 * \brief
 */

#include "atvoss/elewise/elewise_tiling.h"
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <string>
#include "log/log.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SyncBatchNormBackwardReduceTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SyncBatchNormBackwardReduceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SyncBatchNormBackwardReduceTiling TearDown" << std::endl;
    }
};

static void InitPlatForm(
    fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
}

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp16_001)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000}, {1000}};
    gert::StorageShape invert_std_shape = {{1000}, {1000}};

    gert::StorageShape sum_dy_xum_shape = {{1000}, {1000}};
    gert::StorageShape y_shape = {{1000}, {1000}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp16_002)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000,20}, {1000,20}};
    gert::StorageShape sum_dy_shape = {{1000,20}, {1000,20}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000,20}, {1000,20}};
    gert::StorageShape invert_std_shape = {{1000,20}, {1000,20}};

    gert::StorageShape sum_dy_xum_shape = {{1000,20}, {1000,20}};
    gert::StorageShape y_shape = {{1000,20}, {1000,20}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp16_003)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape sum_dy_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape sum_dy_dx_pad_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape invert_std_shape = {{100,20,30,10}, {100,20,30,10}};

    gert::StorageShape sum_dy_xum_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape y_shape = {{100,20,30,10}, {100,20,30,10}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp16_004)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape sum_dy_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape sum_dy_dx_pad_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape invert_std_shape = {{100,20,30,10}, {100,20,30,10}};

    gert::StorageShape sum_dy_xum_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape y_shape = {{100,20,30,10}, {100,20,30,10}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_bf16_001)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000}, {1000}};
    gert::StorageShape invert_std_shape = {{1000}, {1000}};

    gert::StorageShape sum_dy_xum_shape = {{1000}, {1000}};
    gert::StorageShape y_shape = {{1000}, {1000}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp32_001)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000}, {1000}};
    gert::StorageShape invert_std_shape = {{1000}, {1000}};

    gert::StorageShape sum_dy_xum_shape = {{1000}, {1000}};
    gert::StorageShape y_shape = {{1000}, {1000}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}
TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp32_002)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000,20}, {1000,20}};
    gert::StorageShape sum_dy_shape = {{1000,20}, {1000,20}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000,20}, {1000,20}};
    gert::StorageShape invert_std_shape = {{1000,20}, {1000,20}};

    gert::StorageShape sum_dy_xum_shape = {{1000,20}, {1000,20}};
    gert::StorageShape y_shape = {{1000,20}, {1000,20}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp32_003)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape sum_dy_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape sum_dy_dx_pad_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape invert_std_shape = {{100,20,30,10}, {100,20,30,10}};

    gert::StorageShape sum_dy_xum_shape = {{100,20,30,10}, {100,20,30,10}};
    gert::StorageShape y_shape = {{100,20,30,10}, {100,20,30,10}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_fp32_004)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{100,20,30,10,2}, {100,20,30,10,2}};
    gert::StorageShape sum_dy_shape = {{100,20,30,10,2}, {100,20,30,10,2}};
    gert::StorageShape sum_dy_dx_pad_shape = {{100,20,30,10,2}, {100,20,30,10,2}};
    gert::StorageShape invert_std_shape = {{100,20,30,10,2}, {100,20,30,10,2}};

    gert::StorageShape sum_dy_xum_shape = {{100,20,30,10,2}, {100,20,30,10,2}};
    gert::StorageShape y_shape = {{100,20,30,10,2}, {100,20,30,10,2}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult, expectData);
    auto tiling_key = tilingContext->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_shape_diff_001)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000}, {1000}};
    gert::StorageShape invert_std_shape = {{1000}, {1000}};

    gert::StorageShape sum_dy_xum_shape = {{500}, {500}};
    gert::StorageShape y_shape = {{1000}, {1000}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_shape_diff_002)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000}, {1000}};
    gert::StorageShape invert_std_shape = {{500}, {500}};

    gert::StorageShape sum_dy_xum_shape = {{1000}, {1000}};
    gert::StorageShape y_shape = {{1000}, {1000}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_dtype_diff_001)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000}, {1000}};
    gert::StorageShape invert_std_shape = {{1000}, {1000}};

    gert::StorageShape sum_dy_xum_shape = {{1000}, {1000}};
    gert::StorageShape y_shape = {{1000}, {1000}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

TEST_F(SyncBatchNormBackwardReduceTiling, test_tiling_dtype_diff_002)
{
    std::string op_type("SyncBatchNormBackwardReduce");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape mean_Shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_shape = {{1000}, {1000}};
    gert::StorageShape sum_dy_dx_pad_shape = {{1000}, {1000}};
    gert::StorageShape invert_std_shape = {{1000}, {1000}};

    gert::StorageShape sum_dy_xum_shape = {{1000}, {1000}};
    gert::StorageShape y_shape = {{1000}, {1000}};

    Ops::Base::ElewiseCompileInfo compile_info;
    
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&mean_Shape, &sum_dy_shape, &sum_dy_dx_pad_shape, &invert_std_shape})
                      .OutputShapes({&sum_dy_xum_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}
