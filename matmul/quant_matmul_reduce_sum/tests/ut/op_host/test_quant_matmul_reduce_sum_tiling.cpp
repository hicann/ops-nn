/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_matmul_reduce_sum_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../op_host/op_tiling/quant_matmul_reduce_sum_tiling.h"

using namespace std;
using namespace ge;


class QuantMatmulReduceSumTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "QuantMatmulReduceSum SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "QuantMatmulReduceSum TearDown" << std::endl;
  }
};

struct QuantMatmulReduceSumTilingTestData {
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape x1ScaleShape;
    gert::StorageShape x2ScaleShape;
    gert::StorageShape outShape;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType x1ScaleDtype;
    ge::DataType x2ScaleDtype;
    ge::DataType outDType;
};

static void TestInputFunc(struct QuantMatmulReduceSumTilingTestData testCase, uint64_t expectKey)
{
    std::string opType("QuantMatmulReduceSum");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    string compileInfoString = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 1024,
                                            "load3d_constraints": "0",
                                            "Intrinsic_fix_pipe_l0c2out": true,
                                            "Intrinsic_data_move_l12ub": false,
                                            "Intrinsic_data_move_l0c2ub": false,
                                            "Intrinsic_data_move_out2l1_nd2nz": true,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 20
                                        }
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    struct QuantMatmulReduceSumCompileInfo{
    } compileInfo;

    auto param = gert::TilingData::CreateCap(12016);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());

    gert::StorageShape dimsShape = {{1}, {1}};
    gert::StorageShape biasShape = {{}, {}};
    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(6, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1})
            .InputShapes({&(testCase.x1Shape), &(testCase.x2Shape), &dimsShape, &biasShape,
                          &(testCase.x1ScaleShape), &(testCase.x2ScaleShape)})
            .OutputShapes({&(testCase.outShape)})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
            .NodeInputTd(0, testCase.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, testCase.x2Dtype, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
            .NodeInputTd(4, testCase.x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, testCase.x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, testCase.outDType, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(wsSize)
            .SetOpType(opType)
            .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // to check tiling result
    auto tilingKey = tilingContext->GetTilingKey();
    ASSERT_EQ(tilingKey, expectKey);
}

TEST_F(QuantMatmulReduceSumTilingTest, obj_cases_01)
{
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulReduceSum"), nullptr);

  struct QuantMatmulReduceSumTilingTestData testCase = {
      {{8, 2048, 1024}, {8, 2048, 1024}},
      {{8, 1024, 7168}, {8, 1024, 7168}},
      {{8, 2048}, {8, 2048}},
      {{7168}, {7168}},
      {{2048, 7168}, {2048, 7168}},
      ge::DT_INT8,
      ge::DT_INT8,
      ge::DT_FLOAT,
      ge::DT_BF16,
      ge::DT_BF16,
  };

  TestInputFunc(testCase, 0);
}

TEST_F(QuantMatmulReduceSumTilingTest, obj_cases_02)
{
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulReduceSum"), nullptr);

  struct QuantMatmulReduceSumTilingTestData testCase = {
      {{8, 2048, 2048}, {8, 2048, 2048}},
      {{8, 2048, 7168}, {8, 2048, 7168}},
      {{8, 2048}, {8, 2048}},
      {{7168}, {7168}},
      {{2048, 7168}, {2048, 7168}},
      ge::DT_INT8,
      ge::DT_INT8,
      ge::DT_FLOAT,
      ge::DT_BF16,
      ge::DT_BF16,
  };

  TestInputFunc(testCase, 1);
}
