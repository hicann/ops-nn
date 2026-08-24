/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <string>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

using namespace ge;
using namespace ut_util;

namespace {

struct SeluGradTilingCompileInfo {};

std::string GetCompileInfo()
{
    return R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 48
        }
    })";
}

void ExpectTilingStatus(const gert::StorageShape& gradientsShape, const gert::StorageShape& outputsShape,
                        const gert::StorageShape& yShape, ge::graphStatus expectedStatus)
{
    std::string compileInfoString = GetCompileInfo();
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    SeluGradTilingCompileInfo compileInfo;
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(1);
    auto* workspaceSizes = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspaceSizes, nullptr);

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("SeluGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({const_cast<gert::StorageShape*>(&gradientsShape),
                                    const_cast<gert::StorageShape*>(&outputsShape)})
                      .OutputShapes({const_cast<gert::StorageShape*>(&yShape)})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspaceSizes)
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(opImpl->tiling(context), expectedStatus);
}

} // namespace

TEST(SeluGradTilingTest, SameShapeSucceeds)
{
    const gert::StorageShape gradientsShape({2, 3, 4}, {2, 3, 4});
    const gert::StorageShape outputsShape({2, 3, 4}, {2, 3, 4});
    const gert::StorageShape yShape({2, 3, 4}, {2, 3, 4});

    ExpectTilingStatus(gradientsShape, outputsShape, yShape, ge::GRAPH_SUCCESS);
}

TEST(SeluGradTilingTest, SameEmptyShapeSucceeds)
{
    const gert::StorageShape gradientsShape({0, 3, 4}, {0, 3, 4});
    const gert::StorageShape outputsShape({0, 3, 4}, {0, 3, 4});
    const gert::StorageShape yShape({0, 3, 4}, {0, 3, 4});

    ExpectTilingStatus(gradientsShape, outputsShape, yShape, ge::GRAPH_SUCCESS);
}

TEST(SeluGradTilingTest, BroadcastableShapeFails)
{
    const gert::StorageShape gradientsShape({2, 3, 4}, {2, 3, 4});
    const gert::StorageShape outputsShape({1, 3, 1}, {1, 3, 1});
    const gert::StorageShape yShape({2, 3, 4}, {2, 3, 4});

    ExpectTilingStatus(gradientsShape, outputsShape, yShape, ge::GRAPH_FAILED);
}

TEST(SeluGradTilingTest, EmptyBroadcastableShapeFails)
{
    const gert::StorageShape gradientsShape({0, 3, 4}, {0, 3, 4});
    const gert::StorageShape outputsShape({1, 3, 1}, {1, 3, 1});
    const gert::StorageShape yShape({0, 3, 4}, {0, 3, 4});

    ExpectTilingStatus(gradientsShape, outputsShape, yShape, ge::GRAPH_FAILED);
}

TEST(SeluGradTilingTest, OutputShapeMismatchFails)
{
    const gert::StorageShape gradientsShape({2, 3, 4}, {2, 3, 4});
    const gert::StorageShape outputsShape({2, 3, 4}, {2, 3, 4});
    const gert::StorageShape yShape({2, 3, 5}, {2, 3, 5});

    ExpectTilingStatus(gradientsShape, outputsShape, yShape, ge::GRAPH_FAILED);
}
