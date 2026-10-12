/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/sparse_apply_adagrad_tiling_arch35.h"
#include "../../../../op_kernel/arch35/sparse_apply_adagrad_tiling_data.h"
#include "../../../../op_kernel/arch35/sparse_apply_adagrad_tiling_key.h"

using namespace ge;
using namespace ut_util;

class SparseApplyAdagradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseApplyAdagradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SparseApplyAdagradTilingTest TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platformInfo, std::map<std::string, std::string>& socInfos,
                         std::map<std::string, std::string>& aicoreSpec, std::map<std::string, std::string>& intrinsics)
{
    std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

static void DoTilingTest(gert::StorageShape& var, gert::StorageShape& accum, gert::StorageShape& lr,
                         gert::StorageShape& grad, gert::StorageShape& indices, bool useLocking, bool updateSlots,
                         uint64_t expectTilingKey)
{
    optiling::SparseApplyAdagradCompileInfo compileInfo;

    fe::PlatFormInfos platformInfo;
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    std::string opType("SparseApplyAdagrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    gert::StorageShape varOut = var;
    gert::StorageShape accumOut = accum;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&var, &accum, &lr, &grad, &indices})
                      .OutputShapes({&varOut, &accumOut})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(useLocking)},
                                  {"update_slots", Ops::NN::AnyValue::CreateFrom<bool>(updateSlots)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingContext->GetTilingKey(), expectTilingKey);
    EXPECT_GT(tilingContext->GetBlockDim(), 0U);

    // 回读 tiling data，校验 use_locking / update_slots 属性是否正确透传到 tiling 数据
    bool gotUseLocking = false;
    bool gotUpdateSlots = false;
    switch (expectTilingKey) {
        case SPARSE_APPLY_ADAGRAD_TILING_KEY_OPTIMIZE: {
            auto* td = tilingContext->GetTilingData<SparseApplyAdagradOptimizeTilingData>();
            gotUseLocking = td->useLocking;
            gotUpdateSlots = td->updateSlots;
            break;
        }
        case SPARSE_APPLY_ADAGRAD_TILING_KEY_OPTIMIZE_SPLIT_R: {
            auto* td = tilingContext->GetTilingData<SparseApplyAdagradOptimizeSplitRTilingData>();
            gotUseLocking = td->useLocking;
            gotUpdateSlots = td->updateSlots;
            break;
        }
        case SPARSE_APPLY_ADAGRAD_TILING_KEY_SMALL: {
            auto* td = tilingContext->GetTilingData<SparseApplyAdagradSmallDimTilingData>();
            gotUseLocking = td->useLocking;
            gotUpdateSlots = td->updateSlots;
            break;
        }
        case SPARSE_APPLY_ADAGRAD_TILING_KEY_BIG: {
            auto* td = tilingContext->GetTilingData<SparseApplyAdagradBigDimTilingData>();
            gotUseLocking = td->useLocking;
            gotUpdateSlots = td->updateSlots;
            break;
        }
        default:
            break;
    }
    EXPECT_EQ(gotUseLocking, useLocking);
    EXPECT_EQ(gotUpdateSlots, updateSlots);
}

TEST_F(SparseApplyAdagradTilingTest, SelectOptimizeSplitR)
{
    gert::StorageShape var({8, 512}, {8, 512});
    gert::StorageShape accum({8, 512}, {8, 512});
    gert::StorageShape lr({}, {});
    gert::StorageShape grad({2, 512}, {2, 512});
    gert::StorageShape indices({2}, {2});
    DoTilingTest(var, accum, lr, grad, indices, false, true, SPARSE_APPLY_ADAGRAD_TILING_KEY_OPTIMIZE_SPLIT_R);
}

TEST_F(SparseApplyAdagradTilingTest, SelectOptimize)
{
    gert::StorageShape var({128, 16}, {128, 16});
    gert::StorageShape accum({128, 16}, {128, 16});
    gert::StorageShape lr({1}, {1});
    gert::StorageShape grad({80, 16}, {80, 16});
    gert::StorageShape indices({80}, {80});
    DoTilingTest(var, accum, lr, grad, indices, false, true, SPARSE_APPLY_ADAGRAD_TILING_KEY_OPTIMIZE);
}

TEST_F(SparseApplyAdagradTilingTest, SelectSmallWhenUseLocking)
{
    gert::StorageShape var({16, 4}, {16, 4});
    gert::StorageShape accum({16, 4}, {16, 4});
    gert::StorageShape lr({}, {});
    gert::StorageShape grad({4, 4}, {4, 4});
    gert::StorageShape indices({4}, {4});
    DoTilingTest(var, accum, lr, grad, indices, true, true, SPARSE_APPLY_ADAGRAD_TILING_KEY_SMALL);
}

// use_locking=true 且 indices 极大导致 UB 装不下整份 indices 缓冲（ubFactor<=0）时，应选择 big 路径
TEST_F(SparseApplyAdagradTilingTest, SelectBigWhenUbExhausted)
{
    gert::StorageShape var({65536, 512}, {65536, 512});
    gert::StorageShape accum({65536, 512}, {65536, 512});
    gert::StorageShape lr({}, {});
    gert::StorageShape grad({65536, 512}, {65536, 512});
    gert::StorageShape indices({65536}, {65536});
    DoTilingTest(var, accum, lr, grad, indices, true, true, SPARSE_APPLY_ADAGRAD_TILING_KEY_BIG);
}

// update_slots=false 时不影响 tiling key 选择，但须正确透传到 tiling 数据
TEST_F(SparseApplyAdagradTilingTest, SelectOptimizeUpdateSlotsFalse)
{
    gert::StorageShape var({128, 16}, {128, 16});
    gert::StorageShape accum({128, 16}, {128, 16});
    gert::StorageShape lr({1}, {1});
    gert::StorageShape grad({80, 16}, {80, 16});
    gert::StorageShape indices({80}, {80});
    DoTilingTest(var, accum, lr, grad, indices, false, false, SPARSE_APPLY_ADAGRAD_TILING_KEY_OPTIMIZE);
}
