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

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "log/log.h"

#include "op_host/tiling_templates_registry.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/op_tiling/arch35/quant_batch_matmul_v4_tiling.h"
#include "../../../op_host/op_tiling/arch35/quant_batch_matmul_v4_reg_base_tiling.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_v4_compile_info.h"
#include "../../../../common/op_host/math_util_nn.h"
#include "ut_string_utils.h"

using namespace ut_str;
using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;

namespace {
// The op_host UT target is built with -fno-access-control, so private/protected members of the tiling classes are
// reachable from the test. All branch-relevant state is driven directly through those members; the tiling context is
// only required so QuantBatchMatmulV4RegBase can be constructed with a non-null compile info and a tiling data buffer.
void ConfigureSolverResult(QuantBatchMatmulV4RegBase& r, int64_t singleK, int64_t singleN, int64_t stepKb,
                           int64_t baseK, int64_t stepN, int64_t baseN, int64_t b1BufferNum)
{
    auto& p = r.tilingSolver_.basicBlockParam_;
    p.singleK = singleK;
    p.singleN = singleN;
    p.l1Param.stepKb = stepKb;
    p.l1Param.stepKa = stepKb;
    p.l1Param.stepN = stepN;
    p.l1Param.stepM = 1;
    p.l1Param.B1BufferNum = b1BufferNum;
    p.l1Param.A1BufferNum = 2;
    p.l1Param.iterateOrder = 1;
    p.l1Param.scaleFactor = 1;
    p.basicBlock.baseK = baseK;
    p.basicBlock.baseN = baseN;
    p.basicBlock.baseM = 128;
    p.mSize = 128;
    p.nSize = 128;
    p.kSize = 512;
    p.mDim = 1;
    p.nDim = 1;
    p.kDim = 1;
    p.singleM = 128;
}
} // namespace

// White-box coverage for arch35 QuantBatchMatmulV4RegBase, targeting the previously uncovered branches of
// GetBubTilingA8W4 / GetBubTilingA8W4BySize and the private print helpers.
TEST(QuantBatchMatmulV4RegBaseCovTest, CoverBubAndPrintBranches)
{
    constexpr int64_t m = 128;
    constexpr int64_t k = 512;
    constexpr int64_t n = 128;
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape outputShape({m, n}, {m, n});
    x1Shape.MutableStorageShape() = gert::Shape({m, k});
    x1Shape.MutableOriginShape() = x1Shape.MutableStorageShape();
    x2Shape.MutableStorageShape() = gert::Shape({n, k});
    x2Shape.MutableOriginShape() = x2Shape.MutableStorageShape();

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::QuantBatchMatmulV4CompileInfo compileInfo;

    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&x1Shape, &x2Shape, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(int64_t(64))},
                                  {"compute_type", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(int64_t(64))}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType("QuantBatchMatmulV4")
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);

    QuantBatchMatmulV4RegBase regbase(tilingContext);

    // Drive the branch-relevant state directly (no reliance on GetPlatformInfo/GetShapeAttrsInfo/DoOpTiling).
    regbase.inputParams_.opName = "QuantBatchMatmulV4RegBaseCov";
    regbase.inputParams_.groupSize = 64; // avoid div-by-zero in the transB&&weightNz path
    regbase.inputParams_.bDtype = ge::DT_INT8;
    regbase.inputParams_.aDtype = ge::DT_FLOAT8_E4M3FN;
    regbase.inputParams_.cDtype = ge::DT_BF16;
    regbase.inputParams_.hasBias = false;
    regbase.inputParams_.hasX1Scale = false;
    regbase.inputParams_.hasX2Scale = false;
    regbase.inputParams_.hasAntiQuantOffset = false;
    regbase.aicoreParams_.ubSize = 196352;

    // Populate tilingData_ (matmul tiling fields) through the real setter so the print helpers serialize real state.
    ConfigureSolverResult(regbase, 512, 128, 1, 64, 1, 128, 2);
    regbase.SetMatmulTiling();
    regbase.SetBubTiling(); // exercises GetBubTilingA8W4 once (B1BufferNum != 4 -> GetBubTilingA8W4BySize early return)

    // --- GetBubTilingA8W4 dispatcher branches (B1BufferNum == 4) ---

    // Branch: transB && weightNz, kBl1Size > alignSize (NZ_C0_SIZE == 32)
    regbase.inputParams_.transB = true;
    regbase.inputParams_.weightNz = true;
    ConfigureSolverResult(regbase, 512, 128, 1, 64, 1, 128, 4); // kBl1 = min(512, 64) = 64 > 32
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        regbase.GetBubTilingA8W4(nBub, kBub);
        EXPECT_GT(kBub, 0);
    }

    // Branch: transB && weightNz, kBl1Size <= alignSize
    ConfigureSolverResult(regbase, 16, 128, 1, 16, 1, 128, 4); // kBl1 = min(16, 16) = 16 <= 32
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        regbase.GetBubTilingA8W4(nBub, kBub);
        EXPECT_EQ(kBub, 16);
    }

    // Branch: !(transB && weightNz), nBl1Size <= alignSize
    regbase.inputParams_.transB = false;
    regbase.inputParams_.weightNz = false;
    ConfigureSolverResult(regbase, 512, 16, 1, 64, 1, 16, 4); // nBl1 = min(16, 16) = 16 <= 32
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        regbase.GetBubTilingA8W4(nBub, kBub);
        EXPECT_GT(kBub, 0);
    }

    // Branch: !(transB && weightNz), nBl1Size > alignSize (reinforce the already-covered sub-branch)
    ConfigureSolverResult(regbase, 512, 128, 1, 64, 1, 128, 4); // nBl1 = 128 > 32
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        regbase.GetBubTilingA8W4(nBub, kBub);
        EXPECT_GT(nBub, 0);
    }

    // --- GetBubTilingA8W4BySize deeper paths (called directly with crafted kBl1/nBl1) ---
    regbase.inputParams_.weightNz = false;
    regbase.inputParams_.transB = false;
    regbase.aicoreParams_.ubSize = 196352;

    // Early return path: small load that fits UB.
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        int64_t kBl1 = 16;
        int64_t nBl1 = 16;
        regbase.GetBubTilingA8W4BySize(nBub, kBub, kBl1, nBl1);
        EXPECT_EQ(nBub, 16);
        EXPECT_EQ(kBub, 16);
    }

    // Default-solution + ub overflow fallback + kBl1 not cacheline aligned -> early return (244 & 249 branches).
    // weightNz=true makes GetBubSize large; tiny UB forces the overflow branch.
    regbase.inputParams_.weightNz = true;
    regbase.aicoreParams_.ubSize = 10000;
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        int64_t kBl1 = 6000; // 6000 % 256 != 0
        int64_t nBl1 = 256;
        regbase.GetBubTilingA8W4BySize(nBub, kBub, kBl1, nBl1);
        EXPECT_GT(nBub, 0);
    }

    // Cacheline-aligned kBl1, loop finds an ideal divisible pair (kBl1%tmpKBub==0 && nBl1%tmpNBub==0).
    regbase.inputParams_.weightNz = false;
    regbase.aicoreParams_.ubSize = 196352;
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        int64_t kBl1 = 512; // 512%256==0, finds 512/64
        int64_t nBl1 = 128;
        regbase.GetBubTilingA8W4BySize(nBub, kBub, kBl1, nBl1);
        EXPECT_GT(nBub, 0);
        EXPECT_GT(kBub, 0);
    }

    // Cacheline-aligned kBl1, loop exhausts without an ideal pair -> default solution (line 271).
    {
        int64_t nBub = 0;
        int64_t kBub = 0;
        int64_t kBl1 = 256; // no tmp pair satisfies min load
        int64_t nBl1 = 48;
        regbase.GetBubTilingA8W4BySize(nBub, kBub, kBl1, nBl1);
        EXPECT_GT(nBub, 0);
    }

    // --- Print helpers (private, reached directly thanks to -fno-access-control) ---
    EXPECT_EQ(regbase.InstantiateTilingData(), ge::GRAPH_SUCCESS);
    regbase.PrintCVTilingData(true);      // OPS_LOG_D branch
    regbase.PrintCVTilingData(false);     // OPS_LOG_E (else) branch
    regbase.PrintTilingData(true);        // forwards to PrintCVTilingData(true)
    regbase.PrintTilingData(false);       // forwards to PrintCVTilingData(false)
    regbase.DumpCVTilingDataToLog(true);  // debug branch + PrintMatMulTiling
    regbase.DumpCVTilingDataToLog(false); // error branch + PrintMatMulTiling
    regbase.PrintMatMulTiling();          // matmul tiling dump
    SUCCEED();
}
