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
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

using namespace ut_util;

class ForeachAddListTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachAddListTilingArch35 SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachAddListTilingArch35 TearDown" << std::endl; }
};

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType dt, ge::DataType alphaDt, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachAddListCompileInfo {
    } ci;
    auto opType = std::string("ForeachAddList");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xS = {xShape, xShape};
    gert::StorageShape yS = {xShape, xShape};
    gert::StorageShape scalarS = {{1}, {1}};
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(3, 1)
                  .IrInstanceNum({1, 1, 1})
                  .InputShapes({&xS, &xS, &scalarS})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(1, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(2, alphaDt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    EXPECT_EQ(tilingFn(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetTilingKey(), expectKey);
}

TEST_F(ForeachAddListTilingArch35, test_tiling_float16_arch35) { DoCase({32, 4}, ge::DT_FLOAT16, ge::DT_FLOAT16, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_float32_arch35) { DoCase({32, 4}, ge::DT_FLOAT, ge::DT_FLOAT, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_int32_arch35) { DoCase({32, 4}, ge::DT_INT32, ge::DT_INT32, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_bfloat16_arch35) { DoCase({32, 4}, ge::DT_BF16, ge::DT_FLOAT, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_int16_arch35) { DoCase({32, 4}, ge::DT_INT16, ge::DT_INT32, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_int8_arch35) { DoCase({32, 4}, ge::DT_INT8, ge::DT_INT32, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_uint8_arch35) { DoCase({32, 4}, ge::DT_UINT8, ge::DT_INT32, 0); }
