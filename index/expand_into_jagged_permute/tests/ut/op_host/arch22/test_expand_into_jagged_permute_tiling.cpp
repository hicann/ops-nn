/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "base/err_mgr.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch22/expand_into_jagged_permute_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
ge::graphStatus TilingPrepareExpandIntoJaggedPermute(gert::TilingParseContext* context);
} // namespace optiling

namespace {
class ScopedErrorContext {
public:
    ScopedErrorContext() : savedContext_(error_message::GetErrMgrContext())
    {
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::GetErrMgrRawErrorMessages();
    }

    ~ScopedErrorContext()
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

CapturedError CaptureError()
{
    CapturedError captured;
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    captured.count = errors.size();
    if (errors.empty()) {
        return captured;
    }
    const auto& error = errors.back();
    captured.code = error.error_id == nullptr ? "" : error.error_id.get();
    for (const auto& key : error.args_key) {
        captured.keys.emplace_back(key == nullptr ? "" : key.get());
    }
    for (const auto& value : error.args_value) {
        captured.values.emplace_back(value == nullptr ? "" : value.get());
    }
    return captured;
}
} // namespace

class ExpandIntoJaggedPermuteTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExpandIntoJaggedPermuteTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ExpandIntoJaggedPermuteTiling TearDown" << std::endl; }

private:
    ScopedErrorContext errorContext_;
};

TEST_F(ExpandIntoJaggedPermuteTiling, test_tiling_int32)
{
    gert::StorageShape permute_shape = {{3}, {3}};
    gert::StorageShape input_offsets_shape = {{4}, {4}};
    gert::StorageShape output_offsets_shape = {{4}, {4}};
    // output
    gert::StorageShape output_permute_shape = {{6}, {6}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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
    struct ExpandIntoJaggedPermuteTilingCompileInfo {};
    ExpandIntoJaggedPermuteTilingCompileInfo compile_info;

    std::string op_type("ExpandIntoJaggedPermute");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ExpandIntoJaggedPermute")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&permute_shape, &input_offsets_shape, &output_offsets_shape})
                      .OutputShapes({&output_permute_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<int64_t>(6)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto* input_offsets_context_shape = const_cast<gert::StorageShape*>(tiling_context->GetInputShape(1));
    ASSERT_NE(input_offsets_context_shape, nullptr);
    *input_offsets_context_shape = {{5}, {5}};
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

static void ExpectSingleStageError(const std::string& reason)
{
    const auto error = CaptureError();
    ASSERT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ1008");
    ASSERT_EQ(error.keys, (std::vector<std::string>{"opName", "reason"}));
    ASSERT_EQ(error.values.size(), 2U);
    EXPECT_EQ(error.values[0], "ExpandIntoJaggedPermute");
    EXPECT_NE(error.values[1].find(reason), std::string::npos);
}

TEST_F(ExpandIntoJaggedPermuteTiling, null_tiling_context_reports_stage_failure_once)
{
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ExpandIntoJaggedPermute");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
    ExpectSingleStageError("TilingContext is nullptr");
}

TEST_F(ExpandIntoJaggedPermuteTiling, null_parse_context_reports_stage_failure_once)
{
    EXPECT_EQ(optiling::TilingPrepareExpandIntoJaggedPermute(nullptr), ge::GRAPH_FAILED);
    ExpectSingleStageError("TilingParseContext is nullptr");
}
