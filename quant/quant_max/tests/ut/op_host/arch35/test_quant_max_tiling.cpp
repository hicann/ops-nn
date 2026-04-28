/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_quant_max_tiling.cpp
 * @brief Unit test for QuantMax tiling
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "tiling_context_faker.h"
#include "test_cube_util.h"
#include "platform/platform_info.h"

// Include tiling header for QuantMaxCompileInfo struct (implementation is linked separately)
#include "../../../../op_host/arch35/quant_max_tiling.h"

using namespace std;
using namespace ge;

class QuantMaxTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "QuantMaxTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "QuantMaxTilingTest TearDown" << std::endl;
    }
};

// Test that QuantMax OpImpl is registered correctly
TEST_F(QuantMaxTilingTest, quant_max_op_impl_registered) {
    std::string op_type("QuantMax");

    // Check if OpImpl is registered
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);

    // Check if tiling function is registered
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    // Check if tiling_parse function is registered
    auto tiling_parse_func = op_impl->tiling_parse;
    ASSERT_NE(tiling_parse_func, nullptr);

    std::cout << "QuantMax OpImpl registered successfully" << std::endl;
}

// DISABLED: These tests require real hardware platform info (coreNum, ubSize)
// which cannot be properly simulated in UT environment. Enable when running on real hardware.
// Test basic tiling with FP32 input, calling tiling function
TEST_F(QuantMaxTilingTest, DISABLED_quant_max_tiling_fp32_success) {
    std::string op_type("QuantMax");

    // Setup compile info string with hardware info
    std::string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 262144,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 24
        }
    })";

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // Setup platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // Setup compile info
    optiling::QuantMaxCompileInfo compile_info;

    // Get tiling functions
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    auto tiling_parse_func = op_impl->tiling_parse;
    ASSERT_NE(tiling_func, nullptr);
    ASSERT_NE(tiling_parse_func, nullptr);

    // TilingParse simulate
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // Create tiling context
    gert::StorageShape x_shape({1024}, {1024});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024}, {1024});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .SetOpType("QuantMax")
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // DT_FLOAT8_E5M2
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);

    // Initialize platform info in tiling context
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // Actually call tiling function and verify result
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    std::cout << "QuantMax tiling executed successfully for FP32 input" << std::endl;
}

// DISABLED: Requires real hardware platform info
// Test FP16 input to HIFLOAT8 with Round mode
TEST_F(QuantMaxTilingTest, DISABLED_quant_max_tiling_fp16_hifloat8_success) {
    std::string op_type("QuantMax");

    // Setup compile info
    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;  // 256KB UB

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    // Create tiling context
    gert::StorageShape x_shape({1024}, {1024});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024}, {1024});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("round")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}  // DT_HIFLOAT8
        })
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);

    // Initialize platform info
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    // Get tiling function and call it
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    // Actually call tiling function and verify result
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    std::cout << "QuantMax tiling executed successfully for FP16 to HIFLOAT8" << std::endl;
}

// DISABLED: Requires real hardware platform info
// Test BF16 input with FLOAT8_E5M2 output
TEST_F(QuantMaxTilingTest, DISABLED_quant_max_tiling_bf16_success) {
    std::string op_type("QuantMax");

    // Setup compile info
    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;  // 256KB UB

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    // Create tiling context
    gert::StorageShape x_shape({2048}, {2048});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({2048}, {2048});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // DT_FLOAT8_E5M2
        })
        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);

    // Initialize platform info
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    // Get tiling function and call it
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    // Actually call tiling function and verify result
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    std::cout << "QuantMax tiling executed successfully for BF16 input" << std::endl;
}

// DISABLED: Requires real hardware platform info
// Test FLOAT8_E4M3FN output type (dstType = 36)
TEST_F(QuantMaxTilingTest, DISABLED_quant_max_tiling_fp8_e4m3fn_success) {
    std::string op_type("QuantMax");

    // Setup compile info
    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;  // 256KB UB

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    // Create tiling context
    gert::StorageShape x_shape({4096}, {4096});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({4096}, {4096});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)}  // DT_FLOAT8_E4M3FN
        })
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);

    // Initialize platform info
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    // Get tiling function and call it
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    // Actually call tiling function and verify result
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    std::cout << "QuantMax tiling executed successfully for FLOAT8_E4M3FN output" << std::endl;
}

// DISABLED: Requires real hardware platform info
// Test HIFLOAT8 with Hybrid round mode
TEST_F(QuantMaxTilingTest, DISABLED_quant_max_tiling_hifloat8_hybrid_success) {
    std::string op_type("QuantMax");

    // Setup compile info
    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;  // 256KB UB

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    // Create tiling context
    gert::StorageShape x_shape({512}, {512});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({512}, {512});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("hybrid")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}  // DT_HIFLOAT8
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);

    // Initialize platform info
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    // Get tiling function and call it
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    // Actually call tiling function and verify result
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    std::cout << "QuantMax tiling executed successfully for HIFLOAT8 with Hybrid mode" << std::endl;
}

// DISABLED: Requires real hardware platform info
// Test large shape (boundary test)
TEST_F(QuantMaxTilingTest, DISABLED_quant_max_tiling_large_shape_success) {
    std::string op_type("QuantMax");

    // Setup compile info
    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;  // 256KB UB

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    // Create tiling context with large shape (1M elements)
    gert::StorageShape x_shape({1024, 1024}, {1024, 1024});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024, 1024}, {1024, 1024});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // DT_FLOAT8_E5M2
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);

    // Initialize platform info
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    // Get tiling function and call it
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    // Actually call tiling function and verify result
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    std::cout << "QuantMax tiling executed successfully for large shape" << std::endl;
}

// Test CheckDtype failure: x dtype not in FLOAT/FLOAT16/BF16
TEST_F(QuantMaxTilingTest, quant_max_tiling_invalid_x_dtype)
{
    std::string op_type("QuantMax");

    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    gert::StorageShape x_shape({1024}, {1024});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024}, {1024});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
        })
        .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)  // Invalid dtype: INT32 not supported
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

    std::cout << "QuantMax tiling check dtype failure for INT32 input" << std::endl;
}

// Test CheckShape failure: x dim > 8
TEST_F(QuantMaxTilingTest, quant_max_tiling_x_dim_exceed_max)
{
    std::string op_type("QuantMax");

    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    // 9-dimension tensor exceeds max rank 8
    gert::StorageShape x_shape({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

    std::cout << "QuantMax tiling check shape failure for 9-dim input" << std::endl;
}

// Test CheckShape failure: scale dim != 1
TEST_F(QuantMaxTilingTest, quant_max_tiling_invalid_scale_dim)
{
    std::string op_type("QuantMax");

    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    gert::StorageShape x_shape({1024}, {1024});
    gert::StorageShape scale_shape({2, 2}, {2, 2});  // Invalid: scale dim should be 1
    gert::StorageShape y_shape({1024}, {1024});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

    std::cout << "QuantMax tiling check shape failure for invalid scale dim" << std::endl;
}

// Test CheckShape failure: amax dim != 1
TEST_F(QuantMaxTilingTest, quant_max_tiling_invalid_amax_dim)
{
    std::string op_type("QuantMax");

    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    gert::StorageShape x_shape({1024}, {1024});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024}, {1024});
    gert::StorageShape amax_shape({2}, {2});  // Invalid: amax dim should be 1, shape should be [1]

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

    std::cout << "QuantMax tiling check shape failure for invalid amax dim" << std::endl;
}

// Test CheckAttrs failure: dstType not in 34/35/36
TEST_F(QuantMaxTilingTest, quant_max_tiling_invalid_dst_type)
{
    std::string op_type("QuantMax");

    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    gert::StorageShape x_shape({1024}, {1024});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024}, {1024});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(99)}  // Invalid dstType: not 34/35/36
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

    std::cout << "QuantMax tiling check attrs failure for invalid dstType" << std::endl;
}

// Test CheckAttrs failure: roundMode mismatch with dstType
TEST_F(QuantMaxTilingTest, quant_max_tiling_roundmode_mismatch)
{
    std::string op_type("QuantMax");

    optiling::QuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"},
        {"SoCInfo_core_num", "24"},
        {"SoCInfo_ai_core_num", "24"}
    };
    std::map<std::string, std::string> aicore_spec = {
        {"ub_size", "262144"}
    };

    gert::StorageShape x_shape({1024}, {1024});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024}, {1024});
    gert::StorageShape amax_shape({1}, {1});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1}, {1, 1})
        .InputShapes({&x_shape, &scale_shape})
        .OutputShapes({&y_shape, &amax_shape})
        .NodeAttrs({
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("round")},  // Invalid: FLOAT8_E5M2 should use "rint"
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
        })
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tiling_data.get())
        .Workspace(ws_size)
        .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

    std::cout << "QuantMax tiling check attrs failure for roundMode mismatch" << std::endl;
}