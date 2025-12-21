/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "array_ops.h"
#include "kernel_run_context_facker.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;

class FusedLinearCrossEntropyLossGradTiling : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "FusedLinearCrossEntropyLossGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "FusedLinearCrossEntropyLossGradTiling TearDown" << std::endl;
    }
};

void TilingTest(std::string opName, bool isMemFriendly,  // false: high-perf, true: mem
                std::initializer_list<int64_t>& input, std::initializer_list<int64_t>& weight,
                std::initializer_list<int64_t>& softmax, std::initializer_list<int64_t>& target_mask,
                std::initializer_list<int64_t>& masked_target, std::initializer_list<int64_t>& grad_output,
                std::initializer_list<int64_t>& logits_max, std::initializer_list<int64_t>& sum_exp_logits,
                std::initializer_list<int64_t>& grad_input, std::initializer_list<int64_t>& grad_weight,
                ge::DataType dataTypeFp, ge::DataType dataTypeInt, ge::DataType dataTypeBool,
                const ge::graphStatus status) {
    std::string op_type(opName);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    // auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string compile_info_string = R"({
    "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, 
    "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, 
    "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, 
    "CORE_NUM": 24}
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FusedLinearCrossEntropyLossGradCompileInfo {};
    FusedLinearCrossEntropyLossGradCompileInfo compile_info;
    // tilingParseFunc simulate，没有实质tilingParse的可以跳过
    // auto kernel_holder =
    //     gert::KernelRunContextFaker()
    //         .KernelIONum(6, 2)
    //         .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
    //         .Outputs({&compile_info})
    //         .Build();
    // ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    // kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    // kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    // kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    // kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
    //                                                                                         intrinsics);
    // ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    // workspace_size_holer不可太大
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    
    gert::StorageShape input_ss = {input, input};
    gert::StorageShape weight_ss = {weight, weight};
    gert::StorageShape softmax_ss = {softmax, softmax};
    gert::StorageShape target_mask_ss = {target_mask, target_mask};
    gert::StorageShape masked_target_ss = {masked_target, masked_target};
    gert::StorageShape grad_output_ss = {grad_output, grad_output};
    gert::StorageShape logits_max_ss = {logits_max, logits_max};
    gert::StorageShape sum_exp_logits_ss = {sum_exp_logits, sum_exp_logits};

    gert::StorageShape grad_input_ss = {grad_input, grad_input};
    gert::StorageShape grad_weight_ss = {grad_weight, grad_weight};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
        .SetOpType(op_type)
        .NodeIoNum(8, 2)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})  // 输入数量个"1"
        .InputShapes({&grad_output_ss, &input_ss, &weight_ss, &target_mask_ss, &masked_target_ss,
                      (isMemFriendly) ? &logits_max_ss : nullptr,
                      (isMemFriendly) ? &sum_exp_logits_ss : nullptr,
                      (isMemFriendly) ? nullptr : &softmax_ss})
        .OutputShapes({&grad_input_ss, &grad_weight_ss})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, dataTypeFp, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, dataTypeFp, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, dataTypeBool, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, dataTypeInt, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, dataTypeFp, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, dataTypeFp, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_EQ(tiling_func(tiling_context), status);
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_fp16_int32_bool_success) {
    int64_t bt = 8192;
    int64_t h = 4096;
    int64_t v = 19392;
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_bf16_int32_bool_success) {
    int64_t bt = 8192;
    int64_t h = 4096;
    int64_t v = 19392;
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_BF16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_bt_gt_boundary_succcess) {
    int64_t bt = 32768;
    int64_t h = 4096;
    int64_t v = 19392;
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_bt_gt_65536_failed) {
    int64_t bt = 77568;
    int64_t h = 4096;
    int64_t v = 19392;
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_v_lt_512B_failed) {
    int64_t bt = 8192;
    int64_t h = 4096;
    int64_t v = 100;
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_v_gt_65280_failed) {
    int64_t bt = 8192;
    int64_t h = 4096;
    int64_t v = 77568;
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_h_gt_65535_failed) {
    int64_t bt = 8192;
    int64_t h = 77568;
    int64_t v = 19392;
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_tiling_partial_branch_not_implement_failed) {
    int64_t bt = 8192;
    int64_t h = 4096;
    int64_t v = 32768;  // 196608->24572
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> softmax = {bt, v};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", false,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_mem_friendly_tiling_fp16_int32_bool_success) {
    int64_t bt = 8192;
    int64_t h = 4096;
    int64_t v = 19392;
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};
    std::initializer_list<int64_t> softmax = {bt, v};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", true,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_mem_friendly_tiling_bf16_int32_bool_success) {
    int64_t bt = 8192;
    int64_t h = 4096;
    int64_t v = 19392;
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};
    std::initializer_list<int64_t> softmax = {bt, v};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", true,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_BF16, ge::DT_INT32, ge::DT_BOOL, status
    );
}

TEST_F(FusedLinearCrossEntropyLossGradTiling, fused_linear_cross_entropy_loss_grad_mem_friendly_tiling_big_H_success) {
    int64_t bt = 8192;
    int64_t h = 65535;
    int64_t v = 19392;
    std::initializer_list<int64_t> grad_output = {bt};
    std::initializer_list<int64_t> input = {bt, h};
    std::initializer_list<int64_t> weight = {v, h};
    std::initializer_list<int64_t> target_mask = {bt};
    std::initializer_list<int64_t> masked_target = {bt};
    std::initializer_list<int64_t> logits_max = {bt};
    std::initializer_list<int64_t> sum_exp_logits = {bt};
    std::initializer_list<int64_t> softmax = {bt, v};

    std::initializer_list<int64_t> grad_input = {bt, h};
    std::initializer_list<int64_t> grad_weight = {v, h};

    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest(
        "FusedLinearCrossEntropyLossGrad", true,
        input, weight, softmax, target_mask, masked_target, grad_output, logits_max, sum_exp_logits, grad_input, grad_weight,
        ge::DT_BF16, ge::DT_INT32, ge::DT_BOOL, status
    );
}