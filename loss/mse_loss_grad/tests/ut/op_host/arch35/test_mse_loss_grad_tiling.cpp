/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/mse_loss_grad_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
// GET_TPL_TILING_KEY(schMode, doutIsScalar): schMode occupies bit 0..15, doutIsScalar occupies bit 16.
constexpr uint64_t DOUT_IS_SCALAR_MASK = 1UL << 16;
} // namespace

class MseLossGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MseLossGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MseLossGradTiling TearDown" << std::endl; }
};

// Drives one MseLossGrad arch35 tiling invocation on the Ascend950 platform.
// predict/label share the same storage shape; dout can be either a full tensor (doutDims empty ->
// same as predict) or a dedicated shape such as {1} for the scalar path. On success the resulting
// tiling key / block dim are returned for assertions.
static ge::graphStatus RunMseLossGradTiling(gert::StorageShape& predictShape, gert::StorageShape& labelShape,
                                            gert::StorageShape& doutShape, gert::StorageShape& outputShape,
                                            ge::DataType dtype, ge::DataType labelDtype, ge::DataType doutDtype,
                                            ge::DataType outputDtype, const std::string& reduction,
                                            uint64_t& outTilingKey, uint32_t& outBlockDim)
{
    std::string op_type("MseLossGrad");
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    EXPECT_NE(op_impl, nullptr);
    if (op_impl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = op_impl->tiling;
    auto tiling_parse_func = op_impl->tiling_parse;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version_infos);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::MseLossGradCompileInfo compile_info;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    EXPECT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    EXPECT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    EXPECT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, labelDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, doutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto status = tiling_func(tiling_context);
    if (status == ge::GRAPH_SUCCESS) {
        outTilingKey = tiling_context->GetTilingKey();
        outBlockDim = tiling_context->GetBlockDim();
    }
    return status;
}

// fp32, tensor dout, reduction=mean: DoTensorDagOpTiling path, doutIsScalar bit stays 0.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_fp32_tensor_dout_mean)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_key & DOUT_IS_SCALAR_MASK, 0U);
    EXPECT_GT(block_dim, 0U);
}

// fp16, tensor dout, reduction=sum.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_fp16_tensor_dout_sum)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                   ge::DT_FLOAT16, ge::DT_FLOAT16, "sum", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_key & DOUT_IS_SCALAR_MASK, 0U);
    EXPECT_GT(block_dim, 0U);
}

// bf16, tensor dout, reduction=none.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_bf16_tensor_dout_none)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_BF16, ge::DT_BF16,
                                   ge::DT_BF16, ge::DT_BF16, "none", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_key & DOUT_IS_SCALAR_MASK, 0U);
    EXPECT_GT(block_dim, 0U);
}

// fp32, scalar dout (shape {1}): DoScalarDagOpTiling path, doutIsScalar bit is set in the tiling key.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_fp32_scalar_dout)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape dout_shape = {{1}, {1}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, dout_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_NE(tiling_key & DOUT_IS_SCALAR_MASK, 0U);
    EXPECT_GT(block_dim, 0U);
}

// fp16, scalar dout, reduction=sum.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_fp16_scalar_dout)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape dout_shape = {{1}, {1}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, dout_shape, output_shape, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                   ge::DT_FLOAT16, ge::DT_FLOAT16, "sum", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_NE(tiling_key & DOUT_IS_SCALAR_MASK, 0U);
    EXPECT_GT(block_dim, 0U);
}

// bf16, scalar dout, reduction=none.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_bf16_scalar_dout)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape dout_shape = {{1}, {1}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, dout_shape, output_shape, ge::DT_BF16, ge::DT_BF16,
                                   ge::DT_BF16, ge::DT_BF16, "none", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_NE(tiling_key & DOUT_IS_SCALAR_MASK, 0U);
    EXPECT_GT(block_dim, 0U);
}

// scalar predict/label with reduction=mean: EnsureNotScalar maps the scalar to {1}, dimVal=1.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_scalar_input_mean)
{
    gert::StorageShape input_shape = {{}, {}};
    gert::StorageShape dout_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, dout_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_NE(tiling_key & DOUT_IS_SCALAR_MASK, 0U); // scalar dout also takes the scalar dag path
    EXPECT_GT(block_dim, 0U);
}

// 4D input exercises the multi-dim broadcast DAG path.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_4d_tensor_dout)
{
    gert::StorageShape input_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape output_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_key & DOUT_IS_SCALAR_MASK, 0U);
    EXPECT_GT(block_dim, 0U);
}

// predict/label dtype mismatch is rejected.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_label_dtype_mismatch_failed)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT16,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", tiling_key, block_dim),
              ge::GRAPH_FAILED);
}

// predict/dout dtype mismatch is rejected.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_dout_dtype_mismatch_failed)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT16, ge::DT_FLOAT, "mean", tiling_key, block_dim),
              ge::GRAPH_FAILED);
}

// predict/output dtype mismatch is rejected.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_output_dtype_mismatch_failed)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT16, "mean", tiling_key, block_dim),
              ge::GRAPH_FAILED);
}

// unsupported predict dtype (int32) is rejected.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_unsupported_dtype_failed)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_INT32, ge::DT_INT32,
                                   ge::DT_INT32, ge::DT_INT32, "mean", tiling_key, block_dim),
              ge::GRAPH_FAILED);
}

// invalid reduction string is rejected.
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_invalid_reduction_failed)
{
    gert::StorageShape input_shape = {{182, 4}, {182, 4}};
    gert::StorageShape output_shape = {{182, 4}, {182, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "avg", tiling_key, block_dim),
              ge::GRAPH_FAILED);
}

// empty tensor with reduction=mean is rejected (dimVal would be 0).
TEST_F(MseLossGradTiling, mse_loss_grad_tiling_mean_empty_tensor_failed)
{
    gert::StorageShape input_shape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape output_shape = {{2, 0, 4}, {2, 0, 4}};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunMseLossGradTiling(input_shape, input_shape, input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", tiling_key, block_dim),
              ge::GRAPH_FAILED);
}
