/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_in_training_update_grad_tiling.cpp
 * \brief Tiling UT for INTrainingUpdateGrad (arch35)。
 *        TilingKey：reduce-empty 50000，full-load 100000，stream 200000。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/in_training_update_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

class INTrainingUpdateGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingUpdateGradTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingUpdateGradTiling TearDown" << std::endl; }
};

namespace {
constexpr const char* kOpType = "INTrainingUpdateGrad";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 四输入(dy/x/variance/mean) 两输出(res_gamma/res_beta)，format 固定 NDC1HWC0。
// tilingKey 仅在返回 GRAPH_SUCCESS 时被写入。
ge::graphStatus RunTiling(const std::vector<int64_t>& dyDims, const std::vector<int64_t>& paramDims,
                          uint64_t& tilingKey, ge::DataType dyDt = ge::DT_FLOAT, ge::DataType xDt = ge::DT_FLOAT,
                          ge::DataType paramDt = ge::DT_FLOAT, ge::DataType outDt = ge::DT_FLOAT,
                          ge::Format fmt = ge::FORMAT_NDC1HWC0)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                              "Intrinsic_fix_pipe_l0c2out": false,
                              "Intrinsic_data_move_l12ub": true,
                              "Intrinsic_data_move_l0c2ub": true,
                              "Intrinsic_data_move_out2l1_nd2nz": false,
                              "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                              "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // tiling 从 CompileInfo 取平台参数，按 Ascend950 规格填充。
    optiling::InTrainingUpdateGradCompileInfo compile_info;
    compile_info.coreNum = 64U;
    compile_info.ubSize = 245760U;
    compile_info.vectorLength = 256U; // VRegSize，单位字节
    compile_info.ubBlockSize = 32U;

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    if (op_impl == nullptr || op_impl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = op_impl->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    gert::StorageShape dy = MakeShape(dyDims);
    gert::StorageShape x = MakeShape(dyDims);
    gert::StorageShape var = MakeShape(paramDims);
    gert::StorageShape mean = MakeShape(paramDims);
    gert::StorageShape resGamma = MakeShape(paramDims);
    gert::StorageShape resBeta = MakeShape(paramDims);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(kOpType) // 走 TilingRegistry 模板注册表，必须设置 op type 才能命中模板
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&dy, &x, &var, &mean})
                      .OutputShapes({&resGamma, &resBeta})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dyDt, fmt, fmt)
                      .NodeInputTd(1, xDt, fmt, fmt)
                      .NodeInputTd(2, paramDt, fmt, fmt)
                      .NodeInputTd(3, paramDt, fmt, fmt)
                      .NodeOutputTd(0, outDt, fmt, fmt)
                      .NodeOutputTd(1, outDt, fmt, fmt)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr || tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tiling_func(tiling_context);
    if (ret == ge::GRAPH_SUCCESS) {
        tilingKey = tiling_context->GetTilingKey();
    }
    return ret;
}

// [N, D, C1, H, W, C0] 与其对应的参数形状（空间维为 1）。
ge::graphStatus RunTiling6D(int64_t n, int64_t d, int64_t c1, int64_t h, int64_t w, int64_t c0, uint64_t& tilingKey,
                            ge::DataType dyDt = ge::DT_FLOAT)
{
    return RunTiling({n, d, c1, h, w, c0}, {n, 1, c1, 1, 1, c0}, tilingKey, dyDt, dyDt);
}
} // namespace

TEST_F(INTrainingUpdateGradTiling, in_training_update_grad_tiling_registered)
{
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);
    ASSERT_NE(op_impl->tiling_parse, nullptr);
}

// ---------------- 正向：三条模板路径 ----------------

// 归约长度小，UB 放得下 -> full load
TEST_F(INTrainingUpdateGradTiling, tilingkey_full_load_100000)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(2, 2, 2, 8, 8, 16, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 100000U);
}

// 归约长度大，UB 放不下 -> stream
TEST_F(INTrainingUpdateGradTiling, tilingkey_stream_200000)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(1, 64, 1, 128, 128, 16, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 200000U);
}

// 归约轴出现 0（D=0）-> reduce empty
TEST_F(INTrainingUpdateGradTiling, tilingkey_reduce_empty_50000)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(2, 0, 2, 8, 8, 16, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 50000U);
}

// H=0 同样触发空归约
TEST_F(INTrainingUpdateGradTiling, tilingkey_reduce_empty_zero_h)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(2, 2, 2, 0, 8, 16, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 50000U);
}

// dy/x 为 fp16（参数与输出仍为 fp32）
TEST_F(INTrainingUpdateGradTiling, accept_fp16_input)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(2, 2, 2, 8, 8, 16, key, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 100000U);
}

// ---------------- 反向：非法输入必须被拦截 ----------------

// dy 的 dtype 非 fp16/fp32
TEST_F(INTrainingUpdateGradTiling, reject_invalid_dy_dtype)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(2, 2, 2, 8, 8, 16, key, ge::DT_INT32), ge::GRAPH_FAILED);
}

// dy 与 x 的 dtype 不一致
TEST_F(INTrainingUpdateGradTiling, reject_dtype_mismatch_dy_x)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 2, 8, 8, 16}, {2, 1, 2, 1, 1, 16}, key, ge::DT_FLOAT, ge::DT_FLOAT16), ge::GRAPH_FAILED);
}

// variance/mean 必须是 fp32
TEST_F(INTrainingUpdateGradTiling, reject_non_fp32_param)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 2, 8, 8, 16}, {2, 1, 2, 1, 1, 16}, key, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// 两个输出必须是 fp32
TEST_F(INTrainingUpdateGradTiling, reject_non_fp32_output)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 2, 8, 8, 16}, {2, 1, 2, 1, 1, 16}, key, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                        ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// dy 维数不是 6
TEST_F(INTrainingUpdateGradTiling, reject_rank_not_six)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 8, 8, 16}, {2, 1, 1, 1, 16}, key), ge::GRAPH_FAILED);
}

// N / C1 / C0 必须为正
TEST_F(INTrainingUpdateGradTiling, reject_non_positive_n)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(0, 2, 2, 8, 8, 16, key), ge::GRAPH_FAILED);
}

TEST_F(INTrainingUpdateGradTiling, reject_non_positive_c0)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling6D(2, 2, 2, 8, 8, 0, key), ge::GRAPH_FAILED);
}

// format 必须是 NDC1HWC0：ND 与之布局不同，当成 NDC1HWC0 计算会出错，须拦截
TEST_F(INTrainingUpdateGradTiling, reject_non_ndc1hwc0_format)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 2, 8, 8, 16}, {2, 1, 2, 1, 1, 16}, key, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                        ge::DT_FLOAT, ge::FORMAT_ND),
              ge::GRAPH_FAILED);
}
