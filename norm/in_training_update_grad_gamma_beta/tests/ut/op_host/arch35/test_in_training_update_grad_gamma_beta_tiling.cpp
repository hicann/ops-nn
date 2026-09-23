/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_in_training_update_grad_gamma_beta_tiling.cpp
 * \brief Tiling UT for INTrainingUpdateGradGammaBeta on Ascend 950.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
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
#include "../../../../op_host/arch35/in_training_update_grad_gamma_beta_tiling_arch35.h"
#include "../../../../op_kernel/arch35/in_training_update_grad_gamma_beta_tiling_data.h"

using namespace std;
using namespace ge;

class INTrainingUpdateGradGammaBetaTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingUpdateGradGammaBetaTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingUpdateGradGammaBetaTiling TearDown" << std::endl; }
};

namespace {
constexpr const char* kOpType = "INTrainingUpdateGradGammaBeta";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 两输入(res_gamma/res_beta) 两输出(pd_gamma/pd_beta)；输出 shape 默认 = 输入 dim0 置 1。
// tilingKey 仅在返回 GRAPH_SUCCESS 时被写入。
ge::graphStatus RunTiling(const std::vector<int64_t>& inDims, uint64_t& tilingKey,
                          const std::vector<int64_t>& outDims = {}, ge::DataType dt = ge::DT_FLOAT,
                          ge::Format fmt = ge::FORMAT_NCHW, const std::vector<int64_t>& betaDims = {},
                          ge::Format betaFmt = ge::FORMAT_RESERVED, ge::Format originFmt = ge::FORMAT_RESERVED,
                          INTrainingUpdateGradGammaBetaTilingData* tilingData = nullptr)
{
    if (betaFmt == ge::FORMAT_RESERVED) {
        betaFmt = fmt; // 未显式指定 res_beta format 时与 res_gamma 同 format
    }
    if (originFmt == ge::FORMAT_RESERVED) {
        originFmt = fmt; // 未显式指定 origin format 时与 storage format 相同
    }
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
    optiling::INTrainingUpdateGradGammaBetaCompileInfo compile_info;

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

    gert::StorageShape gamma = MakeShape(inDims);
    gert::StorageShape beta = MakeShape(betaDims.empty() ? inDims : betaDims);
    std::vector<int64_t> expected = inDims;
    expected[0] = 1;
    gert::StorageShape pdGamma = MakeShape(outDims.empty() ? expected : outDims);
    gert::StorageShape pdBeta = MakeShape(outDims.empty() ? expected : outDims);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(kOpType) // 走 TilingRegistry 模板注册表，必须设置 op type 才能命中模板
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gamma, &beta})
                      .OutputShapes({&pdGamma, &pdBeta})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dt, originFmt, fmt)
                      .NodeInputTd(1, dt, originFmt, betaFmt)
                      .NodeOutputTd(0, ge::DT_FLOAT, fmt, fmt)
                      .NodeOutputTd(1, ge::DT_FLOAT, fmt, fmt)
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
        if (tilingData != nullptr) {
            std::memcpy(tilingData, tiling_context->GetRawTilingData()->GetData(), sizeof(*tilingData));
        }
    }
    return ret;
}
} // namespace

TEST_F(INTrainingUpdateGradGammaBetaTiling, in_training_update_grad_gamma_beta_tiling_registered)
{
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);
    ASSERT_NE(op_impl->tiling_parse, nullptr);
}

// ---------------- 正向路径 ----------------

// 常规 4 维 NCHW：A 方向分核，R 单轴全驻 -> base
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_base_0)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// 5 维 NCDHW：合轴后 tail-A，R 全驻 -> base
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_base_0_5d)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5, 6}, key, {}, ge::DT_FLOAT, ge::FORMAT_NCDHW), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// ND 稠密别名：与公有格式同走 base
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_base_0_nd)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key, {}, ge::DT_FLOAT, ge::FORMAT_ND), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// NHWC 4 维
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_base_0_nhwc)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 3, 5, 64}, key, {}, ge::DT_FLOAT, ge::FORMAT_NHWC), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// N=1 退化（纯 A 路径，补 R 增广）
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_single_n)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 64, 3, 5}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// 大 N、小输出仍走确定性的逐输出归约，不使用用户 workspace。
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_large_n_reduce_0)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({100000, 8, 1, 1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// A power-of-two magnitude envelope leaves one exponent bit of accumulation headroom.
// Very large N is a host-only check; no tensors are allocated here.
TEST_F(INTrainingUpdateGradGammaBetaTiling, safe_magnitude_and_scale_boundaries)
{
    const std::vector<int64_t> counts{1, 2, 3, 4095, 4096, 4097, 10000, 16777217, 4294967297LL};
    for (const auto count : counts) {
        SCOPED_TRACE(count);
        uint64_t key = 0;
        INTrainingUpdateGradGammaBetaTilingData data{};
        ASSERT_EQ(RunTiling({count, 8, 1, 1}, key, {}, ge::DT_FLOAT, ge::FORMAT_NCHW, {}, ge::FORMAT_RESERVED,
                            ge::FORMAT_RESERVED, &data),
                  ge::GRAPH_SUCCESS);
        EXPECT_EQ(data.reduceCount, count);
        EXPECT_EQ(key, 0U);
        EXPECT_FLOAT_EQ(data.inputScale * data.outputScale, 1.0F);
        if (count == 1) {
            EXPECT_FLOAT_EQ(data.safeMagnitude, std::numeric_limits<float>::max());
            EXPECT_FLOAT_EQ(data.inputScale, 1.0F);
            continue;
        }
        int exponent = 0;
        EXPECT_FLOAT_EQ(std::frexp(data.safeMagnitude, &exponent), 0.5F);
        EXPECT_LE(static_cast<double>(data.safeMagnitude) * static_cast<double>(count), std::ldexp(1.0, 127));
        EXPECT_FLOAT_EQ(data.safeMagnitude, std::ldexp(data.inputScale, 128));
    }
}

TEST_F(INTrainingUpdateGradGammaBetaTiling, expansion_state_fits_ub_with_vector_tails)
{
    for (const int64_t rows : {0, 1, 2, 7, 13, 14, 8193}) {
        for (const int64_t columns : {1, 7, 8, 13, 65, 400001, 1000000}) {
            SCOPED_TRACE(rows);
            SCOPED_TRACE(columns);
            uint64_t key = 0;
            INTrainingUpdateGradGammaBetaTilingData data{};
            ASSERT_EQ(RunTiling({rows, columns, 1, 1}, key, {}, ge::DT_FLOAT, ge::FORMAT_NCHW, {}, ge::FORMAT_RESERVED,
                                ge::FORMAT_RESERVED, &data),
                      ge::GRAPH_SUCCESS);
            const uint64_t partialCount = std::min(
                std::max(rows, int64_t{1}), static_cast<int64_t>(IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_MAX_PARTIALS));
            const uint64_t stateStride = (static_cast<uint64_t>(data.tileElements) + 63U) / 64U * 64U;
            const uint64_t stateBufferCount = rows <= 2 ? 1U : std::max(partialCount, uint64_t{4}) + 2U;
            const uint64_t stateBytes = stateBufferCount * stateStride * sizeof(float);
            const uint64_t inputBytes = static_cast<uint64_t>(data.reduceRowsPerTile) * data.tileElements *
                                            sizeof(float) +
                                        (64U - data.blockElements) * sizeof(float);
            EXPECT_LE(stateBytes + inputBytes + 4096U, 245760U);
            EXPECT_GE(data.reduceRowsPerTile, 1U);
            EXPECT_EQ(data.tileElements % data.blockElements, 0U);
            EXPECT_EQ(key, 0U);
        }
    }
}

TEST_F(INTrainingUpdateGradGammaBetaTiling, two_rows_share_one_wide_input_tile)
{
    uint64_t key = 0;
    INTrainingUpdateGradGammaBetaTilingData data{};
    ASSERT_EQ(RunTiling({2, 800003, 1, 1}, key, {}, ge::DT_FLOAT, ge::FORMAT_NCHW, {}, ge::FORMAT_RESERVED,
                        ge::FORMAT_RESERVED, &data),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(data.reduceRowsPerTile, 2U);
    EXPECT_EQ(key, 0U);
}

// N=0（R=0 且 A 全 >0）-> empty（空集求和，输出全 0）
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_zero_reduce_empty)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({0, 64, 1, 1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// 非规约维为 0 -> empty（输出空张量）
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_zero_output_empty)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 0, 1, 1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// ---------------- 反向：非法输入必须被校验拒绝 ----------------

// dtype 非 fp32
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_non_fp32_input)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key, {}, ge::DT_FLOAT16), ge::GRAPH_FAILED);
}

// format 不在支持列表（FRACTAL_NZ）
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_fractal_nz_format)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key, {}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ), ge::GRAPH_FAILED);
}

// 两输入 format 不一致
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_mixed_input_formats)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key, {}, ge::DT_FLOAT, ge::FORMAT_NCHW, {}, ge::FORMAT_NHWC), ge::GRAPH_FAILED);
}

// rank 不是 4/5（3 维）
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_rank_three)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1}, key), ge::GRAPH_FAILED);
}

// rank 不是 4/5（6 维）
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_rank_six)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 2, 3, 4, 5, 6}, key), ge::GRAPH_FAILED);
}

// 两输入 shape 不一致（不支持广播）
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_shape_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key, {}, ge::DT_FLOAT, ge::FORMAT_NCHW, {4, 32, 1, 1}), ge::GRAPH_FAILED);
}

// 负维（动态 shape 下界防御）
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_negative_dim)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, -1, 1, 1}, key), ge::GRAPH_FAILED);
}

// 输出 shape 契约：dim0 != 1（keepdims 语义）
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_output_dim0_not_one)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key, {4, 64, 1, 1}), ge::GRAPH_FAILED);
}

// 输出 shape 契约：非规约维与输入不一致
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_output_shape_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 64, 1, 1}, key, {1, 32, 1, 1}), ge::GRAPH_FAILED);
}

// 全 1 A 维退化形态：合轴后 [A=1, R=N]（tail-R），树链 + K=ceil(log2(N)) 预缩放路径
TEST_F(INTrainingUpdateGradGammaBetaTiling, tilingkey_tail_r_all_ones_a)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({3, 1, 1, 1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// origin format 与具体 rank 配对：unknown-rank 输入具体化为 5 维但声明 NCHW → 端到端拒绝
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_origin_format_rank_pairing)
{
    uint64_t key = 0;
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5, 6}, key, {}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_RESERVED, ge::FORMAT_NCHW),
        ge::GRAPH_FAILED);
}

// 输入元素总数及字节数必须在 signed int64 GM 偏移范围内。
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_input_byte_size_overflow)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({std::numeric_limits<int64_t>::max(), 2, 1, 1}, key), ge::GRAPH_FAILED);
}

// 非规约维连乘必须执行 checked arithmetic，不能静默回绕。
TEST_F(INTrainingUpdateGradGammaBetaTiling, reject_output_element_count_overflow)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, std::numeric_limits<int64_t>::max(), 2, 1}, key), ge::GRAPH_FAILED);
}
