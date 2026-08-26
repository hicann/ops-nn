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
 * \file test_bn_training_update_grad_tiling.cpp
 * \brief Tiling UT for BNTrainingUpdateGrad (arch35)。
 *        单路径 tilingKey=0；覆盖正向切分（channel 均分 / N 维 nGroups 再切）、dtype 三态、
 *        epsilon 缺省（OPTIONAL，缺省 0.0001 须放行）、
 *        反向校验（dtype/format/rank/空 tensor/统计量元素数/grads≠x shape）。
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
#include "../../../../op_host/arch35/bn_training_update_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

class BNTrainingUpdateGradTilingUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BNTrainingUpdateGradTilingUT SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BNTrainingUpdateGradTilingUT TearDown" << std::endl; }
};

namespace {
constexpr const char* kOpType = "BNTrainingUpdateGrad";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 四输入(grads/x/batch_mean/batch_variance) 两输出(diff_scale/diff_offset)，format 固定 ND。
// hasEpsilon=false 时不下发 epsilon 属性（OPTIONAL 缺省 0.0001，须放行）。
ge::graphStatus RunTiling(const std::vector<int64_t>& gradsDims, int64_t c, uint64_t& tilingKey,
                          int64_t* blockDim = nullptr, size_t* workspaceSize = nullptr,
                          ge::DataType gradsDt = ge::DT_FLOAT, ge::DataType xDt = ge::DT_FLOAT,
                          ge::DataType statDt = ge::DT_FLOAT, ge::Format fmt = ge::FORMAT_ND, bool hasEpsilon = true,
                          const std::vector<int64_t>* xDims = nullptr)
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
    optiling::BNTrainingUpdateGradCompileInfo compile_info;

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    if (op_impl == nullptr || op_impl->tiling == nullptr || op_impl->tiling_parse == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = op_impl->tiling;
    auto tiling_parse_func = op_impl->tiling_parse;

    // 先跑 tiling_parse 填 compile_info（coreNum/ubSize 取自平台）
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::StorageShape grads = MakeShape(gradsDims);
    gert::StorageShape x = MakeShape(xDims == nullptr ? gradsDims : *xDims);
    gert::StorageShape stat = MakeShape({c});

    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs;
    if (hasEpsilon) {
        attrs.emplace_back("epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5f));
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType(kOpType)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&grads, &x, &stat, &stat})
                      .OutputShapes({&stat, &stat})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, gradsDt, fmt, fmt)
                      .NodeInputTd(1, xDt, fmt, fmt)
                      .NodeInputTd(2, statDt, fmt, fmt)
                      .NodeInputTd(3, statDt, fmt, fmt)
                      .NodeOutputTd(0, statDt, fmt, fmt)
                      .NodeOutputTd(1, statDt, fmt, fmt)
                      .NodeAttrs(attrs)
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
        if (blockDim != nullptr) {
            *blockDim = static_cast<int64_t>(tiling_context->GetBlockDim());
        }
        if (workspaceSize != nullptr) {
            *workspaceSize = *reinterpret_cast<const size_t*>(ws_size->GetData());
        }
    }
    return ret;
}
} // namespace

TEST_F(BNTrainingUpdateGradTilingUT, tiling_registered)
{
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);
    ASSERT_NE(op_impl->tiling_parse, nullptr);
}

// ---------------- 正向 ----------------

// channel 主切分：C=3 < 56 核 → channelCores=3，blockDim=3，无 workspace
TEST_F(BNTrainingUpdateGradTilingUT, accept_channel_split)
{
    uint64_t key = 0;
    int64_t blockDim = 0;
    size_t ws = 1;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, &blockDim, &ws), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
    EXPECT_EQ(blockDim, 3); // channelCores=3
    EXPECT_EQ(ws, 0U);      // 零核间通信，无 workspace
}

// channel 充足占满核：C=128 → channelCores=64（UT 假平台 AIV 64 核），blockDim=64
TEST_F(BNTrainingUpdateGradTilingUT, accept_channel_full_split)
{
    uint64_t key = 0;
    int64_t blockDim = 0;
    size_t ws = 1;
    EXPECT_EQ(RunTiling({4, 128, 7, 7}, 128, key, &blockDim, &ws), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
    EXPECT_EQ(blockDim, 64); // channelCores=min(128, coreNum=64)
    EXPECT_EQ(ws, 0U);
}

// rank2 最小形态（R==1 跳过归约路径）
TEST_F(BNTrainingUpdateGradTilingUT, accept_rank2)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 8}, 8, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// num==1（N=1 且 R=1）：归约退化为单元素取值，tiling 须放行
TEST_F(BNTrainingUpdateGradTilingUT, accept_num_equals_one)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 16, 1}, 16, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// R 极大触发 sliceR 分片（cLenCap=1、sliceR<R）
TEST_F(BNTrainingUpdateGradTilingUT, accept_huge_r_slice)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 64, 100000}, 64, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// 深 R 触发快路(cLenCap=1、sliceR 为 1D chunk;小 C 时 channelCores=C)
TEST_F(BNTrainingUpdateGradTilingUT, accept_deep_r_fast_path)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 3, 100000}, 3, key), ge::GRAPH_SUCCESS);          // C=3 深 R
    EXPECT_EQ(RunTiling({2, 24, 15, 128, 7, 3}, 24, key), ge::GRAPH_SUCCESS); // C=24 深 R
    EXPECT_EQ(RunTiling({1, 1, 65}, 1, key), ge::GRAPH_SUCCESS);              // 尾块路径(eff%64!=0)
}

// fp16 / bf16 grads+x（二者同型）
TEST_F(BNTrainingUpdateGradTilingUT, accept_fp16_bf16)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_FLOAT16, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_BF16, ge::DT_BF16), ge::GRAPH_SUCCESS);
}

// GE 图模式可能下发 NCHW 标签，须接受
TEST_F(BNTrainingUpdateGradTilingUT, accept_nchw_tag)
{
    uint64_t key = 0;
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_NCHW),
        ge::GRAPH_SUCCESS);
}

// epsilon 为 OPTIONAL 属性（缺省 0.0001，对齐 A2 proto），缺失须放行
TEST_F(BNTrainingUpdateGradTilingUT, accept_missing_epsilon_default)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND,
                        false),
              ge::GRAPH_SUCCESS);
}

// ---------------- 反向：非法输入必须被拦截 ----------------

// grads dtype 非法（int32）
TEST_F(BNTrainingUpdateGradTilingUT, reject_invalid_grads_dtype)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_INT32, ge::DT_INT32), ge::GRAPH_FAILED);
}

// x 与 grads dtype 不同型
TEST_F(BNTrainingUpdateGradTilingUT, reject_x_dtype_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_FLOAT, ge::DT_FLOAT16), ge::GRAPH_FAILED);
}

// 统计量 dtype 非 fp32
TEST_F(BNTrainingUpdateGradTilingUT, reject_invalid_stat_dtype)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// format 非 ND/NCHW
TEST_F(BNTrainingUpdateGradTilingUT, reject_invalid_format)
{
    uint64_t key = 0;
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_NHWC),
        ge::GRAPH_FAILED);
}

// rank < 2
TEST_F(BNTrainingUpdateGradTilingUT, reject_rank1)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({16}, 1, key), ge::GRAPH_FAILED);
}

// 空 tensor 逐轴：N=0 / C=0 / R 轴=0 / 多轴同 0（归约语义，A2 同样不支持）
TEST_F(BNTrainingUpdateGradTilingUT, reject_empty_tensor)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({0, 3, 4, 5}, 3, key), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 0, 4, 5}, 0, key), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 0, 5}, 3, key), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({0, 0, 0}, 0, key), ge::GRAPH_FAILED);
}

// 统计量元素数必须等于 C
TEST_F(BNTrainingUpdateGradTilingUT, reject_stat_numel_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 4, key), ge::GRAPH_FAILED);
}

// x shape 必须等于 grads shape
TEST_F(BNTrainingUpdateGradTilingUT, reject_x_shape_mismatch)
{
    uint64_t key = 0;
    std::vector<int64_t> xDims = {2, 3, 4, 6};
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND,
                        true, &xDims),
              ge::GRAPH_FAILED);
}
