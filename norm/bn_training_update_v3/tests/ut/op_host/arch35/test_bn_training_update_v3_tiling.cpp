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
 * \file test_bn_training_update_v3_tiling.cpp
 * \brief Tiling UT for BNTrainingUpdateV3 (arch35)。
 *        单路径 tilingKey=0；覆盖正向切分（plane 均分 / inner 再切）、dtype 三态、
 *        反向校验（dtype/format/rank/空 tensor/统计量元素数/epsilon 缺失）。
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
#include "../../../../op_host/arch35/bn_training_update_v3_tiling_arch35.h"

using namespace std;
using namespace ge;

class BNTrainingUpdateV3TilingUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BNTrainingUpdateV3TilingUT SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BNTrainingUpdateV3TilingUT TearDown" << std::endl; }
};

namespace {
constexpr const char* kOpType = "BNTrainingUpdateV3";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 五输入(x/sum/square_sum/scale/offset) 五输出(y/batch_mean/batch_variance/reserve_1/reserve_2)，format 固定 ND。
// hasEpsilon=false 时不下发 epsilon 属性（REQUIRED 缺失须被拒）。
ge::graphStatus RunTiling(const std::vector<int64_t>& xDims, int64_t c, uint64_t& tilingKey,
                          int64_t* blockDim = nullptr, ge::DataType xDt = ge::DT_FLOAT,
                          ge::DataType statDt = ge::DT_FLOAT, ge::Format fmt = ge::FORMAT_ND, bool hasEpsilon = true)
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
    optiling::BNTrainingUpdateV3CompileInfo compile_info;

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

    gert::StorageShape x = MakeShape(xDims);
    gert::StorageShape stat = MakeShape({c});
    gert::StorageShape y = MakeShape(xDims);

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
                      .NodeIoNum(5, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&x, &stat, &stat, &stat, &stat})
                      .OutputShapes({&y, &stat, &stat, &stat, &stat})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDt, fmt, fmt)
                      .NodeInputTd(1, statDt, fmt, fmt)
                      .NodeInputTd(2, statDt, fmt, fmt)
                      .NodeInputTd(3, statDt, fmt, fmt)
                      .NodeInputTd(4, statDt, fmt, fmt)
                      .NodeOutputTd(0, xDt, fmt, fmt)
                      .NodeOutputTd(1, statDt, fmt, fmt)
                      .NodeOutputTd(2, statDt, fmt, fmt)
                      .NodeOutputTd(3, statDt, fmt, fmt)
                      .NodeOutputTd(4, statDt, fmt, fmt)
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
    }
    return ret;
}
} // namespace

TEST_F(BNTrainingUpdateV3TilingUT, tiling_registered)
{
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);
    ASSERT_NE(op_impl->tiling_parse, nullptr);
}

// ---------------- 正向 ----------------

// plane 均分：units=6 ≤ 核数、R=20 < VL → inner 不切；tilingKey=0
TEST_F(BNTrainingUpdateV3TilingUT, accept_basic_plane_split)
{
    uint64_t key = 0;
    int64_t blockDim = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, &blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
    EXPECT_EQ(blockDim, 6); // units=N*C=6，inner 不切
}

// plane 不足且 R 够大 → inner 维再切：units=2，innerCores=32，blockDim=64
TEST_F(BNTrainingUpdateV3TilingUT, accept_inner_split)
{
    uint64_t key = 0;
    int64_t blockDim = 0;
    EXPECT_EQ(RunTiling({1, 2, 10000}, 2, key, &blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
    EXPECT_EQ(blockDim, 64); // unitCores=2 × innerCores=32
}

// rank2 最小形态
TEST_F(BNTrainingUpdateV3TilingUT, accept_rank2)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 8}, 8, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// num==1（N=1 且 R=1）：batchVarScaler=0.0 特判路径，tiling 须放行
TEST_F(BNTrainingUpdateV3TilingUT, accept_num_equals_one)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 16, 1}, 16, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0U);
}

// fp16 / bf16 x
TEST_F(BNTrainingUpdateV3TilingUT, accept_fp16_bf16)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, ge::DT_BF16), ge::GRAPH_SUCCESS);
}

// GE 图模式可能下发 NCHW 标签，须接受
TEST_F(BNTrainingUpdateV3TilingUT, accept_nchw_tag)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_NCHW), ge::GRAPH_SUCCESS);
}

// ---------------- 反向：非法输入必须被拦截 ----------------

// x dtype 非法（int32）
TEST_F(BNTrainingUpdateV3TilingUT, reject_invalid_x_dtype)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, ge::DT_INT32), ge::GRAPH_FAILED);
}

// 统计量 dtype 非 fp32
TEST_F(BNTrainingUpdateV3TilingUT, reject_invalid_stat_dtype)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, ge::DT_FLOAT, ge::DT_FLOAT16), ge::GRAPH_FAILED);
}

// format 非 ND/NCHW
TEST_F(BNTrainingUpdateV3TilingUT, reject_invalid_format)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_NHWC), ge::GRAPH_FAILED);
}

// rank < 2
TEST_F(BNTrainingUpdateV3TilingUT, reject_rank1)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({16}, 1, key), ge::GRAPH_FAILED);
}

// 空 tensor 逐轴：N=0 / C=0 / R 轴=0 / 多轴同 0（N*R 为分母，A2 同样不支持）
TEST_F(BNTrainingUpdateV3TilingUT, reject_empty_tensor)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({0, 3, 4, 5}, 3, key), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 0, 4, 5}, 0, key), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 0, 5}, 3, key), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({0, 0, 0}, 0, key), ge::GRAPH_FAILED);
}

// 统计量元素数必须等于 C
TEST_F(BNTrainingUpdateV3TilingUT, reject_stat_numel_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 4, key), ge::GRAPH_FAILED);
}

// epsilon 为 REQUIRED 属性，缺失须拒
TEST_F(BNTrainingUpdateV3TilingUT, reject_missing_epsilon)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, 3, key, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND, false),
              ge::GRAPH_FAILED);
}
