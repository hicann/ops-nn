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
 * \file test_sgd_tiling.cpp
 * \brief SGD arch35 Tiling 单元测试
 *
 * 覆盖面：
 *   ① 6 个业务 TilingKey（K0~K5 = useNesterov × hasWeightDecay × hasDampening 的合法组合）× 3 dtype
 *   ② 非法组合 nesterov && dampening != 0 被 GetAttr 拦下
 *   ③ weight_decay < 0 被拦下
 *   ④ 本算子相对 910B/910C 补齐的校验：大张量不同形、标量 shape != [1]、dtype 不一致、空 tensor
 *
 * TilingKey 编码（4 位，见 sgd_tiling_key.h）：
 *   bit0 schMode(框架决定) / bit1 useNesterov / bit2 hasWeightDecay / bit3 hasDampening
 * 故期望 key = schMode | (nesterov<<1) | (hasWd<<2) | (hasDamp<<3)。
 * schMode 由 ElewiseBaseTiling 依 shape 自行决定，本 UT 用 GetTilingKey() 的高 3 位做断言，
 * 不硬编码 schMode，避免与框架实现耦合。
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "../../../../op_host/arch35/sgd_tiling.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestSgdTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TestSgdTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestSgdTiling TearDown" << std::endl; }
};

namespace {
constexpr size_t SGD_INPUT_NUM = 6;
constexpr size_t SGD_OUTPUT_NUM = 1;

void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                  map<string, string>& intrinsics, map<string, string>& socVersion)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    platFormInfo.Init();
}

struct SgdUtCompileInfo {};

// 通用 tiling 驱动。shapes 允许逐路不同，便于构造"不同形"负用例。
ge::graphStatus RunSgdTiling(gert::StorageShape& paramShape, gert::StorageShape& gradShape, gert::StorageShape& lrShape,
                             gert::StorageShape& accumShape, gert::StorageShape& momentumShape,
                             gert::StorageShape& statShape, ge::DataType paramDtype, ge::DataType otherDtype,
                             float dampening, float weightDecay, bool nesterov, uint64_t* outTilingKey)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("SGD");
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (impl == nullptr || impl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = impl->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }

    SgdUtCompileInfo compileInfo;
    auto inFormat = ge::FORMAT_ND;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(SGD_INPUT_NUM, SGD_OUTPUT_NUM)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramShape, &gradShape, &lrShape, &accumShape, &momentumShape, &statShape})
                      .OutputShapes({&paramShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, paramDtype, inFormat, inFormat)
                      .NodeInputTd(1, otherDtype, inFormat, inFormat)
                      .NodeInputTd(2, otherDtype, inFormat, inFormat)
                      .NodeInputTd(3, otherDtype, inFormat, inFormat)
                      .NodeInputTd(4, otherDtype, inFormat, inFormat)
                      .NodeInputTd(5, otherDtype, inFormat, inFormat)
                      .NodeOutputTd(0, paramDtype, inFormat, inFormat)
                      .NodeAttrs({{"dampening", Ops::NN::AnyValue::CreateFrom<float>(dampening)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom<float>(weightDecay)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(nesterov)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    auto ret = tiling_func(tiling_context);
    if (ret == ge::GRAPH_SUCCESS && outTilingKey != nullptr) {
        *outTilingKey = tiling_context->GetTilingKey();
    }
    return ret;
}

// 正常路径便捷封装：6 路同形同 dtype
ge::graphStatus RunSgdTilingNormal(std::initializer_list<int64_t> shape, ge::DataType dtype, float dampening,
                                   float weightDecay, bool nesterov, uint64_t* outTilingKey)
{
    gert::StorageShape big = {shape, shape};
    gert::StorageShape one = {{1}, {1}};
    return RunSgdTiling(big, big, one, big, one, big, dtype, dtype, dampening, weightDecay, nesterov, outTilingKey);
}

// TilingKey 的业务位（剥掉 bit0 的 schMode）：nesterov | hasWd<<1 | hasDamp<<2
uint64_t BizBits(uint64_t tilingKey) { return tilingKey >> 1; }
constexpr uint64_t BizExpect(bool nesterov, bool hasWd, bool hasDamp)
{
    return (nesterov ? 1U : 0U) | (hasWd ? 2U : 0U) | (hasDamp ? 4U : 0U);
}
} // namespace

// ───────────────────────── K0 ~ K5 × 3 dtype ─────────────────────────

TEST_F(TestSgdTiling, sgd_tiling_K0_no_branch_fp32)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({16, 26, 16, 19}, ge::DT_FLOAT, 0.0f, 0.0f, false, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(BizBits(key), BizExpect(false, false, false));
}

TEST_F(TestSgdTiling, sgd_tiling_K1_dampening_only_fp32)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({16, 26, 16, 19}, ge::DT_FLOAT, 0.5f, 0.0f, false, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(BizBits(key), BizExpect(false, false, true));
}

TEST_F(TestSgdTiling, sgd_tiling_K2_weight_decay_only_fp16)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({3761, 4, 44, 4}, ge::DT_FLOAT16, 0.0f, 0.01f, false, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(BizBits(key), BizExpect(false, true, false));
}

TEST_F(TestSgdTiling, sgd_tiling_K3_both_branches_fp16)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({3761, 4, 44, 4}, ge::DT_FLOAT16, 0.5f, 0.01f, false, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(BizBits(key), BizExpect(false, true, true));
}

TEST_F(TestSgdTiling, sgd_tiling_K4_nesterov_with_weight_decay_bf16)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({7, 2, 7, 8, 10}, ge::DT_BF16, 0.0f, 0.01f, true, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(BizBits(key), BizExpect(true, true, false));
}

TEST_F(TestSgdTiling, sgd_tiling_K5_nesterov_only_bf16)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({7, 2, 7, 8, 10}, ge::DT_BF16, 0.0f, 0.0f, true, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(BizBits(key), BizExpect(true, false, false));
}

// 三个 dtype 在同一分支下都要能出 tiling（binary.json 有 3 个 dtype 条目）
TEST_F(TestSgdTiling, sgd_tiling_K0_all_three_dtypes)
{
    uint64_t key = 0;
    for (auto dt : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        ASSERT_EQ(RunSgdTilingNormal({256, 256}, dt, 0.0f, 0.0f, false, &key), ge::GRAPH_SUCCESS);
        EXPECT_EQ(BizBits(key), BizExpect(false, false, false));
    }
}

// 尾块 / 非 32B 对齐 shape 也要能出 tiling
TEST_F(TestSgdTiling, sgd_tiling_unaligned_tail_shape)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({33, 2}, ge::DT_FLOAT, 0.0f, 0.0f, false, &key), ge::GRAPH_SUCCESS);
    ASSERT_EQ(RunSgdTilingNormal({7, 7, 33, 2}, ge::DT_FLOAT, 0.0f, 0.0f, false, &key), ge::GRAPH_SUCCESS);
}

// ───────────────────────── 属性非法组合 ─────────────────────────

TEST_F(TestSgdTiling, sgd_tiling_illegal_nesterov_with_dampening)
{
    // nesterov && dampening != 0 —— 被 GetAttr 拦下，且该组合不生成 binary
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({256, 256}, ge::DT_FLOAT, 0.5f, 0.0f, true, &key), ge::GRAPH_FAILED);
}

TEST_F(TestSgdTiling, sgd_tiling_illegal_negative_weight_decay)
{
    uint64_t key = 0;
    ASSERT_EQ(RunSgdTilingNormal({256, 256}, ge::DT_FLOAT, 0.0f, -0.01f, false, &key), ge::GRAPH_FAILED);
}

// ───────────── 本算子相对 910B/910C 补齐的校验（canndev 仅校验 parameters 的 rank）─────────────

TEST_F(TestSgdTiling, sgd_tiling_reject_rank0)
{
    gert::StorageShape zeroRank = {{}, {}};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(RunSgdTiling(zeroRank, zeroRank, one, zeroRank, one, zeroRank, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f,
                           false, nullptr),
              ge::GRAPH_FAILED);
}

TEST_F(TestSgdTiling, sgd_tiling_reject_rank9)
{
    std::initializer_list<int64_t> r9 = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    gert::StorageShape big = {r9, r9};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(RunSgdTiling(big, big, one, big, one, big, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
              ge::GRAPH_FAILED);
}

TEST_F(TestSgdTiling, sgd_tiling_reject_empty_tensor)
{
    // 空 tensor 判非法（不是"空进空出"）—— accum/stat 的原地回写在 numel == 0 下无定义
    gert::StorageShape empty = {{0, 3}, {0, 3}};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(
        RunSgdTiling(empty, empty, one, empty, one, empty, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED);
}

// ── DFX：空 tensor「全空间」枚举 ────────────────────────────────────────────
// changwei-op-dev step5-verify §5.4 要求：空 tensor = **任意一轴或多轴为 0**，
// 必须逐形态真跑分类，不能只验一个维度，结论要写成「仅支持哪个轴为空」。
// 本算子的契约是【所有形态一律拒绝】（无空进空出语义），故下面每一条都断言 GRAPH_FAILED。
// 若将来放开某一轴，本组用例会立刻在该形态上变红，逼迫同步更新契约与文档。
//
// 覆盖：1-D [0] / 2-D 每轴单独为 0 / 2-D 双轴同时为 0 /
//       3-D 每轴单独为 0 / 3-D 多轴组合为 0 / 8-D（rank 上界）末轴为 0
struct EmptyShapeCase {
    const char* desc;
    std::vector<int64_t> dims;
};

TEST_F(TestSgdTiling, sgd_tiling_reject_empty_tensor_full_space)
{
    const std::vector<EmptyShapeCase> cases = {
        {"1D_[0]", {0}},
        {"2D_axis0_[0,3]", {0, 3}},
        {"2D_axis1_[2,0]", {2, 0}},
        {"2D_both_[0,0]", {0, 0}},
        {"3D_axis0_[0,2,3]", {0, 2, 3}},
        {"3D_axis1_[2,0,3]", {2, 0, 3}},
        {"3D_axis2_[2,3,0]", {2, 3, 0}},
        {"3D_axis01_[0,0,3]", {0, 0, 3}},
        {"3D_axis02_[0,2,0]", {0, 2, 0}},
        {"3D_all_[0,0,0]", {0, 0, 0}},
        {"8D_lastaxis_[2,2,2,2,2,2,2,0]", {2, 2, 2, 2, 2, 2, 2, 0}},
    };
    gert::StorageShape one = {{1}, {1}};
    for (const auto& c : cases) {
        // gert::StorageShape 只能用花括号字面量或逐维 AppendDim 构造，
        // 【不能】从 std::vector<int64_t> 隐式转换（会报 could not convert ... to gert::StorageShape）。
        gert::StorageShape empty;
        for (int64_t d : c.dims) {
            empty.MutableOriginShape().AppendDim(d);
            empty.MutableStorageShape().AppendDim(d);
        }
        EXPECT_EQ(
            RunSgdTiling(empty, empty, one, empty, one, empty, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
            ge::GRAPH_FAILED)
            << "空 tensor 形态 " << c.desc << " 未被拒绝；本算子契约为【任意一轴或多轴为 0 一律非法】";
    }
}

TEST_F(TestSgdTiling, sgd_tiling_reject_empty_tensor_partial_inputs)
{
    // 只有【部分】输入为空的形态：parameters 非空但 gradient / accum / stat 为空。
    // 这类先撞 CheckSameShape（形状不等）也算拒绝，但必须确认「不崩、返 GRAPH_FAILED」。
    gert::StorageShape param = {{2, 3}, {2, 3}};
    gert::StorageShape empty = {{0, 3}, {0, 3}};
    gert::StorageShape one = {{1}, {1}};
    EXPECT_EQ(
        RunSgdTiling(param, empty, one, param, one, param, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED)
        << "gradient 为空未被拒绝";
    EXPECT_EQ(
        RunSgdTiling(param, param, one, empty, one, param, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED)
        << "accum 为空未被拒绝";
    EXPECT_EQ(
        RunSgdTiling(param, param, one, param, one, empty, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED)
        << "stat 为空未被拒绝";
}

TEST_F(TestSgdTiling, sgd_tiling_reject_shape_mismatch_gradient)
{
    // 大张量必须严格同形，**不做广播** —— "可广播但不相等"同样非法
    gert::StorageShape param = {{2, 3}, {2, 3}};
    gert::StorageShape grad = {{1, 3}, {1, 3}};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(RunSgdTiling(param, grad, one, param, one, param, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
              ge::GRAPH_FAILED);
}

TEST_F(TestSgdTiling, sgd_tiling_reject_shape_mismatch_stat)
{
    gert::StorageShape param = {{2, 4}, {2, 4}};
    gert::StorageShape stat = {{2, 5}, {2, 5}};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(RunSgdTiling(param, param, one, param, one, stat, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
              ge::GRAPH_FAILED);
}

TEST_F(TestSgdTiling, sgd_tiling_reject_scalar_shape_not_one)
{
    // learning_rate / momentum 必须是 [1] 或 0D 标量
    gert::StorageShape param = {{2, 4}, {2, 4}};
    gert::StorageShape two = {{2}, {2}};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(
        RunSgdTiling(param, param, two, param, one, param, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED);
    ASSERT_EQ(
        RunSgdTiling(param, param, one, param, two, param, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED);
}

TEST_F(TestSgdTiling, sgd_tiling_reject_dtype_mismatch)
{
    // 9 个张量位必须同 dtype（parameters 为 fp32，其余为 fp16）
    gert::StorageShape param = {{2, 4}, {2, 4}};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(
        RunSgdTiling(param, param, one, param, one, param, ge::DT_FLOAT, ge::DT_FLOAT16, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED);
}

TEST_F(TestSgdTiling, sgd_tiling_reject_unsupported_dtype)
{
    gert::StorageShape param = {{2, 4}, {2, 4}};
    gert::StorageShape one = {{1}, {1}};
    ASSERT_EQ(
        RunSgdTiling(param, param, one, param, one, param, ge::DT_INT32, ge::DT_INT32, 0.0f, 0.0f, false, nullptr),
        ge::GRAPH_FAILED);
}
