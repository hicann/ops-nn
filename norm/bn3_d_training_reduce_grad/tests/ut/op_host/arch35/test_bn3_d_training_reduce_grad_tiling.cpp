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
#include <vector>
#include <gtest/gtest.h>

#include "graph/types.h"
#include "../../../../op_host/arch35/bn3_d_training_reduce_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

namespace {

constexpr int64_t kPerBufBytes = 65536; // (ubSize / PHYS_NODES) & ~31 参考值，ubSize = 262144

// 构造公共 Tiling 纯公式输入：8 个张量（0=grads、1=x、2..6=5 个通道参数、7=y）
optiling::PublicTilingInputs MakePublicInputs(ge::DataType dtype, ge::Format format, const vector<int64_t>& gradShape,
                                              int64_t paramLen, float epsilon)
{
    optiling::PublicTilingInputs in{};
    for (int32_t i = 0; i < 8; i++) {
        in.ranks[i] = (i < 2 || i == 7) ? 5 : 1;
        for (int32_t d = 0; d < 5; d++) {
            in.shapes[i][d] = 0;
        }
        in.dtypes[i] = (i >= 2 && i <= 6) ? ge::DT_FLOAT : dtype;
        in.formats[i] = (i >= 2 && i <= 6) ? ge::FORMAT_ND : format;
    }
    for (int32_t d = 0; d < 5; d++) {
        in.shapes[0][d] = gradShape[d];
        in.shapes[1][d] = gradShape[d];
        in.shapes[7][d] = gradShape[d];
    }
    for (int32_t i = 2; i <= 6; i++) {
        in.shapes[i][0] = paramLen;
    }
    in.epsilon = epsilon;
    return in;
}

// 构造 Branch0 Tiling 纯公式输入（rank ∈ [1,4]）
void FillNormalShapes(optiling::Branch0TilingInputs& in, const vector<int64_t>& bro, const vector<int64_t>& paramBro)
{
    in.rank = static_cast<int32_t>(bro.size());
    for (int32_t d = 0; d < static_cast<int32_t>(bro.size()); d++) {
        in.maxBroShape[d] = bro[d];
    }
    for (int32_t i = 0; i < 7; i++) {
        const vector<int64_t>& src = (i < 2) ? bro : paramBro;
        for (int32_t d = 0; d < static_cast<int32_t>(bro.size()); d++) {
            in.normalInputShapes[i][d] = src[d];
        }
    }
    for (int32_t d = 0; d < static_cast<int32_t>(bro.size()); d++) {
        in.normalOutputShapes[0][d] = bro[d];
    }
}

} // namespace

class Bn3DTrainingReduceGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Bn3DTrainingReduceGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Bn3DTrainingReduceGradTilingTest TearDown" << std::endl; }
};

// ============================================================================
// 公共 Tiling：常规 5D NCDHW 正例（key=1 / RANK=8）
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, public_tiling_ncdhw_key1)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kOk);
    EXPECT_EQ(out.channelAxis, 1); // NCDHW 通道轴 = dim1
    EXPECT_EQ(out.num, 240);       // num = 2*4*5*6
    EXPECT_EQ(out.rank, 5);
    EXPECT_EQ(out.mapped, 8);
    EXPECT_EQ(out.tilingKey, 1);
    const vector<int64_t> expectBro = {2, 3, 4, 5, 6};
    for (int32_t d = 0; d < 5; d++) {
        EXPECT_EQ(out.maxBroShape[d], expectBro[d]);
        EXPECT_EQ(out.normalInputShapes[0][d], expectBro[d]);  // grads
        EXPECT_EQ(out.normalOutputShapes[0][d], expectBro[d]); // y
    }
    const vector<int64_t> expectParam = {1, 3, 1, 1, 1}; // 参数张量：C 置 dim1
    for (int32_t d = 0; d < 5; d++) {
        EXPECT_EQ(out.normalInputShapes[2][d], expectParam[d]);
    }
}

// ============================================================================
// 公共 Tiling：常规 5D NDHWC 正例（通道轴 = dim4）
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, public_tiling_ndhwc_key1)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NDHWC, {2, 3, 4, 5, 6}, 6, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kOk);
    EXPECT_EQ(out.channelAxis, 4); // NDHWC 通道轴 = dim4
    EXPECT_EQ(out.num, 120);       // num = 2*3*4*5
    EXPECT_EQ(out.rank, 5);
    EXPECT_EQ(out.mapped, 8);
    EXPECT_EQ(out.tilingKey, 1);
    const vector<int64_t> expectParam = {1, 1, 1, 1, 6}; // 参数张量：C 置 dim4
    for (int32_t d = 0; d < 5; d++) {
        EXPECT_EQ(out.normalInputShapes[2][d], expectParam[d]);
    }
}

// ============================================================================
// 公共 Tiling：C=1 通道轴挤压（去 1 后有效 rank=4 → key=0 / RANK=4）
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, public_tiling_branch0_key0)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 1, 4, 5, 6}, 1, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kOk);
    EXPECT_EQ(out.channelAxis, 1);
    EXPECT_EQ(out.num, 240); // num = 2*4*5*6
    EXPECT_EQ(out.rank, 4);  // dim1 在全部输入输出上均为 1，被 squeeze
    EXPECT_EQ(out.mapped, 4);
    EXPECT_EQ(out.tilingKey, 0);
    const vector<int64_t> expectBro = {2, 4, 5, 6};
    for (int32_t d = 0; d < 4; d++) {
        EXPECT_EQ(out.maxBroShape[d], expectBro[d]);
        EXPECT_EQ(out.normalInputShapes[0][d], expectBro[d]); // grads
    }
    const vector<int64_t> expectParam = {1, 1, 1, 1};
    for (int32_t d = 0; d < 4; d++) {
        EXPECT_EQ(out.normalInputShapes[2][d], expectParam[d]);
    }
}

// ============================================================================
// 公共 Tiling：严格双格式——5D ND（布局未声明）被 format 校验拒绝，不推断通道轴
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, public_tiling_nd_format_rejected)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_ND, {2, 4, 5, 6, 3}, 3, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kFormatError);
}

// ============================================================================
// 公共 Tiling：1-D 参数（diff_scale/scale/offset/mean 类）不校验 format——图编译期
// 可能统一刷成 NCDHW，任意参数 format 均不影响校验结果
TEST_F(Bn3DTrainingReduceGradTilingTest, public_tiling_params_format_ignored)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0001f);
    for (int32_t i = 2; i <= 6; i++) {
        in.formats[i] = ge::FORMAT_NCDHW; // 模拟 9.2.0 图编译统一刷新
    }
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kOk);
}

// ============================================================================
// 公共 Tiling：bf16 / fp16 dtype 正例（def 注册三档 dtype，tiling dtype 校验须放行）
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, public_tiling_bf16)
{
    auto in = MakePublicInputs(ge::DT_BF16, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kOk);
    EXPECT_EQ(out.channelAxis, 1);
}

TEST_F(Bn3DTrainingReduceGradTilingTest, public_tiling_fp16)
{
    auto in = MakePublicInputs(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kOk);
    EXPECT_EQ(out.channelAxis, 1);
}

// 公共 Tiling：错误码（空 tensor / shape 不一致 / dtype / format / attr）
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, error_null_input)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 0, 4, 5, 6}, 3, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kNullInput);
}

TEST_F(Bn3DTrainingReduceGradTilingTest, error_shape_mismatch)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 4, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kShapeMismatch);
}

// ============================================================================
// 公共 Tiling：空 tensor 逐轴枚举（0 依次置于 dim0..dim4）→ kNullInput
//   README「任一维为 0 时返回错误」需逐轴回归，不只通道轴。
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, error_null_input_each_axis)
{
    for (int32_t axis = 0; axis < 5; ++axis) {
        vector<int64_t> shape = {2, 3, 4, 5, 6};
        shape[axis] = 0;
        auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, shape, 3, 0.0001f);
        optiling::PublicTilingOutput out{};
        EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kNullInput) << "axis=" << axis;
    }
}

// ============================================================================
// 公共 Tiling：输出 y 空 tensor（任一维为 0）→ kNullInput
//   y 的 shape 由 InferShape 强制等于 grads，tiling 侧独立复检（防线不退化）。
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, error_null_input_output_y)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0001f);
    in.shapes[7][3] = 0; // 仅输出 y 的 dim3 置 0
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kNullInput);
}

// ============================================================================
// 公共 Tiling：参数长度 != 通道数 C（format-aware）→ kShapeMismatch
//   NDHWC 通道轴为 dim4=6，paramLen=3 不匹配；不依赖 CheckBroadcastShape 的
//   size-1 广播宽松语义（paramLen=1 而 C>1 也必须拒绝）。
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, error_param_len_mismatch)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NDHWC, {2, 3, 4, 5, 6}, 3, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kShapeMismatch);
}

TEST_F(Bn3DTrainingReduceGradTilingTest, error_param_len_one_mismatch)
{
    // paramLen=1 而 C>1：广播语义会放行，5.5 校验必须拒绝（静默算错点）
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 1, 0.0001f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kShapeMismatch);
}

TEST_F(Bn3DTrainingReduceGradTilingTest, error_dtype)
{
    auto in = MakePublicInputs(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0001f);
    in.dtypes[2] = ge::DT_FLOAT16; // 参数张量必须为 FLOAT
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kDtypeError);
}

TEST_F(Bn3DTrainingReduceGradTilingTest, error_format)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0001f);
    in.formats[1] = ge::FORMAT_NDHWC; // grads 与 x format 必须一致
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kFormatError);
}

TEST_F(Bn3DTrainingReduceGradTilingTest, error_attr)
{
    auto in = MakePublicInputs(ge::DT_FLOAT, ge::FORMAT_NCDHW, {2, 3, 4, 5, 6}, 3, 0.0f);
    optiling::PublicTilingOutput out{};
    EXPECT_EQ(optiling::ComputePublicTiling(in, out), optiling::PublicTilingError::kAttrError);
}

// ============================================================================
// 分支 Tiling：Branch-0（rank=4，单 tile 整张量装下）
//   maxBroShape=(2,4,5,6)、perBufBytes=65536、coreNum=32
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, branch0_compute_single_tile)
{
    optiling::Branch0TilingInputs in{};
    FillNormalShapes(in, {2, 4, 5, 6}, {1, 1, 1, 1});
    in.perBufBytes = kPerBufBytes;
    in.coreNum = 32;
    in.epsilon = 0.0001f;
    in.num = 240;

    BN3DTrainingReduceGradTilingData<4> out{};
    optiling::ComputeBranch0Tiling(in, out);

    // UB 切分：240 元素 < 16384（perBufBytes/4）→ 整张量单 tile
    EXPECT_EQ(out.split.axis, 0);
    EXPECT_EQ(out.split.aI, 2);
    EXPECT_EQ(out.split.aO, 1);
    EXPECT_EQ(out.split.aITail, 2);
    // 多核切分：totalTiles=1 → numCores=1
    EXPECT_EQ(out.multicore.numCores, 1);
    EXPECT_EQ(out.multicore.totalTiles, 1);
    EXPECT_EQ(out.multicore.tilesMain, 1);
    EXPECT_EQ(out.multicore.coresTail, 0);
    // 字段填充（delta = 4 - 4 = 0，无前补右移）
    EXPECT_EQ(out.rank, 4);
    EXPECT_EQ(out.perBufBytes, kPerBufBytes);
    EXPECT_EQ(out.epsilon, 0.0001f);
    EXPECT_EQ(out.num, 240);
    const vector<int64_t> expectBro = {2, 4, 5, 6};
    const vector<int64_t> expectStrides = {120, 30, 6, 1};
    for (int32_t d = 0; d < 4; d++) {
        EXPECT_EQ(out.maxBroShape[d], expectBro[d]);
        EXPECT_EQ(out.inputShapes[0][d], expectBro[d]);
        EXPECT_EQ(out.inputStrides[0][d], expectStrides[d]);
        EXPECT_EQ(out.outputShapes[0][d], expectBro[d]);
        EXPECT_EQ(out.outputStrides[0][d], expectStrides[d]);
        // 参数张量：广播轴（size-1）stride=0
        EXPECT_EQ(out.inputShapes[2][d], 1);
        EXPECT_EQ(out.inputStrides[2][d], 0);
    }
}

// ============================================================================
// 分支 Tiling：Branch-1（rank=5，delta=3 前补右移）
//   maxBroShape=(2,3,4,5,6)、perBufBytes=65536、coreNum=32
// ============================================================================
TEST_F(Bn3DTrainingReduceGradTilingTest, branch1_compute_single_tile)
{
    optiling::Branch1TilingInputs in{};
    {
        in.rank = 5;
        const vector<int64_t> bro = {2, 3, 4, 5, 6};
        const vector<int64_t> paramBro = {1, 3, 1, 1, 1};
        for (int32_t d = 0; d < 5; d++) {
            in.maxBroShape[d] = bro[d];
        }
        for (int32_t i = 0; i < 7; i++) {
            const vector<int64_t>& src = (i < 2) ? bro : paramBro;
            for (int32_t d = 0; d < 5; d++) {
                in.normalInputShapes[i][d] = src[d];
            }
        }
        for (int32_t d = 0; d < 5; d++) {
            in.normalOutputShapes[0][d] = bro[d];
        }
    }
    in.perBufBytes = kPerBufBytes;
    in.coreNum = 32;
    in.epsilon = 0.0001f;
    in.num = 240;

    BN3DTrainingReduceGradTilingData<8> out{};
    optiling::ComputeBranch1Tiling(in, out);

    // UB 切分：720 元素 < 16384 → 整张量单 tile；delta=3 → split.axis 0 → 3
    EXPECT_EQ(out.split.axis, 3);
    EXPECT_EQ(out.split.aI, 2);
    EXPECT_EQ(out.split.aO, 1);
    EXPECT_EQ(out.split.aITail, 2);
    // 多核切分：totalTiles=1 → numCores=1
    EXPECT_EQ(out.multicore.numCores, 1);
    EXPECT_EQ(out.multicore.totalTiles, 1);
    EXPECT_EQ(out.multicore.tilesMain, 1);
    EXPECT_EQ(out.multicore.coresTail, 0);
    // 字段填充（delta = 8 - 5 = 3，前 3 维补 1/stride=0）
    EXPECT_EQ(out.rank, 5);
    EXPECT_EQ(out.perBufBytes, kPerBufBytes);
    EXPECT_EQ(out.epsilon, 0.0001f);
    EXPECT_EQ(out.num, 240);
    const vector<int64_t> expectBro = {1, 1, 1, 2, 3, 4, 5, 6};
    const vector<int64_t> expectStrides = {0, 0, 0, 360, 120, 30, 6, 1};
    for (int32_t d = 0; d < 8; d++) {
        EXPECT_EQ(out.maxBroShape[d], expectBro[d]);
        EXPECT_EQ(out.inputShapes[0][d], expectBro[d]);
        EXPECT_EQ(out.inputStrides[0][d], expectStrides[d]);
        EXPECT_EQ(out.outputShapes[0][d], expectBro[d]);
        EXPECT_EQ(out.outputStrides[0][d], expectStrides[d]);
    }
    // 参数张量：norm shape=(1,3,1,1,1) → stride=(0,1,0,0,0)，右移 3 位
    const vector<int64_t> expectParamShape = {1, 1, 1, 1, 3, 1, 1, 1};
    const vector<int64_t> expectParamStrides = {0, 0, 0, 0, 1, 0, 0, 0};
    for (int32_t d = 0; d < 8; d++) {
        EXPECT_EQ(out.inputShapes[2][d], expectParamShape[d]);
        EXPECT_EQ(out.inputStrides[2][d], expectParamStrides[d]);
    }
}
