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
 * \file test_thnn_fused_gru_cell_tiling.cpp
 * \brief ThnnFusedGruCell tiling UT（TilingContextPara + ExecuteTiling 框架）：
 *        覆盖正常路由（tilingKey=0 原路径 / tilingKey=1 h-major 小 H 大 B 路由）、
 *        空 Tensor 兜底与校验链负例，并对 TilingData 字段做字节级校验。
 */

#include <gtest/gtest.h>
#include <vector>

#include "tiling_case_executor.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_host/arch35/thnn_fused_gru_cell_tiling_arch35.h"
#include "../../../../op_kernel/arch35/thnn_fused_gru_cell_struct.h"
#include "../../../../op_kernel/arch35/thnn_fused_gru_cell_tiling_struct.h"

using namespace std;

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
// Ascend950 平台常量（AIV 64 核 / UB 245760B），与 TilingParse 产物一致
constexpr uint64_t UT_CORE_NUM = 64;
constexpr uint64_t UT_UB_SIZE = 245760;
constexpr uint64_t UT_TILING_DATA_SIZE = 4096;
} // namespace

class ThnnFusedGruCellTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ThnnFusedGruCellTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ThnnFusedGruCellTiling TearDown" << std::endl; }
};

// 通用 tiling 测试模板：执行 TilingFunc → 校验状态 / tilingKey / TilingData 字段。
// inputInstances 的 0 表示可选 bias 输入缺省（IrInstanceNum 语义）。
static void RunThnnFusedGruCellTiling(const vector<TensorDesc>& inputs, const vector<TensorDesc>& outputs,
                                      const vector<uint32_t>& inputInstances, const vector<uint32_t>& outputInstances,
                                      ge::DataType dataType, ge::graphStatus expectStatus, uint64_t expectTilingKey,
                                      bool verifyFields = true)
{
    (void)dataType;
    // CompileInfo 由 TilingParse 产物口径预填（coreNum / ubSize），ReadPlatform 源①直接采纳
    optiling::ThnnFusedGruCellCompileInfo compileInfo{UT_CORE_NUM, UT_UB_SIZE};
    gert::TilingContextPara para("ThnnFusedGruCell", inputs, outputs, {}, inputInstances, outputInstances, &compileInfo,
                                 UT_CORE_NUM, UT_UB_SIZE, UT_TILING_DATA_SIZE);

    TilingInfo info;
    const bool succeeded = ExecuteTiling(para, info);
    EXPECT_EQ(succeeded, expectStatus == ge::GRAPH_SUCCESS);
    if (!succeeded) {
        return;
    }
    EXPECT_EQ(info.tilingKey, expectTilingKey);
    if (!verifyFields) {
        return;
    }

    // TilingData 内容校验（真实 ThnnFusedGruCellTilingData<4> 布局）
    auto* tiling = reinterpret_cast<ThnnFusedGruCellTilingData<THNN_FUSED_GRU_CELL_RANK_4>*>(info.tilingData.get());
    ASSERT_NE(tiling, nullptr);
    ASSERT_GE(info.tilingDataSize, sizeof(ThnnFusedGruCellTilingData<THNN_FUSED_GRU_CELL_RANK_4>));
    const bool hasInputBias = (inputInstances[3] != 0);
    const bool hasHiddenBias = (inputInstances[4] != 0);
    const int64_t expectBiasFlags = (hasInputBias ? INPUT_BIAS_FLAG : 0) | (hasHiddenBias ? HIDDEN_BIAS_FLAG : 0);
    EXPECT_EQ(tiling->biasFlags, expectBiasFlags);

    // hx desc 形状为 (B, H) 锚点（TensorDescription 按 shape 拷贝构造）
    const gert::StorageShape& hxShape = inputs[2].shape_;
    const int64_t batch = hxShape.GetStorageShape().GetDim(0);
    const int64_t hidden = hxShape.GetStorageShape().GetDim(1);
    if (expectTilingKey == 0 && batch > 0 && hidden > 0) {
        // 原路径：坐标系 (1,1,B,H) + numInputs/numOutputs/biasFlags 编码 + 多核字段自洽
        EXPECT_EQ(tiling->numInputs, REQUIRED_INPUT_COUNT + ((expectBiasFlags & INPUT_BIAS_FLAG) ? 1 : 0) +
                                         ((expectBiasFlags & HIDDEN_BIAS_FLAG) ? 1 : 0));
        EXPECT_EQ(tiling->numOutputs, MAX_OUTPUT_SLOTS);
        EXPECT_EQ(tiling->maxBroShape[0], 1);
        EXPECT_EQ(tiling->maxBroShape[1], 1);
        EXPECT_EQ(tiling->maxBroShape[2], batch);
        EXPECT_EQ(tiling->maxBroShape[3], hidden);
        EXPECT_GE(tiling->multicore.numCores, 1);
        EXPECT_LE(tiling->multicore.numCores, static_cast<int64_t>(UT_CORE_NUM));
        EXPECT_EQ(tiling->multicore.totalTiles,
                  tiling->multicore.numCores * tiling->multicore.tilesMain + tiling->multicore.coresTail);
        EXPECT_GT(tiling->perBufBytes, 0);
        EXPECT_GT(tiling->tileElems, 0);
        EXPECT_GT(tiling->paddedHidden, 0);
    }
    if (expectTilingKey == 1) {
        // h-major 路由：B/H 复用 maxBroShape，R 为 16 对齐的 [256,768] 区间
        EXPECT_EQ(tiling->maxBroShape[2], batch);
        EXPECT_EQ(tiling->maxBroShape[3], hidden);
        EXPECT_GE(tiling->hmajorRowsPerTile, 256);
        EXPECT_LE(tiling->hmajorRowsPerTile, 768);
        EXPECT_EQ(tiling->hmajorRowsPerTile % 16, 0); // transpose 16 对齐约束
        EXPECT_EQ(tiling->hmajorPadGates, ((3 * hidden + 15) / 16) * 16);
        EXPECT_EQ(tiling->hmajorPadStorage, ((6 * hidden + 15) / 16) * 16);
        EXPECT_GE(tiling->multicore.numCores, 1);
        EXPECT_LE(tiling->multicore.numCores, static_cast<int64_t>(UT_CORE_NUM));
    }
}

// 标准合法 shape 构造：gates (B,3H) / hx (B,H) / bias (3H) / hy (B,H) / storage (B,5H)。
// 注意：inputs desc 仅包含在位槽位（ComputeNodeInfo 的 inputs_num_ 为实例数之和，
// 为缺省可选输入额外传 tensor 会使 platform/compile 槽位索引错位）
static void RunThnnFusedGruCellTilingNormal(int64_t batch, int64_t hidden, ge::DataType dataType, bool hasInputBias,
                                            bool hasHiddenBias, ge::graphStatus expectStatus, uint64_t expectTilingKey)
{
    gert::StorageShape gatesShape = {{batch, 3 * hidden}, {batch, 3 * hidden}};
    gert::StorageShape commonShape = {{batch, hidden}, {batch, hidden}};
    gert::StorageShape biasShape = {{3 * hidden}, {3 * hidden}};
    gert::StorageShape storageShape = {{batch, 5 * hidden}, {batch, 5 * hidden}};

    vector<TensorDesc> inputs = {
        {gatesShape, dataType, ge::FORMAT_ND},  // input_gates
        {gatesShape, dataType, ge::FORMAT_ND},  // hidden_gates
        {commonShape, dataType, ge::FORMAT_ND}, // hx
    };
    if (hasInputBias) {
        inputs.push_back({biasShape, dataType, ge::FORMAT_ND}); // input_bias（可选）
    }
    if (hasHiddenBias) {
        inputs.push_back({biasShape, dataType, ge::FORMAT_ND}); // hidden_bias（可选）
    }
    vector<TensorDesc> outputs = {
        {commonShape, dataType, ge::FORMAT_ND},  // hy
        {storageShape, dataType, ge::FORMAT_ND}, // storage
    };
    vector<uint32_t> inputInstances = {1, 1, 1, hasInputBias ? 1U : 0U, hasHiddenBias ? 1U : 0U};
    vector<uint32_t> outputInstances = {1, 1};
    RunThnnFusedGruCellTiling(inputs, outputs, inputInstances, outputInstances, dataType, expectStatus,
                              expectTilingKey);
}

// ==================== 正例 ====================

// 正例1: fp32 双 bias，原路径 tilingKey=0
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_float_with_bias_succ)
{
    RunThnnFusedGruCellTilingNormal(4, 8, ge::DT_FLOAT, true, true, ge::GRAPH_SUCCESS, 0);
}

// 正例2: fp32 双 bias 缺省，biasFlags=0
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_float_no_bias_succ)
{
    RunThnnFusedGruCellTilingNormal(4, 8, ge::DT_FLOAT, false, false, ge::GRAPH_SUCCESS, 0);
}

// 正例3: fp16 双 bias，多核 batch 维切分
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_fp16_multicore_succ)
{
    RunThnnFusedGruCellTilingNormal(8, 16, ge::DT_FLOAT16, true, true, ge::GRAPH_SUCCESS, 0);
}

// 正例4: bf16 无 bias，非 32B 对齐 H（paddedHidden 行对齐填充）
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_bf16_unaligned_hidden_succ)
{
    RunThnnFusedGruCellTilingNormal(2, 5, ge::DT_BF16, false, false, ge::GRAPH_SUCCESS, 0);
}

// 正例5: 空 Tensor（B == 0）兜底：totalTiles=0 / numCores=0 + SetBlockDim(1)
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_empty_batch_succ)
{
    RunThnnFusedGruCellTilingNormal(0, 8, ge::DT_FLOAT, false, false, ge::GRAPH_SUCCESS, 0);
}

// 正例6: h-major 小 H 大 B 路由（H*2B ≤ 10 且 B ≥ 256）→ tilingKey=1
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_hmajor_route_succ)
{
    RunThnnFusedGruCellTilingNormal(512, 5, ge::DT_FLOAT16, false, false, ge::GRAPH_SUCCESS, 1);
}

// 正例7: B < 256 不触发 h-major 路由，保持原路径 tilingKey=0
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_hmajor_not_routed_small_batch)
{
    RunThnnFusedGruCellTilingNormal(255, 5, ge::DT_FLOAT16, false, false, ge::GRAPH_SUCCESS, 0);
}

// ==================== 负例 ====================

// 负例1: dtype 白名单外（INT32）
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_dtype_not_support)
{
    RunThnnFusedGruCellTilingNormal(4, 8, ge::DT_INT32, false, false, ge::GRAPH_FAILED, 0);
}

// 负例2: input_gates rank 1（期望 2 维）
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_gates_rank_invalid)
{
    gert::StorageShape gatesShape = {{24}, {24}};
    gert::StorageShape commonShape = {{4, 8}, {4, 8}};
    gert::StorageShape storageShape = {{4, 40}, {4, 40}};

    vector<TensorDesc> inputs = {
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<TensorDesc> outputs = {
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<uint32_t> inputInstances = {1, 1, 1, 0, 0};
    vector<uint32_t> outputInstances = {1, 1};
    RunThnnFusedGruCellTiling(inputs, outputs, inputInstances, outputInstances, ge::DT_FLOAT, ge::GRAPH_FAILED, 0,
                              false);
}

// 负例3: input_gates.shape[1] != 3 * hx.shape[1]
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_gates_cols_not_3h)
{
    gert::StorageShape gatesShape = {{4, 32}, {4, 32}}; // 期望 (4, 24)
    gert::StorageShape commonShape = {{4, 8}, {4, 8}};
    gert::StorageShape storageShape = {{4, 40}, {4, 40}};

    vector<TensorDesc> inputs = {
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<TensorDesc> outputs = {
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<uint32_t> inputInstances = {1, 1, 1, 0, 0};
    vector<uint32_t> outputInstances = {1, 1};
    RunThnnFusedGruCellTiling(inputs, outputs, inputInstances, outputInstances, ge::DT_FLOAT, ge::GRAPH_FAILED, 0,
                              false);
}

// 负例4: hidden_gates shape != input_gates shape
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_hidden_gates_shape_mismatch)
{
    gert::StorageShape gatesShape = {{4, 24}, {4, 24}};
    gert::StorageShape hiddenGatesShape = {{5, 24}, {5, 24}};
    gert::StorageShape commonShape = {{4, 8}, {4, 8}};
    gert::StorageShape storageShape = {{4, 40}, {4, 40}};

    vector<TensorDesc> inputs = {
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {hiddenGatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<TensorDesc> outputs = {
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<uint32_t> inputInstances = {1, 1, 1, 0, 0};
    vector<uint32_t> outputInstances = {1, 1};
    RunThnnFusedGruCellTiling(inputs, outputs, inputInstances, outputInstances, ge::DT_FLOAT, ge::GRAPH_FAILED, 0,
                              false);
}

// 负例5: hx rank 1
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_hx_rank_invalid)
{
    gert::StorageShape gatesShape = {{4, 24}, {4, 24}};
    gert::StorageShape hxShape = {{32}, {32}}; // 期望 (4, 8)
    gert::StorageShape commonShape = {{4, 8}, {4, 8}};
    gert::StorageShape storageShape = {{4, 40}, {4, 40}};

    vector<TensorDesc> inputs = {
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {hxShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<TensorDesc> outputs = {
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<uint32_t> inputInstances = {1, 1, 1, 0, 0};
    vector<uint32_t> outputInstances = {1, 1};
    RunThnnFusedGruCellTiling(inputs, outputs, inputInstances, outputInstances, ge::DT_FLOAT, ge::GRAPH_FAILED, 0,
                              false);
}

// 负例6: bias numel != 3H
TEST_F(ThnnFusedGruCellTiling, thnn_fused_gru_cell_bias_numel_mismatch)
{
    gert::StorageShape gatesShape = {{4, 24}, {4, 24}};
    gert::StorageShape commonShape = {{4, 8}, {4, 8}};
    gert::StorageShape biasShape = {{25}, {25}}; // 期望 numel == 24
    gert::StorageShape storageShape = {{4, 40}, {4, 40}};

    vector<TensorDesc> inputs = {
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {biasShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<TensorDesc> outputs = {
        {commonShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    vector<uint32_t> inputInstances = {1, 1, 1, 1, 0};
    vector<uint32_t> outputInstances = {1, 1};
    RunThnnFusedGruCellTiling(inputs, outputs, inputInstances, outputInstances, ge::DT_FLOAT, ge::GRAPH_FAILED, 0,
                              false);
}
