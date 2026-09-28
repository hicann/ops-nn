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
 * \file test_single_layer_lstm_tiling.cpp
 * \brief SingleLayerLstm tiling unit tests.
 *
 * Every case that expects GRAPH_FAILED is a REFUSAL the operator makes on purpose. They matter more
 * than the success case: the alternative to refusing is clamping to a tiling that overflows a
 * buffer on chip, and an on-chip overflow produces no error message anywhere.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/single_layer_lstm_tiling_data.h"
#include "../../../../op_kernel/arch35/single_layer_lstm_layout.h" // RowsOfBlock: the kernel's own block walk

namespace {

constexpr int64_t T = 8;
constexpr int64_t B = 16;
constexpr int64_t I = 32;
constexpr int64_t H = 64;

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

/* The faker builds the tiling context AROUND this object, so it must be a real address: passing
 * nullptr leaves the context without compute-node info and the harness then faults inside
 * GetPlatformInfo(), before the tiling function is entered. The CONTENT is unused here because the
 * harness does not call TilingParse -- the platform info it installs is what tiling reads. */
struct CompileInfoStub {
    uint64_t sysWorkspaceSize = 0;
};
CompileInfoStub g_compileInfo;

/* ascend950 as tiling sees it: 72 AIV, 248 KB of programmable UB. */
constexpr uint64_t AIV_CORE_NUM = 72;
constexpr uint64_t UB_SIZE = 248 * 1024;
constexpr uint64_t TILING_DATA_CAP = 4096;

/* One SingleLayerLstm tiling context. `hidden` is separate from `H` so a case can make init_h disagree with w
 * without rebuilding the whole list.
 *
 * `dt` is the common floating-point input/output dtype.
 * biasDt is overridden only by the case checking its refusal. */
gert::TilingContextPara MakePara(int64_t t, int64_t b, int64_t i, int64_t hidden, const char* direction,
                                 const char* gateOrder, int64_t wRows = -1, bool withSeqLength = false,
                                 ge::DataType dt = ge::DT_FLOAT, ge::DataType biasDt = ge::DT_UNDEFINED,
                                 int64_t logicalInput = -1, int64_t logicalHidden = -1,
                                 bool includeLogicalAttrs = false, int64_t hiddenBiasSize = -1,
                                 ge::DataType hiddenBiasDt = ge::DT_UNDEFINED)
{
    const int64_t gates4H = 4 * hidden;
    const int64_t rows = (wRows < 0) ? (i + hidden) : wRows;
    std::vector<TensorDesc> inputs = {
        {{{t, b, i}, {t, b, i}}, dt, ge::FORMAT_ND},             // x
        {{{rows, gates4H}, {rows, gates4H}}, dt, ge::FORMAT_ND}, // w
        {{{gates4H}, {gates4H}}, biasDt == ge::DT_UNDEFINED ? dt : biasDt, ge::FORMAT_ND},
        {{{b, hidden}, {b, hidden}}, dt, ge::FORMAT_ND}, // init_h
        {{{b, hidden}, {b, hidden}}, dt, ge::FORMAT_ND}, // init_c
    };
    /* seq_length is OPTIONAL, and both shapes of context matter: a graph that
     * omits it leaves the IR index uninstantiated, which is the case that must not be read with
     * GetInputTensor. */
    if (withSeqLength || hiddenBiasSize >= 0) {
        inputs.push_back({{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND});
    }
    if (hiddenBiasSize >= 0) {
        inputs.push_back({{{hiddenBiasSize}, {hiddenBiasSize}},
                          hiddenBiasDt == ge::DT_UNDEFINED ? dt : hiddenBiasDt,
                          ge::FORMAT_ND});
    }
    std::vector<TensorDesc> outputs(8, {{{t, b, hidden}, {t, b, hidden}}, dt, ge::FORMAT_ND});
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom<std::string>(direction)},
        {"gate_order", Ops::NN::AnyValue::CreateFrom<std::string>(gateOrder)},
    };
    if (includeLogicalAttrs || logicalInput != -1 || logicalHidden != -1) {
        attrs.push_back({"logical_input_size", Ops::NN::AnyValue::CreateFrom<int64_t>(logicalInput)});
        attrs.push_back({"logical_hidden_size", Ops::NN::AnyValue::CreateFrom<int64_t>(logicalHidden)});
    }
    return gert::TilingContextPara("SingleLayerLstm", inputs, outputs, attrs, &g_compileInfo, AIV_CORE_NUM, UB_SIZE,
                                   TILING_DATA_CAP);
}

} // namespace

class SingleLayerLstmTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SingleLayerLstmTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SingleLayerLstmTiling TearDown" << std::endl; }
};

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_accepts_a_supported_shape)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo");
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_GT(info.blockNum, 0U);
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->batch, static_cast<uint32_t>(B));
    EXPECT_EQ(tiling->inputSize, static_cast<uint32_t>(I));
    EXPECT_EQ(tiling->hiddenSize, static_cast<uint32_t>(H));
    EXPECT_EQ(tiling->logicalInputSize, static_cast<uint32_t>(I));
    EXPECT_EQ(tiling->logicalHiddenSize, static_cast<uint32_t>(H));
    EXPECT_EQ(tiling->timeStep, static_cast<uint32_t>(T));
    EXPECT_EQ(tiling->hasBiasHh, 0U);
    /* tChunk must divide T and kChunk must divide I: the A tile lays each batch row's timesteps at a
     * fixed NZ row offset, so a short last chunk would leave holes between rows, and a ragged K tail
     * would need a second fractal count in the innermost loop. */
    ASSERT_GT(tiling->tChunk, 0U);
    ASSERT_GT(tiling->kChunk, 0U);
    EXPECT_EQ(static_cast<uint32_t>(T) % tiling->tChunk, 0U);
    EXPECT_GT(tiling->kChunk, 0U); // kChunk need not divide I: the last block is short
    /* rowsPerBlock * blockDim IS NOT THE BATCH, and asserting that it was encoded a rule the
     * operator no longer has. rowsPerBlock is an ON-CHIP extent; the grid walks the batch in blocks
     * of it, striding by blockDim * rowsPerBlock, so the product is the rows covered by ONE sweep
     * of the grid and B may take several. What has to hold is that the walk reaches every row and
     * covers none twice -- which is what this replays, the same loop the kernel runs. */
    ASSERT_GT(tiling->rowsPerBlock, 0U);
    ASSERT_GT(tiling->blockDim, 0U);
    EXPECT_LE(tiling->blockDim, 16U); // MAX_CLUSTERS
    {
        std::vector<uint32_t> visits(static_cast<size_t>(B), 0U);
        for (uint32_t cluster = 0; cluster < tiling->blockDim; ++cluster) {
            for (uint32_t base = cluster * tiling->rowsPerBlock; base < static_cast<uint32_t>(B);
                 base += tiling->blockDim * tiling->rowsPerBlock) {
                const uint32_t rows = SingleLayerLstmFwd::RowsOfBlock(static_cast<uint32_t>(B), tiling->rowsPerBlock,
                                                                      base);
                for (uint32_t r = 0; r < rows; ++r) {
                    visits[base + r] += 1U;
                }
            }
        }
        for (int64_t r = 0; r < B; ++r) {
            ASSERT_EQ(visits[static_cast<size_t>(r)], 1U) << "row " << r << " visited " << visits[r] << " times";
        }
    }
    /* The workspace offsets must be strictly increasing, or two of the four regions alias and the
     * recurrence reads the projection's bytes -- which builds, runs, and returns wrong numbers. */
    EXPECT_LT(tiling->offIgates, tiling->offHAll);
    EXPECT_LT(tiling->offHAll, tiling->offCAll);
    EXPECT_LT(tiling->offCAll, tiling->offStore);
}

/* w's transpose is a different matrix. The kernel reads rows [I, I+H) as the already-transposed
 * hidden weight with a pointer offset, so accepting [4H, I+H] would give plausible wrong numbers. */
/* The same shape, but with the optional seq_length instantiated. Reading an optional input that IS
 * present must work as well as skipping one that is not. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_accepts_an_instantiated_seq_length)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    /* The VALUE only reaches the host when the caller materialised it there; when it does not, the
     * fallback is T, and either outcome is in range. Asserting the range rather than the value is
     * deliberate -- claiming the tail-zeroing path works needs a test with a short sequence on
     * hardware, which this is not. */
    EXPECT_LE(tiling->seqLenMax, static_cast<uint32_t>(T));
}

TEST_F(SingleLayerLstmTiling, separate_bias_allocates_fp32_sum_for_every_dtype)
{
    for (auto dt : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, true, dt, ge::DT_UNDEFINED, -1, -1, false,
                             4 * H);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info));
        ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
        const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
        EXPECT_EQ(tiling->hasBiasHh, 1U);
        EXPECT_GE(tiling->offBF32, tiling->offWF32);
        auto fused = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, true, dt);
        TilingInfo fusedInfo;
        ASSERT_TRUE(ExecuteTiling(fused, fusedInfo));
        ASSERT_EQ(info.workspaceSizes.size(), 1U);
        ASSERT_EQ(fusedInfo.workspaceSizes.size(), 1U);
        const int64_t extraBytes = dt == ge::DT_FLOAT ? 4 * H * sizeof(float) : 0;
        EXPECT_EQ(info.workspaceSizes[0] - fusedInfo.workspaceSizes[0], extraBytes);
    }
}

TEST_F(SingleLayerLstmTiling, separate_bias_refuses_wrong_shape)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, true, ge::DT_FLOAT16, ge::DT_UNDEFINED, -1, -1,
                         false, 4 * H - 1);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmTiling, separate_bias_refuses_mixed_dtype)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, true, ge::DT_FLOAT16, ge::DT_UNDEFINED, -1, -1,
                         false, 4 * H, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_transposed_w)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", 4 * H);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* hidden_size must be a multiple of the fp32 fractal width. Every strided UB copy states its row
 * pitch in 32-byte blocks, so a pitch that is not a whole number of them is not expressible: row 0
 * comes out right and everything after it is garbage. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_hidden_not_c0_aligned)
{
    auto para = MakePara(T, B, I, 12, "UNIDIRECTIONAL", "ifjo");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_input_not_c0_aligned)
{
    auto para = MakePara(T, B, 12, H, "UNIDIRECTIONAL", "ifjo");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* Refused, not ignored. gate_order selects the column order of w and b; "ijfo" against "ifjo" is a
 * permutation that leaves the result smooth and plausible, so nothing downstream would report it. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_unsupported_gate_order)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ijfo");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_logical_defaults_use_physical_extents)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT, ge::DT_FLOAT, -1, -1, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->logicalInputSize, static_cast<uint32_t>(I));
    EXPECT_EQ(tiling->logicalHiddenSize, static_cast<uint32_t>(H));
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_preserves_odd_logical_extents)
{
    auto para = MakePara(3, 5, 40, 24, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_BF16, ge::DT_BF16, 33, 17);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->inputSize, 40U);
    EXPECT_EQ(tiling->hiddenSize, 24U);
    EXPECT_EQ(tiling->logicalInputSize, 33U);
    EXPECT_EQ(tiling->logicalHiddenSize, 17U);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_accepts_zero_logical_input)
{
    auto para = MakePara(T, B, 8, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, H);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->inputSize, 8U);
    EXPECT_EQ(tiling->logicalInputSize, 0U);
    EXPECT_EQ(tiling->logicalHiddenSize, static_cast<uint32_t>(H));
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_invalid_logical_extents)
{
    const int64_t invalid[][2] = {{-2, H}, {I + 1, H}, {I, -2}, {I, 0}, {I, H + 1}};
    for (const auto& dimensions : invalid) {
        auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT, ge::DT_FLOAT, dimensions[0],
                             dimensions[1]);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_unsupported_direction)
{
    auto para = MakePara(T, B, I, H, "REDIRECTIONAL", "ifjo");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* A hidden size far past what fits any on-chip buffer whole. IT IS ACCEPTED: both axes of the
 * recurrence GEMM are tiled, phase A's N axis with them, and each residency degrades into a stream,
 * so no buffer's footprint grows with hidden_size. This test asserted a refusal while one whole gate
 * went to L0B at a time; the refusal was the correct statement about that kernel and is not about
 * this one. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_accepts_hidden_past_every_residency)
{
    auto para = MakePara(T, B, I, 1024, "UNIDIRECTIONAL", "ifjo");
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    EXPECT_GT(tiling->projNChunk, 0U);
    EXPECT_LE(tiling->projNChunk, 1024U);
}

/* THE THREE DTYPES, AND WHAT EACH ONE MOVES.
 *
 * I = 32 is a multiple of both fractal widths, so the same shape is accepted at all three and the
 * cases below differ in nothing but the dtype. The shapes that separate them are in the two cases
 * after these. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_accepts_fp16)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    EXPECT_GT(tiling->kChunk, 0U); // kChunk need not divide I: the last block is short
    EXPECT_EQ(static_cast<uint32_t>(T) % tiling->tChunk, 0U);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_accepts_bf16)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_BF16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
}

/* input_size's rule FOLLOWS THE DTYPE and hidden_size's DOES NOT. C0 is 32 bytes, so the fractal is
 * 8 elements at fp32 and 16 at the narrow widths, and input_size is phase A's K axis -- the one
 * place the caller's width reaches the cube. hidden_size is bounded by phase B and the epilogue,
 * which are fp32 whatever the caller sent.
 *
 * I = 24 is therefore a legal fp32 shape and an illegal fp16 one, and H = 24 is legal at both. Both
 * halves are asserted: a check that only ran at one width would leave the other free to be wrong,
 * and both directions of that mistake are silent -- a refused shape reads as "unsupported" and an
 * accepted one reads as a precision bug. */
/* I=24 is a multiple of 8 and not of 16. IT IS ACCEPTED AT ALL THREE DTYPES, and this test asserted
 * the opposite while phase A took the caller's width into the cube: the narrow fractal is 16
 * elements, so I=24 was an fp32-only shape. Phase A widens x and w[0:I] to fp32 now, so input_size
 * follows the fp32 fractal like hidden_size does, and the two extents finally share one rule. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_input_size_alignment_does_not_follow_the_dtype)
{
    /* ExecuteTiling, not ExecuteTestCase: the latter's success path also compares the workspace list
     * against an expectation, and the default expectation is empty, so a shape that tiles correctly
     * fails on the one workspace this operator requests. */
    TilingInfo info;
    auto fp32 = MakePara(T, B, 24, H, "UNIDIRECTIONAL", "ifjo");
    EXPECT_TRUE(ExecuteTiling(fp32, info)) << "I=24 must tile at fp32";

    auto fp16 = MakePara(T, B, 24, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT16);
    EXPECT_TRUE(ExecuteTiling(fp16, info)) << "I=24 must tile at fp16 too";

    auto bf16 = MakePara(T, B, 24, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_BF16);
    EXPECT_TRUE(ExecuteTiling(bf16, info)) << "I=24 must tile at bf16 too";

    /* 12 is not a multiple of 8 and is refused at every dtype. */
    auto bad = MakePara(T, B, 12, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT16);
    ExecuteTestCase(bad, ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_hidden_size_alignment_does_not_follow_the_dtype)
{
    TilingInfo info;
    auto fp32 = MakePara(T, B, I, 24, "UNIDIRECTIONAL", "ifjo");
    EXPECT_TRUE(ExecuteTiling(fp32, info)) << "H=24 must tile at fp32";

    auto fp16 = MakePara(T, B, I, 24, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT16);
    EXPECT_TRUE(ExecuteTiling(fp16, info)) << "H=24 must tile at fp16 too: phase B is fp32";
}

/* THE WORKSPACE OFFSETS ARE uint32 ELEMENT COUNTS, AND A BIG ENOUGH SHAPE OVERFLOWS THEM.
 *
 * This pair is a boundary, not a ceiling. T=8 B=16384 H=4096 passes every on-chip test -- the batch
 * is walked in blocks and both axes of the recurrence GEMM are tiled, so nothing about the chip
 * turns it away -- and it needs 5.5 G fp32 elements of workspace. igates alone is 2^31 of them, so
 * the running total wraps in uint32 and offStore, offXF32 and offWF32 then point INSIDE earlier
 * buffers: the kernel writes the caller's outputs from bytes the projection owns, which is a
 * shape-correct result with wrong numbers. The tiling refuses instead.
 *
 * The second case is the same batch one eighth as long, at 2.8 G elements, and IS accepted -- so
 * what the check bounds is the workspace, not the batch this operator can take. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_a_workspace_it_cannot_address)
{
    auto para = MakePara(8, 16384, 32, 4096, "UNIDIRECTIONAL", "ifjo");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_accepts_a_large_batch_whose_workspace_fits)
{
    auto para = MakePara(4, 16384, 32, 4096, "UNIDIRECTIONAL", "ifjo");
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(SingleLayerLstmTilingData));
    const auto* tiling = reinterpret_cast<const SingleLayerLstmTilingData*>(info.tilingData.get());
    /* The batch does not fit one sweep of the grid: this is the case the block walk exists for. */
    EXPECT_LT(tiling->rowsPerBlock * tiling->blockDim, 16384U);
    EXPECT_LE(tiling->blockDim, 16U);
    EXPECT_LT(tiling->offIgates, tiling->offHAll);
    EXPECT_LT(tiling->offHAll, tiling->offCAll);
    EXPECT_LT(tiling->offCAll, tiling->offStore);
}

TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_a_mixed_bias)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* Every tensor that follows x has to agree with it: the kernel is built once per dtype, so a node
 * whose tensors disagree runs a binary compiled for one of them and reinterprets the rest. */
TEST_F(SingleLayerLstmTiling, single_layer_lstm_tiling_refuses_mixed_input_dtypes)
{
    auto para = MakePara(T, B, I, H, "UNIDIRECTIONAL", "ifjo", -1, false, ge::DT_FLOAT16);
    para.inputTensorDesc_[3].dtype_ = ge::DT_FLOAT; // init_h out of step with x
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* There is no case for an output dtype out of step with x, and that is not an omission: it is not a
 * combination the def declares, so the framework never matches it and tiling never sees it. The
 * tiling context carries compute-node descriptors for inputs only -- this harness cannot express an
 * output dtype at all, and reading one back faults. */
