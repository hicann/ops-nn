/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Parameterized host-tiling regression for every row in tests/st/arch35/
// ttk_kernel_gru_block_cell_grad_negative.csv.  Each case builds the same
// tensor descriptors as the CSV row and must be rejected by the implementation
// that kernel-mode execution reaches, rather than by VERIFY_FUNC_REG.

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "../../../op_host/arch35/gru_block_cell_grad_tiling_arch35.h"

namespace {
constexpr size_t kNumInputs = 10;
constexpr size_t kNumOutputs = 4;
constexpr std::array<const char*, kNumInputs> kInputNames = {"x",   "h_prev", "w_ru", "w_c", "b_ru",
                                                             "b_c", "r",      "u",    "c",   "d_h"};

enum class Constraint {
    kInputDtype,
    kInputFormat,
    kOutputDtype,
    kOutputFormat,
    kMissingInput,
    kInputRank,
    kInputShape,
    kCellSizeZero,
    kValid,
};

struct NegativeCase {
    std::string name;
    Constraint constraint;
    size_t tensor = 0;
    size_t dim = 0;
    bool greater = false;
    ge::DataType dtype = ge::DT_FLOAT;
};

std::string Numbered(const char* prefix, int number)
{
    char buffer[128] = {};
    (void)snprintf(buffer, sizeof(buffer), "%s%03d", prefix, number);
    return buffer;
}

std::vector<std::vector<int64_t>> MakeInputDims(int64_t batch = 3, int64_t inputSize = 11, int64_t cellSize = 5)
{
    const int64_t k = inputSize + cellSize;
    return {{batch, inputSize}, {batch, cellSize}, {k, 2 * cellSize}, {k, cellSize},     {2 * cellSize},
            {cellSize},         {batch, cellSize}, {batch, cellSize}, {batch, cellSize}, {batch, cellSize}};
}

std::vector<std::vector<int64_t>> MakeOutputDims(int64_t batch = 3, int64_t inputSize = 11, int64_t cellSize = 5)
{
    return {{batch, inputSize}, {batch, cellSize}, {batch, cellSize}, {batch, 2 * cellSize}};
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

std::vector<NegativeCase> BuildNegativeCases()
{
    std::vector<NegativeCase> cases;
    cases.reserve(96);
    for (size_t input = 0; input < kNumInputs; ++input) {
        const int first = static_cast<int>(input * 3 + 1);
        cases.push_back({Numbered("aclnnGRUBlockCellGrad_L2_exc_dtype_", first), Constraint::kInputDtype, input, 0,
                         false, ge::DT_BF16});
        cases.push_back({Numbered("aclnnGRUBlockCellGrad_L2_exc_dtype_", first + 1), Constraint::kInputDtype, input, 0,
                         false, ge::DT_BOOL});
        cases.push_back({Numbered("aclnnGRUBlockCellGrad_L2_exc_format_", first + 2), Constraint::kInputFormat, input});
    }
    for (size_t output = 0; output < kNumOutputs; ++output) {
        const int first = static_cast<int>(output * 3 + 31);
        cases.push_back({Numbered("aclnnGRUBlockCellGrad_L2_exc_dtype_", first), Constraint::kOutputDtype, output, 0,
                         false, ge::DT_BF16});
        cases.push_back({Numbered("aclnnGRUBlockCellGrad_L2_exc_dtype_", first + 1), Constraint::kOutputDtype, output,
                         0, false, ge::DT_BOOL});
        cases.push_back(
            {Numbered("aclnnGRUBlockCellGrad_L2_exc_format_", first + 2), Constraint::kOutputFormat, output});
    }
    cases.push_back({"aclnnGRUBlockCellGrad_L2_exc_missing_047", Constraint::kMissingInput});

    for (size_t input = 0; input < kNumInputs; ++input) {
        cases.push_back({std::string("GRUBlockCellGrad_L2_input_rank_") + kInputNames[input] + "_low",
                         Constraint::kInputRank, input});
        cases.push_back({std::string("GRUBlockCellGrad_L2_input_rank_") + kInputNames[input] + "_high",
                         Constraint::kInputRank, input, 0, true});
    }

    const std::array<std::pair<size_t, size_t>, 15> shapeTargets = {{{1, 0},
                                                                     {2, 0},
                                                                     {2, 1},
                                                                     {3, 0},
                                                                     {3, 1},
                                                                     {4, 0},
                                                                     {5, 0},
                                                                     {6, 0},
                                                                     {6, 1},
                                                                     {7, 0},
                                                                     {7, 1},
                                                                     {8, 0},
                                                                     {8, 1},
                                                                     {9, 0},
                                                                     {9, 1}}};
    for (const auto& target : shapeTargets) {
        const std::string prefix = std::string("GRUBlockCellGrad_L2_input_shape_") + kInputNames[target.first] +
                                   "_dim" + std::to_string(target.second);
        cases.push_back({prefix + "_less", Constraint::kInputShape, target.first, target.second});
        cases.push_back({prefix + "_greater", Constraint::kInputShape, target.first, target.second, true});
    }

    cases.push_back({"GRUBlockCellGrad_L2_exc_cell_zero_032", Constraint::kCellSizeZero});
    cases.push_back({"GRUBlockCellGrad_L2_exc_cell_zero_034", Constraint::kCellSizeZero, 1});
    cases.push_back({"GRUBlockCellGrad_L2_exc_cell_zero_035", Constraint::kCellSizeZero, 2});
    return cases;
}

ge::graphStatus RunTilingCase(const NegativeCase& testCase)
{
    int64_t batch = 3;
    int64_t inputSize = 11;
    int64_t cellSize = 5;
    if (testCase.constraint == Constraint::kCellSizeZero) {
        // The three CSV rows are (B, I, C) = (2, 8, 0), (2, 0, 0), (0, 0, 0).
        batch = testCase.tensor == 2 ? 0 : 2;
        inputSize = testCase.tensor == 1 ? 0 : 8;
        cellSize = 0;
    }
    std::vector<std::vector<int64_t>> inputDims = MakeInputDims(batch, inputSize, cellSize);
    const std::vector<std::vector<int64_t>> outputDims = MakeOutputDims(batch, inputSize, cellSize);
    std::vector<ge::DataType> inputDtypes(kNumInputs, ge::DT_FLOAT);
    std::vector<ge::Format> inputFormats(kNumInputs, ge::FORMAT_ND);
    std::vector<ge::DataType> outputDtypes(kNumOutputs, ge::DT_FLOAT);
    std::vector<ge::Format> outputFormats(kNumOutputs, ge::FORMAT_ND);

    if (testCase.constraint == Constraint::kInputDtype) {
        inputDtypes[testCase.tensor] = testCase.dtype;
    } else if (testCase.constraint == Constraint::kInputFormat) {
        inputFormats[testCase.tensor] = ge::FORMAT_FRACTAL_NZ;
    } else if (testCase.constraint == Constraint::kOutputDtype) {
        outputDtypes[testCase.tensor] = testCase.dtype;
    } else if (testCase.constraint == Constraint::kOutputFormat) {
        outputFormats[testCase.tensor] = ge::FORMAT_FRACTAL_NZ;
    } else if (testCase.constraint == Constraint::kInputRank) {
        if (testCase.greater) {
            inputDims[testCase.tensor].push_back(1);
        } else {
            const int64_t elementCount = std::accumulate(inputDims[testCase.tensor].begin(),
                                                         inputDims[testCase.tensor].end(), int64_t{1},
                                                         std::multiplies<int64_t>());
            inputDims[testCase.tensor].clear();
            if (testCase.tensor != 4 && testCase.tensor != 5) {
                inputDims[testCase.tensor].push_back(elementCount);
            }
        }
    } else if (testCase.constraint == Constraint::kInputShape) {
        inputDims[testCase.tensor][testCase.dim] += testCase.greater ? 1 : -1;
    }

    std::vector<gert::StorageShape> inputShapes;
    std::vector<gert::StorageShape> outputShapes;
    inputShapes.reserve(kNumInputs);
    outputShapes.reserve(kNumOutputs);
    for (const auto& dims : inputDims) {
        inputShapes.emplace_back(MakeStorageShape(dims));
    }
    for (const auto& dims : outputDims) {
        outputShapes.emplace_back(MakeStorageShape(dims));
    }
    std::vector<gert::StorageShape*> inputShapePointers;
    std::vector<gert::StorageShape*> outputShapePointers;
    inputShapePointers.reserve(kNumInputs);
    outputShapePointers.reserve(kNumOutputs);
    for (size_t i = 0; i < kNumInputs; ++i) {
        inputShapePointers.push_back(testCase.constraint == Constraint::kMissingInput && i == 0 ? nullptr :
                                                                                                  &inputShapes[i]);
    }
    for (size_t i = 0; i < kNumOutputs; ++i) {
        outputShapePointers.push_back(&outputShapes[i]);
    }

    // Valid compile-time platform data keeps these negative tests on the
    // descriptor/shape validation path instead of the platform fallback.
    optiling::GruBlockCellGradCompileInfo compileInfo{64, 245760, 524288, 65536, 65536, 131072, 16777216};
    fe::PlatFormInfos platformInfo;
    if (!platformInfo.Init()) {
        ADD_FAILURE() << "failed to initialize the TilingContext platform data";
        return ge::GRAPH_FAILED;
    }
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    if (tilingData == nullptr || workspaceHolder == nullptr) {
        ADD_FAILURE() << "failed to build the TilingContext test infrastructure";
        return ge::GRAPH_FAILED;
    }
    const auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("GRUBlockCellGrad")
                      .NodeIoNum(kNumInputs, kNumOutputs)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputShapePointers)
                      .OutputShapes(outputShapePointers)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, inputDtypes[0], inputFormats[0], inputFormats[0])
                      .NodeInputTd(1, inputDtypes[1], inputFormats[1], inputFormats[1])
                      .NodeInputTd(2, inputDtypes[2], inputFormats[2], inputFormats[2])
                      .NodeInputTd(3, inputDtypes[3], inputFormats[3], inputFormats[3])
                      .NodeInputTd(4, inputDtypes[4], inputFormats[4], inputFormats[4])
                      .NodeInputTd(5, inputDtypes[5], inputFormats[5], inputFormats[5])
                      .NodeInputTd(6, inputDtypes[6], inputFormats[6], inputFormats[6])
                      .NodeInputTd(7, inputDtypes[7], inputFormats[7], inputFormats[7])
                      .NodeInputTd(8, inputDtypes[8], inputFormats[8], inputFormats[8])
                      .NodeInputTd(9, inputDtypes[9], inputFormats[9], inputFormats[9])
                      .NodeOutputTd(0, outputDtypes[0], outputFormats[0], outputFormats[0])
                      .NodeOutputTd(1, outputDtypes[1], outputFormats[1], outputFormats[1])
                      .NodeOutputTd(2, outputDtypes[2], outputFormats[2], outputFormats[2])
                      .NodeOutputTd(3, outputDtypes[3], outputFormats[3], outputFormats[3])
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GRUBlockCellGrad");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        ADD_FAILURE() << "GRUBlockCellGrad host Tiling was not registered";
        return ge::GRAPH_FAILED;
    }
    auto* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr) {
        ADD_FAILURE() << "failed to build the TilingContext test infrastructure";
        return ge::GRAPH_FAILED;
    }
    return opImpl->tiling(context);
}

class GruBlockCellGradNegativeTilingTest : public testing::TestWithParam<NegativeCase> {};

TEST_P(GruBlockCellGradNegativeTilingTest, rejected_by_host_tiling)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GRUBlockCellGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
    EXPECT_EQ(RunTilingCase(GetParam()), ge::GRAPH_FAILED) << GetParam().name;
}

TEST(GruBlockCellGradNegativeTilingBaseline, valid_descriptors_reach_successful_tiling)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GRUBlockCellGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
    EXPECT_EQ(RunTilingCase({"valid_fp32_nd", Constraint::kValid}), ge::GRAPH_SUCCESS);
}

TEST(GruBlockCellGradNegativeTilingManifest, maps_all_negative_csv_rows)
{
    const std::vector<NegativeCase> cases = BuildNegativeCases();
    ASSERT_EQ(cases.size(), 96U);
    std::array<size_t, 9> counts{};
    for (const auto& testCase : cases) {
        ++counts[static_cast<size_t>(testCase.constraint)];
    }
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kInputDtype)], 20U);
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kInputFormat)], 10U);
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kOutputDtype)], 8U);
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kOutputFormat)], 4U);
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kMissingInput)], 1U);
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kInputRank)], 20U);
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kInputShape)], 30U);
    EXPECT_EQ(counts[static_cast<size_t>(Constraint::kCellSizeZero)], 3U);
}

INSTANTIATE_TEST_SUITE_P(NegativeCsv, GruBlockCellGradNegativeTilingTest, testing::ValuesIn(BuildNegativeCases()),
                         [](const testing::TestParamInfo<NegativeCase>& info) { return info.param.name; });
} // namespace
