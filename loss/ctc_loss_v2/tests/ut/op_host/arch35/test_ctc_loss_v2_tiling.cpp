/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "../../../../op_host/arch35/ctc_loss_v2_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
constexpr size_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t CTC_TILING_KEY_IS_FP32 = 1UL;           // isFP32 occupies bit 0..7
constexpr uint64_t CTC_TILING_KEY_INT32_THREAD = 1UL << 8; // threadTypeInt32 occupies bit 8..15

// The serialized CTCLossV2TilingData4AscendC buffer is a compact sequence of the 18 int64 fields
// declared by TILING_DATA_FIELD_DEF (the generated class itself carries extra bookkeeping members,
// so the raw buffer is reinterpreted through this layout-compatible POD mirror).
struct CtcLossV2TilingPod {
    int64_t maxInputLength;
    int64_t maxTargetLength;
    int64_t lpInputStride;
    int64_t lpBatchStride;
    int64_t lpCharStride;
    int64_t laBatchStride;
    int64_t laInputStride;
    int64_t laTargetStride;
    int64_t tgTargetStride;
    int64_t batchSize;
    int64_t blank;
    int64_t blockDimX;
    int64_t blockDimY;
    int64_t targetsDim;
    int64_t tgBatchStride;
    int64_t workspaceSize;
    int64_t gridY;
    int64_t usedCoreNum;
};
} // namespace

template <typename T>
static void SetConstInput(size_t const_index, ge::DataType dtype, const T* const_data, int64_t data_size,
                          std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder = std::make_unique<uint8_t[]>(sizeof(gert::Tensor) +
                                                                                 sizeof(T) * data_size);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor({{data_size}, {data_size}},         // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        dtype,                              // dtype
                        nullptr);
    memcpy_s(input_tensor, sizeof(gert::Tensor), &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

class CTCLossV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CTCLossV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CTCLossV2Tiling TearDown" << std::endl; }
};

// Drives one CTCLossV2 arch35 tiling invocation on the Ascend950 platform.
// log_probs/targets shapes are given as storage shapes, input_lengths/target_lengths are const
// tensors carried by the faker. On success the raw tiling data is reinterpreted as
// CtcLossV2TilingPod for field-level assertions.
static ge::graphStatus RunCtcLossV2Tiling(gert::StorageShape& logProbsShape, gert::StorageShape& targetsShape,
                                          gert::StorageShape& inputLengthsShape, gert::StorageShape& targetLengthsShape,
                                          gert::StorageShape& negLogLikelihoodShape, gert::StorageShape& logAlphaShape,
                                          ge::DataType logProbsDtype, ge::DataType lengthsDtype,
                                          const std::vector<int64_t>& inputLengthsData,
                                          const std::vector<int64_t>& targetLengthsData, int64_t blank,
                                          CtcLossV2TilingPod& outTiling, uint64_t& outTilingKey, uint32_t& outBlockDim)
{
    std::string op_type("CTCLossV2");
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
    optiling::CTCLossV2CompileInfo compile_info;

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

    // const inputs: index 2 = input_lengths, index 3 = target_lengths
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    if (lengthsDtype == ge::DT_INT32) {
        std::vector<int32_t> inputLengths32(inputLengthsData.begin(), inputLengthsData.end());
        std::vector<int32_t> targetLengths32(targetLengthsData.begin(), targetLengthsData.end());
        SetConstInput(2, DT_INT32, inputLengths32.data(), static_cast<int64_t>(inputLengths32.size()), const_tensors);
        SetConstInput(3, DT_INT32, targetLengths32.data(), static_cast<int64_t>(targetLengths32.size()), const_tensors);
    } else {
        SetConstInput(2, DT_INT64, inputLengthsData.data(), static_cast<int64_t>(inputLengthsData.size()),
                      const_tensors);
        SetConstInput(3, DT_INT64, targetLengthsData.data(), static_cast<int64_t>(targetLengthsData.size()),
                      const_tensors);
    }

    auto param = gert::TilingData::CreateCap(4096);
    EXPECT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &inputLengthsShape, &targetLengthsShape})
                      .OutputShapes({&negLogLikelihoodShape, &logAlphaShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, logProbsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, lengthsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, lengthsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, lengthsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, logProbsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, logProbsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(blank)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                                  {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .ConstInput(const_tensors)
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
        auto raw = tiling_context->GetRawTilingData();
        if (raw != nullptr && raw->GetDataSize() >= sizeof(CtcLossV2TilingPod)) {
            outTiling = *reinterpret_cast<const CtcLossV2TilingPod*>(raw->GetData());
        }
        outTilingKey = tiling_context->GetTilingKey();
        outBlockDim = tiling_context->GetBlockDim();
    }
    return status;
}

// 3D log_probs (T=50, N=16, C=8), fp32, int32 lengths, 2D targets:
// small size -> threadTypeInt32=1, isFP32=1 -> tilingKey = 1 | (1 << 8) = 257.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_3d_fp32_int32_success)
{
    gert::StorageShape log_probs_shape = {{50, 16, 8}, {50, 16, 8}};
    gert::StorageShape targets_shape = {{16, 30}, {16, 30}};
    gert::StorageShape input_lengths_shape = {{16}, {16}};
    gert::StorageShape target_lengths_shape = {{16}, {16}};
    gert::StorageShape neg_log_likelihood_shape = {{16}, {16}};
    gert::StorageShape log_alpha_shape = {{16, 50, 3}, {16, 50, 3}};
    std::vector<int64_t> input_lengths(16, 50);
    std::vector<int64_t> target_lengths(16, 1);

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_FLOAT, ge::DT_INT32, input_lengths,
                                 target_lengths, 9, tiling, tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    // strides and dims derived from (T=50, N=16, C=8), maxTargetLength=1
    EXPECT_EQ(tiling.maxInputLength, 50);
    EXPECT_EQ(tiling.maxTargetLength, 1);
    EXPECT_EQ(tiling.lpInputStride, 128); // 16 * 8
    EXPECT_EQ(tiling.lpBatchStride, 8);   // C
    EXPECT_EQ(tiling.lpCharStride, 1);
    EXPECT_EQ(tiling.batchSize, 16);
    EXPECT_EQ(tiling.laBatchStride, 150); // 50 * (2*1+1)
    EXPECT_EQ(tiling.laInputStride, 3);   // 2*1+1
    EXPECT_EQ(tiling.laTargetStride, 1);
    EXPECT_EQ(tiling.targetsDim, 2);
    EXPECT_EQ(tiling.tgBatchStride, 30);
    EXPECT_EQ(tiling.tgTargetStride, 1);
    EXPECT_EQ(tiling.blank, 9);
    // thread config: threadsTarget = 2*1+1 = 3, blockDimY = min(ceil(16/64), 1024/3) = 1
    EXPECT_EQ(tiling.blockDimX, 3);
    EXPECT_EQ(tiling.blockDimY, 1);
    EXPECT_EQ(tiling.gridY, 16);       // (16 + 1 - 1) / 1
    EXPECT_EQ(tiling.usedCoreNum, 16); // min(16, 64)
    EXPECT_EQ(block_dim, 16);
    EXPECT_EQ(tiling.workspaceSize, SYSTEM_WORKSPACE_SIZE);                      // fp32 needs no extra workspace
    EXPECT_EQ(tiling_key, CTC_TILING_KEY_IS_FP32 | CTC_TILING_KEY_INT32_THREAD); // 257
}

// 3D log_probs, fp16: isFP32=0 and extra workspace is appended for the fp16 path.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_3d_fp16_extra_workspace)
{
    gert::StorageShape log_probs_shape = {{50, 16, 8}, {50, 16, 8}};
    gert::StorageShape targets_shape = {{16, 30}, {16, 30}};
    gert::StorageShape input_lengths_shape = {{16}, {16}};
    gert::StorageShape target_lengths_shape = {{16}, {16}};
    gert::StorageShape neg_log_likelihood_shape = {{16}, {16}};
    gert::StorageShape log_alpha_shape = {{16, 50, 3}, {16, 50, 3}};
    std::vector<int64_t> input_lengths(16, 50);
    std::vector<int64_t> target_lengths(16, 1);

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_FLOAT16, ge::DT_INT32, input_lengths,
                                 target_lengths, 0, tiling, tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    // extra workspace = CeilAlign((2*1+1)*4*16*50, 32) = CeilAlign(9600, 32) = 9600
    EXPECT_EQ(tiling.workspaceSize, SYSTEM_WORKSPACE_SIZE + 9600);
    EXPECT_EQ(tiling_key, CTC_TILING_KEY_INT32_THREAD); // isFP32=0 -> 256
    EXPECT_EQ(tiling.blank, 0);
}

// 2D log_probs (T, C) is expanded to (T, 1, C); 1D targets keeps tgBatchStride = 1.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_2d_log_probs_success)
{
    gert::StorageShape log_probs_shape = {{20, 10}, {20, 10}};
    gert::StorageShape targets_shape = {{30}, {30}};
    gert::StorageShape input_lengths_shape = {{1}, {1}};
    gert::StorageShape target_lengths_shape = {{1}, {1}};
    gert::StorageShape neg_log_likelihood_shape = {{1}, {1}};
    gert::StorageShape log_alpha_shape = {{1, 20, 7}, {1, 20, 7}};
    std::vector<int64_t> input_lengths = {20};
    std::vector<int64_t> target_lengths = {3};

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_FLOAT, ge::DT_INT32, input_lengths,
                                 target_lengths, 3, tiling, tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.batchSize, 1);       // N expanded to 1
    EXPECT_EQ(tiling.maxInputLength, 20); // T
    EXPECT_EQ(tiling.lpInputStride, 10);  // 1 * C
    EXPECT_EQ(tiling.lpBatchStride, 10);  // C
    EXPECT_EQ(tiling.maxTargetLength, 3);
    EXPECT_EQ(tiling.laInputStride, 7);   // 2*3+1
    EXPECT_EQ(tiling.laBatchStride, 140); // 20 * 7
    EXPECT_EQ(tiling.targetsDim, 1);
    EXPECT_EQ(tiling.tgBatchStride, 1);
    // threadsTarget = 2*3+1 = 7, blockDimY = min(ceil(1/64), 1024/7) = 1
    EXPECT_EQ(tiling.blockDimX, 7);
    EXPECT_EQ(tiling.blockDimY, 1);
    EXPECT_EQ(tiling.gridY, 1);
    EXPECT_EQ(tiling.usedCoreNum, 1);
    EXPECT_EQ(block_dim, 1);
}

// int64 input_lengths/target_lengths are supported for both validation and maxTargetLength.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_int64_lengths_success)
{
    gert::StorageShape log_probs_shape = {{10, 2, 6}, {10, 2, 6}};
    gert::StorageShape targets_shape = {{2, 10}, {2, 10}};
    gert::StorageShape input_lengths_shape = {{2}, {2}};
    gert::StorageShape target_lengths_shape = {{2}, {2}};
    gert::StorageShape neg_log_likelihood_shape = {{2}, {2}};
    gert::StorageShape log_alpha_shape = {{2, 10, 9}, {2, 10, 9}};
    std::vector<int64_t> input_lengths = {10, 10};
    std::vector<int64_t> target_lengths = {2, 4};

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_FLOAT, ge::DT_INT64, input_lengths,
                                 target_lengths, 5, tiling, tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.maxTargetLength, 4); // max(2, 4)
    EXPECT_EQ(tiling.laInputStride, 9);   // 2*4+1
    EXPECT_EQ(tiling.laBatchStride, 90);  // 10 * 9
    EXPECT_EQ(tiling.blockDimX, 9);       // threadsTarget = 2*4+1
    EXPECT_EQ(tiling.gridY, 2);           // batchSize 2 / blockDimY 1
    EXPECT_EQ(tiling.usedCoreNum, 2);
    EXPECT_EQ(tiling_key, CTC_TILING_KEY_IS_FP32 | CTC_TILING_KEY_INT32_THREAD);
}

// Large size (T*N*C overflows int32) falls back to 512 threads: threadTypeInt32=0 -> tilingKey low bits 0.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_large_size_512_threads)
{
    gert::StorageShape log_probs_shape = {{65536, 1, 32768}, {65536, 1, 32768}};
    gert::StorageShape targets_shape = {{1, 30}, {1, 30}};
    gert::StorageShape input_lengths_shape = {{1}, {1}};
    gert::StorageShape target_lengths_shape = {{1}, {1}};
    gert::StorageShape neg_log_likelihood_shape = {{1}, {1}};
    gert::StorageShape log_alpha_shape = {{1, 65536, 3}, {1, 65536, 3}};
    std::vector<int64_t> input_lengths = {65536};
    std::vector<int64_t> target_lengths = {1};

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    ASSERT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_FLOAT, ge::DT_INT32, input_lengths,
                                 target_lengths, 0, tiling, tiling_key, block_dim),
              ge::GRAPH_SUCCESS);
    // IsLargeSize: 65536 * 1 * 32768 > INT32_MAX -> threadTypeInt32 = 0
    EXPECT_EQ(tiling_key, CTC_TILING_KEY_IS_FP32); // 1, no int32-thread bit
    EXPECT_EQ(tiling.blockDimX, 3);                // threadsTarget = min(512 >= 3) -> 2*1+1
    EXPECT_GT(block_dim, 0);
}

// Unsupported log_probs dtype (int32) is rejected.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_invalid_log_probs_dtype_failed)
{
    gert::StorageShape log_probs_shape = {{50, 16, 8}, {50, 16, 8}};
    gert::StorageShape targets_shape = {{16, 30}, {16, 30}};
    gert::StorageShape input_lengths_shape = {{16}, {16}};
    gert::StorageShape target_lengths_shape = {{16}, {16}};
    gert::StorageShape neg_log_likelihood_shape = {{16}, {16}};
    gert::StorageShape log_alpha_shape = {{16, 50, 3}, {16, 50, 3}};
    std::vector<int64_t> input_lengths(16, 50);
    std::vector<int64_t> target_lengths(16, 1);

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_INT32, ge::DT_INT32, input_lengths,
                                 target_lengths, 0, tiling, tiling_key, block_dim),
              ge::GRAPH_FAILED);
}

// input_lengths greater than maxInputLength (int32 path) is rejected.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_input_lengths_overflow_int32_failed)
{
    gert::StorageShape log_probs_shape = {{10, 2, 6}, {10, 2, 6}};
    gert::StorageShape targets_shape = {{2, 10}, {2, 10}};
    gert::StorageShape input_lengths_shape = {{2}, {2}};
    gert::StorageShape target_lengths_shape = {{2}, {2}};
    gert::StorageShape neg_log_likelihood_shape = {{2}, {2}};
    gert::StorageShape log_alpha_shape = {{2, 10, 5}, {2, 10, 5}};
    std::vector<int64_t> input_lengths = {11, 10}; // 11 > maxInputLength 10

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_FLOAT, ge::DT_INT32, input_lengths,
                                 {2, 2}, 0, tiling, tiling_key, block_dim),
              ge::GRAPH_FAILED);
}

// input_lengths greater than maxInputLength (int64 path) is rejected.
TEST_F(CTCLossV2Tiling, ctc_loss_v2_tiling_input_lengths_overflow_int64_failed)
{
    gert::StorageShape log_probs_shape = {{10, 2, 6}, {10, 2, 6}};
    gert::StorageShape targets_shape = {{2, 10}, {2, 10}};
    gert::StorageShape input_lengths_shape = {{2}, {2}};
    gert::StorageShape target_lengths_shape = {{2}, {2}};
    gert::StorageShape neg_log_likelihood_shape = {{2}, {2}};
    gert::StorageShape log_alpha_shape = {{2, 10, 5}, {2, 10, 5}};
    std::vector<int64_t> input_lengths = {10, 11}; // 11 > maxInputLength 10

    CtcLossV2TilingPod tiling{};
    uint64_t tiling_key = 0;
    uint32_t block_dim = 0;
    EXPECT_EQ(RunCtcLossV2Tiling(log_probs_shape, targets_shape, input_lengths_shape, target_lengths_shape,
                                 neg_log_likelihood_shape, log_alpha_shape, ge::DT_FLOAT, ge::DT_INT64, input_lengths,
                                 {2, 2}, 0, tiling, tiling_key, block_dim),
              ge::GRAPH_FAILED);
}
