/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "tiling_context_faker.h"
#include "platform/platform_infos_def.h"
#include "op_impl_registry.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_mx_basic_api_tiling.h"
#include "matmul/quant_batch_matmul_v4/op_host/op_tiling/arch35/quant_batch_matmul_v4_weight_quant_mx_cube_tiling.h"
#include "matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_compile_info.h"

namespace {
using MxTilingData = DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData;
constexpr uint64_t MX_CUBE_SCALE_ND_KEY = 4369UL;
constexpr uint64_t MX_CUBE_SCALE_ND_A_FULL_LOAD_KEY = 5393UL;
static_assert(sizeof(MxTilingData) == 64, "V4 MX uses the shared 64-byte compact ABI");
static_assert(offsetof(MxTilingData, windowLen) == 56, "windowLen follows the original compact fields");
static_assert(std::is_base_of<optiling::AdaptiveSlidingWindowTiling,
                              optiling::AdaptiveSlidingWindowWeightQuantMxCubeTiling>::value,
              "V4 MX reuses the ASW tiling pipeline");
static_assert(!std::is_base_of<optiling::AdaptiveSlidingWindowMXBasicAPITiling,
                               optiling::AdaptiveSlidingWindowWeightQuantMxCubeTiling>::value,
              "V4 MX must not inherit V3 MX strategy or serialization state");

class MxV4Probe : public optiling::AdaptiveSlidingWindowWeightQuantMxCubeTiling {
public:
    explicit MxV4Probe(gert::TilingContext* context) : AdaptiveSlidingWindowWeightQuantMxCubeTiling(context) {}
    using AdaptiveSlidingWindowWeightQuantMxCubeTiling::GetShapeAttrsInfo;
    using AdaptiveSlidingWindowWeightQuantMxCubeTiling::IsCapable;
    using AdaptiveSlidingWindowWeightQuantMxCubeTiling::ResetTilingData;
    using AdaptiveSlidingWindowWeightQuantMxCubeTiling::SetTilingData;
    const optiling::QuantBatchMatmulInfo& Inputs() const { return inputParams_; }
    const MxTilingData& CompactData() const { return *static_cast<const MxTilingData*>(GetTilingData()); }
    size_t TilingDataSize() const { return tilingDataSize_; }
    void SetAFullLoad(bool enabled) { isAFullLoad_ = enabled; }
    void ConfigureL1Case(bool hif4, bool fullLoad, uint64_t l1Size)
    {
        inputParams_.Reset();
        inputParams_.opName = "QuantBatchMatmulV4";
        inputParams_.mSize = 64UL;
        inputParams_.nSize = 128UL;
        inputParams_.kSize = 96UL;
        inputParams_.aDtype = ge::DT_FLOAT8_E4M3FN;
        inputParams_.bDtype = hif4 ? Ops::NN::DT_HIFLOAT4 : ge::DT_FLOAT4_E2M1;
        inputParams_.perTokenScaleDtype = ge::DT_FLOAT8_E8M0;
        inputParams_.scaleDtype = hif4 ? Ops::NN::DT_HIFLOAT4_SCALE : ge::DT_FLOAT8_E8M0;
        inputParams_.biasDtype = ge::DT_FLOAT16;
        inputParams_.hasBias = true;
        basicTiling_.baseM = 64U;
        basicTiling_.baseN = 128U;
        basicTiling_.baseK = 64U;
        basicTiling_.dbL0c = 2U;
        basicTiling_.stepKa = 2U;
        basicTiling_.stepKb = 1U;
        basicTiling_.scaleFactorA = 3U;
        basicTiling_.scaleFactorB = 2U;
        aicoreParams_.l1Size = l1Size;
        isAFullLoad_ = fullLoad;
    }
    bool HasOriginalL1Windows() const
    {
        return basicTiling_.stepKa == 2U && basicTiling_.stepKb == 1U && basicTiling_.scaleFactorA == 3U &&
               basicTiling_.scaleFactorB == 2U;
    }
    bool EvaluateAFullLoad(NpuArch arch, uint64_t mBlockCnt)
    {
        compileInfo_.npuArch = arch;
        aicoreParams_.aicNum = 64UL;
        aicoreParams_.l1Size = 512UL * 1024UL;
        inputParams_.batchC = 1UL;
        inputParams_.aDtype = ge::DT_FLOAT8_E4M3FN;
        inputParams_.kSize = 96UL;
        adaptiveWin_.baseM = 64UL;
        adaptiveWin_.mBlockCnt = mBlockCnt;
        adaptiveWin_.totalBlockCnt = 128UL;
        AnalyseFullLoadInfo();
        return isAFullLoad_;
    }
};

enum class InvalidInput {
    NONE,
    MISSING_A_SCALE,
    MISSING_B_SCALE,
    WRONG_A_DTYPE,
    NON_MX_A_SCALE,
    NON_MX_B_SCALE,
    WRONG_B_SCALE_DTYPE,
    WRONG_SCALE_SHAPE,
    WRONG_B_SCALE_SHAPE,
    WRONG_NZ_SHAPE,
    WRONG_OUTPUT_SHAPE,
    WRONG_BIAS_SHAPE,
    BATCHED_BIAS,
    WRONG_BIAS_DTYPE,
    NON_NZ_B,
    TRANSPOSE_A,
    TRANSPOSE_B,
    WRONG_GROUP,
    BATCH,
    NEGATIVE_DIMENSION,
    UNALIGNED_N,
    UNALIGNED_K
};

struct MxCase {
    bool hif4 = false;
    bool bias = false;
    bool arch9202 = false;
    int64_t k = 192;
    ge::DataType output = ge::DT_FLOAT16;
    InvalidInput invalid = InvalidInput::NONE;
    size_t capacity = 64;
    uint32_t forbiddenInput = 0;
    int64_t m = 33;
    int64_t n = 24;
    int64_t fp4GroupSize = 32;
};

class MxContext {
public:
    explicit MxContext(const MxCase& param)
    {
        const int64_t m = param.m;
        const int64_t n = param.invalid == InvalidInput::UNALIGNED_N ? 25 : param.n;
        const int64_t k = param.invalid == InvalidInput::UNALIGNED_K ? 65 : param.k;
        const int64_t scaleK = (k + 63) / 64;
        gert::StorageShape x1({m, k}, {m, k});
        gert::StorageShape x2({n, k}, {(k + 63) / 64, (n + 15) / 16, 16, 64});
        gert::StorageShape aScale({m, scaleK, 2}, {m, scaleK, 2});
        const int64_t bScaleLastDim = param.hif4 ? 1 : 2;
        gert::StorageShape bScale({n, scaleK, bScaleLastDim}, {n, scaleK, bScaleLastDim});
        gert::StorageShape bias({1, n}, {1, n});
        gert::StorageShape y({m, n}, {m, n});
        gert::StorageShape offset({1}, {1});
        ge::DataType biasDtype = param.output;
        ge::DataType aDtype = ge::DT_FLOAT8_E4M3FN;
        ge::DataType aScaleDtype = ge::DT_FLOAT8_E8M0;
        ge::DataType bScaleDtype = param.hif4 ? Ops::NN::DT_HIFLOAT4_SCALE : ge::DT_FLOAT8_E8M0;
        ge::Format bFormat = ge::FORMAT_FRACTAL_NZ;
        int64_t groupSize = param.hif4 ? -1 : param.fp4GroupSize;
        switch (param.invalid) {
            case InvalidInput::WRONG_A_DTYPE:
                aDtype = ge::DT_FLOAT8_E5M2;
                break;
            case InvalidInput::NON_MX_A_SCALE:
                aScaleDtype = ge::DT_FLOAT;
                break;
            case InvalidInput::NON_MX_B_SCALE:
                bScaleDtype = ge::DT_FLOAT;
                break;
            case InvalidInput::WRONG_B_SCALE_DTYPE:
                bScaleDtype = param.hif4 ? ge::DT_FLOAT8_E8M0 : Ops::NN::DT_HIFLOAT4_SCALE;
                groupSize = bScaleDtype == ge::DT_FLOAT8_E8M0 ? 32 : -1;
                break;
            case InvalidInput::WRONG_SCALE_SHAPE:
                aScale.MutableStorageShape().SetDim(2, 1);
                break;
            case InvalidInput::WRONG_B_SCALE_SHAPE:
                bScale.MutableStorageShape().SetDim(2, param.hif4 ? 2 : 1);
                break;
            case InvalidInput::WRONG_NZ_SHAPE:
                x2.MutableStorageShape().SetDim(3, 32);
                break;
            case InvalidInput::WRONG_OUTPUT_SHAPE:
                y.MutableStorageShape().SetDim(1, n + 8);
                break;
            case InvalidInput::WRONG_BIAS_SHAPE:
                bias.MutableStorageShape().SetDim(0, 2);
                break;
            case InvalidInput::BATCHED_BIAS:
                bias = gert::StorageShape({2, 1, n}, {2, 1, n});
                break;
            case InvalidInput::WRONG_BIAS_DTYPE:
                biasDtype = ge::DT_FLOAT;
                break;
            case InvalidInput::NON_NZ_B:
                bFormat = ge::FORMAT_ND;
                break;
            case InvalidInput::WRONG_GROUP:
                groupSize = param.hif4 ? 32LL : -65536LL + 64LL;
                break;
            case InvalidInput::BATCH:
                x1.MutableOriginShape() = gert::Shape({1, m, k});
                break;
            case InvalidInput::NEGATIVE_DIMENSION:
                x1.MutableOriginShape().SetDim(0, -1);
                break;
            default:
                break;
        }
        platform_.Init();
        std::map<std::string, std::string> soc = {{"ai_core_cnt", "64"},
                                                  {"cube_core_cnt", "64"},
                                                  {"vector_core_cnt", "128"},
                                                  {"l2_size", "134217728"},
                                                  {"core_type_list", "CubeCore,VectorCore"}};
        std::map<std::string, std::string> spec = {{"ub_size", "253952"},   {"l1_size", "524288"},
                                                   {"l0_a_size", "65536"},  {"l0_b_size", "65536"},
                                                   {"l0_c_size", "262144"}, {"cube_freq", "1650"}};
        std::map<std::string, std::string> version = {
            {"Short_SoC_version", param.arch9202 ? "Ascend960PR" : "Ascend960DT"},
            {"NpuArch", param.arch9202 ? "9202" : "9201"}};
        platform_.SetPlatformRes("SoCInfo", soc);
        platform_.SetPlatformRes("AICoreSpec", spec);
        platform_.SetPlatformRes("version", version);
        platform_.SetCoreNumByCoreType("AICore");
        std::map<std::string, std::string> intrinsics = {{"Intrinsic_fix_pipe_l0c2out", "float16"},
                                                         {"Intrinsic_data_move_l12bt", "bf16"}};
        platform_.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
        raw_ = gert::TilingData::CreateCap(param.capacity);
        workspace_ = gert::ContinuousVector::Create<size_t>(1);
        std::vector<gert::StorageShape*> inputs = {&x1,
                                                   &x2,
                                                   param.bias ? &bias : nullptr,
                                                   param.invalid == InvalidInput::MISSING_A_SCALE ? nullptr : &aScale,
                                                   param.invalid == InvalidInput::MISSING_B_SCALE ? nullptr : &bScale,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr};
        if (param.forbiddenInput != 0) {
            inputs.at(param.forbiddenInput) = &offset;
        }
        holder_ = gert::TilingContextFaker()
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputs)
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo_)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_))
                      .NodeInputTd(0, aDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.hif4 ? Ops::NN::DT_HIFLOAT4 : ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, bFormat)
                      .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, aScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, bScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.output, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(param.output)},
                                  {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"transpose_x1",
                                   Ops::NN::AnyValue::CreateFrom<bool>(param.invalid == InvalidInput::TRANSPOSE_A)},
                                  {"transpose_x2",
                                   Ops::NN::AnyValue::CreateFrom<bool>(param.invalid != InvalidInput::TRANSPOSE_B)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
                      .TilingData(raw_.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace_.get()))
                      .SetOpType("QuantBatchMatmulV4")
                      .Build();
        if (auto* context = Get()) {
            // The SDK builder requires CompileInfo, but these tests exercise platform-based routing.
            const auto* nodeInfo = context->GetComputeNodeInfo();
            const size_t compileInfoIndex = nodeInfo->GetInputsNum() + nodeInfo->GetOutputsNum();
            holder_.GetContext<gert::KernelContext>()->MutableInput(compileInfoIndex)->Set(nullptr, nullptr);
        }
    }
    gert::TilingContext* Get() { return holder_.GetContext<gert::TilingContext>(); }
    bool IsValid()
    {
        const auto* context = Get();
        return context != nullptr && context->GetCompileInfo() == nullptr && context->GetPlatformInfo() == &platform_;
    }

private:
    optiling::QuantBatchMatmulV4CompileInfo compileInfo_{};
    fe::PlatFormInfos platform_;
    decltype(gert::TilingData::CreateCap(64)) raw_;
    decltype(gert::ContinuousVector::Create<size_t>(1)) workspace_;
    gert::KernelRunContextHolder holder_;
};
} // namespace

TEST(QuantBatchMatmulV4MxBlaze, ConstructorAndResetAlwaysUseCompactData)
{
    MxV4Probe tiling(nullptr);
    EXPECT_EQ(tiling.TilingDataSize(), sizeof(MxTilingData));
    EXPECT_EQ(tiling.CompactData().k, 0U);
    EXPECT_EQ(tiling.CompactData().windowLen, 4U);
    tiling.ConfigureL1Case(true, true, 512UL * 1024UL);
    tiling.SetTilingData();
    EXPECT_EQ(tiling.CompactData().m, 64U);
    EXPECT_EQ(tiling.CompactData().n, 128U);
    EXPECT_EQ(tiling.CompactData().k, 96U);
    EXPECT_EQ(tiling.CompactData().baseM, 64U);
    EXPECT_EQ(tiling.CompactData().baseN, 128U);
    EXPECT_EQ(tiling.CompactData().baseK, 64U);
    EXPECT_EQ(tiling.CompactData().isBias, 1U);
    EXPECT_EQ(tiling.CompactData().biasDtype, static_cast<uint8_t>(ge::DT_FLOAT16));
    EXPECT_EQ(tiling.CompactData().dbL0C, 2U);
    ASSERT_EQ(tiling.DoLibApiTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.CompactData().k, 96U);
    EXPECT_EQ(tiling.CompactData().scaleKL1, 128U);

    tiling.ResetTilingData();
    EXPECT_EQ(tiling.TilingDataSize(), sizeof(MxTilingData));
    const auto& data = tiling.CompactData();
    EXPECT_EQ(data.m, 0U);
    EXPECT_EQ(data.n, 0U);
    EXPECT_EQ(data.k, 0U);
    EXPECT_EQ(data.baseK, 0U);
    EXPECT_EQ(data.kAL1, 0U);
    EXPECT_EQ(data.kBL1, 0U);
    EXPECT_EQ(data.scaleKL1, 0U);
    EXPECT_EQ(data.nBufferNum, 0U);
    EXPECT_EQ(data.isBias, 0U);
    EXPECT_EQ(data.mBaseTailSplitCnt, 1U);
    EXPECT_EQ(data.nBaseTailSplitCnt, 1U);
    EXPECT_EQ(data.weightMustHitL2, 1U);
    EXPECT_EQ(data.windowLen, 4U);
}

TEST(QuantBatchMatmulV4MxBlaze, FourBuffersRequireSpareL1AndPreserveSolverWindows)
{
    for (bool hif4 : {false, true}) {
        for (bool fullLoad : {false, true}) {
            // kL1=64, scaleKL1=min(3*2*64, 2*1*64)=128; logical K=96.
            const uint64_t bDataBytes = 128UL * 64UL / 2UL * 4UL;
            const uint64_t bScaleBytes = 128UL * 2UL * (hif4 ? 4UL : 2UL) * 2UL;
            const uint64_t biasBytes = 128UL * 2UL * 2UL;
            const uint64_t aBytes = fullLoad ? 64UL * 128UL + 64UL * 4UL : 64UL * 64UL * 4UL + 64UL * 4UL * 2UL;
            const uint64_t fourBufferBytes = aBytes + bDataBytes + bScaleBytes + biasBytes;
            for (uint64_t spareBytes : {0UL, 1UL}) {
                SCOPED_TRACE(::testing::Message()
                             << "hif4=" << hif4 << ", fullLoad=" << fullLoad << ", spareBytes=" << spareBytes);
                MxV4Probe tiling(nullptr);
                tiling.ConfigureL1Case(hif4, fullLoad, fourBufferBytes + spareBytes);
                tiling.SetTilingData();
                ASSERT_EQ(tiling.DoLibApiTiling(), ge::GRAPH_SUCCESS);
                const auto& data = tiling.CompactData();
                EXPECT_EQ(data.k, 96U);
                EXPECT_EQ(data.baseM, 64U);
                EXPECT_EQ(data.baseN, 128U);
                EXPECT_EQ(data.baseK, 64U);
                EXPECT_EQ(data.isBias, 1U);
                EXPECT_EQ(data.dbL0C, 2U);
                EXPECT_EQ(data.kAL1, 64U);
                EXPECT_EQ(data.kBL1, 64U);
                EXPECT_EQ(data.scaleKL1, 128U);
                EXPECT_EQ(data.nBufferNum, spareBytes == 0UL ? 2U : 4U);
                EXPECT_TRUE(tiling.HasOriginalL1Windows());
            }
        }
    }
}

TEST(QuantBatchMatmulV4MxBlaze, AFullLoadUsesArchitectureWindow)
{
    MxV4Probe tiling(nullptr);
    for (auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        EXPECT_TRUE(tiling.EvaluateAFullLoad(arch, 4UL)); // 64 cores have an ASW window length of 8.
        EXPECT_FALSE(tiling.EvaluateAFullLoad(arch, 8UL));
        EXPECT_FALSE(tiling.EvaluateAFullLoad(arch, 3UL)); // M blocks must divide the core count.
        EXPECT_FALSE(tiling.EvaluateAFullLoad(arch, 0UL));
    }
}

TEST(QuantBatchMatmulV4MxBlaze, BothArchitecturesSerializeMxA8W4ModesAndLogicalKTails)
{
    for (bool arch9202 : {false, true}) {
        for (bool hif4 : {false, true}) {
            for (auto output : {ge::DT_FLOAT16, ge::DT_BF16}) {
                for (bool bias : {false, true}) {
                    for (int64_t k : {8, 64, 96, 192}) {
                        SCOPED_TRACE(::testing::Message() << "arch9202=" << arch9202 << ", hif4=" << hif4
                                                          << ", output=" << output << ", bias=" << bias << ", k=" << k);
                        MxContext context({hif4, bias, arch9202, k, output});
                        ASSERT_TRUE(context.IsValid());
                        MxV4Probe tiling(context.Get());
                        ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
                        ASSERT_EQ(context.Get()->GetRawTilingData()->GetDataSize(), 64U);
                        MxTilingData data{};
                        std::memcpy(&data, context.Get()->GetRawTilingData()->GetData(), sizeof(data));
                        EXPECT_EQ(data.m, 33U);
                        EXPECT_EQ(data.n, 24U);
                        EXPECT_EQ(data.k, static_cast<uint32_t>(k));
                        EXPECT_EQ(data.baseK % 64U, 0U);
                        EXPECT_GT(data.baseK, 0U);
                        EXPECT_EQ(data.groupSizeM, 1U);
                        EXPECT_EQ(data.groupSizeN, 1U);
                        EXPECT_EQ(data.groupSizeK, hif4 ? 64U : 32U);
                        EXPECT_EQ(data.x1QuantMode, 8U);
                        EXPECT_EQ(data.x2QuantMode, hif4 ? 64U : 8U);
                        EXPECT_EQ(data.isBias, bias);
                        EXPECT_EQ(data.windowLen, 8U);
                        EXPECT_TRUE(context.Get()->GetTilingKey() == MX_CUBE_SCALE_ND_KEY ||
                                    context.Get()->GetTilingKey() == MX_CUBE_SCALE_ND_A_FULL_LOAD_KEY);
                        EXPECT_GT(context.Get()->GetBlockDim(), 0U);
                        EXPECT_LE(context.Get()->GetBlockDim(), 64U);
                    }
                }
            }
        }
    }
}

TEST(QuantBatchMatmulV4MxBlaze, Fp4UsesOnlyLow16BitsOfGroupSize)
{
    // M=7, N=23 for the first encoding; all bits above K are set in the negative encoding.
    for (int64_t groupSize : {(7LL << 32) | (23LL << 16) | 32LL, -65536LL + 32LL}) {
        SCOPED_TRACE(::testing::Message() << "group_size=" << groupSize);
        MxCase param;
        param.fp4GroupSize = groupSize;
        MxContext context(param);
        ASSERT_TRUE(context.IsValid());
        MxV4Probe tiling(context.Get());
        ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
        const auto& data = tiling.CompactData();
        EXPECT_EQ(data.groupSizeM, 1U);
        EXPECT_EQ(data.groupSizeN, 1U);
        EXPECT_EQ(data.groupSizeK, 32U);
    }
}

TEST(QuantBatchMatmulV4MxBlaze, FinalCompactRefreshPreservesArchitectureWindow)
{
    for (bool arch9202 : {false, true}) {
        MxContext context({true, false, arch9202, 96});
        ASSERT_TRUE(context.IsValid());
        auto* platform = context.Get()->GetPlatformInfo();
        ASSERT_NE(platform, nullptr);
        // V3 maps s8s4 to DAV_RESV. The MX window must still use the actual 9201/9202 architecture.
        std::map<std::string, std::string> intrinsics = {{"Intrinsic_fix_pipe_l0c2out", "float16"},
                                                         {"Intrinsic_data_move_l12bt", "bf16"},
                                                         {"Intrinsic_mmad", "s8s4"}};
        platform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
        std::string intrinsic;
        ASSERT_TRUE(platform->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", intrinsic));
        ASSERT_EQ(intrinsic, "s8s4");
        MxV4Probe tiling(context.Get());
        ASSERT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_SUCCESS);
        ASSERT_EQ(tiling.GetPlatformInfo(), ge::GRAPH_SUCCESS);
        ASSERT_TRUE(tiling.IsCapable());
        ASSERT_EQ(tiling.DoOpTiling(), ge::GRAPH_SUCCESS);
        EXPECT_EQ(tiling.CompactData().windowLen, 8U);
        const auto weightMustHitL2 = tiling.CompactData().weightMustHitL2;
        ASSERT_EQ(tiling.DoLibApiTiling(), ge::GRAPH_SUCCESS);
        EXPECT_EQ(tiling.CompactData().windowLen, 8U);
        EXPECT_EQ(tiling.CompactData().weightMustHitL2, weightMustHitL2);
        tiling.ResetTilingData();
        EXPECT_EQ(tiling.CompactData().windowLen, 4U);
    }
}

TEST(QuantBatchMatmulV4MxBlaze, RejectsInvalidContractBeforeSolving)
{
    for (bool hif4 : {false, true}) {
        for (auto invalid :
             {InvalidInput::MISSING_A_SCALE, InvalidInput::MISSING_B_SCALE, InvalidInput::WRONG_B_SCALE_DTYPE,
              InvalidInput::WRONG_SCALE_SHAPE, InvalidInput::WRONG_B_SCALE_SHAPE, InvalidInput::WRONG_NZ_SHAPE,
              InvalidInput::WRONG_OUTPUT_SHAPE, InvalidInput::WRONG_BIAS_SHAPE, InvalidInput::BATCHED_BIAS,
              InvalidInput::WRONG_BIAS_DTYPE, InvalidInput::NON_NZ_B, InvalidInput::TRANSPOSE_A,
              InvalidInput::TRANSPOSE_B, InvalidInput::WRONG_GROUP, InvalidInput::BATCH,
              InvalidInput::NEGATIVE_DIMENSION, InvalidInput::UNALIGNED_N, InvalidInput::UNALIGNED_K}) {
            SCOPED_TRACE(::testing::Message() << "hif4=" << hif4 << ", invalid=" << static_cast<int>(invalid));
            MxContext context({hif4, true, false, 192, ge::DT_FLOAT16, invalid});
            ASSERT_TRUE(context.IsValid());
            if (invalid == InvalidInput::MISSING_A_SCALE || invalid == InvalidInput::MISSING_B_SCALE) {
                const bool missingA = invalid == InvalidInput::MISSING_A_SCALE;
                const auto absent = missingA ? optiling::X1_SCALE_INDEX_V4 : optiling::X2_SCALE_INDEX_V4;
                const auto present = missingA ? optiling::X2_SCALE_INDEX_V4 : optiling::X1_SCALE_INDEX_V4;
                ASSERT_EQ(context.Get()->GetOptionalInputDesc(absent), nullptr);
                ASSERT_EQ(context.Get()->GetOptionalInputShape(absent), nullptr);
                ASSERT_NE(context.Get()->GetOptionalInputDesc(present), nullptr);
                ASSERT_NE(context.Get()->GetOptionalInputShape(present), nullptr);
            }
            MxV4Probe tiling(context.Get());
            EXPECT_NE(tiling.DoTiling(), ge::GRAPH_SUCCESS);
            EXPECT_FALSE(tiling.Inputs().initFlag);
            if (invalid == InvalidInput::BATCHED_BIAS) {
                EXPECT_EQ(tiling.Inputs().batchBias, 1UL);
            }
            if (invalid == InvalidInput::WRONG_B_SCALE_DTYPE) {
                // The group matches x2_scale, so tiling reaches the data/scale dtype check.
                EXPECT_EQ(tiling.Inputs().scaleDtype, hif4 ? ge::DT_FLOAT8_E8M0 : Ops::NN::DT_HIFLOAT4_SCALE);
            }
        }
    }
}

TEST(QuantBatchMatmulV4MxBlaze, RejectsForbiddenInputsThroughRegistry)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    for (uint32_t index : {optiling::Y_SCALE_INDEX_V4, optiling::X1_OFFSET_INDEX_V4, optiling::X2_OFFSET_INDEX_V4,
                           optiling::Y_OFFSET_INDEX_V4, optiling::X2_TABLE_INDEX_V4}) {
        SCOPED_TRACE(::testing::Message() << "forbidden IR input=" << index);
        MxCase param;
        param.forbiddenInput = index;
        MxContext context(param);
        ASSERT_TRUE(context.IsValid());
        ASSERT_NE(context.Get()->GetOptionalInputDesc(index), nullptr);
        ASSERT_NE(context.Get()->GetOptionalInputShape(index), nullptr);
        MxV4Probe tiling(context.Get());
        EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_PARAM_INVALID);
        EXPECT_FALSE(tiling.Inputs().initFlag);
        EXPECT_EQ(impl->tiling(context.Get()), ge::GRAPH_FAILED);
    }
}

TEST(QuantBatchMatmulV4MxBlaze, RegistryPublishesMxKeysForBothWeightTypes)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    for (bool arch9202 : {false, true}) {
        for (bool hif4 : {false, true}) {
            for (bool aFullLoad : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "arch9202=" << arch9202 << ", hif4=" << hif4 << ", aFullLoad=" << aFullLoad);
                MxCase param{hif4, false, arch9202, 192};
                if (aFullLoad) {
                    // One M block and more than 64 N blocks qualify for A full load on both platforms.
                    param.m = 16;
                    param.n = 65536;
                }
                MxContext context(param);
                ASSERT_TRUE(context.IsValid());
                ASSERT_EQ(impl->tiling(context.Get()), ge::GRAPH_SUCCESS);
                EXPECT_EQ(context.Get()->GetTilingKey(),
                          aFullLoad ? MX_CUBE_SCALE_ND_A_FULL_LOAD_KEY : MX_CUBE_SCALE_ND_KEY);
            }
        }
    }
}

TEST(QuantBatchMatmulV4MxBlaze, RegistryRejectsIncompleteOrMismatchedMxPairs)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    for (bool arch9202 : {false, true}) {
        for (bool hif4 : {false, true}) {
            for (auto invalid :
                 {InvalidInput::MISSING_A_SCALE, InvalidInput::MISSING_B_SCALE, InvalidInput::WRONG_A_DTYPE,
                  InvalidInput::NON_MX_A_SCALE, InvalidInput::NON_MX_B_SCALE, InvalidInput::WRONG_B_SCALE_DTYPE}) {
                SCOPED_TRACE(::testing::Message() << "arch9202=" << arch9202 << ", hif4=" << hif4
                                                  << ", invalid=" << static_cast<int>(invalid));
                MxContext context({hif4, false, arch9202, 192, ge::DT_FLOAT16, invalid});
                ASSERT_TRUE(context.IsValid());
                EXPECT_EQ(impl->tiling(context.Get()), ge::GRAPH_FAILED);
            }
        }
    }
}

TEST(QuantBatchMatmulV4MxBlaze, ReusedTemplatePublishesSecondContextsMxModeAndTail)
{
    MxCase fp4{false, true, false, 192};
    fp4.m = 16;
    fp4.n = 262144 + 8; // 1025 N blocks. The final round splits its single block along N.
    MxContext first(fp4);
    ASSERT_TRUE(first.IsValid());
    optiling::AdaptiveSlidingWindowWeightQuantMxCubeTiling tiling(first.Get());
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(first.Get()->GetTilingKey(), MX_CUBE_SCALE_ND_A_FULL_LOAD_KEY);
    MxTilingData data{};
    auto* raw = first.Get()->GetRawTilingData();
    ASSERT_EQ(raw->GetDataSize(), sizeof(data));
    std::memcpy(&data, raw->GetData(), sizeof(data));
    ASSERT_EQ(data.x2QuantMode, 8U);
    ASSERT_EQ(data.isBias, 1U);
    ASSERT_GT(data.nTailTile, 1U);

    MxCase hif4{true, false, true, 96, ge::DT_BF16};
    hif4.m = 4096;
    hif4.n = 16384; // 16 x 64 blocks fill whole 64-core rounds. Both tail factors must return to 1.
    MxContext second(hif4);
    ASSERT_TRUE(second.IsValid());
    tiling.Reset(second.Get());
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(second.Get()->GetTilingKey(), MX_CUBE_SCALE_ND_KEY);
    raw = second.Get()->GetRawTilingData();
    ASSERT_EQ(raw->GetDataSize(), sizeof(data));
    std::memcpy(&data, raw->GetData(), sizeof(data));
    EXPECT_EQ(data.m, 4096U);
    EXPECT_EQ(data.n, 16384U);
    EXPECT_EQ(data.k, 96U);
    EXPECT_EQ(data.x1QuantMode, 8U);
    EXPECT_EQ(data.x2QuantMode, 64U);
    EXPECT_EQ(data.isBias, 0U);
    EXPECT_EQ(data.mTailTile, 1U);
    EXPECT_EQ(data.nTailTile, 1U);
}

TEST(QuantBatchMatmulV4MxBlaze, KeysKeepV4SixArgumentLayout)
{
    MxV4Probe tiling(nullptr);
    tiling.SetAFullLoad(false);
    EXPECT_EQ(tiling.GetTilingKey(), MX_CUBE_SCALE_ND_KEY);
    tiling.SetAFullLoad(true);
    EXPECT_EQ(tiling.GetTilingKey(), MX_CUBE_SCALE_ND_A_FULL_LOAD_KEY);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4MxBlaze, CompactBufferCapacityMatches64ByteAbi)
{
    for (size_t capacity : {32UL, 56UL, 63UL, 64UL}) {
        SCOPED_TRACE(::testing::Message() << "capacity=" << capacity);
        MxContext context({false, false, false, 192, ge::DT_FLOAT16, InvalidInput::NONE, capacity});
        ASSERT_TRUE(context.IsValid());
        MxV4Probe tiling(context.Get());
        if (capacity < sizeof(MxTilingData)) {
            EXPECT_NE(tiling.DoTiling(), ge::GRAPH_SUCCESS);
        } else {
            ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
            EXPECT_EQ(context.Get()->GetRawTilingData()->GetDataSize(), 64U);
            EXPECT_EQ(tiling.CompactData().windowLen, 8U);
        }
    }
}
