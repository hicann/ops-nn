/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "any_value.h"

namespace AscendC {
static constexpr auto dec = ::dec;
}

#include "data_utils.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/lp_loss_tiling_data.h"
#include "../../../op_kernel/lp_loss.h"

#ifdef DT_FLOAT
#undef DT_FLOAT
#endif
#ifdef DT_FLOAT16
#undef DT_FLOAT16
#endif
#ifdef FORMAT_ND
#undef FORMAT_ND
#endif

namespace {

enum class LpLossTestDtype : uint32_t
{
    FLOAT32 = 0,
    FLOAT16 = 1,
    BF16 = 2,
};

static LpLossTestDtype gCurrentDtype = LpLossTestDtype::FLOAT32;

struct LpLossCompileInfo {
};

extern "C" __global__ __aicore__ void lp_loss(
    GM_ADDR predict, GM_ADDR label, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LpLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(LpLossTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    auto runKernel = [&](auto dtypeTag) {
        using DType = decltype(dtypeTag);
        if (tilingData.bufferNum == 2) {
            if (tilingData.reduction == 0) {
                MyLpLoss::KernelLpLoss<DType, DType, 2, 0> op;
                op.InitNone(predict, label, loss, &tilingData, &pipe);
                op.Process();
            } else {
                GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
                MyLpLoss::KernelLpLoss<DType, DType, 2, 1> op;
                op.InitReduce(predict, label, loss, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        } else {
            if (tilingData.reduction == 0) {
                MyLpLoss::KernelLpLoss<DType, DType, 1, 0> op;
                op.InitNone(predict, label, loss, &tilingData, &pipe);
                op.Process();
            } else {
                GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
                MyLpLoss::KernelLpLoss<DType, DType, 1, 1> op;
                op.InitReduce(predict, label, loss, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
    };

    if (gCurrentDtype == LpLossTestDtype::FLOAT16) {
        runKernel(half{});
    } else if (gCurrentDtype == LpLossTestDtype::BF16) {
        runKernel(bfloat16_t{});
    } else {
        runKernel(float{});
    }
}

static std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(int64_t p, const std::string& reduction)
{
    using Ops::NN::AnyValue;
    return {{"p", AnyValue::CreateFrom<int64_t>(p)}, {"reduction", AnyValue::CreateFrom<std::string>(reduction)}};
}

static float SumFloat(const std::vector<float>& data)
{
    float sum = 0.0f;
    for (float value : data) {
        sum += value;
    }
    return sum;
}

static float MeanFloat(const std::vector<float>& data)
{
    if (data.empty()) {
        return 0.0f;
    }
    return SumFloat(data) / static_cast<float>(data.size());
}

static void ExpectNearRel(float actual, float expected, float rtol, float atol)
{
    float diff = std::fabs(actual - expected);
    float tol = atol + rtol * std::fabs(expected);
    EXPECT_LE(diff, tol);
}

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
std::vector<T> MakeInputData(uint32_t dataCount, float offset)
{
    std::vector<T> data(dataCount);
    for (uint32_t index = 0; index < dataCount; ++index) {
        float value = (static_cast<int32_t>(index % 97U) - 48) * 0.125f + offset;
        data[index] = static_cast<T>(value);
    }
    return data;
}

template <typename T>
std::vector<float> ComputeL1Golden(const std::vector<T>& predict, const std::vector<T>& label)
{
    std::vector<float> golden(predict.size());
    for (size_t index = 0; index < predict.size(); ++index) {
        float predictValue = static_cast<float>(predict[index]);
        float labelValue = static_cast<float>(label[index]);
        golden[index] = std::fabs(predictValue - labelValue);
    }
    return golden;
}

template <typename T>
std::vector<float> ReadOutputAsFloat(const uint8_t* output, uint32_t dataCount)
{
    const T* typedOutput = reinterpret_cast<const T*>(output);
    std::vector<float> result(dataCount);
    for (uint32_t index = 0; index < dataCount; ++index) {
        result[index] = static_cast<float>(typedOutput[index]);
    }
    return result;
}

static void ExpectVectorNear(
    const std::vector<float>& actual, const std::vector<float>& expected, float rtol, float atol)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        float diff = std::fabs(actual[index] - expected[index]);
        float tol = atol + rtol * std::fabs(expected[index]);
        EXPECT_LE(diff, tol) << "index=" << index;
    }
}

} // namespace

class LpLossTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "lp_loss_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "lp_loss_test TearDown" << std::endl;
    }
};

TEST_F(LpLossTest, test_case_float16_none)
{
    LpLossCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "LpLoss",
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        MakeAttrs(1, "none"), &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint32_t dataCount = 128 * 64;
    std::vector<half> predictData = MakeInputData<half>(dataCount, 0.5f);
    std::vector<half> labelData = MakeInputData<half>(dataCount, -0.75f);
    std::vector<float> golden = ComputeL1Golden(predictData, labelData);
    size_t inputByteSize = dataCount * sizeof(half);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(predict, predictData.data(), inputByteSize);

    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(label, labelData.data(), inputByteSize);

    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    std::memset(loss, 0, CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    gCurrentDtype = LpLossTestDtype::FLOAT16;
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(lp_loss, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    std::vector<float> actual = ReadOutputAsFloat<half>(loss, dataCount);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)label);
    AscendC::GmFree((void*)loss);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ExpectVectorNear(actual, golden, 1e-3f, 1e-3f);
}

TEST_F(LpLossTest, test_case_float32_none)
{
    LpLossCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "LpLoss",
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1, "none"), &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint32_t dataCount = 256 * 33;
    std::vector<float> predictData = MakeInputData<float>(dataCount, 1.25f);
    std::vector<float> labelData = MakeInputData<float>(dataCount, -0.25f);
    std::vector<float> golden = ComputeL1Golden(predictData, labelData);
    size_t inputByteSize = dataCount * sizeof(float);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(predict, predictData.data(), inputByteSize);

    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(label, labelData.data(), inputByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    std::memset(loss, 0, CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    gCurrentDtype = LpLossTestDtype::FLOAT32;
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(lp_loss, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    std::vector<float> actual = ReadOutputAsFloat<float>(loss, dataCount);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)label);
    AscendC::GmFree((void*)loss);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ExpectVectorNear(actual, golden, 1e-5f, 1e-6f);
}

TEST_F(LpLossTest, test_case_float32_sum)
{
    LpLossCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "LpLoss",
        {
            {{{128, 32}, {128, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 32}, {128, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1, "sum"), &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint32_t dataCount = 128 * 32;
    std::vector<float> predictData = MakeInputData<float>(dataCount, 0.0f);
    std::vector<float> labelData = MakeInputData<float>(dataCount, 2.0f);
    std::vector<float> golden = ComputeL1Golden(predictData, labelData);
    size_t inputByteSize = dataCount * sizeof(float);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(predict, predictData.data(), inputByteSize);

    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(label, labelData.data(), inputByteSize);

    size_t outputByteSize = sizeof(float);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    std::memset(loss, 0, CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    gCurrentDtype = LpLossTestDtype::FLOAT32;
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(lp_loss, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    float actual = ReadOutputAsFloat<float>(loss, 1)[0];

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)label);
    AscendC::GmFree((void*)loss);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    float expected = SumFloat(golden);
    ExpectNearRel(actual, expected, 1e-5f, 1e-6f);
}

TEST_F(LpLossTest, test_case_float32_mean)
{
    LpLossCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "LpLoss",
        {
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1, "mean"), &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint32_t dataCount = 64 * 64;
    std::vector<float> predictData = MakeInputData<float>(dataCount, -1.0f);
    std::vector<float> labelData = MakeInputData<float>(dataCount, 0.75f);
    std::vector<float> golden = ComputeL1Golden(predictData, labelData);
    size_t inputByteSize = dataCount * sizeof(float);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(predict, predictData.data(), inputByteSize);

    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    std::memcpy(label, labelData.data(), inputByteSize);

    size_t outputByteSize = sizeof(float);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    std::memset(loss, 0, CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    gCurrentDtype = LpLossTestDtype::FLOAT32;
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(lp_loss, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    float actual = ReadOutputAsFloat<float>(loss, 1)[0];

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)label);
    AscendC::GmFree((void*)loss);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    float expected = MeanFloat(golden);
    ExpectNearRel(actual, expected, 1e-5f, 1e-6f);
}