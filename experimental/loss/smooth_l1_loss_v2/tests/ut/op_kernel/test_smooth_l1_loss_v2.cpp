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
 * \file test_sqrt.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include <cstdint>
#include <fstream>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "any_value.h"

// Fix dec ambiguity in UT build (std::dec vs tikicpulib::dec)
namespace AscendC {
static constexpr auto dec = ::dec;
}

#include "data_utils.h"
#include "tiling_case_executor.h"
#include "../op_host/smooth_l1_loss_v2_tiling.h"
#include "../../../op_kernel/smooth_l1_loss_v2.h"

#ifdef DT_FLOAT
#undef DT_FLOAT
#endif
#ifdef DT_FLOAT16
#undef DT_FLOAT16
#endif
#ifdef FORMAT_ND
#undef FORMAT_ND
#endif

namespace fs = std::filesystem;

extern "C" __global__ __aicore__ void smooth_l1_loss_v2(GM_ADDR predict,
                                                        GM_ADDR label,
                                                        GM_ADDR loss,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SmoothL1LossV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SmoothL1LossV2TilingData, tilingData, tiling);
    MySmoothL1LossV2::KernelSmoothL1LossV2<DTYPE_PREDICT, DTYPE_LABEL, DTYPE_LOSS> op;
    op.Init(predict, label, loss,
            tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
            tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
            tilingData.tileDataNum, tilingData.smallTailDataNum,
            tilingData.bigTailDataNum, tilingData.tailBlockNum,
            tilingData.totalDataNum, tilingData.sigma, tilingData.reduction);
    op.Process();
}

static bool EnsureSmoothL1LossV2DataDir()
{
    const fs::path dst = fs::current_path() / "smooth_l1_loss_v2_data";
    if (fs::exists(dst) && !fs::is_directory(dst)) {
        fs::remove(dst);
    }

    const fs::path srcFromFile = fs::path(__FILE__).parent_path() / "smooth_l1_loss_v2_data";
    if (fs::exists(srcFromFile) && fs::is_directory(srcFromFile)) {
        fs::create_directories(dst);
        fs::copy(srcFromFile, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        return true;
    }

    const fs::path cwd = fs::current_path();
    const fs::path candidates[] = {
        cwd / "../../../experimental/math/smooth_l1_loss_v2/tests/ut/op_kernel/smooth_l1_loss_v2_data",
        cwd / "../../../../experimental/math/smooth_l1_loss_v2/tests/ut/op_kernel/smooth_l1_loss_v2_data",
        cwd / "../../../../../experimental/math/smooth_l1_loss_v2/tests/ut/op_kernel/smooth_l1_loss_v2_data",
    };
    for (const auto& src : candidates) {
        if (fs::exists(src) && fs::is_directory(src)) {
            fs::create_directories(dst);
            fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
            return true;
        }
    }

    std::fprintf(stderr,
                 "[SmoothL1LossV2Test] cannot locate smooth_l1_loss_v2_data, cwd=%s, __FILE__=%s\n",
                 fs::current_path().c_str(), __FILE__);
    return false;
}

static std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(float sigma, const std::string& reduction)
{
    using Ops::Math::AnyValue;
    return {
        {"sigma", AnyValue::CreateFrom<float>(sigma)},
        {"reduction", AnyValue::CreateFrom<std::string>(reduction)}
    };
}

class SmoothL1LossV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "smooth_l1_loss_v2_test SetUp" << std::endl;
        ASSERT_TRUE(EnsureSmoothL1LossV2DataDir());
        system("chmod -R 755 ./smooth_l1_loss_v2_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "smooth_l1_loss_v2_test TearDown" << std::endl;
    }

};

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

static std::vector<float> ReadGoldenFloat32(const std::string& fileName)
{
    std::ifstream ifs(fileName, std::ios::binary);
    if (!ifs) {
        return {};
    }
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    if (!ifs.read(reinterpret_cast<char*>(data.data()), size)) {
        return {};
    }
    return data;
}

static float ReadOutputFloat32(const std::string& fileName)
{
    std::ifstream ifs(fileName, std::ios::binary);
    float value = 0.0f;
    if (!ifs) {
        return value;
    }
    ifs.read(reinterpret_cast<char*>(&value), sizeof(float));
    return value;
}

static float SumFloat(const std::vector<float>& data)
{
    float sum = 0.0f;
    for (float v : data) {
        sum += v;
    }
    return sum;
}

static void ExpectNearRel(float actual, float expected, float rtol, float atol)
{
    float diff = std::fabs(actual - expected);
    float tol = atol + rtol * std::fabs(expected);
    EXPECT_LE(diff, tol);
}

TEST_F(SmoothL1LossV2Test, test_case_float16_1)
{
    optiling::SmoothL1LossV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 0}; // sigma=1.0, reduction=none
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossV2",
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // predict
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // label
        },
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // loss
        },
        MakeAttrs(1.0f, "none"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_v2_data/ && python3 gen_data.py '(128, 64)' 'float16' '1.0'");
    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(half);

    std::string fileName = "./smooth_l1_loss_v2_data/float16_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_v2_data/float16_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_v2, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    fileName = "./smooth_l1_loss_v2_data/float16_output_t_loss.bin";
    WriteFile(fileName, loss, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(loss));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./smooth_l1_loss_v2_data/ && python3 compare_data.py 'float16'");
}

TEST_F(SmoothL1LossV2Test, test_case_float32_1)
{
    optiling::SmoothL1LossV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 0};
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossV2",
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1.0f, "none"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_v2_data/ && python3 gen_data.py '(256, 33)' 'float32' '1.0'");
    uint32_t dataCount = 256 * 33;
    size_t inputByteSize = dataCount * sizeof(float);

    std::string fileName = "./smooth_l1_loss_v2_data/float32_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_v2_data/float32_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_v2, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    fileName = "./smooth_l1_loss_v2_data/float32_output_t_loss.bin";
    WriteFile(fileName, loss, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(loss));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./smooth_l1_loss_v2_data/ && python3 compare_data.py 'float32'");
}

TEST_F(SmoothL1LossV2Test, test_case_float32_sum)
{
    optiling::SmoothL1LossV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 1}; // reduction=sum
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossV2",
        {
            {{{128, 32}, {128, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 32}, {128, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1.0f, "sum"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_v2_data/ && python3 gen_data.py '(128, 32)' 'float32' '1.0'");
    uint32_t dataCount = 128 * 32;
    size_t inputByteSize = dataCount * sizeof(float);

    std::string fileName = "./smooth_l1_loss_v2_data/float32_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_v2_data/float32_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    size_t outputByteSize = sizeof(float);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_v2, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    fileName = "./smooth_l1_loss_v2_data/float32_output_t_loss.bin";
    WriteFile(fileName, loss, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(loss));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    std::vector<float> golden = ReadGoldenFloat32("./smooth_l1_loss_v2_data/float32_golden_t_loss.bin");
    ASSERT_FALSE(golden.empty());
    float expected = SumFloat(golden);
    float actual = ReadOutputFloat32("./smooth_l1_loss_v2_data/float32_output_t_loss.bin");
    ExpectNearRel(actual, expected, 1e-5f, 1e-6f);
}

TEST_F(SmoothL1LossV2Test, test_case_float32_mean)
{
    optiling::SmoothL1LossV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 2}; // reduction=mean
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossV2",
        {
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1.0f, "mean"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_v2_data/ && python3 gen_data.py '(64, 64)' 'float32' '1.0'");
    uint32_t dataCount = 64 * 64;
    size_t inputByteSize = dataCount * sizeof(float);

    std::string fileName = "./smooth_l1_loss_v2_data/float32_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_v2_data/float32_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    size_t outputByteSize = sizeof(float);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_v2, tilingInfo.blockNum, predict, label, loss, workspace, tiling);

    fileName = "./smooth_l1_loss_v2_data/float32_output_t_loss.bin";
    WriteFile(fileName, loss, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(loss));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    std::vector<float> golden = ReadGoldenFloat32("./smooth_l1_loss_v2_data/float32_golden_t_loss.bin");
    ASSERT_FALSE(golden.empty());
    float expected = SumFloat(golden) / static_cast<float>(golden.size());
    float actual = ReadOutputFloat32("./smooth_l1_loss_v2_data/float32_output_t_loss.bin");
    ExpectNearRel(actual, expected, 1e-5f, 1e-6f);
}
