/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef __CCE_KT_TEST__
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#include "data_utils.h"
#include "tikicpulib.h"
#include <string>
#include <iostream>
#include "string.h"
#endif
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_kernel/fused_adamw_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void fused_adamw(GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs,
                                                  GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps, GM_ADDR grad_scale,
                                                  GM_ADDR found_inf, GM_ADDR params_ref, GM_ADDR exp_avgs_ref,
                                                  GM_ADDR exp_avg_sqs_ref, GM_ADDR max_exp_avg_sqs_ref,
                                                  GM_ADDR workspace, GM_ADDR tiling);

class FusedAdamWKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "FusedAdamWKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "FusedAdamWKernelTest TearDown" << endl; }
};

template <typename T1, typename T2>
inline T1 CeilA2B(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
uint8_t* CreateNormTensorList(const std::vector<std::vector<uint64_t>>& shapeInfos, const char* d_type,
                              const char* tag = "fused_adamw")
{
    uint64_t tensorListDescCount = 1 + shapeInfos.size() * 2;
    for (auto s : shapeInfos) {
        tensorListDescCount += s.size();
    }
    std::vector<uint64_t> shapeSizeList;
    uint64_t* tensorListDesc = (uint64_t*)AscendC::GmAlloc(tensorListDescCount * sizeof(uint64_t));
    *tensorListDesc = (tensorListDescCount - shapeInfos.size()) * sizeof(uint64_t);
    uint64_t addrIndex = 0;
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        addrIndex++;
        uint16_t dimCount = shapeInfos[i].size();
        *(tensorListDesc + addrIndex) = ((uint64_t)(i) << 32) + dimCount;
        uint64_t shapeSize = 1;
        for (size_t j = 0; j < dimCount; j++) {
            addrIndex++;
            *(tensorListDesc + addrIndex) = shapeInfos[i][j];
            shapeSize *= shapeInfos[i][j];
        }
        shapeSizeList.push_back(shapeSize);
    }
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        addrIndex++;
        uint64_t dataSize = shapeSizeList[i] * sizeof(T);
        uint8_t* dataPtr = (uint8_t*)AscendC::GmAlloc(CeilA2B(dataSize, 32) * 32);
        std::stringstream fileName;
        fileName << "./adamw_data/" << d_type << "_input_t_" << tag << "_" << i << ".bin";
        ReadFile(fileName.str(), dataSize, dataPtr, dataSize);
        *(tensorListDesc + addrIndex) = (uint64_t)dataPtr;
    }
    return (uint8_t*)tensorListDesc;
}

template <typename T>
void FreeNormTensorList(uint8_t* addr, const std::vector<std::vector<uint64_t>>& shapeInfos, const char* d_type,
                        const char* tag = "fused_adamw")
{
    uint64_t dataPtrOffset = *((uint64_t*)addr);
    uint8_t* dataAddr = addr + dataPtrOffset;
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        uint64_t shapeSize = 1;
        for (size_t j = 0; j < shapeInfos[i].size(); j++) {
            shapeSize *= shapeInfos[i][j];
        }
        uint8_t* tensorAddr = (uint8_t*)(*((uint64_t*)(dataAddr) + i));
        std::stringstream fileName;
        fileName << "./adamw_data/" << d_type << "_output_t_" << tag << "_" << i << ".bin";
        WriteFile(fileName.str(), tensorAddr, shapeSize * sizeof(T));
        AscendC::GmFree((void*)(tensorAddr));
    }
    AscendC::GmFree((void*)addr);
}

template <typename T>
void RunFusedAdamWTest(const std::vector<std::vector<uint64_t>>& shapeInfos, const char* d_type, bool amsgrad = false)
{
    size_t tilingSize = sizeof(FusedAdamWTilingData);
    uint32_t blockDim = 1;

    system("cp -rf "
           "../../../../optim/fused_adamw/tests/ut/op_kernel/adamw_data ./");
    system("chmod -R 755 ./adamw_data/");
    std::stringstream cmd;
    cmd << "cd ./adamw_data/ && python3 gen_data.py '";
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        cmd << "{";
        for (size_t j = 0; j < shapeInfos[i].size(); j++) {
            cmd << shapeInfos[i][j];
            if (j < shapeInfos[i].size() - 1) {
                cmd << ",";
            }
        }
        cmd << "}";
    }
    cmd << "' '" << d_type << "'";
    system(cmd.str().c_str());

    uint8_t* paramsBuf = CreateNormTensorList<T>(shapeInfos, d_type, "params");
    uint8_t* gradsBuf = CreateNormTensorList<T>(shapeInfos, d_type, "grads");
    uint8_t* expAvgsBuf = CreateNormTensorList<T>(shapeInfos, d_type, "exp_avgs");
    uint8_t* expAvgSqsBuf = CreateNormTensorList<T>(shapeInfos, d_type, "exp_avg_sqs");
    uint8_t* maxExpAvgSqsBuf = nullptr;
    if (amsgrad) {
        maxExpAvgSqsBuf = CreateNormTensorList<T>(shapeInfos, d_type, "max_exp_avg_sqs");
    }
    uint8_t* stateStepsBuf = CreateNormTensorList<float>(shapeInfos, d_type, "state_steps");
    uint8_t* paramsRefBuf = CreateNormTensorList<T>(shapeInfos, d_type, "params_ref");
    uint8_t* expAvgsRefBuf = CreateNormTensorList<T>(shapeInfos, d_type, "exp_avgs_ref");
    uint8_t* expAvgSqsRefBuf = CreateNormTensorList<T>(shapeInfos, d_type, "exp_avg_sqs_ref");
    uint8_t* maxExpAvgSqsRefBuf = nullptr;
    if (amsgrad) {
        maxExpAvgSqsRefBuf = CreateNormTensorList<T>(shapeInfos, d_type, "max_exp_avg_sqs_ref");
    }

    uint8_t* gradScaleBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* foundInfBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* gradScalePtr = reinterpret_cast<float*>(gradScaleBuf);
    gradScalePtr[0] = 0.5f;
    float* foundInfPtr = reinterpret_cast<float*>(foundInfBuf);
    foundInfPtr[0] = 0.0f;

    FusedAdamWTilingData* tilingData = reinterpret_cast<FusedAdamWTilingData*>(tiling);
    // 根据实际数据量计算 coreCalcMax，对齐到 8(FP32) 或 16(FP16/BF16)
    uint64_t dataCount = 1;
    for (auto dim : shapeInfos[0]) {
        dataCount *= dim;
    }
    uint64_t alignSize = (sizeof(T) == sizeof(float)) ? 8 : 16;
    uint64_t coreCalcMax = (dataCount + alignSize - 1) / alignSize * alignSize;
    if (coreCalcMax == 0) {
        coreCalcMax = alignSize;
    }

    tilingData->lr = 0.001f;
    tilingData->beta1 = 0.9f;
    tilingData->beta2 = 0.999f;
    tilingData->weightDecay = 0.01f;
    tilingData->eps = 1e-8f;
    tilingData->amsgrad = amsgrad ? 1 : 0;
    tilingData->maximize = 0;
    tilingData->useGradScale = 1;
    tilingData->useFoundInf = 0;
    tilingData->tensorNum = 1;
    tilingData->tensorsPerCore = 1;
    tilingData->usedCoreNum = 1;
    tilingData->coreCalcMax = coreCalcMax;
    tilingData->stepCount = 0;
    tilingData->usedRealCoreNum = 1;
    tilingData->lastCoreTensor = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // tiling key: 0=FP32, 1=FP16, 2=BF16  UT中使用tilingkey去区分不同分支，实际kernel使用宏去判断分支，减少.o大小
    if constexpr (std::is_same_v<T, half>) {
        ICPU_SET_TILING_KEY(1);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        ICPU_SET_TILING_KEY(2);
    } else {
        ICPU_SET_TILING_KEY(0);
    }
    ICPU_RUN_KF(fused_adamw, blockDim, paramsBuf, gradsBuf, expAvgsBuf, expAvgSqsBuf, maxExpAvgSqsBuf, stateStepsBuf,
                gradScaleBuf, foundInfBuf, paramsRefBuf, expAvgsRefBuf, expAvgSqsRefBuf, maxExpAvgSqsRefBuf, workspace,
                (uint8_t*)(tiling));

    FreeNormTensorList<T>(paramsRefBuf, shapeInfos, d_type, "params_ref");
    FreeNormTensorList<T>(expAvgsRefBuf, shapeInfos, d_type, "exp_avgs_ref");
    FreeNormTensorList<T>(expAvgSqsRefBuf, shapeInfos, d_type, "exp_avg_sqs_ref");
    if (amsgrad) {
        FreeNormTensorList<T>(maxExpAvgSqsRefBuf, shapeInfos, d_type, "max_exp_avg_sqs_ref");
    }
    FreeNormTensorList<T>(paramsBuf, shapeInfos, d_type, "params");
    FreeNormTensorList<T>(gradsBuf, shapeInfos, d_type, "grads");
    FreeNormTensorList<T>(expAvgsBuf, shapeInfos, d_type, "exp_avgs");
    FreeNormTensorList<T>(expAvgSqsBuf, shapeInfos, d_type, "exp_avg_sqs");
    if (amsgrad) {
        FreeNormTensorList<T>(maxExpAvgSqsBuf, shapeInfos, d_type, "max_exp_avg_sqs");
    }
    FreeNormTensorList<float>(stateStepsBuf, shapeInfos, d_type, "state_steps");
    AscendC::GmFree(gradScaleBuf);
    AscendC::GmFree(foundInfBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    std::stringstream compareCmd;
    compareCmd << "cd ./adamw_data/ && python3 compare_data.py '" << d_type << "'";
    system(compareCmd.str().c_str());
}

TEST_F(FusedAdamWKernelTest, test_fp32_basic)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{4}};
    RunFusedAdamWTest<float>(shapeInfos, "float32");
}

TEST_F(FusedAdamWKernelTest, test_fp32_large)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{2, 64}};
    RunFusedAdamWTest<float>(shapeInfos, "float32");
}

TEST_F(FusedAdamWKernelTest, test_fp16_basic)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{4}};
    RunFusedAdamWTest<half>(shapeInfos, "float16");
}

TEST_F(FusedAdamWKernelTest, test_fp16_large)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{2, 64}};
    RunFusedAdamWTest<half>(shapeInfos, "float16");
}

TEST_F(FusedAdamWKernelTest, test_bf16_basic)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{4}};
    RunFusedAdamWTest<bfloat16_t>(shapeInfos, "bfloat16");
}

TEST_F(FusedAdamWKernelTest, test_bf16_large)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{2, 64}};
    RunFusedAdamWTest<bfloat16_t>(shapeInfos, "bfloat16");
}
