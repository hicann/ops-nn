/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define __CCE_UT_TEST__

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_host/fused_sgd_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void fused_sgd(
    GM_ADDR params, GM_ADDR grads, GM_ADDR momentum_buffer_list,
    GM_ADDR grad_scale,
    GM_ADDR params_ref, GM_ADDR grads_ref, GM_ADDR momentum_buffer_list_ref,
    GM_ADDR workspace, GM_ADDR tiling);

class FusedSgdKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "FusedSgdKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "FusedSgdKernelTest TearDown" << endl; }
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
uint8_t* CreateNormTensorList(const std::vector<std::vector<uint64_t>>& shapeInfos, char* d_type, const char* tag = "fused_sgd")
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
        fileName << "./sgd_data/" << d_type << "_input_t_" << tag << "_" << i << ".bin";
        ReadFile(fileName.str(), dataSize, dataPtr, dataSize);
        *(tensorListDesc + addrIndex) = (uint64_t)dataPtr;
    }
    return (uint8_t*)tensorListDesc;
}

template <typename T>
void FreeNormTensorList(uint8_t* addr, const std::vector<std::vector<uint64_t>>& shapeInfos, char* d_type, const char* tag = "fused_sgd")
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
        fileName << "./sgd_data/" << d_type << "_output_t_" << tag << "_" << i << ".bin";
        WriteFile(fileName.str(), tensorAddr, shapeSize * sizeof(T));
        AscendC::GmFree((void*)(tensorAddr));
    }
    AscendC::GmFree((void*)addr);
}

TEST_F(FusedSgdKernelTest, test_fp32_basic)
{
    size_t tilingSize = sizeof(FusedSgdTilingData);
    uint32_t blockDim = 1;
    std::vector<std::vector<uint64_t>> shapeInfos = {{4}};

    system(
        "cp -rf "
        "../../../../optim/fused_sgd/tests/ut/op_kernel/sgd_data ./");
    system("chmod -R 755 ./sgd_data/");
    system("cd ./sgd_data/ && python3 gen_data.py '{{4}}' 'float32'");

    uint8_t* paramsBuf = CreateNormTensorList<float>(shapeInfos, "float32", "params");
    uint8_t* gradsBuf = CreateNormTensorList<float>(shapeInfos, "float32", "grads");
    uint8_t* momentumBuf = CreateNormTensorList<float>(shapeInfos, "float32", "momentum");
    uint8_t* paramsRefBuf = CreateNormTensorList<float>(shapeInfos, "float32", "params_ref");
    uint8_t* gradsRefBuf = CreateNormTensorList<float>(shapeInfos, "float32", "grads_ref");
    uint8_t* momentumRefBuf = CreateNormTensorList<float>(shapeInfos, "float32", "momentum_ref");
    uint8_t* gradScaleBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* gradScalePtr = reinterpret_cast<float*>(gradScaleBuf);
    gradScalePtr[0] = 0.5f;

    FusedSgdTilingData* tilingData = reinterpret_cast<FusedSgdTilingData*>(tiling);
    tilingData->weightDecay = 0.01f;
    tilingData->momentum = 0.5f;
    tilingData->lr = 0.001f;
    tilingData->dampening = 0.0f;
    tilingData->nesterov = 0;
    tilingData->maximize = 0;
    tilingData->isFirstStep = 1;
    tilingData->useGradScale = 1;
    tilingData->useMomentum = 1;
    tilingData->tensorNum = 1;
    tilingData->tensorsPerCore = 1;
    tilingData->usedCoreNum = 1;
    tilingData->coreCalcMax = 4088;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fused_sgd, blockDim,
                paramsBuf, gradsBuf, momentumBuf,
                gradScaleBuf,
                paramsRefBuf, gradsRefBuf, momentumRefBuf,
                workspace, (uint8_t*)(tiling));

    FreeNormTensorList<float>(paramsRefBuf, shapeInfos, "float32", "params_ref");
    FreeNormTensorList<float>(gradsRefBuf, shapeInfos, "float32", "grads_ref");
    FreeNormTensorList<float>(momentumRefBuf, shapeInfos, "float32", "momentum_ref");
    FreeNormTensorList<float>(paramsBuf, shapeInfos, "float32", "params_ref");
    FreeNormTensorList<float>(gradsBuf, shapeInfos, "float32", "grads_ref");
    FreeNormTensorList<float>(momentumBuf, shapeInfos, "float32", "momentum_ref");
    AscendC::GmFree(gradScaleBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./sgd_data/ && python3 compare_data.py 'float32'");
}

TEST_F(FusedSgdKernelTest, test_fp16_basic)
{
    size_t tilingSize = sizeof(FusedSgdTilingData);
    uint32_t blockDim = 1;
    std::vector<std::vector<uint64_t>> shapeInfos = {{4}};

    system(
        "cp -rf "
        "../../../../optim/fused_sgd/tests/ut/op_kernel/sgd_data ./");
    system("chmod -R 755 ./sgd_data/");
    system("cd ./sgd_data/ && python3 gen_data.py '{{4}}' 'float16'");

    uint8_t* paramsBuf = CreateNormTensorList<half>(shapeInfos, "float16", "params");
    uint8_t* gradsBuf = CreateNormTensorList<half>(shapeInfos, "float16", "grads");
    uint8_t* momentumBuf = CreateNormTensorList<half>(shapeInfos, "float16", "momentum");
    uint8_t* paramsRefBuf = CreateNormTensorList<half>(shapeInfos, "float16", "params_ref");
    uint8_t* gradsRefBuf = CreateNormTensorList<half>(shapeInfos, "float16", "grads_ref");
    uint8_t* momentumRefBuf = CreateNormTensorList<half>(shapeInfos, "float16", "momentum_ref");
    uint8_t* gradScaleBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* gradScalePtr = reinterpret_cast<float*>(gradScaleBuf);
    gradScalePtr[0] = 0.5f;

    FusedSgdTilingData* tilingData = reinterpret_cast<FusedSgdTilingData*>(tiling);
    tilingData->weightDecay = 0.01f;
    tilingData->momentum = 0.5f;
    tilingData->lr = 0.001f;
    tilingData->dampening = 0.0f;
    tilingData->nesterov = 0;
    tilingData->maximize = 0;
    tilingData->isFirstStep = 1;
    tilingData->useGradScale = 1;
    tilingData->useMomentum = 1;
    tilingData->tensorNum = 1;
    tilingData->tensorsPerCore = 1;
    tilingData->usedCoreNum = 1;
    tilingData->coreCalcMax = 4088;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fused_sgd, blockDim,
                paramsBuf, gradsBuf, momentumBuf,
                gradScaleBuf,
                paramsRefBuf, gradsRefBuf, momentumRefBuf,
                workspace, (uint8_t*)(tiling));

    FreeNormTensorList<half>(paramsRefBuf, shapeInfos, "float16", "params_ref");
    FreeNormTensorList<half>(gradsRefBuf, shapeInfos, "float16", "grads_ref");
    FreeNormTensorList<half>(momentumRefBuf, shapeInfos, "float16", "momentum_ref");
    FreeNormTensorList<half>(paramsBuf, shapeInfos, "float16", "params_ref");
    FreeNormTensorList<half>(gradsBuf, shapeInfos, "float16", "grads_ref");
    FreeNormTensorList<half>(momentumBuf, shapeInfos, "float16", "momentum_ref");
    AscendC::GmFree(gradScaleBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./sgd_data/ && python3 compare_data.py 'float16'");
}
