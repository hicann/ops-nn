/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_list.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "scatter_list_tiling_def.h"
#include "tensor_list_operate.h"

using namespace std;

extern "C" __global__ __aicore__ void scatter_list(
    GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR varOut, GM_ADDR workspace, GM_ADDR tiling);

class scatter_list_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_list_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scatter_list_test TearDown\n" << endl;
    }
};

TEST_F(scatter_list_test, test_case_200)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data.py 8 4 4096 1 256");

    std::vector<std::vector<uint64_t>> varShape = {{4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256},
                                                   {4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 4 * 1 * 256 * sizeof(int8_t);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<int8_t>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 32;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 4;
    tilingData->varDim2Count = 4096;
    tilingData->dim2Count = 1;
    tilingData->dim3Count = 256;
    tilingData->dim3CountAlign = 256;
    tilingData->updatesOneBlock = 32;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 256;
    tilingData->srcBatchStrideAlign = 256;
    tilingData->dstBatchStride = 1048576;
    tilingData->useCoreNum = 32;
    tilingData->preCoreBatchNum = 1;
    tilingData->lastCoreBatchNum = 1;
    tilingData->eachLoopNum = 0;
    tilingData->eachPreLoopEle = 0;
    tilingData->eachLastLoopEle = 0;
    tilingData->eachLastLoopEleAlign = 0;
    tilingData->updatesCount = 256;
    tilingData->updatesUbSize = 256;
    tilingData->tilingKey = 200;

    ICPU_SET_TILING_KEY(200);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_210)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data.py 8 16 4096 256 256");

    std::vector<std::vector<uint64_t>> varShape = {{16, 4096, 256}, {16, 4096, 256}, {16, 4096, 256}, {16, 4096, 256},
                                                   {16, 4096, 256}, {16, 4096, 256}, {16, 4096, 256}, {16, 4096, 256}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 16 * 256 * 256 * sizeof(int8_t);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<int8_t>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 32;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 16;
    tilingData->varDim2Count = 4096;
    tilingData->dim2Count = 256;
    tilingData->dim3Count = 256;
    tilingData->dim3CountAlign = 256;
    tilingData->updatesOneBlock = 32;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 65536;
    tilingData->srcBatchStrideAlign = 65536;
    tilingData->dstBatchStride = 1048576;
    tilingData->useCoreNum = 32;
    tilingData->preCoreBatchNum = 4;
    tilingData->lastCoreBatchNum = 4;
    tilingData->eachLoopNum = 0;
    tilingData->eachPreLoopEle = 0;
    tilingData->eachLastLoopEle = 0;
    tilingData->eachLastLoopEleAlign = 0;
    tilingData->updatesCount = 65536;
    tilingData->updatesUbSize = 65536;
    tilingData->tilingKey = 210;

    ICPU_SET_TILING_KEY(210);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_211)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data.py 8 16 4096 255 255");

    std::vector<std::vector<uint64_t>> varShape = {{16, 4096, 255}, {16, 4096, 255}, {16, 4096, 255}, {16, 4096, 255},
                                                   {16, 4096, 255}, {16, 4096, 255}, {16, 4096, 255}, {16, 4096, 255}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 16 * 255 * 255 * sizeof(int8_t);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<int8_t>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 31;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 16;
    tilingData->varDim2Count = 4096;
    tilingData->dim2Count = 255;
    tilingData->dim3Count = 255;
    tilingData->dim3CountAlign = 256;
    tilingData->updatesOneBlock = 32;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 65025;
    tilingData->srcBatchStrideAlign = 65056;
    tilingData->dstBatchStride = 1044480;
    tilingData->useCoreNum = 32;
    tilingData->preCoreBatchNum = 4;
    tilingData->lastCoreBatchNum = 4;
    tilingData->eachLoopNum = 0;
    tilingData->eachPreLoopEle = 0;
    tilingData->eachLastLoopEle = 0;
    tilingData->eachLastLoopEleAlign = 0;
    tilingData->updatesCount = 65056;
    tilingData->updatesUbSize = 65056;
    tilingData->tilingKey = 211;

    ICPU_SET_TILING_KEY(211);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_220)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data.py 8 1 4096 1024 256");

    std::vector<std::vector<uint64_t>> varShape = {{1, 4096, 256}, {1, 4096, 256}, {1, 4096, 256}, {1, 4096, 256},
                                                   {1, 4096, 256}, {1, 4096, 256}, {1, 4096, 256}, {1, 4096, 256}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 1 * 1024 * 256 * sizeof(int8_t);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<int8_t>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 32;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 1;
    tilingData->varDim2Count = 4096;
    tilingData->dim2Count = 1024;
    tilingData->dim3Count = 256;
    tilingData->dim3CountAlign = 256;
    tilingData->updatesOneBlock = 32;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 262144;
    tilingData->srcBatchStrideAlign = 262144;
    tilingData->dstBatchStride = 1048576;
    tilingData->useCoreNum = 32;
    tilingData->preCoreBatchNum = 256;
    tilingData->lastCoreBatchNum = 256;
    tilingData->eachLoopNum = 0;
    tilingData->eachPreLoopEle = 0;
    tilingData->eachLastLoopEle = 0;
    tilingData->eachLastLoopEleAlign = 0;
    tilingData->updatesCount = 65536;
    tilingData->updatesUbSize = 65536;
    tilingData->tilingKey = 220;

    ICPU_SET_TILING_KEY(220);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_230)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data.py 8 4 4096 1024 256");

    std::vector<std::vector<uint64_t>> varShape = {{4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256},
                                                   {4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256}, {4, 4096, 256}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 4 * 1024 * 256 * sizeof(int8_t);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<int8_t>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 32;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 4;
    tilingData->varDim2Count = 4096;
    tilingData->dim2Count = 1024;
    tilingData->dim3Count = 256;
    tilingData->dim3CountAlign = 256;
    tilingData->updatesOneBlock = 32;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 262144;
    tilingData->srcBatchStrideAlign = 262144;
    tilingData->dstBatchStride = 1048576;
    tilingData->useCoreNum = 32;
    tilingData->preCoreBatchNum = 1;
    tilingData->lastCoreBatchNum = 1;
    tilingData->eachLoopNum = 2;
    tilingData->eachPreLoopEle = 131072;
    tilingData->eachLastLoopEle = 131072;
    tilingData->eachLastLoopEleAlign = 0;
    tilingData->updatesCount = 131072;
    tilingData->updatesUbSize = 131072;
    tilingData->tilingKey = 230;

    ICPU_SET_TILING_KEY(230);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_231)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data.py 8 4 4096 1024 255");

    std::vector<std::vector<uint64_t>> varShape = {{4, 4096, 255}, {4, 4096, 255}, {4, 4096, 255}, {4, 4096, 255},
                                                   {4, 4096, 255}, {4, 4096, 255}, {4, 4096, 255}, {4, 4096, 255}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 4 * 1024 * 255 * sizeof(int8_t);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<int8_t>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 32;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 4;
    tilingData->varDim2Count = 4096;
    tilingData->dim2Count = 1024;
    tilingData->dim3Count = 255;
    tilingData->dim3CountAlign = 256;
    tilingData->updatesOneBlock = 32;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 261120;
    tilingData->srcBatchStrideAlign = 261120;
    tilingData->dstBatchStride = 1044480;
    tilingData->useCoreNum = 32;
    tilingData->preCoreBatchNum = 1;
    tilingData->lastCoreBatchNum = 1;
    tilingData->eachLoopNum = 2;
    tilingData->eachPreLoopEle = 130560;
    tilingData->eachLastLoopEle = 130560;
    tilingData->eachLastLoopEleAlign = 130560;
    tilingData->updatesCount = 130560;
    tilingData->updatesUbSize = 130560;
    tilingData->tilingKey = 231;

    ICPU_SET_TILING_KEY(231);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_100)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data_neg.py [8,4,256,4096] [8,4,256,1]");

    std::vector<std::vector<uint64_t>> varShape = {{4, 256, 4096}, {4, 256, 4096}, {4, 256, 4096}, {4, 256, 4096},
                                                   {4, 256, 4096}, {4, 256, 4096}, {4, 256, 4096}, {4, 256, 4096}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 4 * 256 * 1 * sizeof(half);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<float>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 32;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 4;
    tilingData->varDim2Count = 256;
    tilingData->dim2Count = 256;
    tilingData->dim3Count = 1;
    tilingData->dim3CountAlign = 16;
    tilingData->updatesOneBlock = 16;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 256;
    tilingData->srcBatchStrideAlign = 0;
    tilingData->dstBatchStride = 1048576;
    tilingData->useCoreNum = 32;
    tilingData->preCoreBatchNum = 1;
    tilingData->lastCoreBatchNum = 1;
    tilingData->eachLoopNum = 0;
    tilingData->eachPreLoopEle = 0;
    tilingData->eachLastLoopEle = 0;
    tilingData->eachLastLoopEleAlign = 0;
    tilingData->updatesCount = 0;
    tilingData->updatesUbSize = 512;
    tilingData->dataUbSize = 8192;
    tilingData->transposeUbSize = 8192;
    tilingData->transRepeatTimes = 16;
    tilingData->transRepeatTimesTail = 0;
    tilingData->updateDim23Align = 256;
    tilingData->preCoreUpdateDim23 = 256;
    tilingData->varDim3Stride = 255;
    tilingData->varDim3Count = 4096;
    tilingData->dim3CountSize = 0;
    tilingData->eachLastSize = 256;
    tilingData->tilingKey = 100;

    ICPU_SET_TILING_KEY(100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_101)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data_neg.py [8,400,1024,32] [8,400,1024,1]");

    std::vector<std::vector<uint64_t>> varShape = {{400, 1024, 32}, {400, 1024, 32}, {400, 1024, 32}, {400, 1024, 32},
                                                   {400, 1024, 32}, {400, 1024, 32}, {400, 1024, 32}, {400, 1024, 32}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 8 * 400 * 1024 * 1 * sizeof(half);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<float>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 8;
    tilingData->dim1Count = 400;
    tilingData->varDim2Count = 1024;
    tilingData->dim2Count = 1024;
    tilingData->dim3Count = 1;
    tilingData->dim3CountAlign = 16;
    tilingData->updatesOneBlock = 16;
    tilingData->indiceDims = 1;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 1024;
    tilingData->srcBatchStrideAlign = 0;
    tilingData->dstBatchStride = 32768;
    tilingData->useCoreNum = 48;
    tilingData->preCoreBatchNum = 67;
    tilingData->lastCoreBatchNum = 51;
    tilingData->eachLoopNum = 0;
    tilingData->eachPreLoopEle = 0;
    tilingData->eachLastLoopEle = 0;
    tilingData->eachLastLoopEleAlign = 0;
    tilingData->updatesCount = 0;
    tilingData->updatesUbSize = 2048;
    tilingData->dataUbSize = 32768;
    tilingData->transposeUbSize = 32768;
    tilingData->transRepeatTimes = 64;
    tilingData->transRepeatTimesTail = 0;
    tilingData->updateDim23Align = 1024;
    tilingData->preCoreUpdateDim23 = 68608;
    tilingData->varDim3Stride = 1;
    tilingData->varDim3Count = 32;
    tilingData->dim3CountSize = 0;
    tilingData->eachLastSize = 1024;
    tilingData->tilingKey = 101;

    ICPU_SET_TILING_KEY(101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_list_test, test_case_107)
{
    system(
        "cp -rf "
        "../../../../index/scatter_list/tests/ut/op_kernel/scatter_list_data "
        "./");
    system("chmod -R 755 ./scatter_list_data/");
    system("cd ./scatter_list_data/ && python3 gen_data_neg.py [2,1,1,120016] [2,1,1,120000]");

    std::vector<std::vector<uint64_t>> varShape = {{1, 1, 120016}, {1, 1, 120000}};

    size_t indiceByteSize = 8 * sizeof(int32_t);
    size_t updatesByteSize = 2 * 1 * 1 * 120000 * sizeof(half);
    size_t maskByteSize = 8 * sizeof(int32_t); // 实际只需要8 * sizeof(uint8_t)，为了补齐到一个block才这样设置
    size_t tilingSize = sizeof(ScatterListTilingData);

    uint8_t* var = CreateTensorList<float>(varShape);
    uint8_t* indice = (uint8_t*)AscendC::GmAlloc(indiceByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* varOut = var;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 2;

    ScatterListTilingData* tilingData = reinterpret_cast<ScatterListTilingData*>(tiling);

    tilingData->dim0Count = 2;
    tilingData->dim1Count = 1;
    tilingData->varDim2Count = 1;
    tilingData->dim2Count = 1;
    tilingData->dim3Count = 120000;
    tilingData->dim3CountAlign = 7500;
    tilingData->updatesOneBlock = 16;
    tilingData->indiceDims = 2;
    tilingData->indiceCount = 8;
    tilingData->indiceUbSize = 32;
    tilingData->maskCount = 32;
    tilingData->maskUbSize = 32;
    tilingData->srcBatchStride = 120000;
    tilingData->srcBatchStrideAlign = 0;
    tilingData->dstBatchStride = 120016;
    tilingData->useCoreNum = 2;
    tilingData->preCoreBatchNum = 1;
    tilingData->lastCoreBatchNum = 1;
    tilingData->eachLoopNum = 1;
    tilingData->eachPreLoopEle = 98208;
    tilingData->eachLastLoopEle = 21792;
    tilingData->eachLastLoopEleAlign = 21792;
    tilingData->updatesCount = 0;
    tilingData->updatesUbSize = 196416;
    tilingData->dataUbSize = 512;
    tilingData->transposeUbSize = 512;
    tilingData->transRepeatTimes = 1;
    tilingData->transRepeatTimesTail = 0;
    tilingData->updateDim23Align = 120000;
    tilingData->preCoreUpdateDim23 = 120000;
    tilingData->varDim3Stride = 32;
    tilingData->varDim3Count = 120016;
    tilingData->dim3CountSize = 240000;
    tilingData->eachLastSize = 43584;
    tilingData->tilingKey = 107;

    ICPU_SET_TILING_KEY(107);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_list, blockDim, var, indice, updates, mask, varOut, workspace, tiling);

    FreeTensorList<int8_t>(var, varShape);
    AscendC::GmFree(indice);
    AscendC::GmFree(updates);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}