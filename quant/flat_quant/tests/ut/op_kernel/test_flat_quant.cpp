/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_upsample_nearest.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "flat_quant_tiling_def.h"

extern "C" __global__ __aicore__ void flat_quant(
    GM_ADDR x, GM_ADDR kronecker_p1, GM_ADDR kronecker_p2, GM_ADDR out, GM_ADDR quant_scale, GM_ADDR workspace, GM_ADDR tiling);

class flat_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "flat_quant_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "flat_quant_test TearDown\n" << std::endl;
    }
};

TEST_F(flat_quant_test, test_case_float16_1)
{
    system(
        "cp -rf "
        "../../../../quant/flat_quant/tests/ut/op_kernel/flat_quant_data ./");
    system("chmod -R 755 ./flat_quant_data/");
    system("cd ./flat_quant_data/ && python3 gen_data.py '(1, 128, 128)' '(128, 128)' '(128, 128)' 'float16'");

    size_t xByteSize = 1 * 128 * 128 * sizeof(half);
    size_t p1ByteSize = 128 * 128 * sizeof(half);
    size_t p2ByteSize = 128 * 128 * sizeof(half);
    size_t outByteSize = 1 * 128 * 128 * sizeof(int8_t) / 2;
    size_t scaleByteSize = 1 * sizeof(float);
    size_t workspaceSize = 1 * 128 * 128 * sizeof(half) + 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kronecker_p1 = (uint8_t*)AscendC::GmAlloc(p1ByteSize);
    uint8_t* kronecker_p2 = (uint8_t*)AscendC::GmAlloc(p2ByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* quant_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FlatQuantTilingData));

    ReadFile("./flat_quant_data/float16_input_x_flat_quant.bin", xByteSize, x, xByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p1_flat_quant.bin", p1ByteSize, kronecker_p1, p1ByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p2_flat_quant.bin", p2ByteSize, kronecker_p2, p2ByteSize);

    FlatQuantTilingData* tilingDatafromBin = reinterpret_cast<FlatQuantTilingData*>(tiling);

    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->K = 1;
    tilingDatafromBin->M = 128;
    tilingDatafromBin->N = 128;
    tilingDatafromBin->clipRatio = 1.0f;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(
        flat_quant, blockDim, x, kronecker_p1, kronecker_p2, out, quant_scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    WriteFile("./flat_quant_data/float16_output_quant_scale_flat_quant.bin", quant_scale, scaleByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(kronecker_p1));
    AscendC::GmFree((void*)(kronecker_p2));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)(quant_scale));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./flat_quant_data/ && python3 compare_data.py");
}

TEST_F(flat_quant_test, test_case_float16_2)
{
    system(
        "cp -rf "
        "../../../../quant/flat_quant/tests/ut/op_kernel/flat_quant_data ./");
    system("chmod -R 755 ./flat_quant_data/");
    system("cd ./flat_quant_data/ && python3 gen_data.py '(4, 16, 16)' '(16, 16)' '(16, 16)' 'float16'");

    size_t xByteSize = 4 * 16 * 16 * sizeof(half);
    size_t p1ByteSize = 16 * 16 * sizeof(half);
    size_t p2ByteSize = 16 * 16 * sizeof(half);
    size_t outByteSize = 4 * 16 * 16 * sizeof(int8_t) / 2;
    size_t scaleByteSize = 4 * sizeof(float);
    size_t workspaceSize = (4 * 16 * 16 + 32 * 32) * sizeof(half) + 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kronecker_p1 = (uint8_t*)AscendC::GmAlloc(p1ByteSize);
    uint8_t* kronecker_p2 = (uint8_t*)AscendC::GmAlloc(p2ByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* quant_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FlatQuantTilingData));

    ReadFile("./flat_quant_data/float16_input_x_flat_quant.bin", xByteSize, x, xByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p1_flat_quant.bin", p1ByteSize, kronecker_p1, p1ByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p2_flat_quant.bin", p2ByteSize, kronecker_p2, p2ByteSize);

    FlatQuantTilingData* tilingDatafromBin = reinterpret_cast<FlatQuantTilingData*>(tiling);

    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->K = 4;
    tilingDatafromBin->M = 16;
    tilingDatafromBin->N = 16;
    tilingDatafromBin->clipRatio = 1.0f;

    uint32_t blockDim = 8;
    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(
        flat_quant, blockDim, x, kronecker_p1, kronecker_p2, out, quant_scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    WriteFile("./flat_quant_data/float16_output_quant_scale_flat_quant.bin", quant_scale, scaleByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(kronecker_p1));
    AscendC::GmFree((void*)(kronecker_p2));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)(quant_scale));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./flat_quant_data/ && python3 compare_data.py");
}

TEST_F(flat_quant_test, test_case_float16_3)
{
    system(
        "cp -rf "
        "../../../../quant/flat_quant/tests/ut/op_kernel/flat_quant_data ./");
    system("chmod -R 755 ./flat_quant_data/");
    system("cd ./flat_quant_data/ && python3 gen_data.py '(1, 160, 128)' '(160, 160)' '(128, 128)' 'float16'");

    size_t xByteSize = 1 * 160 * 128 * sizeof(half);
    size_t p1ByteSize = 160 * 160 * sizeof(half);
    size_t p2ByteSize = 128 * 128 * sizeof(half);
    size_t outByteSize = 1 * 160 * 128 * sizeof(int8_t) / 2;
    size_t scaleByteSize = 1 * sizeof(float);
    size_t workspaceSize = 1 * 160 * 128 * sizeof(half) + 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kronecker_p1 = (uint8_t*)AscendC::GmAlloc(p1ByteSize);
    uint8_t* kronecker_p2 = (uint8_t*)AscendC::GmAlloc(p2ByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* quant_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FlatQuantTilingData));

    ReadFile("./flat_quant_data/float16_input_x_flat_quant.bin", xByteSize, x, xByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p1_flat_quant.bin", p1ByteSize, kronecker_p1, p1ByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p2_flat_quant.bin", p2ByteSize, kronecker_p2, p2ByteSize);

    FlatQuantTilingData* tilingDatafromBin = reinterpret_cast<FlatQuantTilingData*>(tiling);

    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->K = 1;
    tilingDatafromBin->M = 160;
    tilingDatafromBin->N = 128;
    tilingDatafromBin->clipRatio = 1.0f;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(3);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(
        flat_quant, blockDim, x, kronecker_p1, kronecker_p2, out, quant_scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    WriteFile("./flat_quant_data/float16_output_quant_scale_flat_quant.bin", quant_scale, scaleByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(kronecker_p1));
    AscendC::GmFree((void*)(kronecker_p2));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)(quant_scale));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./flat_quant_data/ && python3 compare_data.py");
}

TEST_F(flat_quant_test, test_case_float16_4)
{
    system(
        "cp -rf "
        "../../../../quant/flat_quant/tests/ut/op_kernel/flat_quant_data ./");
    system("chmod -R 755 ./flat_quant_data/");
    system("cd ./flat_quant_data/ && python3 gen_data.py '(1, 256, 256)' '(256, 256)' '(256, 256)' 'float16'");

    size_t xByteSize = 1 * 256 * 256 * sizeof(half);
    size_t p1ByteSize = 256 * 256 * sizeof(half);
    size_t p2ByteSize = 256 * 256 * sizeof(half);
    size_t outByteSize = 1 * 256 * 256 * sizeof(int8_t) / 2;
    size_t scaleByteSize = 1 * sizeof(float);
    size_t workspaceSize = (1 * 256 * 256 * 2 + 256 * 256 + 256 * 256) * sizeof(half) + 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kronecker_p1 = (uint8_t*)AscendC::GmAlloc(p1ByteSize);
    uint8_t* kronecker_p2 = (uint8_t*)AscendC::GmAlloc(p2ByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* quant_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FlatQuantTilingData));

    ReadFile("./flat_quant_data/float16_input_x_flat_quant.bin", xByteSize, x, xByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p1_flat_quant.bin", p1ByteSize, kronecker_p1, p1ByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p2_flat_quant.bin", p2ByteSize, kronecker_p2, p2ByteSize);

    FlatQuantTilingData* tilingDatafromBin = reinterpret_cast<FlatQuantTilingData*>(tiling);

    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->K = 1;
    tilingDatafromBin->M = 256;
    tilingDatafromBin->N = 256;
    tilingDatafromBin->clipRatio = 1.0f;
    tilingDatafromBin->matmulTilingL.M = 256;
    tilingDatafromBin->matmulTilingL.N = 256;
    tilingDatafromBin->matmulTilingL.Ka = 256;
    tilingDatafromBin->matmulTilingL.Kb = 256;
    tilingDatafromBin->matmulTilingL.singleCoreM = 256;
    tilingDatafromBin->matmulTilingL.singleCoreN = 256;
    tilingDatafromBin->matmulTilingL.singleCoreK = 256;
    tilingDatafromBin->matmulTilingL.baseM = 128;
    tilingDatafromBin->matmulTilingL.baseN = 256;
    tilingDatafromBin->matmulTilingL.baseK = 64;
    tilingDatafromBin->matmulTilingL.depthA1 = 8;
    tilingDatafromBin->matmulTilingL.depthB1 = 4;
    tilingDatafromBin->matmulTilingL.stepM = 2;
    tilingDatafromBin->matmulTilingL.stepN = 1;
    tilingDatafromBin->matmulTilingL.stepKa = 4;
    tilingDatafromBin->matmulTilingL.stepKb = 4;
    tilingDatafromBin->matmulTilingL.shareL1Size = 262144;
    tilingDatafromBin->matmulTilingL.shareL0CSize = 131072;
    tilingDatafromBin->matmulTilingL.shareUbSize = 0;
    tilingDatafromBin->matmulTilingL.batchM = 1;
    tilingDatafromBin->matmulTilingL.batchN = 1;
    tilingDatafromBin->matmulTilingL.singleBatchM = 1;
    tilingDatafromBin->matmulTilingL.singleBatchN = 1;
    tilingDatafromBin->matmulTilingL.transLength = 0;

    tilingDatafromBin->matmulTilingR.M = 65536;
    tilingDatafromBin->matmulTilingR.N = 256;
    tilingDatafromBin->matmulTilingR.Ka = 256;
    tilingDatafromBin->matmulTilingR.Kb = 256;
    tilingDatafromBin->matmulTilingR.singleCoreM = 1024;
    tilingDatafromBin->matmulTilingR.singleCoreN = 256;
    tilingDatafromBin->matmulTilingR.singleCoreK = 256;
    tilingDatafromBin->matmulTilingR.baseM = 128;
    tilingDatafromBin->matmulTilingR.baseN = 256;
    tilingDatafromBin->matmulTilingR.baseK = 64;
    tilingDatafromBin->matmulTilingR.depthA1 = 4;
    tilingDatafromBin->matmulTilingR.depthB1 = 4;
    tilingDatafromBin->matmulTilingR.stepM = 1;
    tilingDatafromBin->matmulTilingR.stepN = 1;
    tilingDatafromBin->matmulTilingR.stepKa = 2;
    tilingDatafromBin->matmulTilingR.stepKb = 4;
    tilingDatafromBin->matmulTilingR.shareL1Size = 196608;
    tilingDatafromBin->matmulTilingR.shareL0CSize = 131072;
    tilingDatafromBin->matmulTilingR.shareUbSize = 0;
    tilingDatafromBin->matmulTilingR.batchM = 1;
    tilingDatafromBin->matmulTilingR.batchN = 1;
    tilingDatafromBin->matmulTilingR.singleBatchM = 1;
    tilingDatafromBin->matmulTilingR.singleBatchN = 1;
    tilingDatafromBin->matmulTilingR.transLength = 0;
    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(
        flat_quant, blockDim, x, kronecker_p1, kronecker_p2, out, quant_scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    WriteFile("./flat_quant_data/float16_output_quant_scale_flat_quant.bin", quant_scale, scaleByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(kronecker_p1));
    AscendC::GmFree((void*)(kronecker_p2));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)(quant_scale));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./flat_quant_data/ && python3 compare_data.py");
}

TEST_F(flat_quant_test, test_case_float16_5)
{
    system(
        "cp -rf "
        "../../../../quant/flat_quant/tests/ut/op_kernel/flat_quant_data ./");
    system("chmod -R 755 ./flat_quant_data/");
    system("cd ./flat_quant_data/ && python3 gen_data.py '(2, 1, 16)' '(1, 1)' '(16, 16)' 'float16'");

    size_t xByteSize = 2 * 1 * 16 * sizeof(half);
    size_t p1ByteSize = 1 * 1 * sizeof(half);
    size_t p2ByteSize = 16 * 16 * sizeof(half);
    size_t outByteSize = 2 * 1 * 16 * sizeof(int8_t) / 2;
    size_t scaleByteSize = 2 * sizeof(float);
    size_t workspaceSize = 2 * 16 * 16 * sizeof(half) + 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kronecker_p1 = (uint8_t*)AscendC::GmAlloc(p1ByteSize);
    uint8_t* kronecker_p2 = (uint8_t*)AscendC::GmAlloc(p2ByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* quant_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FlatQuantTilingData));

    ReadFile("./flat_quant_data/float16_input_x_flat_quant.bin", xByteSize, x, xByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p1_flat_quant.bin", p1ByteSize, kronecker_p1, p1ByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p2_flat_quant.bin", p2ByteSize, kronecker_p2, p2ByteSize);

    FlatQuantTilingData* tilingDatafromBin = reinterpret_cast<FlatQuantTilingData*>(tiling);

    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->K = 2;
    tilingDatafromBin->M = 1;
    tilingDatafromBin->N = 16;
    tilingDatafromBin->clipRatio = 1.0f;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(5);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(
        flat_quant, blockDim, x, kronecker_p1, kronecker_p2, out, quant_scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    WriteFile("./flat_quant_data/float16_output_quant_scale_flat_quant.bin", quant_scale, scaleByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(kronecker_p1));
    AscendC::GmFree((void*)(kronecker_p2));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)(quant_scale));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./flat_quant_data/ && python3 compare_data.py");
}