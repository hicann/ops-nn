/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "dynamic_quant_v2_tiling_def.h"
#include "../../../../dynamic_quant/op_kernel/arch35/dynamic_quant_arch35_tilingdata.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void dynamic_quant_v2(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                 GM_ADDR offset, GM_ADDR workSpace, GM_ADDR tiling);

class dynamic_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "dynamic_quant_v2_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "dynamic_quant_v2_test TearDown\n" << endl; }
};

TEST_F(dynamic_quant_v2_test, test_case_fp16_bf16)
{
    size_t inputByteSize = 1 * 128 * 1024 * sizeof(uint16_t);
    size_t smoothScalesByteSzie = 1024 * sizeof(uint16_t);
    size_t smoothScalesMoeByteSzie = 8 * 1024 * sizeof(uint16_t);
    size_t groupIndexMoeByteSize = 8 * sizeof(int32_t);
    size_t outputByteSize = 1 * 128 * 1024 * sizeof(int8_t);
    size_t scaleByteSize = 1 * 128 * sizeof(uint32_t);
    size_t offsetByteSize = 1 * 128 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 24;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* group_index = nullptr;
    uint8_t* smooth_scales_moe = (uint8_t*)AscendC::GmAlloc(smoothScalesMoeByteSzie);
    uint8_t* group_index_moe = (uint8_t*)AscendC::GmAlloc(groupIndexMoeByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = 24;
    tilingDatafromBin->rowLen = 1024;
    tilingDatafromBin->headCoreNum = 8;
    tilingDatafromBin->rowPerHeadCore = 6;
    tilingDatafromBin->rowPerTailCore = 5;
    tilingDatafromBin->multiRowNumHeadCore = 6;
    tilingDatafromBin->multiRowNumTailCore = 5;
    tilingDatafromBin->innerLoopEle = 6;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->groupNum = 8;
    tilingDatafromBin->alignGroupNum = 8;
    tilingDatafromBin->hasSmooth = 1;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    tilingDatafromBin->ubSize = 191 * 1024;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // bf16
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales, group_index, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    // fp16
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales, group_index, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    // bf16 db
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales, group_index, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    // fp16 db
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales, group_index, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    // large shape
    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales, group_index, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    // large shape
    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales_moe, group_index_moe, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    // large shape
    ICPU_SET_TILING_KEY(8);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales_moe, group_index_moe, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(smooth_scales_moe);
    AscendC::GmFree(group_index_moe);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_v2_test, test_case_310_01)
{
    size_t inputByteSize = 8 * 1024 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 1024 * sizeof(int8_t);
    size_t scaleByteSize = 8 * sizeof(uint32_t);
    size_t offsetByteSize = 8 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->rowLen = 0;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 0;
    tilingDatafromBin->rowPerTailCore = 0;
    tilingDatafromBin->multiRowNumHeadCore = 0;
    tilingDatafromBin->multiRowNumTailCore = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 1024;
    tilingDatafromBin->sizeX = 1024;
    tilingDatafromBin->sizeZOut = 1024;
    tilingDatafromBin->sizeCopyRow = 8;
    tilingDatafromBin->numCopyRow = 8;
    tilingDatafromBin->numHeadCore = 1;
    tilingDatafromBin->numTailCore = 7;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->alignType = 0;
    tilingDatafromBin->ubSize = 191 * 1024;

    // bf16
    ICPU_SET_TILING_KEY(10100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, nullptr, nullptr, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_v2_test, test_case_310_02)
{
    size_t inputByteSize = 10 * 1088 * sizeof(uint16_t);
    size_t outputByteSize = 10 * 1088 * sizeof(int8_t);
    size_t scaleByteSize = 10 * sizeof(uint32_t);
    size_t offsetByteSize = 10 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->rowLen = 0;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 0;
    tilingDatafromBin->rowPerTailCore = 0;
    tilingDatafromBin->multiRowNumHeadCore = 0;
    tilingDatafromBin->multiRowNumTailCore = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 1088;
    tilingDatafromBin->sizeX = 1088;
    tilingDatafromBin->sizeZOut = 1088;
    tilingDatafromBin->sizeCopyRow = 8;
    tilingDatafromBin->numCopyRow = 8;
    tilingDatafromBin->numHeadCore = 1;
    tilingDatafromBin->numTailCore = 7;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 2;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->alignType = 1;
    tilingDatafromBin->ubSize = 191 * 1024;

    // bf16
    ICPU_SET_TILING_KEY(11000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, nullptr, nullptr, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_v2_test, test_case_310_03)
{
    size_t inputByteSize = 8 * 7168 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 7168 * sizeof(int8_t);
    size_t scaleByteSize = 8 * sizeof(uint32_t);
    size_t offsetByteSize = 8 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->rowLen = 0;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 0;
    tilingDatafromBin->rowPerTailCore = 0;
    tilingDatafromBin->multiRowNumHeadCore = 0;
    tilingDatafromBin->multiRowNumTailCore = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 7168;
    tilingDatafromBin->sizeX = 7168;
    tilingDatafromBin->sizeZOut = 7168;
    tilingDatafromBin->sizeCopyRow = 8;
    tilingDatafromBin->numCopyRow = 8;
    tilingDatafromBin->numHeadCore = 1;
    tilingDatafromBin->numTailCore = 7;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->innerLoopEle = 4096;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 3072;
    tilingDatafromBin->alignType = 1;
    tilingDatafromBin->ubSize = 191 * 1024;

    // bf16
    ICPU_SET_TILING_KEY(11001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, nullptr, nullptr, output, scale, offset, workSpace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_v2_test, test_case_moe_pertensor_fullload)
{
    uint32_t rowLen = 1024;
    uint32_t totalRow = 128;
    uint32_t groupNum = 8;
    uint32_t blockDim = 24;

    size_t inputByteSize = totalRow * rowLen * sizeof(uint16_t);
    size_t smoothScalesMoeByteSize = groupNum * rowLen * sizeof(uint16_t);
    size_t groupIndexMoeByteSize = groupNum * sizeof(int32_t);
    size_t outputByteSize = totalRow * rowLen * sizeof(int8_t);
    size_t scaleByteSize = 1 * sizeof(uint32_t);
    size_t offsetByteSize = 1 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingDataArch35);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales_moe = (uint8_t*)AscendC::GmAlloc(smoothScalesMoeByteSize);
    uint8_t* group_index_moe = (uint8_t*)AscendC::GmAlloc(groupIndexMoeByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    DynamicQuantTilingDataArch35* td = reinterpret_cast<DynamicQuantTilingDataArch35*>(tiling);
    td->coreNum = blockDim;
    td->rowLen = rowLen;
    td->headCoreNum = 8;
    td->rowPerHeadCore = 6;
    td->rowPerTailCore = 5;
    td->multiRowNumHeadCore = 6;
    td->multiRowNumTailCore = 5;
    td->innerLoopEle = 6;
    td->innerLoopTimes = 1;
    td->innerLoopTail = 0;
    td->groupNum = groupNum;
    td->alignGroupNum = groupNum;
    td->hasSmooth = 1;
    td->unused = 0;
    td->ubSize = 191 * 1024;
    td->sizeH = 0;
    td->sizeX = 0;
    td->sizeZOut = 0;
    td->sizeCopyRow = 0;
    td->numCopyRow = 0;
    td->numHeadCore = 0;
    td->numTailCore = 0;
    td->numHeadTimes = 0;
    td->numTailTimes = 0;
    td->numLastTailRow = 0;
    td->alignType = 0;
    td->totalBatchLen = 0;
    td->mLen = 0;
    td->mBlockSize = 0;
    td->mTailBlockSize = 0;
    td->mBlockNum = 0;
    td->nLen = 0;
    td->nBlockSize = 0;
    td->nTailBlockSize = 0;
    td->nBlockNum = 0;
    td->nBaseSize = 0;
    td->nBaseLoopNum = 0;
    td->blockPerHead = 0;
    td->blockPerTail = 0;
    td->totalBlockNum = 0;
    td->batchBlockSize = 0;
    td->batchTailBlockSize = 0;
    td->batchBlockNum = 0;
    td->dstTypeMax = 0.0f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // moe pertensor full load sym
    ICPU_SET_TILING_KEY(117);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales_moe, group_index_moe, output, scale, offset, workSpace,
                (uint8_t*)(td));

    // moe pertensor full load nosym
    ICPU_SET_TILING_KEY(53);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales_moe, group_index_moe, output, scale, offset, workSpace,
                (uint8_t*)(td));

    AscendC::GmFree(input);
    AscendC::GmFree(smooth_scales_moe);
    AscendC::GmFree(group_index_moe);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

TEST_F(dynamic_quant_v2_test, test_case_moe_pertensor_largeshape)
{
    uint32_t rowLen = 32800;
    uint32_t totalRow = 128;
    uint32_t groupNum = 8;
    uint32_t blockDim = 24;

    size_t inputByteSize = totalRow * rowLen * sizeof(uint16_t);
    size_t smoothScalesMoeByteSize = groupNum * rowLen * sizeof(uint16_t);
    size_t groupIndexMoeByteSize = groupNum * sizeof(int32_t);
    size_t outputByteSize = totalRow * rowLen * sizeof(int8_t);
    size_t scaleByteSize = 1 * sizeof(uint32_t);
    size_t offsetByteSize = 1 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingDataArch35);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales_moe = (uint8_t*)AscendC::GmAlloc(smoothScalesMoeByteSize);
    uint8_t* group_index_moe = (uint8_t*)AscendC::GmAlloc(groupIndexMoeByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    DynamicQuantTilingDataArch35* td = reinterpret_cast<DynamicQuantTilingDataArch35*>(tiling);
    td->coreNum = blockDim;
    td->rowLen = rowLen;
    td->headCoreNum = 8;
    td->rowPerHeadCore = 6;
    td->rowPerTailCore = 5;
    td->multiRowNumHeadCore = 6;
    td->multiRowNumTailCore = 5;
    td->innerLoopEle = 4096;
    td->innerLoopTimes = 7;
    td->innerLoopTail = 1056;
    td->groupNum = groupNum;
    td->alignGroupNum = groupNum;
    td->hasSmooth = 1;
    td->unused = 0;
    td->ubSize = 191 * 1024;
    td->sizeH = 0;
    td->sizeX = 0;
    td->sizeZOut = 0;
    td->sizeCopyRow = 0;
    td->numCopyRow = 0;
    td->numHeadCore = 0;
    td->numTailCore = 0;
    td->numHeadTimes = 0;
    td->numTailTimes = 0;
    td->numLastTailRow = 0;
    td->alignType = 0;
    td->totalBatchLen = 0;
    td->mLen = 0;
    td->mBlockSize = 0;
    td->mTailBlockSize = 0;
    td->mBlockNum = 0;
    td->nLen = 0;
    td->nBlockSize = 0;
    td->nTailBlockSize = 0;
    td->nBlockNum = 0;
    td->nBaseSize = 0;
    td->nBaseLoopNum = 0;
    td->blockPerHead = 0;
    td->blockPerTail = 0;
    td->totalBlockNum = 0;
    td->batchBlockSize = 0;
    td->batchTailBlockSize = 0;
    td->batchBlockNum = 0;
    td->dstTypeMax = 0.0f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // moe pertensor large shape sym
    ICPU_SET_TILING_KEY(119);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales_moe, group_index_moe, output, scale, offset, workSpace,
                (uint8_t*)(td));

    // moe pertensor large shape nosym
    ICPU_SET_TILING_KEY(55);
    ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales_moe, group_index_moe, output, scale, offset, workSpace,
                (uint8_t*)(td));

    AscendC::GmFree(input);
    AscendC::GmFree(smooth_scales_moe);
    AscendC::GmFree(group_index_moe);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

// bf16 行内极值交替 ±bf16 最大值时, 行极差 2*bf16max 超出 fp32 表示范围:
// 验证溢出防护生效——scale 为有限值且等于两端分别除以量化粒度再作差的结果, offset 约 -0.5,
// int4 输出正/负极值分别饱和到 +7 / -8, 而非溢出时 scale=inf、输出全 7
TEST_F(dynamic_quant_v2_test, test_case_bf16_extreme_range)
{
    const size_t rowNum = 128;
    const size_t rowLen = 1024;
    size_t inputByteSize = rowNum * rowLen * sizeof(uint16_t);
    size_t smoothScalesByteSize = rowLen * sizeof(uint16_t);
    size_t outputByteSize = rowNum * rowLen; // int4 两元素打包为 1 字节
    size_t scaleByteSize = rowNum * sizeof(float);
    size_t offsetByteSize = rowNum * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 24;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // 输入交替 +bf16max(0x7F7F) / -bf16max(0xFF7F), smooth 置 1.0(bf16 0x3F80)
    uint16_t* inputU16 = reinterpret_cast<uint16_t*>(input);
    uint16_t* smoothU16 = reinterpret_cast<uint16_t*>(smooth_scales);
    for (size_t i = 0; i < rowNum * rowLen; i++) {
        inputU16[i] = (i % 2 == 0) ? 0x7F7F : 0xFF7F;
    }
    for (size_t i = 0; i < rowLen; i++) {
        smoothU16[i] = 0x3F80;
    }

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);
    tilingDatafromBin->coreNum = 24;
    tilingDatafromBin->rowLen = rowLen;
    tilingDatafromBin->headCoreNum = 8;
    tilingDatafromBin->rowPerHeadCore = 6;
    tilingDatafromBin->rowPerTailCore = 5;
    tilingDatafromBin->multiRowNumHeadCore = 6;
    tilingDatafromBin->multiRowNumTailCore = 5;
    tilingDatafromBin->innerLoopEle = 6;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->groupNum = 8;
    tilingDatafromBin->alignGroupNum = 8;
    tilingDatafromBin->hasSmooth = 1;
    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    tilingDatafromBin->ubSize = 191 * 1024;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 期望值按 kernel 溢出分支同款 fp32 算法计算: scale = max/粒度 - min/粒度, offset = 7 - max/scale
    // (int4 量化粒度 15.0, 与 dynamic_quant_base.h 的 DYNAMIC_QUANT_INT4_SCALE/OFFSET 一致)
    const float maxV = 3.3895313892515355e+38f;
    const float minV = -maxV;
    const float expScale = maxV / 15.0f - minV / 15.0f;
    const float expOffset = 7.0f - maxV / expScale;

    // int4 两元素打包为一字节(低位在前), 按有符号 nibble 解包
    auto unpackInt4 = [](uint8_t packed, size_t nibbleIdx) {
        int v = (nibbleIdx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
        return (v > 7) ? (v - 16) : v;
    };

    for (uint32_t tilingKey : {0u, 2u}) { // bf16 常规路径 / bf16 双行路径
        ICPU_SET_TILING_KEY(tilingKey);
        ICPU_RUN_KF(dynamic_quant_v2, blockDim, input, smooth_scales, group_index, output, scale, offset, workSpace,
                    (uint8_t*)(tilingDatafromBin));

        float* scaleF32 = reinterpret_cast<float*>(scale);
        float* offsetF32 = reinterpret_cast<float*>(offset);
        for (size_t r : {size_t(0), size_t(63), size_t(127)}) {
            EXPECT_LT(scaleF32[r], 3.5e+38f); // scale 不应为 inf
            EXPECT_GT(scaleF32[r], 1e+37f);
            EXPECT_NEAR(scaleF32[r], expScale, expScale * 1e-3f);
            EXPECT_NEAR(offsetF32[r], expOffset, 1e-3f);
        }
        EXPECT_EQ(unpackInt4(output[0], 0), 7);  // 正极值饱和 +7
        EXPECT_EQ(unpackInt4(output[0], 1), -8); // 负极值饱和 -8
        EXPECT_EQ(unpackInt4(output[1], 0), 7);
        EXPECT_EQ(unpackInt4(output[1], 1), -8);
    }

    AscendC::GmFree(input);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
