/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "max_pool3d_grad_with_argmax_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void max_pool3d_grad_with_argmax(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class MaxPool3dGradWithArgmaxKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MaxPool3dGradWithArgmax Kernel SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MaxPool3dGradWithArgmax Kernel TearDown\n" << endl;
    }
};

template <typename T>
void TestMaxPool3dGradWithArgmaxKernel(
    vector<uint64_t>& xShape, vector<uint64_t>& gradShape, vector<uint64_t>& ksize, vector<uint64_t>& strides,
    vector<uint64_t>& pads, vector<uint64_t>& dilation, uint64_t workspaceSize, uint64_t blockDim, uint64_t tilingKey)
{
    uint64_t ncDim = xShape[0] * xShape[1];
    uint64_t diDim = xShape[2];
    uint64_t hiDim = xShape[3];
    uint64_t wiDim = xShape[4];
    uint64_t doDim = gradShape[2];
    uint64_t hoDim = gradShape[3];
    uint64_t woDim = gradShape[4];

    size_t xByteSize = ncDim * diDim * hiDim * wiDim * sizeof(T);
    size_t gradByteSize = ncDim * doDim * hoDim * woDim * sizeof(T);
    size_t argmaxByteSize = ncDim * doDim * hoDim * woDim * sizeof(int32_t);
    size_t dxByteSize = ncDim * diDim * hiDim * wiDim * sizeof(T);
    size_t tilingDataSize = sizeof(MaxPool3DGradWithArgmaxTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);
    MaxPool3DGradWithArgmaxTilingData* tilingDatafromBin = reinterpret_cast<MaxPool3DGradWithArgmaxTilingData*>(tiling);

    tilingDatafromBin->ncDim = ncDim;
    tilingDatafromBin->diDim = diDim;
    tilingDatafromBin->hiDim = hiDim;
    tilingDatafromBin->wiDim = wiDim;
    tilingDatafromBin->doDim = doDim;
    tilingDatafromBin->hoDim = hoDim;
    tilingDatafromBin->woDim = woDim;
    tilingDatafromBin->kd = ksize[0];
    tilingDatafromBin->kh = ksize[1];
    tilingDatafromBin->kw = ksize[2];
    tilingDatafromBin->sd = strides[0];
    tilingDatafromBin->sh = strides[1];
    tilingDatafromBin->sw = strides[2];
    tilingDatafromBin->padDTop = pads[0];
    tilingDatafromBin->padHTop = pads[1];
    tilingDatafromBin->padWTop = pads[2];
    tilingDatafromBin->padDBottom = 0;
    tilingDatafromBin->padHBottom = 0;
    tilingDatafromBin->padWBottom = 0;
    tilingDatafromBin->baseNc = ncDim / blockDim;
    tilingDatafromBin->baseDo = 1;
    tilingDatafromBin->baseHo = 1;
    tilingDatafromBin->baseWo = 1;
    tilingDatafromBin->ncTail = ncDim / blockDim;
    tilingDatafromBin->doTail = 1;
    tilingDatafromBin->hoTail = 1;
    tilingDatafromBin->woTail = 1;
    tilingDatafromBin->ncCnt = blockDim;
    tilingDatafromBin->doCnt = 1;
    tilingDatafromBin->hoCnt = 1;
    tilingDatafromBin->woCnt = 1;
    tilingDatafromBin->totalCnt = blockDim;
    tilingDatafromBin->usedCoreNum = blockDim;
    tilingDatafromBin->totalUBSize = 196352;
    tilingDatafromBin->singleCoreNc = ncDim;
    tilingDatafromBin->singleCoreDo = doDim;
    tilingDatafromBin->singleCoreHo = hoDim;
    tilingDatafromBin->singleCoreWo = woDim;
    tilingDatafromBin->ncRound = 1;
    tilingDatafromBin->ncRoundTail = 1;
    tilingDatafromBin->preCoreNum = 0;
    tilingDatafromBin->totalRound = 1;

    // normal fp16
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(max_pool3d_grad_with_argmax, blockDim, x, grad, argmax, dy, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(grad);
    AscendC::GmFree(argmax);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

template <typename T>
void TestMaxPool3dGradWithArgmaxKernelNew(
    vector<uint64_t>& xShape, vector<uint64_t>& gradShape, vector<uint64_t>& ksize, vector<uint64_t>& strides,
    vector<uint64_t>& pads, vector<uint64_t>& dilation, uint64_t workspaceSize, uint64_t blockDim, uint64_t tilingKey)
{
    uint64_t ncDim = xShape[0] * xShape[1];
    uint64_t diDim = xShape[2];
    uint64_t hiDim = xShape[3];
    uint64_t wiDim = xShape[4];
    uint64_t doDim = gradShape[2];
    uint64_t hoDim = gradShape[3];
    uint64_t woDim = gradShape[4];

    size_t xByteSize = ncDim * diDim * hiDim * wiDim * sizeof(T);
    size_t gradByteSize = ncDim * doDim * hoDim * woDim * sizeof(T);
    size_t argmaxByteSize = ncDim * doDim * hoDim * woDim * sizeof(int32_t);
    size_t dxByteSize = ncDim * diDim * hiDim * wiDim * sizeof(T);
    size_t tilingDataSize = 0;

    if (tilingKey == 100000 || tilingKey == 100001 || tilingKey == 100002) {
        tilingDataSize = sizeof(MaxPool3DGradWithArgmaxNoSplitTilingData);
    } else if (tilingKey == 110000 || tilingKey == 110001 || tilingKey == 110002) {
        tilingDataSize = sizeof(MaxPool3DGradWithArgmaxSplitDTilingData);
    } else if (tilingKey == 111000 || tilingKey == 111001 || tilingKey == 111002) {
        tilingDataSize = sizeof(MaxPool3DGradWithArgmaxSplitHTilingData);
    } else {
        tilingDataSize = sizeof(MaxPool3DGradWithArgmaxSplitWTilingData);
    }
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DGradWithArgmaxNoSplitTilingData* tilingDatafromBinNo =
        reinterpret_cast<MaxPool3DGradWithArgmaxNoSplitTilingData*>(tiling);
    MaxPool3DGradWithArgmaxSplitDTilingData* tilingDatafromBinD =
        reinterpret_cast<MaxPool3DGradWithArgmaxSplitDTilingData*>(tiling);
    MaxPool3DGradWithArgmaxSplitHTilingData* tilingDatafromBinH =
        reinterpret_cast<MaxPool3DGradWithArgmaxSplitHTilingData*>(tiling);
    MaxPool3DGradWithArgmaxSplitWTilingData* tilingDatafromBinW =
        reinterpret_cast<MaxPool3DGradWithArgmaxSplitWTilingData*>(tiling);

    if (tilingKey == 100000 || tilingKey == 100001 || tilingKey == 100002) {
        tilingDatafromBinNo->inputShapes[0] = xShape[2];
        tilingDatafromBinNo->inputShapes[1] = xShape[3];
        tilingDatafromBinNo->inputShapes[2] = xShape[4];
        tilingDatafromBinNo->outShapes[0] = gradShape[2];
        tilingDatafromBinNo->outShapes[1] = gradShape[3];
        tilingDatafromBinNo->outShapes[2] = gradShape[4];
        tilingDatafromBinNo->kD = ksize[0];
        tilingDatafromBinNo->kH = ksize[1];
        tilingDatafromBinNo->kW = ksize[2];
        tilingDatafromBinNo->sD = strides[0];
        tilingDatafromBinNo->sH = strides[1];
        tilingDatafromBinNo->sW = strides[2];
        tilingDatafromBinNo->pD = pads[0];
        tilingDatafromBinNo->pH = pads[1];
        tilingDatafromBinNo->pW = pads[2];
        tilingDatafromBinNo->dD = dilation[0];
        tilingDatafromBinNo->dH = dilation[1];
        tilingDatafromBinNo->dW = dilation[2];
        tilingDatafromBinNo->batchesPerCore = 0;
        tilingDatafromBinNo->leftOverBatches = 8;
        tilingDatafromBinNo->partD = xShape[2];
        tilingDatafromBinNo->partH = xShape[3];
        tilingDatafromBinNo->partW = xShape[4];
        tilingDatafromBinNo->partOutD = gradShape[2];
        tilingDatafromBinNo->partOutH = gradShape[3];
        tilingDatafromBinNo->partOutW = gradShape[4];
        tilingDatafromBinNo->ceilD = 0;
        tilingDatafromBinNo->ceilH = 0;
        tilingDatafromBinNo->ceilW = 0;
        tilingDatafromBinNo->sizeUb1 = 8 * xShape[2] * xShape[3] * xShape[4];
        tilingDatafromBinNo->sizeUb2 = 8 * gradShape[2] * gradShape[3] * gradShape[4];
        tilingDatafromBinNo->sizeValues = 4096;
    } else if (tilingKey == 110000 || tilingKey == 110001 || tilingKey == 110002) {
        tilingDatafromBinD->inputShapes[0] = xShape[2];
        tilingDatafromBinD->inputShapes[1] = xShape[3];
        tilingDatafromBinD->inputShapes[2] = xShape[4];
        tilingDatafromBinD->outShapes[0] = gradShape[2];
        tilingDatafromBinD->outShapes[1] = gradShape[3];
        tilingDatafromBinD->outShapes[2] = gradShape[4];
        tilingDatafromBinD->kD = ksize[0];
        tilingDatafromBinD->kH = ksize[1];
        tilingDatafromBinD->kW = ksize[2];
        tilingDatafromBinD->sD = strides[0];
        tilingDatafromBinD->sH = strides[1];
        tilingDatafromBinD->sW = strides[2];
        tilingDatafromBinD->pD = pads[0];
        tilingDatafromBinD->pH = pads[1];
        tilingDatafromBinD->pW = pads[2];
        tilingDatafromBinD->dD = dilation[0];
        tilingDatafromBinD->dH = dilation[1];
        tilingDatafromBinD->dW = dilation[2];
        tilingDatafromBinD->batchesPerCore = 0;
        tilingDatafromBinD->leftOverBatches = 8;
        tilingDatafromBinD->partD = xShape[2];
        tilingDatafromBinD->partH = xShape[3];
        tilingDatafromBinD->partW = xShape[4];
        tilingDatafromBinD->partOutD = gradShape[2];
        tilingDatafromBinD->partOutH = gradShape[3];
        tilingDatafromBinD->partOutW = gradShape[4];
        tilingDatafromBinD->ceilD = 0;
        tilingDatafromBinD->ceilH = 0;
        tilingDatafromBinD->ceilW = 0;
        tilingDatafromBinD->sizeUb1 = 8 * xShape[2] * xShape[3] * xShape[4];
        tilingDatafromBinD->sizeUb2 = 8 * gradShape[2] * gradShape[3] * gradShape[4];
        tilingDatafromBinD->sizeValues = 4096;
    } else if (tilingKey == 111000 || tilingKey == 111001 || tilingKey == 111002) {
        tilingDatafromBinH->inputShapes[0] = xShape[2];
        tilingDatafromBinH->inputShapes[1] = xShape[3];
        tilingDatafromBinH->inputShapes[2] = xShape[4];
        tilingDatafromBinH->outShapes[0] = gradShape[2];
        tilingDatafromBinH->outShapes[1] = gradShape[3];
        tilingDatafromBinH->outShapes[2] = gradShape[4];
        tilingDatafromBinH->kD = ksize[0];
        tilingDatafromBinH->kH = ksize[1];
        tilingDatafromBinH->kW = ksize[2];
        tilingDatafromBinH->sD = strides[0];
        tilingDatafromBinH->sH = strides[1];
        tilingDatafromBinH->sW = strides[2];
        tilingDatafromBinH->pD = pads[0];
        tilingDatafromBinH->pH = pads[1];
        tilingDatafromBinH->pW = pads[2];
        tilingDatafromBinH->dD = dilation[0];
        tilingDatafromBinH->dH = dilation[1];
        tilingDatafromBinH->dW = dilation[2];
        tilingDatafromBinH->batchesPerCore = 0;
        tilingDatafromBinH->leftOverBatches = 8;
        tilingDatafromBinH->partD = xShape[2];
        tilingDatafromBinH->partH = xShape[3];
        tilingDatafromBinH->partW = xShape[4];
        tilingDatafromBinH->partOutD = gradShape[2];
        tilingDatafromBinH->partOutH = gradShape[3];
        tilingDatafromBinH->partOutW = gradShape[4];
        tilingDatafromBinH->ceilD = 0;
        tilingDatafromBinH->ceilH = 0;
        tilingDatafromBinH->ceilW = 0;
        tilingDatafromBinH->sizeUb1 = 8 * xShape[2] * xShape[3] * xShape[4];
        tilingDatafromBinH->sizeUb2 = 8 * gradShape[2] * gradShape[3] * gradShape[4];
        tilingDatafromBinH->sizeValues = 4096;
    } else {
        tilingDatafromBinW->inputShapes[0] = xShape[2];
        tilingDatafromBinW->inputShapes[1] = xShape[3];
        tilingDatafromBinW->inputShapes[2] = xShape[4];
        tilingDatafromBinW->outShapes[0] = gradShape[2];
        tilingDatafromBinW->outShapes[1] = gradShape[3];
        tilingDatafromBinW->outShapes[2] = gradShape[4];
        tilingDatafromBinW->kD = ksize[0];
        tilingDatafromBinW->kH = ksize[1];
        tilingDatafromBinW->kW = ksize[2];
        tilingDatafromBinW->sD = strides[0];
        tilingDatafromBinW->sH = strides[1];
        tilingDatafromBinW->sW = strides[2];
        tilingDatafromBinW->pD = pads[0];
        tilingDatafromBinW->pH = pads[1];
        tilingDatafromBinW->pW = pads[2];
        tilingDatafromBinW->dD = dilation[0];
        tilingDatafromBinW->dH = dilation[1];
        tilingDatafromBinW->dW = dilation[2];
        tilingDatafromBinW->batchesPerCore = 0;
        tilingDatafromBinW->leftOverBatches = 8;
        tilingDatafromBinW->partD = xShape[2];
        tilingDatafromBinW->partH = xShape[3];
        tilingDatafromBinW->partW = xShape[4];
        tilingDatafromBinW->partOutD = gradShape[2];
        tilingDatafromBinW->partOutH = gradShape[3];
        tilingDatafromBinW->partOutW = gradShape[4];
        tilingDatafromBinW->ceilD = 0;
        tilingDatafromBinW->ceilH = 0;
        tilingDatafromBinW->ceilW = 0;
        tilingDatafromBinW->sizeUb1 = 8 * xShape[2] * xShape[3] * xShape[4];
        tilingDatafromBinW->sizeUb2 = 8 * gradShape[2] * gradShape[3] * gradShape[4];
        tilingDatafromBinW->sizeValues = 4096;
    }

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if (tilingKey == 100000 || tilingKey == 100001 || tilingKey == 100002) {
        ICPU_RUN_KF(
            max_pool3d_grad_with_argmax, blockDim, x, grad, argmax, dy, workspace, (uint8_t*)(tilingDatafromBinNo));
    } else if (tilingKey == 110000 || tilingKey == 110001 || tilingKey == 110002) {
        ICPU_RUN_KF(
            max_pool3d_grad_with_argmax, blockDim, x, grad, argmax, dy, workspace, (uint8_t*)(tilingDatafromBinD));
    } else if (tilingKey == 111000 || tilingKey == 111001 || tilingKey == 111002) {
        ICPU_RUN_KF(
            max_pool3d_grad_with_argmax, blockDim, x, grad, argmax, dy, workspace, (uint8_t*)(tilingDatafromBinH));
    } else {
        ICPU_RUN_KF(
            max_pool3d_grad_with_argmax, blockDim, x, grad, argmax, dy, workspace, (uint8_t*)(tilingDatafromBinW));
    }

    AscendC::GmFree(x);
    AscendC::GmFree(grad);
    AscendC::GmFree(argmax);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_normal_case_0)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 0;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_normal_case_tail_empty_kernel)
{
    vector<uint64_t> xShape = {1, 16, 1, 2, 21};
    vector<uint64_t> gradShape = {1, 16, 1, 1, 10};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 0;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_case)
{
    vector<uint64_t> xShape = {40, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {40, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 40;
    uint64_t tilingKey = 51;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_overlap_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 151;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkdh_overlap_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 161;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkdhw_overlap_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 171;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_1_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 1;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_21_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 21;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_31_case)
{
    vector<uint64_t> xShape = {40, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {40, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 40;
    uint64_t tilingKey = 31;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkdh_41_case)
{
    vector<uint64_t> xShape = {40, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {40, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 40;
    uint64_t tilingKey = 41;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_101_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 101;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_121_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 121;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkd_131_case)
{
    vector<uint64_t> xShape = {40, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {40, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 40;
    uint64_t tilingKey = 131;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkdh_141_case)
{
    vector<uint64_t> xShape = {40, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {40, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 40;
    uint64_t tilingKey = 141;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkdhw_case)
{
    vector<uint64_t> xShape = {1, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 71;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_cutkdh_case)
{
    vector<uint64_t> xShape = {40, 64, 1, 2, 2};
    vector<uint64_t> gradShape = {40, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 2, 2};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 40;
    uint64_t tilingKey = 61;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_normal_case_1)
{
    vector<uint64_t> xShape = {1, 64, 1, 3, 3};
    vector<uint64_t> gradShape = {1, 64, 1, 1, 1};
    vector<uint64_t> ksize = {1, 3, 3};
    vector<uint64_t> strides = {1, 2, 2};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {1, 1, 1};
    uint64_t workspaceSize = 92160;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 100;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

TEST_F(MaxPool3dGradWithArgmaxKernel, max_pool3d_grad_with_argmax_scatter_case_0)
{
    vector<uint64_t> xShape = {5, 640, 1, 64, 64};
    vector<uint64_t> gradShape = {5, 640, 1, 1, 1};
    vector<uint64_t> ksize = {1, 64, 64};
    vector<uint64_t> strides = {1, 64, 64};
    vector<uint64_t> pads = {0, 0, 0};
    vector<uint64_t> dilation = {2, 1, 1};
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 40;
    uint64_t tilingKey = 2;
    TestMaxPool3dGradWithArgmaxKernel<float>(
        xShape, gradShape, ksize, strides, pads, dilation, workspaceSize, blockDim, tilingKey);
}

