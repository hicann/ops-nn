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
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "data_utils.h"
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/max_pool2d_with_mask_tiling_data.h"

extern "C" __global__ __aicore__ void max_pool2d_with_mask(GM_ADDR x, GM_ADDR out, GM_ADDR indices, GM_ADDR workspace,
                                                           GM_ADDR tiling);

namespace {
size_t Align32(size_t bytes) { return (bytes + 31) / 32 * 32; }

void RunTwoSliceCase(int64_t kernelSize, int64_t padding, int64_t containerBytes)
{
    constexpr int64_t slices = 2;
    constexpr int64_t inputElementsPerSlice = 16;
    constexpr int64_t outputElementsPerSlice = 4;
    constexpr std::array<float, slices * inputElementsPerSlice> input = {
        1,  4.5199752F, 2,  4,  4.52141F, 0, 6, 1, 7, 8, 9, 2, 4, 3, 1, 10,
        -1, -2,         -3, -4, 1,        1, 2, 3, 4, 3, 2, 1, 5, 6, 7, 8};
    constexpr std::array<float, slices * outputElementsPerSlice> expectedOut = {4.52141F, 6, 8, 10, 1, 3, 6, 8};
    constexpr std::array<int32_t, slices * outputElementsPerSlice> expectedIndices = {4, 6, 9, 15, 4, 7, 13, 15};

    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(input.size() * sizeof(float))));
    auto* out = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(expectedOut.size() * sizeof(float))));
    auto* indices = static_cast<uint8_t*>(AscendC::GmAlloc(containerBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(MaxPool2dWithMaskTilingData)));
    std::memcpy(x, input.data(), input.size() * sizeof(float));
    std::memset(out, 0, Align32(expectedOut.size() * sizeof(float)));
    std::memset(indices, 0x7f, containerBytes);

    std::memset(tiling, 0, sizeof(MaxPool2dWithMaskTilingData));
    auto* tilingData = reinterpret_cast<MaxPool2dWithMaskTilingData*>(tiling);
    tilingData->batchSize = 1;
    tilingData->channels = slices;
    tilingData->inputHeight = 4;
    tilingData->inputWidth = 4;
    tilingData->outputHeight = 2;
    tilingData->outputWidth = 2;
    tilingData->kernelH = kernelSize;
    tilingData->kernelW = kernelSize;
    tilingData->strideH = 2;
    tilingData->strideW = 2;
    tilingData->padH = padding;
    tilingData->padW = padding;
    tilingData->dilationH = 1;
    tilingData->dilationW = 1;
    tilingData->totalSlices = slices;
    tilingData->slicesPerCore = 1;
    tilingData->outputRowsPerTile = 2;
    tilingData->inputWidthAligned = 8;
    tilingData->outputElementsPerTileAligned = 8;
    tilingData->validIndexBytes = expectedIndices.size() * sizeof(int32_t);
    tilingData->containerBytes = containerBytes;
    tilingData->zeroBytesPerCore = (containerBytes - tilingData->validIndexBytes) / slices;
    tilingData->zeroBufferBytes = 8192;
    tilingData->nc = slices;
    tilingData->dx = 1;
    tilingData->hx = 4;
    tilingData->wx = 4;
    tilingData->kd = 1;
    tilingData->kh = kernelSize;
    tilingData->kw = kernelSize;
    tilingData->sd = 1;
    tilingData->sh = 2;
    tilingData->sw = 2;
    tilingData->pf = 0;
    tilingData->pb = 0;
    tilingData->pl = padding;
    tilingData->pr = padding;
    tilingData->pt = padding;
    tilingData->pd = 0;
    tilingData->dy = 1;
    tilingData->hy = 2;
    tilingData->wy = 2;
    tilingData->ncFactor = 2;
    tilingData->ncTail = 2;
    tilingData->ncOuter = 1;
    tilingData->dyFactor = 1;
    tilingData->dyTail = 1;
    tilingData->dyOuter = 1;
    tilingData->hyFactor = 2;
    tilingData->hyTail = 2;
    tilingData->hyOuter = 1;
    tilingData->wyFactor = 2;
    tilingData->wyTail = 2;
    tilingData->wyOuter = 1;
    tilingData->blockFactor = 1;
    tilingData->blockTail = 1;
    tilingData->totalIdx = 1;
    tilingData->coreNums = 1;
    tilingData->inputBufferSize = kernelSize == 2 ? 2048 : 2560;
    tilingData->outputMaxBufferSize = 2048;
    tilingData->outputIndiceBufferSize = 8192;
    tilingData->indiceTempBufferSize = 4096;
    tilingData->maskBufferSize = 512;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(padding == 0 ? 0 : 1);
    auto kernel = [](GM_ADDR xAddress, GM_ADDR outAddress, GM_ADDR indicesAddress, GM_ADDR workspaceAddress,
                     GM_ADDR tilingAddress) {
        ::max_pool2d_with_mask(xAddress, outAddress, indicesAddress, workspaceAddress, tilingAddress);
    };
    ICPU_RUN_KF(kernel, 2, x, out, indices, workspace, tiling);

    const auto* actualOut = reinterpret_cast<const float*>(out);
    const auto* actualIndices = reinterpret_cast<const int32_t*>(indices);
    for (size_t i = 0; i < expectedOut.size(); ++i) {
        EXPECT_FLOAT_EQ(actualOut[i], expectedOut[i]);
        EXPECT_EQ(actualIndices[i], expectedIndices[i]);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

void RunUnalignedSliceCase()
{
    constexpr int64_t slices = 2;
    constexpr int64_t inputHeight = 2;
    constexpr int64_t inputWidth = 7;
    constexpr int64_t outputHeight = 1;
    constexpr int64_t outputWidth = 6;
    constexpr int64_t kernelSize = 2;
    constexpr int64_t containerBytes = 512;
    constexpr std::array<float, slices * inputHeight * inputWidth> input = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    constexpr std::array<float, slices * outputHeight * outputWidth> expectedOut = {9,  10, 11, 12, 13, 14,
                                                                                    14, 13, 12, 11, 10, 9};
    constexpr std::array<int32_t, slices * outputHeight * outputWidth> expectedIndices = {8, 9, 10, 11, 12, 13,
                                                                                          0, 1, 2,  3,  4,  5};

    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(input.size() * sizeof(float))));
    auto* out = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(expectedOut.size() * sizeof(float))));
    auto* indices = static_cast<uint8_t*>(AscendC::GmAlloc(containerBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(MaxPool2dWithMaskTilingData)));
    std::memset(x, 0x7f, Align32(input.size() * sizeof(float)));
    std::memcpy(x, input.data(), input.size() * sizeof(float));
    std::memset(out, 0, Align32(expectedOut.size() * sizeof(float)));
    std::memset(indices, 0x7f, containerBytes);

    std::memset(tiling, 0, sizeof(MaxPool2dWithMaskTilingData));
    auto* tilingData = reinterpret_cast<MaxPool2dWithMaskTilingData*>(tiling);
    tilingData->batchSize = 1;
    tilingData->channels = slices;
    tilingData->inputHeight = inputHeight;
    tilingData->inputWidth = inputWidth;
    tilingData->outputHeight = outputHeight;
    tilingData->outputWidth = outputWidth;
    tilingData->kernelH = kernelSize;
    tilingData->kernelW = kernelSize;
    tilingData->strideH = 1;
    tilingData->strideW = 1;
    tilingData->padH = 0;
    tilingData->padW = 0;
    tilingData->dilationH = 1;
    tilingData->dilationW = 1;
    tilingData->totalSlices = slices;
    tilingData->slicesPerCore = slices;
    tilingData->outputRowsPerTile = outputHeight;
    tilingData->inputWidthAligned = 8;
    tilingData->outputElementsPerTileAligned = 8;
    tilingData->validIndexBytes = expectedIndices.size() * sizeof(int32_t);
    tilingData->containerBytes = containerBytes;
    tilingData->zeroBytesPerCore = containerBytes - tilingData->validIndexBytes;
    tilingData->zeroBufferBytes = 8192;
    tilingData->nc = slices;
    tilingData->dx = 1;
    tilingData->hx = inputHeight;
    tilingData->wx = inputWidth;
    tilingData->kd = 1;
    tilingData->kh = kernelSize;
    tilingData->kw = kernelSize;
    tilingData->sd = 1;
    tilingData->sh = 1;
    tilingData->sw = 1;
    tilingData->pf = 0;
    tilingData->pb = 0;
    tilingData->pl = 0;
    tilingData->pr = 0;
    tilingData->pt = 0;
    tilingData->pd = 0;
    tilingData->dy = 1;
    tilingData->hy = outputHeight;
    tilingData->wy = outputWidth;
    tilingData->ncFactor = slices;
    tilingData->ncTail = slices;
    tilingData->ncOuter = 1;
    tilingData->dyFactor = 1;
    tilingData->dyTail = 1;
    tilingData->dyOuter = 1;
    tilingData->hyFactor = outputHeight;
    tilingData->hyTail = outputHeight;
    tilingData->hyOuter = 1;
    tilingData->wyFactor = outputWidth;
    tilingData->wyTail = outputWidth;
    tilingData->wyOuter = 1;
    tilingData->blockFactor = 1;
    tilingData->blockTail = 1;
    tilingData->totalIdx = 1;
    tilingData->coreNums = 1;
    tilingData->inputBufferSize = 2048;
    tilingData->outputMaxBufferSize = 2048;
    tilingData->outputIndiceBufferSize = 8192;
    tilingData->indiceTempBufferSize = 4096;
    tilingData->maskBufferSize = 512;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    auto kernel = [](GM_ADDR xAddress, GM_ADDR outAddress, GM_ADDR indicesAddress, GM_ADDR workspaceAddress,
                     GM_ADDR tilingAddress) {
        ::max_pool2d_with_mask(xAddress, outAddress, indicesAddress, workspaceAddress, tilingAddress);
    };
    ICPU_RUN_KF(kernel, 1, x, out, indices, workspace, tiling);

    const auto* actualOut = reinterpret_cast<const float*>(out);
    const auto* actualIndices = reinterpret_cast<const int32_t*>(indices);
    for (size_t i = 0; i < expectedOut.size(); ++i) {
        EXPECT_FLOAT_EQ(actualOut[i], expectedOut[i]);
        EXPECT_EQ(actualIndices[i], expectedIndices[i]);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

void RunWindowFallbackCase(int64_t height, int64_t width, bool ties, int64_t slices = 1)
{
    const int64_t elements = height * width;
    const int64_t containerBytes = slices * elements * 64;
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(slices * elements * sizeof(float))));
    auto* out = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(slices * sizeof(float))));
    auto* indices = static_cast<uint8_t*>(AscendC::GmAlloc(containerBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(MaxPool2dWithMaskTilingPacket)));
    auto* input = reinterpret_cast<float*>(x);
    for (int64_t i = 0; i < slices * elements; ++i) {
        input[i] = static_cast<float>(i % elements);
    }
    constexpr int64_t firstMaximum = MAX_POOL2D_WINDOW_INPUT_BYTES / sizeof(float) - 1;
    if (ties) {
        input[firstMaximum] = 10000.0f;
        input[firstMaximum + 1] = 10000.0f;
    }
    std::memset(out, 0x7f, Align32(slices * sizeof(float)));
    std::memset(indices, 0x7f, containerBytes);
    std::memset(tiling, 0, sizeof(MaxPool2dWithMaskTilingPacket));
    auto* data = reinterpret_cast<MaxPool2dWithMaskTilingData*>(tiling);
    data->inputHeight = height;
    data->inputWidth = width;
    data->outputHeight = 1;
    data->outputWidth = 1;
    data->kernelH = height;
    data->kernelW = width;
    data->strideH = height;
    data->strideW = width;
    data->dilationH = data->dilationW = 1;
    data->kh = data->sh = height;
    data->kw = data->sw = width;
    data->validIndexBytes = slices * sizeof(int32_t);
    data->containerBytes = containerBytes;
    data->windowOutputTile = slices >= 16 ? 8 : 1;
    data->windowTailBoundaryCore = 0;
    data->zeroBytesPerCore = containerBytes - Align32(data->validIndexBytes);
    data->zeroBufferBytes = 8192;
    auto* packet = reinterpret_cast<MaxPool2dWithMaskTilingPacket*>(tiling);
    packet->window = {data->inputHeight,
                      data->inputWidth,
                      data->outputHeight,
                      data->outputWidth,
                      data->validIndexBytes,
                      data->containerBytes,
                      data->zeroBytesPerCore,
                      data->zeroBufferBytes,
                      data->nc,
                      data->kh,
                      data->kw,
                      data->sh,
                      data->sw,
                      data->pl,
                      data->pt,
                      data->windowOutputTile,
                      data->windowTailBoundaryCore,
                      data->windowPrefetch};
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    auto kernel = [](GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, GM_ADDR e) { ::max_pool2d_with_mask(a, b, c, d, e); };
    ICPU_RUN_KF(kernel, 1, x, out, indices, workspace, tiling);
    for (int64_t i = 0; i < slices; ++i) {
        EXPECT_FLOAT_EQ(reinterpret_cast<float*>(out)[i], ties ? 10000.0f : static_cast<float>(elements - 1));
        EXPECT_EQ(reinterpret_cast<int32_t*>(indices)[i], ties ? firstMaximum : elements - 1);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(MaxPool2dWithMaskKernel, WindowOutputBufferReuse) { RunWindowFallbackCase(2, 3, false, 34); }

TEST(MaxPool2dWithMaskKernel, LargeWindowFallbackAndZeroTail) { RunWindowFallbackCase(64, 64, false); }

TEST(MaxPool2dWithMaskKernel, WindowChunkBoundaryKeepsFirstMaximum) { RunWindowFallbackCase(2, 2051, true); }

TEST(MaxPool2dWithMaskKernel, TwoSlicesValuesIndicesAndZeroTail) { RunTwoSliceCase(2, 0, 512); }

TEST(MaxPool2dWithMaskKernel, PaddingValuesIndicesAndZeroTail) { RunTwoSliceCase(3, 1, 1152); }

TEST(MaxPool2dWithMaskKernel, Full64KiBZeroTailPerCore)
{
    // A uint16 DMA byte count wraps at 64 KiB. Prefilling with 0x7f detects missing writes.
    RunTwoSliceCase(2, 0, 2 * 65536 + 32);
}

TEST(MaxPool2dWithMaskKernel, UnalignedSliceStartAndInputTail) { RunUnalignedSliceCase(); }
TEST(MaxPool2dWithMaskKernel, OneByOnePreservesValuesAndAll25Indices)
{
    constexpr int64_t elements = 25;
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(128));
    auto* out = static_cast<uint8_t*>(AscendC::GmAlloc(128));
    auto* indices = static_cast<uint8_t*>(AscendC::GmAlloc(128));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(MaxPool2dWithMaskTilingPacket)));
    for (int64_t i = 0; i < elements; ++i) {
        reinterpret_cast<float*>(x)[i] = static_cast<float>(i - 12);
    }
    std::memset(out, 0x7f, 128);
    std::memset(indices, 0x7f, 128);
    std::memset(tiling, 0, sizeof(MaxPool2dWithMaskTilingPacket));
    auto* packet = reinterpret_cast<MaxPool2dWithMaskTilingPacket*>(tiling);
    packet->window.inputHeight = packet->window.inputWidth = 5;
    packet->window.outputHeight = packet->window.outputWidth = 5;
    packet->window.validIndexBytes = elements * sizeof(int32_t);
    packet->window.containerBytes = 128;
    packet->window.nc = 1;
    packet->window.kh = packet->window.kw = packet->window.sh = packet->window.sw = 1;
    packet->window.windowOutputTile = elements;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    auto kernel = [](GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, GM_ADDR e) { ::max_pool2d_with_mask(a, b, c, d, e); };
    ICPU_RUN_KF(kernel, 1, x, out, indices, workspace, tiling);
    for (int64_t i = 0; i < elements; ++i) {
        EXPECT_FLOAT_EQ(reinterpret_cast<float*>(out)[i], reinterpret_cast<float*>(x)[i]);
        EXPECT_EQ(reinterpret_cast<int32_t*>(indices)[i], i);
    }
    AscendC::GmFree(x);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace
