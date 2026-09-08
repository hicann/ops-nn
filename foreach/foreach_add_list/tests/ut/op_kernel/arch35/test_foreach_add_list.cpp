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
 * \file test_foreach_add_list.cpp
 * \brief Kernel UT for foreach_add_list arch35 (ascend950) SIMT implementation.
 *        Covers int16/int8/uint8 wraparound, large-alpha precision regression
 *        (alpha kept as int32, mul-add in int32), multi-core tensor split and
 *        empty tensor.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../op_kernel/arch35/foreach_add_list.cpp"

namespace {
constexpr uint32_t TILING_KEY_INT16 = 4;
constexpr uint32_t TILING_KEY_INT8 = 5;
constexpr uint32_t TILING_KEY_UINT8 = 6;

struct CoreRange {
    uint16_t startT;
    uint16_t endT;
    int64_t startOff;
    int64_t endOff;
};

struct TensorListBuf {
    uint8_t* desc = nullptr;
    std::vector<uint8_t*> datas;
};

template <typename T>
TensorListBuf CreateTensorList(const std::vector<int64_t>& numels)
{
    TensorListBuf buf;
    const size_t n = numels.size();
    const uint64_t descCount = 1 + 3 * n;
    auto* desc = static_cast<uint64_t*>(AscendC::GmAlloc(descCount * sizeof(uint64_t)));
    desc[0] = (descCount - n) * sizeof(uint64_t);
    uint64_t idx = 0;
    for (size_t i = 0; i < n; i++) {
        desc[++idx] = (static_cast<uint64_t>(i) << 32) + 1;
        desc[++idx] = static_cast<uint64_t>(numels[i]);
    }
    for (size_t i = 0; i < n; i++) {
        const int64_t bytes = ((numels[i] * static_cast<int64_t>(sizeof(T))) + 31) / 32 * 32;
        auto* data = (bytes > 0) ? static_cast<uint8_t*>(AscendC::GmAlloc(bytes)) : nullptr;
        desc[++idx] = reinterpret_cast<uint64_t>(data);
        buf.datas.push_back(data);
    }
    buf.desc = reinterpret_cast<uint8_t*>(desc);
    return buf;
}

template <typename T>
T* ListData(const TensorListBuf& buf, size_t index)
{
    const uint64_t ptrOff = *reinterpret_cast<uint64_t*>(buf.desc);
    return reinterpret_cast<T*>(*reinterpret_cast<uint64_t*>(buf.desc + ptrOff + index * sizeof(uint64_t)));
}

void FreeTensorList(TensorListBuf& buf)
{
    for (auto* p : buf.datas) {
        if (p != nullptr) {
            AscendC::GmFree(p);
        }
    }
    AscendC::GmFree(buf.desc);
}

uint8_t* CreateTiling(int32_t coreNum, const std::vector<int64_t>& numels, const std::vector<CoreRange>& ranges)
{
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ForeachAddListTilingData)));
    memset(tiling, 0, sizeof(ForeachAddListTilingData));
    auto* td = reinterpret_cast<ForeachAddListTilingData*>(tiling);
    td->needCoreNum = coreNum;
    td->tensorCount = static_cast<int32_t>(numels.size());
    td->totalElements = std::accumulate(numels.begin(), numels.end(), static_cast<int64_t>(0));
    for (size_t i = 0; i < numels.size(); i++) {
        td->tensorDataCountList[i] = numels[i];
    }
    for (int32_t c = 0; c < coreNum; c++) {
        td->tensorStartList[c] = ranges[c].startT;
        td->tensorEndList[c] = ranges[c].endT;
        td->tensorStartOffsetList[c] = ranges[c].startOff;
        td->tensorEndOffsetList[c] = ranges[c].endOff;
    }
    return tiling;
}

int8_t WrapI8(int64_t r) { return static_cast<int8_t>((r & 0x7F) - (r & 0x80)); }

int16_t WrapI16(int64_t r) { return static_cast<int16_t>((r & 0x7FFF) - (r & 0x8000)); }

uint8_t WrapU8(int64_t r) { return static_cast<uint8_t>(r & 0xFF); }

template <typename T, uint32_t tilingKey>
void RunAndCheck(const std::vector<int64_t>& numels, int32_t alphaVal, const std::vector<std::vector<T>>& x1Vals,
                 const std::vector<std::vector<T>>& x2Vals, int32_t coreNum, const std::vector<CoreRange>& ranges,
                 T (*wrap)(int64_t))
{
    TensorListBuf x1 = CreateTensorList<T>(numels);
    TensorListBuf x2 = CreateTensorList<T>(numels);
    TensorListBuf y = CreateTensorList<T>(numels);
    for (size_t t = 0; t < numels.size(); t++) {
        T* x1d = ListData<T>(x1, t);
        T* x2d = ListData<T>(x2, t);
        for (int64_t i = 0; i < numels[t]; i++) {
            x1d[i] = x1Vals[t][i];
            x2d[i] = x2Vals[t][i];
        }
    }
    uint8_t* tiling = CreateTiling(coreNum, numels, ranges);
    auto* alphaBuf = static_cast<int32_t*>(AscendC::GmAlloc(sizeof(int32_t)));
    *alphaBuf = alphaVal;
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(foreach_add_list<tilingKey>, static_cast<uint32_t>(coreNum), x1.desc, x2.desc,
                reinterpret_cast<uint8_t*>(alphaBuf), y.desc, workspace, tiling);

    int64_t failed = 0;
    for (size_t t = 0; t < numels.size(); t++) {
        const T* out = ListData<T>(y, t);
        for (int64_t i = 0; i < numels[t]; i++) {
            const int64_t r = static_cast<int64_t>(x1Vals[t][i]) + static_cast<int64_t>(x2Vals[t][i]) * alphaVal;
            const T expect = wrap(r);
            if (out[i] != expect) {
                if (failed < 8) {
                    printf("  mismatch t=%zu i=%ld: got=%d expect=%d (r=%ld)\n", t, static_cast<long>(i),
                           static_cast<int>(out[i]), static_cast<int>(expect), static_cast<long>(r));
                }
                failed++;
            }
        }
    }
    EXPECT_EQ(failed, 0);

    AscendC::GmFree(workspace);
    AscendC::GmFree(alphaBuf);
    AscendC::GmFree(tiling);
    FreeTensorList(y);
    FreeTensorList(x2);
    FreeTensorList(x1);
}

template <typename T, uint32_t tilingKey>
void RunSingleCore(const std::vector<int64_t>& numels, int32_t alphaVal, const std::vector<std::vector<T>>& x1Vals,
                   const std::vector<std::vector<T>>& x2Vals, T (*wrap)(int64_t))
{
    std::vector<CoreRange> ranges = {{0, static_cast<uint16_t>(numels.size() - 1), 0, numels.back() - 1}};
    RunAndCheck<T, tilingKey>(numels, alphaVal, x1Vals, x2Vals, 1, ranges, wrap);
}
} // namespace

class foreach_add_list_arch35_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "foreach_add_list arch35 SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "foreach_add_list arch35 TearDown" << std::endl; }
};

TEST_F(foreach_add_list_arch35_test, test_int8_wraparound)
{
    const std::vector<int64_t> numels = {8};
    const std::vector<std::vector<int8_t>> x1 = {{-128, -1, 0, 1, 127, 100, -100, 55}};
    const std::vector<std::vector<int8_t>> x2 = {{1, -1, 2, -2, 1, 100, 100, 0}};
    RunSingleCore<int8_t, TILING_KEY_INT8>(numels, 3, x1, x2, WrapI8);
}

TEST_F(foreach_add_list_arch35_test, test_int16_wraparound)
{
    const std::vector<int64_t> numels = {4};
    const std::vector<std::vector<int16_t>> x1 = {{-32768, 32767, 1234, -1234}};
    const std::vector<std::vector<int16_t>> x2 = {{1, 1, 100, -100}};
    RunSingleCore<int16_t, TILING_KEY_INT16>(numels, -1000, x1, x2, WrapI16);
}

TEST_F(foreach_add_list_arch35_test, test_uint8_wraparound)
{
    const std::vector<int64_t> numels = {4};
    const std::vector<std::vector<uint8_t>> x1 = {{0, 250, 128, 255}};
    const std::vector<std::vector<uint8_t>> x2 = {{1, 3, 200, 0}};
    RunSingleCore<uint8_t, TILING_KEY_UINT8>(numels, -2, x1, x2, WrapU8);
}

TEST_F(foreach_add_list_arch35_test, test_int8_alpha_2p24_plus_1)
{
    const std::vector<int64_t> numels = {2};
    const std::vector<std::vector<int8_t>> x1 = {{0, 0}};
    const std::vector<std::vector<int8_t>> x2 = {{1, 1}};
    RunSingleCore<int8_t, TILING_KEY_INT8>(numels, 16777217, x1, x2, WrapI8);
}

TEST_F(foreach_add_list_arch35_test, test_int16_alpha_2p24_plus_1)
{
    const std::vector<int64_t> numels = {2};
    const std::vector<std::vector<int16_t>> x1 = {{0, 0}};
    const std::vector<std::vector<int16_t>> x2 = {{1, 1}};
    RunSingleCore<int16_t, TILING_KEY_INT16>(numels, 16777217, x1, x2, WrapI16);
}

TEST_F(foreach_add_list_arch35_test, test_int8_alpha_int32_max)
{
    const std::vector<int64_t> numels = {2};
    const std::vector<std::vector<int8_t>> x1 = {{0, 0}};
    const std::vector<std::vector<int8_t>> x2 = {{1, 1}};
    RunSingleCore<int8_t, TILING_KEY_INT8>(numels, 2147483647, x1, x2, WrapI8);
}

TEST_F(foreach_add_list_arch35_test, test_uint8_alpha_int32_max)
{
    const std::vector<int64_t> numels = {2};
    const std::vector<std::vector<uint8_t>> x1 = {{0, 0}};
    const std::vector<std::vector<uint8_t>> x2 = {{1, 1}};
    RunSingleCore<uint8_t, TILING_KEY_UINT8>(numels, 2147483647, x1, x2, WrapU8);
}

TEST_F(foreach_add_list_arch35_test, test_int8_multicore_split)
{
    const std::vector<int64_t> numels = {8, 200, 33};
    std::vector<std::vector<int8_t>> x1(3);
    std::vector<std::vector<int8_t>> x2(3);
    for (size_t t = 0; t < numels.size(); t++) {
        x1[t].resize(numels[t]);
        x2[t].resize(numels[t]);
        for (int64_t i = 0; i < numels[t]; i++) {
            x1[t][i] = static_cast<int8_t>((i * 7 + static_cast<int64_t>(t) * 3) % 251 - 125);
            x2[t][i] = static_cast<int8_t>((i * 13 + static_cast<int64_t>(t)) % 61 - 30);
        }
    }
    const std::vector<CoreRange> ranges = {{0, 1, 0, 99}, {1, 2, 100, 32}};
    RunAndCheck<int8_t, TILING_KEY_INT8>(numels, 2, x1, x2, 2, ranges, WrapI8);
}

TEST_F(foreach_add_list_arch35_test, test_int8_empty_tensor)
{
    const std::vector<int64_t> numels = {4, 0, 3};
    const std::vector<std::vector<int8_t>> x1 = {{1, 2, 3, 4}, {}, {5, 6, 7}};
    const std::vector<std::vector<int8_t>> x2 = {{10, 20, 30, 40}, {}, {50, 60, 70}};
    RunSingleCore<int8_t, TILING_KEY_INT8>(numels, 1, x1, x2, WrapI8);
}
