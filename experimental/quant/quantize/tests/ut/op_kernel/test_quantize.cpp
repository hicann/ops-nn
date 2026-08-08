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
 * \file test_quantize.cpp
 * \brief Quantize kernel UT: run the real __global__ path on the CPU twin (ICPU_RUN_KF) and compare against a
 *        numpy golden = saturate_cast_to_y( rint( x / scales + zero_points ) ).
 *
 * The op-kernel harness compiles ONE binary with fixed DTYPE_* macros (see CMakeLists), so multiple dtype
 * combos are covered by instantiating a local templated __global__ entry (`quantize_ut_entry`) per combo
 * (the same technique ICPU_RUN_KF supports for templated kernels, e.g. relu_v3<0>).
 */
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "quantize_tiling_def.h"
#include "quantize.h"
#include "data_utils.h"

using namespace std;

#define QZ_STR(x) #x
#define QZ_XSTR(x) QZ_STR(x)

// Templated CPU-twin entry: mirrors op_kernel/quantize.cpp dispatch, but with explicit dtypes so all combos
// can be exercised from a single compiled target.
template <typename T, typename S, typename Z, typename Y, uint64_t KEY>
__global__ __aicore__ void quantize_ut_entry(GM_ADDR x, GM_ADDR scales, GM_ADDR zeroPoints, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (KEY == 1) {
        QuantizeOp::QuantizePerTensor<T, S, Z, Y> op;
        op.Init(x, scales, zeroPoints, y, &tilingData);
        op.Process();
    } else {
        QuantizeOp::QuantizePerChannel<T, S, Z, Y> op;
        op.Init(x, scales, zeroPoints, y, &tilingData);
        op.Process();
    }
}

// Non-template wrappers (avoid comma-in-macro issues when passing to ICPU_RUN_KF).
#define DEFINE_QZ_ENTRY(name, T, S, Z, Y, KEY)                                                                   \
    __global__ __aicore__ void name(GM_ADDR x, GM_ADDR scales, GM_ADDR zeroPoints, GM_ADDR y, GM_ADDR workspace, \
                                    GM_ADDR tiling)                                                              \
    {                                                                                                            \
        quantize_ut_entry<T, S, Z, Y, KEY>(x, scales, zeroPoints, y, workspace, tiling);                         \
    }

DEFINE_QZ_ENTRY(qz_pt_f32_f32_i32_i8, float, float, int32_t, int8_t, 1)
DEFINE_QZ_ENTRY(qz_pt_f32_f32_i32_i32, float, float, int32_t, int32_t, 1)
DEFINE_QZ_ENTRY(qz_pt_f32_f32_i32_u8, float, float, int32_t, uint8_t, 1)
DEFINE_QZ_ENTRY(qz_pc_f16_f32_i32_i8, half, float, int32_t, int8_t, 0)
DEFINE_QZ_ENTRY(qz_pc_bf16_bf16_bf16_i8, bfloat16_t, bfloat16_t, bfloat16_t, int8_t, 0)

namespace {
struct KernelIO {
    uint8_t* x = nullptr;
    uint8_t* scales = nullptr;
    uint8_t* zp = nullptr;
    uint8_t* y = nullptr;
    uint8_t* ws = nullptr;
    uint8_t* tiling = nullptr;
    int64_t total = 0;
};

KernelIO Prepare(const string& gendataArgs, size_t xElem, size_t sElem, size_t zElem, size_t yElem, bool hasZp,
                 bool perChannel, int64_t total, int64_t channelNum, int64_t rowLen, int64_t totalRows,
                 uint64_t tilingKey)
{
    string dataDir = QZ_XSTR(QUANTIZE_DATA_DIR);
    string cmd = string("python3 ") + dataDir + "/gen_data.py " + gendataArgs;
    int genRet = system(cmd.c_str());
    EXPECT_EQ(genRet, 0);

    int64_t paramCount = perChannel ? channelNum : 1;
    KernelIO io;
    io.total = total;
    size_t xBytes = static_cast<size_t>(total) * xElem;
    size_t sBytes = static_cast<size_t>(paramCount) * sElem;
    size_t zBytes = static_cast<size_t>(paramCount) * zElem;
    size_t yBytes = static_cast<size_t>(total) * yElem;
    io.x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    io.scales = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    io.zp = hasZp ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(zBytes)) : nullptr;
    io.y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    io.ws = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024));
    io.tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(QuantizeTilingData)));

    ReadFile("./x.bin", xBytes, io.x, xBytes);
    ReadFile("./scales.bin", sBytes, io.scales, sBytes);
    if (hasZp) {
        ReadFile("./zp.bin", zBytes, io.zp, zBytes);
    }

    QuantizeTilingData* td = reinterpret_cast<QuantizeTilingData*>(io.tiling);
    td->numCore = 1;
    td->hasZeroPoint = hasZp ? 1U : 0U;
    td->channelNum = perChannel ? channelNum : 1;
    td->rowLen = perChannel ? rowLen : 1;
    td->totalRows = perChannel ? totalRows : 1;
    td->totalElems = total;
    td->blockFactor = perChannel ? totalRows : total;
    td->blockTailFactor = perChannel ? totalRows : total;
    td->baseLen = 256;
    // zpDtype: the kernel reads zero_points by its runtime dtype, mirroring the host tiling. Derive the
    // ge::DataType code from the zero_points dtype token (3rd token of the gendata args).
    {
        std::istringstream iss(gendataArgs);
        std::vector<std::string> toks;
        std::string tok;
        while (iss >> tok) {
            toks.push_back(tok);
        }
        const std::string zdt = (toks.size() > 2) ? toks[2] : std::string("int32");
        uint32_t code = 0;
        if (zdt == "int8") {
            code = 2;
        } else if (zdt == "int32") {
            code = 3;
        } else if (zdt == "uint8") {
            code = 4;
        } else if (zdt == "bfloat16") {
            code = 27;
        } else if (zdt == "float16") {
            code = 1;
        }
        td->zpDtype = code;
    }

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    return io;
}

template <typename Y>
void CompareGolden(const KernelIO& io)
{
    std::vector<Y> golden(io.total);
    size_t goldenBytes = static_cast<size_t>(io.total) * sizeof(Y);
    ReadFile("./golden.bin", goldenBytes, golden.data(), goldenBytes);
    const Y* yout = reinterpret_cast<const Y*>(io.y);
    int mismatch = 0;
    for (int64_t i = 0; i < io.total; ++i) {
        int64_t diff = static_cast<int64_t>(yout[i]) - static_cast<int64_t>(golden[i]);
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > 1) {
            if (mismatch < 8) {
                std::cout << "mismatch idx " << i << " got " << static_cast<int64_t>(yout[i]) << " expected "
                          << static_cast<int64_t>(golden[i]) << std::endl;
            }
            ++mismatch;
        }
    }
    EXPECT_EQ(mismatch, 0);
}

void FreeIO(KernelIO& io)
{
    AscendC::GmFree(io.x);
    AscendC::GmFree(io.scales);
    if (io.zp != nullptr) {
        AscendC::GmFree(io.zp);
    }
    AscendC::GmFree(io.y);
    AscendC::GmFree(io.ws);
    AscendC::GmFree(io.tiling);
}
} // namespace

class quantize_kernel_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "quantize_kernel_test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "quantize_kernel_test TearDown" << std::endl; }
};

TEST_F(quantize_kernel_test, per_tensor_fp32_int8_no_zp)
{
    auto io = Prepare("float32 float32 int32 int8 pt 0 0 64 0", sizeof(float), sizeof(float), sizeof(int32_t),
                      sizeof(int8_t), false, false, 64, 1, 1, 1, 1);
    ICPU_RUN_KF(qz_pt_f32_f32_i32_i8, 1, io.x, io.scales, io.zp, io.y, io.ws, io.tiling);
    CompareGolden<int8_t>(io);
    FreeIO(io);
}

TEST_F(quantize_kernel_test, per_tensor_fp32_int8_saturate)
{
    auto io = Prepare("float32 float32 int32 int8 pt 0 1 64 0", sizeof(float), sizeof(float), sizeof(int32_t),
                      sizeof(int8_t), false, false, 64, 1, 1, 1, 1);
    ICPU_RUN_KF(qz_pt_f32_f32_i32_i8, 1, io.x, io.scales, io.zp, io.y, io.ws, io.tiling);
    CompareGolden<int8_t>(io);
    FreeIO(io);
}

TEST_F(quantize_kernel_test, per_tensor_fp32_int32_with_zp)
{
    auto io = Prepare("float32 float32 int32 int32 pt 1 0 64 0", sizeof(float), sizeof(float), sizeof(int32_t),
                      sizeof(int32_t), true, false, 64, 1, 1, 1, 1);
    ICPU_RUN_KF(qz_pt_f32_f32_i32_i32, 1, io.x, io.scales, io.zp, io.y, io.ws, io.tiling);
    CompareGolden<int32_t>(io);
    FreeIO(io);
}

TEST_F(quantize_kernel_test, per_tensor_fp32_uint8_saturate_no_zp)
{
    auto io = Prepare("float32 float32 int32 uint8 pt 0 1 64 0", sizeof(float), sizeof(float), sizeof(int32_t),
                      sizeof(uint8_t), false, false, 64, 1, 1, 1, 1);
    ICPU_RUN_KF(qz_pt_f32_f32_i32_u8, 1, io.x, io.scales, io.zp, io.y, io.ws, io.tiling);
    CompareGolden<uint8_t>(io);
    FreeIO(io);
}

TEST_F(quantize_kernel_test, per_channel_fp16_int8_zp_int32_axis1)
{
    // x[2,4,8] axis=1 -> channelNum=4, rowLen=8, totalRows=8, total=64
    auto io = Prepare("float16 float32 int32 int8 pc 1 0 2,4,8 1", sizeof(half), sizeof(float), sizeof(int32_t),
                      sizeof(int8_t), true, true, 64, 4, 8, 8, 0);
    ICPU_RUN_KF(qz_pc_f16_f32_i32_i8, 1, io.x, io.scales, io.zp, io.y, io.ws, io.tiling);
    CompareGolden<int8_t>(io);
    FreeIO(io);
}

TEST_F(quantize_kernel_test, per_channel_bf16_all_consistent_int8_axis1)
{
    // x[2,4,8] axis=1, all bf16 (x/scales/zp) -> channelNum=4, rowLen=8, totalRows=8, total=64
    auto io = Prepare("bfloat16 bfloat16 bfloat16 int8 pc 1 0 2,4,8 1", sizeof(bfloat16_t), sizeof(bfloat16_t),
                      sizeof(bfloat16_t), sizeof(int8_t), true, true, 64, 4, 8, 8, 0);
    ICPU_RUN_KF(qz_pc_bf16_bf16_bf16_i8, 1, io.x, io.scales, io.zp, io.y, io.ws, io.tiling);
    CompareGolden<int8_t>(io);
    FreeIO(io);
}
