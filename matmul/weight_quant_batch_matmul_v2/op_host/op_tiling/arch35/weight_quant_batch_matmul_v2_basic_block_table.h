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
 * \file weight_quant_batch_matmul_v2_basic_block_table.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_BASIC_BLOCK_TABLE_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_BASIC_BLOCK_TABLE_H

#include <array>
#include <cstdint>
#include <tuple>

namespace optiling {
constexpr int64_t BITS_16 = 16;
constexpr int64_t BITS_8 = 8;
constexpr int64_t BITS_4 = 4;
constexpr int64_t BITS_PER_BYTE = 8;

constexpr int64_t BASE_MN_LIMIT = 256 * 256;
constexpr int64_t BASE_MK_LIMIT = 256 * 64;
constexpr int64_t BLOCK_NUM = 32;
constexpr int64_t BASE_BLOCK_MIN = 96;
constexpr int64_t BASE_BLOCK_MAX = 512;
constexpr int64_t BLOCK_CUBE = 16;
constexpr int64_t FACTOR_2 = 2;

struct BasicBlock {
    int64_t baseM;
    int64_t baseN;
    int64_t baseK;
    double mte2BW;
    double mte2MinBW;
    double mte2BWRatio;
    double mte2TailBWRatio;
};

struct RowView {
    const BasicBlock* ptr;
    std::size_t len;

    const BasicBlock* begin() const
    {
        return ptr;
    }
    const BasicBlock* end() const
    {
        return ptr + len;
    }

    const BasicBlock& operator[](std::size_t i) const
    {
        return ptr[i];
    }
    std::size_t size() const
    {
        return len;
    }
};

// 定义在类内无法通过编译(used before its definition)
namespace WeightQuantBatchMatmulV2Detail {
static constexpr double FREQUENCY_v100 = 0.0016;
static constexpr double HBM_BW_V100 = 1.6;
static constexpr double L2_BW_V100 = 5.4;

static constexpr int64_t FloorAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a / b) * b;
}

static constexpr int64_t Bits(int64_t n)
{
    int bits = 0;
    do {
        ++bits;
        n >>= 1;
    } while (n);
    return bits;
}

static constexpr int64_t BitsAndOne(int64_t n)
{
    return Bits(n) + 1;
}

template <int64_t BitsA, int64_t BitsB>
static constexpr double CalcMte2Bw(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim)
{
    double denom =
        static_cast<double>(mDim * baseM * BitsA + nDim * baseN * BitsB) / HBM_BW_V100 +
        static_cast<double>((mDim * nDim - mDim) * baseM * BitsA + baseN * (mDim * nDim - nDim) * BitsB) / L2_BW_V100;
    if (denom == 0) {
        return 0.0;
    }
    return (static_cast<double>(mDim * nDim) * (baseM * BitsA + BitsB * baseN)) / denom *
           (static_cast<double>(mDim * nDim) / static_cast<double>(BLOCK_NUM));
}

template <int64_t BitsA, int64_t BitsB>
static constexpr double CalcMinMte2Bw(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim)
{
    if (baseM == 0 || baseN == 0) {
        return 0.0;
    }
    return BLOCK_CUBE * BLOCK_CUBE * BLOCK_CUBE * FREQUENCY_v100 * static_cast<double>(mDim * nDim) *
           (BitsA / static_cast<double>(baseN) + BitsB / static_cast<double>(baseM)) / BITS_PER_BYTE;
}

template <int64_t BlockNum, int64_t BitsA, int64_t BitsB, int64_t StartM>
static constexpr std::array<std::size_t, BitsAndOne(BlockNum)> CreateOffsetTable()
{
    std::array<std::size_t, BitsAndOne(BlockNum)> out{};
    std::size_t cnt = 0;
    std::size_t idx = 1;
    out[0] = 0;
    for (int64_t mDim = 1; mDim <= BlockNum; mDim *= FACTOR_2) {
        int64_t nDim = BlockNum / mDim;
        for (int64_t baseM = StartM; baseM <= BASE_BLOCK_MAX; baseM += BLOCK_CUBE) {
            int64_t startN = std::min<int64_t>(BASE_BLOCK_MAX, FloorAlign(BASE_MN_LIMIT / baseM, BLOCK_CUBE));
            for (int64_t baseN = startN; baseN >= BLOCK_CUBE; baseN -= BLOCK_CUBE) {
                double mte2Bw = CalcMte2Bw<BitsA, BitsB>(baseM, baseN, mDim, nDim);
                double minMte2Bw = CalcMinMte2Bw<BitsA, BitsB>(baseM, baseN, mDim, nDim);
                if (minMte2Bw == 0) {
                    continue;
                }
                double bwRatio = mte2Bw / minMte2Bw;
                if (bwRatio > 1.0) {
                    ++cnt;
                }
            }
        }
        out[idx++] = cnt;
    }
    return out;
}

template <std::size_t N, int64_t BlockNum, int64_t BitsA, int64_t BitsB, int64_t StartM>
static constexpr std::array<BasicBlock, N> CreateTable()
{
    std::array<BasicBlock, N> out{};
    std::size_t idx = 0;
    for (int64_t mDim = 1; mDim <= BlockNum; mDim *= FACTOR_2) {
        int64_t nDim = BlockNum / mDim;
        for (int64_t baseM = StartM; baseM <= BASE_BLOCK_MAX; baseM += BLOCK_CUBE) {
            int64_t startN = std::min<int64_t>(BASE_BLOCK_MAX, FloorAlign(BASE_MN_LIMIT / baseM, BLOCK_CUBE));
            for (int64_t baseN = startN; baseN >= BLOCK_CUBE; baseN -= BLOCK_CUBE) {
                double mte2Bw = CalcMte2Bw<BitsA, BitsB>(baseM, baseN, mDim, nDim);
                double minMte2Bw = CalcMinMte2Bw<BitsA, BitsB>(baseM, baseN, mDim, nDim);
                if (minMte2Bw == 0) {
                    continue;
                }
                double bwRatio = mte2Bw / minMte2Bw;
                if (bwRatio > 1.0) {
                    int64_t baseK = FloorAlign(BASE_MK_LIMIT / std::max<int64_t>(baseM, baseN), BLOCK_CUBE);
                    out[idx++] = BasicBlock{baseM, baseN, baseK, mte2Bw, minMte2Bw, bwRatio, 0.0};
                }
            }
        }
    }
    return out;
}

template <size_t... Is, std::size_t TableSize, std::size_t OffsetSize>
static constexpr std::array<RowView, sizeof...(Is)> CreateRowViewsImpl(
    std::index_sequence<Is...>, const std::array<BasicBlock, TableSize>& basicBlockTable,
    const std::array<std::size_t, OffsetSize>& rowOffsets)
{
    return {{RowView{&basicBlockTable[rowOffsets[Is]], rowOffsets[Is + 1] - rowOffsets[Is]}...}};
}

template <std::size_t TableSize, std::size_t OffsetSize>
static constexpr auto CreateRowViews(
    const std::array<BasicBlock, TableSize>& basicBlockTable, const std::array<std::size_t, OffsetSize>& rowOffsets)
{
    return CreateRowViewsImpl(std::make_index_sequence<OffsetSize - 1>{}, basicBlockTable, rowOffsets);
}
} // namespace WeightQuantBatchMatmulV2Detail

class WeightQuantBatchMatmulV2BasicBlockTable {
public:
    static const std::array<RowView, 6>& GetBasicBlockTable(int64_t aDtypeBits, int64_t bDtypeBits)
    {
        if (aDtypeBits == BITS_16 && bDtypeBits == BITS_4) {
            return A16W4_ROW_VIEWS;
        }
        return A16W8_ROW_VIEWS;
    }

private:
    static constexpr auto A16W8_ROW_OFFSET =
        WeightQuantBatchMatmulV2Detail::CreateOffsetTable<BLOCK_NUM, BITS_16, BITS_8, BASE_BLOCK_MIN>();
    static constexpr auto A16W8_TABLE_SIZE = A16W8_ROW_OFFSET.back();
    static constexpr auto A16W8_BASIC_BLOCK_TABLE_V100 =
        WeightQuantBatchMatmulV2Detail::CreateTable<A16W8_TABLE_SIZE, BLOCK_NUM, BITS_16, BITS_8, BASE_BLOCK_MIN>();
    static constexpr auto A16W8_ROW_VIEWS =
        WeightQuantBatchMatmulV2Detail::CreateRowViews(A16W8_BASIC_BLOCK_TABLE_V100, A16W8_ROW_OFFSET);

    static constexpr auto A16W4_ROW_OFFSET =
        WeightQuantBatchMatmulV2Detail::CreateOffsetTable<BLOCK_NUM, BITS_16, BITS_4, BASE_BLOCK_MIN / FACTOR_2>();
    static constexpr auto A16W4_TABLE_SIZE = A16W4_ROW_OFFSET.back();
    static constexpr auto A16W4_BASIC_BLOCK_TABLE_V100 = WeightQuantBatchMatmulV2Detail::CreateTable<
        A16W4_TABLE_SIZE, BLOCK_NUM, BITS_16, BITS_4, BASE_BLOCK_MIN / FACTOR_2>();
    static constexpr auto A16W4_ROW_VIEWS =
        WeightQuantBatchMatmulV2Detail::CreateRowViews(A16W4_BASIC_BLOCK_TABLE_V100, A16W4_ROW_OFFSET);
};
} // namespace optiling

#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_BASIC_BLOCK_TABLE_H