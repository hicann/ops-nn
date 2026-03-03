/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_tile_forward.h
 * \brief 完成var/indices/updates向workspace的转置，并在转置过程中进行cast操作
 *      T: float U: float
 *      T: int32_t U: int32_t
 *      T: half U: float
 *      T: bf16 U: float
 *      T: uint8 U: half
 *      T: int8 U: half
 *      T: bool U: half
 *      T: int64 U: int32
 */

#ifndef TRANSPOSE_TILE_FORWARD_H
#define TRANSPOSE_TILE_FORWARD_H
#include "common.h"
#include "transpose_tile_base.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U>
class TransposeTileForward {
public:
    __aicore__ inline TransposeTileForward() {}

    /**
     * @brief 初始化transpose_tile_forward，为transpose_tile_forward分配ub空间
     * @param srcGm 待转置Tensor的地址，非待转置切块的起始地址
     * @param dstGm 切块转置后的起始地址
     * @param allUbLocal 所有ub本地内存
     */
    __aicore__ inline void Init(GlobalTensor<T>& srcGm, GlobalTensor<U>& dstGm, LocalTensor<uint8_t>& allUbLocal) {
        this->srcGm = srcGm;
        this->dstGm = dstGm;
        this->allUbLocal = allUbLocal;
        // 分配ub空间
        this->srcUbLocal = this->allUbLocal.template ReinterpretCast<U>(); // 在有需要时原地cast
        this->dstUbLocal = this->allUbLocal[CACHE_CAPACITY * sizeof(U)].template ReinterpretCast<U>();
    }
    
    /**
     * @brief 设置transpose_tile_forward的参数
     * @param rows 待转置Tensor的行数
     * @param cols 待转置Tensor的列数
     * @param tileStartCol 待转置切块的起始列
     * @param tileCols 待转置切块的列数
     */
    __aicore__ inline void SetShape(uint32_t rows, uint32_t cols, uint32_t tileStartCol,uint32_t tileCols) {
        this->rows = rows;
        this->cols = cols;
        this->tileStartCol = tileStartCol;
        this->tileCols = tileCols;
    }

    __aicore__ inline void Process() {
        if (this->tileCols >= this->rows) { // 该块transpose时，列数大于等于行数，按列分核
            this->CoreSplitByColumns();
        } else {
            this->CoreSplitByRows();
        }
    }

    TRANSPOSE_TILE_SET_CORE_NUMS()
    TRANSPOSE_TILE_SET_CORE_ID()

private:
    TRANSPOSE_TILE_GET_NEXT_CORE()

    __aicore__ inline void CoreSplitByColumns() {
        uint64_t usedCores = this->coreNums;
        // 按列分核
        uint32_t baseCols;
        uint32_t extraCols;
        CALC_ALIGNED_BASE_EXTRA(this->tileCols, usedCores, baseCols, extraCols)

        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraCols == 0) {
                break;
            }
            uint64_t transposeCols = j == usedCores ? extraCols : baseCols; // 要处理的列数
            uint64_t transposeStartCol = j * baseCols; // 要处理的起始列
            this->GetNextCore(this->coreId); // 获取下一个核，用于处理当前块
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            // 当前核处理当前块
            this->DoTileTranspose(transposeStartCol, transposeCols, 0, this->rows);
        }
    }

    __aicore__ inline void CoreSplitByRows() {
        uint64_t usedCores = this->coreNums;
        
        // 按行分核，以复用this->ProcessTask为主要设计思路
        uint32_t baseRows;
        uint32_t extraRows;
        CALC_ALIGNED_BASE_EXTRA(this->rows, usedCores, baseRows, extraRows)
        
        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraRows == 0) {
                break;
            }
            uint64_t transposeRows = j == usedCores ? extraRows : baseRows; // 要处理的行数
            uint64_t transposeStartRow = j * baseRows; // 要处理的起始行
            uint64_t transposeCols = this->tileCols; // 要处理的列数
            uint64_t transposeStartCol = 0; // 要处理的起始列(在this->startCol的基础上)
            this->GetNextCore(this->coreId); // 获取下一个核，用于处理当前块
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            // 当前核处理当前块
            this->DoTileTranspose(transposeStartCol, transposeCols, transposeStartRow, transposeRows);
        }
        
    }

    __aicore__ inline void DoTileTranspose(uint32_t startCol, uint32_t taskCols, uint32_t startRow, uint32_t taskRows) {
        // 对rows和cols都按BASE_TILE_SIZE分核，方便双重for循环处理每一块。
        // 对taskCols按BASE_TILE_SIZE分核
        uint32_t colBlocks = taskCols / BASE_TILE_SIZE;
        uint32_t colLeft = taskCols % BASE_TILE_SIZE;
        // 对taskRows按BASE_TILE_SIZE分核
        uint32_t rowBlocks = taskRows / BASE_TILE_SIZE;
        uint32_t rowLeft = taskRows % BASE_TILE_SIZE;
        
        // 双重for循环处理每一块
        for (uint32_t rowBlockIdx = 0; rowBlockIdx <= rowBlocks; rowBlockIdx++) {
            if (rowBlockIdx == rowBlocks && rowLeft == 0) {
                break;
            }
            uint32_t rowSize = (rowBlockIdx == rowBlocks ? rowLeft : BASE_TILE_SIZE); 
            for (uint32_t colBlockIdx = 0; colBlockIdx <= colBlocks; colBlockIdx++) {
                if (colBlockIdx == colBlocks && colLeft == 0) {
                    break;
                }
                uint32_t colSize = (colBlockIdx == colBlocks ? colLeft : BASE_TILE_SIZE); 
                // 计算当前块的起始地址
                uint32_t srcOffset = ((startRow + rowBlockIdx * BASE_TILE_SIZE) * this->cols + this->tileStartCol + startCol + colBlockIdx * BASE_TILE_SIZE); // 先定位到第几行，再定位到第几列
                uint32_t dstOffset = ((startCol + colBlockIdx * BASE_TILE_SIZE) * this->rows + rowBlockIdx * BASE_TILE_SIZE + startRow); // tileStartCol不参与计算
                // 从全局内存加载当前块到ub
                this->LoadDataToUb(srcOffset, rowSize, colSize, this->cols);
                PIPE_MTE2_S();
                this->DoTranspose();
                PIPE_V_S();
                // 存储当前块到全局内存
                this->StoreDataToGm(dstOffset, rowSize, colSize);
                PIPE_MTE3_S();
            }
        }
    }

    /**
     * @brief 从全局内存加载一块rowSize * colSize的数据块到ub
     * @param srcOffset 全局内存中要加载数据的起始地址，单位为元素
     * @param rowSize 要加载的行数
     * @param colSize 要加载的列数
     * @param rowLength 输入Tensor一行的元素数，单位为元素，等于this->cols
     */
    __aicore__ inline void LoadDataToUb(uint32_t srcOffset, uint32_t rowSize, uint32_t colSize, uint32_t rowLength) {
        INIT_DATA_COPY_PARAMS(copyParams, rowSize, colSize, rowLength)
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value) {
            DataCopyPad(this->srcUbLocal[0], this->srcGm[srcOffset], copyParams, padParams);
        } else {
            auto srcUbLocalT = std::is_same<T, int64_t>::value ? 
                this->srcUbLocal.template ReinterpretCast<T>() : // int64->int32时需要借用this->dstUbLocal的空间
                this->srcUbLocal.template ReinterpretCast<T>()[CACHE_CAPACITY];
            DataCopyPad(srcUbLocalT, this->srcGm[srcOffset], copyParams, padParams);
            PIPE_MTE2_S();
            if constexpr (std::is_same<T, bool>::value) {
                Cast(this->srcUbLocal, srcUbLocalT.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, CACHE_CAPACITY);
            } else {
                Cast(this->srcUbLocal, srcUbLocalT, RoundMode::CAST_NONE, CACHE_CAPACITY);
            }
            PIPE_V_S();
        }
    }

    /**
     * @brief 对128 * 128的srcUbLocal中的数据做转置，结果存储到dstUbLocal中
     */
    __aicore__ inline void DoTranspose() {
        if constexpr (std::is_same<U, float>::value || std::is_same<U, int32_t>::value) {
            LocalTensor<float> dstLocal = this->dstUbLocal.template ReinterpretCast<float>();
            LocalTensor<float> srcLocal = this->srcUbLocal.template ReinterpretCast<float>();
            TransposeFloat(dstLocal, srcLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        } else if constexpr (std::is_same<U, half>::value) {
            LocalTensor<half> dstLocal = this->dstUbLocal.template ReinterpretCast<half>();
            LocalTensor<half> srcLocal = this->srcUbLocal.template ReinterpretCast<half>();
            TransposeHalf(dstLocal, srcLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        }
    }

    /**
     * @brief 将dstUbLocal中colSize * rowSize（中间有气泡）的数据块搬运到全局内存
     * @param dstOffset 全局内存中要存储数据的起始地址，单位为元素
     * @param rowSize 要存储的行数
     * @param colSize 要存储的列数
     */
    __aicore__ inline void StoreDataToGm(uint32_t dstOffset, uint32_t rowSize, uint32_t colSize) {
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(U);
        uint32_t rowSizeAligned = (rowSize + aliged - 1) / aliged * aliged;
        uint32_t srcUbStride = (BASE_TILE_SIZE - rowSizeAligned) / aliged;
        DataCopyExtParams dstCopyParams{
            static_cast<uint16_t>(colSize),
            static_cast<uint32_t>(rowSize * sizeof(U)),
            srcUbStride, // 源数据前一块数据的尾，到下一块数据的头的距离，单位为32字节
            static_cast<uint32_t>(this->rows * sizeof(U) - rowSize * sizeof(U)), // 目的数据前一块数据的尾，到下一块数据的头的距离，单位为字节
            0
        };
        DataCopyPad(this->dstGm[dstOffset], this->dstUbLocal[0], dstCopyParams);
    }

private:
    GlobalTensor<T> srcGm;
    GlobalTensor<U> dstGm;
    LocalTensor<uint8_t> allUbLocal;
    uint32_t rows = 0; // 原始矩阵的行数
    uint32_t cols = 0; // 原始矩阵的列数
    uint32_t tileCols = 0; // 本块要转置的列数（本块要转置的行数就是rows）
    uint32_t coreNums = 0;
    uint32_t* coreId = nullptr;
    uint32_t tileStartCol = 0; // 本块要转置的起始列
    
    LocalTensor<U> srcUbLocal;
    LocalTensor<U> dstUbLocal;
};
}

# endif