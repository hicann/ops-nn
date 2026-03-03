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
 * \file transpose_tile_backward.h
 * \brief 完成workspace向var/indices/updates的转置，并在转置过程中进行cast操作
 *      T是srcGm即workspace的类型。U是dstGm即var/indices/updates的类型。
 *      U: float T: float
 *      U: int32_t T: int32_t
 *      U: half T: float
 *      U: bf16 T: float
 *      U: uint8 T: half
 *      U: int8 T: half
 *      U: bool T: half
 */

#ifndef TRANSPOSE_TILE_BACKWARD_H
#define TRANSPOSE_TILE_BACKWARD_H
#include "transpose_tile_base.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

// T的类型字节宽度高于U
template <typename T, typename U>
class TransposeTileBackward {
public:
    __aicore__ inline TransposeTileBackward() {}

    /**
     * @brief 初始化transpose_tile_backward类
     * @param srcGm 指向workspace
     * @param dstGm 指向var/indices/updates
     * @param allUbLocal 所有的ub内存空间
     */
    __aicore__ inline void Init(GlobalTensor<T>& srcGm, GlobalTensor<U>& dstGm, LocalTensor<uint8_t>& allUbLocal) {
        this->srcGm = srcGm;
        this->dstGm = dstGm;
        this->allUbLocal = allUbLocal;
        // 分配ub空间
        this->srcUbLocal = this->allUbLocal.template ReinterpretCast<T>(); // 在有需要时原地cast
        this->dstUbLocal = this->allUbLocal[CACHE_CAPACITY * sizeof(T)].template ReinterpretCast<T>();
    }
    
    /**
     * @brief 设置transpose_tile_backward类的参数
     * @param rows workspace的行数
     * @param cols workspace的列数
     * @param tileStartCol 该块transpose后，在var/indices/updates上的起始列
     * @param tensorCols 原始tensor var/indices/updates的列数
     */
    __aicore__ inline void SetShape(uint32_t rows, uint32_t cols, uint32_t tileStartCol,uint32_t tensorCols) {
        this->rows = rows;
        this->cols = cols;
        this->tileStartCol = tileStartCol;
        this->tensorCols = tensorCols;
    }

    TRANSPOSE_TILE_SET_CORE_NUMS()
    TRANSPOSE_TILE_SET_CORE_ID()

    __aicore__ inline void Process() {
        if (this->cols >= this->rows) { 
            // workspace上的列数超过行数，按列分核
            this->CoreSplitByColumns();
        } else {
            this->CoreSplitByRows();
        }
    }

private:
    __aicore__ inline void CoreSplitByRows() {
        uint64_t usedCores = this->coreNums;
        
        uint32_t baseRows;
        uint32_t extraRows;
        CALC_ALIGNED_BASE_EXTRA(this->rows, usedCores, baseRows, extraRows)
        
        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraRows == 0) {
                break;
            }
            uint64_t transposeStartRow = j * baseRows; // 要处理的起始行
            uint64_t transposeRows = j == usedCores ? extraRows : baseRows; // 要处理的行数
            uint64_t transposeCols = this->cols; // 要处理的列数
            uint64_t transposeStartCol = 0;
            this->GetNextCore(this->coreId); // 获取下一个核，用于处理当前块
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            // 当前核处理当前块
            this->DoTileTranspose(transposeStartCol, transposeCols, transposeStartRow, transposeRows);
        }
        
    }

    TRANSPOSE_TILE_GET_NEXT_CORE()

    __aicore__ inline void CoreSplitByColumns() {
        uint64_t usedCores = this->coreNums;
        // 按列分核
        uint32_t baseCols;
        uint32_t extraCols;
        CALC_ALIGNED_BASE_EXTRA(this->cols, usedCores, baseCols, extraCols)

        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraCols == 0) {
                break;
            }
            uint64_t transposeCols = j == usedCores ? extraCols : baseCols; // 要处理的列数
            uint64_t transposeColStart = j * baseCols; // 要处理的起始列
            this->GetNextCore(this->coreId); // 获取下一个核，用于处理当前块
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            // 当前核处理当前块
            this->DoTileTranspose(transposeColStart, transposeCols, 0, this->rows);
        }
    }

    __aicore__ inline void DoTileTranspose(uint32_t startCol, uint32_t taskCols, uint32_t startRow, uint32_t taskRows) {
        // 对rows和cols都按BASE_TILE_SIZE分核，方便双重for循环处理每一块。
        // 对taskCols按BASE_TILE_SIZE分核
        uint32_t colBlocks = taskCols / BASE_TILE_SIZE;
        uint32_t colLeft = taskCols % BASE_TILE_SIZE;
        // 对taskRows按BASE_TILE_SIZE分核
        uint32_t rowBlocks = taskRows / BASE_TILE_SIZE;
        uint32_t rowsLeft = taskRows % BASE_TILE_SIZE;
        
        // 双重for循环处理每一块
        for (uint32_t rowBlockIdx = 0; rowBlockIdx <= rowBlocks; rowBlockIdx++) {
            if (rowBlockIdx == rowBlocks && rowsLeft == 0) {
                break;
            }
            uint32_t rowNums = (rowBlockIdx == rowBlocks ? rowsLeft : BASE_TILE_SIZE); 
            for (uint32_t colBlockIdx = 0; colBlockIdx <= colBlocks; colBlockIdx++) {
                if (colBlockIdx == colBlocks && colLeft == 0) {
                    break;
                }
                uint32_t colSize = (colBlockIdx == colBlocks ? colLeft : BASE_TILE_SIZE); 
                // 计算当前块的起始地址
                uint32_t srcOffset = (startRow + rowBlockIdx * BASE_TILE_SIZE) * this->cols +
                                      startCol + colBlockIdx * BASE_TILE_SIZE;
                // 先定位到原始tensor的行号，再定位到列号
                uint32_t dstOffset = (startCol + colBlockIdx * BASE_TILE_SIZE) * this->tensorCols +
                                      this->tileStartCol + startRow + rowBlockIdx * BASE_TILE_SIZE;

                // 从全局内存加载当前块到ub
                this->LoadDataToUb(srcOffset, rowNums, colSize, this->cols);
                PIPE_MTE2_S();
                this->DoTranspose();
                PIPE_V_S();
                // 存储当前块到全局内存
                this->StoreDataToGm(dstOffset, rowNums, colSize);
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
        DataCopyPadExtParams<T> padParam{true, 0, 0, 0};
        DataCopyPad(this->srcUbLocal[0], this->srcGm[srcOffset], copyParams, padParam);
    }

    /**
     * @brief 将dstUbLocal中colSize * rowSize（中间有气泡）的数据块搬运到全局内存
     * @param dstOffset 全局内存中要存储数据的起始地址，单位为元素
     * @param rowSize 128中有rowSize个元素有效
     * @param colSize colSize块128
     */
    __aicore__ inline void StoreDataToGm(uint32_t dstOffset, uint32_t rowSize, uint32_t colSize) {
        auto dstUbLocalU = this->dstUbLocal.template ReinterpretCast<U>();
        if constexpr (std::is_same<U, half>::value || std::is_same<U, bfloat16_t>::value) {
            Cast(dstUbLocalU, this->dstUbLocal, RoundMode::CAST_RINT, colSize * BASE_TILE_SIZE);
            PIPE_V_S();
        }
        if constexpr (std::is_same<U, uint8_t>::value || std::is_same<U, int8_t>::value) {
            Cast(dstUbLocalU, this->dstUbLocal, RoundMode::CAST_NONE, colSize * BASE_TILE_SIZE);
            PIPE_V_S();
        }
        if constexpr (std::is_same<U, bool>::value) {
            // 非0值需要先归1
            // 1. dstUbLocal与0比较大小，结果存入maskLocal
            // 2. dstUbLocal置1
            // 3. dstUbLocal根据maskLocal赋值为0
            // 4. dstUbLocal转为bool类型
            auto compareMaskLocal = this->srcUbLocal.template ReinterpretCast<uint8_t>();
            CompareScalar(compareMaskLocal, this->dstUbLocal, static_cast<half>(1), CMPMODE::LT, colSize * BASE_TILE_SIZE);
            PipeBarrier<PIPE_V>();
            Duplicate(this->dstUbLocal, static_cast<half>(0), colSize * BASE_TILE_SIZE);
            PipeBarrier<PIPE_V>();
            Select(this->dstUbLocal, compareMaskLocal, this->dstUbLocal, static_cast<half>(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, colSize * BASE_TILE_SIZE);
            PipeBarrier<PIPE_V>();
            Cast(this->dstUbLocal.template ReinterpretCast<uint8_t>(), this->dstUbLocal, RoundMode::CAST_NONE, colSize * BASE_TILE_SIZE);
            PIPE_V_S();
        }

        uint32_t aliged = BYTE_ALIGNMENT / sizeof(U);
        uint32_t rowSizeAligned = (rowSize + aliged - 1) / aliged * aliged;
        uint32_t srcStride = (BASE_TILE_SIZE - rowSizeAligned) / aliged;
        DataCopyExtParams dstCopyParams{
            static_cast<uint16_t>(colSize),
            static_cast<uint32_t>(rowSize * sizeof(U)),
            srcStride, // 源数据前一块数据的尾，到下一块数据的头的距离，单位为32字节
            static_cast<uint32_t>(this->tensorCols * sizeof(U) - rowSize * sizeof(U)), // 目的数据前一块数据的尾，到下一块数据的头的距离，单位为字节
            0
        };
        DataCopyPad(this->dstGm[dstOffset], dstUbLocalU, dstCopyParams);
    }

    /**
     * @brief 对128 * 128的srcUbLocal中的数据做转置，结果存储到dstUbLocal中
     * @param colSize 转置后的行数。gather到colSize * BASE_TILE_SIZE个元素
     */
    __aicore__ inline void DoTranspose() {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value) {
            LocalTensor<float> dstUbLocal = this->dstUbLocal.template ReinterpretCast<float>();
            LocalTensor<float> srcUbLocal = this->srcUbLocal.template ReinterpretCast<float>();
            TransposeFloat(dstUbLocal, srcUbLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        } else if constexpr (std::is_same<T, half>::value) {
            LocalTensor<half> dstUbLocal = this->dstUbLocal.template ReinterpretCast<half>();
            LocalTensor<half> srcUbLocal = this->srcUbLocal.template ReinterpretCast<half>();
            TransposeHalf(dstUbLocal, srcUbLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        }
    }

private:
    GlobalTensor<T> srcGm;
    GlobalTensor<U> dstGm;
    LocalTensor<uint8_t> allUbLocal;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t tensorCols = 0;
    uint32_t coreNums = 0;
    uint32_t* coreId = nullptr;
    uint32_t tileStartCol = 0;
    
    LocalTensor<T> srcUbLocal;
    LocalTensor<T> dstUbLocal;
};
}

# endif