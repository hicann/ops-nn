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
#include "common.h"
#include "transpose_tile_forward.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U, typename Q>
class TransposeTileBackward : public TransposeTileBase<T, U, Q> {
public:
    __aicore__ inline void SetShape(uint32_t rows, uint32_t cols, uint32_t tileStartCol, uint32_t tensorCols) {
        this->rows = rows;
        this->cols = cols;
        this->tileStartCol = tileStartCol;
        this->tensorCols = tensorCols;
    }

    __aicore__ inline void Process() {
        if (this->cols >= this->rows) {
            this->CoreSplitByColumns(this->cols);
        } else {
            if (this->cols <= BASE_TILE_SIZE) {
                this->isSpecial = true; // weight <= BASE_TILE_SIZE 且按行分核，才特殊处理。
            }
            this->CoreSplitByRows(this->cols);
        }
        this->isSpecial = false; // 不同的part，不一定全是按行分块
    }

private:
    __aicore__ inline void CoreSplitByRows(uint32_t taskCols) {
        uint32_t baseRows, extraRows;
        uint64_t usedCores;
        this->CalcSplitParams(this->rows, baseRows, extraRows, usedCores);
        
        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraRows == 0) {
                break;
            }
            uint64_t transposeStartRow = j * baseRows;
            uint64_t transposeRows = j == usedCores ? extraRows : baseRows;
            uint64_t transposeCols = taskCols;
            uint64_t transposeStartCol = 0;
            this->GetNextCore(this->coreId);
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            this->DoTileTranspose(transposeStartCol, transposeCols, transposeStartRow, transposeRows);
        }
    }

    __aicore__ inline void CoreSplitByColumns(uint32_t taskCols) {
        uint32_t baseCols, extraCols;
        uint64_t usedCores;
        this->CalcSplitParams(taskCols, baseCols, extraCols, usedCores);

        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraCols == 0) {
                break;
            }
            uint64_t transposeCols = j == usedCores ? extraCols : baseCols;
            uint64_t transposeColStart = j * baseCols;
            this->GetNextCore(this->coreId);
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            this->DoTileTranspose(transposeColStart, transposeCols, 0, this->rows);
        }
    }

    __aicore__ inline void DoTileTranspose(uint32_t startCol, uint32_t taskCols, uint32_t startRow, uint32_t taskRows) {
        uint32_t colTileSize = BASE_TILE_SIZE;
        uint32_t rowTileSize = BASE_TILE_SIZE;
        uint32_t taskColsAligned = (taskCols + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t multiple = BASE_TILE_SIZE / taskColsAligned;
        multiple = multiple > 0 ? multiple : 1;
        colTileSize = colTileSize / multiple;
        rowTileSize = rowTileSize * multiple;

        uint32_t colBlocks = taskCols / colTileSize;
        uint32_t colLeft = taskCols % colTileSize;
        uint32_t rowBlocks = taskRows / rowTileSize;
        uint32_t rowsLeft = taskRows % rowTileSize;

        if (this->isSpecial) {
            if (rowBlocks > 0) {
                // 行上有整块，读入做transpose的gather offset
                this->ReadTransposeOffset();
            } else {
                // 行上无整块，读入做pad的gather offset
                this->ReadPadOffset();
            }
        }

        for (uint32_t rowBlockIdx = 0; rowBlockIdx <= rowBlocks; rowBlockIdx++) {
            if (rowBlockIdx == rowBlocks && rowsLeft == 0) {
                break;
            }

            if (this->isSpecial && rowBlockIdx == rowBlocks && rowBlocks > 0) {
                // 行上有整块，有尾块。给尾块读入做pad的gather offset
                this->ReadPadOffset();
            }

            uint32_t rowNums = (rowBlockIdx == rowBlocks ? rowsLeft : rowTileSize); 
            for (uint32_t colBlockIdx = 0; colBlockIdx <= colBlocks; colBlockIdx++) {
                if (colBlockIdx == colBlocks && colLeft == 0) {
                    break;
                }
                uint32_t colSize = (colBlockIdx == colBlocks ? colLeft : colTileSize); 
                uint32_t srcOffset = (startRow + rowBlockIdx * rowTileSize) * this->cols +
                                      startCol + colBlockIdx * colTileSize;
                uint32_t dstOffset = (startCol + colBlockIdx * colTileSize) * this->tensorCols +
                                      this->tileStartCol + startRow + rowBlockIdx * rowTileSize;
                this->LoadDataToUb(srcOffset, rowNums, colSize, this->cols);
                PIPE_MTE2_S();
                this->DoTranspose(rowNums, colSize);
                PIPE_V_S();
                this->StoreDataToGm(dstOffset, rowNums, colSize, this->tensorCols);
                PIPE_MTE3_S();
            }
        }
    }

private:
    uint32_t tensorCols = 0;
};
}

# endif