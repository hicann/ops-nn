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
 * \file transpose_batch_base.h
 * \brief Transpose Batch操作的基类和宏函数定义
 */

#ifndef TRANSPOSE_BATCH_BASE_H
#define TRANSPOSE_BATCH_BASE_H
#include "common.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U>
class TransposeBatchBase {
protected:
    GlobalTensor<T> srcGm;
    GlobalTensor<U> dstGm;
    LocalTensor<uint8_t> allUbLocal;
    uint32_t batchSize;
    uint32_t rows;
    uint32_t cols;
    uint32_t coreNums;
    uint32_t* coreId;
};


#define TRANSPOSE_BATCH_SET_SHAPE() \
    __aicore__ inline void SetShape(uint32_t batchSize, uint32_t rows, uint32_t cols) { \
        this->batchSize = batchSize; \
        this->rows = rows; \
        this->cols = cols; \
    }

#define TRANSPOSE_BATCH_SET_CORE_ID() \
    __aicore__ inline void SetCoreId(uint32_t* coreId) { \
        this->coreId = coreId; \
    }

#define TRANSPOSE_BATCH_SET_CORE_NUMS() \
    __aicore__ inline void SetCoreNums(uint32_t coreNums) { \
        this->coreNums = coreNums; \
    }

#define TRANSPOSE_BATCH_PROCESS() \
    __aicore__ inline void Process() { \
        if (this->cols >= this->rows) { \
            this->ProcessByColumns(); \
        } else { \
            this->ProcessByRows(); \
        } \
    }

#define TRANSPOSE_BATCH_PROCESS_BY_COLUMNS() \
    __aicore__ inline void ProcessByColumns() { \
        uint64_t batchCores = this->coreNums / this->batchSize; \
        batchCores = batchCores > 0 ? batchCores : 1; \
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(T); \
        uint32_t baseCols = this->cols / batchCores; \
        uint32_t extraCols = this->cols % batchCores; \
        if (baseCols > 0) { \
            baseCols = ((baseCols + aliged - 1) / aliged) * aliged; \
            batchCores = this->cols / baseCols; \
            extraCols = this->cols - batchCores * baseCols; \
        } \
        for (uint32_t i = 0; i < this->batchSize; i++) { \
            uint32_t batchSrcGm = i * this->rows * this->cols; \
            uint32_t batchDstGm = i * this->cols * this->rows; \
            for (uint32_t j = 0; j <= batchCores; j++) { \
                if (j == batchCores && extraCols == 0) { \
                    break; \
                } \
                uint64_t transposeCols = j == batchCores ? extraCols : baseCols; \
                uint64_t transposeStartCol = j * baseCols; \
                this->GetNextCore(this->coreId); \
                if (GetBlockIdx() != *this->coreId) { \
                    continue; \
                } \
                this->ProcessTask(transposeStartCol, transposeCols, batchSrcGm, batchDstGm, 0, this->rows); \
            } \
        } \
    }

#define TRANSPOSE_BATCH_PROCESS_BY_ROWS() \
    __aicore__ inline void ProcessByRows() { \
        uint64_t batchCores = this->coreNums / this->batchSize; \
        batchCores = batchCores > 0 ? batchCores : 1; \
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(T); \
        uint32_t baseRows = this->rows / batchCores; \
        uint32_t extraRows = this->rows % batchCores; \
        if (baseRows > 0) { \
            baseRows = ((baseRows + aliged - 1) / aliged) * aliged; \
            batchCores = this->rows / baseRows; \
            extraRows = this->rows - batchCores * baseRows; \
        } \
        for (uint32_t i = 0; i < this->batchSize; i++) { \
            uint32_t batchSrcGm = i * this->rows * this->cols; \
            uint32_t batchDstGm = i * this->cols * this->rows; \
            for (uint32_t j = 0; j <= batchCores; j++) { \
                if (j == batchCores && extraRows == 0) { \
                    break; \
                } \
                uint64_t transposeRows = j == batchCores ? extraRows : baseRows; \
                uint64_t transposeStartRow = j * baseRows; \
                uint64_t transposeCols = this->cols; \
                uint64_t transposeStartCol = 0; \
                this->GetNextCore(this->coreId); \
                if (GetBlockIdx() != *this->coreId) { \
                    continue; \
                } \
                this->ProcessTask(transposeStartCol, transposeCols, batchSrcGm, batchDstGm, transposeStartRow, transposeRows); \
            } \
        } \
    }

#define TRANSPOSE_BATCH_GET_NEXT_CORE() \
    __aicore__ inline void GetNextCore(uint32_t* coreId) { \
        *coreId += 1; \
        if (*coreId == this->coreNums) { \
            *coreId = 0; \
        } \
    }

#define TRANSPOSE_BATCH_PROCESS_TASK() \
    __aicore__ inline void ProcessTask(uint32_t startCol, uint32_t taskCols, uint32_t batchSrcGm, uint32_t batchDstGm, \
                                       uint32_t startRow, uint32_t taskRows) { \
        uint32_t colBlocks = taskCols / BASE_TILE_SIZE; \
        uint32_t colLeft = taskCols % BASE_TILE_SIZE; \
        uint32_t rowBlocks = taskRows / BASE_TILE_SIZE; \
        uint32_t rowLeft = taskRows % BASE_TILE_SIZE; \
        for (uint32_t rowBlockIdx = 0; rowBlockIdx <= rowBlocks; rowBlockIdx++) { \
            if (rowBlockIdx == rowBlocks && rowLeft == 0) { \
                break; \
            } \
            uint32_t rowSize = (rowBlockIdx == rowBlocks ? rowLeft : BASE_TILE_SIZE); \
            for (uint32_t colBlockIdx = 0; colBlockIdx <= colBlocks; colBlockIdx++) { \
                if (colBlockIdx == colBlocks && colLeft == 0) { \
                    break; \
                } \
                uint32_t colSize = (colBlockIdx == colBlocks ? colLeft : BASE_TILE_SIZE); \
                uint32_t srcOffset = (batchSrcGm + (startRow + rowBlockIdx * BASE_TILE_SIZE) * this->cols + startCol + colBlockIdx * BASE_TILE_SIZE); \
                uint32_t dstOffset = (batchDstGm + (startCol + colBlockIdx * BASE_TILE_SIZE) * this->rows + rowBlockIdx * BASE_TILE_SIZE + startRow); \
                this->LoadDataToUb(srcOffset, rowSize, colSize); \
                PIPE_MTE2_S(); \
                this->DoTranspose(); \
                PIPE_V_S(); \
                this->StoreDataToGm(dstOffset, rowSize, colSize); \
                PIPE_MTE3_S(); \
            } \
        } \
    }

}

#endif
