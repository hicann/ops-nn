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
 * \file transpose_batch_forward.h
 * \brief 完成transpose与cast功能
 */

#ifndef TRANSPOSE_BATCH_FORWARD_H
#define TRANSPOSE_BATCH_FORWARD_H
#include "common.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U, typename Q>
class TransposeBatchForward {
public:
    __aicore__ inline TransposeBatchForward() {}

    __aicore__ inline void Init(GlobalTensor<T>& srcGm, GlobalTensor<Q>& dstGm,
                                LocalTensor<uint8_t>& allUbLocal) {
        this->srcGm = srcGm;
        this->dstGm = dstGm;
        this->allUbLocal = allUbLocal;
        // 分配ub空间
        this->srcUbLocal = this->allUbLocal.template ReinterpretCast<U>(); // 在有需要时原地cast
        auto dstUbLocalStart = this->allUbLocal[CACHE_CAPACITY * sizeof(U)];
        this->dstUbLocal = dstUbLocalStart.template ReinterpretCast<U>();
    }

    __aicore__ inline void SetShape(uint32_t batchSize, uint32_t rows, uint32_t cols) {
        this->batchSize = batchSize;
        this->rows = rows;
        this->cols = cols;
    }

    __aicore__ inline void SetCoreId(uint32_t* coreId) {
        this->coreId = coreId;
    }

    __aicore__ inline void SetCoreNums(uint32_t coreNums) {
        this->coreNums = coreNums;
    }

    __aicore__ inline void SetTransposeOffset(GlobalTensor<int32_t>& transposeOffsetGm) {
        this->transposeOffsetGm = transposeOffsetGm;
    }

    __aicore__ inline void Process() {
        if (this->rows <= BASE_TILE_SIZE) {
            this->ReadTransposeOffset();
            this->isRowSmall = true;
        } else if (this->cols <= BASE_TILE_SIZE) {
            this->ReadTransposeOffset();
            this->isColSmall = true;
        }

        if (this->cols >= this->rows) {
            this->ProcessByColumns();
        } else {
            this->ProcessByRows();
        }
    }

private:
    struct TileSize {
        uint32_t colTileSize;
        uint32_t rowTileSize;
    };

    struct SplitParams {
        uint32_t baseSize;
        uint32_t extraSize;
        uint64_t usedCores;
    };

    __aicore__ inline TileSize CalcTileSize() {
        TileSize tileSize{BASE_TILE_SIZE, BASE_TILE_SIZE};
        if (this->rows <= BASE_TILE_SIZE) {
            uint32_t rowsAlign = (this->rows + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            uint32_t multiple = BASE_TILE_SIZE / rowsAlign;
            tileSize.rowTileSize = BASE_TILE_SIZE / multiple;
            tileSize.colTileSize = BASE_TILE_SIZE * multiple;
        } else if (this->cols <= BASE_TILE_SIZE) {
            uint32_t colsAlign = (this->cols + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            uint32_t multiple = BASE_TILE_SIZE / colsAlign;
            tileSize.colTileSize = BASE_TILE_SIZE / multiple;
            tileSize.rowTileSize = BASE_TILE_SIZE * multiple;
        }
        return tileSize;
    }

    __aicore__ inline uint32_t CalcAlignedSize(bool byColumns) {
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(T);
        uint32_t dim = byColumns ? this->rows : this->cols;
        if (dim <= BASE_TILE_SIZE) {
            uint32_t dimAlign = (dim + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            uint32_t multiple = BASE_TILE_SIZE / dimAlign;
            aliged = BASE_TILE_SIZE * multiple;
        }
        return aliged;
    }

    __aicore__ inline SplitParams CalcSplitParams(uint32_t totalSize, uint64_t batchCores, uint32_t aliged) {
        SplitParams params;
        params.usedCores = batchCores > 0 ? batchCores : 1;
        params.baseSize = totalSize / params.usedCores;
        params.extraSize = totalSize % params.usedCores;
        aliged = aliged > 0 ? aliged : 1;
        if (params.baseSize > 0) {
            params.baseSize = ((params.baseSize + aliged - 1) / aliged) * aliged;
            params.usedCores = totalSize / params.baseSize;
            params.extraSize = totalSize - params.usedCores * params.baseSize;
        }
        return params;
    }

    __aicore__ inline void ProcessTask(uint32_t startCol, uint32_t taskCols, uint32_t batchSrcGm, 
                                       uint32_t batchDstGm, uint32_t startRow, uint32_t taskRows) {
        TileSize tileSize = this->CalcTileSize();
        
        uint32_t colBlocks = taskCols / tileSize.colTileSize;
        uint32_t colLeft = taskCols % tileSize.colTileSize;
        uint32_t rowBlocks = taskRows / tileSize.rowTileSize;
        uint32_t rowLeft = taskRows % tileSize.rowTileSize;
        
        for (uint32_t rowBlockIdx = 0; rowBlockIdx <= rowBlocks; rowBlockIdx++) {
            if (rowBlockIdx == rowBlocks && rowLeft == 0) break;
            uint32_t rowSize = (rowBlockIdx == rowBlocks ? rowLeft : tileSize.rowTileSize);
            
            for (uint32_t colBlockIdx = 0; colBlockIdx <= colBlocks; colBlockIdx++) {
                if (colBlockIdx == colBlocks && colLeft == 0) break;
                uint32_t colSize = (colBlockIdx == colBlocks ? colLeft : tileSize.colTileSize);
                
                uint32_t srcOffset = batchSrcGm + (startRow + rowBlockIdx * tileSize.rowTileSize) * this->cols + 
                                     startCol + colBlockIdx * tileSize.colTileSize;
                uint32_t dstOffset = batchDstGm + (startCol + colBlockIdx * tileSize.colTileSize) * this->rows + 
                                     rowBlockIdx * tileSize.rowTileSize + startRow;
                this->LoadDataToUb(srcOffset, rowSize, colSize);
                PIPE_MTE2_S();
                this->DoTranspose(rowSize, colSize);
                PIPE_V_S();
                this->StoreDataToGm(dstOffset, rowSize, colSize);
                PIPE_MTE3_S();
            }
        }
    }

    __aicore__ inline void GetNextCore(uint32_t* coreId) {
        *coreId += 1;
        if (*coreId == this->coreNums) {
            *coreId = 0;
        }
    }

    __aicore__ inline void ProcessByRows() {
        uint64_t batchCores = this->coreNums / this->batchSize;
        uint32_t aliged = this->CalcAlignedSize(false);
        SplitParams params = this->CalcSplitParams(this->rows, batchCores, aliged);

        for (uint32_t i = 0; i < this->batchSize; i++) {
            uint32_t batchSrcGm = i * this->rows * this->cols;
            uint32_t batchDstGm = i * this->cols * this->rows;
            for (uint32_t j = 0; j <= params.usedCores; j++) {
                if (j == params.usedCores && params.extraSize == 0) break;
                uint32_t transposeRows = (j == params.usedCores ? params.extraSize : params.baseSize);
                uint32_t transposeStartRow = j * params.baseSize;
                this->GetNextCore(this->coreId);
                if (GetBlockIdx() != *this->coreId) continue;
                this->ProcessTask(0, this->cols, batchSrcGm, batchDstGm, transposeStartRow, transposeRows);
            }
        }
    }

    __aicore__ inline void ProcessByColumns() {
        uint64_t batchCores = this->coreNums / this->batchSize;
        uint32_t aliged = this->CalcAlignedSize(true);
        SplitParams params = this->CalcSplitParams(this->cols, batchCores, aliged);

        for (uint32_t i = 0; i < this->batchSize; i++) {
            uint32_t batchSrcGm = i * this->rows * this->cols;
            uint32_t batchDstGm = i * this->cols * this->rows;
            for (uint32_t j = 0; j <= params.usedCores; j++) {
                if (j == params.usedCores && params.extraSize == 0) break;
                uint32_t transposeCols = (j == params.usedCores ? params.extraSize : params.baseSize);
                uint32_t transposeStartCol = j * params.baseSize;
                this->GetNextCore(this->coreId);
                if (GetBlockIdx() != *this->coreId) continue;
                this->ProcessTask(transposeStartCol, transposeCols, batchSrcGm, batchDstGm, 0, this->rows);
            }
        }
    }

    __aicore__ inline void LoadDataToUb(uint32_t srcOffset, uint32_t rowSize, uint32_t colSize) {
        uint32_t colSizeAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t castSize = rowSize * colSizeAligned;
        
        DataCopyExtParams copyParams;
        if (this->isColSmall) {
            auto multiple = BASE_TILE_SIZE / colSizeAligned;
            if (rowSize == BASE_TILE_SIZE * multiple) {
                copyParams = {1, static_cast<uint32_t>(rowSize * colSize * sizeof(T)), 0, 0, 0};
                castSize = rowSize * colSize;
                this->DoDataCopyAndCast(srcOffset, copyParams, castSize);
                return;
            }
        }

        uint32_t dstStride = (colSizeAligned - colSize) / (BYTE_ALIGNMENT / sizeof(T));
        copyParams = {
            static_cast<uint16_t>(rowSize),
            static_cast<uint32_t>(colSize * sizeof(T)),
            static_cast<uint32_t>(this->cols * sizeof(T) - colSize * sizeof(T)),
            dstStride,
            0
        };
        this->DoDataCopyAndCast(srcOffset, copyParams, castSize);
    }

    __aicore__ inline void DoDataCopyAndCast(uint32_t srcOffset, DataCopyExtParams& copyParams, uint32_t castSize) {
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        if constexpr (std::is_same<T, U>::value) {
            DataCopyPad(this->srcUbLocal[0], this->srcGm[srcOffset], copyParams, padParams);
        } else {
            auto srcUbLocalT = std::is_same<T, int64_t>::value ? 
                this->srcUbLocal.template ReinterpretCast<T>() :
                this->srcUbLocal.template ReinterpretCast<T>()[CACHE_CAPACITY];
            DataCopyPad(srcUbLocalT, this->srcGm[srcOffset], copyParams, padParams);
            PIPE_MTE2_S();
            if constexpr (std::is_same<T, bool>::value) {
                Cast(this->srcUbLocal, srcUbLocalT.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, castSize);
            } else {
                Cast(this->srcUbLocal, srcUbLocalT, RoundMode::CAST_NONE, castSize);
            }
            PIPE_V_S();
        }
    }

    __aicore__ inline void StoreDataToGm(uint32_t dstOffset, uint32_t rowSize, uint32_t colSize) {
        uint32_t rowSizeAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t castSize = colSize * rowSizeAligned;
        
        DataCopyExtParams dstCopyParams;
        if (this->isRowSmall) {
            auto multiple = BASE_TILE_SIZE / rowSizeAligned;
            if (colSize == BASE_TILE_SIZE * multiple) {
                castSize = colSize * rowSize;
                LocalTensor<Q> srcLocalQ = this->PrepareSrcLocalQ(castSize);
                dstCopyParams = {1, static_cast<uint32_t>(rowSize * colSize * sizeof(Q)), 0, 0, 0};
                DataCopyPad(this->dstGm[dstOffset], srcLocalQ, dstCopyParams);
                return;
            }
        }

        LocalTensor<Q> srcLocalQ = this->PrepareSrcLocalQ(castSize);
        uint32_t srcUbStride = (rowSizeAligned - rowSize) / (BYTE_ALIGNMENT / sizeof(Q));
        dstCopyParams = {
            static_cast<uint16_t>(colSize),
            static_cast<uint32_t>(rowSize * sizeof(Q)),
            srcUbStride,
            static_cast<uint32_t>(this->rows * sizeof(Q) - rowSize * sizeof(Q)),
            0
        };
        DataCopyPad(this->dstGm[dstOffset], srcLocalQ, dstCopyParams);
    }

    __aicore__ inline LocalTensor<Q> PrepareSrcLocalQ(uint32_t dataSize) {
        LocalTensor<Q> srcLocalQ;
        if constexpr (!std::is_same<U, Q>::value) {
            auto srcLocalU = this->srcUbLocal.template ReinterpretCast<U>();
            DataCopy(srcLocalU, this->dstUbLocal, dataSize);
            PIPE_V_S();
            srcLocalQ = this->dstUbLocal.template ReinterpretCast<Q>();
            if constexpr (!std::is_same<Q, bool>::value) {
                Cast(srcLocalQ, srcLocalU, RoundMode::CAST_RINT, dataSize);
            } else {
                Cast(srcLocalQ.template ReinterpretCast<uint8_t>(), srcLocalU, RoundMode::CAST_RINT, dataSize);
            }
            PIPE_V_S();
        } else {
            srcLocalQ = this->dstUbLocal;
        }
        return srcLocalQ;
    }

    __aicore__ inline void DoTranspose(uint32_t rowSize, uint32_t colSize) {
        if (this->CanUseGatherTranspose(rowSize, colSize)) {
            uint32_t totalSize = rowSize * colSize;
            Gather(this->dstUbLocal, this->srcUbLocal, this->gatherIndicesLocal, 0, totalSize / 2); // 2 is the half of totalSize
            PipeBarrier<PIPE_V>();
            Gather(this->dstUbLocal[totalSize / 2], this->srcUbLocal, this->gatherIndicesLocal[totalSize / 2], 0, totalSize / 2); // 2 is the half of totalSize
            return;
        }

        uint32_t rowSizeAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t colSizeAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        if constexpr (std::is_same<U, float>::value || std::is_same<U, int32_t>::value) {
            LocalTensor<float> dstUbLocal = this->dstUbLocal.template ReinterpretCast<float>();
            LocalTensor<float> srcUbLocal = this->srcUbLocal.template ReinterpretCast<float>();
            TransposeFloat(dstUbLocal, srcUbLocal, rowSizeAligned, colSizeAligned);
        } else if constexpr (std::is_same<U, half>::value || std::is_same<U, bfloat16_t>::value) {
            LocalTensor<half> dstUbLocal = this->dstUbLocal.template ReinterpretCast<half>();
            LocalTensor<half> srcUbLocal = this->srcUbLocal.template ReinterpretCast<half>();
            TransposeHalf(dstUbLocal, srcUbLocal, rowSizeAligned, colSizeAligned);
        }
    }

    __aicore__ inline bool CanUseGatherTranspose(uint32_t rowSize, uint32_t colSize) {
        if (this->isRowSmall) {
            uint32_t rowSizeAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            uint32_t multiple = BASE_TILE_SIZE / rowSizeAligned;
            return colSize == BASE_TILE_SIZE * multiple;
        }
        if (this->isColSmall) {
            uint32_t colSizeAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            uint32_t multiple = BASE_TILE_SIZE / colSizeAligned;
            return rowSize == BASE_TILE_SIZE * multiple;
        }
        return false;
    }

    __aicore__ inline void ReadTransposeOffset() {
        auto gatherLocalStart = CACHE_CAPACITY * sizeof(int32_t) * 2; // 2 is srcUblocal + dstUbLocal
        auto offsetLocal = this->allUbLocal[gatherLocalStart].template ReinterpretCast<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(OFFSET_TABLE_SIZE * OFFSET_TABLE_SIZE * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(offsetLocal, this->transposeOffsetGm, copyParams, padParams);
        PIPE_MTE2_S();
        this->gatherIndicesLocal = offsetLocal.template ReinterpretCast<uint32_t>();
    }

private:
    bool isRowSmall = false;
    bool isColSmall = false;
    LocalTensor<U> srcUbLocal;
    LocalTensor<U> dstUbLocal;
    LocalTensor<uint32_t> gatherIndicesLocal;
    GlobalTensor<T> srcGm;
    GlobalTensor<Q> dstGm;
    LocalTensor<uint8_t> allUbLocal;
    uint32_t batchSize;
    uint32_t rows;
    uint32_t cols;
    uint32_t coreNums;
    uint32_t* coreId;
    GlobalTensor<int32_t> transposeOffsetGm;
};
}
#endif
