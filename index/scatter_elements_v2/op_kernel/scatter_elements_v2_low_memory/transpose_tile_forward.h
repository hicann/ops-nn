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

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U, typename Q>
class TransposeTileBase {
public:
    __aicore__ inline TransposeTileBase() {}

    __aicore__ inline void Init(GlobalTensor<T>& srcGm, GlobalTensor<Q>& dstGm, 
                                LocalTensor<uint8_t>& allUbLocal) {
        this->srcGm = srcGm;
        this->dstGm = dstGm;
        this->allUbLocal = allUbLocal;
        this->srcUbLocal = this->allUbLocal.template ReinterpretCast<U>();
        auto tempUbLoal = this->allUbLocal[CACHE_CAPACITY * sizeof(U)];
        this->dstUbLocal = tempUbLoal.template ReinterpretCast<U>();
    }
    
    __aicore__ inline void SetCoreNums(uint32_t coreNums) {
        this->coreNums = coreNums;
    }

    __aicore__ inline void SetCoreId(uint32_t* coreId) {
        this->coreId = coreId;
    }

    __aicore__ inline void SetPadOffset(GlobalTensor<int32_t>& padGmTensor) {
        this->padGmTensor = padGmTensor;
 	}

    __aicore__ inline void SetTransposeOffset(GlobalTensor<int32_t>& transposeGmTensor) {
        this->transposeGmTensor = transposeGmTensor;
    }

    __aicore__ inline void SetIsForward(bool isForward) {
        this->isForward = isForward;
    }

protected:
    __aicore__ inline void GetNextCore(uint32_t* coreId) {
        *coreId += 1;
        if (*coreId == this->coreNums) {
            *coreId = 0;
        }
    }

    __aicore__ inline void CalcSplitParams(uint32_t totalSize, uint32_t& baseSize, uint32_t& extraSize, uint64_t& usedCores) {
        usedCores = this->coreNums;
        usedCores = usedCores > 0 ? usedCores : 1;
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(T);
        baseSize = totalSize / usedCores;
        extraSize = totalSize % usedCores;
        if (baseSize > 0) {
            baseSize = ((baseSize + aliged - 1) / aliged) * aliged;
            usedCores = totalSize / baseSize;
            extraSize = totalSize - usedCores * baseSize;
        } else {
            usedCores = 0;
        }
    }

    __aicore__ inline void LoadDataToUbContinuoues(uint32_t srcOffset, uint32_t rowSize, uint32_t colSize) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(colSize * rowSize * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        if constexpr (std::is_same<T, U>::value) {
            DataCopyPad(this->srcUbLocal[0], this->srcGm[srcOffset], copyParams, padParams);
        } else {
            auto srcLocalT = std::is_same<T, int64_t>::value ? 
                this->srcUbLocal.template ReinterpretCast<T>() :
                this->srcUbLocal.template ReinterpretCast<T>()[CACHE_CAPACITY];
            DataCopyPad(srcLocalT, this->srcGm[srcOffset], copyParams, padParams);
            PIPE_MTE2_S();
            if constexpr (std::is_same<T, bool>::value) {
                Cast(this->srcUbLocal, srcLocalT.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, rowSize * colSize);
            } else {
                Cast(this->srcUbLocal, srcLocalT, RoundMode::CAST_NONE, rowSize * colSize);
            }
            PIPE_V_S();
        }

        uint32_t colsAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t multiple = BASE_TILE_SIZE / colsAligned;
        if (colSize % BYTE_ALIGNMENT != 0 && rowSize != BASE_TILE_SIZE * multiple) {
            // 反向（isForward == false），行分块且cols <= BASE_TILE_SIZE (isSpecial == true)，且非整块，且cols不是32的整数倍。需要pad
            PIPE_MTE2_S();
            auto colSizeAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            auto gatherTimes = rowSize / BYTE_ALIGNMENT;
            auto gatherLeft = rowSize % BYTE_ALIGNMENT;
            for (uint32_t i = 0; i <= gatherTimes; i++) {
                if (i == gatherTimes && gatherLeft == 0) {
                    break;
                }
                auto gatherNums = (i == gatherTimes ? gatherLeft : BYTE_ALIGNMENT) * colSizeAligned;
                auto srcStart = i * BYTE_ALIGNMENT * colSize;
                auto dstStart = i * BYTE_ALIGNMENT * colSizeAligned;
                Gather(this->dstUbLocal[dstStart], this->srcUbLocal[srcStart], this->gatherIndicesLocal.template ReinterpretCast<uint32_t>(), 0, gatherNums);
                PipeBarrier<PIPE_V>();
            }
            auto temp = this->dstUbLocal;
            this->dstUbLocal = this->srcUbLocal;
            this->srcUbLocal = temp;
            PIPE_V_S();
        }
    }

    __aicore__ inline void LoadDataToUb(uint32_t srcOffset, uint32_t rowSize, uint32_t colSize, uint32_t rowLength) {
        if (this->isSpecial && (!this->isForward)) {
            this->LoadDataToUbContinuoues(srcOffset, rowSize, colSize);
            return;
        }

        uint32_t colSizeAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t dstStride = (colSizeAligned - colSize);
        uint32_t aligned = BYTE_ALIGNMENT / sizeof(T);
        dstStride = dstStride / aligned;

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowSize),
            static_cast<uint32_t>(colSize * sizeof(T)),
            static_cast<uint32_t>(rowLength * sizeof(T) - colSize * sizeof(T)),
            dstStride,
            0
        };
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        if constexpr (std::is_same<T, U>::value) {
            DataCopyPad(this->srcUbLocal[0], this->srcGm[srcOffset], copyParams, padParams);
        } else {
            auto srcUbT = std::is_same<T, int64_t>::value ? 
                this->srcUbLocal.template ReinterpretCast<T>() :
                this->srcUbLocal.template ReinterpretCast<T>()[CACHE_CAPACITY];
            DataCopyPad(srcUbT, this->srcGm[srcOffset], copyParams, padParams);
            PIPE_MTE2_S();
            if constexpr (std::is_same<T, bool>::value) {
                Cast(this->srcUbLocal, srcUbT.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, rowSize * colSizeAligned);
            } else {
                Cast(this->srcUbLocal, srcUbT, RoundMode::CAST_NONE, rowSize * colSizeAligned);
            }
            PIPE_V_S();
        }
    }

    __aicore__ inline void DoTranspose(uint32_t rowSize, uint32_t colSize) {
        if (this->isForward && this->isSpecial) {
            uint32_t rowsAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            uint32_t multiple = BASE_TILE_SIZE / rowsAligned;
            if (colSize == BASE_TILE_SIZE * multiple) {
                //正向，列分块，且rows <= BASE_TILE_SIZE, 且是整块
                Gather(this->dstUbLocal, this->srcUbLocal, this->gatherIndicesLocal, 0, rowSize * colSize / 2); // 2 is the half of rowSize * colSize
                PipeBarrier<PIPE_V>();
                Gather(this->dstUbLocal[rowSize * colSize / 2], this->srcUbLocal, this->gatherIndicesLocal[rowSize * colSize / 2], 0, rowSize * colSize / 2); // 2 is the half of rowSize * colSize
                return;
            }
        }
        if ((!this->isForward) && this->isSpecial) {
            uint32_t colsAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            uint32_t multiple = BASE_TILE_SIZE / colsAligned;
            if (rowSize == BASE_TILE_SIZE * multiple) {
                // 反向，行分块且cols <= BASE_TILE_SIZE，且是整块
                Gather(this->dstUbLocal, this->srcUbLocal, this->gatherIndicesLocal, 0, rowSize * colSize / 2); // 2 is the half of rowSize * colSize
                PipeBarrier<PIPE_V>();
                Gather(this->dstUbLocal[rowSize * colSize / 2], this->srcUbLocal, this->gatherIndicesLocal[rowSize * colSize / 2], 0, rowSize * colSize / 2); // 2 is the half of rowSize * colSize
                return;
            }
        }

        uint32_t colSizeAligned = (colSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t rowSizeAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        if constexpr (std::is_same<U, float>::value || std::is_same<U, int32_t>::value) {
            LocalTensor<float> dstLocal = this->dstUbLocal.template ReinterpretCast<float>();
            LocalTensor<float> srcLocal = this->srcUbLocal.template ReinterpretCast<float>();
            TransposeFloat(dstLocal, srcLocal, rowSizeAligned, colSizeAligned);
        } else if constexpr (std::is_same<U, half>::value || std::is_same<U, bfloat16_t>::value) {
            LocalTensor<half> dstLocal = this->dstUbLocal.template ReinterpretCast<half>();
            LocalTensor<half> srcLocal = this->srcUbLocal.template ReinterpretCast<half>();
            TransposeHalf(dstLocal, srcLocal, rowSizeAligned, colSizeAligned);
        }
    }

    __aicore__ inline void StoreDataToGmContinuous(uint32_t dstOffset, uint32_t rowSize, uint32_t colSize) {
        uint32_t rowsAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t multiple = BASE_TILE_SIZE / rowsAligned;
        if (rowSize % BYTE_ALIGNMENT != 0 && colSize != BASE_TILE_SIZE * multiple) {
            // 非对齐，非整块
            int32_t blockNums = BYTE_ALIGNMENT;
            auto tasks = colSize / blockNums;
            auto leftTasks = colSize % blockNums;
            uint32_t rowSizeAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
            

            for (int32_t i = 0; i <= tasks; i++) {
                if (i == tasks && leftTasks == 0) {
                    break;
                }
                auto taskSize = (i == tasks ? leftTasks : blockNums);
                auto dataNums = taskSize * rowSize;
                auto startSrc = i * blockNums * rowSizeAligned;
                auto startDst = i * blockNums * rowSize;
                Gather(this->srcUbLocal[startDst], this->dstUbLocal[startSrc], this->gatherIndicesLocal, 0, dataNums);
                PipeBarrier<PIPE_V>();
            }
            auto temp = this->dstUbLocal;
            this->dstUbLocal = this->srcUbLocal;
            this->srcUbLocal = temp;
            PIPE_V_S();
        }

        LocalTensor<Q> srcLocalQ;
        if constexpr (!std::is_same<U, Q>::value) {
            srcLocalQ = this->srcUbLocal.template ReinterpretCast<Q>();
            if constexpr (!std::is_same<Q, bool>::value) {
                Cast(srcLocalQ, this->dstUbLocal, RoundMode::CAST_RINT, colSize * rowSize);
            } else {
                Cast(srcLocalQ.template ReinterpretCast<uint8_t>(), this->dstUbLocal, RoundMode::CAST_RINT, colSize * rowSize);
            }
            PIPE_V_S();
        } else {
            srcLocalQ = this->dstUbLocal;
        }

        DataCopyExtParams dstCopyParams{1, static_cast<uint32_t>(colSize * rowSize * sizeof(Q)), 0, 0, 0};
        DataCopyPad(this->dstGm[dstOffset], srcLocalQ, dstCopyParams);
    }

    __aicore__ inline void StoreDataToGm(uint32_t dstOffset, uint32_t rowSize, uint32_t colSize, uint32_t rowLength) {
        if (this->isForward && this->isSpecial) {
            this->StoreDataToGmContinuous(dstOffset, rowSize, colSize);
            return;
        }

        uint32_t rowSizeAligned = (rowSize + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        LocalTensor<Q> srcLocalQ;
        if constexpr (!std::is_same<U, Q>::value) {
            srcLocalQ = this->srcUbLocal.template ReinterpretCast<Q>();
            if constexpr (!std::is_same<Q, bool>::value) {
                Cast(srcLocalQ, this->dstUbLocal, RoundMode::CAST_RINT, colSize * rowSizeAligned);
            } else {
                Cast(srcLocalQ.template ReinterpretCast<uint8_t>(), this->dstUbLocal, RoundMode::CAST_RINT, colSize * rowSizeAligned);
            }
            PIPE_V_S();
        } else {
            srcLocalQ = this->dstUbLocal;
        }

        uint32_t srcUbStride = rowSizeAligned - rowSize;
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(Q);
        srcUbStride = srcUbStride / aliged;
        DataCopyExtParams dstCopyParams{
            static_cast<uint16_t>(colSize),
            static_cast<uint32_t>(rowSize * sizeof(Q)),
            srcUbStride,
            static_cast<uint32_t>(rowLength * sizeof(Q) - rowSize * sizeof(Q)),
            0
        };
        DataCopyPad(this->dstGm[dstOffset], srcLocalQ, dstCopyParams);
    }

    __aicore__ inline void ReadPadOffset() {
        auto gatherLocalStart = CACHE_CAPACITY * sizeof(int32_t) * 2;
        auto offsetLocal = this->allUbLocal[gatherLocalStart].template ReinterpretCast<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(BLOCK_SIZE * OFFSET_TABLE_SIZE * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(offsetLocal, this->padGmTensor, copyParams, padParams);
        PIPE_MTE2_S();
        this->gatherIndicesLocal = offsetLocal.template ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void ReadTransposeOffset() {
        auto gatherLocalStart = CACHE_CAPACITY * sizeof(int32_t) * 2;
        auto offsetLocal = this->allUbLocal[gatherLocalStart].template ReinterpretCast<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(OFFSET_TABLE_SIZE * OFFSET_TABLE_SIZE * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(offsetLocal, this->transposeGmTensor, copyParams, padParams);
        PIPE_MTE2_S();
        this->gatherIndicesLocal = offsetLocal.template ReinterpretCast<uint32_t>();
    }

protected:
    bool isSpecial = false; // true 启动only gather
    GlobalTensor<T> srcGm;
    GlobalTensor<Q> dstGm;
    LocalTensor<uint8_t> allUbLocal;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t coreNums = 0;
    uint32_t* coreId = nullptr;
    uint32_t tileStartCol = 0;
    bool isForward = true;
    LocalTensor<U> srcUbLocal;
    LocalTensor<U> dstUbLocal;
    LocalTensor<uint32_t> gatherIndicesLocal;
    GlobalTensor<int32_t> padGmTensor;
    GlobalTensor<int32_t> transposeGmTensor;
};

template <typename T, typename U, typename Q>
class TransposeTileForward : public TransposeTileBase<T, U, Q> {
public:
    __aicore__ inline void SetShape(uint32_t rows, uint32_t cols, uint32_t tileStartCol, uint32_t tileCols) {
        this->rows = rows;
        this->cols = cols;
        this->tileStartCol = tileStartCol;
        this->tileCols = tileCols;
    }

    __aicore__ inline void Process() {
        if (this->tileCols >= this->rows) {
            if (this->rows <= BASE_TILE_SIZE) {
                this->isSpecial = true; // hight <= BASE_TILE_SIZE 且按列分核，才特殊处理。
            }
            this->CoreSplitByColumns(this->tileCols);
        } else {
            this->CoreSplitByRows(this->tileCols);
        }
        this->isSpecial = false; // 不同的part，不一定全是按列分块
    }

private:
    __aicore__ inline void CoreSplitByColumns(uint32_t taskCols) {
        uint32_t baseCols, extraCols;
        uint64_t usedCores;
        this->CalcSplitParams(taskCols, baseCols, extraCols, usedCores);

        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraCols == 0) {
                break;
            }
            uint64_t transposeCols = j == usedCores ? extraCols : baseCols;
            uint64_t transposeStartCol = j * baseCols;
            this->GetNextCore(this->coreId);
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            this->DoTileTranspose(transposeStartCol, transposeCols, 0, this->rows);
        }
    }

    __aicore__ inline void CoreSplitByRows(uint32_t taskCols) {
        uint32_t baseRows, extraRows;
        uint64_t usedCores;
        this->CalcSplitParams(this->rows, baseRows, extraRows, usedCores);
        
        for (uint32_t j = 0; j <= usedCores; j++) {
            if (j == usedCores && extraRows == 0) {
                break;
            }
            uint64_t transposeRows = j == usedCores ? extraRows : baseRows;
            uint64_t transposeStartRow = j * baseRows;
            uint64_t transposeCols = taskCols;
            uint64_t transposeStartCol = 0;
            this->GetNextCore(this->coreId);
            if (GetBlockIdx() != *this->coreId) {
                continue;
            }
            this->DoTileTranspose(transposeStartCol, transposeCols, transposeStartRow, transposeRows);
        }
    }

    __aicore__ inline void DoTileTranspose(uint32_t startCol, uint32_t taskCols, uint32_t startRow, uint32_t taskRows) {
        uint32_t colTileSize = BASE_TILE_SIZE;
        uint32_t rowTileSize = BASE_TILE_SIZE;
        // rows少，colTileSize变大
        uint32_t taskRowsAligned = (taskRows + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t multiple = BASE_TILE_SIZE / taskRowsAligned;
        multiple = multiple > 0 ? multiple : 1;
        colTileSize = colTileSize * multiple;
        rowTileSize = rowTileSize / multiple;

        uint32_t colBlocks = taskCols / colTileSize;
        uint32_t colLeft = taskCols % colTileSize;
        uint32_t rowBlocks = taskRows / rowTileSize;
        uint32_t rowLeft = taskRows % rowTileSize;

        if (this->isSpecial) {
            if (colBlocks > 0) {
                // 列上有整块，读入做transpose的gather offset
                this->ReadTransposeOffset();
            } else {
                // 列上无整块，读入做pad的gather offset
                this->ReadPadOffset();
            }
        }
        
        for (uint32_t rowBlockIdx = 0; rowBlockIdx <= rowBlocks; rowBlockIdx++) {
            if (rowBlockIdx == rowBlocks && rowLeft == 0) {
                break;
            }
            uint32_t rowSize = (rowBlockIdx == rowBlocks ? rowLeft : rowTileSize); 
            for (uint32_t colBlockIdx = 0; colBlockIdx <= colBlocks; colBlockIdx++) {
                if (colBlockIdx == colBlocks && colLeft == 0) {
                    break;
                }
                uint32_t colSize = (colBlockIdx == colBlocks ? colLeft : colTileSize);
                if (this->isSpecial && colBlockIdx == colBlocks && colBlocks > 0) {
                    // 列上有整块，有尾块。给尾块读入做pad的gather offset
                    this->ReadPadOffset();
                }

                uint32_t srcOffset = ((startRow + rowBlockIdx * rowTileSize) * this->cols + this->tileStartCol + startCol + colBlockIdx * colTileSize);
                uint32_t dstOffset = ((startCol + colBlockIdx * colTileSize) * this->rows + rowBlockIdx * rowTileSize + startRow);
                this->LoadDataToUb(srcOffset, rowSize, colSize, this->cols);
                PIPE_MTE2_S();
                this->DoTranspose(rowSize, colSize);
                PIPE_V_S();
                this->StoreDataToGm(dstOffset, rowSize, colSize, this->rows);
                PIPE_MTE3_S();
            }
        }
    }

private:
    uint32_t tileCols = 0;
};
}

# endif