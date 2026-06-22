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
 * \file exec_transpose_and_scatter_elements.h
 * \brief 调用transpose操作与scatter操作，实现低内存的scatter_elements
 */

#ifndef EXECUTE_TRANSPOSE_AND_SCATTER_H
#define EXECUTE_TRANSPOSE_AND_SCATTER_H
#include <type_traits>
#include "scatter_elements.h"
#include "transpose_tile_forward.h"
#include "transpose_tile_backward.h"
#include "transpose_batch_backward.h"
#include "transpose_batch_forward.h"
#include "init_gather_offset.h"

namespace ScatterElementsV2NS {
using namespace AscendC;

template <typename T, typename U, const uint32_t MODE, const bool IsScalar>
class ExecTransposeAndScatterElements {
public:
    __aicore__ inline ExecTransposeAndScatterElements() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates,
                                const ScatterElementsV2TilingData* tilingData, TPipe* tPipe, GM_ADDR workspace) {
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        this->updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates));
        this->indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(indices));
        this->workspace = workspace;
        this->x = x;
        this->indices = indices;
        this->updates = updates;

        this->batchSize = tilingData->batchSize;
        this->xDim0 = tilingData->xDim0;
        this->xDim1 = tilingData->xDim1;
        this->indicesDim0 = tilingData->indicesDim0;
        this->indicesDim1 = tilingData->indicesDim1;
        this->updatesDim0 = tilingData->updatesDim0;
        this->updatesDim1 = tilingData->updatesDim1;
        this->coreNums = tilingData->coreNums;
        this->dim = tilingData->realDim;
        
        TPipe* pipe = tPipe;
        TBuf<TPosition::VECCALC> allUbBuf;
        pipe->InitBuffer(allUbBuf, ALL_UB_SIZE);
        this->allUbLocal = allUbBuf.Get<uint8_t>();
    }
    
    __aicore__ inline void Process() {
        if (this->dim == 1) {
            this->ProcessMiddleDim();
        } else {
            this->ProcessLastDim();
        }
    }

    __aicore__ inline void InitAggOffset(uint64_t indicesDim, uint64_t xDim, uint64_t updatesDim) {
        if (indicesDim <= BLOCK_SIZE) {
            GlobalTensor<int32_t> offsetGmTensor;
            offsetGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace));
            InitGatherOffset<int32_t> initGatherOffset;
            initGatherOffset.Init(offsetGmTensor, this->allUbLocal, indicesDim, BLOCK_SIZE);
            initGatherOffset.SetCoreNums(this->coreNums);
            initGatherOffset.ProcessAggIndices(xDim);
            if constexpr (!IsScalar) {
                if (indicesDim != updatesDim) {
                    GlobalTensor<int32_t> offsetGmTensor1;
                    offsetGmTensor1.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace + AGG_INDICES_NUM * sizeof(int32_t)));
                    InitGatherOffset<T, MODE> initGatherOffset1;
                    initGatherOffset1.Init(offsetGmTensor1, this->allUbLocal, indicesDim, BLOCK_SIZE);
                    initGatherOffset1.SetCoreNums(this->coreNums);
                    initGatherOffset1.ProcessAggUpdates(xDim, updatesDim);
                }
            }
            SyncAll();
        }
    }

    __aicore__ inline void ProcessMiddleDim() {
        this->InitAggOffset(this->indicesDim0, this->xDim0, this->updatesDim0);

        if (this->batchSize == 1) {
            if (this->indicesDim0 == 1 && this->updatesDim0 == 1) {
                this->ProcessMiddleDimBatchSizeOneSpecial();
            } else {
                this->ProcessMiddleDimBatchSizeOne();
            }
        } else {
            this->ProcessMiddleDimBatchSizeMore();
        }
    }

    __aicore__ inline void ProcessMiddleDimBatchSizeMore() {
        uint32_t targetParts, partSize, extraBatch, maxBatch;
        this->CalcBatchSplitParams(this->batchSize, targetParts, partSize, extraBatch, maxBatch);

        using V = typename std::conditional<
                  std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                  half,
                  T
                  >::type;
        GlobalTensor<T> xDstGm, updatesDstGm;
        GlobalTensor<int32_t> indicesDstGm;
        this->SetDstGlobalBuffersForBatch(maxBatch, xDstGm, indicesDstGm, updatesDstGm);

        ScatterElementsV2NS::TransposeBatchForward<T, V, T> transposeVarBatchForward, transposeUpdatesBatchForward;
        ScatterElementsV2NS::TransposeBatchForward<U, int32_t, int32_t> transposeIndicesBatchForward;
        ScatterElementsV2NS::TransposeBatchBackward<T, V, T> transposeDataBatchBackward;
        ScatterElementsV2NS::ScatterElements<T, int32_t, MODE, IsScalar> scatterElements;

        this->SetTransposeOffsets(transposeVarBatchForward, transposeIndicesBatchForward, 
                                  transposeUpdatesBatchForward, transposeDataBatchBackward, maxBatch);
        SyncAll();

        GlobalTensor<T> xSrcGm, updatesSrcGm;
        GlobalTensor<U> indicesSrcGm;
        for (uint32_t i = 0; i <= targetParts; i++) {
            if (i == targetParts && extraBatch == 0) {
                break;
            }
            uint32_t tileBatchSize = (i == targetParts ? extraBatch : partSize);
            uint32_t tileStart = i * partSize;

            uint32_t coreId = 0;
            this->SetSrcGlobalBuffers(tileStart, xSrcGm, indicesSrcGm, updatesSrcGm);
            
            this->ProcessBatchTranspose(transposeVarBatchForward, xSrcGm, xDstGm, coreId, 
                                   BatchTransposeParams{tileBatchSize, this->xDim0, this->xDim1});
            
            if constexpr (!IsScalar) {
                this->ProcessBatchTranspose(transposeUpdatesBatchForward, updatesSrcGm, updatesDstGm, coreId,
                                       BatchTransposeParams{tileBatchSize, this->updatesDim0, this->updatesDim1});
            } else {
                updatesDstGm = this->updatesGm;
            }
            this->ProcessBatchTranspose(transposeIndicesBatchForward, indicesSrcGm, indicesDstGm, coreId,
                                   BatchTransposeParams{tileBatchSize, this->indicesDim0, this->indicesDim1});
            SyncAll();
            this->ProcessScatter(scatterElements, xDstGm, indicesDstGm, updatesDstGm,
                                 ScatterParams{tileBatchSize * this->xDim1, this->xDim0,
                                               tileBatchSize * this->indicesDim1, this->indicesDim0,
                                               tileBatchSize * this->updatesDim1, this->updatesDim0});
            SyncAll();
            coreId = 0;
            this->ProcessBatchTranspose(transposeDataBatchBackward, xDstGm, xSrcGm, coreId,
                                   BatchTransposeParams{tileBatchSize, this->xDim1, this->xDim0});
            SyncAll();
        }
    }

    __aicore__ inline void ProcessMiddleDimBatchSizeOneSpecial() {
        uint32_t targetParts, partSize, extraSize;
        this->CalcPartSplitParams(this->indicesDim1, targetParts, partSize, extraSize);
        uint32_t maxSize = partSize > extraSize ? partSize : extraSize;
        
        using V = typename std::conditional<
                  std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                  half, T >::type;
        ScatterElementsV2NS::TransposeTileForward<T, V, T> transposeVarTileForward;
        ScatterElementsV2NS::TransposeTileBackward<T, V, T> transposeDataTileBackward;
        ScatterElementsV2NS::ScatterElements<T, U, MODE, IsScalar> scatterElements;

        uint32_t offsetIndex = 0;
        this->InitTileOffset<V>(transposeVarTileForward, this->xDim0, maxSize, offsetIndex++, OffsetType::PAD_UNPAD);
        offsetIndex++;
        if constexpr (!IsScalar) {
            offsetIndex++;
        }
        this->InitTileOffset<V>(transposeDataTileBackward, this->xDim0, maxSize, offsetIndex++, OffsetType::PAD_PAD);
        this->InitTileOffset<V>(transposeVarTileForward, this->xDim0, maxSize, offsetIndex++, OffsetType::TRANSPOSE_FORWARD);
        offsetIndex++;
        if constexpr (!IsScalar) {
            offsetIndex++;
        }
        this->InitTileOffset<V>(transposeDataTileBackward, this->xDim0, maxSize, offsetIndex++, OffsetType::TRANSPOSE_BACKWARD);
        SyncAll();
        for (uint32_t i = 0; i <= targetParts; i++) {
            if (i == targetParts && extraSize == 0) {
                break;
            }
            auto tileSize = (i == targetParts) ? extraSize : partSize;
            GlobalTensor<T> xDstGm, updatesDstGm;
            GlobalTensor<U> indicesDstGm;
            auto offsetGmStart = AGG_INDICES_NUM * 2 * sizeof(int32_t); // 每个core需要2个offset
            xDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->workspace + offsetGmStart));

            uint32_t workCoreId = 0;
            // 转置var
            this->ProcessTileTranspose(transposeVarTileForward, this->xGm, xDstGm, workCoreId, TileTransposeParams{this->xDim0, this->xDim1, i * partSize, tileSize, true});
            // 转置indices
            indicesDstGm = this->indicesGm[i * partSize];
            // 转置updates
            if constexpr (!IsScalar) {
                updatesDstGm = this->updatesGm[i * partSize];
            } else {
                updatesDstGm = this->updatesGm;
            }
            SyncAll();
            // 执行scatter
            this->ProcessScatter(scatterElements, xDstGm, indicesDstGm, updatesDstGm, ScatterParams{tileSize, this->xDim0, tileSize, this->indicesDim0, tileSize, this->updatesDim0});
            SyncAll();
            // 转置回来
            workCoreId = 0;
            this->ProcessTileTranspose(transposeDataTileBackward, xDstGm, this->xGm, workCoreId, TileTransposeParams{tileSize, this->xDim0, i * partSize, this->xDim1, false});
            SyncAll();
        }
    }

    __aicore__ inline void ProcessMiddleDimBatchSizeOne() {
        uint32_t targetParts, partSize, extraSize;
        this->CalcPartSplitParams(this->indicesDim1, targetParts, partSize, extraSize);
        uint32_t maxSize = partSize > extraSize ? partSize : extraSize;
        
        using V = typename std::conditional<
                  std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                  half, T >::type;
        ScatterElementsV2NS::TransposeTileForward<T, V, T> transposeVarTileForward, transposeUpdatesTileForward;
        ScatterElementsV2NS::TransposeTileForward<U, int32_t, int32_t> transposeIndicesTileForward;
        ScatterElementsV2NS::TransposeTileBackward<T, V, T> transposeDataTileBackward;
        ScatterElementsV2NS::ScatterElements<T, int32_t, MODE, IsScalar> scatterElements;

        uint32_t offsetIndex = 0;
        this->InitTileOffset<V>(transposeVarTileForward, this->xDim0, maxSize, offsetIndex++, OffsetType::PAD_UNPAD);
        this->InitTileOffset<int32_t>(transposeIndicesTileForward, this->indicesDim0, maxSize, offsetIndex++, OffsetType::PAD_UNPAD);
        if constexpr (!IsScalar) {
            this->InitTileOffset<V>(transposeUpdatesTileForward, this->updatesDim0, maxSize, offsetIndex++, OffsetType::PAD_UNPAD);
        }

        this->InitTileOffset<V>(transposeDataTileBackward, this->xDim0, maxSize, offsetIndex++, OffsetType::PAD_PAD);

        this->InitTileOffset<V>(transposeVarTileForward, this->xDim0, maxSize, offsetIndex++, OffsetType::TRANSPOSE_FORWARD);
        this->InitTileOffset<int32_t>(transposeIndicesTileForward, this->indicesDim0, maxSize, offsetIndex++, OffsetType::TRANSPOSE_FORWARD);
        if constexpr (!IsScalar) {
            this->InitTileOffset<V>(transposeUpdatesTileForward, this->updatesDim0, maxSize, offsetIndex++, OffsetType::TRANSPOSE_FORWARD);
        }

        this->InitTileOffset<V>(transposeDataTileBackward, this->xDim0, maxSize, offsetIndex++, OffsetType::TRANSPOSE_BACKWARD);

        SyncAll();

        for (uint32_t i = 0; i <= targetParts; i++) {
            if (i == targetParts && extraSize == 0) {
                break;
            }
            auto tileSize = (i == targetParts) ? extraSize : partSize;
            GlobalTensor<T> xDstGm, updatesDstGm;
            GlobalTensor<int32_t> indicesDstGm;
            this->SetDstGlobalBuffersForTile(maxSize, xDstGm, indicesDstGm, updatesDstGm);

            uint32_t coreId = 0;
            // 转置var
            this->ProcessTileTranspose(transposeVarTileForward, this->xGm, xDstGm, coreId, TileTransposeParams{this->xDim0, this->xDim1, i * partSize, tileSize, true});
            // 转置indices
            this->ProcessTileTranspose(transposeIndicesTileForward, this->indicesGm, indicesDstGm, coreId, TileTransposeParams{this->indicesDim0, this->indicesDim1, i * partSize, tileSize, true});

            // 转置updates
            if constexpr (!IsScalar) {
                this->ProcessTileTranspose(transposeUpdatesTileForward, this->updatesGm, updatesDstGm, coreId, TileTransposeParams{this->updatesDim0, this->updatesDim1, i * partSize, tileSize, true});
            } else {
                updatesDstGm = this->updatesGm;
            }
            SyncAll();
            
            // 执行scatter
            this->ProcessScatter(scatterElements, xDstGm, indicesDstGm, updatesDstGm, ScatterParams{tileSize, this->xDim0, tileSize, this->indicesDim0, tileSize, this->updatesDim0});
            SyncAll();
            
            // 转置回来
            coreId = 0;
            this->ProcessTileTranspose(transposeDataTileBackward, xDstGm, this->xGm, coreId, TileTransposeParams{tileSize, this->xDim0, i * partSize, this->xDim1, false});
            SyncAll();
        }
    }

    __aicore__ inline void ProcessLastDim() {
        ScatterElements<T, U, MODE, IsScalar> scatterElements;
        this->InitAggOffset(this->indicesDim1, this->xDim1, this->updatesDim1);
        this->ProcessScatter(scatterElements, this->xGm, this->indicesGm, this->updatesGm,
                             ScatterParams{this->xDim0, this->xDim1, this->indicesDim0, this->indicesDim1,
                                           this->updatesDim0, this->updatesDim1});
    }

    template <typename DataType>
    __aicore__ inline GlobalTensor<int32_t> InitTransposeOffset(uint32_t dim0, uint32_t dim1, 
                                                                 uint32_t offsetIndex, bool isForward, uint32_t partSize) {
        GlobalTensor<int32_t> offsetGmTensor;
        uint32_t dimValue = 0;
        bool useForward = isForward;
        
        if (dim0 <= BASE_TILE_SIZE) {
            dimValue = dim0;
            useForward = isForward;
        } else if (dim1 <= BASE_TILE_SIZE) {
            dimValue = dim1;
            useForward = !isForward;
        } else {
            return offsetGmTensor;
        }

        uint32_t dimValueAlign = (dimValue + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        uint32_t offsetGmStart = AGG_INDICES_NUM * 2 * sizeof(int32_t); // 每个core需要2个offset
        offsetGmStart += partSize * this->xDim0 * this->xDim1 * sizeof(T);
        offsetGmStart += partSize * this->indicesDim0 * this->indicesDim1 * sizeof(int32_t);
        if constexpr (!IsScalar) {
            offsetGmStart += partSize * this->updatesDim0 * this->updatesDim1 * sizeof(T);
        }
        offsetGmStart += OFFSET_TABLE_SIZE * OFFSET_TABLE_SIZE * sizeof(int32_t) * offsetIndex;
        offsetGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace + offsetGmStart));

        InitGatherOffset<DataType> initGatherOffset;
        initGatherOffset.Init(offsetGmTensor, this->allUbLocal, dimValue, dimValueAlign);
        initGatherOffset.SetCoreNums(this->coreNums);
        if (useForward) {
            initGatherOffset.ProcessTransposeForward();
        } else {
            initGatherOffset.ProcessTransposeBackward();
        }
        return offsetGmTensor;
    }

    template <typename TransposeVar, typename TransposeIndices, typename TransposeUpdates, typename TransposeData>
    __aicore__ inline void SetTransposeOffsets(TransposeVar& transposeVar, TransposeIndices& transposeIndices,
                                               TransposeUpdates& transposeUpdates, TransposeData& transposeData,
                                               uint32_t partSize) {
        using V = typename std::conditional<
                  std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                  half,
                  T
                  >::type;

        uint32_t offsetIndex = 0;
        auto offsetGmTensor = this->InitTransposeOffset<V>(this->xDim0, this->xDim1, offsetIndex++, true, partSize);
        if (offsetGmTensor.GetSize() > 0) {
            transposeVar.SetTransposeOffset(offsetGmTensor);
        }
        
        offsetGmTensor = this->InitTransposeOffset<int32_t>(this->indicesDim0, this->indicesDim1, offsetIndex++, true, partSize);
        if (offsetGmTensor.GetSize() > 0) {
            transposeIndices.SetTransposeOffset(offsetGmTensor);
        }

        if constexpr (!IsScalar) {
            offsetGmTensor = this->InitTransposeOffset<V>(this->updatesDim0, this->updatesDim1, offsetIndex++, true, partSize);
            if (offsetGmTensor.GetSize() > 0) {
                transposeUpdates.SetTransposeOffset(offsetGmTensor);
            }
        }

        offsetGmTensor = this->InitTransposeOffset<V>(this->xDim0, this->xDim1, offsetIndex++, false, partSize);
        if (offsetGmTensor.GetSize() > 0) {
            transposeData.SetTransposeOffset(offsetGmTensor);
        }
    }

    struct BatchTransposeParams {
        uint64_t batchSize;
        uint64_t dim0;
        uint64_t dim1;
    };

    template <typename TransposeType, typename SrcType, typename DstType>
    __aicore__ inline void ProcessBatchTranspose(TransposeType& transpose, 
                                             GlobalTensor<SrcType>& srcGm, 
                                             GlobalTensor<DstType>& dstGm,
                                             uint32_t& coreId,
                                             const BatchTransposeParams& params) {
        transpose.Init(srcGm, dstGm, this->allUbLocal);
        transpose.SetCoreId(&coreId);
        transpose.SetShape(params.batchSize, params.dim0, params.dim1);
        transpose.SetCoreNums(this->coreNums);
        transpose.Process();
    }

    enum class OffsetType {
        PAD_UNPAD,          // ProcessUnPad -> SetPadOffset
        PAD_PAD,            // ProcessPad -> SetPadOffset
        TRANSPOSE_FORWARD,  // ProcessTransposeForward -> SetTransposeOffset
        TRANSPOSE_BACKWARD  // ProcessTransposeBackward -> SetTransposeOffset
    };

    struct TileTransposeParams {
        uint64_t dim0;
        uint64_t dim1;
        uint64_t tileStart;
        uint64_t tileSize;
        bool isForward;
    };

    struct ScatterParams {
        uint64_t xDim0;
        uint64_t xDim1;
        uint64_t indicesDim0;
        uint64_t indicesDim1;
        uint64_t updatesDim0;
        uint64_t updatesDim1;
    };

    __aicore__ inline void CalcPartSplitParams(uint64_t totalSize, uint32_t& targetParts, 
                                                uint32_t& partSize, uint32_t& extraSize) {
        targetParts = static_cast<uint32_t>(totalSize / this->coreNums);
        targetParts = targetParts > 0 ? targetParts : 1;
        targetParts = targetParts > MAX_BATCH_PARTS ? MAX_BATCH_PARTS : targetParts;
        partSize = static_cast<uint32_t>(totalSize / targetParts);
        extraSize = static_cast<uint32_t>(totalSize % targetParts);
    }

    __aicore__ inline void CalcBatchSplitParams(uint64_t batchSize, uint32_t& targetParts, 
                                                 uint32_t& partSize, uint32_t& extraBatch, uint32_t& maxBatch) {
        targetParts = static_cast<uint32_t>(batchSize <= MAX_BATCH_PARTS ? batchSize : MAX_BATCH_PARTS);
        targetParts = targetParts > 0 ? targetParts : 1;
        partSize = static_cast<uint32_t>(batchSize / targetParts);
        extraBatch = static_cast<uint32_t>(batchSize % targetParts);
        maxBatch = partSize > extraBatch ? partSize : extraBatch;
    }

    __aicore__ inline void SetSrcGlobalBuffers(uint32_t tileStart, 
                                               GlobalTensor<T>& xSrcGm, 
                                               GlobalTensor<U>& indicesSrcGm,
                                               GlobalTensor<T>& updatesSrcGm) {
        xSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->x) + tileStart * this->xDim1 * this->xDim0);
        indicesSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(this->indices) + tileStart * this->indicesDim1 * this->indicesDim0);
        updatesSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->updates) + tileStart * this->updatesDim1 * this->updatesDim0);
    }

    __aicore__ inline void SetDstGlobalBuffersForBatch(uint32_t maxBatch,
                                                       GlobalTensor<T>& xDstGm,
                                                       GlobalTensor<int32_t>& indicesDstGm,
                                                       GlobalTensor<T>& updatesDstGm) {
        auto offsetGmStart = 1024 * 2 * sizeof(int32_t);
        xDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->workspace + offsetGmStart));
        indicesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace + offsetGmStart + maxBatch * this->xDim1 * this->xDim0 * sizeof(T)));
        updatesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->workspace + offsetGmStart + maxBatch * (this->xDim1 * this->xDim0 * sizeof(T) +
                                                                 this->indicesDim1 * this->indicesDim0 * sizeof(int32_t))));
    }

    __aicore__ inline void SetDstGlobalBuffersForTile(uint32_t extraSize,
                                                      GlobalTensor<T>& xDstGm,
                                                      GlobalTensor<int32_t>& indicesDstGm,
                                                      GlobalTensor<T>& updatesDstGm) {
        auto offsetGmStart = AGG_INDICES_NUM * 2 * sizeof(int32_t); // 每个core需要2个offset
        xDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->workspace + offsetGmStart));
        indicesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace + offsetGmStart + extraSize * this->xDim0 * sizeof(T)));
        updatesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->workspace + offsetGmStart + extraSize * this->xDim0 * sizeof(T) +
                                                                 extraSize * this->indicesDim0 * sizeof(int32_t)));
    }

    template <typename ScatterType, typename XType, typename IndicesType, typename UpdatesType>
    __aicore__ inline void ProcessScatter(ScatterType& scatter, 
                                           GlobalTensor<XType>& xGm, 
                                           GlobalTensor<IndicesType>& indicesGm,
                                           GlobalTensor<UpdatesType>& updatesGm,
                                           const ScatterParams& params) {
        scatter.Init(xGm, indicesGm, updatesGm, this->allUbLocal, this->workspace);
        scatter.SetXInfo(params.xDim0, params.xDim1);
        scatter.SetIndicesInfo(params.indicesDim0, params.indicesDim1);
        scatter.SetUpdatesInfo(params.updatesDim0, params.updatesDim1);
        scatter.SetCoreNums(this->coreNums);
        scatter.Process();
    }

    template <typename TransposeType, typename SrcType, typename DstType>
    __aicore__ inline void ProcessTileTranspose(TransposeType& transpose, 
                                                 GlobalTensor<SrcType>& srcGm, 
                                                 GlobalTensor<DstType>& dstGm,
                                                 uint32_t& coreId,
                                                 const TileTransposeParams& params) {
        transpose.SetCoreId(&coreId);
        transpose.Init(srcGm, dstGm, this->allUbLocal);
        transpose.SetShape(params.dim0, params.dim1, params.tileStart, params.tileSize);
        transpose.SetCoreNums(this->coreNums);
        transpose.SetIsForward(params.isForward);
        transpose.Process();
    }

    template <typename DataType, typename TransposeType>
    __aicore__ inline void InitTileOffset(TransposeType& transpose, uint64_t dimValue, 
                                           uint32_t extraSize, uint32_t offsetIndex, OffsetType offsetType) {
        if (dimValue > BASE_TILE_SIZE) return;

        uint32_t value = static_cast<uint32_t>(dimValue);
        uint32_t valueAlign = (value + BYTE_ALIGNMENT - 1) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        
        uint32_t offsetGmStart = AGG_INDICES_NUM * 2 * sizeof(int32_t); // 每个core需要2个offset
        offsetGmStart += extraSize * this->xDim0 * sizeof(T);
        offsetGmStart += extraSize * this->indicesDim0 * sizeof(int32_t);
        if constexpr (!IsScalar) {
            offsetGmStart += extraSize * this->updatesDim0 * sizeof(T);
        }
        
        // 计算额外的偏移量
        // offsetIndex 0-3: BLOCK_SIZE * OFFSET_TABLE_SIZE * sizeof(uint32_t) * offsetIndex
        // offsetIndex 4-7: BLOCK_SIZE * OFFSET_TABLE_SIZE * sizeof(uint32_t) * 4 + OFFSET_TABLE_SIZE * OFFSET_TABLE_SIZE * sizeof(uint32_t) * (offsetIndex - 4)
        if (offsetIndex < 4) {
            offsetGmStart += BLOCK_SIZE * OFFSET_TABLE_SIZE * sizeof(uint32_t) * offsetIndex;
        } else {
            offsetGmStart += BLOCK_SIZE * OFFSET_TABLE_SIZE * sizeof(uint32_t) * 4;
            offsetGmStart += OFFSET_TABLE_SIZE * OFFSET_TABLE_SIZE * sizeof(uint32_t) * (offsetIndex - 4);
        }

        GlobalTensor<int32_t> offsetGmTensor;
        offsetGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace + offsetGmStart));

        InitGatherOffset<DataType> initGatherOffset;
        initGatherOffset.Init(offsetGmTensor, this->allUbLocal, value, valueAlign);
        initGatherOffset.SetCoreNums(this->coreNums);

        switch (offsetType) {
            case OffsetType::PAD_UNPAD:
                initGatherOffset.ProcessUnPad();
                transpose.SetPadOffset(offsetGmTensor);
                break;
            case OffsetType::PAD_PAD:
                initGatherOffset.ProcessPad();
                transpose.SetPadOffset(offsetGmTensor);
                break;
            case OffsetType::TRANSPOSE_FORWARD:
                initGatherOffset.ProcessTransposeForward();
                transpose.SetTransposeOffset(offsetGmTensor);
                break;
            case OffsetType::TRANSPOSE_BACKWARD:
                initGatherOffset.ProcessTransposeBackward();
                transpose.SetTransposeOffset(offsetGmTensor);
                break;
        }
    }

private:
    GM_ADDR workspace;
    GlobalTensor<T> xGm;
    GlobalTensor<T> updatesGm;
    GlobalTensor<U> indicesGm;

    GM_ADDR x;
    GM_ADDR indices;
    GM_ADDR updates;

    uint32_t batchSize;
    uint64_t xDim0 = 0; // x.shape[0]
    uint64_t xDim1 = 0; // x.shape[1]
    uint64_t indicesDim0 = 0; // indices.shape[0]
    uint64_t indicesDim1 = 0; // indices.shape[1]
    uint64_t updatesDim0 = 0; // updates.shape[0]
    uint64_t updatesDim1 = 0; // updates.shape[1]
    uint32_t coreNums;
    uint32_t dim; // 1: 中间轴scatter 2: 最后轴scatter
    LocalTensor<uint8_t> allUbLocal;
};
}
#endif