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
 * \file execute_scatter.h
 * \brief 调用transpose操作与scatter操作，实现低内存scatter_elements_v2算子。
 */

#ifndef EXECUTE_SCATTER_H
#define EXECUTE_SCATTER_H
#include <type_traits>
#include "scatter_elements_two_dims.h"
#include "transpose_tile_forward.h"
#include "transpose_tile_backward.h"
#include "transpose_batch_forward.h"
#include "transpose_batch_backward.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
constexpr uint32_t ALL_UB_SIZE = 16384 * 3 * 4; // 192 kb

// 宏函数：执行转置操作
#define EXECUTE_TRANSPOSE_OPERATION(obj, srcGm, dstGm, dim0, dim1, start, size) \
    do { \
        obj.SetCoreId(&coreId); \
        obj.Init(srcGm, dstGm, this->allUbLocal); \
        obj.SetShape(dim0, dim1, start, size); \
        obj.SetCoreNums(this->coreNums); \
        obj.Process(); \
    } while(0)

// 宏函数：执行批量转置操作
#define EXECUTE_TRANSPOSE_BATCH_OPERATION(obj, srcGm, dstGm, tileBatchSize, dim0, dim1) \
    do { \
        obj.Init(srcGm, dstGm, this->allUbLocal); \
        obj.SetCoreId(&coreId); \
        obj.SetShape(tileBatchSize, dim0, dim1); \
        obj.SetCoreNums(this->coreNums); \
        obj.Process(); \
    } while(0)

// 宏函数：执行scatter操作
#define EXECUTE_SCATTER_ELEMENTS_TWO_DIMS(obj, xGm, indicesGm, updatesGm, tileBatchSize) \
    do { \
        obj.Init(xGm, indicesGm, updatesGm, this->allUbLocal); \
        obj.SetMode(this->mode); \
        obj.SetUpdatesIsScalar(this->updatesIsScalar); \
        obj.SetXInfo(this->xDim1 * tileBatchSize, this->xDim0); \
        obj.SetIndicesInfo(this->indicesDim1 * tileBatchSize, this->indicesDim0); \
        obj.SetUpdatesInfo(this->updatesDim1 * tileBatchSize, this->updatesDim0); \
        obj.SetCoreNums(this->coreNums); \
        obj.Process(); \
    } while(0)

// 宏函数：声明转置和scatter操作所需的对象
#define DECLARE_TRANSPOSE_BATCH_OBJECTS(T, U, Q) \
    ScatterElementsV2NS::TransposeBatchForward<T, Q> transposeVarBatchForward; \
    ScatterElementsV2NS::TransposeBatchForward<T, Q> transposeUpdatesBatchForward; \
    ScatterElementsV2NS::TransposeBatchForward<U, int32_t> transposeIndicesBatchForward; \
    ScatterElementsV2NS::TransposeBatchBackward<Q, T> transposeDataBatchBackward; \
    ScatterElementsV2NS::ScatterElementsTwoDims<Q, int32_t> scatterElementsTwoDims;

// 宏函数：声明Tile转置和scatter操作所需的对象
#define DECLARE_TRANSPOSE_TILE_OBJECTS(T, U, Q) \
    ScatterElementsV2NS::TransposeTileForward<T, Q> transposeVarTileForward; \
    ScatterElementsV2NS::TransposeTileForward<T, Q> transposeUpdatesTileForward; \
    ScatterElementsV2NS::TransposeTileForward<U, int32_t> transposeIndicesTileForward; \
    ScatterElementsV2NS::TransposeTileBackward<Q, T> transposeDataTileBackward; \
    ScatterElementsV2NS::ScatterElementsTwoDims<Q, int32_t> scatterElementsTwoDims;

template <typename T, typename U>
class ExecuteScatter {
public:
    __aicore__ inline ExecuteScatter() {}
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
        this->mode = tilingData->mode;
        this->coreNums = tilingData->coreNums;
        this->dim = tilingData->realDim;
        this->updatesIsScalar = tilingData->updatesIsScalar;
        
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

    __aicore__ inline void ProcessMiddleDim() {
        if (this->batchSize == 1) {
            this->ProcessMiddleDimBatchSizeOne();
        } else {
            this->ProcessMiddleDimBatchSizeN();
        }
    }

    __aicore__ inline void ProcessMiddleDimBatchSizeN() {
        int32_t targetParts = this->batchSize;
        uint32_t partSize = this->batchSize / targetParts;
        uint32_t extraBatch = this->batchSize % targetParts;

        using Q = typename std::conditional<
            std::is_same<T, int32_t>::value,
            int32_t,
            typename std::conditional<
                std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                half,
                float
            >::type
        >::type;
        GlobalTensor<Q> xDstGm;
        GlobalTensor<int32_t> indicesDstGm;
        GlobalTensor<Q> updatesDstGm;
        xDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q*>(this->workspace));
        indicesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace + partSize * this->xDim1 * this->xDim0 * sizeof(Q)));
        updatesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q*>(this->workspace + partSize * (this->xDim1 * this->xDim0 * sizeof(Q) +
                                                                this->indicesDim1 * this->indicesDim0 * sizeof(int32_t))));

        DECLARE_TRANSPOSE_BATCH_OBJECTS(T, U, Q)

        GlobalTensor<T> xSrcGm;
        GlobalTensor<U> indicesSrcGm;
        GlobalTensor<T> updatesSrcGm;
        for (uint32_t i = 0; i <= targetParts; i++) {
            if (i == targetParts && extraBatch == 0) {
                break;
            }
            uint32_t tileBatchSize = (i == targetParts ? extraBatch : partSize);
            uint32_t tileStart = i * partSize;

            uint32_t coreId = 0;
            xSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->x) + tileStart * this->xDim1 * this->xDim0);
            indicesSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(this->indices) + tileStart * this->indicesDim1 * this->indicesDim0);
            updatesSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->updates) + tileStart * this->updatesDim1 * this->updatesDim0);

            EXECUTE_TRANSPOSE_BATCH_OPERATION(transposeVarBatchForward, xSrcGm, xDstGm, tileBatchSize, this->xDim0, this->xDim1);
            if (!this->updatesIsScalar) {
                EXECUTE_TRANSPOSE_BATCH_OPERATION(transposeUpdatesBatchForward, updatesSrcGm, updatesDstGm, tileBatchSize, this->updatesDim0, this->updatesDim1);
            } else {
                this->MoveSrcToDst<T, Q>(updatesSrcGm, updatesDstGm);
            }
            EXECUTE_TRANSPOSE_BATCH_OPERATION(transposeIndicesBatchForward, indicesSrcGm, indicesDstGm, tileBatchSize, this->indicesDim0, this->indicesDim1);

            SyncAll();
            EXECUTE_SCATTER_ELEMENTS_TWO_DIMS(scatterElementsTwoDims, xDstGm, indicesDstGm, updatesDstGm, tileBatchSize);

            SyncAll();
            coreId = 0;
            EXECUTE_TRANSPOSE_BATCH_OPERATION(transposeDataBatchBackward, xDstGm, xSrcGm, tileBatchSize, this->xDim1, this->xDim0);
            SyncAll();
        }
    }

    __aicore__ inline void ProcessMiddleDimBatchSizeOne() {
        // 二维场景，先分块，再转置，再scatter，再转置回来
        auto targetParts = this->indicesDim1 / this->coreNums; // 一份coreNum，可切成多少份
        targetParts = targetParts > 0 ? targetParts : 1;
        targetParts = targetParts > 5 ? 5 : targetParts;
        auto partSize = this->indicesDim1 / targetParts; // 每个part的大小，todo: 分核之后不一定是16
        auto extraSize = this->indicesDim1 % targetParts; // 额外的大小
        
        using Q = typename std::conditional<
            std::is_same<T, int32_t>::value,
            int32_t,
            typename std::conditional<
                std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                half,
                float
            >::type
        >::type;
        DECLARE_TRANSPOSE_TILE_OBJECTS(T, U, Q)

        for (uint32_t i = 0; i <= targetParts; i++) {
            if (i == targetParts && extraSize == 0) {
                break;
            }
            auto tileSize = (i == targetParts) ? extraSize : partSize;
            GlobalTensor<Q> xDstGm;
            GlobalTensor<int32_t> indicesDstGm;
            GlobalTensor<Q> updatesDstGm;
            xDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q*>(this->workspace));
            indicesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace + partSize * this->xDim0 * sizeof(Q)));
            updatesDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q*>(this->workspace + partSize * this->xDim0 * sizeof(Q) +
                                                                     partSize * this->indicesDim0 * sizeof(int32_t)));
            uint32_t coreId = 0;
            // 先转置var
            EXECUTE_TRANSPOSE_OPERATION(transposeVarTileForward, this->xGm, xDstGm, this->xDim0, this->xDim1, i * partSize, tileSize);
            
            // 再转置indices
            EXECUTE_TRANSPOSE_OPERATION(transposeIndicesTileForward, this->indicesGm, indicesDstGm, this->indicesDim0, this->indicesDim1, i * partSize, tileSize);
            
            // 最后转置updates
            if (!this->updatesIsScalar) {
                EXECUTE_TRANSPOSE_OPERATION(transposeUpdatesTileForward, this->updatesGm, updatesDstGm, this->updatesDim0, this->updatesDim1, i * partSize, tileSize);
            } else {
                this->MoveSrcToDst<T, Q>(this->updatesGm, updatesDstGm);
            }
            SyncAll(); // todo: tiling里需要做相应的设置
            
            // 最后scatter
            scatterElementsTwoDims.Init(xDstGm, indicesDstGm, updatesDstGm, this->allUbLocal);
            scatterElementsTwoDims.SetMode(this->mode);
            scatterElementsTwoDims.SetUpdatesIsScalar(this->updatesIsScalar);
            scatterElementsTwoDims.SetXInfo(tileSize, this->xDim0);
            scatterElementsTwoDims.SetIndicesInfo(tileSize, this->indicesDim0);
            scatterElementsTwoDims.SetUpdatesInfo(tileSize, this->updatesDim0);
            scatterElementsTwoDims.SetCoreNums(this->coreNums);
            scatterElementsTwoDims.Process();
            SyncAll();
            
            // 最后转置回来
            coreId = 0;
            EXECUTE_TRANSPOSE_OPERATION(transposeDataTileBackward, xDstGm, this->xGm, tileSize, this->xDim0, i * partSize, this->xDim1);
            SyncAll();
        }

    }

    __aicore__ inline void ProcessLastDim() {
        if constexpr (std::is_same<T, bool>::value || std::is_same<T, float>::value || std::is_same<T, int32_t>::value) {
            ScatterElementsTwoDims<T, U> scatterElementsTwoDims;
            scatterElementsTwoDims.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal);
            scatterElementsTwoDims.SetMode(this->mode);
            scatterElementsTwoDims.SetUpdatesIsScalar(this->updatesIsScalar);
            scatterElementsTwoDims.SetXInfo(this->xDim0, this->xDim1);
            scatterElementsTwoDims.SetIndicesInfo(this->indicesDim0, this->indicesDim1);
            scatterElementsTwoDims.SetUpdatesInfo(this->updatesDim0, this->updatesDim1);
            scatterElementsTwoDims.SetCoreNums(this->coreNums);
            scatterElementsTwoDims.Process();
        }
    }

    template <typename SrcT, typename DstT>
    __aicore__ inline void MoveSrcToDst(GlobalTensor<SrcT> srcGm, GlobalTensor<DstT> dstGm) {
        if constexpr (std::is_same<SrcT, bool>::value || std::is_same<SrcT, uint8_t>::value) {
            SrcT srcValue = srcGm.GetValue(0);
            dstGm.SetValue(0, static_cast<DstT>(static_cast<int32_t>(srcValue)));
            PipeBarrier<PIPE_ALL>();
        } else if constexpr (!std::is_same<SrcT, bfloat16_t>::value) {
            SrcT srcValue = srcGm.GetValue(0);
            dstGm.SetValue(0, static_cast<DstT>(srcValue));
            PipeBarrier<PIPE_ALL>();
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
    uint32_t mode; // 0: scatter, 1: scatter_add
    uint32_t coreNums;
    uint32_t dim; // 1: 中间轴scatter 2: 最后轴scatter
    uint32_t updatesIsScalar;// 0: 非scalar 1: scalar
    LocalTensor<uint8_t> allUbLocal;
};
}
#endif