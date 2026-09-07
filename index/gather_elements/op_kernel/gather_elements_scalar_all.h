/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements_scalar_all.h
 * \brief Pure scalar fallback branch for gather_elements.
 */
#ifndef GATHER_ELEMENTS_SCALAR_ALL_H
#define GATHER_ELEMENTS_SCALAR_ALL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gather_elements.h"

namespace AscendC {
template <typename X_T, typename INDEX_T>
class GatherElementsScalarKernel {
public:
    __aicore__ inline GatherElementsScalarKernel() = delete;
    __aicore__ inline GatherElementsScalarKernel(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                                 const GatherElementsTilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(x, index, y);
        ComputeStrides();
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= needCoreNum_) {
            return;
        }
        // balanced partition of the whole index tensor among cores
        int64_t perCore = indicesNum_ / needCoreNum_;
        int64_t remain = indicesNum_ % needCoreNum_;
        int64_t start = blockIdx * perCore + (blockIdx < remain ? blockIdx : remain);
        int64_t count = perCore + (blockIdx < remain ? 1 : 0);
        // process in chunks so that index/y go through UB via DataCopy instead of
        // per-element gm scalar access; x is still read per-element since its
        // gather position is arbitrary.
        for (int64_t done = 0; done < count; done += SCALAR_CHUNK_NUM) {
            int64_t curNum = (count - done) < SCALAR_CHUNK_NUM ? (count - done) : SCALAR_CHUNK_NUM;
            int64_t flatBase = start + done;
            CopyInIdx(flatBase, curNum);
            Compute(flatBase, curNum);
            CopyOutY(flatBase, curNum);
        }
    }

private:
    static constexpr int64_t SCALAR_CHUNK_NUM = 512;
    static constexpr int32_t BUFFER_NUM = 1;

    TQue<TPosition::VECIN, BUFFER_NUM> idxInQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> yOutQue_;

    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDEX_T> indexGm_;
    GlobalTensor<X_T> yGm_;
    int64_t axis_;
    int64_t dims_;
    int64_t indicesNum_;
    int64_t needCoreNum_;
    int64_t xGatherDim_;
    int64_t paramsShape_[8];
    int64_t indicesShape_[8];
    // row-major flat stride of each dim, for index and x respectively
    int64_t idxStride_[8];
    int64_t xStride_[8];

    __aicore__ inline void InitParams(const GatherElementsTilingData& tiling)
    {
        axis_ = tiling.axis;
        dims_ = tiling.dims;
        indicesNum_ = tiling.indices_num;
        needCoreNum_ = tiling.need_core_num;
        for (int32_t i = 0; i < 8; i++) {
            paramsShape_[i] = tiling.params_shape[i];
            indicesShape_[i] = tiling.indices_shape[i];
            idxStride_[i] = 1;
            xStride_[i] = 1;
        }
        xGatherDim_ = (axis_ >= 0 && axis_ < dims_) ? paramsShape_[axis_] : 1;
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(idxInQue_, BUFFER_NUM, SCALAR_CHUNK_NUM * sizeof(INDEX_T));
        pipe.InitBuffer(yOutQue_, BUFFER_NUM, SCALAR_CHUNK_NUM * sizeof(X_T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR index, GM_ADDR y)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T*>(x));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INDEX_T*>(index));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T*>(y));
    }

    __aicore__ inline void ComputeStrides()
    {
        int64_t xStride = 1;
        int64_t idxStride = 1;
        for (int32_t d = dims_ - 1; d >= 0; d--) {
            xStride_[d] = xStride;
            idxStride_[d] = idxStride;
            xStride *= paramsShape_[d];
            idxStride *= indicesShape_[d];
        }
    }

    // Decompose the flat index position into per-dim coordinates and re-compose the
    // corresponding x offset, skipping the gather dim (d == axis).
    __aicore__ inline int64_t ComputeXOffset(int64_t flatIdx) const
    {
        int64_t xOffset = 0;
        int64_t rem = flatIdx;
        for (int32_t d = 0; d < dims_; d++) {
            int64_t coord = rem / idxStride_[d];
            rem -= coord * idxStride_[d];
            if (d != axis_) {
                xOffset += coord * xStride_[d];
            }
        }
        return xOffset;
    }

    __aicore__ inline void CopyInIdx(int64_t flatBase, int64_t curNum)
    {
        LocalTensor<INDEX_T> idxLocal = idxInQue_.AllocTensor<INDEX_T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curNum * sizeof(INDEX_T)), 0, 0, 0};
        DataCopyPadExtParams<INDEX_T> padParams{true, 0, 0, 0};
        DataCopyPad(idxLocal, indexGm_[flatBase], copyParams, padParams);
        idxInQue_.EnQue<INDEX_T>(idxLocal);
    }

    __aicore__ inline void Compute(int64_t flatBase, int64_t curNum)
    {
        LocalTensor<INDEX_T> idxLocal = idxInQue_.DeQue<INDEX_T>();
        LocalTensor<X_T> yLocal = yOutQue_.AllocTensor<X_T>();
        for (int64_t i = 0; i < curNum; i++) {
            int64_t flatIdx = flatBase + i;
            int64_t gatherPos = static_cast<int64_t>(idxLocal.GetValue(i));
            // normalize negative index, keep the same semantics as the optimized branches
            gatherPos = (gatherPos + xGatherDim_) % xGatherDim_;
            int64_t xOffset = ComputeXOffset(flatIdx);
            X_T value = xGm_[xOffset + gatherPos * xStride_[axis_]].GetValue(0);
            yLocal.SetValue(i, value);
        }
        idxInQue_.FreeTensor<INDEX_T>(idxLocal);
        yOutQue_.EnQue<X_T>(yLocal);
    }

    __aicore__ inline void CopyOutY(int64_t flatBase, int64_t curNum)
    {
        LocalTensor<X_T> yLocal = yOutQue_.DeQue<X_T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curNum * sizeof(X_T)), 0, 0, 0};
        DataCopyPad(yGm_[flatBase], yLocal, copyParams);
        yOutQue_.FreeTensor<X_T>(yLocal);
    }
};
} // namespace AscendC

#endif // GATHER_ELEMENTS_SCALAR_H
