/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_EXPERIMENTAL_INDEX_KERNEL_INDEX_KERNEL_H_
#define OPS_NN_EXPERIMENTAL_INDEX_KERNEL_INDEX_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "index_tiling_data.h"

namespace IndexExperimental {
using namespace AscendC;
constexpr uint32_t INDEX_FAST_COPY_BYTES = 16 * 1024;

template <typename T, typename IndexT>
class IndexKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices, GM_ADDR y,
                                const optiling::IndexTilingData* tiling)
    {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        sizesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indexedSizes));
        stridesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indexedStrides));
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(indices));
        for (uint32_t i = 0; i < tiling_->get_indexedDimNum(); ++i) {
            indexGm_[i].SetGlobalBuffer(inputList_.GetDataPtr<__gm__ IndexT>(i));
        }
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        pipe_.InitBuffer(copyQueue_, 1, INDEX_FAST_COPY_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->get_totalNum() == 0 || blockNum_ == 0) {
            return;
        }
        if (IsLeadingContinuous()) {
            if (GetTailSize() > 1) {
                ProcessLeadingContinuous();
                return;
            }
            ProcessLeadingScalar();
            return;
        }
        ProcessGeneral();
    }

private:
    __aicore__ inline bool IsIndexedDim(uint32_t dim, uint32_t& indexOrdinal) const
    {
        for (uint32_t i = 0; i < tiling_->get_indexedDimNum(); ++i) {
            if (tiling_->get_indexedDims()[i] == dim) {
                indexOrdinal = i;
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool IsContinuous() const
    {
        for (uint32_t i = 1; i < tiling_->get_indexedDimNum(); ++i) {
            if (tiling_->get_indexedDims()[i] != tiling_->get_indexedDims()[i - 1] + 1) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool IsLeadingContinuous() const
    {
        for (uint32_t i = 0; i < tiling_->get_indexedDimNum(); ++i) {
            if (tiling_->get_indexedDims()[i] != i) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void DecodeOffset(uint64_t offset, const uint64_t* shape, const uint64_t* stride, uint32_t rank,
                                        uint64_t* coord) const
    {
        (void)shape;
        for (uint32_t i = 0; i < rank; ++i) {
            coord[i] = offset / stride[i];
            offset %= stride[i];
        }
    }

    __aicore__ inline uint64_t CalcIndexOffset(const uint64_t* yCoord, bool continuous, uint32_t indexOrdinal) const
    {
        uint64_t indexOffset = 0;
        uint32_t indexStartInY = continuous ? tiling_->get_indexedDims()[0] : 0;
        const uint64_t* indexInputStrides = tiling_->get_indexInputStrides() + indexOrdinal * INDEX_MAX_RANK;
        for (uint32_t i = 0; i < tiling_->get_indexRank(); ++i) {
            indexOffset += yCoord[indexStartInY + i] * indexInputStrides[i];
        }
        return indexOffset;
    }

    __aicore__ inline uint64_t GetUnindexedYDim(uint32_t xDim, bool continuous) const
    {
        if (!continuous) {
            uint32_t pos = tiling_->get_indexRank();
            for (uint32_t i = 0; i < xDim; ++i) {
                uint32_t ordinal = 0;
                if (!IsIndexedDim(i, ordinal)) {
                    ++pos;
                }
            }
            return pos;
        }
        const uint32_t first = tiling_->get_indexedDims()[0];
        const uint32_t last = tiling_->get_indexedDims()[tiling_->get_indexedDimNum() - 1];
        if (xDim < first) {
            return xDim;
        }
        return xDim - tiling_->get_indexedDimNum() + tiling_->get_indexRank();
    }

    __aicore__ inline uint64_t CalcInputOffset(uint64_t outOffset)
    {
        uint64_t yCoord[INDEX_MAX_RANK] = {0};
        DecodeOffset(outOffset, tiling_->get_yShape(), tiling_->get_yStride(), tiling_->get_yRank(), yCoord);
        const bool continuous = IsContinuous();

        uint64_t srcOffset = 0;
        for (uint32_t xDim = 0; xDim < tiling_->get_xRank(); ++xDim) {
            uint32_t indexOrdinal = 0;
            if (IsIndexedDim(xDim, indexOrdinal)) {
                const uint64_t indexOffset = CalcIndexOffset(yCoord, continuous, indexOrdinal);
                int64_t indexValue = static_cast<int64_t>(indexGm_[indexOrdinal].GetValue(indexOffset));
                const int64_t dimSize = static_cast<int64_t>(tiling_->get_xShape()[xDim]);
                if (indexValue < 0) {
                    indexValue += dimSize;
                }
                srcOffset += static_cast<uint64_t>(indexValue) * static_cast<uint64_t>(stridesGm_.GetValue(xDim));
            } else {
                const uint64_t yDim = GetUnindexedYDim(xDim, continuous);
                srcOffset += yCoord[yDim] * tiling_->get_xStride()[xDim];
            }
        }
        return srcOffset;
    }

    __aicore__ inline uint64_t CalcLeadingInputBase(const uint64_t* indexCoord)
    {
        uint64_t srcOffset = 0;
        for (uint32_t indexOrdinal = 0; indexOrdinal < tiling_->get_indexedDimNum(); ++indexOrdinal) {
            const uint64_t indexOffset = CalcIndexOffset(indexCoord, true, indexOrdinal);
            int64_t indexValue = static_cast<int64_t>(indexGm_[indexOrdinal].GetValue(indexOffset));
            const int64_t dimSize = static_cast<int64_t>(tiling_->get_xShape()[indexOrdinal]);
            if (indexValue < 0) {
                indexValue += dimSize;
            }
            srcOffset += static_cast<uint64_t>(indexValue) * static_cast<uint64_t>(stridesGm_.GetValue(indexOrdinal));
        }
        return srcOffset;
    }

    __aicore__ inline bool IsDenseIndexInputs() const
    {
        for (uint32_t indexOrdinal = 0; indexOrdinal < tiling_->get_indexedDimNum(); ++indexOrdinal) {
            const uint64_t* indexInputStrides = tiling_->get_indexInputStrides() + indexOrdinal * INDEX_MAX_RANK;
            for (uint32_t i = 0; i < tiling_->get_indexRank(); ++i) {
                if (indexInputStrides[i] != tiling_->get_indexStride()[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    __aicore__ inline uint64_t CalcLeadingInputBaseDense(uint64_t indexOffset)
    {
        uint64_t srcOffset = 0;
        for (uint32_t indexOrdinal = 0; indexOrdinal < tiling_->get_indexedDimNum(); ++indexOrdinal) {
            int64_t indexValue = static_cast<int64_t>(indexGm_[indexOrdinal].GetValue(indexOffset));
            const int64_t dimSize = static_cast<int64_t>(tiling_->get_xShape()[indexOrdinal]);
            if (indexValue < 0) {
                indexValue += dimSize;
            }
            srcOffset += static_cast<uint64_t>(indexValue) * static_cast<uint64_t>(stridesGm_.GetValue(indexOrdinal));
        }
        return srcOffset;
    }

    __aicore__ inline uint64_t GetTailSize() const
    {
        uint64_t tailSize = 1;
        for (uint32_t dim = tiling_->get_indexedDimNum(); dim < tiling_->get_xRank(); ++dim) {
            tailSize *= tiling_->get_xShape()[dim];
        }
        return tailSize;
    }

    __aicore__ inline void GetCoreRange(uint64_t total, uint64_t& start, uint64_t& end) const
    {
        const uint64_t base = total / blockNum_;
        const uint64_t tail = total % blockNum_;
        const uint64_t extra = blockIdx_ < tail ? 1 : 0;
        start = blockIdx_ * base + (blockIdx_ < tail ? blockIdx_ : tail);
        end = start + base + extra;
    }

    __aicore__ inline void ProcessLeadingContinuous()
    {
        const uint64_t tailSize = GetTailSize();
        if (tailSize == 0) {
            return;
        }
        const uint64_t maxCopyElems = INDEX_FAST_COPY_BYTES / sizeof(T);
        const uint64_t prefixNum = tiling_->get_totalNum() / tailSize;
        const bool denseIndexInputs = IsDenseIndexInputs();
        uint64_t prefixStart = 0;
        uint64_t prefixEnd = 0;
        GetCoreRange(prefixNum, prefixStart, prefixEnd);
        for (uint64_t prefixOffset = prefixStart; prefixOffset < prefixEnd; ++prefixOffset) {
            uint64_t srcBase = 0;
            if (denseIndexInputs) {
                srcBase = CalcLeadingInputBaseDense(prefixOffset);
            } else {
                uint64_t indexCoord[INDEX_MAX_RANK] = {0};
                DecodeOffset(prefixOffset, tiling_->get_indexShape(), tiling_->get_indexStride(),
                             tiling_->get_indexRank(), indexCoord);
                srcBase = CalcLeadingInputBase(indexCoord);
            }
            const uint64_t dstBase = prefixOffset * tailSize;
            for (uint64_t tailOffset = 0; tailOffset < tailSize; tailOffset += maxCopyElems) {
                const uint64_t copyElems = (tailSize - tailOffset) < maxCopyElems ? (tailSize - tailOffset) :
                                                                                    maxCopyElems;
                LocalTensor<T> local = copyQueue_.template AllocTensor<T>();
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyElems * sizeof(T)), 0, 0, 0};
                DataCopyPad(local, xGm_[srcBase + tailOffset], copyParams, {});
                copyQueue_.template EnQue<T>(local);
                local = copyQueue_.template DeQue<T>();
                DataCopyPad(yGm_[dstBase + tailOffset], local, copyParams);
                copyQueue_.FreeTensor(local);
            }
        }
    }

    __aicore__ inline void ProcessGeneral()
    {
        uint64_t rangeStart = 0;
        uint64_t rangeEnd = 0;
        GetCoreRange(tiling_->get_totalNum(), rangeStart, rangeEnd);
        for (uint64_t outOffset = rangeStart; outOffset < rangeEnd; ++outOffset) {
            yGm_.SetValue(outOffset, xGm_.GetValue(CalcInputOffset(outOffset)));
        }
    }

    __aicore__ inline void ProcessLeadingScalar()
    {
        const bool denseIndexInputs = IsDenseIndexInputs();
        uint64_t prefixStart = 0;
        uint64_t prefixEnd = 0;
        GetCoreRange(tiling_->get_totalNum(), prefixStart, prefixEnd);
        for (uint64_t prefixOffset = prefixStart; prefixOffset < prefixEnd; ++prefixOffset) {
            uint64_t srcOffset = 0;
            if (denseIndexInputs) {
                srcOffset = CalcLeadingInputBaseDense(prefixOffset);
            } else {
                uint64_t indexCoord[INDEX_MAX_RANK] = {0};
                DecodeOffset(prefixOffset, tiling_->get_indexShape(), tiling_->get_indexStride(),
                             tiling_->get_indexRank(), indexCoord);
                srcOffset = CalcLeadingInputBase(indexCoord);
            }
            yGm_.SetValue(prefixOffset, xGm_.GetValue(srcOffset));
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> copyQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int64_t> sizesGm_;
    GlobalTensor<int64_t> stridesGm_;
    GlobalTensor<IndexT> indexGm_[INDEX_MAX_RANK];
    ListTensorDesc inputList_;
    const optiling::IndexTilingData* tiling_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 1;
};
} // namespace IndexExperimental
#endif // OPS_NN_EXPERIMENTAL_INDEX_KERNEL_INDEX_KERNEL_H_
