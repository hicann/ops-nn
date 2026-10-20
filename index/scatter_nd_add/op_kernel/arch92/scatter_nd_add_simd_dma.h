/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_ND_ADD_SIMD_DMA_H_
#define SCATTER_ND_ADD_SIMD_DMA_H_

#include "../arch35/scatter_nd_add_common.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ScatterNdAdd {
using namespace AscendC;

constexpr Reg::CastTrait castTraitB322B64 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
class ScatterNdAddDma {
public:
    using OffsetType = typename std::conditional<isCastIndices, int64_t, U>::type;
    using IndexRegType = typename std::conditional<
        IsSameType<U, int64_t>::value, typename AscendC::Reg::RegTensor<uint64_t, AscendC::Reg::RegTraitNumTwo>,
        typename AscendC::Reg::RegTensor<uint32_t>>::type;
    using CalRegType = typename std::conditional<
        IsSameType<U, int64_t>::value, typename AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo>,
        typename AscendC::Reg::RegTensor<int32_t>>::type;

    using OffsetRegType = typename std::conditional<
        isCastIndices || IsSameType<U, int64_t>::value,
        typename AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo>,
        typename AscendC::Reg::RegTensor<int32_t>>::type;

    __aicore__ inline ScatterNdAddDma(const ScatterNdAddRegBaseTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyInUpdate(LocalTensor<T>& dstTensor, int64_t updatesOffset, int64_t rowLen,
                                        int64_t colLen);
    template <typename OffsetT>
    __aicore__ inline void CopyOutUpdate(LocalTensor<T>& srcTensor, LocalTensor<OffsetT>& offsetLocal, int64_t rowLen,
                                         int64_t colLen);
    __aicore__ inline void ComputeOffset(LocalTensor<U> indicesLocal, LocalTensor<OffsetType> offsetLocal,
                                         int32_t indicesLen, int32_t rankSize);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessNoSplitCol(int64_t rowIdx, int64_t curRow);
    __aicore__ inline void ProcessSplitCol(int64_t rowIdx, int64_t curRow);

private:
    GlobalTensor<T> varGm_;
    GlobalTensor<U> indicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> yGm_;

    TPipe& pipe_;
    const ScatterNdAddRegBaseTilingData& tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> dataQueue_;
    TQue<QuePosition::VECIN, 1> indicesQue_;
    TQue<QuePosition::VECOUT, 1> offsetQue_;
    TBuf<QuePosition::VECCALC> strideBuf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;

    int64_t curCoreIndexCount_{0};
    int64_t curCoreAfterAxisCount_{0};
    int64_t rowStart_{0};
    int64_t colStart_{0};
};

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
__aicore__ inline void ScatterNdAddDma<T, U, isCastIndices, isSplitCol>::Init(GM_ADDR var, GM_ADDR indices,
                                                                              GM_ADDR updates, GM_ADDR y,
                                                                              GM_ADDR workspace)
{
    varGm_.SetGlobalBuffer((__gm__ T*)(var));
    indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
    yGm_.SetGlobalBuffer((__gm__ T*)(y));

    pipe_.InitBuffer(dataQueue_, DOUBLE_BUFFER,
                     tilingData_.indicesFactor * ROUND_UP32(tilingData_.afterAxisFactor * sizeof(T)));
    pipe_.InitBuffer(indicesQue_, 1, ROUND_UP32(tilingData_.indicesFactor * tilingData_.rankSize * sizeof(U)));
    pipe_.InitBuffer(offsetQue_, 1, ROUND_UP32(tilingData_.indicesFactor * sizeof(OffsetType)));
    pipe_.InitBuffer(strideBuf_, MAX_RANK_COUNT * sizeof(U));
    if constexpr (isSplitCol) {
        pipe_.InitBuffer(offsetBuf_, ROUND_UP32(tilingData_.indicesFactor * sizeof(OffsetType)));
    }

    int64_t updateLoopSize = tilingData_.updateLoopSize;
    int64_t rowGroup = static_cast<int64_t>(GetBlockIdx()) / updateLoopSize;
    int64_t colGroup = static_cast<int64_t>(GetBlockIdx()) % updateLoopSize;

    rowStart_ = rowGroup * tilingData_.eachCoreIndexCount;
    colStart_ = colGroup * tilingData_.eachCoreAfterAxisCount;

    int64_t totalRowGroups = ops::CeilDiv(static_cast<int64_t>(tilingData_.blockNum), updateLoopSize);
    curCoreIndexCount_ = (rowGroup == totalRowGroups - 1) ? tilingData_.tailCoreIndexCount :
                                                            tilingData_.eachCoreIndexCount;
    curCoreAfterAxisCount_ = (colGroup == updateLoopSize - 1) ? tilingData_.tailCoreAfterAxisCount :
                                                                tilingData_.eachCoreAfterAxisCount;
}

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
__aicore__ inline void ScatterNdAddDma<T, U, isCastIndices, isSplitCol>::CopyInUpdate(LocalTensor<T>& dstTensor,
                                                                                      int64_t updatesOffset,
                                                                                      int64_t rowLen, int64_t colLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
    DataCopyPad(dstTensor, updatesGm_[updatesOffset], copyParams, updatePadParams);
}

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
template <typename OffsetT>
__aicore__ inline void ScatterNdAddDma<T, U, isCastIndices, isSplitCol>::CopyOutUpdate(
    LocalTensor<T>& srcTensor, LocalTensor<OffsetT>& offsetLocal, int64_t rowLen, int64_t colLen)
{
    // srcTensor 行方向对齐步长: 列按 UB_AGLIN_VALUE 对齐
    int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
    ScatterParams scatterParams;
    scatterParams.offsetNum = static_cast<uint16_t>(rowLen);
    scatterParams.burstLen = static_cast<uint32_t>(colLen * sizeof(T));
    scatterParams.srcStride = static_cast<uint32_t>(colLenAlignSize * sizeof(T));

    if constexpr (IsSameType<T, bool>::value) {
        SetAtomicMax<int8_t>();
    } else if constexpr (IsSameType<T, int64_t>::value) {
        // int64 不支持 SetAtomicAdd, DMA 路径运行时不处理 int64 (SelectTiling 已排除),
        // 此处仅保证编译通过, 不设置 atomic
    } else {
        SetAtomicAdd<T>();
    }

    ScatterDataCopy(yGm_, srcTensor, offsetLocal, scatterParams);
    SetAtomicNone();
}

template <typename U, typename OffsetType, typename OffsetRegType, typename CalRegType, typename IndexRegType,
          bool isCastIndices>
__simd_vf__ inline void ComputeOffsetVF(__ubuf__ U* strideLocalAddr, __ubuf__ U* indicesLocalPtr,
                                        __ubuf__ OffsetType* offsetAddr, OffsetType afterAxisBytes,
                                        OffsetType colStartBytes, OffsetType maxFlatOffset, uint32_t dataLen,
                                        uint32_t indexLen, uint32_t vfLen, uint32_t loopCnt, int32_t rankSize)
{
    OffsetRegType inReg;
    OffsetRegType outReg;
    OffsetRegType tmpReg;
    OffsetRegType invalidFlatReg;
    CalRegType orderReg;
    IndexRegType indexReg;
    Reg::MaskReg pregIndex;
    Reg::MaskReg pregLoop;
    Reg::MaskReg validMask;
    uint16_t rankSizeLoops = static_cast<uint16_t>(rankSize);
    for (uint16_t i = 0; i < static_cast<uint16_t>(loopCnt); i++) {
        if constexpr (IsSameType<U, int32_t>::value && !isCastIndices) {
            pregLoop = Reg::UpdateMask<int32_t>(dataLen);
        } else {
            pregLoop = Reg::UpdateMask<int64_t, Reg::RegTraitNumTwo>(dataLen);
        }
        if constexpr (IsSameType<U, int32_t>::value) {
            pregIndex = Reg::UpdateMask<int32_t>(indexLen);
        } else {
            pregIndex = Reg::UpdateMask<int64_t, Reg::RegTraitNumTwo>(indexLen);
        }
        Reg::Duplicate(outReg, 0, pregLoop);
        // order/index 用 int32 计算 (indices 是 int32, Gather 要求 b32 src + u32 index),
        // Gather 结果 Cast 到 int64 累加
        Reg::Arange(orderReg, i * vfLen);
        Reg::Muls(orderReg, orderReg, rankSize, pregIndex);
        for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
            OffsetType strideValue = static_cast<OffsetType>(strideLocalAddr[dim]);
            indexReg = (IndexRegType&)orderReg;
            if constexpr (IsSameType<U, int32_t>::value && isCastIndices) {
                Reg::RegTensor<int32_t> castReg;
                Reg::Gather(castReg, indicesLocalPtr, indexReg, pregIndex);
                Reg::Cast<int64_t, int32_t, castTraitB322B64>(inReg, castReg, pregLoop);
            } else {
                Reg::Gather(inReg, indicesLocalPtr, indexReg, pregLoop);
            }
            Reg::Muls(inReg, inReg, strideValue, pregLoop);
            Reg::Add(outReg, inReg, outReg, pregLoop);
            Reg::Adds(orderReg, orderReg, (U)(1), pregIndex);
        }
        // 越界索引保护: flat 行偏移(sum(indices*stride))须落在 [0, varInAxis), 越界项改写为 -1,
        // 经 afterAxisBytes 换算成负字节偏移后由 ScatterDataCopy 丢弃该行; 负索引折叠出的负 flat
        // 偏移同样得到负字节偏移被丢弃)
        Reg::Duplicate(invalidFlatReg, static_cast<OffsetType>(-1), pregLoop);
        Reg::Compares<OffsetType, CMPMODE::LT>(validMask, outReg, maxFlatOffset, pregLoop);
        Reg::Select(outReg, outReg, invalidFlatReg, validMask);
        Reg::Muls(tmpReg, outReg, afterAxisBytes, pregLoop);
        Reg::Adds(outReg, tmpReg, colStartBytes, pregLoop);
        Reg::StoreAlign(offsetAddr + i * vfLen, outReg, pregLoop);
    }
}

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
__aicore__ inline void ScatterNdAddDma<T, U, isCastIndices, isSplitCol>::ComputeOffset(
    LocalTensor<U> indicesLocal, LocalTensor<OffsetType> offsetLocal, int32_t indicesLen, int32_t rankSize)
{
    // 一步到位: indices -> flat_row_offset(越界裁剪到 -1) -> (flat_row_offset * afterAxis + colStart)
    // * sizeof(T) -> offsetLocal(字节地址); ScatterDataCopy 的 offset 参数为字节偏移, 故此处直接计算字节地址
    // isCastIndices=true 时用 int64 寄存器; isCastIndices=false 时用 U 类型寄存器(int32 性能更好)
    LocalTensor<U> strideLocal = strideBuf_.Get<U>();
    __ubuf__ U* indicesLocalPtr = (__ubuf__ U*)indicesLocal.GetPhyAddr();
    __ubuf__ OffsetType* offsetAddr = (__ubuf__ OffsetType*)offsetLocal.GetPhyAddr();
    __ubuf__ U* strideLocalAddr = (__ubuf__ U*)strideLocal.GetPhyAddr();
    OffsetType afterAxisBytes = static_cast<OffsetType>(tilingData_.afterAxis * sizeof(T));
    OffsetType colStartBytes = static_cast<OffsetType>(colStart_ * sizeof(T));
    // varInAxis = prod(shape[:rankSize]) 为 flat 行偏移上界, 用于越界索引裁剪;
    // under-uint32 路径 host 已保证 varInAxis*afterAxis*sizeof(T) < INT32_MAX, 可安全转 OffsetType
    OffsetType maxFlatOffset = static_cast<OffsetType>(tilingData_.varInAxis);

    uint32_t dataLen = indicesLen;
    uint32_t indexLen = indicesLen;
    uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
    uint32_t loopCnt = ops::CeilDiv(dataLen, vfLen);

    ComputeOffsetVF<U, OffsetType, OffsetRegType, CalRegType, IndexRegType, isCastIndices>(
        strideLocalAddr, indicesLocalPtr, offsetAddr, afterAxisBytes, colStartBytes, maxFlatOffset, dataLen, indexLen,
        vfLen, loopCnt, rankSize);
}

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
__aicore__ inline void ScatterNdAddDma<T, U, isCastIndices, isSplitCol>::ProcessNoSplitCol(int64_t rowIdx,
                                                                                           int64_t curRow)
{
    int64_t updatesOffset = (rowStart_ + rowIdx * tilingData_.indicesFactor) * tilingData_.afterAxis + colStart_;
    LocalTensor<T> updateLocal = dataQueue_.AllocTensor<T>();
    CopyInUpdate(updateLocal, updatesOffset, curRow, curCoreAfterAxisCount_);
    dataQueue_.EnQue(updateLocal);

    updateLocal = dataQueue_.DeQue<T>();
    LocalTensor<OffsetType> offsetLocal = offsetQue_.DeQue<OffsetType>();

    CopyOutUpdate(updateLocal, offsetLocal, curRow, curCoreAfterAxisCount_);

    dataQueue_.FreeTensor(updateLocal);
    offsetQue_.FreeTensor(offsetLocal);
}

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
__aicore__ inline void ScatterNdAddDma<T, U, isCastIndices, isSplitCol>::ProcessSplitCol(int64_t rowIdx, int64_t curRow)
{
    LocalTensor<OffsetType> tmpOffsetLocal = offsetBuf_.Get<OffsetType>();
    LocalTensor<OffsetType> offsetLocal;
    int64_t colLoop = ops::CeilDiv(curCoreAfterAxisCount_, tilingData_.afterAxisFactor);
    for (int64_t colIdx = 0; colIdx < colLoop; colIdx++) {
        int64_t curCol = (colIdx == colLoop - 1) ? (curCoreAfterAxisCount_ - colIdx * tilingData_.afterAxisFactor) :
                                                   tilingData_.afterAxisFactor;

        int64_t updatesOffset = (rowStart_ + rowIdx * tilingData_.indicesFactor) * tilingData_.afterAxis + colStart_ +
                                colIdx * tilingData_.afterAxisFactor;
        LocalTensor<T> updateLocal = dataQueue_.AllocTensor<T>();
        CopyInUpdate(updateLocal, updatesOffset, curRow, curCol);
        dataQueue_.EnQue(updateLocal);

        offsetLocal = offsetQue_.DeQue<OffsetType>();
        Copy(tmpOffsetLocal, offsetLocal, curRow);

        updateLocal = dataQueue_.DeQue<T>();

        CopyOutUpdate(updateLocal, offsetLocal, curRow, curCol);

        dataQueue_.FreeTensor(updateLocal);
        offsetQue_.FreeTensor(offsetLocal);

        if (colIdx < colLoop - 1) {
            offsetLocal = offsetQue_.AllocTensor<OffsetType>();
            Adds(offsetLocal, tmpOffsetLocal, static_cast<OffsetType>(tilingData_.afterAxisFactor * sizeof(T)), curRow);
            offsetQue_.EnQue(offsetLocal);
        }
    }
}

template <typename T, typename U, bool isCastIndices, bool isSplitCol>
__aicore__ inline void ScatterNdAddDma<T, U, isCastIndices, isSplitCol>::Process()
{
    if (GetBlockIdx() >= tilingData_.blockNum) {
        return;
    }

    LocalTensor<U> strideLocal = strideBuf_.Get<U>();
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideLocal(i) = static_cast<U>(tilingData_.strideList[i]);
    }

    int64_t rowLoop = ops::CeilDiv(curCoreIndexCount_, tilingData_.indicesFactor);

    for (int64_t rowIdx = 0; rowIdx < rowLoop; rowIdx++) {
        int64_t curRow = (rowIdx == rowLoop - 1) ? (curCoreIndexCount_ - rowIdx * tilingData_.indicesFactor) :
                                                   tilingData_.indicesFactor;

        LocalTensor<U> indicesLocal = indicesQue_.AllocTensor<U>();
        int64_t indicesOffset = (rowStart_ + rowIdx * tilingData_.indicesFactor) * tilingData_.rankSize;
        DataCopyExtParams indicesCopyParams = {
            static_cast<uint16_t>(1), static_cast<uint32_t>(curRow * tilingData_.rankSize * sizeof(U)),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<U> indicesPadParams = {false, 0, 0, 0};
        DataCopyPad(indicesLocal, indicesGm_[indicesOffset], indicesCopyParams, indicesPadParams);
        indicesQue_.EnQue(indicesLocal);

        indicesLocal = indicesQue_.DeQue<U>();

        LocalTensor<OffsetType> offsetLocal = offsetQue_.AllocTensor<OffsetType>();
        ComputeOffset(indicesLocal, offsetLocal, static_cast<int32_t>(curRow),
                      static_cast<int32_t>(tilingData_.rankSize));
        offsetQue_.EnQue(offsetLocal);

        if constexpr (isSplitCol) {
            ProcessSplitCol(rowIdx, curRow);
        } else {
            ProcessNoSplitCol(rowIdx, curRow);
        }
        indicesQue_.FreeTensor(indicesLocal);
    }
}

} // namespace ScatterNdAdd

#endif // SCATTER_ND_ADD_SIMD_DMA_H_
