/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_v2_simd.h
 * \brief
 */

#ifndef INDEX_PUT_V2_SIMD
#define INDEX_PUT_V2_SIMD

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "kernel_operator_list_tensor_intf.h"

namespace IndexPutV2 {
using namespace AscendC;

constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint64_t ubBlockSize = 32;
constexpr uint32_t MAX_DIM_NUM = 8;


template<typename TX, typename TIDX, bool IsAtomicAdd>
class IndexPutV2Simd {
public:
    __aicore__ inline IndexPutV2Simd(){};
    __aicore__ inline void Init(
        GM_ADDR inputX, GM_ADDR value, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
        IndexPutV2SimdTilingData tilingData_);
    __aicore__ inline void Process();
    __aicore__ inline void CopyOut(LocalTensor<TX> valueLocal, int64_t ubRowOffset, int64_t inputGmOffset, int64_t curCols);
    __aicore__ inline void CopyInIndices(LocalTensor<TIDX> xLocal, GlobalTensor<TIDX> xGm, int64_t gmOffset, int64_t localOffset, uint32_t copyCount, uint32_t copyLen);
    __aicore__ inline void HandleIndices(int64_t curRows);
    __aicore__ inline void CopyInValue(int64_t copyCount, int64_t copyLen, int64_t blockOffset);
    __aicore__ inline void GetIndices(int64_t curRows);

private:
    TPipe pipe_;
    GlobalTensor<TX> inputGM_;
    GlobalTensor<TIDX> indicesGM_;
    GlobalTensor<TX> valueGM_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQue_;    // 一次存储rowsFactor组索引
    TQueBind<TPosition::VECIN, TPosition::VECOUT, DOUBLE_BUFFER> valueQue_;   // 一次存储rowsFactor * cowsFactor个元素
    TBuf<QuePosition::VECCALC> linearIndexBuf_;          // 将indicesQue通过HandleIndices转换为一维索引
    TBuf<QuePosition::VECCALC> inputShapeBuf_;
    TBuf<QuePosition::VECCALC> indexedStridesBuf_;

    IndexPutV2SimdTilingData tilingData_;

    uint32_t needCoreNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t rowCoreIdx_ = 0;
    uint32_t colCoreIdx_ = 0;

    uint64_t rowGmOffset_ = 0;
    uint64_t colGmOffset_ = 0;
    uint64_t blockOffset_ = 0;
    uint64_t rowsUbLoop = 0;
    uint64_t colsUbLoop = 0;
    uint64_t indicesOffsetBase_ = 0;
    uint64_t rowsFactor_ = 0;
    uint64_t colsFactor_ = 0;
    int64_t actualRowsNum_ = 0;
    int64_t actualColsNum_ = 0;
    int64_t invalidIndex = -1;
    ListTensorDesc indicesList_;

};

template<typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::Init(
    GM_ADDR inputX, GM_ADDR value, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
    IndexPutV2SimdTilingData tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = tilingData_.coreNum;
    indicesList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(indices));

    if (blockIdx_ >= needCoreNum_) {
        return;
    }

    rowCoreIdx_ = blockIdx_ / tilingData_.blockNumInCol;
    colCoreIdx_ = blockIdx_ % tilingData_.blockNumInCol;

    rowGmOffset_ = rowCoreIdx_ * tilingData_.normalCoreRowsNum;
    colGmOffset_ = colCoreIdx_ * tilingData_.normalCoreColsNum;

    blockOffset_ = rowGmOffset_ * tilingData_.nonIndexedLength + colGmOffset_;

    rowsFactor_ = tilingData_.rowsFactor;
    colsFactor_ = tilingData_.colsFactor;

    actualRowsNum_ = (rowCoreIdx_ == tilingData_.blockNumInRow - 1)
        ? tilingData_.tailCoreRowsNum : tilingData_.normalCoreRowsNum;
    actualColsNum_ = (colCoreIdx_ == tilingData_.blockNumInCol - 1)
        ? tilingData_.tailCoreColsNum : tilingData_.normalCoreColsNum;

    rowsUbLoop = Ops::Base::CeilDiv(actualRowsNum_, tilingData_.rowsFactor);
    colsUbLoop = Ops::Base::CeilDiv(actualColsNum_, tilingData_.colsFactor);

    inputGM_.SetGlobalBuffer((__gm__ TX*)inputX);
    valueGM_.SetGlobalBuffer((__gm__ TX*)value);

    indicesOffsetBase_ = rowGmOffset_;

    // indicesQue_: indexedDimNum 个维度，每维 rowsFactor 个索引
    uint64_t valueBuf = tilingData_.rowsFactor * Ops::Base::CeilAlign(tilingData_.colsFactor * sizeof(TX), ubBlockSize);
    uint64_t indicesBuf = tilingData_.indexedDimNum * Ops::Base::CeilAlign(tilingData_.rowsFactor * sizeof(TIDX), ubBlockSize);
    uint64_t linearIndexBuf = Ops::Base::CeilAlign(tilingData_.rowsFactor * sizeof(TIDX), ubBlockSize);
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, indicesBuf);
    pipe_.InitBuffer(valueQue_, DOUBLE_BUFFER, valueBuf);
    pipe_.InitBuffer(linearIndexBuf_, linearIndexBuf);
    pipe_.InitBuffer(inputShapeBuf_, MAX_DIM_NUM * sizeof(int64_t));
    pipe_.InitBuffer(indexedStridesBuf_, MAX_DIM_NUM * sizeof(int64_t));

    LocalTensor<int64_t> inputShapeTensor = inputShapeBuf_.Get<int64_t>();
    LocalTensor<int64_t> indexedStridesTensor = indexedStridesBuf_.Get<int64_t>();
    for (uint16_t i = 0; i < MAX_DIM_NUM; i++) {
        inputShapeTensor.SetValue(i, tilingData_.inputShapes[i]);
        indexedStridesTensor.SetValue(i, tilingData_.indexedStrides[i]);
    }
}

// 索引搬入：将 indexedDimNum 个维度的索引搬入同一块 UB buffer
template<typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::GetIndices(int64_t curRows)
{
    LocalTensor<TIDX> indicesLocal = indicesQue_.AllocTensor<TIDX>();
    for (uint32_t i = 0; i < tilingData_.indexedDimNum; i++) {
        indicesGM_.SetGlobalBuffer(indicesList_.GetDataPtr<TIDX>(i));
        int64_t localOffset = i * Ops::Base::CeilAlign(rowsFactor_ * sizeof(TIDX), ubBlockSize) / sizeof(TIDX);
        CopyInIndices(indicesLocal, indicesGM_, indicesOffsetBase_, localOffset, 1, curRows);
    }
    indicesQue_.EnQue(indicesLocal);
    indicesOffsetBase_ += curRows;
}

// 将 UB 中一行 value 数据搬出到 inputGM_ 的目标位置
template<typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::CopyOut(
    LocalTensor<TX> valueLocal, int64_t ubRowOffset, int64_t inputGmOffset, int64_t curCols)
{
    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = 1;
    dataCopyExtParams.blockLen = curCols * sizeof(TX);
    dataCopyExtParams.srcStride = 0;
    dataCopyExtParams.dstStride = 0;
    DataCopyPad(inputGM_[inputGmOffset], valueLocal[ubRowOffset], dataCopyExtParams);
}

// 将一个维度的索引从 GM 搬入 UB 的指定偏移位置
template<typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::CopyInIndices(
    LocalTensor<TIDX> xLocal, GlobalTensor<TIDX> xGm,
    int64_t gmOffset, int64_t localOffset, uint32_t copyCount, uint32_t copyLen)
{
    DataCopyPadExtParams<TIDX> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = copyCount;
    dataCoptExtParams.blockLen = copyLen * sizeof(TIDX);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal[localOffset], xGm[gmOffset], dataCoptExtParams, dataCopyPadExtParams);
}

template<typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::CopyInValue(int64_t copyCount, int64_t copyLen, int64_t gmOffset)
{
    int64_t gmStride = tilingData_.nonIndexedLength - copyLen;
    LocalTensor<TX> valueLocal = valueQue_.AllocTensor<TX>();
    DataCopyPadExtParams<TX> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = copyCount;
    dataCoptExtParams.blockLen = copyLen * sizeof(TX);
    dataCoptExtParams.srcStride = gmStride * sizeof(TX);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(valueLocal, valueGM_[gmOffset], dataCoptExtParams, dataCopyPadExtParams);
    valueQue_.EnQue(valueLocal);
}

template<typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::Process()
{
    if (blockIdx_ >= tilingData_.coreNum) {
        return;
    }

    for (uint16_t rowId = 0; rowId < rowsUbLoop; rowId++) {
        int64_t curRows = (rowId == rowsUbLoop - 1)
            ? (actualRowsNum_ - rowId * rowsFactor_) : static_cast<int64_t>(rowsFactor_);

        GetIndices(curRows);
        HandleIndices(curRows);

        LocalTensor<TIDX> tmpLocal = linearIndexBuf_.Get<TIDX>();
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);

        for (uint16_t colId = 0; colId < colsUbLoop; colId++) {
            int64_t curCols = (colId == colsUbLoop - 1)
                ? (actualColsNum_ - colId * colsFactor_) : static_cast<int64_t>(colsFactor_);

            // value GM 偏移 = 核起始偏移 + 行迭代推进 + 列迭代推进
            int64_t valueGmOffset = blockOffset_
                + rowId * rowsFactor_ * tilingData_.nonIndexedLength
                + colId * colsFactor_;

            CopyInValue(curRows, curCols, valueGmOffset);
            PipeBarrier<PIPE_MTE2>();
            LocalTensor<TX> valueLocal = valueQue_.DeQue<TX>();

            for (int64_t i = 0; i < curRows; i++) {
                int64_t linearIdx = static_cast<int64_t>(tmpLocal.GetValue(i));
                if (linearIdx == invalidIndex) {
                    continue; 
                }
                // inputGM 目标偏移 = 线性索引（索引合一结果） + 列偏移
                int64_t inputGmTarget = linearIdx + colGmOffset_ + colId * colsFactor_;
                int64_t ubRowOffset = i * Ops::Base::CeilAlign(curCols * sizeof(TX), ubBlockSize) / sizeof(TX);
                if constexpr(IsAtomicAdd) {
                    SetAtomicAdd<TX>();
                    CopyOut(valueLocal, ubRowOffset, inputGmTarget, curCols);
                    SetAtomicNone();
                } else {
                    CopyOut(valueLocal, ubRowOffset, inputGmTarget, curCols);
                }
            }

            valueQue_.FreeTensor(valueLocal);
        }
    }
}

//负索引转正、越界标记（-1）、多维合一（linear = sum(idx[k] * stride[k]))
template<typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::HandleIndices(int64_t curRows)
{
    uint16_t oneRepeatNum = GetVecLen() / sizeof(TIDX);
    uint16_t repeatTimes = Ops::Base::CeilDiv(curRows, static_cast<int64_t>(oneRepeatNum));

    LocalTensor<TIDX> indicesLocal = indicesQue_.DeQue<TIDX>();
    __local_mem__ TIDX *indicesAddr = (__local_mem__ TIDX *)indicesLocal.GetPhyAddr();
    LocalTensor<TIDX> linearIndex = linearIndexBuf_.Get<TIDX>();
    __local_mem__ TIDX * linearIndexAddr = (__local_mem__ TIDX *)linearIndex.GetPhyAddr();

    LocalTensor<int64_t> inputShapesTensor = inputShapeBuf_.Get<int64_t>();
    LocalTensor<int64_t> indexedStridesTensor = indexedStridesBuf_.Get<int64_t>();
    uint16_t indexedDimNum = static_cast<uint16_t>(tilingData_.indexedDimNum);
    uint16_t idxDtypeSize = sizeof(TIDX);
    uint16_t rowsFactorU16 = Ops::Base::CeilAlign(rowsFactor_ * idxDtypeSize, ubBlockSize) / idxDtypeSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<TIDX> idxReg;          // 当前维度的索引值
        MicroAPI::RegTensor<TIDX> idxFusionReg;    // 累加的线性索引
        MicroAPI::RegTensor<TIDX> zeroReg;          // 零值寄存器，用于负索引比较
        MicroAPI::RegTensor<TIDX> selectReg;
        MicroAPI::RegTensor<TIDX> dimSizeReg;
        uint32_t curRepeatNum = curRows;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<TIDX>(curRepeatNum);

            MicroAPI::Duplicate(idxFusionReg, static_cast<TIDX>(0), maskReg);
            // 初始化越界融合掩码为全 0（无越界），后续用 OR 累积
            MicroAPI::MaskReg crossFusionMask = MicroAPI::CreateMask<TIDX, MicroAPI::MaskPattern::ALLF>();
            MicroAPI::Duplicate(zeroReg, static_cast<TIDX>(0), maskReg);

            for (uint16_t k = 0; k < indexedDimNum; k++) {
                TIDX inputShape = static_cast<TIDX>(inputShapesTensor.GetValue(k));
                TIDX indexedStride = static_cast<TIDX>(indexedStridesTensor.GetValue(k));

                // 偏移 = k * rowsFactor_ + i * oneRepeatNum（元素单位 → 字节偏移）
                auto indicesAddrUpdate = indicesAddr + (k * rowsFactorU16 + i * oneRepeatNum);
                MicroAPI::DataCopy(idxReg, indicesAddrUpdate);

                // ---- 1. 负索引转正：idx < 0 → idx += inputShape[k] ----
                MicroAPI::MaskReg negMask;
                MicroAPI::Compare<TIDX, CMPMODE::LT>(negMask, idxReg, zeroReg, maskReg);
                MicroAPI::Duplicate(selectReg, inputShape, maskReg);
                MicroAPI::Select(selectReg, selectReg, zeroReg, negMask);
                MicroAPI::Add(idxReg, idxReg, selectReg, maskReg);

                // ---- 2. 越界检查：idx >= inputShape[k] → 标记为越界 ----
                // 0-based 索引，有效范围 [0, inputShape[k]-1]，等于 inputShape[k] 时也越界
                MicroAPI::MaskReg crossMask;
                MicroAPI::Duplicate(dimSizeReg, inputShape, maskReg);
                MicroAPI::Compare<TIDX, CMPMODE::GE>(crossMask, idxReg, dimSizeReg, maskReg);
                MicroAPI::MaskOr(crossFusionMask, crossFusionMask, crossMask, maskReg);
                MicroAPI::Compare<TIDX, CMPMODE::LT>(crossMask, idxReg, zeroReg, maskReg);
                MicroAPI::MaskOr(crossFusionMask, crossFusionMask, crossMask, maskReg);

                // ---- 3. 索引合一：idxFusion += idx * stride[k] ----
                MicroAPI::Muls(idxReg, idxReg, indexedStride, maskReg);
                MicroAPI::Add(idxFusionReg, idxFusionReg, idxReg, maskReg);
            }

            // ---- 4. 所有维度处理完后，将越界位置标记为 -1 ----
            MicroAPI::Duplicate(selectReg, static_cast<TIDX>(-1), maskReg);
            MicroAPI::Select(idxFusionReg, selectReg, idxFusionReg, crossFusionMask);
            auto linearIndexAddrUpdate = linearIndexAddr + i * oneRepeatNum;
            MicroAPI::DataCopy(linearIndexAddrUpdate, idxFusionReg, maskReg);
        }
    }

    indicesQue_.FreeTensor(indicesLocal);
}

} // namespace IndexPutV2Simd

#endif // INDEX_PUT_V2_SIMD
