/**
 * Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_tail_axis_nvfp4.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_TAIL_AXIS_NVFP4_H
#define DYNAMIC_MX_QUANT_TAIL_AXIS_NVFP4_H
#include "kernel_operator.h"
#include "dynamic_mx_quant_common.h"
namespace DynamicMxQuant {
using namespace AscendC;

constexpr int64_t NVFP4_UB_SLICE_SIZE = 256; // 尾轴 tiling 标准切片的元素数量

template <typename T, typename U>
class DynamicMxQuantTailAxisNvfp4 {
public:
    __aicore__ inline DynamicMxQuantTailAxisNvfp4() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void GetGmParams();
    __aicore__ inline void GetUbParams();

    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                  int64_t ubFactorColNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum, int64_t ubFactorColNum);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                   int64_t ubFactorColNum, int64_t ubFactorColBlockNum);

    // SCALE_ALG==3: NVFP4实现，量化块固定为 block16，scale 为 FP8 E4M3FN
    __aicore__ inline void ComputeMaxExpNvfp4(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                              uint16_t loopNum2VF);
    template <bool colNumIs16>
    __aicore__ inline void ComputeScaleNvfp4(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint8_t* mxScaleLocalAddr,
                                             __ubuf__ float* recipScaleLocalAddr, uint16_t loopNumHalfVF,
                                             float maxLowBound);
    // Compact padded scale rows and explicitly zero the unused member of the final scale pair.
    __aicore__ inline void ProcessScale(__ubuf__ uint8_t* mxScaleLocalAddr, uint16_t srcRowWidth,
                                        uint16_t validRowWidth, uint32_t count);
    // 奇数块行的 scale 布局修正：行宽 srcRowWidth（每行 srcRowWidth 个有效 scale），原地展宽为
    // srcRowWidth + 1（行尾插 1 个 0）；count 为目的侧总字节数（行数 × (srcRowWidth + 1)），需 < 65536
    __aicore__ inline void InsertZeroScale(__ubuf__ uint8_t* mxScaleLocalAddr, uint16_t srcRowWidth, uint32_t count);
    template <Reg::RegLayout layout = Reg::RegLayout::ZERO>
    __aicore__ inline void ComputeDataToBf16(Reg::RegTensor<bfloat16_t>& xBF16, Reg::RegTensor<float>& xFP32,
                                             Reg::RegTensor<float>& invScaleFP32, Reg::MaskReg& dataMaskB32);
    template <RoundMode roundMode>
    __aicore__ inline void ComputeDataNvfp4(__ubuf__ T* xLocalAddr, __ubuf__ float* recipScaleLocalAddr,
                                            __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF);
    template <bool colNumIs16>
    __aicore__ inline void ComputeNvfp4(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr,
                                        __ubuf__ uint8_t* mxScaleLocalAddr, __ubuf__ int8_t* yLocalAddr,
                                        __ubuf__ float* recipScaleLocalAddr, uint16_t loopNum2VF,
                                        uint16_t loopNumHalfVF);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<uint8_t> yGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    GlobalTensor<uint8_t> scaleGm_;

    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> recipScaleBuffer_;

    int64_t roundMode_ = 0;
    int64_t blockSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;

    int64_t rowTileNum_ = 0;        // row 方向上的切核数
    int64_t colTileNum_ = 0;        // col 方向上的切核数
    int64_t rowNum_ = 1;            // 合轴之后 -2 轴大小
    int64_t colNum_ = 1;            // 合轴之后 -1 轴大小
    int64_t colNormalBlockNum_ = 0; // 列方向头核处理的块数 (1 x 256)
    int64_t colTailLen_ = 0;        // 列方向尾块长度
    int64_t rowNormalBlockNum_ = 0; // 行方向头核处理的块数 (1 行)
    int64_t rowTailLen_ = 0;        // 行方向尾块长度
    int64_t maxUbBlockNum_ = 0;     // UB最大能放下的处理块数 (blockSize 的倍数)
    float maxLowBound_ = 0.0;

    // GM Params
    int64_t coreIdx_ = 0;       // Core ID
    int64_t coreColIdx_ = 0;    // Core 处理的 GM 数据块列方向的核ID数
    int64_t coreRowIdx_ = 0;    // Core 处理的 GM 数据块行方向的核ID数
    int64_t xGmOffset_ = 0;     // Core 处理的数据块在 GM 的起始地址偏移
    int64_t scaleGmOffset_ = 0; // Core 处理的数据块得到的Scale在 GM 的地址偏移
    int64_t scaleColNum_ = 0;   // 合轴后数据块在尾轴方向上的 scale 数量大小（偶数对齐）

    // UB Params
    int64_t ubFactorColNum_ = 0;            // Core 处理的 GM 数据块列方向元素个数
    int64_t ubFactorColBlockNum_ = 0;       // Core 处理的 GM 数据块列方向 block 个数
    int64_t ubFactorRow1BlockNum_ = 0;      // Core 处理的 GM 数据块行方向 1 长度块个数
    int64_t ubFactorColLoopNum_ = 0;        // Core 处理的 GM 数据块列方向循环次数
    int64_t ubFactorRowLoopNum_ = 0;        // Core 处理的 GM 数据块行方向循环次数
    int64_t ubFactorColNormalBlockNum_ = 0; // Core 处理的 GM 数据块搬入 UB 列方向 Normal 数据块
    int64_t ubFactorColTailBlockNum_ = 0;   // Core 处理的 GM 数据块搬入 UB 列方向 Tail 数据块
    int64_t ubFactorColNormalBlockLen_ = 0; // Core 处理的 GM 数据块搬入 UB 列方向 Normal 数据块元素个数
    int64_t ubFactorColTailBlockLen_ = 0;   // Core 处理的 GM 数据块搬入 UB 列方向 Tail 数据块元素个数
    int64_t ubFactorRowNormalBlockNum_ = 0; // Core 处理的 GM 数据块搬入 UB 行方向 Normal 1 长度数据块
    int64_t ubFactorRowTailBlockNum_ = 0;   // Core 处理的 GM 数据块搬入 UB 行方向 Tail 1 长度数据块
};

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                                               const DynamicMxQuantTailAxisTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData); // 获取TilingData数据

    GetGmParams(); // 计算核间GM地址偏移

    GetUbParams(); // 计算核内切分参数

    // NVFP4 的倒数为 FP32（保边界乘法需 FP32 精度），recipScaleBuffer_ 按 4 字节/block 分配
    // maxUbBlockNum_ * blockSize_ 是 256 对齐的
    int64_t scaleBufferBlockNum = Ops::Base::CeilAlign(maxUbBlockNum_, static_cast<int64_t>(VF_LEN_16));
    pipe_.InitBuffer(inQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(T));
    pipe_.InitBuffer(outQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(uint8_t) / DIGIT_TWO);
    pipe_.InitBuffer(mxScaleQueue_, DB_BUFFER, scaleBufferBlockNum * sizeof(uint8_t));
    pipe_.InitBuffer(maxExpBuffer_, maxUbBlockNum_ * sizeof(T) * DB_BUFFER);
    pipe_.InitBuffer(recipScaleBuffer_, maxUbBlockNum_ * sizeof(float) * DB_BUFFER);

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset_ / DIGIT_TWO);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleGmOffset_);
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::GetGmParams()
{
    coreIdx_ = GetBlockIdx();
    coreColIdx_ = coreIdx_ % colTileNum_;
    coreRowIdx_ = coreIdx_ / colTileNum_;
    // colNormalBlockNum_ 以 256 元素为单位：block16 时每单位 16 个 block
    xGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * colNum_ +
                 coreColIdx_ * colNormalBlockNum_ * NVFP4_UB_SLICE_SIZE;
    scaleColNum_ = ops::CeilDiv(ops::CeilDiv(colNum_, blockSize_), DIGIT_TWO) * DIGIT_TWO;
    scaleGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * scaleColNum_ +
                     coreColIdx_ * colNormalBlockNum_ * (NVFP4_UB_SLICE_SIZE / blockSize_);
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::GetUbParams()
{
    if (coreColIdx_ == colTileNum_ - 1) {
        ubFactorColBlockNum_ = ops::CeilDiv(colTailLen_, blockSize_);
        ubFactorColNum_ = colTailLen_;
    } else {
        ubFactorColBlockNum_ = colNormalBlockNum_ * (NVFP4_UB_SLICE_SIZE / blockSize_);
        ubFactorColNum_ = ubFactorColBlockNum_ * blockSize_;
    }

    if (coreRowIdx_ == rowTileNum_ - 1) {
        ubFactorRow1BlockNum_ = ops::CeilDiv(rowTailLen_, DIGIT_ONE);
    } else {
        ubFactorRow1BlockNum_ = rowNormalBlockNum_ * DIGIT_ONE;
    }

    ubFactorColLoopNum_ = ops::CeilDiv(ubFactorColBlockNum_,
                                       maxUbBlockNum_); // 列方向上UB循环次数（列方向上的切 UB 块数）
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorColBlockNum_,
                                              ubFactorColLoopNum_); // 列方向上头块的Block数量（均衡处理） 用于VF内处理
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorColNormalBlockNum_, DIGIT_FOUR) *
                                 DIGIT_FOUR; // 列方向上头块的Block数量补成偶数，只有尾块有可能需要补Pad
    ubFactorColTailBlockNum_ = ubFactorColBlockNum_ -
                               (ubFactorColLoopNum_ - DIGIT_ONE) *
                                   ubFactorColNormalBlockNum_; // 列方向上尾块的Block数量（只有一个尾块） 用于VF内处理
    ubFactorColNormalBlockLen_ = ubFactorColNormalBlockNum_ *
                                 blockSize_; // 列方向上头块的元素数量（均衡处理）           用于数据搬运
    ubFactorColTailBlockLen_ = ubFactorColNum_ -
                               (ubFactorColLoopNum_ - DIGIT_ONE) *
                                   ubFactorColNormalBlockLen_; // 列方向上尾块的元素数量（只有一个尾块） 用于数据搬运

    ubFactorRowNormalBlockNum_ = ops::FloorDiv(maxUbBlockNum_, ubFactorColNormalBlockNum_); // UB 一次至多载入的行数
    ubFactorRowLoopNum_ = ops::CeilDiv(ubFactorRow1BlockNum_,
                                       ubFactorRowNormalBlockNum_); // 行方向上 UB 循环次数（行方向上的切 UB 块数）
    ubFactorRowNormalBlockNum_ = ops::CeilDiv(ubFactorRow1BlockNum_,
                                              ubFactorRowLoopNum_); // 行方向上的均衡处理 用于VF内处理
    ubFactorRowTailBlockNum_ = ubFactorRow1BlockNum_ -
                               (ubFactorRowLoopNum_ - DIGIT_ONE) *
                                   ubFactorRowNormalBlockNum_; // 行方向上尾块的行数（只有一个尾块） 用于VF内处理
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::ParseTilingData(
    const DynamicMxQuantTailAxisTilingData* tilingData)
{
    roundMode_ = tilingData->roundMode;
    blockSize_ = tilingData->blockSize;
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    rowTileNum_ = tilingData->rowTileNum;
    colTileNum_ = tilingData->colTileNum;
    rowNum_ = tilingData->rowNum;
    colNum_ = tilingData->colNum;
    colNormalBlockNum_ = tilingData->colNormalBlockNum;
    colTailLen_ = tilingData->colTailLen;
    rowNormalBlockNum_ = tilingData->rowNormalBlockNum;
    rowTailLen_ = tilingData->rowTailLen;
    maxUbBlockNum_ = tilingData->maxUbBlockNum;
    maxLowBound_ = tilingData->maxLowBound;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::Process()
{
    if (coreIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t dim0LoopIdx = 0;
    int64_t dim1LoopIdx = 0;
    for (dim0LoopIdx = 0; dim0LoopIdx < ubFactorRowLoopNum_ - 1; dim0LoopIdx++) {
        for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
            CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_,
                   ubFactorColNormalBlockNum_);
            Compute(ubFactorRowNormalBlockNum_, ubFactorColNormalBlockNum_, ubFactorColNormalBlockLen_);
            CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_,
                    ubFactorColNormalBlockNum_);
        }
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_,
               ubFactorColTailBlockNum_);
        Compute(ubFactorRowNormalBlockNum_, ubFactorColTailBlockNum_, ubFactorColTailBlockLen_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_,
                ubFactorColTailBlockNum_);
    }
    for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_,
               ubFactorColNormalBlockNum_);
        Compute(ubFactorRowTailBlockNum_, ubFactorColNormalBlockNum_, ubFactorColNormalBlockLen_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_,
                ubFactorColNormalBlockNum_);
    }
    CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    Compute(ubFactorRowTailBlockNum_, ubFactorColTailBlockNum_, ubFactorColTailBlockLen_);
    CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                                 int64_t ubFactorRowNum, int64_t ubFactorColNum,
                                                                 int64_t ubFactorColBlockNum)
{
    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();

    DataCopyExtParams copyInParam;
    DataCopyPadExtParams<T> padParams;

    // 当补的数(rightPad)是16时，一定能补到UB的下一个DataBlock，实现补到32个数对齐的功能（pad 上限 16，
    // 不足 64 元素对齐的部分由 dstStride 挖洞补足；无效块的 scale 由 ProcessScale 丢弃或置零）
    int64_t padRightNumTmp = (ubFactorColNum % 32 > 16) ? (32 - ubFactorColNum % 32) : 16;
    int64_t padRightNum = (ubFactorColNum % 32 == 0) ? 0 : padRightNumTmp;
    int64_t dstStride = (ubFactorColNum % 64 < 32 && ubFactorColNum % 64 > 0) ? 2 : 0;
    if (ubFactorColNum % blockSize_ == DIGIT_ZERO) { // colNum 为块大小整数倍：blockLen 本身 32B 对齐，无需 pad/挖洞
        dstStride = 0;
        padRightNum = 0;
    }
    copyInParam = {static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(T)),
                   static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(T)), static_cast<uint32_t>(dstStride),
                   static_cast<uint32_t>(0)};
    // Padding must not turn an all-NaN partial block into a zero block.
    union {
        uint16_t bits;
        T value;
    } nan;
    nan.bits = 0x7fff;
    padParams = {true, static_cast<uint8_t>(0), static_cast<uint8_t>(padRightNum), nan.value};

    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;

    DataCopyPad(xLocal, xGm_[offset], copyInParam, padParams);
    inQueue_.EnQue(xLocal);
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                                  int64_t ubFactorRowNum, int64_t ubFactorColNum,
                                                                  int64_t ubFactorColBlockNum)
{
    LocalTensor<uint8_t> scaleLocal = mxScaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    int64_t yOffset = (dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_) /
                      DIGIT_TWO;

    DataCopyExtParams copyOutParamY = {static_cast<uint16_t>(ubFactorRowNum),
                                       static_cast<uint32_t>(ubFactorColNum / DIGIT_TWO), static_cast<uint32_t>(0),
                                       static_cast<uint32_t>((colNum_ - ubFactorColNum) / DIGIT_TWO),
                                       static_cast<uint32_t>(0)};
    if (ubFactorColNum % blockSize_ == DIGIT_ZERO) {
        // 精确块数路径：y 行流紧凑连续（行距 colNum/2），需按 Compact 搬出
        DataCopyPad<uint8_t, PaddingMode::Compact>(yGm_[yOffset], yLocal, copyOutParamY);
    } else {
        // 旧路径（pad/挖洞）：y 行距 = footprint×8 = align32(colNum/2)，32B 对齐，按 Normal 搬出
        DataCopyPad<uint8_t>(yGm_[yOffset], yLocal, copyOutParamY);
    }

    DataCopyExtParams copyOutParamScale = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColBlockNum + scaleIsNotOdd),
        static_cast<uint32_t>(0), static_cast<uint32_t>(scaleColNum_ - (ubFactorColBlockNum + scaleIsNotOdd)),
        static_cast<uint32_t>(0)};

    int64_t scaleOffset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * scaleColNum_ +
                          dim1LoopIdx * ubFactorColNormalBlockNum_;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[scaleOffset], scaleLocal, copyOutParamScale);

    outQueue_.FreeTensor(yLocal);
    mxScaleQueue_.FreeTensor(scaleLocal);
}

// ========================================== Compute MaxExp =====================================================
// 每个完整向量产生 8 个 block16 归约结果
template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::ComputeMaxExpNvfp4(__ubuf__ T* xLocalAddr,
                                                                             __ubuf__ uint16_t* maxExpAddr,
                                                                             uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        // fp16 与 bf16 的 Inf/NaN 屏蔽阈值不同
        constexpr uint16_t infBits = IsSame<T, half>::value ? FP16_INF : BF16_MAX_EXP;
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<uint16_t> absBits;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, BF16_ABS_MASK);
        Reg::UnalignRegForStore ureg;
        Reg::MaskReg finiteOrInfMask;
        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(xSrc, xLocalAddr, VF_LEN_16);
            Reg::And(absBits, (Reg::RegTensor<uint16_t>&)xSrc, absMask16Bit, Mask);
            Reg::Compares<uint16_t, CMPMODE::LE>(finiteOrInfMask, absBits, infBits,
                                                 Mask); // Filter out non-NaN values before doing ReduceMax.
            Reg::ReduceDataBlock<ReduceType::MAX>((Reg::RegTensor<int16_t>&)xMaxExp, (Reg::RegTensor<int16_t>&)absBits,
                                                  finiteOrInfMask);
            // DataBlocks with all NaN values do not participate in ReduceMax and are written as the minimum value of
            // the corresponding data type For int16, the minimum is 0x8000, and the corresponding T data type parses
            // out as -0.0.
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

// ========================================== Compute Scale ======================================================
// 全空符号 MAX 组归约结果为 0x8000，经 Cast 后是 -0 标记。每个 FP32 lane 计算一个 block 的 scale 并写出一个 FP32 倒数
template <typename T, typename U>
template <bool colNumIs16>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::ComputeScaleNvfp4(__ubuf__ uint16_t* maxExpAddr,
                                                                            __ubuf__ uint8_t* mxScaleLocalAddr,
                                                                            __ubuf__ float* recipScaleLocalAddr,
                                                                            uint16_t loopNumHalfVF, float maxLowBound)
{
    __VEC_SCOPE__
    {
        static constexpr Reg::CastTrait castTraitB16ToFp32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32ToFp8 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        Reg::RegTensor<T> xMaxExp;
        Reg::RegTensor<float> amaxFP32;
        Reg::RegTensor<float> oneFP32;
        Reg::RegTensor<float> lowBoundFP32;
        Reg::RegTensor<float> usedFP32;
        Reg::RegTensor<float> scaleFP32;
        Reg::RegTensor<float> largestFP32;
        Reg::RegTensor<float> backFP32;
        Reg::RegTensor<float> invFP32;
        Reg::RegTensor<uint32_t> fp8NanU32;
        Reg::RegTensor<fp8_e4m3fn_t> scaleFP8;
        Reg::RegTensor<uint8_t> scalePackFP8;
        Reg::MaskReg storeMaskB8Half = Reg::CreateMask<uint8_t, Reg::MaskPattern::H>();

        Reg::Duplicate(oneFP32, 1.0f);
        Reg::Duplicate(lowBoundFP32, maxLowBound);
        Reg::Duplicate(fp8NanU32, uint32_t(0x0000007f)); // FP8 规范 NaN
        Reg::Duplicate(largestFP32, 448.0f);             // FP8 E4M3FN 最大值
        __ubuf__ T* maxPtr = (__ubuf__ T*)maxExpAddr;
        Reg::MaskReg nonZeroMask;
        Reg::MaskReg allNanMask;
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg storeMaskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < loopNumHalfVF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(xMaxExp, maxPtr,
                                                                                                  VF_LEN_32);
            Reg::Cast<float, T, castTraitB16ToFp32>(amaxFP32, xMaxExp, dataMaskB32);
            // The maximum value of the full Nan block was cast from 0x8000 to 0x80000000.
            Reg::Compares<uint32_t, CMPMODE::NE>(nonZeroMask, (Reg::RegTensor<uint32_t>&)amaxFP32, uint32_t(0),
                                                 dataMaskB32);
            // If the maximum value is 0, the corresponding scale and reciprocal should also be 0.0.
            Reg::Compares<uint32_t, CMPMODE::EQ>(allNanMask, (Reg::RegTensor<uint32_t>&)amaxFP32, uint32_t(0x80000000),
                                                 dataMaskB32);
            // Filter out the blocks that are all NaN and handle their maximum values specially.
            Reg::Duplicate(scaleFP32, 0.0f);
            // NVFP4 乘以 6 的 FP32 倒数（0x1.555556p-3）；maxLowBound 扩展在同一乘法前生效
            Reg::Max(usedFP32, amaxFP32, lowBoundFP32, dataMaskB32);
            Reg::Muls(scaleFP32, usedFP32, 0x1.555556p-3f,
                      nonZeroMask); // usedFP32 存在正常值，0x80000000(Nan)情况，Inf 情况
            Reg::Cast<fp8_e4m3fn_t, float, castTraitFp32ToFp8>(scaleFP8, scaleFP32, dataMaskB32); // 开饱和模式
            Reg::Select((Reg::RegTensor<uint32_t>&)scaleFP8, fp8NanU32, (Reg::RegTensor<uint32_t>&)scaleFP8,
                        allNanMask); // 全 Nan Block, 其对应的 Scale 也是 Nan
            if constexpr (colNumIs16) {
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)scalePackFP8,
                                                                        (Reg::RegTensor<uint32_t>&)scaleFP8);
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM>(
                    mxScaleLocalAddr, scalePackFP8, VF_LEN_16, storeMaskB8Half);
            } else {
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                    mxScaleLocalAddr, (Reg::RegTensor<uint8_t>&)scaleFP8, VF_LEN_32, storeMaskB8);
            }
            Reg::Cast<float, fp8_e4m3fn_t, castTraitB16ToFp32>(backFP32, scaleFP8, dataMaskB32);
            Reg::Duplicate(invFP32, 0.0f);
            Reg::Div(invFP32, oneFP32, backFP32, nonZeroMask);
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM>(
                recipScaleLocalAddr, invFP32, VF_LEN_32, dataMaskB32);
        }
    }
    return;
}

// ========================================== Process Scale ======================================================
// The destination row contains validRowWidth scales, rounded up to an even byte count.
// Read only real scales: skipped DMA blocks and implicit padding have no defined value.
// The source row is at least as wide as the destination, so forward in-place compaction is safe.
template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::ProcessScale(__ubuf__ uint8_t* mxScaleLocalAddr,
                                                                       uint16_t srcRowWidth, uint16_t validRowWidth,
                                                                       uint32_t count)
{
    const uint16_t windows = static_cast<uint16_t>((count + VF_LEN_16 - 1) / VF_LEN_16);
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> lane, q, row, col, offset, values, zero;
        Reg::RegTensor<uint16_t> rowWidth, sourceRowWidth;
        Reg::MaskReg realScale;
        Reg::Arange((Reg::RegTensor<int16_t>&)lane, int16_t(0));
        Reg::Duplicate(zero, uint16_t(0));
        Reg::Duplicate(rowWidth, static_cast<uint16_t>((validRowWidth + 1) / 2 * 2));
        Reg::Duplicate(sourceRowWidth, srcRowWidth);
        for (uint16_t step = 0; step < windows; ++step) {
            const uint32_t start = static_cast<uint32_t>(step) * VF_LEN_16;
            uint32_t remaining = count - start;
            Reg::MaskReg valid = Reg::UpdateMask<uint16_t>(remaining);
            Reg::Adds(q, lane, uint16_t(start), valid);
            Reg::Div(row, q, rowWidth, valid);
            Reg::Mul(col, row, rowWidth, valid);
            Reg::Sub(col, q, col, valid);
            Reg::Mul(offset, row, sourceRowWidth, valid);
            Reg::Add(offset, offset, col, valid);
            Reg::Compares<uint16_t, CMPMODE::LT>(realScale, col, validRowWidth, valid);
            Reg::DataCopyGather(values, mxScaleLocalAddr, offset, realScale);
            Reg::Select(values, values, zero, realScale);
            uint32_t bytes = (count - start < VF_LEN_16 ? count - start : VF_LEN_16) * 2;
            Reg::MaskReg storeMask = Reg::UpdateMask<uint8_t>(bytes);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr + start,
                                                                    (Reg::RegTensor<uint8_t>&)values, storeMask);
        }
    }
    return;
}

// ========================================== Insert Zero Scale ======================================================
// 奇数块行的 scale 布局修正：行宽 srcRowWidth（每行 srcRowWidth 个有效 scale），原地展宽为
// srcRowWidth + 1（行尾插 1 个 0），供 CopyOut 按 ceil2(blocks) 紧凑搬出。
// count 为目的侧总字节数（行数 × (srcRowWidth + 1)），需 < 65536。
// 映射：目的位置 q → row = q / (srcRowWidth + 1)，col = q % (srcRowWidth + 1)；
// col < srcRowWidth 时源位置 = row * srcRowWidth + col = q - row，col == srcRowWidth 处填 0。
// 展宽时源位置 ≤ 目的位置，必须自高向低按窗处理（先尾窗，整窗降序）保证原地安全。
template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::InsertZeroScale(__ubuf__ uint8_t* mxScaleLocalAddr,
                                                                          uint16_t srcRowWidth, uint32_t count)
{
    const uint16_t fullWindows = static_cast<uint16_t>(count / VF_LEN_16);
    uint32_t remaining = fullWindows * VF_LEN_16;
    uint32_t bytes = remaining * 2;
    uint32_t start = fullWindows * VF_LEN_16;
    uint32_t tail = count - start;
    uint32_t taildouble = tail * 2;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> lane, q, row, col, index, values, zero;
        Reg::RegTensor<uint16_t> rowWidth;
        Reg::MaskReg realScale;
        Reg::Arange((Reg::RegTensor<int16_t>&)lane, int16_t(0));
        Reg::Duplicate(zero, uint16_t(0));
        Reg::Duplicate(rowWidth, static_cast<uint16_t>(srcRowWidth + DIGIT_ONE)); // 目的行宽
        if (tail > DIGIT_ZERO) { // 尾窗（最高地址窗）先处理
            Reg::MaskReg valid = Reg::UpdateMask<uint16_t>(tail);
            Reg::MaskReg storeMask = Reg::UpdateMask<uint8_t>(taildouble); // PACK_B16 掩码按 uint16 粒度
            Reg::Adds(q, lane, uint16_t(start), valid);                    // 目的线性位置
            Reg::Div(row, q, rowWidth, valid);                             // 行号 row = q / 目的行宽
            Reg::Mul(col, row, rowWidth, valid);
            Reg::Sub(col, q, col, valid); // col = q % 目的行宽
            Reg::Compares<uint16_t, CMPMODE::LT>(realScale, col, uint16_t(srcRowWidth), valid); // 真 scale 判定
            Reg::Sub(index, q, row, valid);                                  // 源位置 = 目的位置 - row
            Reg::DataCopyGather(values, mxScaleLocalAddr, index, realScale); // vgather2 逐 lane 取字节
            Reg::Select(values, values, zero, realScale);                    // 行尾位置填 0
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr + start,
                                                                    (Reg::RegTensor<uint8_t>&)values, storeMask);
        }
        for (uint16_t step = 0; step < fullWindows; ++step) { // 整窗自高向低
            start -= VF_LEN_16;
            Reg::MaskReg valid = Reg::UpdateMask<uint16_t>(remaining);
            Reg::MaskReg storeMask = Reg::UpdateMask<uint8_t>(bytes);
            Reg::Adds(q, lane, uint16_t(start), valid); // 目的线性位置
            Reg::Div(row, q, rowWidth, valid);          // 行号 row = q / 目的行宽
            Reg::Mul(col, row, rowWidth, valid);
            Reg::Sub(col, q, col, valid); // col = q % 目的行宽
            Reg::Compares<uint16_t, CMPMODE::LT>(realScale, col, uint16_t(srcRowWidth), valid); // 真 scale 判定
            Reg::Sub(index, q, row, valid);                                  // 源位置 = 目的位置 - row
            Reg::DataCopyGather(values, mxScaleLocalAddr, index, realScale); // vgather2 逐 lane 取字节
            Reg::Select(values, values, zero, realScale);                    // 行尾位置填 0
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr + start,
                                                                    (Reg::RegTensor<uint8_t>&)values, storeMask);
        }
    }
    return;
}

// ================================================ Compute Data =================================================
// 载入 128 个源元素后拆分为独立的奇/偶 FP32 数据流，两路共享同一组 8 个 FP32 block 倒数（每个广播 8 次）
template <typename T, typename U>
template <Reg::RegLayout layout>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::ComputeDataToBf16(Reg::RegTensor<bfloat16_t>& xBF16,
                                                                            Reg::RegTensor<float>& xFP32,
                                                                            Reg::RegTensor<float>& invScaleFP32,
                                                                            Reg::MaskReg& dataMaskB32)
{
    static constexpr Reg::CastTrait castTraitFp32ToBf16 = {layout, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                                           RoundMode::CAST_TRUNC};
    Reg::RegTensor<float> scaledFP32;
    Reg::RegTensor<uint32_t> lowMask32Bit;
    Reg::RegTensor<uint32_t> stickyBit;
    Reg::RegTensor<uint32_t> discarded;
    Reg::RegTensor<uint32_t> prepared;
    Reg::MaskReg hasDiscardedMask;
    Reg::Duplicate(lowMask32Bit, FP32_LAST_16_MAN_MASK);
    Reg::Duplicate(stickyBit, FP32_MAN_ODD_MASK);
    Reg::Mul(scaledFP32, xFP32, invScaleFP32, dataMaskB32);
    // E2M1 值与中间点的 BF16 低位为偶数。将丢弃的 FP32 低位保留为奇数 BF16 低位，保持每个边界的取舍方向
    // 最终的 E2M1 Cast 仍按 rint/round/floor 选择；±0 与 Inf 精确转换
    Reg::And(discarded, (Reg::RegTensor<uint32_t>&)scaledFP32, lowMask32Bit, dataMaskB32);
    Reg::Compares<uint32_t, CMPMODE::NE>(hasDiscardedMask, discarded, uint32_t(0), dataMaskB32);
    Reg::Or(prepared, (Reg::RegTensor<uint32_t>&)scaledFP32, stickyBit, dataMaskB32);
    Reg::Select(prepared, prepared, (Reg::RegTensor<uint32_t>&)scaledFP32, hasDiscardedMask);
    Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16>(xBF16, (Reg::RegTensor<float>&)prepared, dataMaskB32);
    return;
}

template <typename T, typename U>
template <RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::ComputeDataNvfp4(__ubuf__ T* xLocalAddr,
                                                                           __ubuf__ float* recipScaleLocalAddr,
                                                                           __ubuf__ int8_t* yLocalAddr,
                                                                           uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        static constexpr Reg::CastTrait castTraitB16ToFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitB16ToFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, roundMode};
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<float> invFP32;
        Reg::RegTensor<float> xZeroFP32;
        Reg::RegTensor<float> xOneFP32;
        Reg::RegTensor<bfloat16_t> xZeroBF16;
        Reg::RegTensor<bfloat16_t> xOneBF16;
        Reg::RegTensor<bfloat16_t> sixBF16;
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)sixBF16, uint16_t(0x40c0)); // bf16(6.0)
        Reg::MaskReg nanMask;
        Reg::RegTensor<U> xFP4;
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg storeMaskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(xSrc, xLocalAddr, VF_LEN_16);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B32>(
                invFP32, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);
            Reg::Cast<float, T, castTraitB16ToFp32Zero>(xZeroFP32, xSrc, dataMaskB16);
            Reg::Cast<float, T, castTraitB16ToFp32One>(xOneFP32, xSrc, dataMaskB16);
            ComputeDataToBf16<>(xZeroBF16, xZeroFP32, invFP32, dataMaskB32);
            ComputeDataToBf16<Reg::RegLayout::ONE>(xOneBF16, xOneFP32, invFP32, dataMaskB32);
            Reg::Or((Reg::RegTensor<uint32_t>&)xZeroBF16, (Reg::RegTensor<uint32_t>&)xZeroBF16,
                    (Reg::RegTensor<uint32_t>&)xOneBF16, dataMaskB32);
            Reg::Compare<bfloat16_t, CMPMODE::NE>(nanMask, xZeroBF16, xZeroBF16, dataMaskB16);
            Reg::Select(xZeroBF16, sixBF16, xZeroBF16, nanMask);
            Reg::Cast<U, bfloat16_t, castTrait>(xFP4, xZeroBF16, dataMaskB16);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xFP4, FP4_OUT_ELE_PER_BLK, storeMaskB8);
        }
    }
    return;
}

template <typename T, typename U>
template <bool colNumIs16>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::ComputeNvfp4(
    __ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr, __ubuf__ uint8_t* mxScaleLocalAddr,
    __ubuf__ int8_t* yLocalAddr, __ubuf__ float* recipScaleLocalAddr, uint16_t loopNum2VF, uint16_t loopNumHalfVF)
{
    ComputeMaxExpNvfp4(xLocalAddr, maxExpLocalAddr, loopNum2VF);
    ComputeScaleNvfp4<colNumIs16>(maxExpLocalAddr, mxScaleLocalAddr, recipScaleLocalAddr, loopNumHalfVF, maxLowBound_);
    if (roundMode_ == MODE_RINT) {
        ComputeDataNvfp4<RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, loopNum2VF);
    } else if (roundMode_ == MODE_ROUND) {
        ComputeDataNvfp4<RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, loopNum2VF);
    } else if (roundMode_ == MODE_FLOOR) {
        ComputeDataNvfp4<RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, loopNum2VF);
    }
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxisNvfp4<T, U>::Compute(int64_t ubFactorRowBlockNum,
                                                                  int64_t ubFactorColBlockNum, int64_t ubFactorColNum)
{
    bool colBlockAligned = (ubFactorColNum % blockSize_ == DIGIT_ZERO); // colNum 为块大小整数倍：按精确块数计算
    int64_t ubFactorColBlockNumAlign = ops::CeilDiv(ubFactorColBlockNum, DIGIT_FOUR) * DIGIT_FOUR;
    uint32_t totalBlockNum = colBlockAligned ?
                                 ubFactorRowBlockNum * ubFactorColBlockNum :
                                 ubFactorRowBlockNum *
                                     ubFactorColBlockNumAlign; // 当前Ub内总的Block数量 = 当前Ub内总的计算出Scale数量

    // 数据方向：每个向量 8 个 block16 = 128 个元素
    uint16_t loopNum2VF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), ELEMENT_AFTER_REDUCE_);
    // scale 方向：每次 VF 处理 64 个 block 的倒数
    uint16_t loopNumHalfVF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), static_cast<uint16_t>(VF_LEN_32));

    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue_.AllocTensor<uint8_t>();
    LocalTensor<int8_t> yLocal = outQueue_.AllocTensor<int8_t>();
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    LocalTensor<float> recipScaleLocal = recipScaleBuffer_.Get<float>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(mxScaleLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yLocal.GetPhyAddr());
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    auto recipScaleLocalAddr = reinterpret_cast<__ubuf__ float*>(recipScaleLocal.GetPhyAddr());

    if (ubFactorColNum == 16) {
        ComputeNvfp4<true>(xLocalAddr, maxExpLocalAddr, mxScaleLocalAddr, yLocalAddr, recipScaleLocalAddr, loopNum2VF,
                           loopNumHalfVF);
    } else {
        ComputeNvfp4<false>(xLocalAddr, maxExpLocalAddr, mxScaleLocalAddr, yLocalAddr, recipScaleLocalAddr, loopNum2VF,
                            loopNumHalfVF);
    }

    if (colBlockAligned) {
        // colNum 为块大小整数倍（16 特例由 ComputeScaleNvfp4<true> 直接产出，不走到这里）：
        // 奇数块行行尾插 0 补到 ceil2(blocks)，供 CopyOut 按 blocks+奇偶 紧凑搬出
        if (ubFactorColNum != 16 && ubFactorColBlockNum % DIGIT_TWO != DIGIT_ZERO) {
            InsertZeroScale(mxScaleLocalAddr, static_cast<uint16_t>(ubFactorColBlockNum),
                            static_cast<uint32_t>(ubFactorRowBlockNum * (ubFactorColBlockNum + DIGIT_ONE)));
        }
    } else {
        const int64_t scaleRowWidth = ops::CeilDiv(ubFactorColBlockNum, DIGIT_TWO) * DIGIT_TWO;
        if (scaleRowWidth != ubFactorColBlockNumAlign || scaleRowWidth != ubFactorColBlockNum) {
            ProcessScale(mxScaleLocalAddr, static_cast<uint16_t>(ubFactorColBlockNumAlign),
                         static_cast<uint16_t>(ubFactorColBlockNum),
                         static_cast<uint32_t>(ubFactorRowBlockNum * scaleRowWidth));
        }
    }

    inQueue_.FreeTensor(xLocal);
    mxScaleQueue_.EnQue(mxScaleLocal);
    outQueue_.EnQue(yLocal);
    return;
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_TAIL_AXIS_NVFP4_H
