/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file ms_deform_attn_310p.h
 * \brief
 */
#ifndef MS_DEFORM_ATTN_310P
#define MS_DEFORM_ATTN_310P

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MultiScaleDeformableAttn
{

using namespace AscendC;

template <typename T>
class KernelMultiScaleDeformableAttn310P
{
public:
    __aicore__ inline KernelMultiScaleDeformableAttn310P(){};
    __aicore__ inline void Init(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
                                GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
                                const MultiScaleDeformableAttnFunctionTilingData* tilingData);
    __aicore__ inline void MSDAProcess();

private:
    __aicore__ inline void InitGM(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
                                  GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output);
    __aicore__ inline void ClearGM();
    __aicore__ inline void GridSamplerProcess(LocalTensor<float> location, LocalTensor<float> output,
                                              LocalTensor<float> gridOutput, int64_t valueOffset,
                                              int64_t samplingOffset, int64_t attentionOffset);
    __aicore__ inline void InputGenerate(int64_t locationOffset, LocalTensor<float> location, int32_t calHWElemsAlign);
    __aicore__ inline void OutputGenerate(int16_t hwIdx, int16_t cIdx, int16_t loop_idx, int32_t calCElems,
                                          int64_t loopOffset, int64_t outBaseOffset, int16_t loopElems,
                                          int64_t weightBaseOffset, LocalTensor<float> output,
                                          LocalTensor<float> gridOutput);
    __aicore__ inline void ParseTilingData(const MultiScaleDeformableAttnFunctionTilingData* tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign,
                                          LocalTensor<float> location, int64_t locationOffset,
                                          LocalTensor<float> output, LocalTensor<float> gridOutput, int64_t valueOffset,
                                          int64_t attentionOffset);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                           LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
                                           LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> weightMaskUb, uint16_t id,
                                           int32_t nIdx, int32_t hwIdx);
    __aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound);
    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                                       LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
                                                       LocalTensor<uint8_t> maskTmpXUb,
                                                       LocalTensor<uint8_t> maskTmpYUb);
    __aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
                                                   LocalTensor<uint8_t> maskUb, const float scalarVal,
                                                   const uint32_t calNum);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int16_t cIdx, int32_t calCElems, int32_t channelAlign,
                                       int32_t loopOffset, int16_t loopElems, LocalTensor<int32_t> coorUb,
                                       LocalTensor<float> xLocal, int64_t valueOffset);
    __aicore__ inline void ValueGMCopy(int16_t cIdx, int32_t calCElems, int64_t doubleChannelAlign, int16_t i,
                                       int64_t base, int64_t offset, LocalTensor<float> xLocal, LocalTensor<int32_t> coorUb,
                                       DataCopyParams params);
    __aicore__ inline void MTE2ForLargeC(int16_t cIdx, int32_t calCElems, int64_t doubleChannelAlign, int64_t xLocation,
                                         int64_t coordVal, LocalTensor<float> xLocal);
    __aicore__ inline void calculateEachPointValue(int32_t nIdx, int32_t calCElems, int32_t channelAlign,
                                                   int32_t loopOffset, LocalTensor<float> weightUb,
                                                   LocalTensor<float> outValueUb, LocalTensor<float> outValueUbSum);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb);
    __aicore__ inline void PointBilinear2(int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign,
                                          LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
                                          LocalTensor<float> weightUb2, LocalTensor<float> weightUb3,
                                          LocalTensor<float> weightUb4, LocalTensor<uint8_t> weightMaskUb,
                                          LocalTensor<uint8_t> weightMaskUb2, LocalTensor<uint8_t> weightMaskUb3,
                                          LocalTensor<uint8_t> weightMaskUb4, LocalTensor<float> outValueUb,
                                          bool isAutomicAdd, LocalTensor<float> output, LocalTensor<float> gridOutput,
                                          int64_t valueOffset, int64_t attentionOffset);
    __aicore__ inline void GridOutForNCHW(int16_t hwIdx, int16_t cIdx, int16_t loop_idx, int32_t calCElems,
                                          int32_t loopOffset, int16_t loopElems, int64_t outBaseOffset,
                                          int64_t weightBaseOffset, LocalTensor<float> outValueUbSum,
                                          LocalTensor<float> output, LocalTensor<float> gridOutput);
    __aicore__ inline void outTransCal(int16_t loopElems, LocalTensor<float> outTransTensor, LocalTensor<float> tmpOut,
                                       int32_t sumOutputLen, int32_t outRepeatLen, int32_t tailNum, int32_t copyNum, bool tailByteAlign);
    __aicore__ inline void outTransCopyOut(bool isDataCopy, int32_t copyDstStride, int32_t copySrcStride, int64_t outputGmBaseOffset,
                                           int16_t loopElems, LocalTensor<float> outTransTensor, int32_t outRepeatLen, int32_t tailNum,
                                           int32_t outRepeatLenAlign, int32_t copyNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gridQueue_;

    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> inputXYFPBuf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> inputXFpBuf_;
    TBuf<QuePosition::VECCALC> inputYFpBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> weightTmpBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;

    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf2_;

    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf2_;

    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> maskBuf3_;
    TBuf<QuePosition::VECCALC> maskBuf4_;
    TBuf<QuePosition::VECCALC> maskBuf6_;
    TBuf<QuePosition::VECCALC> maskBuf8_;
    TBuf<QuePosition::VECCALC> maskBuf9_;

    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf2_;
    TBuf<QuePosition::VECCALC> weightMaskBuf3_;
    TBuf<QuePosition::VECCALC> weightMaskBuf4_;
    TBuf<QuePosition::VECCALC> weightMaskBuf5_;
    TBuf<QuePosition::VECCALC> weightMaskBuf6_;
    TBuf<QuePosition::VECCALC> weightMaskBuf7_;
    TBuf<QuePosition::VECCALC> weightMaskBuf8_;
    TBuf<QuePosition::VECCALC> weightMaskBuf9_;

    TBuf<QuePosition::VECCALC> dupBuf_;
    TBuf<QuePosition::VECCALC> dupBuf3_;
    TBuf<QuePosition::VECCALC> dupBuf4_;
    TBuf<QuePosition::VECCALC> dupBuf6_;
    TBuf<QuePosition::VECCALC> dupBuf8_;
    TBuf<QuePosition::VECCALC> dupBuf9_;

    TBuf<QuePosition::VECCALC> bufferMaskBuf_;
    TBuf<QuePosition::VECCALC> bufferBuf_;

    GlobalTensor<float> valueGm_;
    GlobalTensor<float> locationGm_;
    GlobalTensor<float> attentionWeightsGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_;
    GlobalTensor<int32_t> valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> gridOutQue_;
    TBuf<QuePosition::VECCALC> tmpOutQue_;
    TBuf<TPosition::VECCALC> workLocalQue_;

    TBuf<TPosition::VECCALC> locationQue_;
    TBuf<TPosition::VECCALC> attentionWeightsQue_;
    TBuf<TPosition::VECCALC> outputQue_;

    const int64_t CHANNEL_BLOCK = 32;
    const int64_t BLOCK_SIZE = 32;
    const int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);

    const int64_t B32_MASK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;
    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t CAL_H_W_BLOCK = 512;
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / BLOCK_NUM;
    const int64_t MASK_SIZE = 960;
    const int64_t WEIGHT_MASK_SIZE = 320;

    const int64_t OUT_UB_SIZE_4_GENERAL = 65536;
    const int64_t OUT_UB_SIZE_GENERAL = 16384;
    const int64_t X_UB_SIZE_4_GENERAL = 65536;

    int64_t blockIDX = 0;

    // tiling params
    int64_t batchSize_ = 0;
    int64_t numKeys_ = 0;
    int64_t numHeads_ = 0;
    int64_t embedDims_ = 0;
    int64_t numLevels_ = 0;
    int64_t numQueries_ = 0;
    int64_t numPoints_ = 0;
    int64_t pointLoops_ = 0;
    int64_t realLevels_ = 0;
    int64_t optPoint_ = 0;

    int64_t coreNum_ = 0;
    int64_t inputN_ = 0;
    int64_t inputC_ = 0;
    int64_t inputH_ = 0;
    int64_t inputW_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    int64_t needCoreNum_ = 0;

    int64_t gridHW_ = 0;
    int64_t lastLoopHW_ = 0;
    int64_t lastLoopHWAlign_ = 0;
    int64_t preNUbLoop_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;
    int64_t valueNumPerLevel = 0;

    int64_t alignmentType_ = 0;

    int64_t locationLen = 0;
    int64_t attentionWeightsLen = 0;
    int64_t outputLen = 0;

    int32_t calCElemsAlign = 0;

    int32_t currentLevel = 0;

    // const define
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;
    constexpr static int64_t LAYOUT_NHWC = 1;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;

    constexpr static int64_t SLIDING_WINDOW_C_LIMIT = 16;

    constexpr static int64_t ALIGNMENT_TYPE_1 = 1;

    constexpr static uint16_t U_INT16_T_MAX_VALUE = 65535;
    constexpr static uint32_t TWO = 2;
    constexpr static uint32_t FOUR = 4;
};

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::Init(
    GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations,
    GM_ADDR attentionWeights, GM_ADDR output, const MultiScaleDeformableAttnFunctionTilingData* tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output);                               // 1KB

    pipe.InitBuffer(gridOutQue_, OUT_UB_SIZE_GENERAL);  // 16KB
    pipe.InitBuffer(tmpOutQue_, OUT_UB_SIZE_GENERAL);   // 16KB
    pipe.InitBuffer(workLocalQue_, 1024);               // 1KB

    // buffer申请初始化
    pipe.InitBuffer(gridQueue_, 1, GRID_UB_SIZE_4_GENERAL);  // 4KB
    pipe.InitBuffer(dupBuf_, 2048);                          // 2KB
    pipe.InitBuffer(dupBuf3_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf4_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf6_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf8_, 2048);                         // 2KB
    pipe.InitBuffer(dupBuf9_, 2048);                         // 2KB

    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);              // 64KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);   // 4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);     // 8KB
    pipe.InitBuffer(weightTmpBuf_, Y_UB_SIZE_4_GENERAL * 4);  // 8KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);         // 2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);           // 2KB

    pipe.InitBuffer(outValueBuf_, OUT_UB_SIZE_4_GENERAL);  // 64KB
    pipe.InitBuffer(outValueBuf2_, OUT_UB_SIZE_GENERAL);   // 16KB

    pipe.InitBuffer(maskBuf_, MASK_SIZE);   // 960B
    pipe.InitBuffer(maskBuf3_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf4_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf6_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf8_, MASK_SIZE);  // 960B
    pipe.InitBuffer(maskBuf9_, MASK_SIZE);  // 960B

    pipe.InitBuffer(weightMaskBuf_, WEIGHT_MASK_SIZE);   // 320B
    pipe.InitBuffer(weightMaskBuf2_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf3_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf4_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf5_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf6_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf7_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf8_, WEIGHT_MASK_SIZE);  // 320B
    pipe.InitBuffer(weightMaskBuf9_, WEIGHT_MASK_SIZE);  // 320B

    pipe.InitBuffer(bufferMaskBuf_, BLOCK_SIZE);              // 32B
    pipe.InitBuffer(bufferBuf_, BLOCK_SIZE * CHANNEL_BLOCK);  // 4K

    LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
    bufPattern.SetValue(0, 0b11111111);
    ClearGM();
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::InitGM(GM_ADDR value, GM_ADDR valueSpatialShapes,
                                                                     GM_ADDR valueLevelStartIndex,
                                                                     GM_ADDR samplingLocations,
                                                                     GM_ADDR attentionWeights, GM_ADDR output)
{
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
    locationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(samplingLocations));
    attentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionWeights));
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));

    valueSpatialShapesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueSpatialShapes));
    valueLevelStartIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueLevelStartIndex));

    locationLen = 1024;                                            // 1024
    attentionWeightsLen = GRID_UB_SIZE_4_GENERAL;                  // 4096
    outputLen = GRID_UB_SIZE_4_GENERAL / numPoints_;               // 1024

    pipe.InitBuffer(locationQue_, locationLen * B32_BYTE_SIZE);                                 // 4KB
    pipe.InitBuffer(attentionWeightsQue_, (attentionWeightsLen / embedDims_) * B32_BYTE_SIZE);  // 2KB
    pipe.InitBuffer(outputQue_, outputLen * B32_BYTE_SIZE);
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::ClearGM()
{
    LocalTensor<float> zeroTensor = tmpOutQue_.Get<float>();
    Duplicate(zeroTensor, (float)0.0, zeroTensor.GetSize());

    int32_t coreSplitCount = batchSize_ * numHeads_ * embedDims_ / needCoreNum_;
    int64_t gmSize = coreSplitCount * numQueries_;
    int32_t loopTime = gmSize / 4096;
    int32_t tailGmsize = gmSize  % 4096;

    event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
    int64_t outGMOffset = gmSize * blockIDX ;
    for (int32_t i = 0; i < loopTime; i++) {
        DataCopy(outputGm_[outGMOffset + i * 4096], zeroTensor, 4096);
    }
    if (tailGmsize > 0) {
        int32_t lastGmSize = tailGmsize / 8 * 8;
        int32_t lastTailGmsize = tailGmsize % 8;
        outGMOffset = outGMOffset + loopTime * 4096 ;
        if (lastGmSize > 0) {
            DataCopy(outputGm_[outGMOffset], zeroTensor, lastGmSize);
        }
        if (lastTailGmsize > 0) {
            outGMOffset = outGMOffset + lastGmSize + lastTailGmsize - 8;
            DataCopy(outputGm_[outGMOffset], zeroTensor, 8);
        }
    }
}

/**
 * @description: 解析tiling数据，计算分核数据
 * @param {MultiScaleDeformableAttnFunctionTilingData*} tilingData
 * @return {*}
 */
template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::ParseTilingData(
    const MultiScaleDeformableAttnFunctionTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    numKeys_ = tilingData->numKeys;
    numHeads_ = tilingData->numHeads;
    embedDims_ = tilingData->embedDims;
    numLevels_ = tilingData->numLevels;
    numQueries_ = tilingData->numQueries;
    numPoints_ = tilingData->numPoints;
    pointLoops_ = tilingData->pointLoops;
    realLevels_ = tilingData->realLevels;
    coreNum_ = tilingData->coreNum;

    inputN_ = batchSize_ * numHeads_;
    inputC_ = embedDims_;
    outputH_ = numQueries_;
    outputW_ = numPoints_;
    needCoreNum_ = coreNum_;
    gridHW_ = outputH_ * outputW_;
    preNUbLoop_ = Ceil(gridHW_, CAL_H_W_BLOCK);
    lastLoopHW_ = gridHW_ - CAL_H_W_BLOCK * (preNUbLoop_ - 1);
    lastLoopHWAlign_ = Ceil(lastLoopHW_, BLOCK_NUM) * BLOCK_NUM;
    totalUbLoop_ = preNUbLoop_ * inputN_;
    preCoreLoop_ = Ceil(totalUbLoop_, needCoreNum_);
    needCoreNum_ = Ceil(totalUbLoop_, preCoreLoop_);
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);

    channelLoop_ = Ceil(inputC_, CHANNEL_BLOCK);
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);

    if (gridHW_ % TRANSE_REP_STRIDE < BLOCK_NUM && gridHW_ % TRANSE_REP_STRIDE != 0) {
        alignmentType_ = ALIGNMENT_TYPE_1;
    }
}

/**
 * @description: 坐标超过上下界的处理
 * @param {LocalTensor<int32_t>} x or y
 * @param {int32_t} 上界
 * @return {*}
 */
template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::CoordinatesFrameRange(LocalTensor<int32_t> iIntUb,
                                                                                    int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();;
    Maxs(iIntUb, iIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();;
}

/**
 * @description: 取出合法坐标点：maskXUb，maskYUb
 * @return {*}
 */
template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::CoordinatesGetMaskWithRange(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    // maskTmpXUb存的是大于0的合法点
    CompareScalar<float, uint8_t, false>(maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, MASK_PLACEHOLDER,
                                         CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});

    // maskXUb存的是小于inputW_的合法点
    CompareScalar<float, uint8_t, false>(maskXUb, iXFpUb, static_cast<float>(inputW_ - 1), CMPMODE::LE,
                                         MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    // maskTmpYUb存的是大于0的合法点
    CompareScalar<float, uint8_t, false>(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, MASK_PLACEHOLDER,
                                         CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    // maskYUb存的是小于inputH_的合法点
    CompareScalar<float, uint8_t, false>(maskYUb, iYFpUb, static_cast<float>(inputH_ - 1), CMPMODE::LE,
                                         MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});

    PipeBarrier<PIPE_V>();;

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    // 合并上面的两个结果，得到最终合法点
    And<uint16_t, false>(maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, MASK_PLACEHOLDER, MASK_UB_SIZE / B32_MASK,
                         {1, 1, 1, 8, 8, 8});
    And<uint16_t, false>(maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, MASK_PLACEHOLDER, MASK_UB_SIZE / B32_MASK,
                         {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();;
    And<uint16_t, false>(maskXUbTmp, maskYUbTmp, maskXUbTmp, MASK_PLACEHOLDER, MASK_UB_SIZE / B32_MASK,
                         {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();;
    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
}

/**
 * @description: 计算grid中的x、y的一维坐标和越界点的mask值
 * @param {LocalTensor<float>} iXFpUb
 * @param {LocalTensor<float>} iYFpUb
 * @param {LocalTensor<int32_t>} iXIntUb
 * @param {LocalTensor<int32_t>} iYIntUb
 * @param {LocalTensor<int32_t>} out coorUb
 * @param {LocalTensor<uint8_t>} out wMaskUb
 * @return {*}
 */
template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::ClipCoordinates(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
    LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> wMaskUb, uint16_t id, int32_t nIdx, int32_t hwIdx)
{
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];  // 2: iY temp mask
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);

    if (id == 1) {
        LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
        LocalTensor<int32_t> inputXIntTmpUb = coorUb;
        LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;
        Adds<int32_t, false>(inputXIntTmpUb, iXIntUb, 0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
        Adds<int32_t, false>(inputYIntTmpUb, iYIntUb, 0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();;
        // 重计算坐标，使坐标不超过边界
        CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(inputW_ - 1));
        CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(inputH_ - 1));

        PipeBarrier<PIPE_V>();;

        // cood = y + x * IW
        Muls<int32_t, false>(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, MASK_PLACEHOLDER,
                             CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();;
        Add<int32_t, false>(coorUb, coorUb, inputYIntTmpUb, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                            {1, 1, 1, 8, 8, 8});
    }
}

/**
 * @description: x是NHWC的format，连续搬运align(c)个，按hw循环
 * @param {int32_t} nIdx
 * @param {int32_t} cIdx
 * @param {int32_t} calCElems
 * @param {int32_t} channelAlign
 * @param {int32_t} loopOffset
 * @param {int32_t} loopElems
 * @param {LocalTensor<int32_t>} coorUb
 * @param {LocalTensor<float>} xLocal
 * @return {*}
 */
template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::MTE2ForNHWC(
    int32_t nIdx, int16_t cIdx, int32_t calCElems, int32_t channelAlign, int32_t loopOffset, int16_t loopElems,
    LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal, int64_t valueOffset)
{
    int64_t base = valueOffset + (int64_t)nIdx * numKeys_ * inputC_ + cIdx * CHANNEL_BLOCK;

    auto timeStep = loopElems / 8;

    calCElemsAlign = Ceil(calCElems, 8) * 8;
    int64_t doubleChannelAlign = static_cast<int64_t>(channelAlign) * 4;
    uint16_t blockCount = 2;
    uint16_t blockLen = calCElemsAlign * 2 / 8;
    uint16_t wcLength = inputW_ * inputC_ / 8;
    uint16_t srcStride = wcLength - blockLen;
    uint16_t dstStride = 0;
    if (wcLength < blockLen) {
        blockLen = calCElemsAlign / 8;
        srcStride = 0;
        dstStride = blockLen;
    }
    DataCopyParams params{blockCount, blockLen, srcStride, dstStride};

    // 这边为了不打断流水，提高性能
    for (int16_t i = 0; i < timeStep; i++) {
        int64_t offset = loopOffset + i * 8;
        ValueGMCopy(cIdx, calCElems, doubleChannelAlign, i, base, offset, xLocal, coorUb, params);
    }

    if (inputC_ == 32) {
        for (int16_t i = loopElems / 8 * 8; i < loopElems; i++) {
            int64_t coordVal_0 = base + coorUb.GetValue(loopOffset + i);
            DataCopy(xLocal[i * doubleChannelAlign], valueGm_[coordVal_0], params);
        }
    } else {
        for (int16_t i = loopElems / 8 * 8; i < loopElems; i++) {
            int64_t coordVal_0 = base + coorUb.GetValue(loopOffset + i);
            MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, i * doubleChannelAlign, coordVal_0, xLocal);
        }
    }
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::ValueGMCopy(int16_t cIdx, int32_t calCElems,
                                                                          int64_t doubleChannelAlign, int16_t i,
                                                                          int64_t base, int64_t offset,
                                                                          LocalTensor<float> xLocal, LocalTensor<int32_t> coorUb,
                                                                          DataCopyParams params)
{
    event_t eventSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));

    int64_t coordVal_0 = base + coorUb.GetValue(offset);
    int64_t coordVal_1 = base + coorUb.GetValue(offset + 1);
    int64_t coordVal_2 = base + coorUb.GetValue(offset + 2);
    int64_t coordVal_3 = base + coorUb.GetValue(offset + 3);
    int64_t coordVal_4 = base + coorUb.GetValue(offset + 4);
    int64_t coordVal_5 = base + coorUb.GetValue(offset + 5);
    int64_t coordVal_6 = base + coorUb.GetValue(offset + 6);
    int64_t coordVal_7 = base + coorUb.GetValue(offset + 7);

    int64_t xLocation_0 = (i * 8) * doubleChannelAlign;

    SetFlag<HardEvent::S_MTE2>(eventSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventSToMTE2);

    if (inputC_ == 32) {
        DataCopy(xLocal[xLocation_0], valueGm_[coordVal_0], params);
        DataCopy(xLocal[xLocation_0 + doubleChannelAlign], valueGm_[coordVal_1], params);
        DataCopy(xLocal[xLocation_0 + doubleChannelAlign * 2], valueGm_[coordVal_2], params);
        DataCopy(xLocal[xLocation_0 + doubleChannelAlign * 3], valueGm_[coordVal_3], params);
        DataCopy(xLocal[xLocation_0 + doubleChannelAlign * 4], valueGm_[coordVal_4], params);
        DataCopy(xLocal[xLocation_0 + doubleChannelAlign * 5], valueGm_[coordVal_5], params);
        DataCopy(xLocal[xLocation_0 + doubleChannelAlign * 6], valueGm_[coordVal_6], params);
        DataCopy(xLocal[xLocation_0 + doubleChannelAlign * 7], valueGm_[coordVal_7], params);
    } else {
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0, coordVal_0, xLocal);
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0 + doubleChannelAlign, coordVal_1, xLocal);
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0 + doubleChannelAlign * 2, coordVal_2,
                      xLocal);
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0 + doubleChannelAlign * 3, coordVal_3,
                      xLocal);
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0 + doubleChannelAlign * 4, coordVal_4,
                      xLocal);
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0 + doubleChannelAlign * 5, coordVal_5,
                      xLocal);
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0 + doubleChannelAlign * 6, coordVal_6,
                      xLocal);
        MTE2ForLargeC(cIdx, calCElems, doubleChannelAlign, xLocation_0 + doubleChannelAlign * 7, coordVal_7,
                      xLocal);
    }
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::MTE2ForLargeC(int16_t cIdx, int32_t calCElems,
                                                                            int64_t doubleChannelAlign,
                                                                            int64_t xLocation, int64_t coordVal,
                                                                            LocalTensor<float> xLocal)
{
    uint16_t blockCount = 2;
    uint16_t blockLen = calCElems / 8;
    DataCopyParams params{blockCount, blockLen, 0, 0};
    for (int32_t i = 0; i < 2; i++) {
        params.srcStride = (inputC_ - calCElems) / 8;
        int64_t xLocationOffset = xLocation + i * doubleChannelAlign / 2;
        int64_t valueGMOffset = coordVal + i * inputW_ * inputC_;
        DataCopy(xLocal[xLocationOffset], valueGm_[valueGMOffset], params);
        PipeBarrier<PIPE_ALL>();;
    }
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
                                                                                      LocalTensor<float> oFpUb,
                                                                                      LocalTensor<uint8_t> maskUb,
                                                                                      const float scalarVal,
                                                                                      const uint32_t calNum)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = Ceil(calNum, B32_VECTOR_MASK);
    Select<float, uint8_t, false>(oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK,
                                  repeat, repParams);
    PipeBarrier<PIPE_V>();;
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::calculateEachPointValue(
    int32_t nIdx, int32_t calCElems, int32_t channelAlign, int32_t loopOffset, LocalTensor<float> weightUb,
    LocalTensor<float> outValueUb, LocalTensor<float> outValueUbSum)
{
    for (int16_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
        int32_t outOffset = i * B32_MASK;
        int32_t weightsOffset = loopOffset + i * B32_MASK;
        Mul<float, false>(outValueUb[outOffset], outValueUb[outOffset], weightUb[weightsOffset], MASK_PLACEHOLDER,
                          calCElems, {1, 1, 1, 16, 16, 0});
    }
    PipeBarrier<PIPE_V>();;
    Add<float, false>(outValueUbSum, outValueUbSum, outValueUb, MASK_PLACEHOLDER,
                      TRANSE_REP_STRIDE * channelAlign / B32_MASK, {1, 1, 1, 8, 8, 8});
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::PointBilinear2(
    int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign, LocalTensor<int32_t> coordinatesUb,
    LocalTensor<float> weightUb, LocalTensor<float> weightUb2, LocalTensor<float> weightUb3,
    LocalTensor<float> weightUb4, LocalTensor<uint8_t> weightMaskUb, LocalTensor<uint8_t> weightMaskUb2,
    LocalTensor<uint8_t> weightMaskUb3, LocalTensor<uint8_t> weightMaskUb4, LocalTensor<float> outValueUb,
    bool isAutomicAdd, LocalTensor<float> output, LocalTensor<float> gridOutput, int64_t valueOffset,
    int64_t attentionOffset)
{
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);  // 逐bit模式

    Muls<int32_t, false>(coordinatesUb, coordinatesUb, (int32_t)inputC_, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                         {1, 1, 8, 8});

    // 非法的点的weight置0
    CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
    CoordinatesSelectScalar(weightUb2, weightUb2, weightMaskUb2, 0.0f, CAL_H_W_BLOCK);
    CoordinatesSelectScalar(weightUb3, weightUb3, weightMaskUb3, 0.0f, CAL_H_W_BLOCK);
    CoordinatesSelectScalar(weightUb4, weightUb4, weightMaskUb4, 0.0f, CAL_H_W_BLOCK);

    LocalTensor<float> outValueUbSum = outValueBuf2_.Get<float>();

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半

    auto weightMaskUbTmp1 = weightMaskUb.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp2 = weightMaskUb2.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp3 = weightMaskUb3.ReinterpretCast<uint16_t>();
    auto weightMaskUbTmp4 = weightMaskUb4.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb5 = weightMaskBuf5_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp5 = weightMaskUb5.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb6 = weightMaskBuf6_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp6 = weightMaskUb6.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb7 = weightMaskBuf7_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp7 = weightMaskUb7.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb8 = weightMaskBuf8_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp8 = weightMaskUb8.ReinterpretCast<uint16_t>();
    LocalTensor<uint8_t> weightMaskUb9 = weightMaskBuf9_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp9 = weightMaskUb9.ReinterpretCast<uint16_t>();

    if (calHWElemsAlign < CAL_H_W_BLOCK) {
        int maskOffset = Ceil(calHWElemsAlign, 16);
        Duplicate(weightMaskUbTmp1[maskOffset], (uint16_t)0, maskNum - maskOffset);
    }

    And<uint16_t, false>(weightMaskUbTmp5, weightMaskUbTmp1, weightMaskUbTmp3, MASK_PLACEHOLDER,
                         MASK_UB_SIZE / B32_MASK, {1, 1, 1, 8, 8, 8});
    weightMaskUb5 = weightMaskUbTmp5.ReinterpretCast<uint8_t>();
    And<uint16_t, false>(weightMaskUbTmp7, weightMaskUbTmp1, weightMaskUbTmp2, MASK_PLACEHOLDER,
                         MASK_UB_SIZE / B32_MASK, {1, 1, 1, 8, 8, 8});
    weightMaskUb7 = weightMaskUbTmp7.ReinterpretCast<uint8_t>();
    And<uint16_t, false>(weightMaskUbTmp6, weightMaskUbTmp3, weightMaskUbTmp4, MASK_PLACEHOLDER,
                         MASK_UB_SIZE / B32_MASK, {1, 1, 1, 8, 8, 8});
    weightMaskUb6 = weightMaskUbTmp6.ReinterpretCast<uint8_t>();
    And<uint16_t, false>(weightMaskUbTmp9, weightMaskUbTmp2, weightMaskUbTmp4, MASK_PLACEHOLDER,
                         MASK_UB_SIZE / B32_MASK, {1, 1, 1, 8, 8, 8});
    weightMaskUb9 = weightMaskUbTmp9.ReinterpretCast<uint8_t>();
    Or<uint16_t, false>(weightMaskUbTmp8, weightMaskUbTmp2, weightMaskUbTmp3, MASK_PLACEHOLDER, MASK_UB_SIZE / B32_MASK,
                        {1, 1, 1, 8, 8, 8});
    weightMaskUb8 = weightMaskUbTmp8.ReinterpretCast<uint8_t>();

    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb3 = maskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb4 = maskBuf4_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb6 = maskBuf6_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb8 = maskBuf8_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> maskUb9 = maskBuf9_.Get<uint8_t>(MASK_UB_SIZE);

    auto weightMaskUbTmp = weightMaskUb7.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_3 = weightMaskUb5.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_4 = weightMaskUb4.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_6 = weightMaskUb6.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_8 = weightMaskUb8.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp_9 = weightMaskUb9.ReinterpretCast<uint64_t>();

    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();
    auto maskUbTmp3 = maskUb3.ReinterpretCast<uint64_t>();
    auto maskUbTmp4 = maskUb4.ReinterpretCast<uint64_t>();
    auto maskUbTmp6 = maskUb6.ReinterpretCast<uint64_t>();
    auto maskUbTmp8 = maskUb8.ReinterpretCast<uint64_t>();
    auto maskUbTmp9 = maskUb9.ReinterpretCast<uint64_t>();

    int32_t trans_loop = Ceil(calHWElemsAlign, TRANSE_REP_STRIDE);
    int16_t loopElems = TRANSE_REP_STRIDE;
    int32_t loopOffset = 0;
    int32_t tmpBaseOffset = loopElems / numPoints_;
    int64_t outBaseOffset = (int64_t)nIdx * outputH_ * inputC_ + hwIdx * tmpBaseOffset * 4;
    int64_t weightBaseOffset = attentionOffset + (int64_t)nIdx * gridHW_ + hwIdx * CAL_H_W_BLOCK;
    int32_t ubOffset = 0;
    int32_t maskOffset = 0;
    BinaryRepeatParams repParams{1, 1, 1, 8, 8, 8};

    LocalTensor<float> xLocal = xBuf_.AllocTensor<float>();
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

    auto dupUb4 = dupBuf4_.Get<float>();
    auto dupUbU32_4 = dupUb4.ReinterpretCast<uint32_t>();
    auto dupUb6 = dupBuf6_.Get<float>();
    auto dupUbU32_6 = dupUb6.ReinterpretCast<uint32_t>();
    auto dupUb8 = dupBuf8_.Get<float>();
    auto dupUbU32_8 = dupUb8.ReinterpretCast<uint32_t>();
    auto dupUb = dupBuf_.Get<float>();
    auto dupUbU32 = dupUb.ReinterpretCast<uint32_t>();
    auto dupUb3 = dupBuf3_.Get<float>();
    auto dupUbU32_3 = dupUb3.ReinterpretCast<uint32_t>();
    auto dupUb9 = dupBuf9_.Get<float>();
    auto dupUbU32_9 = dupUb9.ReinterpretCast<uint32_t>();

    // 按vmask(128)分块，循环处理
    for (int16_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop - 1) {
            loopElems = calHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }

        loopOffset = loop_idx * TRANSE_REP_STRIDE;
        maskOffset = loop_idx * 2;

        maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));
        maskUbTmp.SetValue(2, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(3, weightMaskUbTmp.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32 = maskUbTmp.ReinterpretCast<float>();

        maskUbTmp3.SetValue(0, weightMaskUbTmp_3.GetValue(maskOffset));
        maskUbTmp3.SetValue(1, weightMaskUbTmp_3.GetValue(maskOffset + 1));
        maskUbTmp3.SetValue(2, weightMaskUbTmp_3.GetValue(maskOffset));
        maskUbTmp3.SetValue(3, weightMaskUbTmp_3.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_3 = maskUbTmp3.ReinterpretCast<float>();

        maskUbTmp4.SetValue(0, weightMaskUbTmp_4.GetValue(maskOffset));
        maskUbTmp4.SetValue(1, weightMaskUbTmp_4.GetValue(maskOffset + 1));
        maskUbTmp4.SetValue(2, weightMaskUbTmp_4.GetValue(maskOffset));
        maskUbTmp4.SetValue(3, weightMaskUbTmp_4.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_4 = maskUbTmp4.ReinterpretCast<float>();

        maskUbTmp6.SetValue(0, weightMaskUbTmp_6.GetValue(maskOffset));
        maskUbTmp6.SetValue(1, weightMaskUbTmp_6.GetValue(maskOffset + 1));
        maskUbTmp6.SetValue(2, weightMaskUbTmp_6.GetValue(maskOffset));
        maskUbTmp6.SetValue(3, weightMaskUbTmp_6.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_6 = maskUbTmp6.ReinterpretCast<float>();

        maskUbTmp8.SetValue(0, weightMaskUbTmp_8.GetValue(maskOffset));
        maskUbTmp8.SetValue(1, weightMaskUbTmp_8.GetValue(maskOffset + 1));
        maskUbTmp8.SetValue(2, weightMaskUbTmp_8.GetValue(maskOffset));
        maskUbTmp8.SetValue(3, weightMaskUbTmp_8.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_8 = maskUbTmp8.ReinterpretCast<float>();

        maskUbTmp9.SetValue(0, weightMaskUbTmp_9.GetValue(maskOffset));
        maskUbTmp9.SetValue(1, weightMaskUbTmp_9.GetValue(maskOffset + 1));
        maskUbTmp9.SetValue(2, weightMaskUbTmp_9.GetValue(maskOffset));
        maskUbTmp9.SetValue(3, weightMaskUbTmp_9.GetValue(maskOffset + 1));
        auto weightMaskUbTmpfp32_9 = maskUbTmp9.ReinterpretCast<float>();

        // channel先按32大小循环
        for (int16_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ - 1) {
                calCElems = lastLoopChannel_;
            }
            int32_t channelAlign = Ceil(calCElems, BLOCK_NUM) * BLOCK_NUM;
            MTE2ForNHWC(nIdx, cIdx, calCElems, channelAlign, loopOffset, loopElems, coordinatesUb, xLocal, valueOffset);

            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);
            OutTranspose(channelAlign, xLocal, outValueUb);

            LocalTensor<float> outValueUb2 = outValueUb[channelAlign * (TRANSE_REP_STRIDE)];
            LocalTensor<float> outValueUb3 = outValueUb2[channelAlign * (TRANSE_REP_STRIDE)];
            LocalTensor<float> outValueUb4 = outValueUb3[channelAlign * (TRANSE_REP_STRIDE)];

            if (loop_idx > 0 || (loop_idx == 0 && cIdx > 0)) {
                WaitFlag<HardEvent::MTE3_V>(eventMte3V);
            }

            Duplicate(outValueUbSum, (float)0.0, outValueUbSum.GetSize());
            uint32_t dstShape[2] = {Ceil(calCElems, 32 * 8 / TRANSE_REP_STRIDE), (uint32_t)8};
            uint32_t srcShape[2] = {1, (uint32_t)8};

            BroadCast<float, 2, 0>(dupUb9, weightMaskUbTmpfp32_9, dstShape, srcShape);
            PipeBarrier<PIPE_V>();;

            Select<float, uint32_t, false>(outValueUb4, dupUbU32_9, outValueUb4, outValueUb2,
                                           SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, calCElems * (TRANSE_REP_STRIDE / 64),
                                           repParams);

            BroadCast<float, 2, 0>(dupUb6, weightMaskUbTmpfp32_6, dstShape, srcShape);

            PipeBarrier<PIPE_V>();;
            Select<float, uint32_t, false>(outValueUb4, dupUbU32_6, outValueUb4, outValueUb3,
                                           SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, calCElems * (TRANSE_REP_STRIDE / 64),
                                           repParams);

            BroadCast<float, 2, 0>(dupUb8, weightMaskUbTmpfp32_8, dstShape, srcShape);

            PipeBarrier<PIPE_V>();;
            Select<float, uint32_t, false>(outValueUb4, dupUbU32_8, outValueUb4, outValueUb,
                                           SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, calCElems * (TRANSE_REP_STRIDE / 64),
                                           repParams);

            BroadCast<float, 2, 0>(dupUb4, weightMaskUbTmpfp32_4, dstShape, srcShape);

            BroadCast<float, 2, 0>(dupUb, weightMaskUbTmpfp32, dstShape, srcShape);

            BroadCast<float, 2, 0>(dupUb3, weightMaskUbTmpfp32_3, dstShape, srcShape);

            PipeBarrier<PIPE_V>();;
            Select<float, uint32_t, false>(outValueUb2, dupUbU32, outValueUb2, outValueUb,
                                           SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, calCElems * (TRANSE_REP_STRIDE / 64),
                                           repParams);
            Select<float, uint32_t, false>(outValueUb3, dupUbU32_3, outValueUb3, outValueUb,
                                           SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, calCElems * (TRANSE_REP_STRIDE / 64),
                                           repParams);
            Select<float, uint32_t, false>(outValueUb4, dupUbU32_4, outValueUb4, (float)0.0,
                                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, calCElems * (TRANSE_REP_STRIDE / 64),
                                           repParams);
            PipeBarrier<PIPE_V>();;

            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb, outValueUb, outValueUbSum);
            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb2, outValueUb2, outValueUbSum);
            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb3, outValueUb3, outValueUbSum);
            calculateEachPointValue(nIdx, calCElems, channelAlign, loopOffset, weightUb4, outValueUb4, outValueUbSum);

            GridOutForNCHW(hwIdx, cIdx, loop_idx, calCElems, loopOffset, loopElems, outBaseOffset, weightBaseOffset,
                           outValueUbSum, output, gridOutput);
            if ((loop_idx < trans_loop - 1 && trans_loop != 1) ||
                (loop_idx == trans_loop - 1 && cIdx < channelLoop_ - 1 && trans_loop != 1) ||
                (trans_loop == 1 && cIdx < channelLoop_ - 1)) {
                SetFlag<HardEvent::MTE3_V>(eventMte3V);
            }
        }
    }
    ResetMask();
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::GridOutForNCHW(
    int16_t hwIdx, int16_t cIdx, int16_t loop_idx, int32_t calCElems, int32_t loopOffset, int16_t loopElems,
    int64_t outBaseOffset, int64_t weightBaseOffset, LocalTensor<float> outValueUbSum, LocalTensor<float> output,
    LocalTensor<float> gridOutput)
{
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    int16_t loopElemsAlign = Ceil(loopElems, 8) * 8;
    uint32_t mask = 8;
    uint64_t rsvdCnt = 0;
    LocalTensor<uint32_t> bufPattern = bufferMaskBuf_.Get<uint32_t>();
    LocalTensor<float> bufTensor = bufferBuf_.Get<float>();
    if (alignmentType_ != ALIGNMENT_TYPE_1) {
        if (loopElemsAlign != loopElems) {
            PipeBarrier<PIPE_V>();;
            for (int16_t i = 0; i < calCElems; i++) {
                GatherMask(bufTensor[i * BLOCK_NUM], outValueUbSum[i * TRANSE_REP_STRIDE + loopElems - BLOCK_NUM], bufPattern,
                           true, mask, {1, 1, 8, 8}, rsvdCnt);
            }
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte2);
        for (int16_t i = 0; i < calCElems; i++) {
            DataCopy(gridOutput[i * TRANSE_REP_STRIDE], outValueUbSum[i * TRANSE_REP_STRIDE], loopElems);
            if (loopElemsAlign != loopElems) {
                DataCopy(gridOutput[i * TRANSE_REP_STRIDE + loopElems - BLOCK_NUM], bufTensor[i * BLOCK_NUM], BLOCK_NUM);
            }
        }
    } else {
        if (loopElemsAlign != loopElems) {
            PipeBarrier<PIPE_V>();;
            for (int16_t i = 0; i < calCElems; i++) {
                Duplicate(outValueUbSum[i * TRANSE_REP_STRIDE + loopElems], 0.0f, loopElemsAlign - loopElems);
            }
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte2);
        for (int16_t i = 0; i < calCElems; i++) {
            if (loopElemsAlign == loopElems) {
                DataCopy(gridOutput[i * TRANSE_REP_STRIDE], outValueUbSum[i * TRANSE_REP_STRIDE], loopElems);
            } else {
                SetAtomicAdd<float>();
                DataCopy(gridOutput[i * TRANSE_REP_STRIDE], outValueUbSum[i * TRANSE_REP_STRIDE], loopElemsAlign);
                SetAtomicNone();
            }
        }
    }

    OutputGenerate(hwIdx, cIdx, loop_idx, calCElems, loopOffset, outBaseOffset, loopElems, weightBaseOffset, output,
                   gridOutput);
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::OutTranspose(int32_t channelAlign,
                                                                           LocalTensor<float> xLocal,
                                                                           LocalTensor<float> outValueUb)
{
    if (channelAlign == 8) {
        v4dtrans(((__ubuf__ uint32_t*)outValueUb.GetPhyAddr()), ((__ubuf__ uint32_t*)xLocal.GetPhyAddr()),
                 TRANSE_REP_STRIDE, channelAlign * 4, 1);
    } else if (channelAlign <= 64) {
        v4dtrans(((__ubuf__ uint32_t*)outValueUb.GetPhyAddr()), ((__ubuf__ uint32_t*)xLocal.GetPhyAddr()),
                 TRANSE_REP_STRIDE, channelAlign * 4, 1);
    }
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::PerLoopCompute(
    int32_t nIdx, int32_t hwIdx, int32_t calHWElems, int32_t calHWElemsAlign, LocalTensor<float> location,
    int64_t locationOffset, LocalTensor<float> output, LocalTensor<float> gridOutput, int64_t valueOffset,
    int64_t attentionOffset)
{
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);

    LocalTensor<T> gridLocal = gridQueue_.AllocTensor<T>();
    DataCopy(gridLocal, location, calHWElemsAlign * 2);  // calHWElemsAlign = 512

    gridQueue_.EnQue(gridLocal);
    gridQueue_.DeQue();

    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    LocalTensor<T> inputXYUb = inputXYFPBuf_.Get<T>();
    // 加1后，grid的datarange从-1~1到0~2
    Adds(inputXYUb, gridLocal, (float)1.0, CAL_H_W_BLOCK * 2);

    uint32_t mask = CAL_H_W_BLOCK * 2;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;

    uint8_t src0RepeatStride = 8;
    uint8_t src1RepeatStride = 8;
    PipeBarrier<PIPE_V>();;
    LocalTensor<float> inputXFpLocal = gridLocal;
    LocalTensor<float> inputYFpLocal = gridLocal[CAL_H_W_BLOCK];
    // 分别取x和y
    GatherMask(inputXFpLocal, inputXYUb, xPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(inputYFpLocal, inputXYUb, yPattern, true, mask, {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();;

    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);  // 逐bit模式

    // unnormlize处理
    Muls<float, false>(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * inputW_), MASK_PLACEHOLDER,
                       CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    Muls<float, false>(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * inputH_), MASK_PLACEHOLDER,
                       CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});

    PipeBarrier<PIPE_V>();;
    Adds<float, false>(inputXFpLocal, inputXFpLocal, (float)(-0.5), MASK_PLACEHOLDER, CAL_H_W_BLOCK * 2 / B32_MASK,
                       {1, 1, 8, 8});

    PipeBarrier<PIPE_V>();;

    // tmpOutQue_
    LocalTensor<int32_t> inputXWIntLocal = tmpOutQue_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXEIntLocal = tmpOutQue_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<int32_t> inputYWIntLocal = tmpOutQue_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4 * 2);
    LocalTensor<int32_t> inputYEIntLocal = tmpOutQue_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4 * 3);

    LocalTensor<float> inputXWFpLocal = tmpOutQue_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4 * 4);
    LocalTensor<float> inputXEFpLocal = tmpOutQue_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4 * 5);
    LocalTensor<float> inputYWFpLocal = tmpOutQue_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4 * 6);
    LocalTensor<float> inputYEFpLocal = tmpOutQue_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4 * 7);

    Cast(inputXWIntLocal, inputXFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    Cast(inputYWIntLocal, inputYFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();;
    Cast(inputXWFpLocal, inputXWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(inputYWFpLocal, inputYWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    // 分别计算左上，右上，左下，右下的坐标
    Adds<int32_t, false>(inputXEIntLocal, inputXWIntLocal, 1, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    Adds<int32_t, false>(inputYEIntLocal, inputYWIntLocal, 1, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK, {1, 1, 8, 8});
    Adds<float, false>(inputXEFpLocal, inputXWFpLocal, (float)1.0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                       {1, 1, 8, 8});
    Adds<float, false>(inputYEFpLocal, inputYWFpLocal, (float)1.0, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                       {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();;

    LocalTensor<float> nwWeightLocal = weightBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> neWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> swWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> seWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

    LocalTensor<float> weightTmpLocal = weightTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> weightTmp1Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> weightTmp2Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> weightTmp3Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

    Sub<float, false>(weightTmpLocal, inputXEFpLocal, inputXFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weightTmp1Local, inputXFpLocal, inputXWFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weightTmp2Local, inputYEFpLocal, inputYFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weightTmp3Local, inputYFpLocal, inputYWFpLocal, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});

    PipeBarrier<PIPE_V>();;
    Mul<float, false>(nwWeightLocal, weightTmpLocal, weightTmp2Local, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});
    Mul<float, false>(neWeightLocal, weightTmp1Local, weightTmp2Local, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});
    Mul<float, false>(swWeightLocal, weightTmpLocal, weightTmp3Local, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});
    Mul<float, false>(seWeightLocal, weightTmp1Local, weightTmp3Local, MASK_PLACEHOLDER, CAL_H_W_BLOCK / B32_MASK,
                      {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();;

    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> weightMaskUb2 = weightMaskBuf2_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> weightMaskUb3 = weightMaskBuf3_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<uint8_t> weightMaskUb4 = weightMaskBuf4_.Get<uint8_t>(MASK_UB_SIZE);

    // 划窗条件不满足，走兜底分支
    ClipCoordinates(inputXWFpLocal, inputYWFpLocal, inputXWIntLocal, inputYWIntLocal, coordinatesLocal, weightMaskUb, 1,
                    nIdx, hwIdx);
    ClipCoordinates(inputXEFpLocal, inputYWFpLocal, inputXEIntLocal, inputYWIntLocal, coordinatesLocal, weightMaskUb2,
                    2, nIdx, hwIdx);
    ClipCoordinates(inputXWFpLocal, inputYEFpLocal, inputXWIntLocal, inputYEIntLocal, coordinatesLocal, weightMaskUb3,
                    3, nIdx, hwIdx);
    ClipCoordinates(inputXEFpLocal, inputYEFpLocal, inputXEIntLocal, inputYEIntLocal, coordinatesLocal, weightMaskUb4,
                    4, nIdx, hwIdx);
    ResetMask();

    PointBilinear2(nIdx, hwIdx, calHWElems, calHWElemsAlign, coordinatesLocal, nwWeightLocal, neWeightLocal,
                   swWeightLocal, seWeightLocal, weightMaskUb, weightMaskUb2, weightMaskUb3, weightMaskUb4,
                   outValueLocal, true, output, gridOutput, valueOffset, attentionOffset);

    gridQueue_.FreeTensor(gridLocal);
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::GridSamplerProcess(
    LocalTensor<float> location, LocalTensor<float> output, LocalTensor<float> gridOutput, int64_t valueOffset,
    int64_t samplingOffset, int64_t attentionOffset)
{
    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t preLoopNum = blockIDX * preCoreLoop_;
    int32_t calHWElems = 0;
    int32_t calHWElemsAlign = 0;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    int32_t loopTimes = 0;
    for (int16_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        loopTimes += 1;
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calHWElems = CAL_H_W_BLOCK;
        calHWElemsAlign = CAL_H_W_BLOCK;
        if (hwIdx == preNUbLoop_ - 1) {
            calHWElems = lastLoopHW_;
            calHWElemsAlign = lastLoopHWAlign_;
        }

        int64_t locationOffset = samplingOffset + (int64_t)nIdx * gridHW_ * 2 + hwIdx * CAL_H_W_BLOCK * 2;
        InputGenerate(locationOffset, location, calHWElemsAlign);
        PerLoopCompute(nIdx, hwIdx, calHWElems, calHWElemsAlign, location, locationOffset, output, gridOutput,
                       valueOffset, attentionOffset);
    }
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::InputGenerate(int64_t locationOffset,
                                                                            LocalTensor<float> location,
                                                                            int32_t calHWElemsAlign)
{
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));

    SetFlag<HardEvent::S_MTE2>(eventIdSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMTE2);

    // cp location_tensor
    // calHWElemsAlign
    int32_t locationLength = 1024;
    int32_t copyLength = numPoints_ * 2;
    int32_t copyLengthAlign = Ceil(copyLength, 8) * 8;

    int32_t copyLoop = Ceil(locationLength, copyLengthAlign);
    for (int32_t i = 0; i < copyLoop; i++) {
        DataCopy(location[i * copyLengthAlign], locationGm_[locationOffset + i * copyLengthAlign], copyLengthAlign);
    }

    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    Muls(location, location, (float)2, locationLength);
    Adds(location, location, (float)(-1), locationLength);
    PipeBarrier<PIPE_ALL>();;
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::OutputGenerate(
    int16_t hwIdx, int16_t cIdx, int16_t loop_idx, int32_t calCElems, int64_t loopOffset, int64_t outBaseOffset,
    int16_t loopElems, int64_t weightBaseOffset, LocalTensor<float> output, LocalTensor<float> gridOutput)
{
    int64_t tmpOffset = 128 / numPoints_;
    int64_t outputGmBaseOffset = outBaseOffset + cIdx * outputH_ * CHANNEL_BLOCK + tmpOffset * loop_idx;
    int16_t loopElemsAlign = Ceil(loopElems, 8) * 8;

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

    LocalTensor<float> attentionWeight = attentionWeightsQue_.Get<float>();
    int64_t attentionWeightOffset = weightBaseOffset + loopOffset;

    DataCopy(attentionWeight, attentionWeightsGm_[attentionWeightOffset], loopElemsAlign);

    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    LocalTensor<float> tmpOut = tmpOutQue_.Get<float>();

    for (int32_t i = 0; i < 2; i++) {
        int32_t outOffset = i * 64;
        Mul(tmpOut[outOffset], gridOutput[outOffset], attentionWeight[outOffset], 64, 32, {1, 1, 1, 16, 16, 0});
    }

    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    LocalTensor<float> outTransTensor = outValueBuf2_.Get<float>();

    int32_t sumOutputLen = attentionWeightsLen / numPoints_;
    int32_t outRepeatLen = sumOutputLen / perLoopChannel_;
    int32_t outRepeatLenAlign = Ceil(outRepeatLen, 8) * 8;
    int32_t tailNum = numQueries_ % outRepeatLen;
    int32_t copyNum = Ceil(tailNum, 8) * 8;
    bool tailByteAlign = tailNum % 8 == 0 ? true : false;

    outTransCal(loopElems, outTransTensor, tmpOut, sumOutputLen, outRepeatLen, tailNum, copyNum, tailByteAlign);

    bool isDataCopy = numQueries_ % outRepeatLen == 0;
    int32_t copyDstStride = 0;
    int32_t copySrcStride = 0;

    if (loopElems == 128) {
        copyDstStride = (numQueries_ - outRepeatLen) / 8;
    } else {
        copySrcStride = Ceil(CHANNEL_BLOCK - Ceil(loopElems, numPoints_), 8);
        copyDstStride = (numQueries_ - tailNum) / 8;
    }

    outTransCopyOut(isDataCopy, copyDstStride, copySrcStride, outputGmBaseOffset, loopElems,
        outTransTensor, outRepeatLen, tailNum, outRepeatLenAlign, copyNum);
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::outTransCal(
    int16_t loopElems, LocalTensor<float> outTransTensor, LocalTensor<float> tmpOut,
    int32_t sumOutputLen, int32_t outRepeatLen, int32_t tailNum, int32_t copyNum, bool tailByteAlign)
{
    if (loopElems == 128) {
        // sumOutputLen * numPoints_ 转置成 numPoints_ * sumOutputLen
        v4dtrans(((__ubuf__ uint32_t*)outTransTensor.GetPhyAddr()), ((__ubuf__ uint32_t*)tmpOut.GetPhyAddr()),
                 sumOutputLen, numPoints_, 1);
        PipeBarrier<PIPE_V>();;
        for (int32_t i = 1; i < numPoints_; i++) {
            Add(outTransTensor, outTransTensor, outTransTensor[sumOutputLen * i], sumOutputLen);
            PipeBarrier<PIPE_V>();;
        }
    } else {
        for (int32_t i = 0; i < sumOutputLen; i++) {
            float sumValue = 0;
            for (int32_t j = 0; j < numPoints_; j++) {
                sumValue += tmpOut.GetValue(i * numPoints_ + j);
            }
            outTransTensor.SetValue(i, sumValue);
        }
        if (!tailByteAlign) {
            for (int32_t i = 0; i < sumOutputLen; i += outRepeatLen) {
                for (int32_t j = 0; j < copyNum - tailNum; j++) {
                    outTransTensor.SetValue(i + j + tailNum, 0);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::outTransCopyOut(
    bool isDataCopy, int32_t copyDstStride, int32_t copySrcStride, int64_t outputGmBaseOffset,
    int16_t loopElems, LocalTensor<float> outTransTensor, int32_t outRepeatLen, int32_t tailNum,
    int32_t outRepeatLenAlign, int32_t copyNum)
{
    event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);

    if (isDataCopy && numPoints_ == 4) {
        DataCopyParams copyParams = {static_cast<uint16_t>(calCElemsAlign), 0, 0, 0};
        copyParams.blockLen = Ceil(Ceil(loopElems, numPoints_), 8);
        if (loopElems != 128) {
            copyParams.srcStride = static_cast<uint16_t>(copySrcStride);
        }
        copyParams.dstStride = static_cast<uint16_t>(copyDstStride);

        SetAtomicAdd<T>();
        DataCopy(outputGm_[outputGmBaseOffset], outTransTensor, copyParams);
        SetAtomicNone();
    } else {
        for (int32_t i = 0; i < perLoopChannel_; i++) {
            if (loopElems == 128) {
                if (outRepeatLen == outRepeatLenAlign) {
                    SetAtomicAdd<T>();
                    DataCopy(outputGm_[outputGmBaseOffset + i * numQueries_], outTransTensor[i * outRepeatLenAlign],
                             outRepeatLenAlign);
                    SetAtomicNone();
                } else {
                    // outRepeatLen没有32对齐处理：临时存储outTransTensor中一行的值,补0 32对齐后再拷入
                    PipeBarrier<PIPE_V>();;
                    LocalTensor<float> tmpOutTransTensor = workLocalQue_.Get<float>();
                    Duplicate(tmpOutTransTensor, 0.0f, tmpOutTransTensor.GetSize());
                    PipeBarrier<PIPE_V>();;
                    Add(tmpOutTransTensor, tmpOutTransTensor, outTransTensor[i * outRepeatLen], outRepeatLen);
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

                    SetAtomicAdd<T>();
                    DataCopy(outputGm_[outputGmBaseOffset + i * numQueries_], tmpOutTransTensor, outRepeatLenAlign);
                    SetAtomicNone();
                }
            } else {
                SetAtomicAdd<T>();
                DataCopy(outputGm_[outputGmBaseOffset + i * numQueries_], outTransTensor[i * outRepeatLenAlign],
                         copyNum);
                SetAtomicNone();
            }
        }
    }
}

template <typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn310P<T>::MSDAProcess()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    LocalTensor<float> location = locationQue_.Get<float>();
    LocalTensor<float> output = outputQue_.Get<float>();
    LocalTensor<float> gridOutput = gridOutQue_.Get<float>();

    int64_t valueOffset = 0;
    int64_t samplingOffset = 0;
    int64_t attentionOffset = 0;

    int64_t samplingNumPerLevel = batchSize_ * numHeads_ * numQueries_ * numPoints_ * 2;
    int64_t attentionNumPerLevel = batchSize_ * numHeads_ * numQueries_ * numPoints_;

    for (uint32_t i = 0; i < realLevels_; i++) {
        currentLevel = i;
        inputH_ = valueSpatialShapesGm_.GetValue(2 * i);
        inputW_ = valueSpatialShapesGm_.GetValue(2 * i + 1);

        GridSamplerProcess(location, output, gridOutput, valueOffset, samplingOffset, attentionOffset);

        valueNumPerLevel = inputH_ * inputW_ * embedDims_;
        valueOffset += valueNumPerLevel;

        samplingOffset += samplingNumPerLevel;
        attentionOffset += attentionNumPerLevel;
    }
}

}  // namespace MultiScaleDeformableAttn
#endif  // MS_DEFORM_ATTN_310P
