/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_lstm_backward.cpp
 * \brief
 */
#include "aclnn_lstm_backward.h"
#include "single_layer_lstm_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include"aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/greater.h"
#include "aclnn_kernels/cast.h"
#include "level0/concat.h"
#include "level0/arange.h"
#include "aclnn_kernels/reshape.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/slice.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transpose.h"
#include "level0/add.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
// 通用维度常量
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;

// LSTM特定常量
constexpr int64_t GATE_COUNT = 4;  // i, j, f, o 四个门
constexpr int64_t SINGLE_DIRECTION = 1;
constexpr int64_t BI_DIRECTION = 2;
constexpr int64_t HC_TENSOR_COUNT = 2;  // h和c两个张量
constexpr int64_t REDUCE_DIM = 1;
constexpr int64_t CONCAT_DIM_HIDDEN = 2;
constexpr int64_t CONCAT_DIM_LAYER = 0;
constexpr int64_t WEIGHT_INPUT_INDEX = 0;
constexpr int64_t WEIGHT_HIDDEN_INDEX = 1;
constexpr int64_t BIAS_INPUT_INDEX = 2;
constexpr int64_t BIAS_HIDDEN_INDEX = 3;
constexpr int64_t SEQUENCE_DIM = 0;
constexpr int64_t BATCH_DIM = 1;
constexpr int64_t HIDDEN_DIM = 2;
constexpr int64_t OUT_NUM = 5;

// 函数返回结果索引
constexpr int64_t RESULT_WEIGHT_GRAD_INDEX = 0;
constexpr int64_t RESULT_BIAS_GRAD_INDEX = 1;
constexpr int64_t RESULT_INPUT_GRAD_INDEX = 2;
constexpr int64_t RESULT_HIDDEN_GRAD_INDEX = 3;
constexpr int64_t RESULT_CELL_GRAD_INDEX = 4;

// 切片张量数组索引
constexpr int64_t SLICE_INIT_H_INDEX = 0;
constexpr int64_t SLICE_INIT_C_INDEX = 1;
constexpr int64_t SLICE_DH_INDEX = 2;
constexpr int64_t SLICE_DC_INDEX = 3;

// 双向张量数组索引
constexpr int64_t BIDIR_INIT_H_INDEX = 0;
constexpr int64_t BIDIR_INIT_C_INDEX = 1;
constexpr int64_t BIDIR_DH_INDEX = 2;
constexpr int64_t BIDIR_DC_INDEX = 3;
constexpr int64_t BIDIR_DY_INDEX = 4;

// 参数索引计算
constexpr int64_t NUM_NO_B_NO_BIDIR = 2;
constexpr int64_t NUM_WITH_B_OR_BID = 4;
constexpr int64_t NUM_WITH_B_AND_BID = 8;

// 门索引
constexpr int64_t GATE_I_INDEX = 0;
constexpr int64_t GATE_J_INDEX = 1;
constexpr int64_t GATE_F_INDEX = 2;
constexpr int64_t GATE_O_INDEX = 3;

// 函数返回索引
constexpr int64_t INDEX_ZERO = 0;
constexpr int64_t INDEX_ONE = 1;
constexpr int64_t INDEX_TWO = 2;
constexpr int64_t INDEX_THREE = 3;
constexpr int64_t INDEX_FOUR = 4;

// 拼接最大数量
constexpr size_t CONCAT_MAX_NUM = 32;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
}

static const std::initializer_list<op::DataType> BATCH_SIZES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64};

struct LSTMContinuousTensors {
    const aclTensor* inputContiguous = nullptr;
    const aclTensorList* hxContiguous = nullptr;
    const aclTensorList* paramsContiguous = nullptr;
    const aclTensor* dyContiguous = nullptr;
    const aclTensor* dhContiguous = nullptr;
    const aclTensor* dcContiguous = nullptr;
    const aclTensorList* iContiguous = nullptr;
    const aclTensorList* gContiguous = nullptr;
    const aclTensorList* fContiguous = nullptr;
    const aclTensorList* oContiguous = nullptr;
    const aclTensorList* hContiguous = nullptr;
    const aclTensorList* cContiguous = nullptr;
    const aclTensorList* tanhcContiguous = nullptr;
    const aclTensor* batchSizesContiguous = nullptr;
};

static const aclTensor* SplitToConcat(std::vector<const aclTensor*> tensorListA, int64_t dim, aclOpExecutor* executor)
{
    if (tensorListA.size() == 1) {
        return tensorListA[0];
    }

    while (tensorListA.size() > 1) {
        std::vector<const aclTensor*> tensorListOnce;
        std::vector<const aclTensor*> tensorListB;
        for (auto tensor : tensorListA) {
            tensorListOnce.emplace_back(tensor);
            if (tensorListOnce.size() == CONCAT_MAX_NUM) {
                auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensor = l0op::ConcatD(tensorList, dim, executor);
                CHECK_RET(concatTensor != nullptr, nullptr);
                tensorListB.emplace_back(concatTensor);
                tensorListOnce.clear();
            }
        }
        if (!tensorListOnce.empty()) {
            if (tensorListOnce.size() == 1) {
                tensorListB.emplace_back(tensorListOnce.front());
            } else {
                auto aclTensorListTail = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensorTail = l0op::ConcatD(aclTensorListTail, dim, executor);
                CHECK_RET(concatTensorTail != nullptr, nullptr);
                tensorListB.emplace_back(concatTensorTail);
            }
            tensorListOnce.clear();
        }
        tensorListA = tensorListB;
    }

    CHECK_RET (!tensorListA.empty(), nullptr);
    return tensorListA.front();
}

static const aclTensor* GetMask(const aclTensor *input, const aclTensor *batchSizes, const aclTensor *h,
    aclOpExecutor *executor)
{
    auto inputShape = input->GetViewShape();
    auto hShape = h->GetViewShape();
    auto inputDtype = input->GetDataType();
    auto batchSize = inputShape[BATCH_DIM];
    auto timeStep = inputShape[SEQUENCE_DIM];
    auto start = executor->AllocScalar(0);
    auto end = executor->AllocScalar(batchSize);
    auto step = executor->AllocScalar(1);
    gert::Shape arrangeShape;
    arrangeShape.AppendDim(batchSize);
    auto arrangeSizeTensor = executor->AllocTensor(arrangeShape, op::DataType::DT_INT64, Format::FORMAT_ND);
    CHECK_RET(arrangeSizeTensor != nullptr, nullptr);
    auto arrangeTensor = l0op::Arange(start, end, step, arrangeSizeTensor, false, executor);
    CHECK_RET(arrangeTensor != nullptr, nullptr);
    FVector<int64_t> broadTVector{timeStep, batchSize};
    aclIntArray* broadTArray = executor->AllocIntArray(broadTVector.data(), DIM_TWO);
    CHECK_RET(broadTArray != nullptr, nullptr);
    auto arrangeReshapeTensor = l0op::UnsqueezeNd(arrangeTensor, DIM_ZERO, executor);
    CHECK_RET(arrangeReshapeTensor != nullptr, nullptr);
    auto arrangeBroadTTensor = l0op::BroadcastTo(arrangeReshapeTensor, broadTArray, executor);
    CHECK_RET(arrangeBroadTTensor != nullptr, nullptr);

    auto batchSizesReshapeTensor = l0op::UnsqueezeNd(batchSizes, DIM_ONE, executor);
    CHECK_RET(batchSizesReshapeTensor != nullptr, nullptr);
    auto batchSizesBroadTTensor = l0op::BroadcastTo(batchSizesReshapeTensor, broadTArray, executor);
    CHECK_RET(batchSizesBroadTTensor != nullptr, nullptr);

    const aclTensor* mask = l0op::Greater(batchSizesBroadTTensor, arrangeBroadTTensor, executor);
    CHECK_RET(mask != nullptr, nullptr);
    auto seqLengthWithoutHidden = l0op::Cast(mask, inputDtype, executor);
    CHECK_RET(seqLengthWithoutHidden != nullptr, nullptr);

    FVector<int64_t> broadVector{timeStep, batchSize, hShape[HIDDEN_DIM]};
    aclIntArray* broadArray = executor->AllocIntArray(broadVector.data(), DIM_THREE);
    CHECK_RET(broadArray != nullptr, nullptr);
    auto seqLengthReshape = l0op::UnsqueezeNd(seqLengthWithoutHidden, DIM_TWO, executor);
    CHECK_RET(seqLengthReshape != nullptr, nullptr);
    auto seqlength = l0op::BroadcastTo(seqLengthReshape, broadArray, executor);
    CHECK_RET(seqlength != nullptr, nullptr);
    return seqlength;
}

static std::array<const aclTensor *, OUT_NUM> ExecLstmBackward (
    const aclTensor *input,
    const aclTensor *initH,
    const aclTensor *initC, 
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *dy, 
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensor *seqLength,
    const aclTensor *i, 
    const aclTensor *j,
    const aclTensor *f, 
    const aclTensor *o, 
    const aclTensor*h, 
    const aclTensor *c, 
    const aclTensor *tanhc,
    aclOpExecutor *executor,
    bool flagDirection=false)
{
    const char *direction = flagDirection ? "REDIRECTIONAL" : "UNIDIRECTIONAL";
    const char *gateOrder = "ifjo";
    std::array<const aclTensor *, OUT_NUM> nullptrRes{nullptr, nullptr, nullptr, nullptr, nullptr};
    auto result = l0op::SingleLayerLstmGrad(input, weight, bias, nullptr, initH, initC, h, c, dy, dh, dc, i, j, f, o,
        tanhc, seqLength, direction, gateOrder, executor);
    CHECK_RET(result[RESULT_WEIGHT_GRAD_INDEX] != nullptr &&
              result[RESULT_BIAS_GRAD_INDEX] != nullptr &&
              result[RESULT_INPUT_GRAD_INDEX] != nullptr &&
              result[RESULT_HIDDEN_GRAD_INDEX] != nullptr &&
              result[RESULT_CELL_GRAD_INDEX] != nullptr, nullptrRes);
    return result;
}

static FVector<const aclTensor *> GetWeightBiasFromParams(const aclTensorList *params, bool hasBias, bool bidirectional,
    int64_t paramNumPerLayer, int64_t layerIdx, aclOpExecutor *executor)
{
    FVector<const aclTensor *> result{};
    FVector<const aclTensor *> nullptrRes{};
    int64_t layerOffset = (layerIdx - 1) * paramNumPerLayer;
    
    // 前向权重
    FVector<const aclTensor *> weightForwardVector = {(*params)[layerOffset + WEIGHT_INPUT_INDEX], 
                                                     (*params)[layerOffset + WEIGHT_HIDDEN_INDEX]};
    const aclTensorList* weightForwardList = executor->AllocTensorList(weightForwardVector.data(),
                                                                       weightForwardVector.size());
    CHECK_RET(weightForwardList != nullptr, nullptrRes);
    auto weightForward = l0op::ConcatD(weightForwardList, 1, executor);
    CHECK_RET(weightForward != nullptr, nullptrRes);
    result.emplace_back(weightForward);
    
    if (hasBias && bidirectional) {
        auto biasForward = l0op::Add((*params)[layerOffset + BIAS_INPUT_INDEX], 
                                    (*params)[layerOffset + BIAS_HIDDEN_INDEX], executor);
        result.emplace_back(biasForward);
        // 后向权重
        FVector<const aclTensor *> weightBackwardVector = {
            (*params)[layerOffset + NUM_WITH_B_AND_BID / BI_DIRECTION + WEIGHT_INPUT_INDEX], 
            (*params)[layerOffset + NUM_WITH_B_AND_BID / BI_DIRECTION + WEIGHT_HIDDEN_INDEX]};
        const aclTensorList* weightBackwardList = executor->AllocTensorList(weightBackwardVector.data(),
                                                                            weightBackwardVector.size());
        CHECK_RET(weightBackwardList != nullptr, nullptrRes);
        auto weightBackward = l0op::ConcatD(weightBackwardList, REDUCE_DIM, executor);
        CHECK_RET(weightBackward != nullptr, nullptrRes);
        result.emplace_back(weightBackward);

        auto biasBackward = l0op::Add(
            (*params)[layerOffset + NUM_WITH_B_AND_BID / BI_DIRECTION + BIAS_INPUT_INDEX], 
            (*params)[layerOffset + NUM_WITH_B_AND_BID / BI_DIRECTION + BIAS_HIDDEN_INDEX], executor);
        CHECK_RET(biasBackward != nullptr, nullptrRes);
        result.emplace_back(biasBackward);
    } else if (!hasBias && bidirectional) {
        FVector<const aclTensor *> weightBackwardVector = {
            (*params)[layerOffset + NUM_WITH_B_OR_BID / BI_DIRECTION + WEIGHT_INPUT_INDEX], 
            (*params)[layerOffset + NUM_WITH_B_OR_BID / BI_DIRECTION + WEIGHT_HIDDEN_INDEX]};
        const aclTensorList* weightBackwardList = executor->AllocTensorList(weightBackwardVector.data(),
                                                                            weightBackwardVector.size());
        CHECK_RET(weightBackwardList != nullptr, nullptrRes);
        auto weightBackward = l0op::ConcatD(weightBackwardList, REDUCE_DIM, executor);
        CHECK_RET(weightBackward != nullptr, nullptrRes);
        const aclTensor* emptyTensor = nullptr;
        result.emplace_back(emptyTensor);
        result.emplace_back(weightBackward);
        result.emplace_back(emptyTensor);
    } else if (hasBias && !bidirectional) {
        auto biasForward = l0op::Add((*params)[layerOffset + BIAS_INPUT_INDEX], 
                                    (*params)[layerOffset + BIAS_HIDDEN_INDEX], executor);
        CHECK_RET(biasForward != nullptr, nullptrRes);
        result.emplace_back(biasForward);
    } else {
        const aclTensor *biasForward = nullptr;
        result.emplace_back(biasForward);
    }
    return result;
}

static std::array<const aclTensor*, HC_TENSOR_COUNT * 2> CreateSliceTensors(
    int64_t layerIndex,
    const aclTensor* initHMultiLayer,
    const aclTensor* initCMultiLayer,
    const aclTensor* dh,
    const aclTensor* dc,
    aclOpExecutor* executor)
{
    std::array<const aclTensor*, HC_TENSOR_COUNT * 2> result{nullptr, nullptr, nullptr, nullptr};
    auto hShape = initHMultiLayer->GetViewShape();
    // 创建偏移量数组
    FVector<int64_t> offsetVector{layerIndex, 0, 0};
    aclIntArray* offsetArray = executor->AllocIntArray(offsetVector.data(), offsetVector.size());
    CHECK_RET(offsetArray != nullptr, result);

    // 创建大小数组
    FVector<int64_t> sizeVector{1, hShape[BATCH_DIM], hShape[HIDDEN_DIM]};
    aclIntArray* sizeArray = executor->AllocIntArray(sizeVector.data(), sizeVector.size());
    CHECK_RET(sizeArray != nullptr, result);
    
    // 创建四个切片张量
    result[SLICE_INIT_H_INDEX] = l0op::Slice(initHMultiLayer, offsetArray, sizeArray, executor);
    result[SLICE_INIT_C_INDEX] = l0op::Slice(initCMultiLayer, offsetArray, sizeArray, executor);
    result[SLICE_DH_INDEX] = l0op::Slice(dh, offsetArray, sizeArray, executor);
    result[SLICE_DC_INDEX] = l0op::Slice(dc, offsetArray, sizeArray, executor);
    return result;
}

static std::tuple<const aclTensor*, std::vector<const aclTensor *>, std::vector<const aclTensor *>,
                  std::vector<const aclTensor *>, std::vector<const aclTensor *>> LstmBackwardSingleLayerDirec(
    const aclTensor *input,
    const aclTensor *initHMultiLayer,
    const aclTensor *initCMultiLayer,
    const aclTensorList *params,
    const aclTensor *dy, 
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensor *seqLength,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    int64_t layersTemp,
    int64_t numLayers,
    bool hasBias,
    int64_t paramNumPerLayer,
    aclOpExecutor *executor)
{
    auto inputCur = layersTemp == 1 ? input : (*h)[layersTemp - BI_DIRECTION];
    auto nullptrRes = std::make_tuple(
        nullptr,
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>()
    );
    const aclTensor *initHCur = initHMultiLayer;
    const aclTensor *initCCur = initCMultiLayer;
    const aclTensor *dhCur = dh;
    const aclTensor *dcCur = dc;
    auto weightBias = GetWeightBiasFromParams(params, hasBias, false, paramNumPerLayer, layersTemp, executor);
    const aclTensor *weightCur = weightBias[0];
    const aclTensor *biasCur = weightBias[1];
    CHECK_RET(weightCur != nullptr, nullptrRes);
    if (hasBias) {
        CHECK_RET(biasCur != nullptr, nullptrRes);
    }

    if (numLayers > 1) {
        auto sliceTensors = CreateSliceTensors(layersTemp - 1, initHMultiLayer, initCMultiLayer, dh, dc, executor);
        CHECK_RET (sliceTensors[SLICE_INIT_H_INDEX] != nullptr &&
                  sliceTensors[SLICE_INIT_C_INDEX] != nullptr &&
                  sliceTensors[SLICE_DH_INDEX] != nullptr &&
                  sliceTensors[SLICE_DC_INDEX] != nullptr, nullptrRes);
        initHCur = sliceTensors[SLICE_INIT_H_INDEX];
        initCCur = sliceTensors[SLICE_INIT_C_INDEX];
        dhCur = sliceTensors[SLICE_DH_INDEX];
        dcCur = sliceTensors[SLICE_DC_INDEX];
    }
    auto result = ExecLstmBackward(inputCur, initHCur, initCCur, weightCur, biasCur, dy, dhCur, dcCur, seqLength,
        (*i)[layersTemp - 1], (*j)[layersTemp - 1], (*f)[layersTemp - 1], (*o)[layersTemp - 1], (*h)[layersTemp - 1],
        (*c)[layersTemp - 1], (*tanhc)[layersTemp - 1], executor, false);    
    std::vector<const aclTensor *> dwVector = {result[RESULT_WEIGHT_GRAD_INDEX]};
    std::vector<const aclTensor *> dbVector = {result[RESULT_BIAS_GRAD_INDEX]};
    std::vector<const aclTensor *> dhPrevVector = {result[RESULT_HIDDEN_GRAD_INDEX]};
    std::vector<const aclTensor *> dcPrevVector = {result[RESULT_CELL_GRAD_INDEX]};
    return std::tie(result[RESULT_INPUT_GRAD_INDEX], dhPrevVector, dcPrevVector, dwVector, dbVector);
}

static bool CheckTupleNotNull(const std::tuple<const aclTensor*, std::vector<const aclTensor *>,
    std::vector<const aclTensor *>, std::vector<const aclTensor *>, std::vector<const aclTensor *>>& tuple)
{
    if (!std::get<0>(tuple)) {
        return false;
    }
    const auto& dhVec = std::get<INDEX_ONE>(tuple);
    const auto& dcVec = std::get<INDEX_TWO>(tuple);
    const auto& dwVec = std::get<INDEX_THREE>(tuple);
    const auto& dbVec = std::get<INDEX_FOUR>(tuple);
    return std::all_of(dwVec.begin(), dwVec.end(), [](auto* ptr) { return ptr != nullptr; }) &&
           std::all_of(dbVec.begin(), dbVec.end(), [](auto* ptr) { return ptr != nullptr; }) &&
           std::all_of(dhVec.begin(), dhVec.end(), [](auto* ptr) { return ptr != nullptr; }) &&
           std::all_of(dcVec.begin(), dcVec.end(), [](auto* ptr) { return ptr != nullptr; });
}

static std::tuple<const aclTensor*, std::vector<const aclTensor *>, std::vector<const aclTensor *>,
    std::vector<const aclTensor *>, std::vector<const aclTensor *>> LstmBackwardMultiLayerDirec (
    const aclTensor *input,
    const aclTensor *initHMultiLayer,
    const aclTensor *initCMultiLayer,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensor *seqLength,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    int64_t numLayers,
    int64_t layersTemp,
    bool hasBias,
    int64_t paramNumPerLayer,
    aclOpExecutor *executor)
{
    const aclTensor* lastLayerDy = nullptr;
    std::vector<const aclTensor *> lastLayerDhPrevVector{};
    std::vector<const aclTensor *> lastLayerDcPrevVector{};
    std::vector<const aclTensor *> lastLayerDwVector{};
    std::vector<const aclTensor *> lastLayerDbVector{};
    auto nullptrRes = std::make_tuple(nullptr, std::vector<const aclTensor*>(), std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(), std::vector<const aclTensor*>());
    auto hShape = initHMultiLayer->GetViewShape();
    CHECK_RET (hShape.GetDimNum() == DIM_THREE, nullptrRes);

    if (layersTemp == numLayers) {
        auto lastLayerOutput = LstmBackwardSingleLayerDirec(input, initHMultiLayer, initCMultiLayer, params, dy, dh,
            dc, seqLength, i, j, f, o, h, c, tanhc, layersTemp, numLayers, hasBias, paramNumPerLayer, executor);
        CHECK_RET(CheckTupleNotNull(lastLayerOutput), nullptrRes);
        return lastLayerOutput;
    } else {
        std::tie(lastLayerDy, lastLayerDhPrevVector, lastLayerDcPrevVector, lastLayerDwVector, lastLayerDbVector) = 
            LstmBackwardMultiLayerDirec(input, initHMultiLayer, initCMultiLayer, params, dy, dh, dc, seqLength, i, j, f,
                o, h, c, tanhc, numLayers, layersTemp + 1, hasBias, paramNumPerLayer, executor);
        CHECK_RET(CheckTupleNotNull(std::tie(lastLayerDy, lastLayerDhPrevVector, lastLayerDcPrevVector,
                lastLayerDwVector, lastLayerDbVector)), nullptrRes);
    }
    auto weightBias = GetWeightBiasFromParams(params, hasBias, false, paramNumPerLayer, layersTemp, executor);
    const aclTensor *weightCur = weightBias[0];
    const aclTensor *biasCur = weightBias[1];
    CHECK_RET(weightCur != nullptr, nullptrRes);
    if (hasBias) {
        CHECK_RET(biasCur != nullptr, nullptrRes);
    }

    auto sliceTensors = CreateSliceTensors(layersTemp - 1, initHMultiLayer, initCMultiLayer, dh, dc, executor);

    CHECK_RET (sliceTensors[SLICE_INIT_H_INDEX] != nullptr &&
              sliceTensors[SLICE_INIT_C_INDEX] != nullptr &&
              sliceTensors[SLICE_DH_INDEX] != nullptr &&
              sliceTensors[SLICE_DC_INDEX] != nullptr, nullptrRes);

    auto inputCur = layersTemp == 1 ? input : (*h)[layersTemp - BI_DIRECTION];
    auto result = ExecLstmBackward(inputCur, sliceTensors[SLICE_INIT_H_INDEX], sliceTensors[SLICE_INIT_C_INDEX],
        weightCur, biasCur, lastLayerDy, sliceTensors[SLICE_DH_INDEX], sliceTensors[SLICE_DC_INDEX], seqLength, 
        (*i)[layersTemp - 1], (*j)[layersTemp - 1], (*f)[layersTemp - 1], (*o)[layersTemp - 1], 
        (*h)[layersTemp - 1], (*c)[layersTemp - 1], (*tanhc)[layersTemp - 1], executor, false);

    lastLayerDwVector.emplace_back(result[RESULT_WEIGHT_GRAD_INDEX]);
    lastLayerDbVector.emplace_back(result[RESULT_BIAS_GRAD_INDEX]);
    lastLayerDhPrevVector.emplace_back(result[RESULT_HIDDEN_GRAD_INDEX]);
    lastLayerDcPrevVector.emplace_back(result[RESULT_CELL_GRAD_INDEX]);
    return std::tie(result[RESULT_INPUT_GRAD_INDEX], lastLayerDhPrevVector, lastLayerDcPrevVector, lastLayerDwVector,
                    lastLayerDbVector);
}

static std::array<const aclTensor*, 5> CreateBidDirectionTensors(const aclTensor* initHMultiLayer,
    const aclTensor* initCMultiLayer, const aclTensor* dh, const aclTensor* dc, const aclTensor* dy,
    int64_t layersTemp, aclOpExecutor* executor, bool isBackward)
{
    // 定义统一的错误返回值
    const std::array<const aclTensor*, 5> nullptrRes = {nullptr, nullptr, nullptr, nullptr, nullptr};
    auto hShape = initHMultiLayer->GetViewShape();
    CHECK_RET (hShape.GetDimNum() == DIM_THREE, nullptrRes);
    auto dyShape = dy->GetViewShape();
    CHECK_RET (dyShape.GetDimNum() == DIM_THREE, nullptrRes);

    // 根据方向计算偏移量
    int64_t initOffset = isBackward ? (BI_DIRECTION * layersTemp - 1) : (BI_DIRECTION * layersTemp - BI_DIRECTION);
    int64_t dyStart = isBackward ? (dyShape[HIDDEN_DIM] / BI_DIRECTION) : 0;

    // 创建初始隐藏状态和细胞状态
    FVector<int64_t> offsetVectorInit{initOffset, DIM_ZERO, DIM_ZERO};
    aclIntArray* offsetArrayInit = executor->AllocIntArray(offsetVectorInit.data(), offsetVectorInit.size());
    CHECK_RET(offsetArrayInit != nullptr, nullptrRes);
    
    FVector<int64_t> sizeVectorInit{1, hShape[BATCH_DIM], hShape[HIDDEN_DIM]};
    aclIntArray* sizeArrayInit = executor->AllocIntArray(sizeVectorInit.data(), sizeVectorInit.size());
    CHECK_RET(sizeArrayInit != nullptr, nullptrRes);
    
    auto initH = l0op::Slice(initHMultiLayer, offsetArrayInit, sizeArrayInit, executor);
    CHECK_RET(initH != nullptr, nullptrRes);
    
    auto initC = l0op::Slice(initCMultiLayer, offsetArrayInit, sizeArrayInit, executor);
    CHECK_RET(initC != nullptr, nullptrRes);
    
    auto dhDir = l0op::Slice(dh, offsetArrayInit, sizeArrayInit, executor);
    CHECK_RET(dhDir != nullptr, nullptrRes);
    
    auto dcDir = l0op::Slice(dc, offsetArrayInit, sizeArrayInit, executor);
    CHECK_RET(dcDir != nullptr, nullptrRes);

    // 创建输出梯度
    FVector<int64_t> offsetVectorDy{DIM_ZERO, DIM_ZERO, dyStart};
    aclIntArray* offsetArrayDy = executor->AllocIntArray(offsetVectorDy.data(), offsetVectorDy.size());
    CHECK_RET(offsetArrayDy != nullptr, nullptrRes);
    
    FVector<int64_t> sizeVectorDy{dyShape[SEQUENCE_DIM], dyShape[BATCH_DIM], dyShape[HIDDEN_DIM] / BI_DIRECTION};
    aclIntArray* sizeArrayDy = executor->AllocIntArray(sizeVectorDy.data(), sizeVectorDy.size());
    CHECK_RET(sizeArrayDy != nullptr, nullptrRes);
    
    auto dyDir = l0op::Slice(dy, offsetArrayDy, sizeArrayDy, executor);
    CHECK_RET(dyDir != nullptr, nullptrRes);

    return {initH, initC, dhDir, dcDir, dyDir};
}

static const aclTensor* CreateInputTensor(const aclTensor* input, const aclTensorList* h,
                                          int64_t layersTemp, aclOpExecutor* executor)
{
    if (layersTemp == 1) {
        return input;
    } else {
        auto inputCurForward = (*h)[BI_DIRECTION * (layersTemp - BI_DIRECTION)];
        CHECK_RET(inputCurForward != nullptr, nullptr);
        
        auto inputCurBackward = (*h)[BI_DIRECTION * (layersTemp - BI_DIRECTION) + 1];
        CHECK_RET(inputCurBackward != nullptr, nullptr);
        
        op::FVector<const aclTensor*> inputCurVector = {inputCurForward, inputCurBackward};
        const aclTensorList* inputCurList = executor->AllocTensorList(inputCurVector.data(), inputCurVector.size());
        CHECK_RET(inputCurList != nullptr, nullptr);
        
        auto result = l0op::ConcatD(inputCurList, CONCAT_DIM_HIDDEN, executor);
        CHECK_RET(result != nullptr, nullptr);
        
        return result;
    }
}

static std::tuple<const aclTensor*, std::vector<const aclTensor *>, std::vector<const aclTensor *>,
                  std::vector<const aclTensor *>, std::vector<const aclTensor *>> MergeResults (
    std::array<const aclTensor*, 5> resultForward, std::array<const aclTensor*, 5> resultBackward,
    std::vector<const aclTensor*>& dhVector, std::vector<const aclTensor*>& dcVector,
    std::vector<const aclTensor*>& dwVector, std::vector<const aclTensor*>& dbVector, aclOpExecutor* executor)
{
    auto nullptrRes = std::make_tuple(
        nullptr,
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>()
    );
    std::vector<const aclTensor*> catDhPrevVector = {
        resultForward[RESULT_HIDDEN_GRAD_INDEX], resultBackward[RESULT_HIDDEN_GRAD_INDEX]};
    std::vector<const aclTensor*> catDcPrevVector = {
        resultForward[RESULT_CELL_GRAD_INDEX], resultBackward[RESULT_CELL_GRAD_INDEX]};
    dwVector.emplace_back(resultForward[RESULT_WEIGHT_GRAD_INDEX]);
    dwVector.emplace_back(resultBackward[RESULT_WEIGHT_GRAD_INDEX]);
    dbVector.emplace_back(resultForward[RESULT_BIAS_GRAD_INDEX]);
    dbVector.emplace_back(resultBackward[RESULT_BIAS_GRAD_INDEX]);
    dhVector.emplace_back(resultForward[RESULT_HIDDEN_GRAD_INDEX]);
    dhVector.emplace_back(resultBackward[RESULT_HIDDEN_GRAD_INDEX]);
    dcVector.emplace_back(resultForward[RESULT_CELL_GRAD_INDEX]);
    dcVector.emplace_back(resultBackward[RESULT_CELL_GRAD_INDEX]);

    auto dx = l0op::Add(resultForward[RESULT_INPUT_GRAD_INDEX], resultBackward[RESULT_INPUT_GRAD_INDEX], executor);
    CHECK_RET(dx != nullptr, nullptrRes);
    return std::make_tuple(dx, dhVector, dcVector, dwVector, dbVector);
}

static std::tuple<const aclTensor*, std::vector<const aclTensor *>, std::vector<const aclTensor *>,
                  std::vector<const aclTensor *>, std::vector<const aclTensor *>> LstmBackwardSingleLayerBidirec(
    const aclTensor *input,
    const aclTensor *initHMultiLayer,
    const aclTensor *initCMultiLayer,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensor *seqLength,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    int64_t layersTemp,
    bool hasBias,
    int64_t paramNumPerLayer,
    aclOpExecutor *executor)
{
    auto nullptrRes = std::make_tuple(
        nullptr,
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>()
    );
    
    const aclTensor* inputCur = nullptr;
    auto weightBias = GetWeightBiasFromParams(params, hasBias, true, paramNumPerLayer, layersTemp, executor);
    const aclTensor *weightForwardCur = weightBias[0];
    CHECK_RET(weightForwardCur != nullptr, nullptrRes);
    const aclTensor *weightBackwardCur = weightBias[INDEX_TWO];
    CHECK_RET(weightBackwardCur != nullptr, nullptrRes);
    const aclTensor *biasForwardCur = weightBias[1];
    const aclTensor *biasBackwardCur = weightBias[INDEX_THREE];
    if (hasBias) {
        CHECK_RET(biasForwardCur != nullptr, nullptrRes);
        CHECK_RET(biasBackwardCur != nullptr, nullptrRes);
    }

    auto forwardTensors = CreateBidDirectionTensors(initHMultiLayer, initCMultiLayer, dh, dc, dy, layersTemp, executor,
                                                    false);
    CHECK_RET(forwardTensors[BIDIR_INIT_H_INDEX] != nullptr, nullptrRes); // initHForward
    CHECK_RET(forwardTensors[BIDIR_INIT_C_INDEX] != nullptr, nullptrRes); // initCForward
    CHECK_RET(forwardTensors[BIDIR_DH_INDEX] != nullptr, nullptrRes); // dhForward
    CHECK_RET(forwardTensors[BIDIR_DC_INDEX] != nullptr, nullptrRes); // dcForward
    CHECK_RET(forwardTensors[BIDIR_DY_INDEX] != nullptr, nullptrRes); // dyForward

    // 创建输入
    inputCur = CreateInputTensor(input, h, layersTemp, executor);
    CHECK_RET(inputCur != nullptr, nullptrRes);
    // 前向LSTM计算
    auto resultForward = ExecLstmBackward(inputCur, forwardTensors[BIDIR_INIT_H_INDEX],
        forwardTensors[BIDIR_INIT_C_INDEX], weightForwardCur, biasForwardCur, forwardTensors[BIDIR_DY_INDEX],
        forwardTensors[BIDIR_DH_INDEX], forwardTensors[BIDIR_DC_INDEX], seqLength,
        (*i)[layersTemp * BI_DIRECTION - BI_DIRECTION], (*j)[layersTemp * BI_DIRECTION - BI_DIRECTION],
        (*f)[layersTemp * BI_DIRECTION - BI_DIRECTION], (*o)[layersTemp * BI_DIRECTION - BI_DIRECTION],
        (*h)[layersTemp * BI_DIRECTION - BI_DIRECTION], (*c)[layersTemp * BI_DIRECTION - BI_DIRECTION],
        (*tanhc)[layersTemp * BI_DIRECTION - BI_DIRECTION], executor, false);

    auto backwardTensors = CreateBidDirectionTensors(initHMultiLayer, initCMultiLayer, dh, dc, dy, layersTemp, executor, true);
    CHECK_RET(backwardTensors[BIDIR_INIT_H_INDEX] != nullptr, nullptrRes); // initHBackward
    CHECK_RET(backwardTensors[BIDIR_INIT_C_INDEX] != nullptr, nullptrRes); // initCBackward
    CHECK_RET(backwardTensors[BIDIR_DH_INDEX] != nullptr, nullptrRes); // dhBackward
    CHECK_RET(backwardTensors[BIDIR_DC_INDEX] != nullptr, nullptrRes); // dcBackward
    CHECK_RET(backwardTensors[BIDIR_DY_INDEX] != nullptr, nullptrRes); // dyBackward

    // 后向LSTM计算
    auto resultBackward = ExecLstmBackward(inputCur, backwardTensors[BIDIR_INIT_H_INDEX],
        backwardTensors[BIDIR_INIT_C_INDEX], weightBackwardCur, biasBackwardCur, backwardTensors[BIDIR_DY_INDEX],
        backwardTensors[BIDIR_DH_INDEX], backwardTensors[BIDIR_DC_INDEX], seqLength,
        (*i)[layersTemp * BI_DIRECTION - 1], (*j)[layersTemp * BI_DIRECTION - 1],
        (*f)[layersTemp * BI_DIRECTION - 1], (*o)[layersTemp * BI_DIRECTION - 1],
        (*h)[layersTemp * BI_DIRECTION - 1], (*c)[layersTemp * BI_DIRECTION - 1],
        (*tanhc)[layersTemp * BI_DIRECTION - 1], executor, true);

    // 合并结果
    std::vector<const aclTensor*> dwVector{};
    std::vector<const aclTensor*> dbVector{};
    std::vector<const aclTensor*> dhPrevVector{};
    std::vector<const aclTensor*> dcPrevVector{};
    auto mergedResult = MergeResults(resultForward, resultBackward, dhPrevVector, dcPrevVector, dwVector, dbVector, executor);
    return mergedResult;
}

static std::tuple<const aclTensor*, std::vector<const aclTensor *>, std::vector<const aclTensor *>,
                  std::vector<const aclTensor *>, std::vector<const aclTensor *>> LstmBackwardMultiLayerBidirec(
    const aclTensor *input,
    const aclTensor *initHMultiLayer,
    const aclTensor *initCMultiLayer,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensor *seqLength,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    int64_t numLayers,
    int64_t layersTemp,
    bool hasBias,
    int64_t paramNumPerLayer,
    aclOpExecutor *executor)
{
    const aclTensor* lastLayerDy = nullptr;
    std::vector<const aclTensor *> lastLayerDhPrevVector{};
    std::vector<const aclTensor *> lastLayerDcPrevVector{};
    std::vector<const aclTensor *> lastLayerDwVector{};
    std::vector<const aclTensor *> lastLayerDbVector{};

    auto nullptrRes = std::make_tuple(
        nullptr,
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>(),
        std::vector<const aclTensor*>()
    );
    
    const aclTensor* inputCur = nullptr;
    
    // 递归终止条件：处理最后一层
    if (layersTemp == numLayers) {
        auto lastLayerOutput = LstmBackwardSingleLayerBidirec(input, initHMultiLayer, initCMultiLayer, params, dy, dh,
            dc, seqLength, i, j, f, o, h, c, tanhc, layersTemp, hasBias, paramNumPerLayer, executor);
        CHECK_RET(CheckTupleNotNull(lastLayerOutput), nullptrRes);
        return lastLayerOutput;
    } else {
        // 递归调用处理下一层
        std::tie(lastLayerDy, lastLayerDhPrevVector, lastLayerDcPrevVector, lastLayerDwVector, lastLayerDbVector) =
            LstmBackwardMultiLayerBidirec(input, initHMultiLayer, initCMultiLayer, params, dy, dh, dc, seqLength,
                i, j, f, o, h, c, tanhc, numLayers, layersTemp + 1, hasBias, paramNumPerLayer, executor);
        CHECK_RET(CheckTupleNotNull(std::tie(lastLayerDy, lastLayerDhPrevVector, lastLayerDcPrevVector,
                  lastLayerDwVector, lastLayerDbVector)), nullptrRes);
    }
    // 获取当前层权重和偏置
    auto weightBias = GetWeightBiasFromParams(params, hasBias, true, paramNumPerLayer, layersTemp, executor);
    const aclTensor *weightForwardCur = weightBias[0];
    CHECK_RET(weightForwardCur != nullptr, nullptrRes);
    const aclTensor *weightBackwardCur = weightBias[INDEX_TWO];
    CHECK_RET(weightBackwardCur != nullptr, nullptrRes);
    const aclTensor *biasForwardCur = weightBias[1];
    const aclTensor *biasBackwardCur = weightBias[INDEX_THREE];
    if (hasBias) {
        CHECK_RET(biasForwardCur != nullptr, nullptrRes);
        CHECK_RET(biasBackwardCur != nullptr, nullptrRes);
    }

    // 创建输入张量
    inputCur = CreateInputTensor(input, h, layersTemp, executor);
    CHECK_RET(inputCur != nullptr, nullptrRes);

    // 使用 std::array 接收前向张量
    auto forwardTensors = CreateBidDirectionTensors(initHMultiLayer, initCMultiLayer, dh, dc, lastLayerDy,
                                                    layersTemp, executor, false);
    CHECK_RET(forwardTensors[BIDIR_INIT_H_INDEX] != nullptr, nullptrRes); // initHForward
    CHECK_RET(forwardTensors[BIDIR_INIT_C_INDEX] != nullptr, nullptrRes); // initCForward
    CHECK_RET(forwardTensors[BIDIR_DH_INDEX] != nullptr, nullptrRes); // dhForward
    CHECK_RET(forwardTensors[BIDIR_DC_INDEX] != nullptr, nullptrRes); // dcForward
    CHECK_RET(forwardTensors[BIDIR_DY_INDEX] != nullptr, nullptrRes); // dyForward

    // 前向LSTM计算
    auto resultForward = ExecLstmBackward(inputCur, forwardTensors[BIDIR_INIT_H_INDEX],
        forwardTensors[BIDIR_INIT_C_INDEX], weightForwardCur, biasForwardCur, forwardTensors[BIDIR_DY_INDEX],
        forwardTensors[BIDIR_DH_INDEX], forwardTensors[BIDIR_DC_INDEX], seqLength,
        (*i)[layersTemp * BI_DIRECTION - BI_DIRECTION], (*j)[layersTemp * BI_DIRECTION - BI_DIRECTION],
        (*f)[layersTemp * BI_DIRECTION - BI_DIRECTION], (*o)[layersTemp * BI_DIRECTION - BI_DIRECTION],
        (*h)[layersTemp * BI_DIRECTION - BI_DIRECTION], (*c)[layersTemp * BI_DIRECTION - BI_DIRECTION],
        (*tanhc)[layersTemp * BI_DIRECTION - BI_DIRECTION], executor, false);

    auto backwardTensors = CreateBidDirectionTensors(initHMultiLayer, initCMultiLayer, dh, dc, lastLayerDy,
                                                              layersTemp, executor, true);
    CHECK_RET(backwardTensors[BIDIR_INIT_H_INDEX] != nullptr, nullptrRes); // initHBackward
    CHECK_RET(backwardTensors[BIDIR_INIT_C_INDEX] != nullptr, nullptrRes); // initCBackward
    CHECK_RET(backwardTensors[BIDIR_DH_INDEX] != nullptr, nullptrRes); // dhBackward
    CHECK_RET(backwardTensors[BIDIR_DC_INDEX] != nullptr, nullptrRes); // dcBackward
    CHECK_RET(backwardTensors[BIDIR_DY_INDEX] != nullptr, nullptrRes); // dyBackward

    // 后向LSTM计算
    auto resultBackward = ExecLstmBackward(inputCur, backwardTensors[BIDIR_INIT_H_INDEX],
        backwardTensors[BIDIR_INIT_C_INDEX], weightBackwardCur, biasBackwardCur, backwardTensors[BIDIR_DY_INDEX],
        backwardTensors[BIDIR_DH_INDEX], backwardTensors[BIDIR_DC_INDEX], seqLength,
        (*i)[layersTemp * BI_DIRECTION - 1], (*j)[layersTemp * BI_DIRECTION - 1],
        (*f)[layersTemp * BI_DIRECTION - 1], (*o)[layersTemp * BI_DIRECTION - 1], 
        (*h)[layersTemp * BI_DIRECTION - 1], (*c)[layersTemp * BI_DIRECTION - 1],
        (*tanhc)[layersTemp * BI_DIRECTION - 1], executor, true);

    // 合并结果并更新权重向量
    auto mergedResult = MergeResults(resultForward, resultBackward, lastLayerDhPrevVector, lastLayerDcPrevVector,
                                     lastLayerDwVector, lastLayerDbVector, executor);
    return mergedResult;
}

static bool CheckTensorListNotNull(const aclTensorList *tensorList)
{
    for (uint64_t index = 0; index < tensorList->Size(); index++) {
        OP_CHECK_NULL((*tensorList)[index], return false);
    }
    return true;
}

static bool CheckNotNull(const aclTensor *input,
    const aclTensorList *hc,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    const aclTensor *dx,
    const aclTensor *dhPrev,
    const aclTensor *dcPrev,
    const aclTensorList *dparams,
    bool hasBias,
    int64_t numLayers,
    bool bidirectional)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(hc, return false);
    OP_CHECK_NULL(params, return false);
    OP_CHECK_NULL(dy, return false);
    OP_CHECK_NULL(dh, return false);
    OP_CHECK_NULL(dc, return false);
    OP_CHECK_NULL(i, return false);
    OP_CHECK_NULL(j, return false);
    OP_CHECK_NULL(f, return false);
    OP_CHECK_NULL(o, return false);
    OP_CHECK_NULL(h, return false);
    OP_CHECK_NULL(c, return false);
    OP_CHECK_NULL(tanhc, return false);
    OP_CHECK_NULL(dx, return false);
    OP_CHECK_NULL(dhPrev, return false);
    OP_CHECK_NULL(dcPrev, return false);
    OP_CHECK_NULL(dparams, return false);
    uint64_t gateLength = bidirectional ? numLayers * BI_DIRECTION : numLayers;
    bool tensorLengthCheck = gateLength == i->Size() && gateLength == j->Size() && gateLength == f->Size() &&
        gateLength == o->Size() && gateLength == h->Size() && gateLength == c->Size() && gateLength == tanhc->Size();
    if (hc->Size() != HC_TENSOR_COUNT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "For inithc tensorlist, the tensor quantities %ld should be %d.",
            hc->Size(), HC_TENSOR_COUNT);
        return false;
    }
    if (!tensorLengthCheck) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
            "For tensor lists such as the 4 gates, the tensor quantities should follow consistent patterns.");
        return false;
    }
    uint64_t paramsLength = hasBias && bidirectional ? NUM_WITH_B_AND_BID * numLayers :
                          (hasBias || bidirectional) ? NUM_WITH_B_OR_BID * numLayers :
                          NUM_NO_B_NO_BIDIR * numLayers;
    bool paramsLengthCheck = paramsLength == params->Size() && paramsLength == dparams->Size();

    if (!paramsLengthCheck) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
            "For tensor lists include params and dparams, the tensor quantities should follow the pattern related to the weigths.");
        return false;
    }

    return CheckTensorListNotNull(hc) && CheckTensorListNotNull(i) && CheckTensorListNotNull(j) && CheckTensorListNotNull(f) &&
        CheckTensorListNotNull(o) && CheckTensorListNotNull(h) && CheckTensorListNotNull(c) && CheckTensorListNotNull(tanhc) &&
        CheckTensorListNotNull(dparams);
}

static bool CheckTensorListFormat(const aclTensorList* tensors, const char* listName)
{
    for (uint64_t idx = 0; idx < tensors->Size(); idx++) {
        if ((*tensors)[idx]->GetStorageFormat() != Format::FORMAT_ND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor %lu format only support ND", listName, idx);
            return false;
        }
    }
    return true;
}

static bool CheckFormatValid(const aclTensor *input,
    const aclTensorList *hc,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    const aclTensor *dx,
    const aclTensor *dhPrev,
    const aclTensor *dcPrev,
    const aclTensorList *dparams,
    const aclTensor *batchSizes=nullptr)
{
    if (input->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input format only support ND");
        return false;
    }
    if (dy->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dy format only support ND");
        return false;
    }
    if (dh->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dh format only support ND");
        return false;
    }
    if (dc->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dc format only support ND");
        return false;
    }
    if (dx->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dx format only support ND");
        return false;
    }
    if (dhPrev->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dhPrev format only support ND");
        return false;
    }
    if (dcPrev->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dcPrev format only support ND");
        return false;
    }
    if (batchSizes != nullptr && batchSizes->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batchSizes only support ND");
        return false;
    }
    if (!CheckTensorListFormat(hc, "hc")) return false;
    if (!CheckTensorListFormat(params, "params")) return false;
    if (!CheckTensorListFormat(i, "i")) return false;
    if (!CheckTensorListFormat(j, "j")) return false;
    if (!CheckTensorListFormat(f, "f")) return false;
    if (!CheckTensorListFormat(o, "o")) return false;
    if (!CheckTensorListFormat(h, "h")) return false;
    if (!CheckTensorListFormat(c, "c")) return false;
    if (!CheckTensorListFormat(tanhc, "tanhc")) return false;
    if (!CheckTensorListFormat(dparams, "dparams")) return false;
    return true;
}

// 检查单个张量的数据类型支持和一致性
static bool CheckSingleTensorDtype(const aclTensor* tensor, const char* tensorName,
                                   ge::DataType baseDtype)
{
    // 检查是否在支持的数据类型列表中
    OP_CHECK_DTYPE_NOT_SUPPORT(tensor, DTYPE_SUPPORT_LIST, return false);
    
    // 检查数据类型一致性
    if (tensor->GetDataType() != baseDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
            "%s tensor dtype inconsistent, expected: %s, actual: %s.", 
            tensorName,
            op::ToString(baseDtype).GetString(),
            op::ToString(tensor->GetDataType()).GetString());
        return false;
    }
    
    return true;
}

// 检查张量列表的数据类型支持和一致性
static bool CheckTensorListDtype(const aclTensorList* tensors, const char* listName,
                                 ge::DataType baseDtype)
{
    for (uint64_t idx = 0; idx < tensors->Size(); idx++) {
        const aclTensor* tensor = (*tensors)[idx];
        // 检查是否在支持的数据类型列表中
        OP_CHECK_DTYPE_NOT_SUPPORT(tensor, DTYPE_SUPPORT_LIST, return false);
        
        // 检查数据类型一致性
        if (tensor->GetDataType() != baseDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "%s tensor %lu dtype inconsistent, expected: %s, actual: %s.",
                listName, idx,
                op::ToString(baseDtype).GetString(),
                op::ToString(tensor->GetDataType()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *input,
    const aclTensorList *hc,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    const aclTensor *dx,
    const aclTensor *dhPrev,
    const aclTensor *dcPrev,
    const aclTensorList *dparams,
    const aclTensor *batchSizes=nullptr)
{
    ge::DataType baseDtype = input->GetDataType();
    // 检查所有单个张量
    if (!CheckSingleTensorDtype(dy, "dy", baseDtype)) return false;
    if (!CheckSingleTensorDtype(dh, "dh", baseDtype)) return false;
    if (!CheckSingleTensorDtype(dc, "dc", baseDtype)) return false;
    if (!CheckSingleTensorDtype(dx, "dx", baseDtype)) return false;
    if (!CheckSingleTensorDtype(dhPrev, "dhPrev", baseDtype)) return false;
    if (!CheckSingleTensorDtype(dcPrev, "dcPrev", baseDtype)) return false;
    if (batchSizes != nullptr && batchSizes->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
            "batchSizes tensor dtype inconsistent, expected: %s, actual: %s.", 
            op::ToString(op::DataType::DT_INT64).GetString(),
            op::ToString(batchSizes->GetDataType()).GetString());
        return false;
    };

    // 检查所有张量列表
    if (!CheckTensorListDtype(hc, "hc", baseDtype)) return false;
    if (!CheckTensorListDtype(params, "params", baseDtype)) return false;
    if (!CheckTensorListDtype(i, "i", baseDtype)) return false;
    if (!CheckTensorListDtype(j, "j", baseDtype)) return false;
    if (!CheckTensorListDtype(f, "f", baseDtype)) return false;
    if (!CheckTensorListDtype(o, "o", baseDtype)) return false;
    if (!CheckTensorListDtype(h, "h", baseDtype)) return false;
    if (!CheckTensorListDtype(c, "c", baseDtype)) return false;
    if (!CheckTensorListDtype(tanhc, "tanhc", baseDtype)) return false;
    if (!CheckTensorListDtype(dparams, "dparams", baseDtype)) return false;
    return true;
}

static bool ValidateInputShape(const aclTensor *input, std::vector<int64_t>& expected_dims, const char* tensorName) {
  auto shape = input->GetStorageShape();
  if (shape.GetDimNum() != expected_dims.size()) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor %s has wrong dimension count", tensorName);
      return false;
  }

  for (size_t i = 0; i < expected_dims.size(); ++i) {
      if (expected_dims[i] != shape.GetDim(i)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor %s dim %zu mismatch", tensorName, i);
        return false;
      }
  }
  return true;
};

static bool CheckShapeValid(const aclTensor *input,
    const aclTensorList *hc,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    const aclTensor *dx,
    const aclTensor *dhPrev,
    const aclTensor *dcPrev,
    const aclTensorList *dparams,
    bool hasBias,
    int64_t numLayers,
    bool bidirectional,
    bool batchFirst,
    const aclTensor *batchSizes=nullptr)
{
    OP_CHECK_WRONG_DIMENSION(dh, DIM_THREE, return false);
    bool hasSeqlength = batchSizes != nullptr;
    size_t inputDimsNum = hasSeqlength ? DIM_TWO : DIM_THREE;
    OP_CHECK_WRONG_DIMENSION(input, inputDimsNum, return false);
    if (hasSeqlength) {
        OP_CHECK_WRONG_DIMENSION(batchSizes, DIM_ONE, return false);
    }
    auto dhShape = dh->GetViewShape();
    auto inputShape = input->GetViewShape();
    int64_t inputSize = inputShape[inputDimsNum - 1];
    int64_t hiddenSize = dhShape[HIDDEN_DIM];
    int64_t batch = dhShape[BATCH_DIM];
    
    int64_t timeStep = hasSeqlength ? batchSizes->GetViewShape()[SEQUENCE_DIM] :
                       batchFirst ? inputShape[BATCH_DIM] : inputShape[SEQUENCE_DIM];
    int64_t bid = bidirectional ? BI_DIRECTION : SINGLE_DIRECTION;
    
    std::vector<int64_t> biasDim = {GATE_COUNT * hiddenSize};
    std::vector<int64_t> inithDim = {numLayers * bid, batch, hiddenSize};
    std::vector<int64_t> hiddenDim = {timeStep, batch, hiddenSize};
    std::vector<int64_t> outHiddenDim = hasSeqlength
                                        ? std::vector<int64_t>{timeStep * batch, hiddenSize * bid}
                                        : batchFirst ? std::vector<int64_t>{batch, timeStep, hiddenSize * bid} :
                                        std::vector<int64_t>{timeStep, batch, hiddenSize * bid};
    std::vector<int64_t> weightHiddenDim = {GATE_COUNT * hiddenSize, hiddenSize};
    std::vector<int64_t> inputDim = hasSeqlength
                                    ? std::vector<int64_t>{timeStep * batch, inputSize}
                                    : batchFirst ? std::vector<int64_t>{batch, timeStep, inputSize} :
                                    std::vector<int64_t>{timeStep, batch, inputSize};
    bool ret = numLayers > 0 &&
                ValidateInputShape(input, inputDim, "input") &&
                ValidateInputShape((*hc)[0], inithDim, "inith") &&
                ValidateInputShape((*hc)[1], inithDim, "initc") &&
                ValidateInputShape(dy, outHiddenDim, "dy") &&
                ValidateInputShape(dh, inithDim, "dh") &&
                ValidateInputShape(dc, inithDim, "dc") &&
                ValidateInputShape(dx, inputDim, "dx") &&
                ValidateInputShape(dhPrev, inithDim, "dhPrev") &&
                ValidateInputShape(dcPrev, inithDim, "dcPrev");
    
    for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        inputSize = layerIdx == 0 ? inputSize : bid * hiddenSize;
        std::vector<int64_t> weightInputDim = {GATE_COUNT * hiddenSize, inputSize};
        if (hasBias && bidirectional) {
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_AND_BID * layerIdx + WEIGHT_INPUT_INDEX], weightInputDim, "wi");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_AND_BID * layerIdx + WEIGHT_HIDDEN_INDEX], weightHiddenDim, "wh");
            ret = ret && ValidateInputShape((*params)[NUM_WITH_B_AND_BID * layerIdx + BIAS_INPUT_INDEX], biasDim, "bi");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_AND_BID * layerIdx + BIAS_HIDDEN_INDEX], biasDim, "bh");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + WEIGHT_INPUT_INDEX], weightInputDim,
                "wi");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + WEIGHT_HIDDEN_INDEX],
                weightHiddenDim, "wh");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + BIAS_INPUT_INDEX], biasDim, "bi");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + BIAS_HIDDEN_INDEX], biasDim, "bh");

            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + WEIGHT_INPUT_INDEX], weightInputDim, "dwi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + WEIGHT_HIDDEN_INDEX], weightHiddenDim, "dwh");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + BIAS_INPUT_INDEX], biasDim, "dbi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + BIAS_HIDDEN_INDEX], biasDim, "dbh");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + WEIGHT_INPUT_INDEX], weightInputDim,
                "dwi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + WEIGHT_HIDDEN_INDEX],
                weightHiddenDim, "dwh");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + BIAS_INPUT_INDEX], biasDim, "dbi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_AND_BID * layerIdx + NUM_WITH_B_AND_BID / BI_DIRECTION + BIAS_HIDDEN_INDEX], biasDim, "dbh");
        } else if (hasBias && !bidirectional) {
            ret = ret && ValidateInputShape((*params)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_INPUT_INDEX],
                weightInputDim, "wi");
            ret = ret && ValidateInputShape((*params)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_HIDDEN_INDEX],
                weightHiddenDim, "wh");
            ret = ret && ValidateInputShape((*params)[NUM_WITH_B_OR_BID * layerIdx + BIAS_INPUT_INDEX], biasDim, "bi");
            ret = ret && ValidateInputShape((*params)[NUM_WITH_B_OR_BID * layerIdx + BIAS_HIDDEN_INDEX], biasDim, "bh");

            ret = ret && ValidateInputShape((*dparams)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_INPUT_INDEX],
                weightInputDim, "dwi");
            ret = ret && ValidateInputShape((*dparams)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_HIDDEN_INDEX],
                weightHiddenDim, "dwh");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_OR_BID * layerIdx + BIAS_INPUT_INDEX], biasDim, "dbi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_OR_BID * layerIdx + BIAS_HIDDEN_INDEX], biasDim, "dbh");
        } else if (!hasBias && bidirectional) {
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_INPUT_INDEX], weightInputDim, "wi");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_HIDDEN_INDEX], weightHiddenDim, "wh");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_OR_BID * layerIdx + NUM_WITH_B_OR_BID / BI_DIRECTION + WEIGHT_INPUT_INDEX], weightInputDim,
                "wi");
            ret = ret && ValidateInputShape(
                (*params)[NUM_WITH_B_OR_BID * layerIdx + NUM_WITH_B_OR_BID / BI_DIRECTION + WEIGHT_HIDDEN_INDEX], weightHiddenDim,
                "wh");

            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_INPUT_INDEX], weightInputDim, "dwi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_OR_BID * layerIdx + WEIGHT_HIDDEN_INDEX], weightHiddenDim, "dwh");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_OR_BID * layerIdx + NUM_WITH_B_OR_BID / BI_DIRECTION + WEIGHT_INPUT_INDEX], weightInputDim,
                "dwi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_WITH_B_OR_BID * layerIdx + NUM_WITH_B_OR_BID / BI_DIRECTION + WEIGHT_HIDDEN_INDEX], weightHiddenDim,
                "dwh");
        } else {
            ret = ret && ValidateInputShape(
                (*params)[NUM_NO_B_NO_BIDIR * layerIdx + WEIGHT_INPUT_INDEX], weightInputDim, "wi");
            ret = ret && ValidateInputShape(
                (*params)[NUM_NO_B_NO_BIDIR * layerIdx + WEIGHT_HIDDEN_INDEX], weightHiddenDim, "wh");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_NO_B_NO_BIDIR * layerIdx + WEIGHT_INPUT_INDEX], weightInputDim, "dwi");
            ret = ret && ValidateInputShape(
                (*dparams)[NUM_NO_B_NO_BIDIR * layerIdx + WEIGHT_HIDDEN_INDEX], weightHiddenDim, "dwh");
        }
    }

    for (int64_t gateIdx = 0; gateIdx < numLayers * bid; gateIdx++) {
        ret = ret && ValidateInputShape((*i)[gateIdx], hiddenDim, "i");
        ret = ret && ValidateInputShape((*j)[gateIdx], hiddenDim, "j");
        ret = ret && ValidateInputShape((*f)[gateIdx], hiddenDim, "f");
        ret = ret && ValidateInputShape((*o)[gateIdx], hiddenDim, "o");
        ret = ret && ValidateInputShape((*h)[gateIdx], hiddenDim, "h");
        ret = ret && ValidateInputShape((*c)[gateIdx], hiddenDim, "c");
        ret = ret && ValidateInputShape((*tanhc)[gateIdx], hiddenDim, "tanhc");
    }

    return ret;
}

static aclnnStatus CheckParams(const aclTensor *input,
    const aclTensorList *hc,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensorList *i,
    const aclTensorList *j,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    const aclTensor *dx,
    const aclTensor *dhPrev,
    const aclTensor *dcPrev,
    const aclTensorList *dparams,
    bool hasBias,
    int64_t numLayers,
    bool bidirectional,
    bool batchFirst,
    const aclTensor *batchSizes=nullptr)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, hc, params, dy, dh, dc, i, j, f, o, h, c, tanhc, dx, dhPrev, dcPrev, dparams, hasBias,
                           numLayers, bidirectional), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(input, hc, params, dy, dh, dc, i, j, f, o, h, c, tanhc, dx, dhPrev, dcPrev, dparams, batchSizes),
                              ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShapeValid(input, hc, params, dy, dh, dc, i, j, f, o, h, c, tanhc, dx, dhPrev, dcPrev, dparams,
                              hasBias, numLayers, bidirectional, batchFirst, batchSizes), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查format是否满足约束
    CHECK_RET(CheckFormatValid(input, hc, params, dy, dh, dc, i, j, f, o, h, c, tanhc, dx, dhPrev, dcPrev, dparams,
                               batchSizes), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool EmptyCheck(const aclTensor *input,
    const aclTensorList *hc,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensorList *i,
    const aclTensorList *j,  // 修正参数名：j -> g
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    const aclTensor *batchSizes=nullptr)
{
    if (input->IsEmpty() || dy->IsEmpty() || dh->IsEmpty() || dc->IsEmpty()) {
        return false;
    }

    if (batchSizes != nullptr && batchSizes->IsEmpty()) {
        return false;
    }
    auto checkTensorList = [](const aclTensorList* tensorList) {
        for (uint64_t idx = 0; idx < tensorList->Size(); idx++) {
            if ((*tensorList)[idx]->IsEmpty()) {
                return false;
            }
        }
        return true;
    };
    if (!checkTensorList(hc) || !checkTensorList(params) ||
        !checkTensorList(i) || !checkTensorList(j) ||
        !checkTensorList(f) || !checkTensorList(o) ||
        !checkTensorList(h) || !checkTensorList(c) ||
        !checkTensorList(tanhc)) {
        return false;
    }
    return true;
}

static const aclTensor* GetSliceTensor(const FVector<int64_t> offsetVector, const FVector<int64_t> sizeVector,
                                       const aclTensor* input, aclOpExecutor *executor)
{
    aclIntArray* offsetArray = executor->AllocIntArray(offsetVector.data(), offsetVector.size());
    CHECK_RET(offsetArray != nullptr, nullptr);
    aclIntArray* sizeArray = executor->AllocIntArray(sizeVector.data(), sizeVector.size());
    CHECK_RET(sizeArray != nullptr, nullptr);
    auto res = l0op::Slice(input, offsetArray, sizeArray, executor);
    CHECK_RET(res != nullptr, nullptr);
    return res;
}

// 抽取处理LSTM梯度输出的函数
static std::tuple<const aclTensor*, const aclTensor*, std::vector<const aclTensor*>> GetLstmGradExceptDx(
    std::tuple<const aclTensor*, std::vector<const aclTensor*>, std::vector<const aclTensor*>,
    std::vector<const aclTensor*>, std::vector<const aclTensor*>>& output,
    const aclTensor* input,
    const aclTensor* dhPrevOut,
    bool bidirectional,
    bool hasBias,
    int64_t numLayers,
    aclOpExecutor *executor)
{
    auto nullptrRes = std::make_tuple(
        nullptr,
        nullptr,
        std::vector<const aclTensor*>()
    );
    std::vector<const aclTensor *> dhPrevVectorReverse = std::get<INDEX_ONE>(output);
    std::vector<const aclTensor *> dcPrevVectorReverse = std::get<INDEX_TWO>(output);
    std::vector<const aclTensor *> dwVectorReverse = std::get<INDEX_THREE>(output);
    std::vector<const aclTensor *> dbVectorReverse = std::get<INDEX_FOUR>(output);
    std::vector<const aclTensor *> dhPrevVector{};
    std::vector<const aclTensor *> dcPrevVector{};
    std::vector<const aclTensor *> dparamsVector{};

    int64_t inputSize = input->GetViewShape()[DIM_TWO];
    int64_t hiddenSize = dhPrevOut->GetViewShape()[HIDDEN_DIM];
    int64_t bid = bidirectional ? BI_DIRECTION : SINGLE_DIRECTION;

    // 处理每一层的梯度
    for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        auto dwFCur = dwVectorReverse[(numLayers - layerIdx - 1) * bid + 0];
        auto inputSizeCur = layerIdx == 0 ? inputSize : hiddenSize * bid;
        
        // 前向层的输入权重梯度
        FVector<int64_t> offsetVectorF{DIM_ZERO, DIM_ZERO};
        FVector<int64_t> sizeVectorF{GATE_COUNT * hiddenSize, inputSizeCur};
        auto dwFInputCur = GetSliceTensor(offsetVectorF, sizeVectorF, dwFCur, executor);

        // 前向层的隐藏状态权重梯度
        FVector<int64_t> offsetVectorB{DIM_ZERO, inputSizeCur};
        FVector<int64_t> sizeVectorB{GATE_COUNT * hiddenSize, hiddenSize};
        auto dwFHiddenCur = GetSliceTensor(offsetVectorB, sizeVectorB, dwFCur, executor);
        dparamsVector.emplace_back(dwFInputCur);
        dparamsVector.emplace_back(dwFHiddenCur);

        // 处理偏置和双向情况
        if (hasBias && bidirectional) {
            auto dwBCur = dwVectorReverse[(numLayers - layerIdx - 1) * bid + 1];
            auto dwBInputCur = GetSliceTensor(offsetVectorF, sizeVectorF, dwBCur, executor);
            auto dwBHiddenCur = GetSliceTensor(offsetVectorB, sizeVectorB, dwBCur, executor);
            auto dbFCur = dbVectorReverse[(numLayers - layerIdx - 1) * bid + 0];
            auto dbBCur = dbVectorReverse[(numLayers - layerIdx - 1) * bid + 1];
            dparamsVector.emplace_back(dbFCur);
            dparamsVector.emplace_back(dbFCur);
            dparamsVector.emplace_back(dwBInputCur);
            dparamsVector.emplace_back(dwBHiddenCur);
            dparamsVector.emplace_back(dbBCur);
            dparamsVector.emplace_back(dbBCur);
        } else if (hasBias && !bidirectional) {
            auto dbFCur = dbVectorReverse[(numLayers - layerIdx - 1) * bid + 0];
            dparamsVector.emplace_back(dbFCur);
            dparamsVector.emplace_back(dbFCur);
        } else if (!hasBias && bidirectional) {
            auto dwBCur = dwVectorReverse[(numLayers - layerIdx - 1) * bid + 1];
            auto dwBInputCur = GetSliceTensor(offsetVectorF, sizeVectorF, dwBCur, executor);
            auto dwBHiddenCur = GetSliceTensor(offsetVectorB, sizeVectorB, dwBCur, executor);
            dparamsVector.emplace_back(dwBInputCur);
            dparamsVector.emplace_back(dwBHiddenCur);
        }

        // 处理隐藏状态和细胞状态的梯度
        dhPrevVector.emplace_back(dhPrevVectorReverse[(numLayers - layerIdx - 1) * bid + 0]);
        dcPrevVector.emplace_back(dcPrevVectorReverse[(numLayers - layerIdx - 1) * bid + 0]);
        if (bidirectional) {
            dhPrevVector.emplace_back(dhPrevVectorReverse[(numLayers - layerIdx - 1) * bid + 1]);
            dcPrevVector.emplace_back(dcPrevVectorReverse[(numLayers - layerIdx - 1) * bid + 1]);
        }
    }

    // 拼接隐藏状态梯度
    auto dhPrev = SplitToConcat(dhPrevVector, CONCAT_DIM_LAYER, executor);

    // 拼接细胞状态梯度
    auto dcPrev = SplitToConcat(dcPrevVector, CONCAT_DIM_LAYER, executor);
    return std::make_tuple(dhPrev, dcPrev, dparamsVector);
}

static const aclTensorList* MakeContiguousList(const aclTensorList* tensor_list, aclOpExecutor *executor)
{
    std::vector<const aclTensor*> tensors_vec;
    for (size_t i = 0; i < tensor_list->Size(); ++i) {
        auto contiguous = l0op::Contiguous((*tensor_list)[i], executor);
        if (contiguous == nullptr) {
            return nullptr;
        }
        tensors_vec.push_back(contiguous);
    }
    return executor->AllocTensorList(tensors_vec.data(), tensors_vec.size());
}

static aclnnStatus CreateContiguousTensors(
    const aclTensor* input,
    const aclTensorList* hx,
    const aclTensorList* params,
    const aclTensor* dy,
    const aclTensor* dh,
    const aclTensor* dc,
    const aclTensorList* i,
    const aclTensorList* g,
    const aclTensorList* f,
    const aclTensorList* o,
    const aclTensorList* h,
    const aclTensorList* c,
    const aclTensorList* tanhc,
    aclOpExecutor *executor,
    LSTMContinuousTensors* output,
    const aclTensor* batchSizes=nullptr)
{
    output->inputContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(output->inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->hxContiguous = MakeContiguousList(hx, executor);
    CHECK_RET(output->hxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->paramsContiguous = MakeContiguousList(params, executor);
    CHECK_RET(output->paramsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->dyContiguous = l0op::Contiguous(dy, executor);
    CHECK_RET(output->dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->dhContiguous = l0op::Contiguous(dh, executor);
    CHECK_RET(output->dhContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->dcContiguous = l0op::Contiguous(dc, executor);
    CHECK_RET(output->dcContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->iContiguous = MakeContiguousList(i, executor);
    CHECK_RET(output->iContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->gContiguous = MakeContiguousList(g, executor);
    CHECK_RET(output->gContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->fContiguous = MakeContiguousList(f, executor);
    CHECK_RET(output->fContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->oContiguous = MakeContiguousList(o, executor);
    CHECK_RET(output->oContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->hContiguous = MakeContiguousList(h, executor);
    CHECK_RET(output->hContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->cContiguous = MakeContiguousList(c, executor);
    CHECK_RET(output->cContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->tanhcContiguous = MakeContiguousList(tanhc, executor);
    CHECK_RET(output->tanhcContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    output->batchSizesContiguous = batchSizes == nullptr ? nullptr : l0op::Contiguous(batchSizes, executor);
    if (batchSizes != nullptr) {
        CHECK_RET(output->batchSizesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACL_SUCCESS;
}

// 校验LSTM反向传播输出结果
static aclnnStatus ValidateLstmBackwardOutput(
    const std::tuple<const aclTensor*, std::vector<const aclTensor*>, 
                     std::vector<const aclTensor*>, std::vector<const aclTensor*>,
                     std::vector<const aclTensor*>>& output)
{
    CHECK_RET(std::get<0>(output) != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    
    for (uint64_t idx = 0; idx < std::get<INDEX_THREE>(output).size(); idx++) {
        CHECK_RET(std::get<INDEX_ONE>(output)[idx] != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        CHECK_RET(std::get<INDEX_TWO>(output)[idx] != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        CHECK_RET(std::get<INDEX_THREE>(output)[idx] != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }
    
    for (uint64_t idx = 0; idx < std::get<INDEX_FOUR>(output).size(); idx++) {
        CHECK_RET(std::get<INDEX_FOUR>(output)[idx] != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }
    
    return ACLNN_SUCCESS;
}

// 搬出LSTM反向传播结果
static aclnnStatus CopyLstmBackwardResults(
    const aclTensor* dx, const aclTensor* dhPrev, const aclTensor* dcPrev,
    const std::vector<const aclTensor*>& dparamsVector,
    aclTensor* dxOut, aclTensor* dhPrevOut, aclTensor* dcPrevOut,
    aclTensorList* dparamsOut, int64_t numLayers, int64_t paramNumPerLayer,
    aclOpExecutor *executor)
{
    auto dxCopyResult = l0op::ViewCopy(dx, dxOut, executor);
    CHECK_RET(dxCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    auto dhPrevCopyResult = l0op::ViewCopy(dhPrev, dhPrevOut, executor);
    CHECK_RET(dhPrevCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    auto dcPrevCopyResult = l0op::ViewCopy(dcPrev, dcPrevOut, executor);
    CHECK_RET(dcPrevCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    for (int64_t idx = 0; idx < numLayers * paramNumPerLayer; idx++) {
        auto dparamsCopyResult = l0op::ViewCopy(dparamsVector[idx], (*dparamsOut)[idx], executor);
        CHECK_RET(dparamsCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecLstmInputBackward(
    const LSTMContinuousTensors* allInput,
    aclTensor *dxOut,
    aclTensor *dhPrevOut,
    aclTensor *dcPrevOut,
    aclTensorList *dparamsOut,
    bool batchFirst,
    bool bidirectional,
    bool hasBias,
    int64_t numLayers,
    aclOpExecutor *executor)
{
    std::tuple<const aclTensor*, std::vector<const aclTensor*>, std::vector<const aclTensor*>,
               std::vector<const aclTensor*>, std::vector<const aclTensor *>> output;
    // T转到第1维
    FVector<int64_t> newShapeDims = {1, 0, 2};
    auto perm = executor->AllocIntArray(newShapeDims.data(), newShapeDims.size());
    CHECK_RET(perm != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto inputTranspose = batchFirst ?
                          l0op::Transpose(allInput->inputContiguous, perm, executor) :
                          allInput->inputContiguous;
    CHECK_RET(inputTranspose != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto dyTranspose = batchFirst ?
                       l0op::Transpose(allInput->dyContiguous, perm, executor) :
                       allInput->dyContiguous;
    CHECK_RET(dyTranspose != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    int64_t paramNumPerLayer = (hasBias && bidirectional) ? NUM_WITH_B_AND_BID :
                              (hasBias || bidirectional) ? NUM_WITH_B_OR_BID :
                              NUM_NO_B_NO_BIDIR;
    if (!bidirectional) {
        output = LstmBackwardMultiLayerDirec(inputTranspose, (*(allInput->hxContiguous))[0],
            (*(allInput->hxContiguous))[1], allInput->paramsContiguous, dyTranspose,
            allInput->dhContiguous, allInput->dcContiguous, nullptr,
            allInput->iContiguous, allInput->gContiguous, allInput->fContiguous,
            allInput->oContiguous, allInput->hContiguous, allInput->cContiguous,
            allInput->tanhcContiguous, numLayers, 1, hasBias, paramNumPerLayer, executor);
    } else {
        output = LstmBackwardMultiLayerBidirec(inputTranspose, (*allInput->hxContiguous)[0],
            (*allInput->hxContiguous)[1], allInput->paramsContiguous, dyTranspose,
            allInput->dhContiguous, allInput->dcContiguous, nullptr,
            allInput->iContiguous, allInput->gContiguous, allInput->fContiguous,
            allInput->oContiguous, allInput->hContiguous, allInput->cContiguous,
            allInput->tanhcContiguous, numLayers, 1, hasBias, paramNumPerLayer, executor);
    }
    CHECK_RET(ValidateLstmBackwardOutput(output) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    const aclTensor *dhPrev = nullptr;
    const aclTensor *dcPrev = nullptr;
    std::vector<const aclTensor *> dparamsVector{};
    // 输出梯度处理
    std::tie(dhPrev, dcPrev, dparamsVector) = GetLstmGradExceptDx(output, allInput->inputContiguous,
        dhPrevOut, bidirectional, hasBias, numLayers, executor);
    auto dx = batchFirst ? l0op::Transpose(std::get<0>(output), perm, executor) : std::get<0>(output);
    CHECK_RET(dx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return CopyLstmBackwardResults(dx, dhPrev, dcPrev, dparamsVector,
                                   dxOut, dhPrevOut, dcPrevOut, dparamsOut,
                                   numLayers, paramNumPerLayer, executor);
}

static aclnnStatus ExecLstmDataBackward(
    const LSTMContinuousTensors* allInput,
    aclTensor *dxOut,
    aclTensor *dhPrevOut,
    aclTensor *dcPrevOut,
    aclTensorList *dparamsOut,
    bool bidirectional,
    bool hasBias,
    int64_t numLayers,
    aclOpExecutor *executor)
{
    std::tuple<const aclTensor*, std::vector<const aclTensor*>, std::vector<const aclTensor*>,
               std::vector<const aclTensor*>, std::vector<const aclTensor *>> output;
    auto batchSizeShape = allInput->batchSizesContiguous->GetViewShape();
    auto dataShape = allInput->inputContiguous->GetViewShape();

    // 输入reshape为[T, N, D]
    FVector<int64_t> reshapeInputVector{
        batchSizeShape[SEQUENCE_DIM], dataShape[0] / batchSizeShape[SEQUENCE_DIM], dataShape[1]};
    aclIntArray* reshapeInputArray = executor->AllocIntArray(reshapeInputVector.data(), DIM_THREE);
    CHECK_RET(reshapeInputArray != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto input = l0op::Reshape(allInput->inputContiguous, reshapeInputArray, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    FVector<int64_t> reshapeDyVector{
        batchSizeShape[SEQUENCE_DIM], dataShape[0] / batchSizeShape[SEQUENCE_DIM],
        allInput->dyContiguous->GetViewShape()[1]};
    aclIntArray* reshapeDyArray = executor->AllocIntArray(reshapeDyVector.data(), DIM_THREE);
    CHECK_RET(reshapeDyArray != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto dyReshape = l0op::Reshape(allInput->dyContiguous, reshapeDyArray, executor);
    CHECK_RET(dyReshape != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // batchSizes转Mask
    const aclTensor* seqLength = GetMask(input, allInput->batchSizesContiguous, 
                                         allInput->dhContiguous, executor);
    CHECK_RET(seqLength != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    int64_t paramNumPerLayer = (hasBias && bidirectional) ? NUM_WITH_B_AND_BID :
                              (hasBias || bidirectional) ? NUM_WITH_B_OR_BID :
                              NUM_NO_B_NO_BIDIR;                                    

    if (!bidirectional) {
        output = LstmBackwardMultiLayerDirec(input, (*(allInput->hxContiguous))[0],
            (*(allInput->hxContiguous))[1], allInput->paramsContiguous, dyReshape,
            allInput->dhContiguous, allInput->dcContiguous, seqLength,
            allInput->iContiguous, allInput->gContiguous, allInput->fContiguous,
            allInput->oContiguous, allInput->hContiguous, allInput->cContiguous,
            allInput->tanhcContiguous, numLayers, 1, hasBias, paramNumPerLayer, executor);
    } else {
        output = LstmBackwardMultiLayerBidirec(input, (*(allInput->hxContiguous))[0],
            (*(allInput->hxContiguous))[1], allInput->paramsContiguous, dyReshape,
            allInput->dhContiguous, allInput->dcContiguous, seqLength,
            allInput->iContiguous, allInput->gContiguous, allInput->fContiguous,
            allInput->oContiguous, allInput->hContiguous, allInput->cContiguous,
            allInput->tanhcContiguous, numLayers, 1, hasBias, paramNumPerLayer, executor);
    }
    CHECK_RET(ValidateLstmBackwardOutput(output) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    const aclTensor *dhPrev = nullptr;
    const aclTensor *dcPrev = nullptr;
    std::vector<const aclTensor *> dparamsVector{};
    std::tie(dhPrev, dcPrev, dparamsVector) = GetLstmGradExceptDx(output, input, dhPrevOut, bidirectional, hasBias,
        numLayers, executor);

    FVector<int64_t> reshapeVector{dataShape[0], dataShape[1]};
    aclIntArray* reshapeArray = executor->AllocIntArray(reshapeVector.data(), DIM_TWO);
    CHECK_RET(reshapeArray != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto dx = l0op::Reshape(std::get<0>(output), reshapeArray, executor);
    CHECK_RET(dx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return CopyLstmBackwardResults(dx, dhPrev, dcPrev, dparamsVector,
                                   dxOut, dhPrevOut, dcPrevOut, dparamsOut,
                                   numLayers, paramNumPerLayer, executor);;
}

aclnnStatus aclnnLstmBackwardGetWorkspaceSize(
    const aclTensor *input,
    const aclTensorList *hx,
    const aclTensorList *params,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensorList *i,
    const aclTensorList *g,
    const aclTensorList *f,
    const aclTensorList *o,
    const aclTensorList *h,
    const aclTensorList *c,
    const aclTensorList *tanhc,
    const aclTensor *batchSizesOptional,
    bool hasBias,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst,
    const aclBoolArray *outputMask,
    aclTensor *dxOut,
    aclTensor *dhPrevOut,
    aclTensor *dcPrevOut,
    aclTensorList *dparamsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnLstmBackward,
        DFX_IN(input, hx, params, dy, dh, dc, i, g, f, o, h, c, tanhc, batchSizesOptional, hasBias, numLayers, dropout, train,
               bidirectional, batchFirst),
        DFX_OUT(outputMask, dxOut, dhPrevOut, dcPrevOut, dparamsOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(input, hx, params, dy, dh, dc, i, g, f, o, h, c, tanhc, dxOut, dhPrevOut, dcPrevOut,
                           dparamsOut, hasBias, numLayers, bidirectional, batchFirst, batchSizesOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (!EmptyCheck(input, hx, params, dy, dh, dc, i, g, f, o, h, c, tanhc)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    LSTMContinuousTensors allInputContiguous;
    CHECK_RET(CreateContiguousTensors(input, hx, params, dy, dh, dc, i, g, f, o, h, c, tanhc, uniqueExecutor.get(),
                            &allInputContiguous, batchSizesOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    
    if (batchSizesOptional == nullptr) {
        ret = ExecLstmInputBackward(&allInputContiguous, dxOut, dhPrevOut, dcPrevOut, dparamsOut,
                                    batchFirst, bidirectional, hasBias, numLayers, uniqueExecutor.get());
    } else {
        ret = ExecLstmDataBackward(&allInputContiguous, dxOut, dhPrevOut, dcPrevOut, dparamsOut,
                                   bidirectional, hasBias, numLayers, uniqueExecutor.get());
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLstmBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLstmBackward);
    //  固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif