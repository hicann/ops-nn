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
 * \file aclnn_gru.cpp
 * \brief
 */

#include "aclnn_gru.h"
#include "gru.h"
#include "level0/zero_op.h"
#include "level0/add.h"
#include "level0/arange.h"
#include "level0/broadcast_to.h"
#include "level0/concat.h"
#include "level0/greater.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"

using namespace op;
namespace {
//  =========PackedSequence模式的数据结构
struct GruDataParamsIn {
    const aclTensor* input;
    const aclTensorList* params;
    const aclTensor* hx;
    const aclTensor* batchSizes;
    int64_t numLayers;
    bool hasBias;
    bool train;
    bool bidirection;
};

struct GruDataParamsOut {
    aclTensor* output;
    aclTensor* hy;
    aclTensorList* rOut;
    aclTensorList* zOut;
    aclTensorList* nOut;
    aclTensorList* hnOut;
    aclTensorList* hOut;
};

struct GruBaseOpInputs {
    const aclTensor* input;
    const aclTensor* weightInput;
    const aclTensor* weightHidden;
    const aclTensor* biasInput;
    const aclTensor* biasHidden;
    const aclTensor* seqLengthOptional; //  batch_sizes数组
    const aclTensor* initHOptional;     //  初始隐状态h0
    const char* direction;
    bool isTraining;
};

struct GruBaseOpOutputs {
    aclTensor* l0_yOut;
    aclTensor* l0_outputHOut;
    aclTensor* l0_rOut;
    aclTensor* l0_zOut;
    aclTensor* l0_nOut;
    aclTensor* l0_nHOut;
};

struct GruDataInfo {
    int64_t T;                   //  时间步数
    int64_t B;                   //  最大batch数
    int64_t I;                   //  输入特征维度 inputSize
    int64_t H;                   //  隐层维度 hiddenSize
    int64_t L;                   //  层数 numLayers
    int64_t D;                   //  方向 1=单向  2=双向
    int64_t groupLen;            //  每组参数个数 有偏置=4(w_ih, w_hh, b_ih, b_hh) 无偏置=2
    int64_t LD;                  //  L * D
    ge::DataType dtype;          //  数据类型
    const aclTensor* batchSizes; //  batchSize数组
    const aclTensor* lastResult; //  上一层的推理结果，作为下一层的输入
    int64_t totalValidSteps;     //  sum(batch_size)
};
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t INPUT_DIMS = 3;  //  输入tensor维度数
static const int64_t WEIGHT_DIMS = 2; //  权重tensor维度数
static const int64_t BIAS_DIMS = 1;   //  偏置tensor维度数
static const int64_t INIT_DIMS = 3;   //  初始隐状态维度数
static const int64_t OUTPUT_DIMS = 3; //  输出tensor维度数
static const int64_t INDEX_0 = 0;
static const int64_t INDEX_1 = 1;
static const int64_t INDEX_2 = 2;
static const int64_t INDEX_3 = 3;
static const int64_t INDEX_4 = 4;
static const int64_t GRU_GATE_NUM = 3;
static const int64_t BIAS_PARAM_FACTOR = 2;     //  含偏置时参数个数乘数(input bias + hidden bias)
static const int64_t BIDIRECTIONAL_DIR_NUM = 2; //  双向GRU的方向数
static const size_t CONCAT_MAX_NUM = 32;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> INT_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};

//  L0失败返回
auto gruNullptrInner = std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

static inline bool CheckNotNull(const aclTensor* input, const aclTensorList* params, const aclTensor* hx, bool train,
                                aclTensor* output, aclTensor* hy, aclTensorList* rOut, aclTensorList* zOut,
                                aclTensorList* nOut, aclTensorList* hnOut, aclTensorList* hOut)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(params, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(hy, return false);
    if (train) {
        OP_CHECK_NULL(rOut, return false);
        OP_CHECK_NULL(zOut, return false);
        OP_CHECK_NULL(nOut, return false);
        OP_CHECK_NULL(hnOut, return false);
        OP_CHECK_NULL(hOut, return false);
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* input, const aclTensorList* params, const aclTensor* hx, bool train,
                                   aclTensor* output, aclTensor* hy, aclTensorList* rOut, aclTensorList* zOut,
                                   aclTensorList* nOut, aclTensorList* hnOut, aclTensorList* hOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
    auto data_type = input->GetDataType();

    for (uint64_t i = 0; i < params->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*params)[i], data_type, return false);
    }

    if (hx != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(hx, data_type, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(output, data_type, return false);
    OP_CHECK_DTYPE_NOT_MATCH(hy, data_type, return false);

    if (train) {
        for (uint64_t i = 0; i < rOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*rOut)[i], data_type, return false);
        }
        for (uint64_t i = 0; i < zOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*zOut)[i], data_type, return false);
        }
        for (uint64_t i = 0; i < nOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*nOut)[i], data_type, return false);
        }
        for (uint64_t i = 0; i < hnOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*hnOut)[i], data_type, return false);
        }
        for (uint64_t i = 0; i < hOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*hOut)[i], data_type, return false);
        }
    }
    return true;
}

//  校验tensorList长度
static bool CheckDimsSize(const aclTensorList* params, const aclTensor* hx, bool hasBias, int64_t numLayers, bool train,
                          bool bidirection, aclTensorList* rOut, aclTensorList* zOut, aclTensorList* nOut,
                          aclTensorList* hnOut, aclTensorList* hOut)
{
    uint64_t dScale = bidirection ? 2 : 1;
    uint64_t bScale = hasBias ? 2 : 1;
    uint64_t output_nums = dScale * numLayers;
    uint64_t param_nums = 2 * bScale * dScale * numLayers;

    if (params->Size() != param_nums) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The number of tensors required for the params lists should be %lu, but %lu was obtained.", param_nums,
                params->Size());
        return false;
    }
    if (train) {
        if (rOut->Size() != output_nums) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The number of tensors required for the rOut lists should be %lu, but %lu was obtained.",
                    output_nums, rOut->Size());
            return false;
        }
        if (zOut->Size() != output_nums) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The number of tensors required for the zOut lists should be %lu, but %lu was obtained.",
                    output_nums, zOut->Size());
            return false;
        }
        if (nOut->Size() != output_nums) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The number of tensors required for the nOut lists should be %lu, but %lu was obtained.",
                    output_nums, nOut->Size());
            return false;
        }
        if (hnOut->Size() != output_nums) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The number of tensors required for the hnOut lists should be %lu, but %lu was obtained.",
                    output_nums, hnOut->Size());
            return false;
        }
        if (hOut->Size() != output_nums) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The number of tensors required for the hOut lists should be %lu, but %lu was obtained.",
                    output_nums, hOut->Size());
            return false;
        }
    }
    return true;
}

//  校验tensor维度
static bool CheckDims(const aclTensor* input, const aclTensorList* params, const aclTensor* hx, bool hasBias,
                      int64_t numLayers, bool train, bool bidirection, aclTensor* output, aclTensor* hy,
                      aclTensorList* rOut, aclTensorList* zOut, aclTensorList* nOut, aclTensorList* hnOut,
                      aclTensorList* hOut)
{
    OP_CHECK_WRONG_DIMENSION(input, INPUT_DIMS, return false);
    uint64_t bScale = hasBias ? 2 : 1;
    uint64_t dScale = bidirection ? 2 : 1;
    uint64_t oneLayerParams = 2 * bScale * dScale;
    for (uint64_t i = 0; i < (uint64_t)numLayers; i++) {
        for (uint64_t j = 0; j < dScale; j++) {
            uint64_t offsets = i * oneLayerParams + j * oneLayerParams / 2;
            OP_CHECK_WRONG_DIMENSION((*params)[offsets], WEIGHT_DIMS, return false);
            OP_CHECK_WRONG_DIMENSION((*params)[offsets + 1], WEIGHT_DIMS, return false);
            if (hasBias) {
                OP_CHECK_WRONG_DIMENSION((*params)[offsets + 2], BIAS_DIMS, return false);
                OP_CHECK_WRONG_DIMENSION((*params)[offsets + 3], BIAS_DIMS, return false);
            }
        }
    }

    if (hx != nullptr) {
        OP_CHECK_WRONG_DIMENSION(hx, INIT_DIMS, return false);
    }

    OP_CHECK_WRONG_DIMENSION(output, OUTPUT_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(hy, OUTPUT_DIMS, return false);

    if (train) {
        for (uint64_t i = 0; i < rOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*rOut)[i], OUTPUT_DIMS, return false);
        }
        for (uint64_t i = 0; i < zOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*zOut)[i], OUTPUT_DIMS, return false);
        }
        for (uint64_t i = 0; i < nOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*nOut)[i], OUTPUT_DIMS, return false);
        }
        for (uint64_t i = 0; i < hnOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*hnOut)[i], OUTPUT_DIMS, return false);
        }
        for (uint64_t i = 0; i < hOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*hOut)[i], OUTPUT_DIMS, return false);
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* input, const aclTensorList* params, const aclTensor* hx, bool hasBias,
                       int64_t numLayers, bool train, bool bidirection, bool batchFirst, aclTensor* output,
                       aclTensor* hy, aclTensorList* rOut, aclTensorList* zOut, aclTensorList* nOut,
                       aclTensorList* hnOut, aclTensorList* hOut)
{
    //  根据batchFirst确定T和B的位置
    auto timeStep = batchFirst ? input->GetViewShape().GetDim(1) : input->GetViewShape().GetDim(0);
    auto batchSize = batchFirst ? input->GetViewShape().GetDim(0) : input->GetViewShape().GetDim(1);
    auto inputSize = input->GetViewShape().GetDim(2);
    //  W_ih[0] = 3H
    auto hiddenSize = (*params)[0]->GetViewShape().GetDim(0) / GRU_GATE_NUM;
    auto curLayerInputSize = inputSize;
    uint64_t dScale = bidirection ? 2 : 1;
    auto bScale = hasBias ? 2 : 1;
    uint64_t oneLayerParams = 2 * bScale * dScale;

    //  逐层校验
    for (uint64_t i = 0; i < (uint64_t)numLayers; i++) {
        op::Shape expWiShape = {GRU_GATE_NUM * hiddenSize, curLayerInputSize};
        op::Shape expWhShape = {GRU_GATE_NUM * hiddenSize, hiddenSize};
        op::Shape expBShape = {GRU_GATE_NUM * hiddenSize};
        for (uint64_t j = 0; j < dScale; j++) {
            uint64_t offsets = i * oneLayerParams + j * oneLayerParams / 2;
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets], expWiShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets + 1], expWhShape, return false);
            if (hasBias) {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets + 2], expBShape, return false);
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets + 3], expBShape, return false);
            }
        }
        //  第2层起，输入shape变成[3H, D*H]
        curLayerInputSize = dScale * hiddenSize;
    }

    if (hx != nullptr) {
        op::Shape expHxShape = {numLayers * dScale, batchSize, hiddenSize};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(hx, expHxShape, return false);
    }
    op::Shape expOutputShape = {timeStep, batchSize, dScale * hiddenSize};
    if (batchFirst) {
        expOutputShape = {batchSize, timeStep, dScale * hiddenSize};
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, expOutputShape, return false);

    op::Shape expHyShape = {numLayers * dScale, batchSize, hiddenSize};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(hy, expHyShape, return false);

    if (train) {
        for (uint64_t i = 0; i < rOut->Size(); i++) {
            op::Shape expShape = {timeStep, batchSize, hiddenSize};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*rOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*zOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*nOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*hnOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*hOut)[i], expShape, return false);
        }
    }
    return true;
}

//  标准模式参数校验函数
static aclnnStatus CheckParams(const aclTensor* input, const aclTensorList* params, const aclTensor* hx, bool hasBias,
                               int64_t numLayers, bool train, bool bidirection, bool batchFirst, aclTensor* output,
                               aclTensor* hy, aclTensorList* rOut, aclTensorList* zOut, aclTensorList* nOut,
                               aclTensorList* hnOut, aclTensorList* hOut)
{
    CHECK_RET(CheckNotNull(input, params, hx, train, output, hy, rOut, zOut, nOut, hnOut, hOut),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(input, params, hx, train, output, hy, rOut, zOut, nOut, hnOut, hOut),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimsSize(params, hx, hasBias, numLayers, train, bidirection, rOut, zOut, nOut, hnOut, hOut),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckDims(input, params, hx, hasBias, numLayers, train, bidirection, output, hy, rOut, zOut, nOut, hnOut, hOut),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(input, params, hx, hasBias, numLayers, train, bidirection, batchFirst, output, hy, rOut, zOut,
                         nOut, hnOut, hOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

//  通用工具函数
static aclTensorList* ProcessInputContiguous(const aclTensorList* inputParam, aclOpExecutor* executor)
{
    std::vector<const aclTensor*> inputVec;
    for (size_t i = 0; i < inputParam->Size(); ++i) {
        auto secondContiguous = l0op::Contiguous((*inputParam)[i], executor);
        inputVec.push_back(secondContiguous);
    }
    auto inputContiguous = executor->AllocTensorList(inputVec.data(), inputVec.size());
    return inputContiguous;
}

//  拼接tensor
static const aclTensor* SplitToConcat(std::vector<const aclTensor*> tensorListA, int64_t dim, aclOpExecutor* executor)
{
    if (tensorListA.size() == 1) {
        return tensorListA[0];
    }

    //  多轮归并
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
    CHECK_RET(!tensorListA.empty(), nullptr);
    return tensorListA.front();
}

//  校验tensorList长度
static aclnnStatus CheckTensorListLength(const aclTensorList* tensorList, int64_t length, const char* name)
{
    OP_CHECK(int64_t(tensorList->Size()) == length,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "length of %s should be %lld, but %lld was obtained.", name, length,
                     tensorList->Size()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListNullptr(const aclTensorList* tensorList)
{
    OP_CHECK_NULL(tensorList, return ACLNN_ERR_PARAM_NULLPTR);
    for (int64_t idx = INDEX_0; idx < int64_t(tensorList->Size()); idx++) {
        OP_CHECK_NULL((*tensorList)[idx], return ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListShape(const aclTensorList* tensorList, op::Shape shape, const char* name)
{
    for (int64_t idx = INDEX_0; idx < int64_t(tensorList->Size()); idx++) {
        auto tensor = (*tensorList)[idx];
        if (tensor->GetViewShape() != shape) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor in %s to have same size as %s, but got %s.", name,
                    op::ToString(shape).GetString(), op::ToString(tensor->GetViewShape()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListDtype(const aclTensorList* tensorList, op::DataType dtype, const char* name)
{
    for (int64_t idx = INDEX_0; idx < int64_t(tensorList->Size()); idx++) {
        auto tensor = (*tensorList)[idx];
        if (tensor->GetDataType() != dtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor in %s to have dtype as %s, but got %s.", name,
                    op::ToString(dtype).GetString(), op::ToString(tensor->GetDataType()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDataParamsNullptr(const GruDataParamsIn& inputs, const GruDataParamsOut& outputs)
{
    OP_CHECK_NULL(inputs.input, return ACLNN_ERR_PARAM_NULLPTR);
    auto ret = CheckTensorListNullptr(inputs.params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (inputs.hx) {
        OP_CHECK_NULL(inputs.hx, return ACLNN_ERR_PARAM_NULLPTR);
    }

    OP_CHECK_NULL(outputs.output, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(outputs.hy, return ACLNN_ERR_PARAM_NULLPTR);
    if (inputs.train) {
        ret = CheckTensorListNullptr(outputs.rOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.zOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.nOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.hnOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.hOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

//  PackedSequence模式dim和list长度校验
static aclnnStatus CheckDataDimsAndListLength(const GruDataParamsIn& inputs, const GruDataParamsOut& outputs,
                                              GruDataInfo& info)
{
    aclnnStatus ret;

    OP_CHECK_WRONG_DIMENSION(inputs.input, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    ret = CheckTensorListLength(inputs.params, info.groupLen * info.LD, "params");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    for (int64_t group = 0; group < info.LD; group++) {
        int64_t currOffset = group * info.groupLen;
        OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_0], INDEX_2, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_1], INDEX_2, return ACLNN_ERR_PARAM_INVALID);
        if (inputs.hasBias) {
            OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_2], INDEX_1, return ACLNN_ERR_PARAM_INVALID);
            OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_3], INDEX_1, return ACLNN_ERR_PARAM_INVALID);
        }
    }
    OP_CHECK_WRONG_DIMENSION(inputs.batchSizes, INDEX_1, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_WRONG_DIMENSION(outputs.output, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(outputs.hy, INDEX_3, return ACLNN_ERR_PARAM_INVALID);
    if (inputs.train) {
        ret = CheckTensorListLength(outputs.rOut, info.LD, "rOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.zOut, info.LD, "zOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.nOut, info.LD, "nOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.hnOut, info.LD, "hnOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.hOut, info.LD, "hOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

//  PackedSequence模式的shape校验
static aclnnStatus CheckDataShapes(const GruDataParamsIn& inputs, const GruDataParamsOut& outputs, GruDataInfo& info)
{
    aclnnStatus ret;

    auto data2dShape = inputs.input->GetViewShape();
    info.T = inputs.batchSizes->Numel();
    OP_CHECK(info.T != INDEX_0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batchSizes should not be empty wen it is a non-null pointer."),
             return ACLNN_ERR_PARAM_INVALID);

    info.totalValidSteps = data2dShape.GetDim(INDEX_0);

    auto outputShape = outputs.output->GetViewShape();
    info.B = outputs.hy->GetViewShape().GetDim(INDEX_1);
    info.H = outputShape.GetDim(INDEX_1);
    OP_CHECK(info.B > INDEX_0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch size should be positive, but %lld wad obtained.", info.B),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(info.totalValidSteps <= info.T * info.B,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input.shape[0]=%lld should not exceed T*B=%lld (T=%lld, B=%lld).",
                     info.totalValidSteps, info.T * info.B, info.T, info.B),
             return ACLNN_ERR_PARAM_INVALID);

    info.I = data2dShape.GetDim(INDEX_1);
    if (inputs.bidirection) {
        OP_CHECK(info.H % 2 == INDEX_0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output last dim should be even in bidirection scenarios."),
                 return ACLNN_ERR_PARAM_INVALID);
        info.H = info.H / INDEX_2;
    }

    op::Shape weightIhFirstLayerShape = {GRU_GATE_NUM * info.H, info.I};
    op::Shape weightIhShape = {GRU_GATE_NUM * info.H, info.H * info.D};
    op::Shape weightHhShape = {GRU_GATE_NUM * info.H, info.H};
    op::Shape biasShape = {GRU_GATE_NUM * info.H};
    op::Shape hxShape = {info.D * info.L, info.B, info.H};
    op::Shape hyShape = {info.LD, info.B, info.H};

    for (int64_t group = INDEX_0; group < info.LD; group++) {
        int64_t currOffset = group * info.groupLen;
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.params)[currOffset + INDEX_0],
                                                    ((group < info.D) ? weightIhFirstLayerShape : weightIhShape),
                                                    return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.params)[currOffset + INDEX_1], weightHhShape,
                                                    return ACLNN_ERR_PARAM_INVALID);
        if (inputs.hasBias) {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.params)[currOffset + INDEX_2], biasShape,
                                                        return ACLNN_ERR_PARAM_INVALID);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.params)[currOffset + INDEX_3], biasShape,
                                                        return ACLNN_ERR_PARAM_INVALID);
        }
    }

    if (inputs.hx) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputs.hx, hxShape, return ACLNN_ERR_PARAM_INVALID);
    }

    op::Shape output2dShape = {info.totalValidSteps, info.H * info.D};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputs.output, output2dShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputs.hy, hyShape, return ACLNN_ERR_PARAM_INVALID);

    if (inputs.train) {
        op::Shape gateShape2d = {info.totalValidSteps, info.H};
        ret = CheckTensorListShape(outputs.rOut, gateShape2d, "rOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.zOut, gateShape2d, "zOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.nOut, gateShape2d, "nOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.hnOut, gateShape2d, "hnOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.hOut, gateShape2d, "hOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

//  PackedSequence模式的数据类型校验
static aclnnStatus CheckDataDtypes(const GruDataParamsIn& inputs, const GruDataParamsOut& outputs, GruDataInfo& info)
{
    aclnnStatus ret;

    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.input, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    info.dtype = inputs.input->GetDataType();
    ret = CheckTensorListDtype(inputs.params, info.dtype, "params");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (inputs.hx) {
        OP_CHECK_DTYPE_NOT_MATCH(inputs.hx, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.batchSizes, INT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_MATCH(outputs.output, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(outputs.hy, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    if (inputs.train) {
        ret = CheckTensorListDtype(outputs.rOut, info.dtype, "rOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListDtype(outputs.zOut, info.dtype, "zOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListDtype(outputs.nOut, info.dtype, "nOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListDtype(outputs.hnOut, info.dtype, "hnOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListDtype(outputs.hOut, info.dtype, "hOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

//  PackedSequence模式参数校验
static aclnnStatus CheckDataParamsValid(const GruDataParamsIn& inputs, const GruDataParamsOut& outputs,
                                        GruDataInfo& info)
{
    aclnnStatus ret;
    ret = CheckDataParamsNullptr(inputs, outputs);
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ret, "CheckDataParamsNullptr failed, certain incoming nullptrs may be invalid."), return ret);

    OP_CHECK(inputs.numLayers >= INDEX_1,
             OP_LOGE(ret, "numLayers should be a positive integer, but %lld was obtained.", inputs.numLayers),
             return ACLNN_ERR_PARAM_NULLPTR);

    if (inputs.input->IsEmpty())
        return ACLNN_SUCCESS;

    //  填充维度信息
    info.L = inputs.numLayers;
    info.D = inputs.bidirection ? INDEX_2 : INDEX_1;
    info.groupLen = inputs.hasBias ? INDEX_4 : INDEX_2;
    info.LD = info.L * info.D;

    ret = CheckDataDimsAndListLength(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckDataShapes(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckDataDtypes(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

//  收集各层各方向最终隐状态
//  L*D个(1,B,H) -> (L*D,B,H) hy
static aclnnStatus ProcessViewCopyOutputH(std::vector<const aclTensor*>& hOut, aclTensor* hy, aclOpExecutor* executor)
{
    auto outputHConcat = SplitToConcat(hOut, 0, executor);
    auto viewCopyResultOutputH = l0op::ViewCopy(outputHConcat, hy, executor);
    CHECK_RET(viewCopyResultOutputH != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

//  PackedSequence模式 准备当前层\方向的输入参数
static aclnnStatus GruDataProcessParams(const GruDataParamsIn& inputs, const GruDataInfo& info, int64_t layerIdx,
                                        int64_t directIdx, GruBaseOpInputs& baseIn, aclOpExecutor* executor)
{
    //  从hx中slice出当前层\方向的h0 (L*D, B, H) ->(B, H)
    if (inputs.hx) {
        const int64_t offsetData[] = {layerIdx * info.D + directIdx, 0, 0};
        aclIntArray* offsets = executor->AllocIntArray(offsetData, 3);
        CHECK_RET(offsets != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t sizeData[] = {1, info.B, info.H};
        aclIntArray* size = executor->AllocIntArray(sizeData, 3);
        CHECK_RET(size != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* initH3d = l0op::Slice(inputs.hx, offsets, size, executor);
        CHECK_RET(initH3d != nullptr, ACLNN_ERR_INNER_NULLPTR);
        op::Shape initH2dShape = {info.B, info.H};
        baseIn.initHOptional = l0op::Reshape(initH3d, initH2dShape, executor);
        CHECK_RET(baseIn.initHOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    //  从params中取出当前组(层*方向)的权重
    //  W_ih[3H,I]->[I,3H] W_hh[3H,H]->[H,3H]
    int64_t currOffset = (layerIdx * info.D + directIdx) * info.groupLen;
    const int64_t permData[] = {INDEX_1, INDEX_0};
    aclIntArray* perm = executor->AllocIntArray(permData, INDEX_2);
    CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    baseIn.weightInput = l0op::Transpose((*inputs.params)[currOffset + INDEX_0], perm, executor);
    CHECK_RET(baseIn.weightInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    baseIn.weightHidden = l0op::Transpose((*inputs.params)[currOffset + INDEX_1], perm, executor);
    CHECK_RET(baseIn.weightHidden != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (inputs.hasBias) {
        baseIn.biasInput = (*inputs.params)[currOffset + INDEX_2];
        baseIn.biasHidden = (*inputs.params)[currOffset + INDEX_3];
    }
    return ACLNN_SUCCESS;
}

//  PackedSequence模式调用Gru
static aclnnStatus CallGruBaseOp(const GruBaseOpInputs& baseIn, const GruDataInfo& info, GruBaseOpOutputs& baseOut,
                                 aclOpExecutor* executor)
{
    //  y和门输出为紧凑shape [sum(batch_size), H]  output_h也紧凑[B, H]
    op::Shape compactShape = {info.totalValidSteps, info.H};
    op::Shape hyShape = {info.B, info.H};
    baseOut.l0_yOut = executor->AllocTensor(compactShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_outputHOut = executor->AllocTensor(hyShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_rOut = executor->AllocTensor(compactShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_zOut = executor->AllocTensor(compactShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_nOut = executor->AllocTensor(compactShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_nHOut = executor->AllocTensor(compactShape, info.dtype, op::Format::FORMAT_ND);

    auto ret = l0op::Gru(baseIn.input, baseIn.weightInput, baseIn.weightHidden, baseIn.biasInput, baseIn.biasHidden,
                         baseIn.seqLengthOptional, baseIn.initHOptional, baseIn.direction, baseIn.isTraining,
                         baseOut.l0_yOut, baseOut.l0_outputHOut, baseOut.l0_rOut, baseOut.l0_zOut, baseOut.l0_nOut,
                         baseOut.l0_nHOut, executor);
    CHECK_RET(ret != gruNullptrInner, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

//  PackedSequence模式 获取单层单向GRU输出
static aclnnStatus GruDataGetBaseOpOut(const GruDataParamsIn& inputs, const GruDataInfo& info, int64_t layerIdx,
                                       int64_t directIdx, std::vector<GruBaseOpOutputs>& baseOutVec,
                                       aclOpExecutor* executor)
{
    aclnnStatus ret;

    //  构建L0输入
    GruBaseOpInputs baseIn = {
        info.lastResult, nullptr,         nullptr, nullptr,
        nullptr,         info.batchSizes, nullptr, (directIdx == INDEX_0) ? "UNIDIRECTIONAL" : "REDIRECTIONAL",
        inputs.train};
    ret = GruDataProcessParams(inputs, info, layerIdx, directIdx, baseIn, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    //  调用GRU
    GruBaseOpOutputs baseOut = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    ret = CallGruBaseOp(baseIn, info, baseOut, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    baseOutVec.push_back(baseOut);

    return ACLNN_SUCCESS;
}

//  PackedSequence模式，拷贝Gru输出到L2 L0:2D(T*B,H) -> L2(sum(batch_size), D*H)
static aclnnStatus GruDataGetParamsOut(const GruDataParamsIn& inputs, const GruDataInfo& info,
                                       const std::vector<GruBaseOpOutputs>& baseOutVec, GruDataParamsOut& outputs,
                                       aclOpExecutor* executor)
{
    const aclTensor* res = nullptr;

    res = l0op::ViewCopy(info.lastResult, outputs.output, executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //  从各层各方向的outputHOut中取出最后时刻的h，拼成hy
    //  outputHOut紧凑[B,H]，直接Reshape为(1,B,H)
    op::Shape size3dShape = {1, info.B, info.H};
    //  所有层*方向 切片+拼接
    std::vector<const aclTensor*> hyVec;
    for (int64_t idx = INDEX_0; idx < info.LD; idx++) {
        auto hyOut = l0op::Reshape(baseOutVec.at(idx).l0_outputHOut, size3dShape, executor);
        CHECK_RET(hyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        hyVec.push_back(hyOut);
    }
    auto hy = SplitToConcat(hyVec, INDEX_0, executor);
    CHECK_RET(hy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    res = l0op::ViewCopy(hy, outputs.hy, executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (inputs.train) {
        for (int64_t idx = INDEX_0; idx < info.LD; idx++) {
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_rOut, (*outputs.rOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_zOut, (*outputs.zOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_nOut, (*outputs.nOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_nHOut, (*outputs.hnOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_yOut, (*outputs.hOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    return ACLNN_SUCCESS;
}

//  PackedSequence模式的主运行函数 L0 y输出已是紧凑2D
static aclnnStatus GruDataRun(const GruDataParamsIn& inputs, GruDataInfo& info, GruDataParamsOut& outputs,
                              aclOpExecutor* executor)
{
    aclnnStatus ret;

    std::vector<GruBaseOpOutputs> baseOutVec;
    info.lastResult = inputs.input;
    for (int64_t layerIdx = INDEX_0; layerIdx < info.L; layerIdx++) {
        for (int64_t directIdx = INDEX_0; directIdx < info.D; directIdx++) {
            ret = GruDataGetBaseOpOut(inputs, info, layerIdx, directIdx, baseOutVec, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
        //  拼接当前层输出，作为下一层输入
        int64_t offset = info.D * layerIdx;
        if (info.D == INDEX_1) {
            //  单向
            info.lastResult = baseOutVec.at(offset).l0_yOut;
        } else {
            std::vector<const aclTensor*> resVec;
            resVec.push_back(baseOutVec.at(offset).l0_yOut);
            resVec.push_back(baseOutVec.at(offset + INDEX_1).l0_yOut);
            auto resTensorList = executor->AllocTensorList(resVec.data(), resVec.size());
            info.lastResult = l0op::ConcatD(resTensorList, INDEX_1, executor);
            CHECK_RET(info.lastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    ret = GruDataGetParamsOut(inputs, info, baseOutVec, outputs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

//  PackedSequence模式入库函数
static aclnnStatus GruDataGetWorkspaceSize(GruDataParamsIn& inputs, GruDataParamsOut& outputs, uint64_t* workspaceSize,
                                           aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus ret;
    GruDataInfo info;

    //  参数校验，填充info信息
    ret = CheckDataParamsValid(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    //  空tensor处理
    if (inputs.input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    //  转连续内存
    auto inputCtg = l0op::Contiguous(inputs.input, uniqueExecutor.get());
    auto paramsCtg = ProcessInputContiguous(inputs.params, uniqueExecutor.get());
    auto hxCtg = l0op::Contiguous(inputs.hx, uniqueExecutor.get());
    auto batchSizeCtg = l0op::Contiguous(inputs.batchSizes, uniqueExecutor.get());
    inputs.input = inputCtg;
    inputs.params = paramsCtg;
    inputs.hx = hxCtg;
    inputs.batchSizes = batchSizeCtg;

    info.batchSizes = inputs.batchSizes;

    //  执行计算
    ret = GruDataRun(inputs, info, outputs, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

//  门输出
//  输出tensorList先层后方向排布 [L0-F, L0-B, L1-F, L1-B,...]
static aclnnStatus ProcessViewCopy(std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                              const aclTensor*, const aclTensor*>
                                       layerResult,
                                   const aclTensorList* rOut, const aclTensorList* zOut, const aclTensorList* nOut,
                                   const aclTensorList* hnOut, const aclTensorList* hOut, int64_t numLayers,
                                   bool bidirection, const char* direction, aclOpExecutor* executor)
{
    auto paramsNumSingleLayer = bidirection ? 2 : 1;
    auto directionStart = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : 1;
    auto idx = paramsNumSingleLayer * numLayers + directionStart;

    auto viewCopyResultR = l0op::ViewCopy(std::get<2>(layerResult), (*rOut)[idx], executor);
    CHECK_RET(viewCopyResultR != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultZ = l0op::ViewCopy(std::get<3>(layerResult), (*zOut)[idx], executor);
    CHECK_RET(viewCopyResultZ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultN = l0op::ViewCopy(std::get<4>(layerResult), (*nOut)[idx], executor);
    CHECK_RET(viewCopyResultN != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultNH = l0op::ViewCopy(std::get<5>(layerResult), (*hnOut)[idx], executor);
    CHECK_RET(viewCopyResultNH != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultH = l0op::ViewCopy(std::get<1>(layerResult), (*hOut)[idx], executor);
    CHECK_RET(viewCopyResultH != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

//  从outputHOut中提取最后时刻的隐状态
//  outputHOut (T, B, H) -> (1, B, H)
static aclnnStatus ProcessOutputH(std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                             const aclTensor*, const aclTensor*>
                                      layerResult,
                                  std::vector<const aclTensor*>& hyVector, const char* direction,
                                  aclOpExecutor* executor)
{
    int64_t numStep = std::get<0>(layerResult)->GetViewShape().GetDim(0);
    int64_t batch = std::get<0>(layerResult)->GetViewShape().GetDim(1);
    int64_t hidden = std::get<0>(layerResult)->GetViewShape().GetDim(2);

    int64_t copyStep = strcmp(direction, "UNIDIRECTIONAL") == 0 ? numStep - 1 : 0;

    const int64_t offsetData[] = {copyStep, 0, 0};
    aclIntArray* offsets = executor->AllocIntArray(offsetData, 3);
    const int64_t sizeData[] = {1, batch, hidden};
    aclIntArray* size = executor->AllocIntArray(sizeData, 3);

    auto thOutput = l0op::Slice(std::get<1>(layerResult), offsets, size, executor);
    hyVector.emplace_back(thOutput);

    return ACLNN_SUCCESS;
}

//  单层单向GRU
std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*>
GruSingleLayerDirec(const aclTensor* input, const aclTensorList* params, const aclTensor* hx, aclTensor* yOutDirec,
                    aclTensor* outputHOutDirec, aclTensor* rOutDirec, aclTensor* zOutDirec, aclTensor* nOutDirec,
                    aclTensor* nHOutDirec, const char* direction, bool bidirection, bool train, int64_t num_layers,
                    bool hasBias, aclOpExecutor* executor)
{
    //  计算当前层单向的参数个数
    auto oneLayerParams = bidirection ? 4 : 2;
    oneLayerParams = hasBias ? oneLayerParams * BIAS_PARAM_FACTOR : oneLayerParams;
    //  正向从0开始，反向从半组开始
    auto weightStart = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : oneLayerParams / 2;
    auto paramsOffsets = oneLayerParams * num_layers + weightStart;

    //  取当前层/方向的权重
    //  weight[3H,I] -> [I,3H]  [3H,H]->[H,3H]
    const int64_t permData[] = {INDEX_1, INDEX_0};
    aclIntArray* perm = executor->AllocIntArray(permData, INDEX_2);
    OP_CHECK_NULL(perm, return gruNullptrInner);

    const aclTensor* weightInput = l0op::Transpose((*params)[paramsOffsets], perm, executor);
    OP_CHECK_NULL(weightInput, return gruNullptrInner);
    const aclTensor* weightHidden = l0op::Transpose((*params)[paramsOffsets + 1], perm, executor);
    OP_CHECK_NULL(weightHidden, return gruNullptrInner);

    const aclTensor* biasInput = nullptr;
    const aclTensor* biasHidden = nullptr;
    if (hasBias) {
        biasInput = (*params)[paramsOffsets + 2];
        biasHidden = (*params)[paramsOffsets + 3];
    }

    //  从hx中Slice出当前层/方向的初始隐状态h0
    //  hx(L*D, B, H) -> 2D(B, H)
    const aclTensor* initH = nullptr;
    if (hx != nullptr) {
        auto batch = hx->GetViewShape().GetDim(1);
        auto hidden = hx->GetViewShape().GetDim(2);
        auto oneLayerInit = bidirection ? 2 : 1;
        auto initStart = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : 1;

        const int64_t offsetData[] = {oneLayerInit * num_layers + initStart, 0, 0};
        aclIntArray* offsets = executor->AllocIntArray(offsetData, INDEX_3);
        OP_CHECK_NULL(offsets, return gruNullptrInner);
        const int64_t sizeData[] = {1, batch, hidden};
        aclIntArray* size = executor->AllocIntArray(sizeData, INDEX_3);
        OP_CHECK_NULL(size, return gruNullptrInner);
        const aclTensor* initH3d = l0op::Slice(hx, offsets, size, executor);
        OP_CHECK_NULL(initH3d, return gruNullptrInner);
        op::Shape initH2dShape = {batch, hidden};
        initH = l0op::Reshape(initH3d, initH2dShape, executor);
        OP_CHECK_NULL(initH, return gruNullptrInner);
    }

    //  无batch_size
    auto layerResult = l0op::Gru(input, weightInput, weightHidden, biasInput, biasHidden, nullptr, initH, direction,
                                 train, yOutDirec, outputHOutDirec, rOutDirec, zOutDirec, nOutDirec, nHOutDirec,
                                 executor);

    OP_CHECK_NULL(std::get<0>(layerResult), return gruNullptrInner);
    OP_CHECK_NULL(std::get<1>(layerResult), return gruNullptrInner);
    OP_CHECK_NULL(std::get<2>(layerResult), return gruNullptrInner);
    OP_CHECK_NULL(std::get<3>(layerResult), return gruNullptrInner);
    OP_CHECK_NULL(std::get<4>(layerResult), return gruNullptrInner);
    OP_CHECK_NULL(std::get<5>(layerResult), return gruNullptrInner);

    return layerResult;
}

aclnnStatus aclnnGRUGetWorkspaceSize(const aclTensor* input, const aclTensorList* params, const aclTensor* hx,
                                     const aclTensor* batchSizes, bool hasBias, int64_t numLayers, double dropout,
                                     bool train, bool bidirection, bool batchFirst, aclTensor* output, aclTensor* hy,
                                     aclTensorList* rOut, aclTensorList* zOut, aclTensorList* nOut,
                                     aclTensorList* hnOut, aclTensorList* hOut, uint64_t* workspaceSize,
                                     aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGRU,
                   DFX_IN(input, params, hx, batchSizes, hasBias, numLayers, dropout, train, bidirection, batchFirst),
                   DFX_OUT(output, hy, rOut, zOut, nOut, hnOut, hOut));

    //  判断是否进入PackedSequence模式
    if (batchSizes != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GRU PackedSequence(variable-length) mode is not supported yet, please use "
                                         "fixed-length 3D input with batchSizes=nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    //  定长模式
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空tensor处理
    if (input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，参数检查
    auto ret = CheckParams(input, params, hx, hasBias, numLayers, train, bidirection, batchFirst, output, hy, rOut,
                           zOut, nOut, hnOut, hOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 转连续内存布局
    // input
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    // params
    auto paramsContiguous = ProcessInputContiguous(params, uniqueExecutor.get());
    CHECK_RET(paramsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // hx
    auto hxContiguous = hx;
    if (hx != nullptr) {
        hxContiguous = l0op::Contiguous(hx, uniqueExecutor.get());
        CHECK_RET(hxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // batchFirst转置：(B, T, I) -> (T, B, I)
    auto curInput = inputContiguous;
    if (batchFirst) {
        std::vector<int64_t> perm = {1, 0, 2};
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
        curInput = l0op::Transpose(inputContiguous, valuePerm, uniqueExecutor.get());
    }

    // 计算单方向输出的hiddenSize和shape
    int64_t hiddenSize = output->GetViewShape().GetDim(2);
    hiddenSize = bidirection ? hiddenSize / BIDIRECTIONAL_DIR_NUM : hiddenSize;
    op::Shape outShape = {curInput->GetViewShape().GetDim(0), curInput->GetViewShape().GetDim(1), hiddenSize};

    std::vector<const aclTensor*> hyVector = {};

    //  训练模式分支
    if (train) {
        for (uint64_t i = 0U; i < uint64_t(numLayers); ++i) {
            auto yOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(yOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto outputHOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                       op::Format::FORMAT_ND);
            CHECK_RET(outputHOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto rOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(rOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto zOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(zOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto nOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(nOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto nHOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                  op::Format::FORMAT_ND);
            CHECK_RET(nHOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto layerResultForward = GruSingleLayerDirec(
                curInput, paramsContiguous, hxContiguous, yOutForward, outputHOutForward, rOutForward, zOutForward,
                nOutForward, nHOutForward, "UNIDIRECTIONAL", bidirection, train, i, hasBias, uniqueExecutor.get());

            ProcessViewCopy(layerResultForward, rOut, zOut, nOut, hnOut, hOut, i, bidirection, "UNIDIRECTIONAL",
                            uniqueExecutor.get());
            ProcessOutputH(layerResultForward, hyVector, "UNIDIRECTIONAL", uniqueExecutor.get());

            if (bidirection) {
                auto yOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                      op::Format::FORMAT_ND);
                CHECK_RET(yOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto outputHOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                            op::Format::FORMAT_ND);
                CHECK_RET(outputHOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto rOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                      op::Format::FORMAT_ND);
                CHECK_RET(rOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto zOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                      op::Format::FORMAT_ND);
                CHECK_RET(zOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto nOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                      op::Format::FORMAT_ND);
                CHECK_RET(nOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto nHOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                       op::Format::FORMAT_ND);
                CHECK_RET(nHOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);

                auto layerResultBackward = GruSingleLayerDirec(curInput, paramsContiguous, hxContiguous, yOutBackward,
                                                               outputHOutBackward, rOutBackward, zOutBackward,
                                                               nOutBackward, nHOutBackward, "REDIRECTIONAL",
                                                               bidirection, train, i, hasBias, uniqueExecutor.get());

                op::FVector<const aclTensor*> inputConcat;
                inputConcat.emplace_back(std::get<0>(layerResultForward));
                inputConcat.emplace_back(std::get<0>(layerResultBackward));
                auto tensorListInput = uniqueExecutor.get()->AllocTensorList(inputConcat.data(), inputConcat.size());
                curInput = l0op::ConcatD(tensorListInput, INDEX_2, uniqueExecutor.get());
                ProcessViewCopy(layerResultBackward, rOut, zOut, nOut, hnOut, hOut, i, bidirection, "REDIRECTIONAL",
                                uniqueExecutor.get());
                // 从outputHout切出最后时刻的h -> hyVector
                ProcessOutputH(layerResultBackward, hyVector, "REDIRECTIONAL", uniqueExecutor.get());
            } else {
                curInput = std::get<0>(layerResultForward);
            }
        }

        auto outputY = curInput;
        if (batchFirst) {
            std::vector<int64_t> perm = {1, 0, 2};
            auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
            outputY = l0op::Transpose(curInput, valuePerm, uniqueExecutor.get());
        }
        auto viewCopyResultInput = l0op::ViewCopy(outputY, output, uniqueExecutor.get());
        CHECK_RET(viewCopyResultInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        ProcessViewCopyOutputH(hyVector, hy, uniqueExecutor.get());
        //  推理模式分支
        //  门输出tensor在循环外分配，所有层复用同一块
    } else {
        auto rOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(rOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto zOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(zOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto nOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(nOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto nHOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(nHOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);

        aclTensor* rOutBackward = nullptr;
        aclTensor* zOutBackward = nullptr;
        aclTensor* nOutBackward = nullptr;
        aclTensor* nHOutBackward = nullptr;

        if (bidirection) {
            rOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(rOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            zOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(zOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            nOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(nOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            nHOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(nHOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 多层
        for (uint64_t i = 0U; i < uint64_t(numLayers); ++i) {
            auto yOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(yOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto outputHOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                       op::Format::FORMAT_ND);
            CHECK_RET(outputHOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);

            //  单层单向 正向
            auto layerResultForward = GruSingleLayerDirec(
                curInput, paramsContiguous, hxContiguous, yOutForward, outputHOutForward, rOutForward, zOutForward,
                nOutForward, nHOutForward, "UNIDIRECTIONAL", bidirection, train, i, hasBias, uniqueExecutor.get());
            // 从outputHout切出最后时刻的h -> hyVector
            ProcessOutputH(layerResultForward, hyVector, "UNIDIRECTIONAL", uniqueExecutor.get());

            if (bidirection) {
                auto yOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                      op::Format::FORMAT_ND);
                CHECK_RET(yOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto outputHOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(),
                                                                            op::Format::FORMAT_ND);
                CHECK_RET(outputHOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);

                //  单层单向 反向
                auto layerResultBackward = GruSingleLayerDirec(curInput, paramsContiguous, hxContiguous, yOutBackward,
                                                               outputHOutBackward, rOutBackward, zOutBackward,
                                                               nOutBackward, nHOutBackward, "REDIRECTIONAL",
                                                               bidirection, train, i, hasBias, uniqueExecutor.get());

                op::FVector<const aclTensor*> inputConcat;
                inputConcat.emplace_back(std::get<0>(layerResultForward));
                inputConcat.emplace_back(std::get<0>(layerResultBackward));
                auto tensorListInput = uniqueExecutor.get()->AllocTensorList(inputConcat.data(), inputConcat.size());
                curInput = l0op::ConcatD(tensorListInput, INDEX_2, uniqueExecutor.get());
                // 从outputHout切出最后时刻的h -> hyVector
                ProcessOutputH(layerResultBackward, hyVector, "REDIRECTIONAL", uniqueExecutor.get());
            } else {
                curInput = std::get<0>(layerResultForward);
            }
        }
        auto outY = curInput;
        if (batchFirst) {
            std::vector<int64_t> perm = {1, 0, 2};
            auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
            outY = l0op::Transpose(curInput, valuePerm, uniqueExecutor.get());
        }
        auto viewCopyResultInput = l0op::ViewCopy(outY, output, uniqueExecutor.get());
        CHECK_RET(viewCopyResultInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        ProcessViewCopyOutputH(hyVector, hy, uniqueExecutor.get());
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief aclnnGRU的第二段接口，用于执行计算。
 */
aclnnStatus aclnnGRU(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGRU);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
