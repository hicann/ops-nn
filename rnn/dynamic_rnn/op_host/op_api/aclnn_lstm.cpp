/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_lstm.h"
#include "dynamic_rnn.h"
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

struct LstmDataParamsIn {
    const aclTensor *input;
    const aclTensorList *params;
    const aclTensorList *hx;
    const aclTensor *batchSizes;
    int64_t numLayers;
    bool has_biases;
    bool train;
    bool bidirectional;
};

struct LstmDataParamsOut {
    aclTensor *output;
    aclTensorList *iOut;
    aclTensorList *jOut;
    aclTensorList *fOut;
    aclTensorList *oOut;
    aclTensorList *hOut;
    aclTensorList *cOut;
    aclTensorList *tanhCOut;
};

struct BaseOpInputs {
    const aclTensor *input;
    const aclTensor *weight;
    const aclTensor *bias;
    const aclTensor *initHOptional;
    const aclTensor *initCOptional;
    const aclTensor *seqLengthOptional;
    const char *direction;
    bool isTraining;
};

struct BaseOpOutputs {
    aclTensor *l0_yOut;
    aclTensor *l0_iOut;
    aclTensor *l0_jOut;
    aclTensor *l0_fOut;
    aclTensor *l0_oOut;
    aclTensor *l0_hOut;
    aclTensor *l0_cOut;
    aclTensor *l0_tanhcOut;
};

struct LstmDataInfo {
    int64_t T;  // 时间步数
    int64_t B;  // 总Batch数
    int64_t I;  // input size，特征数量
    int64_t H;  // hidden size
    int64_t L;  // numLayers
    int64_t D;  // direction数量
    int64_t groupLen;  // 一层一向的param数量
    int64_t LD;  // L * D
    ge::DataType dtype;  // 输入输出数据类型
    const aclTensor *mask;  // data的mask，即seqLength
    const aclTensor *lastResult;  // 上一层的推理结果
};

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t INPUT_DIMS = 3;
static const int64_t WEIGHT_DIMS = 2;
static const int64_t BIAS_DIMS = 1;
static const int64_t INIT_DIMS = 3;
static const int64_t OUTPUT_DIMS = 3;
static const int64_t OUTPUT_DIMS_INFER = 2;
static const int64_t INDEX_0 = 0;
static const int64_t INDEX_1 = 1;
static const int64_t INDEX_2 = 2;
static const int64_t INDEX_3 = 3;
static const int64_t INDEX_4 = 4;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> INT_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};

auto nullptrInner = std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *> LstmSingleLayerDirec(
    const aclTensor * input, const aclTensorList * params, const aclTensorList * hx, aclTensor *yOutDirec, aclTensor *iOutDirec, aclTensor *jOutDirec, aclTensor *fOutDirec, aclTensor *oOutDirec, aclTensor *hOutDirec, aclTensor *cOutDirec, aclTensor *tanhCOutDirec, 
    const char *direction,  bool bidirectional, bool train, int64_t num_layers, bool has_biases, aclOpExecutor* executor)
{
    auto oneLayerParams = bidirectional == true ? 4 : 2;
    oneLayerParams = has_biases == true ? oneLayerParams * 2 : oneLayerParams;
    auto weightStart = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : oneLayerParams / 2;
    auto paramsOffsets = oneLayerParams * num_layers + weightStart;
    op::FVector<const aclTensor*> weightConcatList;
    weightConcatList.emplace_back((*params)[paramsOffsets]);
    weightConcatList.emplace_back((*params)[paramsOffsets + 1]);
    auto weightListInput = executor->AllocTensorList(weightConcatList.data(), weightConcatList.size());
    OP_CHECK_NULL(weightListInput, return nullptrInner);
    auto weightConcat = l0op::ConcatD(weightListInput, 1, executor);
    OP_CHECK_NULL(weightConcat, return nullptrInner);
    
    std::vector<int64_t> perm={1, 0};
    auto valuePerm = executor->AllocIntArray(perm.data(), 2);
    OP_CHECK_NULL(valuePerm, return nullptrInner);
    auto weightTrans = l0op::Transpose(weightConcat, valuePerm, executor);
    OP_CHECK_NULL(weightTrans, return nullptrInner);

    const aclTensor * bias = nullptr;
    if (has_biases) {
        bias = l0op::Add((*params)[paramsOffsets + 2], (*params)[paramsOffsets + 3], executor);
        OP_CHECK_NULL(bias, return nullptrInner);
    } else {
        op::Shape biasShape = {weightTrans->GetViewShape().GetDim(1)};
        auto biasTmp = executor->AllocTensor(biasShape, weightTrans->GetDataType(), op::Format::FORMAT_ND);
        OP_CHECK_NULL(biasTmp, return nullptrInner);
        bias = l0op::ZerosLike(biasTmp, executor);
        OP_CHECK_NULL(bias, return nullptrInner);
    }

    const aclTensor * initH = nullptr;
    const aclTensor * initC = nullptr;
    if (hx != nullptr && hx->Size() != 0) {
        auto batch = (*hx)[0]->GetViewShape().GetDim(1);
        auto hidden = (*hx)[0]->GetViewShape().GetDim(2);
        auto oneLayerInit = bidirectional == true ? 2 : 1;
        auto initStart = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : 1;
        
        const int64_t offsetData[] = {oneLayerInit * num_layers + initStart, 0, 0};
        aclIntArray* offsets = executor->AllocIntArray(offsetData, 3);
        OP_CHECK_NULL(offsets, return nullptrInner);
        const int64_t sizeData[] = {1, batch, hidden};
        aclIntArray* size = executor->AllocIntArray(sizeData, 3);
        OP_CHECK_NULL(size, return nullptrInner);
        initH = l0op::Slice((*hx)[0], offsets, size, executor);
        OP_CHECK_NULL(initH, return nullptrInner);
        initC = l0op::Slice((*hx)[1], offsets, size, executor);
        OP_CHECK_NULL(initC, return nullptrInner);
    }
    auto layerResult = l0op::DynamicRNN(input, weightTrans, bias, initH, initC, nullptr, direction, train, yOutDirec, iOutDirec, jOutDirec, fOutDirec, oOutDirec, hOutDirec, cOutDirec, tanhCOutDirec, executor); 

    OP_CHECK_NULL(std::get<0>(layerResult), return nullptrInner);
    OP_CHECK_NULL(std::get<1>(layerResult), return nullptrInner);
    OP_CHECK_NULL(std::get<2>(layerResult), return nullptrInner);
    OP_CHECK_NULL(std::get<3>(layerResult), return nullptrInner);
    OP_CHECK_NULL(std::get<4>(layerResult), return nullptrInner);
    OP_CHECK_NULL(std::get<5>(layerResult), return nullptrInner);
    OP_CHECK_NULL(std::get<6>(layerResult), return nullptrInner);
    OP_CHECK_NULL(std::get<7>(layerResult), return nullptrInner);
    
    return layerResult;
}

static aclnnStatus ProcessViewCopy(std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *> layerResult, const aclTensorList *iOut, const aclTensorList *jOut, const aclTensorList *fOut,
    const aclTensorList *oOut, const aclTensorList *hOut, const aclTensorList *cOut, const aclTensorList *tanhCOut, 
    int64_t numLayers,  bool bidirectional, const char *direction, aclOpExecutor* executor) {
    auto paramsNumSingleLayer = bidirectional == true ? 2 : 1;
    auto directionStart = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : 1;
    auto viewCopyResultI = l0op::ViewCopy(std::get<1>(layerResult), (*iOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultI != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultJ = l0op::ViewCopy(std::get<2>(layerResult), (*jOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultJ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultF = l0op::ViewCopy(std::get<3>(layerResult), (*fOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultF != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultO = l0op::ViewCopy(std::get<4>(layerResult), (*oOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultO != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultH = l0op::ViewCopy(std::get<5>(layerResult), (*hOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultH != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultC = l0op::ViewCopy(std::get<6>(layerResult), (*cOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultTanhc = l0op::ViewCopy(std::get<7>(layerResult), (*tanhCOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultTanhc != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessViewCopyWithInfer(std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *> layerResult, const aclTensorList *hOut, const aclTensorList *cOut, 
    int64_t numLayers,  bool bidirectional, const char *direction, aclOpExecutor* executor) {
    auto paramsNumSingleLayer = bidirectional == true ? 2 : 1;
    auto directionStart = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : 1;
    
    int64_t numStep = std::get<0>(layerResult)->GetViewShape().GetDim(0);
    int64_t batch = std::get<0>(layerResult)->GetViewShape().GetDim(1);
    int64_t hidden = std::get<0>(layerResult)->GetViewShape().GetDim(2);
    int64_t copyStep = strcmp(direction, "UNIDIRECTIONAL") == 0 ? numStep - 1: 0;

    const int64_t offsetData[] = {copyStep, 0, 0};
    aclIntArray* offsets = executor->AllocIntArray(offsetData, 3);
    const int64_t sizeData[] = {1, batch, hidden};
    aclIntArray* size = executor->AllocIntArray(sizeData, 3);
    
    auto thOutput = l0op::Slice(std::get<5>(layerResult), offsets, size, executor);
    auto thOutputSq = l0op::SqueezeNd(thOutput, (int64_t)0, executor);
    auto viewCopyResultH = l0op::ViewCopy(thOutputSq, (*hOut)[paramsNumSingleLayer * numLayers + directionStart], executor); 
    CHECK_RET(viewCopyResultH != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto tcOutput = l0op::Slice(std::get<6>(layerResult), offsets, size, executor);
    auto tcOutputSq = l0op::SqueezeNd(tcOutput, (int64_t)0, executor);
    auto viewCopyResultC = l0op::ViewCopy(tcOutputSq, (*cOut)[paramsNumSingleLayer * numLayers + directionStart], executor);
    CHECK_RET(viewCopyResultC != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static inline bool CheckNotNull(const aclTensor *input,  const aclTensorList *params,  bool train, 
    aclTensor *output, aclTensorList *iOut, aclTensorList *jOut, aclTensorList *fOut,  aclTensorList *oOut, 
    aclTensorList *hOut, aclTensorList *cOut, aclTensorList *tanhCOut) {
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(params, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(hOut, return false);
    OP_CHECK_NULL(cOut, return false);
    if (train) {
        OP_CHECK_NULL(iOut, return false);
        OP_CHECK_NULL(jOut, return false);
        OP_CHECK_NULL(fOut, return false);
        OP_CHECK_NULL(oOut, return false);
        OP_CHECK_NULL(tanhCOut, return false);
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *input,  const aclTensorList *params, const aclTensorList *hx, bool train,  
    aclTensor *output, aclTensorList *iOut, aclTensorList *jOut, aclTensorList *fOut,  aclTensorList *oOut, 
    aclTensorList *hOut, aclTensorList *cOut, aclTensorList *tanhCOut) {
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
    auto data_type = input->GetDataType();

    for (uint64_t i = 0; i < params->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*params)[i], data_type, return false);
	}

    if (hx != nullptr) {
        for (uint64_t i = 0; i < hx->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*hx)[i], data_type, return false);
        }
    }

    OP_CHECK_DTYPE_NOT_MATCH(output, data_type, return false);

    for (uint64_t i = 0; i < hOut->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*hOut)[i], data_type, return false);
	}

    for (uint64_t i = 0; i < cOut->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*cOut)[i], data_type, return false);
	}

    if (train) {
        for (uint64_t i = 0; i < iOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*iOut)[i], data_type, return false);
        }

        for (uint64_t i = 0; i < jOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*jOut)[i], data_type, return false);
        }

        for (uint64_t i = 0; i < fOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*fOut)[i], data_type, return false);
        }

        for (uint64_t i = 0; i < oOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*oOut)[i], data_type, return false);
        }
        
        for (uint64_t i = 0; i < tanhCOut->Size(); i++) {
            OP_CHECK_DTYPE_NOT_MATCH((*tanhCOut)[i], data_type, return false);
        }
    }

    return true;
}

static bool CheckDimsSize(const aclTensorList *params, const aclTensorList *hx, bool has_biases, int64_t numLayers, bool bidirectional, aclTensorList *iOut, aclTensorList *jOut, aclTensorList *fOut,  aclTensorList *oOut, 
    aclTensorList *hOut, aclTensorList *cOut, aclTensorList *tanhCOut) {
    uint64_t dScale = bidirectional == true ? 2 : 1;
    uint64_t bScale = has_biases == true ? 2 : 1;
    uint64_t output_nums = dScale * numLayers;
    uint64_t param_nums = 2 * bScale * dScale * numLayers;

    if (params->Size() != param_nums) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the params lists should be %lu, but %lu was obtained.", param_nums, params->Size());
        return false;
    }
    if (hx != nullptr && hx->Size() != 2 &&  hx->Size() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the hx lists should be 2 or 0, but %lu was obtained.", hx->Size());
        return false;
    }
    if (iOut != nullptr && iOut->Size() != output_nums && iOut->Size() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the output_i lists should be %lu, but %lu was obtained.", output_nums, iOut->Size());
        return false;
    }
    if (jOut != nullptr && jOut->Size() != output_nums && jOut->Size() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the output_j lists should be %lu, but %lu was obtained.", output_nums, jOut->Size());
        return false;
    }
    if (fOut != nullptr && fOut->Size() != output_nums && fOut->Size() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the output_f lists should be %lu, but %lu was obtained.", output_nums, fOut->Size());
        return false;
    }
    if (oOut != nullptr && oOut->Size() != output_nums && oOut->Size() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the output_o lists should be %lu, but %lu was obtained.", output_nums, oOut->Size());
        return false;
    }
    if ( hOut->Size() != output_nums) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the output_h lists should be %lu, but %lu was obtained.", output_nums, hOut->Size());
        return false;
    }
    if ( cOut->Size() != output_nums) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the output_c lists should be %lu, but %lu was obtained.", output_nums, cOut->Size());
        return false;
    }
    if (tanhCOut != nullptr && tanhCOut->Size() != output_nums && tanhCOut->Size() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of tensors required for the output_tanhc lists should be %lu, but %lu was obtained.", output_nums, tanhCOut->Size());
        return false;
    }
    return true;
}

static bool CheckDims(const aclTensor *input,  const aclTensorList *params, const aclTensorList *hx, bool has_biases, int64_t numLayers, bool train, bool bidirectional, 
    aclTensor *output, aclTensorList *iOut, aclTensorList *jOut, aclTensorList *fOut,  aclTensorList *oOut, 
    aclTensorList *hOut, aclTensorList *cOut, aclTensorList *tanhCOut) {
    OP_CHECK_WRONG_DIMENSION(input, INPUT_DIMS, return false);
    uint64_t bScale = has_biases == true ? 2 : 1;
    uint64_t dScale = bidirectional == true ? 2 : 1;
    uint64_t oneLayerParams = 2 * bScale * dScale;
    for (uint64_t i = 0; i < (uint64_t)numLayers; i++) {
        for (uint64_t j = 0; j < dScale; j++) {
            uint64_t offsets = i * oneLayerParams + j * oneLayerParams / 2;
            OP_CHECK_WRONG_DIMENSION((*params)[offsets], WEIGHT_DIMS, return false);
            OP_CHECK_WRONG_DIMENSION((*params)[offsets + 1], WEIGHT_DIMS, return false);
            if (has_biases) {
                OP_CHECK_WRONG_DIMENSION((*params)[offsets + 2], BIAS_DIMS, return false);
                OP_CHECK_WRONG_DIMENSION((*params)[offsets + 3], BIAS_DIMS, return false);
            }
        }
    }

    if (hx != nullptr) {
        for (uint64_t i = 0; i < hx->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*hx)[i], INIT_DIMS, return false);
        }
    }

    OP_CHECK_WRONG_DIMENSION(output, OUTPUT_DIMS, return false);

    if (iOut != nullptr) {
        for (uint64_t i = 0; i < iOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*iOut)[i], OUTPUT_DIMS, return false);
        }
    }

    if (jOut != nullptr) {
        for (uint64_t i = 0; i < jOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*jOut)[i], OUTPUT_DIMS, return false);
        }
    }

    if (fOut != nullptr) {
        for (uint64_t i = 0; i < fOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*fOut)[i], OUTPUT_DIMS, return false);
        }
    }

    if (oOut != nullptr) {
        for (uint64_t i = 0; i < oOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*oOut)[i], OUTPUT_DIMS, return false);
        }
    }

    if (tanhCOut != nullptr) {
        for (uint64_t i = 0; i < tanhCOut->Size(); i++) {
            OP_CHECK_WRONG_DIMENSION((*tanhCOut)[i], OUTPUT_DIMS, return false);
        }
    }

    int64_t outputDims = train ? OUTPUT_DIMS : OUTPUT_DIMS_INFER;
    for (uint64_t i = 0; i < hOut->Size(); i++) {
        OP_CHECK_WRONG_DIMENSION((*hOut)[i], outputDims, return false);
	}

    for (uint64_t i = 0; i < cOut->Size(); i++) {
        OP_CHECK_WRONG_DIMENSION((*cOut)[i], outputDims, return false);
	}

    return true;
}

static bool CheckShape(const aclTensor *input,  const aclTensorList *params, const aclTensorList *hx, bool has_biases, int64_t numLayers, bool train, bool bidirectional, bool batch_first,
    aclTensor *output, aclTensorList *iOut, aclTensorList *jOut, aclTensorList *fOut,  aclTensorList *oOut, 
    aclTensorList *hOut, aclTensorList *cOut, aclTensorList *tanhCOut) {
    auto timeStep = batch_first == true ? input->GetViewShape().GetDim(1) : input->GetViewShape().GetDim(0);
    auto batchSize = batch_first == true ? input->GetViewShape().GetDim(0) : input->GetViewShape().GetDim(1);
    auto inputSize = input->GetViewShape().GetDim(2);
    auto hiddenSize = (*params)[0]->GetViewShape().GetDim(0) / 4;
    auto curLayerInputSize = inputSize;
    uint64_t dScale = bidirectional == true ? 2 : 1;
    auto bScale = has_biases == true ? 2 : 1;
    uint64_t oneLayerParams = 2 * bScale * dScale;

    for (uint64_t i = 0; i < (uint64_t)numLayers; i++) {
        op::Shape expWiShape = {4 * hiddenSize, curLayerInputSize};
        op::Shape expWhShape = {4 * hiddenSize, hiddenSize};
        op::Shape expBShape = {4 * hiddenSize};
        for (uint64_t j = 0; j < dScale; j++) {
            uint64_t offsets = i * oneLayerParams + j * oneLayerParams / 2;
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets], expWiShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets + 1], expWhShape, return false);
            if (has_biases) {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets + 2], expBShape, return false);
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*params)[offsets + 3], expBShape, return false);
            }
        }
        curLayerInputSize = dScale * hiddenSize;
    }

    if (hx != nullptr) {
        for (uint64_t i = 0; i < hx->Size(); i++) {
            op::Shape expShape = {numLayers * dScale, batchSize, hiddenSize};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*hx)[i], expShape, return false);
        }
    }

    op::Shape expOutputShape = {timeStep, batchSize, dScale * hiddenSize};
    if (batch_first) {
        expOutputShape = {batchSize, timeStep, dScale * hiddenSize};
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, expOutputShape, return false);

    if (train) {
        for (uint64_t i = 0; i < iOut->Size(); i++) {
            op::Shape expShape = {timeStep, batchSize, hiddenSize};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*iOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*jOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*fOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*oOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*tanhCOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*hOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*cOut)[i], expShape, return false);
        }
    } else {
        for (uint64_t i = 0; i < hOut->Size(); i++) {
            op::Shape expShape = {batchSize, hiddenSize};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*hOut)[i], expShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*cOut)[i], expShape, return false);
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *input,  const aclTensorList *params, const aclTensorList *hx, bool has_biases, int64_t numLayers, bool train, bool bidirectional, 
    bool batch_first, aclTensor *output, aclTensorList *iOut, aclTensorList *jOut, aclTensorList *fOut,  aclTensorList *oOut, 
    aclTensorList *hOut, aclTensorList *cOut, aclTensorList *tanhCOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, params, train, output, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(
        CheckDtypeValid(input, params, hx, train, output, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的tensor list长度是否满足
    CHECK_RET(
        CheckDimsSize(params, hx, has_biases, numLayers, bidirectional, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的dim 是否满足
    CHECK_RET(
        CheckDims(input, params, hx, has_biases, numLayers, train, bidirectional, output, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut),
        ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入形状是否满足
    CHECK_RET(
        CheckShape(input, params, hx, has_biases, numLayers, train, bidirectional, batch_first, output, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclTensorList *ProcessInputContiguous(const aclTensorList *inputParam, aclOpExecutor* executor) {
    std::vector<const aclTensor *> inputVec;
    for (size_t i = 0; i < inputParam->Size(); ++i) {
        auto secondContiguous = l0op::Contiguous((*inputParam)[i], executor);
        inputVec.push_back(secondContiguous);
    }
    auto inputContiguous = executor->AllocTensorList(inputVec.data(), inputVec.size());
    return inputContiguous;
}

static aclnnStatus CheckTensorListNullptr(const aclTensorList *tensorList)
{
    OP_CHECK_NULL(tensorList, return ACLNN_ERR_PARAM_NULLPTR);
    for (int64_t idx = INDEX_0; idx < int64_t(tensorList->Size()); idx++) {
        OP_CHECK_NULL((*tensorList)[idx], return ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsNullptr(const LstmDataParamsIn& inputs, const LstmDataParamsOut& outputs)
{
    aclnnStatus ret;
    // 输入校验
    OP_CHECK_NULL(inputs.input, return ACLNN_ERR_PARAM_NULLPTR);
    ret = CheckTensorListNullptr(inputs.params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 如果hx非空指针，则认为意图传入有效hx。顺便校验hx长度
    if (inputs.hx) {
        OP_CHECK(
            inputs.hx->Size() == INDEX_2,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "length of hx should be 2 when it is a non-null pointer."),
            return ACLNN_ERR_PARAM_INVALID
        );
        OP_CHECK_NULL((*inputs.hx)[INDEX_0], return ACLNN_ERR_PARAM_NULLPTR);
        OP_CHECK_NULL((*inputs.hx)[INDEX_1], return ACLNN_ERR_PARAM_NULLPTR);
    }
    
    // 输出校验
    OP_CHECK_NULL(outputs.output, return ACLNN_ERR_PARAM_NULLPTR);
    ret = CheckTensorListNullptr(outputs.hOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListNullptr(outputs.cOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (inputs.train) {
        ret = CheckTensorListNullptr(outputs.iOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.jOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.fOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.oOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListNullptr(outputs.tanhCOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListLength(const aclTensorList *tensorList, int64_t length, const char* name)
{
    OP_CHECK(
        int64_t(tensorList->Size()) == length,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "length of %s should be %lld, but %lld was obtained",
            name,
            length,
            tensorList->Size()
        ),
        return ACLNN_ERR_PARAM_INVALID
    );

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListShape(const aclTensorList *tensorList, op::Shape shape, const char* name)
{
    for (int64_t idx = INDEX_0; idx < int64_t(tensorList->Size()); idx++) {
        auto tensor = (*tensorList)[idx];
        if (tensor->GetViewShape() != shape) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected tensor in %s to have same size as %s, but got %s.",
                name,
                op::ToString(shape).GetString(),
                op::ToString(tensor->GetViewShape()).GetString()
            );
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListDtype(const aclTensorList *tensorList, op::DataType dtype, const char* name)
{
    for (int64_t idx = INDEX_0; idx < int64_t(tensorList->Size()); idx++) {
        auto tensor = (*tensorList)[idx];
        if (tensor->GetDataType() != dtype) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected tensor in %s to have dtype of %s, but got %s.",
                name,
                op::ToString(dtype).GetString(),
                op::ToString(tensor->GetDataType()).GetString()
            );
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimsAndListLength(const LstmDataParamsIn& inputs, const LstmDataParamsOut& outputs, LstmDataInfo &info)
{
    aclnnStatus ret;

    OP_CHECK_WRONG_DIMENSION(inputs.input, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    ret = CheckTensorListLength(inputs.params, info.groupLen * info.LD, "params");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    for (int64_t group = INDEX_0; group < info.LD; group++) {
        // weight为2维，bias为1维
        int64_t currOffset = group * info.groupLen;
        OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_0], INDEX_2, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_1], INDEX_2, return ACLNN_ERR_PARAM_INVALID);
        if (inputs.has_biases) {
            OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_2], INDEX_1, return ACLNN_ERR_PARAM_INVALID);
            OP_CHECK_WRONG_DIMENSION((*inputs.params)[currOffset + INDEX_3], INDEX_1, return ACLNN_ERR_PARAM_INVALID);
        }
    }
    OP_CHECK_WRONG_DIMENSION(inputs.batchSizes, INDEX_1, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_WRONG_DIMENSION(outputs.output, INDEX_3, return ACLNN_ERR_PARAM_INVALID);
    ret = CheckTensorListLength(outputs.hOut, info.LD, "hOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListLength(outputs.cOut, info.LD, "cOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (inputs.train) {
        ret = CheckTensorListLength(outputs.iOut, info.LD, "iOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.jOut, info.LD, "jOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.fOut, info.LD, "fOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.oOut, info.LD, "oOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListLength(outputs.tanhCOut, info.LD, "tanhCOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShapes(const LstmDataParamsIn& inputs, const LstmDataParamsOut& outputs, LstmDataInfo &info)
{
    aclnnStatus ret;

    auto data2dShape = inputs.input->GetViewShape();
    info.T = inputs.batchSizes->Numel();
    OP_CHECK(
        info.T != INDEX_0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batchSizes should not be empty when it is a non-null pointer."),
        return ACLNN_ERR_PARAM_INVALID
    );
    info.B = data2dShape.GetDim(INDEX_0) / info.T;
    OP_CHECK(
        info.T * info.B == data2dShape.GetDim(INDEX_0),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input.shape[0] should be a multiple of the time step (i.e., the length of batchSizes)."),
        return ACLNN_ERR_PARAM_INVALID
    );
    info.I = data2dShape.GetDim(INDEX_1);
    info.H = outputs.output->GetViewShape().GetDim(INDEX_2);
    if (inputs.bidirectional) {
        OP_CHECK(
            info.H % 2 == INDEX_0,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output.shape[2] (i.e., hidden size * 2) should be even in bidirectional scenarios."),
            return ACLNN_ERR_PARAM_INVALID
        );
        info.H = info.H / INDEX_2;
    }

    // shape关系校验
    op::Shape weightIhFirstLayerShape = {INDEX_4 * info.H, info.I};
    op::Shape weightIhShape = {INDEX_4 * info.H, info.H * info.D};
    op::Shape weightHhShape = {INDEX_4 * info.H, info.H};
    op::Shape biasShape = {INDEX_4 * info.H};
    op::Shape hxShape = {info.D * info.L, info.B, info.H};
    op::Shape outShape = {info.T, info.B, info.H};
    op::Shape outInferenceShape = {info.B, info.H};
    op::Shape outputConcatShape = {info.T, info.B, info.H * info.D};

    for (int64_t group = INDEX_0; group < info.LD; group++) {
        int64_t currOffset = group * info.groupLen;
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(
            (*inputs.params)[currOffset + INDEX_0],
            ((group < info.D) ? weightIhFirstLayerShape : weightIhShape),
            return ACLNN_ERR_PARAM_INVALID
        );
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.params)[currOffset + INDEX_1], weightHhShape, return ACLNN_ERR_PARAM_INVALID);
        if (inputs.has_biases) {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.params)[currOffset + INDEX_2], biasShape, return ACLNN_ERR_PARAM_INVALID);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.params)[currOffset + INDEX_3], biasShape, return ACLNN_ERR_PARAM_INVALID);
        }
    }
    if (inputs.hx) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.hx)[INDEX_0], hxShape, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*inputs.hx)[INDEX_1], hxShape, return ACLNN_ERR_PARAM_INVALID);
    }

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputs.output, outputConcatShape, return ACLNN_ERR_PARAM_INVALID);
    if (inputs.train) {
        ret = CheckTensorListShape(outputs.iOut, outShape, "iOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.jOut, outShape, "jOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.fOut, outShape, "fOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.oOut, outShape, "oOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.hOut, outShape, "hOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.cOut, outShape, "cOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.tanhCOut, outShape, "tanhCOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        ret = CheckTensorListShape(outputs.hOut, outInferenceShape, "hOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckTensorListShape(outputs.cOut, outInferenceShape, "cOut");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypes(const LstmDataParamsIn& inputs, const LstmDataParamsOut& outputs, LstmDataInfo &info)
{
    aclnnStatus ret;

    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.input, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    info.dtype = inputs.input->GetDataType();
    ret = CheckTensorListDtype(inputs.params, info.dtype, "params");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListDtype(inputs.hx, info.dtype, "hx");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.batchSizes, INT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_MATCH(outputs.output, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    ret = CheckTensorListDtype(outputs.iOut, info.dtype, "iOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListDtype(outputs.jOut, info.dtype, "jOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListDtype(outputs.fOut, info.dtype, "fOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListDtype(outputs.oOut, info.dtype, "oOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListDtype(outputs.hOut, info.dtype, "hOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListDtype(outputs.cOut, info.dtype, "cOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorListDtype(outputs.tanhCOut, info.dtype, "tanhCOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsValid(const LstmDataParamsIn& inputs, const LstmDataParamsOut& outputs, LstmDataInfo &info)
{
    aclnnStatus ret;

    // 空指针校验
    ret = CheckParamsNullptr(inputs, outputs);
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ret, "CheckParamsNullptr failed, certain incoming nullptrs may be invalid."),
        return ret
    );

    // 属性值校验
    OP_CHECK(
        inputs.numLayers >= INDEX_1,
        OP_LOGE(ret, "numLayers should be a positive integer, but %lld was obtained.", inputs.numLayers),
        return ACLNN_ERR_PARAM_INVALID
    );

    if (inputs.input->IsEmpty()) return ACLNN_SUCCESS;  // 跳至空Tensor处理流程

    info.L = inputs.numLayers;
    info.D = (inputs.bidirectional) ? INDEX_2 : INDEX_1;
    info.groupLen = (inputs.has_biases) ? INDEX_4 : INDEX_2;
    info.LD = info.L * info.D;

    // list长度与tensor dim校验
    ret = CheckDimsAndListLength(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // shape关系校验
    ret = CheckShapes(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    
    // dtype校验
    CheckDtypes(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetMask(
    const LstmDataParamsIn& inputs,
    const LstmDataInfo& info,
    aclOpExecutor *executor)
{
    // arange [B]
    auto start = executor->AllocScalar(INDEX_0);
    auto end = executor->AllocScalar(info.B);
    auto step = executor->AllocScalar(INDEX_1);
    op::Shape expectShape = {info.B};
    auto expect = executor->AllocTensor(expectShape, inputs.batchSizes->GetDataType());
    CHECK_RET(expect != nullptr, nullptr);
    auto arangedB = l0op::Arange(start, end, step, expect, false, executor);
    CHECK_RET(arangedB != nullptr, nullptr);

    // [B] --unsqueeze--> [1, B, 1] --broad--> [T, B, H]
    // prepare data
    const int64_t shapeData[] = {info.T, info.B, info.H};
    aclIntArray* shape = executor->AllocIntArray(shapeData, INDEX_3);
    CHECK_RET(shape != nullptr, nullptr);
    const int64_t indexUnsqueezeData[] = {INDEX_0, INDEX_2};
    aclIntArray* indexUnsqueezeDim = executor->AllocIntArray(indexUnsqueezeData, INDEX_2);
    CHECK_RET(indexUnsqueezeDim != nullptr, nullptr);
    // broadcast
    const aclTensor *broadedIndex = nullptr;
    broadedIndex = l0op::UnsqueezeNd(arangedB, indexUnsqueezeDim, executor);
    CHECK_RET(broadedIndex != nullptr, nullptr);
    broadedIndex = l0op::BroadcastTo(broadedIndex, shape, executor);
    CHECK_RET(broadedIndex != nullptr, nullptr);

    // batchSizes [T] --unsqueeze--> [T, 1, 1] --broad--> [T, B, H]
    // prepare data
    const int64_t sizeUnsqueezeData[] = {INDEX_1, INDEX_2};
    aclIntArray* sizeUnsqueezeDim = executor->AllocIntArray(sizeUnsqueezeData, INDEX_2);
    CHECK_RET(sizeUnsqueezeDim != nullptr, nullptr);
    // broadcast
    const aclTensor *broadedSize = nullptr;
    broadedSize = l0op::UnsqueezeNd(inputs.batchSizes, sizeUnsqueezeDim, executor);
    CHECK_RET(broadedSize != nullptr, nullptr);
    broadedSize = l0op::BroadcastTo(broadedSize, shape, executor);
    CHECK_RET(broadedSize != nullptr, nullptr);

    // broadedSize > broadedIndex --> Bool Mask --cast--> seqLength
    auto boolMask = l0op::Greater(broadedSize, broadedIndex, executor);
    CHECK_RET(boolMask != nullptr, nullptr);
    auto mask = l0op::Cast(boolMask, info.dtype, executor);
    return mask;
}

static aclnnStatus CallBaseOp(
    const BaseOpInputs& baseIn,
    const LstmDataInfo& info,
    BaseOpOutputs& baseOut,
    aclOpExecutor *executor)
{
    op::Shape outShape = {info.T, info.B, info.H};
    baseOut.l0_yOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_iOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_jOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_fOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_oOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_hOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_cOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    baseOut.l0_tanhcOut = executor->AllocTensor(outShape, info.dtype, op::Format::FORMAT_ND);
    auto ret = l0op::DynamicRNN(
        baseIn.input, baseIn.weight, baseIn.bias, baseIn.initHOptional, baseIn.initCOptional,
        baseIn.seqLengthOptional, baseIn.direction, baseIn.isTraining,
        baseOut.l0_yOut, baseOut.l0_iOut, baseOut.l0_jOut, baseOut.l0_fOut, baseOut.l0_oOut,
        baseOut.l0_hOut, baseOut.l0_cOut, baseOut.l0_tanhcOut,
        executor
    );
    CHECK_RET(ret != nullptrInner, ACLNN_ERR_INNER_NULLPTR);  // 对齐l0
    return ACLNN_SUCCESS;
}

static aclnnStatus LstmDataProcessParams(
    const LstmDataParamsIn& inputs,
    const LstmDataInfo& info,
    int64_t layerIdx,
    int64_t directIdx,
    BaseOpInputs& baseIn,
    aclOpExecutor *executor)
{
    // h0、c0。slice
    if (inputs.hx) {
        // 假设l0op::Slice元素共享内存
        const int64_t offsetData[] = {
            layerIdx * info.D + directIdx,
            INDEX_0,
            INDEX_0
        };
        aclIntArray* offsets = executor->AllocIntArray(offsetData, INDEX_3);
        CHECK_RET(offsets != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t sizeData[] = {INDEX_1, info.B, info.H};
        aclIntArray* size = executor->AllocIntArray(sizeData, INDEX_3);
        CHECK_RET(size != nullptr, ACLNN_ERR_INNER_NULLPTR);

        baseIn.initHOptional = l0op::Slice((*inputs.hx)[0], offsets, size, executor);
        CHECK_RET(baseIn.initHOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        baseIn.initCOptional = l0op::Slice((*inputs.hx)[1], offsets, size, executor);
        CHECK_RET(baseIn.initCOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // weight。concat、T
    int64_t currOffset = (layerIdx * info.D + directIdx) * info.groupLen;  // 当前组所在偏移
    std::vector<const aclTensor *> weightVec;
    weightVec.push_back((*inputs.params)[currOffset + INDEX_0]);
    weightVec.push_back((*inputs.params)[currOffset + INDEX_1]);
    aclTensorList *weightTensorList = executor->AllocTensorList(weightVec.data(), weightVec.size());
    CHECK_RET(weightTensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    baseIn.weight = l0op::ConcatD(weightTensorList, INDEX_1, executor);
    CHECK_RET(baseIn.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    std::vector<int64_t> permData = {INDEX_1, INDEX_0};
    auto perm = executor->AllocIntArray(permData.data(), INDEX_2);
    CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    baseIn.weight = l0op::Transpose(baseIn.weight, perm, executor);
    CHECK_RET(baseIn.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // bias。add
    if (inputs.has_biases) {
        baseIn.bias = l0op::Add(
            (*inputs.params)[currOffset + INDEX_2],
            (*inputs.params)[currOffset + INDEX_3],
            executor
        );
    } else {
        op::Shape biasShape = {info.H * INDEX_4};
        auto biasTmp = executor->AllocTensor(biasShape, info.dtype, op::Format::FORMAT_ND);
        CHECK_RET(biasTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        baseIn.bias = l0op::ZerosLike(biasTmp, executor);
    }
    CHECK_RET(baseIn.bias != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus LstmDataGetBaseOpOut(
    const LstmDataParamsIn& inputs,
    const LstmDataInfo& info,
    int64_t layerIdx,
    int64_t directIdx,
    std::vector<BaseOpOutputs>& baseOutVec,
    aclOpExecutor *executor)
{
    aclnnStatus ret;

    // 构建输入
    BaseOpInputs baseIn = {
        info.lastResult, nullptr, nullptr, nullptr, nullptr, info.mask,
        (directIdx == INDEX_0) ? "UNIDIRECTIONAL" : "REDIRECTIONAL", inputs.train
    };
    ret = LstmDataProcessParams(inputs, info, layerIdx, directIdx, baseIn, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调用底层算子
    BaseOpOutputs baseOut = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    ret = CallBaseOp(baseIn, info, baseOut, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    baseOutVec.push_back(baseOut);

    return ACLNN_SUCCESS;
}

static aclnnStatus LstmDataGetParamsOut(
    const LstmDataParamsIn& inputs,
    const LstmDataInfo& info,
    const std::vector<BaseOpOutputs>& baseOutVec,
    LstmDataParamsOut& outputs,
    aclOpExecutor *executor)
{
    const aclTensor *res = nullptr;

    // 处理outputs.output
    res = l0op::ViewCopy(info.lastResult, outputs.output, executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理其他tensorList
    /* 由于输出也是先层后方向排布，因此可以直接按baseOutVec顺序拷贝
     */
    if (inputs.train) {
        for (int64_t idx = INDEX_0; idx < info.LD; idx++) {
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_iOut, (*outputs.iOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_jOut, (*outputs.jOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_fOut, (*outputs.fOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_oOut, (*outputs.oOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_hOut, (*outputs.hOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_cOut, (*outputs.cOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(baseOutVec.at(idx).l0_tanhcOut, (*outputs.tanhCOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        // 准备取最后1个step，正序为T-1，反序为0
        const int64_t offsetUniDirectData[] = {info.T - INDEX_1, INDEX_0, INDEX_0};
        aclIntArray* offsetUniDirect = executor->AllocIntArray(offsetUniDirectData, INDEX_3);
        CHECK_RET(offsetUniDirect != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t offsetReDirectData[] = {INDEX_0, INDEX_0, INDEX_0};
        aclIntArray* offsetReDirect = executor->AllocIntArray(offsetReDirectData, INDEX_3);
        CHECK_RET(offsetReDirect != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t sizeData[] = {INDEX_1, info.B, info.H};
        aclIntArray* size = executor->AllocIntArray(sizeData, INDEX_3);
        CHECK_RET(size != nullptr, ACLNN_ERR_INNER_NULLPTR);

        for (int64_t idx = INDEX_0; idx < info.LD; idx++) {
            res = l0op::Slice(
                baseOutVec.at(idx).l0_hOut,
                ((info.D == INDEX_2 and idx % INDEX_2 != INDEX_0) ? offsetReDirect : offsetUniDirect),
                size,
                executor
            );
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::SqueezeNd(res, INDEX_0, executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(res, (*outputs.hOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);

            res = l0op::Slice(
                baseOutVec.at(idx).l0_cOut,
                ((info.D == INDEX_2 and idx % INDEX_2 != INDEX_0) ? offsetReDirect : offsetUniDirect),
                size,
                executor
            );
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::SqueezeNd(res, INDEX_0, executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
            res = l0op::ViewCopy(res, (*outputs.cOut)[idx], executor);
            CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    
    return ACLNN_SUCCESS;
}

static aclnnStatus LstmDataRun(
    const LstmDataParamsIn& inputs,
    LstmDataInfo& info,
    LstmDataParamsOut& outputs,
    aclOpExecutor *executor)
{
    aclnnStatus ret;

    // 依层数和方向数构建底层算子输入并获取其输出
    std::vector<BaseOpOutputs> baseOutVec;
    info.lastResult = inputs.input;
    for (int64_t layerIdx = INDEX_0; layerIdx < info.L; layerIdx++) {
        for (int64_t directIdx = INDEX_0; directIdx < info.D; directIdx++) {
            ret = LstmDataGetBaseOpOut(inputs, info, layerIdx, directIdx, baseOutVec, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
        /* baseIn.input首层为外部输入(T, B, I)，后续为前一层所有方向的的输出(T, B, D*H)
         * 同时，aclnn输出也需要(T, B, D*H)
         * 因此在本层双向结果产生后，立即进行拼接备用
         */
        int64_t offset = info.D * layerIdx;
        if (info.D == INDEX_1) {
            info.lastResult = baseOutVec.at(offset).l0_yOut;
        } else {
            std::vector<aclTensor *> resVec;
            resVec.push_back(baseOutVec.at(offset).l0_yOut);
            resVec.push_back(baseOutVec.at(offset + INDEX_1).l0_yOut);
            aclTensorList *resTensorList = executor->AllocTensorList(resVec.data(), resVec.size());
            info.lastResult = l0op::ConcatD(resTensorList, INDEX_2, executor);
            CHECK_RET(info.lastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 构建目标输出并拷贝到目标地址
    ret = LstmDataGetParamsOut(inputs, info, baseOutVec, outputs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

static aclnnStatus LstmDataGetWorkspaceSize(
    LstmDataParamsIn& inputs,
    LstmDataParamsOut& outputs,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus ret;
    LstmDataInfo info;

    // 参数校验，顺便填充info
    ret = CheckParamsValid(inputs, outputs, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 预处理
    // 空tensor处理
    if (inputs.input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 转连续
    auto inputCtg = l0op::Contiguous(inputs.input, uniqueExecutor.get());
    auto paramsCtg = ProcessInputContiguous(inputs.params, uniqueExecutor.get());
    auto hxCtg = ProcessInputContiguous(inputs.hx, uniqueExecutor.get());
    auto batchSizesCtg = l0op::Contiguous(inputs.batchSizes, uniqueExecutor.get());
    inputs.input = inputCtg;
    inputs.params = paramsCtg;
    inputs.hx = hxCtg;
    inputs.batchSizes = batchSizesCtg;

    // 共性处理
    op::Shape data3dShape = {info.T, info.B, info.I};
    auto data = l0op::Reshape(inputs.input, data3dShape, uniqueExecutor.get());
    OP_CHECK(data != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "reshape input to 3D failed."), return ACLNN_ERR_INNER_NULLPTR);
    inputs.input = data;
    info.mask = GetMask(inputs, info, uniqueExecutor.get());
    OP_CHECK(info.mask != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GetMask failed."), return ACLNN_ERR_INNER_NULLPTR);

    // 运行并完成输出
    ret = LstmDataRun(inputs, info, outputs, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLSTMGetWorkspaceSize(
    const aclTensor *input,
    const aclTensorList *params,
    const aclTensorList *hx,
    const aclTensor *batchSizes,
    bool has_biases,
    int64_t numLayers,
    double droupout,
    bool train,
    bool bidirectional,
    bool batch_first,
    aclTensor *output,
    aclTensorList *iOut,
    aclTensorList *jOut,
    aclTensorList *fOut,
    aclTensorList *oOut,
    aclTensorList *hOut,
    aclTensorList *cOut,
    aclTensorList *tanhCOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor){
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnLSTM, DFX_IN(input, params, hx, batchSizes, has_biases, numLayers, droupout, train, bidirectional, batch_first), 
    DFX_OUT(output, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut));

    // 判断是否进入data模式
    if (batchSizes) {
        LstmDataParamsIn inputs = {input, params, hx, batchSizes, numLayers, has_biases, train, bidirectional};
        LstmDataParamsOut outputs = {output, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut};
        return LstmDataGetWorkspaceSize(inputs, outputs, workspaceSize, executor);
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(input, params, hx, has_biases, numLayers, train, bidirectional, batch_first, output, 
            iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 先将tensor转为连续性, 
    //input
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());

    //params
    auto paramsContiguous = ProcessInputContiguous(params, uniqueExecutor.get());
    CHECK_RET(paramsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //init_hc
    auto hxContiguous = hx;
    if (hx != nullptr) {
        hxContiguous = ProcessInputContiguous(hxContiguous, uniqueExecutor.get());
        CHECK_RET(hxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 输入batchFirst转换
    auto curInput = inputContiguous;
    if (batch_first == true) {
        std::vector<int64_t> perm={1, 0, 2};
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
        curInput = l0op::Transpose(inputContiguous, valuePerm, uniqueExecutor.get());
    } 
    
    int64_t hiddenSize = output->GetViewShape().GetDim(2);
    hiddenSize = bidirectional == true ? hiddenSize / 2: hiddenSize;
    op::Shape outShape = {curInput->GetViewShape().GetDim(0), curInput->GetViewShape().GetDim(1), hiddenSize};
    
    //isTraing = True
    if (train == true) {
        for (uint64_t i = 0U; i < uint64_t(numLayers); ++i) {
            auto yOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(yOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto iOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(iOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto jOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(jOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto fOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(fOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto oOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(oOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto hOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(hOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto cOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(cOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto tanhCOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(tanhCOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto layerResultForward = LstmSingleLayerDirec(curInput, paramsContiguous, hxContiguous, yOutForward, iOutForward, jOutForward, fOutForward, oOutForward, hOutForward, cOutForward, tanhCOutForward, 
                                    "UNIDIRECTIONAL", bidirectional, train, i, has_biases, uniqueExecutor.get());
    
            ProcessViewCopy(layerResultForward, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut, i, bidirectional, "UNIDIRECTIONAL", uniqueExecutor.get());
            
            if (bidirectional == true) {
                auto yOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(yOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto iOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(iOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto jOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(jOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto fOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(fOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto oOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(oOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto hOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(hOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto cOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(cOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto tanhCOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(tanhCOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);

                auto layerResultBackward = LstmSingleLayerDirec(curInput, paramsContiguous, hxContiguous, yOutBackward, iOutBackward, jOutBackward, fOutBackward, oOutBackward, hOutBackward, cOutBackward, tanhCOutBackward, 
                                            "REDIRECTIONAL", bidirectional, train, i, has_biases, uniqueExecutor.get());
                // ConcatInput
                op::FVector<const aclTensor*> inputConcat;
                inputConcat.emplace_back(std::get<0>(layerResultForward));
                inputConcat.emplace_back(std::get<0>(layerResultBackward));
                auto tensorListInput = uniqueExecutor.get()->AllocTensorList(inputConcat.data(), inputConcat.size());
                curInput = l0op::ConcatD(tensorListInput, 2, uniqueExecutor.get());
                ProcessViewCopy(layerResultBackward, iOut, jOut, fOut, oOut, hOut, cOut, tanhCOut, i, bidirectional, "REDIRECTIONAL", uniqueExecutor.get());
            } else {
                curInput = std::get<0>(layerResultForward);
            }
        }

        auto outputY = curInput;
        if (batch_first) {
             std::vector<int64_t> perm={1, 0, 2};
            auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
            outputY = l0op::Transpose(curInput, valuePerm, uniqueExecutor.get());
        }
        auto viewCopyResultInput = l0op::ViewCopy(outputY, output, uniqueExecutor.get()); 
        CHECK_RET(viewCopyResultInput != nullptr, ACLNN_ERR_INNER_NULLPTR);  
    } else {
        auto iOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(iOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto jOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(jOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto fOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(fOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto oOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(oOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto tanhCOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(tanhCOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        
        aclTensor *iOutBackward = nullptr;
        aclTensor *jOutBackward = nullptr;
        aclTensor *fOutBackward = nullptr;
        aclTensor *oOutBackward = nullptr;
        aclTensor *tanhCOutBackward = nullptr;

        if (bidirectional == true) {
            iOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(iOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            jOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(jOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            fOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(fOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            oOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(oOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            tanhCOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(tanhCOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        for (uint64_t i = 0U; i < uint64_t(numLayers); ++i) {
            auto yOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(yOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto hOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(hOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto cOutForward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
            CHECK_RET(cOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto layerResultForward = LstmSingleLayerDirec(curInput, paramsContiguous, hxContiguous, yOutForward, iOutForward, jOutForward, fOutForward, oOutForward, hOutForward, cOutForward, tanhCOutForward, 
                                        "UNIDIRECTIONAL", bidirectional, train, i, has_biases, uniqueExecutor.get());
    
            ProcessViewCopyWithInfer(layerResultForward, hOut, cOut, i, bidirectional, "UNIDIRECTIONAL", uniqueExecutor.get());
            
            if (bidirectional == true) {
                auto hOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(hOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto cOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(cOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto yOutBackward = uniqueExecutor.get()->AllocTensor(outShape, input->GetDataType(), op::Format::FORMAT_ND);
                CHECK_RET(yOutBackward != nullptr, ACLNN_ERR_INNER_NULLPTR);

                auto layerResultBackward = LstmSingleLayerDirec(curInput, paramsContiguous, hxContiguous, yOutBackward, iOutBackward, jOutBackward, fOutBackward, oOutBackward, hOutBackward, cOutBackward, tanhCOutBackward, 
                                            "REDIRECTIONAL", bidirectional, train, i, has_biases, uniqueExecutor.get());
                // ConcatInput
                op::FVector<const aclTensor*> inputConcat;
                inputConcat.emplace_back(std::get<0>(layerResultForward));
                inputConcat.emplace_back(std::get<0>(layerResultBackward));
                auto tensorListInput = uniqueExecutor.get()->AllocTensorList(inputConcat.data(), inputConcat.size());
                curInput = l0op::ConcatD(tensorListInput, 2, uniqueExecutor.get());
                ProcessViewCopyWithInfer(layerResultBackward, hOut, cOut, i, bidirectional, "REDIRECTIONAL", uniqueExecutor.get());
            } else {
                curInput = std::get<0>(layerResultForward);
            }
        }
        auto outputY = curInput;
        if (batch_first) {
            std::vector<int64_t> perm={1, 0, 2};
            auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
            outputY = l0op::Transpose(curInput, valuePerm, uniqueExecutor.get());
        }
        auto viewCopyResultInput = l0op::ViewCopy(outputY, output, uniqueExecutor.get()); 
        CHECK_RET(viewCopyResultInput != nullptr, ACLNN_ERR_INNER_NULLPTR);  
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLSTM(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnLSTM);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif