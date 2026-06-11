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
 * \file aclnn_quant_convolution.cpp
 * \brief
 */

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_common/log/log.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"

#include "convolution.h"
#include "quant_convolution.h"
#include "level0/padv3.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transpose.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "aclnn_quant_convolution.h"
#include "convolution_util.h"


using namespace op;
using namespace ge;
using namespace l0op;
using namespace ConvolutionUtil;

using L0FUNCTION = void (*) ();
using QUANT_CONV_FUNCTION = const aclTensor* (*) (const aclTensor* input, const aclTensor* weight,
    const aclTensor* bias, const aclTensor* scale, const aclTensor *offset, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int groups, aclOpExecutor* executor);
using QUANT_CONV_V2_FUNCTION = const aclTensor* (*) (const aclTensor* input, const aclTensor* weight,
    const aclTensor* scale, const aclTensor* bias, DataType outputDtype, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int groups, int32_t offsetx, const char* roundMode,
    aclOpExecutor* executor);
using CONV3D_V2_FUNCTION = const aclTensor* (*) (const aclTensor* input, const aclTensor* weight,
    const aclTensor* bias, const aclTensor* scale, DataType outputDtype, Format outputFormat, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int groups, bool useHf32, aclOpExecutor* executor);

#define CHECK_NULLPTR(param, ret_value)                                                                              \
    do {                                                                                                             \
        if ((param) == (nullptr)) {                                                                                  \
            OP_LOGE(ret_value, "expected %s != nullptr, get nullptr", #param);                                       \
            return ret_value;                                                                                        \
        }                                                                                                            \
    } while (0)

#define REG_L0_FUNCTION_BY_OPTYPE(map, function, functionType) \
    ((map).emplace(functionType, (L0FUNCTION(&(function)))))

namespace op {

const size_t QUANT_CONV_2D_DIM_SIZE = 4;
const size_t QUANT_CONV_3D_DIM_SIZE = 5;
const size_t QUANT_CONV_2D_PAD_TOP_INDEX = 0;
const size_t QUANT_CONV_2D_PAD_BOTTOM_INDEX = 1;
const size_t QUANT_CONV_2D_PAD_LEFT_INDEX = 2;
const size_t QUANT_CONV_2D_PAD_RIGHT_INDEX = 3;
const size_t QUANT_CONV_3D_PAD_HEAD_INDEX = 0;
const size_t QUANT_CONV_3D_PAD_TAIL_INDEX = 1;
const size_t QUANT_CONV_3D_PAD_TOP_INDEX = 2;
const size_t QUANT_CONV_3D_PAD_BOTTOM_INDEX = 3;
const size_t QUANT_CONV_3D_PAD_LEFT_INDEX = 4;
const size_t QUANT_CONV_3D_PAD_RIGHT_INDEX = 5;
const size_t QUANT_CONV_2D_PAD_DIM = 2;
const size_t QUANT_CONV_3D_PAD_DIM = 3;
const size_t QUANT_CONV_2D_STRIDE_DIM = 2;
const size_t QUANT_CONV_3D_STRIDE_DIM = 3;
const size_t QUANT_CONV_2D_DILATION_DIM = 2;
const size_t QUANT_CONV_3D_DILATION_DIM = 3;
const size_t CONV_3D_INPUT_DIM = 5;
const size_t INPUT_C_INDEX = 1;
const size_t CONST_VALUE_2 = 2;
const int32_t OFFSET_X_MAX_VALUE = 127;
const int32_t OFFSET_X_MIN_VALUE = -128;
const size_t MAX_STR_LEN = 1024;
const size_t FRACTAL_Z_3D_DIM = 4;

const std::vector<std::vector<DataType>> SUPPORTED_DTYPES_GROUPS_DEFAULT = {
    // input, weight, output, bias
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_BF16, DataType::DT_FLOAT},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_BF16, DataType::DT_BF16},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_FLOAT},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_FLOAT16}
};

const std::vector<std::vector<DataType>> SUPPORTED_DTYPES_GROUPS_DAV = {
    // input, weight, output, bias
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_INT32},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_BF16, DataType::DT_FLOAT},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_BF16, DataType::DT_BF16},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_FLOAT},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_FLOAT16},
    {DataType::DT_HIFLOAT8, DataType::DT_HIFLOAT8, DataType::DT_FLOAT, DataType::DT_FLOAT},
    {DataType::DT_HIFLOAT8, DataType::DT_HIFLOAT8, DataType::DT_FLOAT16, DataType::DT_FLOAT},
    {DataType::DT_HIFLOAT8, DataType::DT_HIFLOAT8, DataType::DT_BF16, DataType::DT_FLOAT},
    {DataType::DT_HIFLOAT8, DataType::DT_HIFLOAT8, DataType::DT_HIFLOAT8, DataType::DT_FLOAT},
    {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT, DataType::DT_FLOAT},
    {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT16, DataType::DT_FLOAT},
    {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E4M3FN, DataType::DT_BF16, DataType::DT_FLOAT},
    {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT}
};

static inline ge::AscendString ToString(const std::int64_t value)
{
    return ge::AscendString(std::to_string(value).c_str());
}

}  // namespace op

namespace {

static bool IsSocSupportND()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
}

}

static std::vector<std::vector<DataType>> GetSupportedDtypesBySoc()
{
    return IsSocSupportND() ? SUPPORTED_DTYPES_GROUPS_DAV : SUPPORTED_DTYPES_GROUPS_DEFAULT;
}

static FVector<int64_t> ConstructV2Padding(const FVector<int64_t> &oldPad, const FVector<int64_t> &inputShape)
{
    // quant conv2d 支持 pad dim = 2 or 4; quant conv3d 支持 pad dim = 3 or 6;
    // 用于outputshape计算时合并pad，实际传入L0时还是按4 or 6维传入
    FVector<int64_t> newPad;
    if (inputShape.size() == QUANT_CONV_2D_DIM_SIZE) {
        if (oldPad.size() == QUANT_CONV_2D_PAD_DIM) {
            newPad = {(oldPad[0] + oldPad[0]), (oldPad[1] + oldPad[1])};
        } else if (oldPad.size() == QUANT_CONV_2D_PAD_DIM * CONST_VALUE_2) {
            newPad = {(oldPad[QUANT_CONV_2D_PAD_TOP_INDEX] + oldPad[QUANT_CONV_2D_PAD_BOTTOM_INDEX]),
                      (oldPad[QUANT_CONV_2D_PAD_LEFT_INDEX] + oldPad[QUANT_CONV_2D_PAD_RIGHT_INDEX])};
        } else {
            newPad = {0, 0};
        }
    } else if (inputShape.size() == QUANT_CONV_3D_DIM_SIZE) {
        if (oldPad.size() == QUANT_CONV_3D_PAD_DIM) {
            newPad = {(oldPad[0] + oldPad[0]), (oldPad[1] + oldPad[1]), (oldPad[2] + oldPad[2])};
        } else if (oldPad.size() == QUANT_CONV_3D_PAD_DIM * CONST_VALUE_2) {
            newPad = {(oldPad[QUANT_CONV_3D_PAD_HEAD_INDEX] + oldPad[QUANT_CONV_3D_PAD_TAIL_INDEX]),
                      (oldPad[QUANT_CONV_3D_PAD_TOP_INDEX] + oldPad[QUANT_CONV_3D_PAD_BOTTOM_INDEX]),
                      (oldPad[QUANT_CONV_3D_PAD_LEFT_INDEX] + oldPad[QUANT_CONV_3D_PAD_RIGHT_INDEX])};
        } else {
            newPad = {0, 0, 0};
        }
    } else {
        return oldPad;
    }
    return newPad;
}

static FVector<int64_t> ConstructPadding(const FVector<int64_t> &oldPad, const FVector<int64_t> &inputShape)
{
    if (IsSocSupportND()) {
        return ConstructV2Padding(oldPad, inputShape);
    }
    // quant conv3d 支持 pad dim = 3;
    // 用于outputshape计算时合并pad
    FVector<int64_t> newPad;
    newPad = {(oldPad[0] + oldPad[0]), (oldPad[1] + oldPad[1]), (oldPad[2] + oldPad[2])};

    return newPad;
}

struct TensorMeta {
public:
    TensorMeta() = default;
    explicit TensorMeta(const aclTensor* tensor) { this->SetFromTensor(tensor); }

    void SetFromTensor(const aclTensor* tensor)
    {
        if (tensor == nullptr) {
            return;
        }
        format = tensor->GetViewFormat();
        dataType = tensor->GetDataType();
        string formatStr = op::ToString(format).GetString();
        tensorShape = tensor->GetViewShape();
        shape = op::ToFVector(tensorShape);
        storageFormat = tensor->GetStorageFormat();
        storageTensorShape = tensor->GetStorageShape();
        storageShape = op::ToFVector(storageTensorShape);

        auto len = shape.size();
        auto npos = formatStr.npos;
        auto index = formatStr.find('N');
        nIdx = index;
        n = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('C');
        cIdx = index;
        c = (index == npos || index >= len) ? 1 : shape[index];

        // 不含D轴时默认为1
        index = formatStr.find('D');
        dIdx = index;
        d = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('H');
        hIdx = index;
        h = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('W');
        wIdx = index;
        w = (index == npos || index >= len) ? 1 : shape[index];
    }

    int64_t N() const { return n; }
    int64_t C() const { return c; }
    int64_t D() const { return d; }
    int64_t H() const { return h; }
    int64_t W() const { return w; }
    size_t NIdx() const { return nIdx; }
    size_t CIdx() const { return cIdx; }
    size_t DIdx() const { return dIdx; }
    size_t HIdx() const { return hIdx; }
    size_t WIdx() const { return wIdx; }

public:
    op::Format format = Format::FORMAT_ND;
    op::DataType dataType = DataType::DT_UNDEFINED;
    FVector<int64_t> shape;
    op::Shape tensorShape;
    op::Format storageFormat = Format::FORMAT_ND;
    FVector<int64_t> storageShape;
    op::Shape storageTensorShape;

private:
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
    size_t nIdx = 0;
    size_t cIdx = 0;
    size_t dIdx = 0;
    size_t hIdx = 0;
    size_t wIdx = 0;
};

struct QuantConvParams {
    const aclTensor* input;
    const aclTensor* weight;
    const aclTensor* bias;
    const aclTensor* scale;
    const aclTensor* offset;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    const bool transposed;
    const aclIntArray* outputPadding;
    const int64_t groups;
    int32_t offsetx;
    const char* roundMode;
    aclTensor* output;
    const bool isWeightNz;
};

struct QuantConvMeta {
public:
    TensorMeta input;
    TensorMeta weight;
    TensorMeta scale;
    TensorMeta bias;
    TensorMeta output;
    FVector<int64_t> stride;
    FVector<int64_t> dilation;
    FVector<int64_t> padding;

    void FromParams(QuantConvParams &params)
    {
        input.SetFromTensor(params.input);
        weight.SetFromTensor(params.weight);
        output.SetFromTensor(params.output);
        stride = ToVector(params.stride);
        dilation = ToVector(params.dilation);
        padding = ToVector(params.padding);

        if (params.scale) {
            scale.format = params.scale->GetViewFormat();
            scale.dataType = params.scale->GetDataType();
            scale.tensorShape = params.scale->GetViewShape();
            scale.shape = op::ToFVector(scale.tensorShape);
        }
        if (params.bias) {
            bias.format = params.bias->GetViewFormat();
            bias.dataType = params.bias->GetDataType();
            bias.tensorShape = params.bias->GetViewShape();
            bias.shape = op::ToFVector(bias.tensorShape);
        }
    }

private:
    FVector<int64_t> ToVector(const aclIntArray* array) const
    {
        FVector<int64_t> v;
        if (array != nullptr) {
            for (uint64_t i = 0; i < array->Size(); ++i) {
                v.push_back((*array)[i]);
            }
        }
        return v;
    }
};

struct QuantConvEngine {
public:
    QuantConvParams params;
    QuantConvMeta meta;
    std::string entityName;
    explicit QuantConvEngine(QuantConvParams &quantConvParams) : params(quantConvParams) { meta.FromParams(params); }
    FVector<int64_t> CalcOutputShape() { return InferOutShape(); }

private:
    FVector<int64_t> InferOutShape()
    {
        FVector<int64_t> output = {meta.input.N(), meta.weight.N()};
        FVector<int64_t> inputShape = meta.input.shape;
        FVector<int64_t> weightShape = meta.weight.shape;

        auto newPad = ConstructPadding(meta.padding, inputShape);
        int64_t inferredShapeSize = inputShape.size() - 2;
        for (int64_t i = 0; i < inferredShapeSize; ++i) {
            int64_t xOut = (inputShape[i + INPUT_C_INDEX + 1] + newPad[i] - meta.dilation[i] *
                            (weightShape[i + INPUT_C_INDEX + 1] - 1) - 1) / meta.stride[i] + 1;
            output.push_back(xOut);
        }
        return output;
    }
};

static const aclTensor* L0FuncWarper(const std::map<std::string, L0FUNCTION>& l0Functions, std::string functionType,
                                     const aclTensor* input, const aclTensor* weight, const aclTensor* bias,
                                     const aclTensor* scale, const aclTensor *offset,
                                     const aclIntArray* stride, const aclIntArray* padding,
                                     const aclIntArray* dilation, const int64_t groups,
                                     int32_t offsetx, const char* roundMode, op::DataType outputDtype,
                                     op::Format outputFormat, aclOpExecutor *executor)
{
    const aclTensor* result = nullptr;
    if (l0Functions.find(functionType) == l0Functions.end()) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "no matched L0 function");
        return result;
    }

    L0FUNCTION fn = l0Functions.at(functionType);
    if (IsSocSupportND()) {
        if (functionType == "Conv3dV2L0") {
            result = (reinterpret_cast<CONV3D_V2_FUNCTION>(fn))(input, weight, bias, scale, outputDtype, outputFormat,
                stride, padding, dilation, groups, false, executor);
        } else {
            result = (reinterpret_cast<QUANT_CONV_V2_FUNCTION>(fn))(input, weight, scale, bias, outputDtype, stride,
                padding, dilation, groups, offsetx, roundMode, executor);
        }
    } else {
        result = (reinterpret_cast<QUANT_CONV_FUNCTION>(fn))(input, weight, bias, scale, offset, stride, padding,
            dilation, groups, executor);
    }

    return result;
}

#define FUNCTION_CALL_BY_OPTYPE(l0Functions, functionType, input, weight, bias, scale, offset, stride,\
                                padding, dilation, groups, offsetx, roundMode, \
                                outputDtype, outputFormat, executor)                                              \
    L0FuncWarper(l0Functions, functionType, input, weight, bias, scale, offset, stride, padding,    \
                 dilation, groups, offsetx, roundMode, outputDtype, outputFormat, executor)

class QuantConvolutionChecker {
public:
    QuantConvolutionChecker() = default;
    virtual ~QuantConvolutionChecker() = default;
    virtual aclnnStatus Check(QuantConvEngine &engine) = 0;
};

class DimsChecker : public QuantConvolutionChecker {
public:
    DimsChecker() = default;
    ~DimsChecker() override = default;
    aclnnStatus Check(QuantConvEngine &engine) override
    {
        if (CheckInOutDim(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckAttrDim(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        return ACLNN_SUCCESS;
    }
private:
    aclnnStatus CheckInOutDimND(QuantConvEngine &engine) const
    {
        size_t inputDim = engine.meta.input.shape.size();
        size_t weightDim = engine.meta.weight.shape.size();
        size_t outputDim = engine.meta.output.shape.size();
        if (inputDim != QUANT_CONV_2D_DIM_SIZE && inputDim != QUANT_CONV_3D_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "x", std::to_string(inputDim),
                std::to_string(QUANT_CONV_2D_DIM_SIZE) + " or " + std::to_string(QUANT_CONV_3D_DIM_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (weightDim != QUANT_CONV_2D_DIM_SIZE && weightDim != QUANT_CONV_3D_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "filter", std::to_string(weightDim),
                std::to_string(QUANT_CONV_2D_DIM_SIZE) + " or " + std::to_string(QUANT_CONV_3D_DIM_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (outputDim != QUANT_CONV_2D_DIM_SIZE && outputDim != QUANT_CONV_3D_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "y", std::to_string(outputDim),
                std::to_string(QUANT_CONV_2D_DIM_SIZE) + " or " + std::to_string(QUANT_CONV_3D_DIM_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (inputDim != weightDim || inputDim != outputDim || weightDim != outputDim) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(engine.entityName, "x, filter, y", std::to_string(inputDim) +
                ", " + std::to_string(weightDim) + ", " + std::to_string(outputDim),
                "the dims of these parameters must be the same");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInOutDim(QuantConvEngine &engine) const
    {
        size_t scaleDim = engine.meta.scale.shape.size();
        if (scaleDim != static_cast<size_t>(1)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "scale", std::to_string(scaleDim),
                std::to_string(static_cast<size_t>(1)));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (engine.params.bias) {
            size_t biasDim = engine.meta.bias.shape.size();
            if (biasDim != static_cast<size_t>(1)) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "bias", std::to_string(biasDim),
                std::to_string(static_cast<size_t>(1)));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (IsSocSupportND()) {
            return CheckInOutDimND(engine);
        } else {
            size_t inputDim = engine.meta.input.shape.size();
            size_t weightDim = engine.meta.weight.shape.size();
            size_t outputDim = engine.meta.output.shape.size();
            if (inputDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "x", std::to_string(inputDim),
                    std::to_string(QUANT_CONV_3D_DIM_SIZE));
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (weightDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "filter", std::to_string(weightDim),
                    std::to_string(QUANT_CONV_3D_DIM_SIZE));
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (outputDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(engine.entityName, "y", std::to_string(outputDim),
                    std::to_string(QUANT_CONV_3D_DIM_SIZE));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        return ACLNN_SUCCESS;
    }

    aclnnStatus Check3DAttrDim(QuantConvEngine &engine) const
    {
        size_t strideDim = engine.meta.stride.size();
        size_t dilationDim = engine.meta.dilation.size();
        size_t padDim = engine.meta.padding.size();
        if (strideDim != QUANT_CONV_3D_STRIDE_DIM) {
            OP_LOGE_FOR_INVALID_LISTSIZE(engine.entityName, "strides", std::to_string(strideDim),
                std::to_string(QUANT_CONV_3D_STRIDE_DIM));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (dilationDim != QUANT_CONV_3D_DILATION_DIM) {
            OP_LOGE_FOR_INVALID_LISTSIZE(engine.entityName, "dilations", std::to_string(dilationDim),
                std::to_string(QUANT_CONV_3D_DILATION_DIM));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (IsSocSupportND()) {
            if (padDim != QUANT_CONV_3D_PAD_DIM && padDim != QUANT_CONV_3D_PAD_DIM * CONST_VALUE_2) {
                OP_LOGE_FOR_INVALID_LISTSIZE(engine.entityName, "pads", std::to_string(padDim),
                    std::to_string(QUANT_CONV_3D_PAD_DIM) + " or " +
                    std::to_string(QUANT_CONV_3D_PAD_DIM * CONST_VALUE_2));
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (padDim != QUANT_CONV_3D_PAD_DIM) {
                OP_LOGE_FOR_INVALID_LISTSIZE(engine.entityName, "pads", std::to_string(padDim),
                    std::to_string(QUANT_CONV_3D_PAD_DIM));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        return ACLNN_SUCCESS;
    }

    aclnnStatus Check2DAttrDim(QuantConvEngine &engine) const
    {
        size_t strideDim = engine.meta.stride.size();
        size_t dilationDim = engine.meta.dilation.size();
        size_t padDim = engine.meta.padding.size();
        if (strideDim != QUANT_CONV_2D_STRIDE_DIM) {
            OP_LOGE_FOR_INVALID_LISTSIZE(engine.entityName, "strides", std::to_string(strideDim),
                std::to_string(QUANT_CONV_2D_STRIDE_DIM));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (dilationDim != QUANT_CONV_2D_DILATION_DIM) {
            OP_LOGE_FOR_INVALID_LISTSIZE(engine.entityName, "dilations", std::to_string(dilationDim),
                std::to_string(QUANT_CONV_2D_DILATION_DIM));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (padDim != QUANT_CONV_2D_PAD_DIM && padDim != QUANT_CONV_2D_PAD_DIM * CONST_VALUE_2) {
            OP_LOGE_FOR_INVALID_LISTSIZE(engine.entityName, "pads", std::to_string(padDim),
                std::to_string(QUANT_CONV_2D_PAD_DIM) + " or " +
                std::to_string(QUANT_CONV_2D_PAD_DIM * CONST_VALUE_2));
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckAttrDim(QuantConvEngine &engine) const
    {
        size_t inputDim = engine.meta.input.shape.size();
        switch (inputDim) {
            case QUANT_CONV_2D_DIM_SIZE:
                if (IsSocSupportND()) {
                    return Check2DAttrDim(engine);
                }
                break;
            case QUANT_CONV_3D_DIM_SIZE:
                return Check3DAttrDim(engine);
            default:
                break;
        }
        return ACLNN_SUCCESS;
    }
};

class NullPtrChecker : public QuantConvolutionChecker {
public:
    NullPtrChecker() = default;
    ~NullPtrChecker() override = default;
    aclnnStatus Check(QuantConvEngine &engine) override
    {
        if (!IsSocSupportND()) {
            CHECK_PARAM_NULLPTR(engine.entityName, engine.params.bias, "bias");
        }
        CHECK_PARAM_NULLPTR(engine.entityName, engine.params.input, "x");
        CHECK_PARAM_NULLPTR(engine.entityName, engine.params.weight, "filter");
        CHECK_PARAM_NULLPTR(engine.entityName, engine.params.scale, "scale");
        CHECK_PARAM_NULLPTR(engine.entityName, engine.params.stride, "strides");
        CHECK_PARAM_NULLPTR(engine.entityName, engine.params.padding, "pads");
        CHECK_PARAM_NULLPTR(engine.entityName, engine.params.output, "y");
        return ACLNN_SUCCESS;
    }
};

class FormatsChecker : public QuantConvolutionChecker {
public:
    FormatsChecker() = default;
    ~FormatsChecker() override = default;
    aclnnStatus CheckBias(QuantConvEngine &engine) const
    {
        if (engine.params.bias) {
            auto biasFormat = engine.meta.bias.format;
            if (biasFormat != Format::FORMAT_ND) {
                OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "bias", GeFormatToString(biasFormat),
                    GeFormatToString(Format::FORMAT_ND));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckScale(QuantConvEngine &engine) const
    {
        auto scaleFormat = engine.meta.scale.format;
        if (scaleFormat != Format::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "scale", GeFormatToString(scaleFormat),
                GeFormatToString(Format::FORMAT_ND));
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckFormat2D(QuantConvEngine &engine) const
    {
        if (!IsSocSupportND()) {
            return ACLNN_SUCCESS;
        }

        auto inputFormat = engine.meta.input.format;
        auto weightFormat = engine.meta.weight.format;
        auto outputFormat = engine.meta.output.format;
        if (inputFormat != Format::FORMAT_NCHW) {
            OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "x", GeFormatToString(inputFormat),
                GeFormatToString(Format::FORMAT_NCHW));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (weightFormat != Format::FORMAT_NCHW) {
            OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "filter", GeFormatToString(weightFormat),
                GeFormatToString(Format::FORMAT_NCHW));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (outputFormat != Format::FORMAT_NCHW) {
            OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "y", GeFormatToString(outputFormat),
                GeFormatToString(Format::FORMAT_NCHW));
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckFormat3D(QuantConvEngine &engine)
    {
        auto inputFormat = engine.meta.input.format;
        auto weightFormat = engine.meta.weight.format;
        auto outputFormat = engine.meta.output.format;
        if (inputFormat != Format::FORMAT_NCDHW) {
            OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "x", GeFormatToString(inputFormat),
                GeFormatToString(Format::FORMAT_NCDHW));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (engine.params.isWeightNz) {
            if (engine.meta.weight.storageFormat != Format::FORMAT_FRACTAL_Z_3D) {
                OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "filter",
                    GeFormatToString(engine.meta.weight.storageFormat),
                    GeFormatToString(Format::FORMAT_FRACTAL_Z_3D));
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (weightFormat != Format::FORMAT_NCDHW) {
                OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "filter",
                    GeFormatToString(weightFormat), GeFormatToString(Format::FORMAT_NCDHW));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (outputFormat != Format::FORMAT_NCDHW) {
            OP_LOGE_FOR_INVALID_FORMAT(engine.entityName, "y", GeFormatToString(outputFormat),
                GeFormatToString(Format::FORMAT_NCDHW));
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus Check(QuantConvEngine &engine) override
    {
        if (CheckBias(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckScale(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        size_t inputDimNum = engine.meta.input.shape.size();
        if (inputDimNum == QUANT_CONV_2D_DIM_SIZE) {
            return CheckFormat2D(engine);
        } else if (inputDimNum == QUANT_CONV_3D_DIM_SIZE) {
            return CheckFormat3D(engine);
        }
        return ACLNN_SUCCESS;
    }
};

class DtypesChecker : public QuantConvolutionChecker {
public:
    DtypesChecker() = default;
    ~DtypesChecker() override = default;
    aclnnStatus CheckScaleDtypeBySoc(const QuantConvEngine &engine) const
    {
        DataType scaleDtype = engine.meta.scale.dataType;
        if (IsSocSupportND()) {
            if (scaleDtype != DataType::DT_INT64 && scaleDtype != DataType::DT_UINT64 &&
                scaleDtype != DataType::DT_FLOAT) {
                OP_LOGE_FOR_INVALID_DTYPE(engine.entityName, "scale", GeDtypeToString(scaleDtype),
                    "one of {" + GeDtypeToString(DataType::DT_INT64) + ", " +
                    GeDtypeToString(DataType::DT_UINT64) + ", " + GeDtypeToString(DataType::DT_FLOAT) + "}");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (scaleDtype != DataType::DT_FLOAT) {
                OP_LOGE_FOR_INVALID_DTYPE(engine.entityName, "scale", GeDtypeToString(scaleDtype),
                    GeDtypeToString(DataType::DT_FLOAT));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus Check(QuantConvEngine &engine) override
    {
        auto ret = CheckScaleDtypeBySoc(engine);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        DataType inputDtype = engine.meta.input.dataType;
        DataType weightDtype = engine.meta.weight.dataType;
        DataType outputDtype = engine.meta.output.dataType;
        DataType biasDtype;
        vector<DataType> dtypesGroup = {inputDtype, weightDtype, outputDtype};
        size_t checkedLength = dtypesGroup.size();
        if (engine.params.bias) {
            biasDtype = engine.meta.bias.dataType;
            dtypesGroup.push_back(biasDtype);
            checkedLength++;
        }
        auto supportedDtypesGroups = GetSupportedDtypesBySoc();
        for (uint64_t kindsId = 0; kindsId < supportedDtypesGroups.size(); kindsId++) {
            if (QuantConvDtypesMatch(dtypesGroup, supportedDtypesGroups[kindsId], checkedLength)) {
                return ACLNN_SUCCESS;
            }
        }
        std::stringstream reason;
        reason << "The dtypes of these parameters support only the following combinations: ";
        if (engine.params.bias) {
            reason << VectorsToString(supportedDtypesGroups, GeDtypeToString);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(engine.entityName, "x, filter, bias, y",
                GeDtypeToString(inputDtype) + ", " + GeDtypeToString(weightDtype) + ", " +
                GeDtypeToString(biasDtype) + ", " + GeDtypeToString(outputDtype), reason.str());
        } else {
            size_t skippedBiasIdx = checkedLength;
            reason << VectorsToString(supportedDtypesGroups, GeDtypeToString, skippedBiasIdx);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(engine.entityName, "x, filter, y",
                GeDtypeToString(inputDtype) + ", " + GeDtypeToString(weightDtype) + ", " +
                GeDtypeToString(outputDtype), reason.str());
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
private:
    bool QuantConvDtypesMatch(const std::vector<DataType>& matchedList, const std::vector<DataType>& supportedList,
                              size_t checkedLength) const
    {
        if (matchedList.size() > supportedList.size()) {
            return false;
        }
        if (checkedLength > supportedList.size()) {
            return false;
        }
        for (size_t i = 0; i < checkedLength; i++) {
            if (matchedList[i] != supportedList[i]) {
                return false;
            }
        }
        return true;
    }
};

class ValuesChecker : public QuantConvolutionChecker {
public:
    ValuesChecker() = default;
    ~ValuesChecker() override = default;
    aclnnStatus Check(QuantConvEngine &engine) override
    {
        if (CheckShapeValue(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (CheckAttrValue(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (CheckOutputValue(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

private:
    static inline aclnnStatus CheckVectorValueGt0(const std::string &entityName, const std::string &paramName,
        const FVector<int64_t> &param)
    {
        for (size_t i = 0; i < param.size(); ++i) {
            if (param[i] <= 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entityName, paramName, op::FVectorToString(param),
                    "shape[" + std::to_string(i) + "] of this parameter should be greater than 0");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static inline aclnnStatus CheckVectorValueGte0(const std::string &entityName, const std::string &paramName,
        const FVector<int64_t> &param)
    {
        for (size_t i = 0; i < param.size(); ++i) {
            if (param[i] < 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entityName, paramName, op::FVectorToString(param),
                    "shape[" + std::to_string(i) + "] of this parameter should be greater than or equal to 0");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckValidString(const std::string &entityName, const std::string &paramName,
        const string &inputStr)
    {
        if (inputStr.empty()) {
            return ACLNN_SUCCESS;
        }

        if (inputStr.size() > MAX_STR_LEN) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entityName, paramName, inputStr, "The string length " +
                std::to_string(inputStr.size()) + " of this parameter exceeds the maximum value " +
                std::to_string(MAX_STR_LEN));
            return ACLNN_ERR_PARAM_INVALID;
        }
        for (char c : inputStr) {
            if (!std::isalnum(c) && c != '_') {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entityName, paramName, inputStr,
                    "The parameter value contains invalid character '" + std::string(1, c) +
                    "'. Only '0-9', 'a-z', 'A-Z' and '_' is supported");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckAttrRoundMode(QuantConvEngine &engine)
    {
        auto outputDtype = engine.meta.output.dataType;
        std::string roundModeStr;
        switch (outputDtype) {
            case DataType::DT_INT8:
            case DataType::DT_FLOAT8_E4M3FN:
                CHECK_NULLPTR(engine.params.roundMode, ACLNN_ERR_PARAM_NULLPTR);
                roundModeStr = std::string(engine.params.roundMode);
                if (roundModeStr != "rint") {
                    OP_LOGE_FOR_INVALID_VALUE(engine.entityName, "round_mode", roundModeStr, "'rint'");
                    return ACLNN_ERR_PARAM_INVALID;
                }
                break;
            case DataType::DT_HIFLOAT8:
                CHECK_NULLPTR(engine.params.roundMode, ACLNN_ERR_PARAM_NULLPTR);
                roundModeStr = std::string(engine.params.roundMode);
                if (roundModeStr != "round") {
                    OP_LOGE_FOR_INVALID_VALUE(engine.entityName, "round_mode", roundModeStr, "'round'");
                    return ACLNN_ERR_PARAM_INVALID;
                }
                break;
            default:
                if (engine.params.roundMode == nullptr) {
                    break;
                }
                OP_LOGW("the input roundMode is suggested to be set as a nullptr");
                roundModeStr = std::string(engine.params.roundMode);
                if (CheckValidString(engine.entityName, "round_mode", roundModeStr) != ACLNN_SUCCESS) {
                    return ACLNN_ERR_PARAM_INVALID;
                }
                break;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckAttrOffsetX(QuantConvEngine &engine)
    {
        auto inputDtype = engine.meta.input.dataType;
        if (inputDtype == DataType::DT_HIFLOAT8 || inputDtype == DataType::DT_FLOAT8_E4M3FN) {
            if (engine.params.offsetx != 0) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(engine.entityName, "offset_x",
                    std::to_string(engine.params.offsetx),
                    "If the dtype of x is " + GeDtypeToString(DataType::DT_HIFLOAT8) + " or " +
                    GeDtypeToString(DataType::DT_FLOAT8_E4M3FN) + ", attribute offset_x should be 0");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (engine.params.offsetx > OFFSET_X_MAX_VALUE || engine.params.offsetx < OFFSET_X_MIN_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(engine.entityName, "offset_x",
                std::to_string(engine.params.offsetx),
                "The current value is not within the valid range. The valid range is [" +
                std::to_string(OFFSET_X_MIN_VALUE) + ", " + std::to_string(OFFSET_X_MAX_VALUE) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckAttrGroups(QuantConvEngine &engine)
    {
        int64_t groups = engine.params.groups;
        int64_t fMapCin = engine.meta.input.C();
        int64_t weightCin = engine.meta.weight.C();
        int64_t cOut = engine.meta.weight.N();
        if (IsSocSupportND()) {
            if (groups <= 0L) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(engine.entityName, "groups", std::to_string(groups),
                    "the value of this parameter must be greater than 0");
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (cOut % groups != 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                    op::FVectorToString(engine.meta.weight.shape),
                    "shape[" + std::to_string(engine.meta.weight.NIdx()) +
                    "] of this parameter should be exactly divided by attribute groups(" +
                    std::to_string(groups) + ")");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (engine.params.groups != 1) {
                OP_LOGE_FOR_INVALID_VALUE(engine.entityName, "groups", std::to_string(engine.params.groups), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (weightCin * groups != fMapCin) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(engine.entityName, "x, filter",
                op::FVectorToString(engine.meta.input.shape) + ", " +
                op::FVectorToString(engine.meta.weight.shape),
                "shape[" + std::to_string(engine.meta.input.CIdx()) + "] of x must be equal to shape[" +
                std::to_string(engine.meta.weight.CIdx()) + "] of filter multiplied by attribute groups(" +
                    std::to_string(groups) + ")");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckXShape(QuantConvEngine &engine)
    {
        int64_t inputDimN = engine.meta.input.N();
        int64_t inputDimC = engine.meta.input.C();
        int64_t inputDimD = engine.meta.input.D();
        int64_t inputDimH = engine.meta.input.H();
        int64_t inputDimW = engine.meta.input.W();

        if (inputDimN <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "x",
                op::FVectorToString(engine.meta.input.shape),
                "Shape[" + std::to_string(engine.meta.input.NIdx()) + "] of x must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (inputDimC <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "x",
                op::FVectorToString(engine.meta.input.shape),
                "Shape[" + std::to_string(engine.meta.input.CIdx()) + "] of x must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (inputDimD <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "x",
                op::FVectorToString(engine.meta.input.shape),
                "Shape[" + std::to_string(engine.meta.input.DIdx()) + "] of x must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (inputDimH <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "x",
                op::FVectorToString(engine.meta.input.shape),
                "Shape[" + std::to_string(engine.meta.input.HIdx()) + "] of x must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (inputDimW <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "x",
                op::FVectorToString(engine.meta.input.shape),
                "Shape[" + std::to_string(engine.meta.input.WIdx()) + "] of x must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckFilterShape(QuantConvEngine &engine)
    {
        int64_t weightDimN = engine.meta.weight.N();
        int64_t weightDimC = engine.meta.weight.C();
        int64_t weightDimD = engine.meta.weight.D();
        int64_t weightDimH = engine.meta.weight.H();
        int64_t weightDimW = engine.meta.weight.W();
        if (weightDimN <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                op::FVectorToString(engine.meta.weight.shape),
                "Shape[" + std::to_string(engine.meta.weight.NIdx()) + "] of filter must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (weightDimC <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                op::FVectorToString(engine.meta.weight.shape),
                "Shape[" + std::to_string(engine.meta.weight.CIdx()) + "] of filter must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (weightDimD <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                op::FVectorToString(engine.meta.weight.shape),
                "Shape[" + std::to_string(engine.meta.weight.DIdx()) + "] of filter must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (weightDimH <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                op::FVectorToString(engine.meta.weight.shape),
                "Shape[" + std::to_string(engine.meta.weight.HIdx()) + "] of filter must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (weightDimW <= 0L) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                op::FVectorToString(engine.meta.weight.shape),
                "Shape[" + std::to_string(engine.meta.weight.WIdx()) + "] of filter must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckShapeValue(QuantConvEngine &engine)
    {
        if (CheckXShape(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckFilterShape(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (engine.params.isWeightNz) {
            if (CheckWeightTransValue(engine) != ACLNN_SUCCESS) {
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        int64_t weightDimN = engine.meta.weight.N();
        if (engine.params.bias) {
            if (engine.meta.bias.shape[0] != weightDimN) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(engine.entityName, "bias, filter",
                    op::FVectorToString(engine.meta.bias.shape) + ", " +
                    op::FVectorToString(engine.meta.weight.shape),
                    "shape[" + std::to_string(engine.meta.weight.NIdx()) + "] of bias must be equal to shape[" +
                    std::to_string(engine.meta.weight.NIdx()) + "] of filter");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (engine.meta.scale.shape[0] != weightDimN) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(engine.entityName, "scale, filter",
                op::FVectorToString(engine.meta.scale.shape) + ", " +
                op::FVectorToString(engine.meta.weight.shape),
                "shape[" + std::to_string(engine.meta.weight.NIdx()) + "] of scale must be equal to shape[" +
                std::to_string(engine.meta.weight.NIdx()) + "] of filter");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckWeightTransValue(QuantConvEngine &engine)
    {
        FVector<int64_t> storageshape = engine.meta.weight.storageShape;
        auto len = storageshape.size();
        for (size_t i = 0; i < len; ++i) {
            if (storageshape[i] <= 0L) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                    op::FVectorToString(engine.meta.weight.storageShape),
                    "Storage shape[" + std::to_string(i) + "] of filter must be greater than 0");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        int64_t c0 = 32;
        int64_t n0 = 16;
        int64_t n1 = (engine.meta.weight.N() + n0 - 1) / n0;
        int64_t dc1hw = engine.meta.weight.D() * ((engine.meta.weight.C() + c0 - 1) / c0) * engine.meta.weight.H() *
                        engine.meta.weight.W();
        if (len != FRACTAL_Z_3D_DIM || storageshape[0] != dc1hw || storageshape[1] != n1 || storageshape[2] != n0
           || storageshape[3] != c0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "filter",
                op::FVectorToString(engine.meta.weight.storageShape),
                "The storage shape of filter should be [" + std::to_string(dc1hw) + ", " + std::to_string(n1) + ", " +
                std::to_string(n0) + ", " + std::to_string(c0) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckAttrValue(QuantConvEngine &engine)
    {
        if (CheckVectorValueGt0(engine.entityName, "strides", engine.meta.stride) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckVectorValueGt0(engine.entityName, "dilations", engine.meta.dilation) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckVectorValueGte0(engine.entityName, "pads", engine.meta.padding) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckAttrGroups(engine) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (IsSocSupportND()) {
            if (CheckAttrRoundMode(engine) != ACLNN_SUCCESS) {
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (CheckAttrOffsetX(engine) != ACLNN_SUCCESS) {
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckOutputValue(QuantConvEngine &engine)
    {
        FVector<int64_t> outputShape = engine.meta.output.shape;
        if (CheckVectorValueGt0(engine.entityName, "y", outputShape) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto inferredOutputShape = engine.CalcOutputShape();
        for (size_t i = 0; i < inferredOutputShape.size(); i++) {
            if (inferredOutputShape[i] != outputShape[i]) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(engine.entityName, "y",
                    op::FVectorToString(engine.meta.output.shape),
                    "shape[" + std::to_string(i) + "] of y must be equal to its inferred value(" +
                    std::to_string(inferredOutputShape[i]) + ")");
                    return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }
};

class TemporaryLimitChecker : public QuantConvolutionChecker {
public:
    TemporaryLimitChecker() = default;
    ~TemporaryLimitChecker() override = default;
    aclnnStatus Check(QuantConvEngine &engine) override
    {
        SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            OP_LOGD("Get Current NpuArch: DAV_3510.");
            return ACLNN_SUCCESS;
        }
        switch (socVersion) {
            case SocVersion::ASCEND910B:
                break;
            case SocVersion::ASCEND910_93:
                break;
            default:
                OP_LOGE_FOR_INVALID_VALUE(engine.entityName, "SoC version",
                    std::string(op::ToString(socVersion).GetString()), "one of {" +
                    std::string(op::ToString(SocVersion::ASCEND910B).GetString()) + ", " +
                    std::string(op::ToString(SocVersion::ASCEND910_93).GetString()) + ", " +
                    std::string(op::ToString(SocVersion::ASCEND950).GetString()) +
                    "}");
                return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
};

static aclnnStatus CheckQuantConvParams(QuantConvEngine &engine)
{
    std::vector<unique_ptr<QuantConvolutionChecker>> checkList;
    checkList.push_back(make_unique<NullPtrChecker>());
    checkList.push_back(make_unique<DtypesChecker>());
    checkList.push_back(make_unique<DimsChecker>());
    checkList.push_back(make_unique<FormatsChecker>());
    checkList.push_back(make_unique<ValuesChecker>());
    checkList.push_back(make_unique<TemporaryLimitChecker>());
    for (size_t i = 0; i < checkList.size(); i++) {
        aclnnStatus ret = checkList[i]->Check(engine);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quant conv checker failed, index: %zu", i);
            return ret;
        }
    }

    return ACLNN_SUCCESS;
}

static aclIntArray* ViewQuantConv3dPad3dAs6d(const aclIntArray* intArray, aclOpExecutor* executor)
{
    const uint64_t newDimSize = QUANT_CONV_3D_PAD_DIM + QUANT_CONV_3D_PAD_DIM;
    int64_t data[newDimSize];
    data[QUANT_CONV_3D_PAD_HEAD_INDEX] = (*intArray)[0];
    data[QUANT_CONV_3D_PAD_TAIL_INDEX] = (*intArray)[0];
    data[QUANT_CONV_3D_PAD_TOP_INDEX] = (*intArray)[1];
    data[QUANT_CONV_3D_PAD_BOTTOM_INDEX] = (*intArray)[1];
    data[QUANT_CONV_3D_PAD_LEFT_INDEX] = (*intArray)[QUANT_CONV_3D_PAD_DIM - 1];
    data[QUANT_CONV_3D_PAD_RIGHT_INDEX] = (*intArray)[QUANT_CONV_3D_PAD_DIM - 1];
    aclIntArray* newArray = executor->AllocIntArray(data, newDimSize);
    return newArray;
}

static aclIntArray* ViewQuantConv2dPad2dAs4d(const aclIntArray* intArray, aclOpExecutor* executor)
{
    const uint64_t newDimSize = QUANT_CONV_2D_PAD_DIM * CONST_VALUE_2;
    int64_t data[newDimSize];
    data[QUANT_CONV_2D_PAD_TOP_INDEX] = (*intArray)[0];
    data[QUANT_CONV_2D_PAD_BOTTOM_INDEX] = (*intArray)[0];
    data[QUANT_CONV_2D_PAD_LEFT_INDEX] = (*intArray)[1];
    data[QUANT_CONV_2D_PAD_RIGHT_INDEX] = (*intArray)[1];
    aclIntArray* newArray = executor->AllocIntArray(data, newDimSize);
    return newArray;
}

static aclnnStatus TransDataPreProcess(const aclTensor* &input, const aclTensor* &weight, const int64_t groups,
                                       aclOpExecutor* executor, bool isWeightNz)
{
    input = l0op::TransData(input, Format::FORMAT_NDC1HWC0, groups, executor);
    CHECK_NULLPTR(input, ACLNN_ERR_INNER_NULLPTR);

    if (!isWeightNz) {
        weight = l0op::TransData(weight, Format::FORMAT_FRACTAL_Z_3D, groups, executor);
        CHECK_NULLPTR(weight, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ContiguousPreProcess(const aclTensor* &input, const aclTensor* &weight, const aclTensor* &scale,
                                        const aclTensor* &bias, aclOpExecutor* executor, bool isWeightNz)
{
    input = l0op::Contiguous(input, executor);
    CHECK_NULLPTR(input, ACLNN_ERR_INNER_NULLPTR);
    if (!isWeightNz) {
        weight = l0op::Contiguous(weight, executor);
        CHECK_NULLPTR(weight, ACLNN_ERR_INNER_NULLPTR);
    }

    scale = l0op::Contiguous(scale, executor);
    CHECK_NULLPTR(scale, ACLNN_ERR_INNER_NULLPTR);

    if (bias != nullptr) {
        bias = l0op::Contiguous(bias, executor);
        CHECK_NULLPTR(bias, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

namespace AclnnQuantConvolution {

class QuantConvolutionImpl {
public:
    virtual aclnnStatus PreProcess() = 0;
    virtual aclnnStatus Impl() = 0;
    virtual aclnnStatus PostProcess() = 0;
    virtual ~QuantConvolutionImpl()
    {
        input = nullptr;
        weight = nullptr;
        bias = nullptr;
        scale = nullptr;
        offset = nullptr;
        stride = nullptr;
        padding = nullptr;
        dilation = nullptr;
        outputPadding = nullptr;
        output = nullptr;
        executor = nullptr;
        quantConvOut = nullptr;
        roundMode = nullptr;
    }

    QuantConvolutionImpl(const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,
                         const aclTensor* scaleParam, const aclTensor* offsetParam, const aclIntArray* strideParam,
                         const aclIntArray* paddingParam, const aclIntArray* dilationParam, const bool transposedParam,
                         const aclIntArray* outputPaddingParam, const int64_t groupsParam, int32_t offsetxParam,
                         const char* roundModeParam, aclTensor* outputParam, aclOpExecutor* executorParam)
        : input(inputParam),
        weight(weightParam),
        bias(biasParam),
        scale(scaleParam),
        offset(offsetParam),
        stride(strideParam),
        padding(paddingParam),
        dilation(dilationParam),
        transposed(transposedParam),
        outputPadding(outputPaddingParam),
        groups(groupsParam),
        offsetx(offsetxParam),
        roundMode(roundModeParam),
        output(outputParam),
        executor(executorParam) {};

protected:
    const aclTensor* input = nullptr;
    const aclTensor* weight = nullptr;
    const aclTensor* bias = nullptr;
    const aclTensor* scale = nullptr;
    const aclTensor* offset = nullptr;
    const aclIntArray* stride = nullptr;
    const aclIntArray* padding = nullptr;
    const aclIntArray* dilation = nullptr;
    const bool transposed = false;
    const aclIntArray *outputPadding = nullptr;
    const int64_t groups = {1};
    int32_t offsetx = {0};
    const char* roundMode = nullptr;
    aclTensor* output = nullptr;
    aclOpExecutor* executor = nullptr;

    const aclTensor* quantConvOut = nullptr;
    std::map<std::string, L0FUNCTION> l0Functions;
    DataType outputDtype = DataType::DT_UNDEFINED;
    Format outputFormat = Format::FORMAT_ND;
};

class QuantConv3dImpl : public QuantConvolutionImpl {
private:
    bool is_weight_nz;
public:
    QuantConv3dImpl(const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,
                    const aclTensor* scaleParam, const aclTensor* offsetParam, const aclIntArray* strideParam,
                    const aclIntArray* paddingParam, const aclIntArray* dilationParam, const bool transposedParam,
                    const aclIntArray* outputPaddingParam, const int64_t groupsParam, int32_t offsetxParam,
                    const char* roundModeParam, aclTensor* outputParam, aclOpExecutor* executorParam, bool isWeightNz)
    : QuantConvolutionImpl(inputParam, weightParam, biasParam, scaleParam, offsetParam, strideParam,
                           paddingParam, dilationParam, transposedParam, outputPaddingParam, groupsParam, offsetxParam,
                           roundModeParam, outputParam, executorParam), is_weight_nz(isWeightNz) {}
    ~QuantConv3dImpl() override = default;

    aclnnStatus PreProcessV2()
    {
        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, QuantConv3dNCDHW, "QuantConvV2L0");
        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, Conv3dv2L0Func, "Conv3dV2L0");
        outputDtype = output->GetDataType();
        outputFormat = output->GetViewFormat();
        if (padding->Size() == QUANT_CONV_3D_PAD_DIM) {
            padding = ViewQuantConv3dPad3dAs6d(padding, executor);
            if (padding == nullptr) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv3d view padding as 6dim failed");
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
        auto ret = ContiguousPreProcess(input, weight, scale, bias, executor, is_weight_nz);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus PreProcess() override
    {
        if (IsSocSupportND()) {
            return PreProcessV2();
        }

        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, QuantConv3d6HdInt8ToNCDHWBf16, "QuantConv3d6HdInt8ToNCDHWBf16");
        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, QuantConv3d6HdInt8ToNCDHWFp16, "QuantConv3d6HdInt8ToNCDHWFp16");
        outputDtype = output->GetDataType();
        auto retContiguous = ContiguousPreProcess(input, weight, scale, bias, executor, is_weight_nz);
        if (retContiguous != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv3d contiguous preprocess failed");
            return retContiguous;
        }
        auto retTransData = TransDataPreProcess(input, weight, groups, executor, is_weight_nz);
        if (retTransData != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv3d transdata preprocess failed");
            return retTransData;
        }
        if (bias->GetDataType() == DataType::DT_BF16 || bias->GetDataType() == DataType::DT_FLOAT16) {
            bias = l0op::Cast(bias, DataType::DT_FLOAT, executor);
            CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        return ACLNN_SUCCESS;
    }

    aclnnStatus Impl() override
    {
        if (IsSocSupportND()) {
            bool isConv3DQuant = input->GetViewShape().GetDimNum() == CONV_3D_INPUT_DIM &&
                                 input->GetDataType() == DataType::DT_INT8 &&
                                 scale->GetDataType() == DataType::DT_FLOAT;
            if (isConv3DQuant) {
                outputFormat = op::Format::FORMAT_NDHWC;
                quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "Conv3dV2L0", input, weight, bias, scale, offset,
                    stride, padding, dilation, groups, offsetx, roundMode, outputDtype, outputFormat, executor);
            } else {
                quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "QuantConvV2L0", input, weight, bias, scale, offset,
                    stride, padding, dilation, groups, offsetx, roundMode, outputDtype, outputFormat, executor);
            }
        } else {
            if (outputDtype == DataType::DT_FLOAT16) {
                quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "QuantConv3d6HdInt8ToNCDHWFp16", input, weight, bias,
                    scale, offset, stride, padding, dilation, groups, offsetx, roundMode, outputDtype,
                    outputFormat, executor);
            } else if (outputDtype == DataType::DT_BF16) {
                quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "QuantConv3d6HdInt8ToNCDHWBf16", input, weight, bias,
                    scale, offset, stride, padding, dilation, groups, offsetx, roundMode, outputDtype,
                    outputFormat, executor);
            }
        }

        if (quantConvOut == nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv3d impl raise an unknown error");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus PostProcess() override
    {
        const aclTensor* resConvOut = quantConvOut;
        if (IsSocSupportND()) {
            bool isConv3DQuant = input->GetViewShape().GetDimNum() == CONV_3D_INPUT_DIM &&
                                 input->GetDataType() == DataType::DT_INT8 &&
                                 scale->GetDataType() == DataType::DT_FLOAT;
            if (isConv3DQuant) {
                FVector<int64_t> inputDims = {0, 4, 1, 2, 3}; // NDHWC -> NCDHW
                auto permPre = executor->AllocIntArray(inputDims.data(), inputDims.size());
                CHECK_RET(permPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
                resConvOut = l0op::Transpose(quantConvOut, permPre, executor);
                resConvOut = l0op::ReFormat(resConvOut, op::Format::FORMAT_NCDHW);
            }
        }
        auto quantConv3dViewCopyRet = l0op::ViewCopy(resConvOut, output, executor);
        CHECK_NULLPTR(quantConv3dViewCopyRet, ACLNN_ERR_RUNTIME_ERROR);
        return ACLNN_SUCCESS;
    }
};

class ExtendConv2dImpl : public QuantConvolutionImpl {
public:
    ExtendConv2dImpl(const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,
                     const aclTensor* scaleParam, const aclTensor* offsetParam, const aclIntArray* strideParam,
                     const aclIntArray* paddingParam, const aclIntArray* dilationParam, const bool transposedParam,
                     const aclIntArray* outputPaddingParam, const int64_t groupsParam, int32_t offsetxParam,
                     const char* roundModeParam, aclTensor* outputParam, aclOpExecutor* executorParam)
    : QuantConvolutionImpl(inputParam, weightParam, biasParam, scaleParam, offsetParam, strideParam,
                           paddingParam, dilationParam, transposedParam, outputPaddingParam, groupsParam, offsetxParam,
                           roundModeParam, outputParam, executorParam) {}
    ~ExtendConv2dImpl() override = default;

    aclnnStatus PreProcess() override
    {
        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, ExtendConv2dNCHW, "ExtendConv2DL0");
        outputDtype = output->GetDataType();
        if (padding->Size() == QUANT_CONV_2D_PAD_DIM) {
            padding = ViewQuantConv2dPad2dAs4d(padding, executor);
            if (padding == nullptr) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv2d view padding as 4dim failed");
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
        auto ret = ContiguousPreProcess(input, weight, scale, bias, executor, false);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv2d contiguous preprocess failed");
            return ret;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus Impl() override
    {
        quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "ExtendConv2DL0", input, weight, bias, scale, offset,
            stride, padding, dilation, groups, offsetx, roundMode, outputDtype, outputFormat, executor);
        if (quantConvOut == nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv2d impl raise an unknown error");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus PostProcess() override
    {
        auto quantConv2dViewCopyRet = l0op::ViewCopy(quantConvOut, output, executor);
        CHECK_NULLPTR(quantConv2dViewCopyRet, ACLNN_ERR_RUNTIME_ERROR);
        return ACLNN_SUCCESS;
    }
};

std::shared_ptr<QuantConvolutionImpl> CreateQuantConvolutionImpl(const std::string& entityName, const aclTensor* input,
    const aclTensor* weight, const aclTensor* bias, const aclTensor* scale, const aclTensor *offset,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, const bool transposed,
    const aclIntArray *outputPadding, const int64_t groups, int32_t offsetx, const char* roundMode,
    aclTensor* output, bool isWeightNz, aclOpExecutor* executor)

{
    size_t inputDim = input->GetViewShape().GetDimNum();
    switch (inputDim) {
        case QUANT_CONV_2D_DIM_SIZE:
            if (IsSocSupportND()) {
                return std::make_shared<ExtendConv2dImpl>(input, weight, bias, scale, offset, stride, padding, dilation,
                    transposed, outputPadding, groups, offsetx, roundMode, output, executor);
            } else {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(entityName, "x", std::to_string(inputDim),
                    "The shape dim of this parameter can be " + std::to_string(QUANT_CONV_2D_DIM_SIZE) +
                    " only when the SoC version is " + std::string(op::ToString(SocVersion::ASCEND950).GetString()));
                return nullptr;
            }

        case QUANT_CONV_3D_DIM_SIZE:
            return std::make_shared<QuantConv3dImpl>(input, weight, bias, scale, offset, stride, padding, dilation,
                transposed, outputPadding, groups, offsetx, roundMode, output, executor, isWeightNz);
        default:
            return nullptr;
    }
}

} // namespace AclnnQuantConvolution

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                                  const aclTensor *scale, const aclTensor *offset, const aclIntArray *stride,
                                                  const aclIntArray *padding, const aclIntArray *dilation,
                                                  bool transposed, const aclIntArray *outputPadding, int64_t groups,
                                                  int32_t offsetx, const char* roundMode, aclTensor* output,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantConvolution,
        DFX_IN(input, weight, bias, scale, offset, stride, padding, dilation, transposed,
               outputPadding, groups, offsetx, roundMode),
        DFX_OUT(output));

    QuantConvParams params = {input, weight, bias, scale, offset, stride, padding, dilation,
                              transposed, outputPadding, groups, offsetx, roundMode, output, false};
    QuantConvEngine quantConvEngine(params);
    quantConvEngine.entityName = "aclnnQuantConvolutionGetWorkspaceSize";
    aclnnStatus ret = CheckQuantConvParams(quantConvEngine);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    std::shared_ptr<AclnnQuantConvolution::QuantConvolutionImpl> quantConvImpl =
        AclnnQuantConvolution::CreateQuantConvolutionImpl(
            quantConvEngine.entityName, input, weight, bias, scale, offset, stride, padding, dilation, transposed,
            outputPadding, groups, offsetx, roundMode, output, false, uniqueExecutor.get());
    if (quantConvImpl == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "create quant convolution failed, convolution = nullptr");
        return ACLNN_ERR_INNER;
    }

    aclnnStatus preProcessRet = quantConvImpl->PreProcess();
    if (preProcessRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "quant convolution run preprocess failed");
        return ACLNN_ERR_INNER;
    }

    aclnnStatus implRet = quantConvImpl->Impl();
    if (implRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "quant convolution run impl failed");
        return ACLNN_ERR_INNER;
    }

    aclnnStatus postProcessRet = quantConvImpl->PostProcess();
    if (postProcessRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "quant convolution run impl postprocess failed");
        return ACLNN_ERR_INNER;
    }

    *workspaceSize = (uniqueExecutor.get())->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnQuantConvolution(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantConvolution);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantConvolutionWeightNzGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                                  const aclTensor *scale, const aclTensor *offset, const aclIntArray *stride,
                                                  const aclIntArray *padding, const aclIntArray *dilation,
                                                  bool transposed, const aclIntArray *outputPadding, int64_t groups,
                                                  int32_t offsetx, const char* roundMode, aclTensor* output,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantConvolutionWeightNz,
        DFX_IN(input, weight, bias, scale, offset, stride, padding, dilation, transposed,
               outputPadding, groups, offsetx, roundMode),
        DFX_OUT(output));

    QuantConvParams params = {input, weight, bias, scale, offset, stride, padding, dilation,
                              transposed, outputPadding, groups, offsetx, roundMode, output, true};
    QuantConvEngine quantConvEngine(params);
    quantConvEngine.entityName = "aclnnQuantConvolutionWeightNzGetWorkspaceSize";
    aclnnStatus ret = CheckQuantConvParams(quantConvEngine);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    std::shared_ptr<AclnnQuantConvolution::QuantConvolutionImpl> quantConvImpl =
        AclnnQuantConvolution::CreateQuantConvolutionImpl(
            quantConvEngine.entityName, input, weight, bias, scale, offset, stride, padding, dilation, transposed,
            outputPadding, groups, offsetx, roundMode, output, true, uniqueExecutor.get());
    if (quantConvImpl == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "create quant convolution weight nz failed, convolution = nullptr");
        return ACLNN_ERR_INNER;
    }

    aclnnStatus preProcessRet = quantConvImpl->PreProcess();
    if (preProcessRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "quant convolution weight nz run preprocess failed");
        return ACLNN_ERR_INNER;
    }

    aclnnStatus implRet = quantConvImpl->Impl();
    if (implRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "quant convolution weight nz run impl failed");
        return ACLNN_ERR_INNER;
    }

    aclnnStatus postProcessRet = quantConvImpl->PostProcess();
    if (postProcessRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "quant convolution weight nz run impl postprocess failed");
        return ACLNN_ERR_INNER;
    }

    *workspaceSize = (uniqueExecutor.get())->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnQuantConvolutionWeightNz(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantConvolutionWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif