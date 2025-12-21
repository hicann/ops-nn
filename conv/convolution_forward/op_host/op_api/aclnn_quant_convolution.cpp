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
#include "opdev/tensor_view_utils.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"

#include "convolution.h"
#include "quant_convolution.h"
#include "level0/padv3.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"

#include "matmul/common/op_host/op_api/cube_util.h"
#include "aclnn_quant_convolution.h"


using namespace op;
using namespace ge;
using namespace l0op;

using L0FUNCTION = void (*) ();
using QUANT_CONV_FUNCTION = const aclTensor* (*) (const aclTensor* input, const aclTensor* weight,
    const aclTensor* bias, const aclTensor* scale, const aclTensor *offset, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int groups, aclOpExecutor* executor);
using QUANT_CONV_V2_FUNCTION = const aclTensor* (*) (const aclTensor* input, const aclTensor* weight,
    const aclTensor* scale, const aclTensor* bias, DataType outputDtype, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int groups, int32_t offsetx, const char* roundMode,
    aclOpExecutor* executor);

#define CHECK_PARAMS_EQ(param, value)                                                                             \
    do {                                                                                                          \
        if ((param) != (value)) {                                                                                 \
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s = %s, get %s", #param, op::ToString(value).GetString(), \
                    op::ToString(param).GetString());                                                             \
            return ACLNN_ERR_PARAM_INVALID;                                                                       \
        }                                                                                                         \
    } while (0)

#define CHECK_PARAMS_GT(param, boundary)                                                                             \
    do {                                                                                                             \
        if ((param) <= (boundary)) {                                                                                 \
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %s larger than %s, but get %s", #param,                       \
                    op::ToString(boundary).GetString(), op::ToString(param).GetString());                            \
            return ACLNN_ERR_PARAM_INVALID;                                                                          \
        }                                                                                                            \
    } while (0)

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
const size_t INPUT_C_INDEX = 1;
const size_t CONST_VALUE_2 = 2;
const int32_t OFFSET_X_MAX_VALUE = 127;
const int32_t OFFSET_X_MIN_VALUE = -128;
const size_t MAX_STR_LEN = 1024;

const std::vector<std::vector<DataType>> SUPPORTED_DTYPES_GROUPS_DEFAULT = {
    // input, weight, output, bias
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_BF16, DataType::DT_FLOAT},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_BF16, DataType::DT_BF16},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_FLOAT},
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_FLOAT16}
};

const std::vector<std::vector<DataType>> SUPPORTED_DTYPES_GROUPS_910_95 = {
    // input, weight, output, bias
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_FLOAT16, DataType::DT_INT32},
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
    return GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95;
}

}

static std::vector<std::vector<DataType>> GetSupportedDtypesBySoc()
{
    return IsSocSupportND() ? SUPPORTED_DTYPES_GROUPS_910_95 : SUPPORTED_DTYPES_GROUPS_DEFAULT;
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

    FVector<int64_t> ToFVector(const op::Shape &shapeT) const
    {
        FVector<int64_t> vShape;
        if (shapeT.GetDimNum() != 0) {
            size_t dimNum = shapeT.GetDimNum();
            for (size_t idx = 0; idx < dimNum; idx++) {
                int64_t tmpVal = shapeT.GetDim(idx);
                vShape.push_back(tmpVal);
            }
        }
        return vShape;
    }

    void SetFromTensor(const aclTensor* tensor)
    {
        if (tensor == nullptr) {
            return;
        }
        format = tensor->GetViewFormat();
        dataType = tensor->GetDataType();
        string formatStr = op::ToString(format).GetString();
        tensorShape = tensor->GetViewShape();
        shape = ToFVector(tensorShape);

        auto len = shape.size();
        auto npos = formatStr.npos;
        auto index = formatStr.find('N');
        n = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('C');
        c = (index == npos || index >= len) ? 1 : shape[index];

        // 不含D轴时默认为1
        index = formatStr.find('D');
        d = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('H');
        h = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('W');
        w = (index == npos || index >= len) ? 1 : shape[index];
    }

    int64_t N() const { return n; }
    int64_t C() const { return c; }
    int64_t D() const { return d; }
    int64_t H() const { return h; }
    int64_t W() const { return w; }

public:
    op::Format format = Format::FORMAT_ND;
    op::DataType dataType = DataType::DT_UNDEFINED;
    FVector<int64_t> shape;
    op::Shape tensorShape;

private:
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
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
            scale.shape = scale.ToFVector(scale.tensorShape);
        }
        if (params.bias) {
            bias.format = params.bias->GetViewFormat();
            bias.dataType = params.bias->GetDataType();
            bias.tensorShape = params.bias->GetViewShape();
            bias.shape = bias.ToFVector(bias.tensorShape);
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
    // 存储输入输出的元数据，可被直接访问，避免多次调用Get函数带来性能损失
    QuantConvParams params;
    QuantConvMeta meta;
    explicit QuantConvEngine(QuantConvParams &quantConvParams) : params(quantConvParams) { meta.FromParams(params); }
    FVector<int64_t> CalcOutputShape() { return InferOutShape(); }

private:
    FVector<int64_t> InferOutShape()
    {
        FVector<int64_t> output = {meta.input.N(), meta.weight.N()};
        FVector<int64_t> inputShape = meta.input.shape;
        FVector<int64_t> weightShape = meta.weight.shape;

        auto newPad = ConstructPadding(meta.padding, inputShape);
        int64_t inferedShapeSize = inputShape.size() - 2;
        for (int64_t i = 0; i < inferedShapeSize; ++i) {
            int64_t xOut = (inputShape[i + INPUT_C_INDEX + 1] + newPad[i] - meta.dilation[i] *
                            (weightShape[i + INPUT_C_INDEX + 1] - 1) - 1) / meta.stride[i] + 1;
            output.push_back(xOut);
        }
        return output;
    }
};

static const aclTensor* L0FuncWarper(std::map<std::string, L0FUNCTION> l0Functions, std::string functionType,
                                     const aclTensor* input, const aclTensor* weight, const aclTensor* bias,
                                     const aclTensor* scale, const aclTensor *offset,
                                     const aclIntArray* stride, const aclIntArray* padding,
                                     const aclIntArray* dilation, const int64_t groups,
                                     int32_t offsetx, const char* roundMode, op::DataType outputDtype,
                                     aclOpExecutor *executor)
{
    const aclTensor* result = nullptr;
    if (l0Functions.find(functionType) == l0Functions.end()) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "no matched L0 function");
        return result;
    }

    L0FUNCTION fn = l0Functions.at(functionType);
    if (IsSocSupportND()) {
        result = (reinterpret_cast<QUANT_CONV_V2_FUNCTION>(fn))(input, weight, scale, bias, outputDtype, stride, padding, dilation,
                                              groups, offsetx, roundMode, executor);
    } else {
        result = (reinterpret_cast<QUANT_CONV_FUNCTION>(fn))(input, weight, bias, scale, offset, stride, padding, dilation,
                                           groups, executor);
    }

    return result;
}

#define FUNCTION_CALL_BY_OPTYPE(l0Functions, functionType, input, weight, bias, scale, offset, stride,\
                                padding, dilation, groups, offsetx, roundMode, \
                                outputDtype, executor)                                              \
    L0FuncWarper(l0Functions, functionType, input, weight, bias, scale, offset, stride, padding,    \
                 dilation, groups, offsetx, roundMode, outputDtype, executor)

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
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check inputs and output dim failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckAttrDim(engine) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check attrs dim failed");
            return ACLNN_ERR_PARAM_INVALID;
        }

        return ACLNN_SUCCESS;
    }
private:
    aclnnStatus CheckInOutDim(QuantConvEngine &engine) const
    {
        size_t inputDim = engine.meta.input.shape.size();
        size_t weightDim = engine.meta.weight.shape.size();
        size_t outputDim = engine.meta.output.shape.size();
        size_t scaleDim = engine.meta.scale.shape.size();
        CHECK_PARAMS_EQ(scaleDim, static_cast<size_t>(1));
        if (engine.params.bias) {
            size_t biasDim = engine.meta.bias.shape.size();
            CHECK_PARAMS_EQ(biasDim, static_cast<size_t>(1));
        }
        if (IsSocSupportND()) {
            if (inputDim != QUANT_CONV_2D_DIM_SIZE && inputDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected inputDim equals one of [%zu, %zu], get %zu",
                    QUANT_CONV_2D_DIM_SIZE, QUANT_CONV_3D_DIM_SIZE, inputDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (weightDim != QUANT_CONV_2D_DIM_SIZE && weightDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected weightDim equals one of [%zu, %zu], get %zu",
                    QUANT_CONV_2D_DIM_SIZE, QUANT_CONV_3D_DIM_SIZE, weightDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (outputDim != QUANT_CONV_2D_DIM_SIZE && outputDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected outputDim equals one of [%zu, %zu], get %zu",
                    QUANT_CONV_2D_DIM_SIZE, QUANT_CONV_3D_DIM_SIZE, outputDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (inputDim != weightDim || inputDim != outputDim || weightDim != outputDim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected inputDim = weightDim = outputDim, get %zu, %zu, %zu",
                    inputDim, weightDim, outputDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (inputDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected inputDim equals %zu, get %zu",
                        QUANT_CONV_3D_DIM_SIZE, inputDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (weightDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected weightDim equals %zu, get %zu",
                        QUANT_CONV_3D_DIM_SIZE, weightDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (outputDim != QUANT_CONV_3D_DIM_SIZE) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected outputDim equals one of %zu, get %zu",
                        QUANT_CONV_3D_DIM_SIZE, outputDim);
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
        CHECK_PARAMS_EQ(strideDim, QUANT_CONV_3D_STRIDE_DIM);
        CHECK_PARAMS_EQ(dilationDim, QUANT_CONV_3D_DILATION_DIM);
        if (IsSocSupportND()) {
            if (padDim != QUANT_CONV_3D_PAD_DIM && padDim != QUANT_CONV_3D_PAD_DIM * CONST_VALUE_2) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected padDim equals one of [%zu, %zu], get %zu",
                    QUANT_CONV_3D_PAD_DIM, QUANT_CONV_3D_PAD_DIM * CONST_VALUE_2, padDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (padDim != QUANT_CONV_3D_PAD_DIM) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected padDim equals %zu, get %zu",
                    QUANT_CONV_3D_PAD_DIM, padDim);
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
        CHECK_PARAMS_EQ(strideDim, QUANT_CONV_2D_STRIDE_DIM);
        CHECK_PARAMS_EQ(dilationDim, QUANT_CONV_2D_DILATION_DIM);
        if (padDim != QUANT_CONV_2D_PAD_DIM && padDim != QUANT_CONV_2D_PAD_DIM * CONST_VALUE_2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected padDim equals one of [%zu, %zu], get %zu",
                QUANT_CONV_2D_PAD_DIM, QUANT_CONV_2D_PAD_DIM * CONST_VALUE_2, padDim);
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
            CHECK_NULLPTR(engine.params.bias, ACLNN_ERR_PARAM_NULLPTR);
        }
        CHECK_NULLPTR(engine.params.input, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_NULLPTR(engine.params.weight, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_NULLPTR(engine.params.scale, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_NULLPTR(engine.params.stride, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_NULLPTR(engine.params.padding, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_NULLPTR(engine.params.dilation, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_NULLPTR(engine.params.output, ACLNN_ERR_PARAM_NULLPTR);
        return ACLNN_SUCCESS;
    }
};

class FormatsChecker : public QuantConvolutionChecker {
public:
    FormatsChecker() = default;
    ~FormatsChecker() override = default;
    aclnnStatus Check(QuantConvEngine &engine) override
    {
        auto biasFormat = engine.meta.bias.format;
        auto scaleFormat = engine.meta.scale.format;
        if (engine.params.bias) {
            CHECK_PARAMS_EQ(biasFormat, Format::FORMAT_ND);
        }
        CHECK_PARAMS_EQ(scaleFormat, Format::FORMAT_ND);

        auto inputFormat = engine.meta.input.format;
        auto weightFormat = engine.meta.weight.format;
        auto outputFormat = engine.meta.output.format;

        size_t inputDimNum = engine.meta.input.shape.size();
        if (inputDimNum == QUANT_CONV_2D_DIM_SIZE && IsSocSupportND()) {
            CHECK_PARAMS_EQ(inputFormat, Format::FORMAT_NCHW);
            CHECK_PARAMS_EQ(weightFormat, Format::FORMAT_NCHW);
            CHECK_PARAMS_EQ(outputFormat, Format::FORMAT_NCHW);
        } else if (inputDimNum == QUANT_CONV_3D_DIM_SIZE) {
            CHECK_PARAMS_EQ(inputFormat, Format::FORMAT_NCDHW);
            CHECK_PARAMS_EQ(weightFormat, Format::FORMAT_NCDHW);
            CHECK_PARAMS_EQ(outputFormat, Format::FORMAT_NCDHW);
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
            if (scaleDtype != DataType::DT_INT64 && scaleDtype != DataType::DT_UINT64) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected scaleDtype equals one of [%s, %s], get %s",
                    op::ToString(DataType::DT_INT64).GetString(), op::ToString(DataType::DT_UINT64).GetString(),
                    op::ToString(scaleDtype).GetString());
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (scaleDtype != DataType::DT_FLOAT) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected scaleDtype equals %s, get %s",
                    op::ToString(DataType::DT_FLOAT).GetString(), op::ToString(scaleDtype).GetString());
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
        if (engine.params.bias) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "quant Conv Dtypes Match failed, get [fmap, weight, bias, output]: [%s, %s, %s, %s]",
                op::ToString(inputDtype).GetString(), op::ToString(weightDtype).GetString(),
                op::ToString(biasDtype).GetString(), op::ToString(outputDtype).GetString());
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "quant Conv Dtypes Match failed, get [fmap, weight, output]: [%s, %s, %s]",
                op::ToString(inputDtype).GetString(), op::ToString(weightDtype).GetString(),
                op::ToString(outputDtype).GetString());
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
private:
    bool QuantConvDtypesMatch(const std::vector<DataType>& matchedList, const std::vector<DataType>& supporedList,
                              size_t checkedLength) const
    {
        if (matchedList.size() > supporedList.size()) {
            return false;
        }
        if (checkedLength > supporedList.size()) {
            return false;
        }
        for (size_t i = 0; i < checkedLength; i++) {
            if (matchedList[i] != supporedList[i]) {
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
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input shape check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (CheckAttrValue(engine) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "attr check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (CheckOutputValue(engine) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output shape check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

private:
    static inline aclnnStatus CheckVectorValueGt0(const FVector<int64_t> &param)
    {
        for (size_t i = 0; i < param.size(); ++i) {
            if (param[i] <= 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %zuth value > 0, but get %ld", i + 1, param[i]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static inline aclnnStatus CheckVectorValueGte0(const FVector<int64_t> &param)
    {
        for (size_t i = 0; i < param.size(); ++i) {
            if (param[i] < 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %luth value >= 0, but get %ld", i + 1, param[i]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckVaildString(const string &inputStr)
    {
        if (inputStr.empty()) {
            return ACLNN_SUCCESS;
        }

        if (inputStr.size() > MAX_STR_LEN) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check input string length: %zu failed, string length exceed %zu.",
                inputStr.size(), MAX_STR_LEN);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!std::all_of(inputStr.begin(), inputStr.end(), [](char c) { return std::isalnum(c) || c == '_'; })) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check input string: %s failed, only support 0-9, a-z, A-Z and '_'.",
                inputStr.c_str());
            return ACLNN_ERR_PARAM_INVALID;
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
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected roundMode equals 'rint', get: %s",
                        roundModeStr.c_str());
                    return ACLNN_ERR_PARAM_INVALID;
                }
                break;
            case DataType::DT_HIFLOAT8:
                CHECK_NULLPTR(engine.params.roundMode, ACLNN_ERR_PARAM_NULLPTR);
                roundModeStr = std::string(engine.params.roundMode);
                if (roundModeStr != "round") {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected roundMode equals 'round', get: %s",
                        roundModeStr.c_str());
                    return ACLNN_ERR_PARAM_INVALID;
                }
                break;
            default:
                if (engine.params.roundMode == nullptr) {
                    break;
                }
                OP_LOGW("the input roundMode is suggested to be set as a nullptr");
                roundModeStr = std::string(engine.params.roundMode);
                if (CheckVaildString(roundModeStr) != ACLNN_SUCCESS) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the input roundMode has invalid str");
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
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "expected offsetx = 0 when input dtype is hifloat8 or float8_e4m3, get offsetx: %d",
                    engine.params.offsetx);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (engine.params.offsetx > OFFSET_X_MAX_VALUE || engine.params.offsetx < OFFSET_X_MIN_VALUE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected offsetx belong to [-128, 127], get: %d", engine.params.offsetx);
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
            CHECK_PARAMS_GT(groups, 0L);
            if (cOut % groups != 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "expected weight_cout mod groups equals 0, get weight_cout %ld, groups %ld", cOut, groups);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (engine.params.groups != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "expected groups in quantConv must be 1, but get value is: %ld.", engine.params.groups);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (weightCin * groups != fMapCin) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "expected weight_cin * groups = fMap_cin, get weight_cin %ld, groups %ld, fMap_cin %ld",
                weightCin, groups, fMapCin);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckShapeValue(QuantConvEngine &engine)
    {
        int64_t inputDimN = engine.meta.input.N();
        int64_t iuputDimC = engine.meta.input.C();
        int64_t inputDimD = engine.meta.input.D();
        int64_t inputDimH = engine.meta.input.H();
        int64_t inputDimW = engine.meta.input.W();
        int64_t weightDimN = engine.meta.weight.N();
        int64_t weightDimC = engine.meta.weight.C();
        int64_t weightDimD = engine.meta.weight.D();
        int64_t weightDimH = engine.meta.weight.H();
        int64_t weightDimW = engine.meta.weight.W();

        // enbale empty tensor
        CHECK_PARAMS_GT(inputDimN, 0L);
        CHECK_PARAMS_GT(inputDimD, 0L);
        CHECK_PARAMS_GT(inputDimH, 0L);
        CHECK_PARAMS_GT(inputDimW, 0L);
        CHECK_PARAMS_GT(weightDimN, 0L);

        CHECK_PARAMS_GT(iuputDimC, 0L);
        CHECK_PARAMS_GT(weightDimC, 0L);
        CHECK_PARAMS_GT(weightDimD, 0L);
        CHECK_PARAMS_GT(weightDimH, 0L);
        CHECK_PARAMS_GT(weightDimW, 0L);

        if (engine.params.bias) {
            if (engine.meta.bias.shape[0] != weightDimN) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected bias shape equal cout %ld, get %ld",
                    weightDimN, engine.meta.bias.shape[0]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (engine.meta.scale.shape[0] != weightDimN) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected scale shape equal cout %ld, get %ld",
                weightDimN, engine.meta.scale.shape[0]);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckAttrValue(QuantConvEngine &engine)
    {
        if (CheckVectorValueGt0(engine.meta.stride) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check stride value greater than 0 failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckVectorValueGt0(engine.meta.dilation) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check dilation value greater than 0 failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckVectorValueGte0(engine.meta.padding) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check pad value greater than or equal to 0 failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (CheckAttrGroups(engine) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groups check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (IsSocSupportND()) {
            if (CheckAttrRoundMode(engine) != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundMode check failed");
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (CheckAttrOffsetX(engine) != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offsetx check failed");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckOutputValue(QuantConvEngine &engine)
    {
        FVector<int64_t> outputShape = engine.meta.output.shape;
        if (CheckVectorValueGt0(outputShape) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check output value greater than 0 failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto inferedOutputShape = engine.CalcOutputShape();
        for (size_t i = 0; i < inferedOutputShape.size(); i++) {
            if (inferedOutputShape[i] != outputShape[i]) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected output %zuth dim equal %ld, get %ld", i + 1,
                    inferedOutputShape[i], outputShape[i]);
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
    aclnnStatus Check([[maybe_unused]] QuantConvEngine &engine) override
    {
        SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
        switch (socVersion) {
            case SocVersion::ASCEND910B:
                break;
            case SocVersion::ASCEND910_93:
                break;
            case SocVersion::ASCEND910_95:
                break;
            default:
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "support for %s is not implemented", op::ToString(socVersion).GetString());
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
                                       aclOpExecutor* executor)
{
    input = l0op::TransData(input, Format::FORMAT_NDC1HWC0, groups, executor);
    CHECK_NULLPTR(input, ACLNN_ERR_INNER_NULLPTR);

    weight = l0op::TransData(weight, Format::FORMAT_FRACTAL_Z_3D, groups, executor);
    CHECK_NULLPTR(input, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus ContiguousPreProcess(const aclTensor* &input, const aclTensor* &weight, const aclTensor* &scale,
                                        const aclTensor* &bias, aclOpExecutor* executor)
{
    input = l0op::Contiguous(input, executor);
    CHECK_NULLPTR(input, ACLNN_ERR_INNER_NULLPTR);

    weight = l0op::Contiguous(weight, executor);
    CHECK_NULLPTR(weight, ACLNN_ERR_INNER_NULLPTR);

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
};

class QuantConv3dImpl : public QuantConvolutionImpl {
public:
    QuantConv3dImpl(const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,
                    const aclTensor* scaleParam, const aclTensor* offsetParam, const aclIntArray* strideParam,
                    const aclIntArray* paddingParam, const aclIntArray* dilationParam, const bool transposedParam,
                    const aclIntArray* outputPaddingParam, const int64_t groupsParam, int32_t offsetxParam,
                    const char* roundModeParam, aclTensor* outputParam, aclOpExecutor* executorParam)
    : QuantConvolutionImpl(inputParam, weightParam, biasParam, scaleParam, offsetParam, strideParam,
                           paddingParam, dilationParam, transposedParam, outputPaddingParam, groupsParam, offsetxParam,
                           roundModeParam, outputParam, executorParam) {}
    ~QuantConv3dImpl() override = default;

    aclnnStatus PreProcessV2()
    {
        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, QuantConv3dNCDHW, "QuantConvV2L0");
        outputDtype = output->GetDataType();
        if (padding->Size() == QUANT_CONV_3D_PAD_DIM) {
            padding = ViewQuantConv3dPad3dAs6d(padding, executor);
            if (padding == nullptr) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv3d view padding as 6dim failed");
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
        auto ret = ContiguousPreProcess(input, weight, scale, bias, executor);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv3d contiguous preprocess failed");
            return ret;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus PreProcess() override
    {
        if (IsSocSupportND()) {
            return PreProcessV2();
        }

        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, QuantConv3d6HdInt8To6HdBf16, "QuantConv3d6HdInt8To6HdBf16");
        REG_L0_FUNCTION_BY_OPTYPE(l0Functions, QuantConv3d6HdInt8To6HdFp16, "QuantConv3d6HdInt8To6HdFp16");
        outputDtype = output->GetDataType();
        auto retContiguous = ContiguousPreProcess(input, weight, scale, bias, executor);
        if (retContiguous != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv3d contiguous preprocess failed");
            return retContiguous;
        }
        auto retTransData = TransDataPreProcess(input, weight, groups, executor);
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
            quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "QuantConvV2L0", input, weight, bias, scale, offset,
                                                   stride, padding, dilation, groups, offsetx, roundMode, outputDtype, executor);
        } else {
            if (outputDtype == DataType::DT_FLOAT16) {
                quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "QuantConv3d6HdInt8To6HdFp16", input, weight, bias, scale, offset,
                                                       stride, padding, dilation, groups, offsetx, roundMode, outputDtype, executor);
            } else if (outputDtype == DataType::DT_BF16) {
                quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "QuantConv3d6HdInt8To6HdBf16", input, weight, bias, scale, offset,
                                                       stride, padding, dilation, groups, offsetx, roundMode, outputDtype, executor);
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
        if (!IsSocSupportND()) {
            resConvOut = l0op::TransData(quantConvOut, output->GetStorageFormat(), groups, executor);
            CHECK_RET(resConvOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
        auto ret = ContiguousPreProcess(input, weight, scale, bias, executor);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quant conv2d contiguous preprocess failed");
            return ret;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus Impl() override
    {
        quantConvOut = FUNCTION_CALL_BY_OPTYPE(l0Functions, "ExtendConv2DL0", input, weight, bias, scale, offset,
            stride, padding, dilation, groups, offsetx, roundMode, outputDtype, executor);
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

std::shared_ptr<QuantConvolutionImpl> CreateQuantConvolutionImpl(const aclTensor* input, const aclTensor* weight,
    const aclTensor* bias, const aclTensor* scale, const aclTensor *offset,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, const bool transposed,
    const aclIntArray *outputPadding, const int64_t groups, int32_t offsetx, const char* roundMode,
    aclTensor* output, aclOpExecutor* executor)

{
    size_t inputDim = input->GetViewShape().GetDimNum();
    switch (inputDim) {
        case QUANT_CONV_2D_DIM_SIZE:
            if (IsSocSupportND()) {
                return std::make_shared<ExtendConv2dImpl>(input, weight, bias, scale, offset, stride, padding, dilation,
                    transposed, outputPadding, groups, offsetx, roundMode, output, executor);
            } else {
                OP_LOGE(ACLNN_ERR_INNER, "cur soc not support quant conv2d");
                return nullptr;
            }

        case QUANT_CONV_3D_DIM_SIZE:
            return std::make_shared<QuantConv3dImpl>(input, weight, bias, scale, offset, stride, padding, dilation,
                transposed, outputPadding, groups, offsetx, roundMode, output, executor);
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
                              transposed, outputPadding, groups, offsetx, roundMode, output};
    QuantConvEngine quantConvEngine(params);
    aclnnStatus ret = CheckQuantConvParams(quantConvEngine);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ret, "check quant params failed");
        return ret;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    std::shared_ptr<AclnnQuantConvolution::QuantConvolutionImpl> quantConvImpl =
        AclnnQuantConvolution::CreateQuantConvolutionImpl(
            input, weight, bias, scale, offset, stride, padding, dilation, transposed, outputPadding, groups, offsetx,
            roundMode, output, uniqueExecutor.get());
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

#ifdef __cplusplus
}
#endif