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
 * \file aclnn_convolution.cpp
 * \brief
 */

#include "aclnn_convolution.h"
#include "convolution_util.h"

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include "level0/add.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "convolution.h"
#include "level0/padv3.h"
#include "aclnn_kernels/reshape.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/unsqueeze.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"

using namespace op;
using namespace ge;
using namespace l0op;
using namespace ConvolutionUtil;
using namespace Ops::NN;

namespace op {
static inline ge::AscendString ToString(const std::int64_t value)
{
    return ge::AscendString(std::to_string(value).c_str());
}
} // namespace op

namespace op {
static constexpr int64_t specialStride = 63;
static constexpr int64_t specialChannelIndex = 3;
static constexpr int64_t SMALL_CHANNEL = 4;
static constexpr int64_t CONV2D_SHAPE_SIZE = 4;
static constexpr int64_t CONV3D_SHAPE_SIZE = 5;
static const std::string REFLECTION_MODE = "constant";

// 根据API定义，需要列出所能支持的所有dtype
static constexpr const std::initializer_list<op::DataType> BIAS_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static constexpr const std::initializer_list<op::DataType> BIAS_SUPPORT_LIST_ASCEND310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
} // namespace op
/**
 * --------------------------------------L0函数注册机制start------------------------------------------------
 * 以下逻辑支持将L0函数注册到一个map里，在各convXX类中实例化这个map
 * L0FUNCTION类型代表通用的L0函数定义，作为map的value类型, 逻辑上相当于一个占位符
 * XXX_FUNCTION类型代表不同L0类别的函数指针
 * GET_FUNC_ID宏通过输入输出的类别、format确定一个唯一的ID，作为函数map的key
 * REG_L0_FUNCTION宏，可将function注册进map
 * FUNCTION_CALL进行实际的函数调用，此处调用了L0函数的适配器ConvL0Warper
 */
using L0FUNCTION = void (*)();

using CONV_FUNCTION = const aclTensor* (*)(const aclTensor* input, const aclTensor* weight, const aclTensor* bias,
                                           const aclIntArray* stride, const aclIntArray* padding,
                                           const aclIntArray* dilation, int groups, aclOpExecutor* executor);

using CONV_WITHFLAG_FUNCTION = const aclTensor* (*)(const aclTensor* input, const aclTensor* weight,
                                                    const aclTensor* bias, const aclIntArray* stride,
                                                    const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                                    bool useHf32, aclOpExecutor* executor);

using CONVTRANSPOSE_FUNCTION = const aclTensor* (*)(const aclTensor* input, const aclTensor* weight,
                                                    const aclTensor* bias, const aclIntArray* stride,
                                                    const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                                    const aclIntArray* outputPadding, aclOpExecutor* executor);

using CONVTRANSPOSE_WITHFLAG_FUNCTION = const aclTensor* (*)(const aclTensor* input, const aclTensor* weight,
                                                             const aclTensor* bias, const aclIntArray* stride,
                                                             const aclIntArray* padding, const aclIntArray* dilation,
                                                             int groups, const aclIntArray* outputPadding, bool useHf32,
                                                             aclOpExecutor* executor);

using CONV_WITHDTYPE_FUNCTION = const aclTensor* (*)(const aclTensor* input, const aclTensor* weight,
                                                     const aclTensor* bias, op::DataType outputDtype,
                                                     const aclIntArray* stride, const aclIntArray* padding,
                                                     const aclIntArray* dilation, int groups, bool useHf32,
                                                     aclOpExecutor* executor);

namespace op {
std::string CharToString(const char* a)
{
    return std::string(a);
}
} // namespace op

#define GET_FUNC_ID(inputDtype, inputFormat, outputDtype, outputFormat)                                         \
    (CharToString(op::ToString(inputDtype).GetString()) + CharToString(op::ToString(inputFormat).GetString()) + \
     CharToString(op::ToString(outputDtype).GetString()) + CharToString(op::ToString(outputFormat).GetString()))

#define REG_L0_FUNCTION(map, function, inputDtype, inputFormat, outputDtype, outputFormat) \
    ((map).emplace(                                                                        \
        (GET_FUNC_ID((inputDtype), (inputFormat), (outputDtype), (outputFormat))), (L0FUNCTION(&(function)))))

namespace op {

static const aclTensor* ConvL0Warper(
    std::map<std::string, L0FUNCTION> l0Functions, ConvolutionOpInfo& opInfo, const aclTensor* input,
    const aclTensor* weight, const aclTensor* bias, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, const bool transposed, const aclIntArray* outputPadding, const int64_t groups,
    bool useHf32, aclOpExecutor* executor)
{
    const aclTensor* result = nullptr;

    std::string funcId = GET_FUNC_ID(opInfo.inputDtype, opInfo.inputFormat, opInfo.outputDtype, opInfo.outputFormat);
    if (l0Functions.find(funcId) == l0Functions.end()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Not support the given data type and format combination: "
            "inputDtype: %s, outputDtype: %s, inputFormat:%s, outputFormat:%s",
            op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
            op::ToString(opInfo.inputFormat).GetString(), op::ToString(opInfo.outputFormat).GetString());

        return result;
    }

    L0FUNCTION fn = l0Functions.at(funcId);

    OP_LOGI("The opInfo.inputDtype is %s", op::ToString(opInfo.inputDtype).GetString());
    if (opInfo.inputDtype == op::DataType::DT_FLOAT16 || opInfo.inputDtype == op::DataType::DT_BF16 ||
        opInfo.inputDtype == op::DataType::DT_HIFLOAT8 || opInfo.inputDtype == op::DataType::DT_FLOAT8_E4M3FN) {
        if (!transposed) {
            result =
                (reinterpret_cast<CONV_FUNCTION>(fn))(input, weight, bias, stride, padding, dilation, groups, executor);
        } else {
            result = (reinterpret_cast<CONVTRANSPOSE_FUNCTION>(fn))(
                input, weight, bias, stride, padding, dilation, groups, outputPadding, executor);
        }
    } else {
        if (!transposed) {
            result = (reinterpret_cast<CONV_WITHFLAG_FUNCTION>(fn))(
                input, weight, bias, stride, padding, dilation, groups, useHf32, executor);
        } else {
            result = (reinterpret_cast<CONVTRANSPOSE_WITHFLAG_FUNCTION>(fn))(
                input, weight, bias, stride, padding, dilation, groups, outputPadding, useHf32, executor);
        }
    }
    return result;
}

#define FUNCTION_CALL(                                                                                               \
    l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, useHf32, \
    executor)                                                                                                        \
    ConvL0Warper(                                                                                                    \
        l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,      \
        useHf32, executor)
} // namespace op

#define REG_L0_FUNCTION_BY_OPTYPE(map, function, opType) ((map).emplace(opType, (L0FUNCTION(&(function)))))

namespace op {

static bool IsSupportND()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    return socVersion == SocVersion::ASCEND910_95;
}
static const aclTensor* L0FuncWarperByOpType(
    std::map<std::string, L0FUNCTION> l0Functions, std::string functionType, const aclTensor* input,
    const aclTensor* weight, const aclTensor* bias, op::DataType outputDtype, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, const bool transposed, const int64_t groups, bool useHf32,
    aclOpExecutor* executor)
{
    const aclTensor* result = nullptr;
    if (l0Functions.find(functionType) == l0Functions.end()) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "no matched L0 function");
        return result;
    }
    L0FUNCTION fn = l0Functions.at(functionType);
    if (op::IsSupportND() && input->GetViewShape().GetDimNum() == op::CONV2D_SHAPE_SIZE && !transposed) {
        result = (reinterpret_cast<CONV_WITHDTYPE_FUNCTION>(fn))(
            input, weight, bias, outputDtype, stride, padding, dilation, groups, useHf32, executor);
    }
    return result;
}

#define FUNCTION_CALL_BY_OPTYPE(                                                                                       \
    l0Functions, functionType, input, weight, bias, outputDtype, stride, padding, dilation, transposed, outputPadding, \
    groups, useHf32, executor)                                                                                         \
    L0FuncWarperByOpType(                                                                                              \
        l0Functions, functionType, input, weight, bias, outputDtype, stride, padding, dilation, transposed, groups,    \
        useHf32, executor)

} // namespace op

/* --------------------------------------L0函数注册机制end------------------------------------------------ */

/* --------------------------------------公共check能力start------------------------------------------------ */

namespace {
template <typename T>
static inline bool Equal(T a, T b)
{
    return a == b;
}

template <typename T>
static inline bool Greater(T a, T b)
{
    return a > b;
}

template <typename T>
static inline bool LessEqual(T a, T b)
{
    return a <= b;
}

template <typename T>
static inline bool Less(T a, T b)
{
    return a < b;
}

template <typename T, typename Func>
static inline bool Any([[maybe_unused]] T value, [[maybe_unused]] Func f)
{
    return false;
}
// 参数仅需满足任一参数列表判断
template <typename T, typename Func, typename... LIST>
static inline bool Any(T value, Func f, T compare, LIST... list)
{
    bool result = f(value, compare);
    if (!result) {
        return Any(value, f, list...);
    }
    return true;
}

// 参数需要满足所有参数列表判断
template <typename T, typename Func, typename... LIST>
static inline bool All(T value, Func f, T compare, LIST... list)
{
    bool result = f(value, compare);
    if (result) {
        return Any(value, f, list...);
    }
    return false;
}
} // namespace

// param必须等于给定值
#define CHECK_PARAM_EQ(param, value)                                                                          \
    do {                                                                                                      \
        if ((param) != (value)) {                                                                             \
            OP_LOGE(                                                                                          \
                ACLNN_ERR_PARAM_INVALID, "expected %s = %s, get %s", #param, op::ToString(value).GetString(), \
                op::ToString(param).GetString());                                                             \
            return ACLNN_ERR_PARAM_INVALID;                                                                   \
        }                                                                                                     \
    } while (0)

// param必须大于给定值
#define CHECK_PARAM_GT(param, boundary)                                                                          \
    do {                                                                                                         \
        if ((param) <= (boundary)) {                                                                             \
            OP_LOGE(                                                                                             \
                ACLNN_ERR_PARAM_INVALID, "expected %s > %s, get %s", #param, op::ToString(boundary).GetString(), \
                op::ToString(param).GetString());                                                                \
            return ACLNN_ERR_PARAM_INVALID;                                                                      \
        }                                                                                                        \
    } while (0)

// param必须小于给定值
#define CHECK_PARAM_LT(param, boundary)                                                                          \
    do {                                                                                                         \
        if ((param) >= (boundary)) {                                                                             \
            OP_LOGE(                                                                                             \
                ACLNN_ERR_PARAM_INVALID, "expected %s < %s, get %s", #param, op::ToString(boundary).GetString(), \
                op::ToString(param).GetString());                                                                \
            return ACLNN_ERR_PARAM_INVALID;                                                                      \
        }                                                                                                        \
    } while (0)

// param必须大于等于给定值
#define CHECK_PARAM_GTE(param, boundary)                                                                          \
    do {                                                                                                          \
        if ((param) < (boundary)) {                                                                               \
            OP_LOGE(                                                                                              \
                ACLNN_ERR_PARAM_INVALID, "expected %s >= %s, get %s", #param, op::ToString(boundary).GetString(), \
                op::ToString(param).GetString());                                                                 \
            return ACLNN_ERR_PARAM_INVALID;                                                                       \
        }                                                                                                         \
    } while (0)
/**
 * 定义CHECK系列宏，支持任意param和变长参数的比较
 */
// param满足等于参数列表之一
#define CHECK_PARAM_EQ_ONE(param, type, ...)                                                           \
    do {                                                                                               \
        if (!Any((param), Equal<type>, __VA_ARGS__)) {                                                 \
            OP_LOGE(                                                                                   \
                ACLNN_ERR_PARAM_INVALID, "expected %s equals one of %s, get %s", #param, #__VA_ARGS__, \
                op::ToString(param).GetString());                                                      \
            return ACLNN_ERR_PARAM_INVALID;                                                            \
        }                                                                                              \
    } while (0)

// 参数列表满足都等于value
#define CHECK_PARAM_ALL_EQ(value, type, ...)                                          \
    do {                                                                              \
        if (!All((value), Equal<type>, __VA_ARGS__)) {                                \
            OP_LOGE(                                                                  \
                ACLNN_ERR_PARAM_INVALID, "expected all of %s equal %s", #__VA_ARGS__, \
                op::ToString(value).GetString());                                     \
            return ACLNN_ERR_PARAM_INVALID;                                           \
        }                                                                             \
    } while (0)

// 参数列表满足都大于等于value
#define CHECK_PARAM_ALL_GTE(value, type, ...)                                                                        \
    do {                                                                                                             \
        if (!All((value), LessEqual<type>, __VA_ARGS__)) {                                                           \
            OP_LOGE(                                                                                                 \
                ACLNN_ERR_PARAM_INVALID, "expected all of %s >= %s", #__VA_ARGS__, op::ToString(value).GetString()); \
            return ACLNN_ERR_PARAM_INVALID;                                                                          \
        }                                                                                                            \
    } while (0)

// param满足小于所有参数列表
#define CHECK_PARAM_LT_ALL(param, type, ...)                                                              \
    do {                                                                                                  \
        if (!All((param), Less<type>, __VA_ARGS__)) {                                                     \
            OP_LOGE(                                                                                      \
                ACLNN_ERR_PARAM_INVALID, "expected %s less than all of %s, get %s", #param, #__VA_ARGS__, \
                op::ToString(param).GetString());                                                         \
            return ACLNN_ERR_PARAM_INVALID;                                                               \
        }                                                                                                 \
    } while (0)

// param满足大于所有参数列表
#define CHECK_PARAM_GT_ALL(param, type, ...)                                                            \
    do {                                                                                                \
        if (!All((param), Greater<type>, __VA_ARGS__)) {                                                \
            OP_LOGE(                                                                                    \
                ACLNN_ERR_PARAM_INVALID, "expected %s greater all of %s, get %s", #param, #__VA_ARGS__, \
                op::ToString(param).GetString());                                                       \
            return ACLNN_ERR_PARAM_INVALID;                                                             \
        }                                                                                               \
    } while (0)

#define CHECK_RET_RELEASE(condition, impl, ret_value)                                                         \
    do {                                                                                                      \
        if (!(condition)) {                                                                                   \
            OP_LOGE(                                                                                          \
                ACLNN_ERR_INNER, "check the condition which is \"%s\" failed, except the condition is true.", \
                #condition);                                                                                  \
            delete (impl);                                                                                    \
            return ret_value;                                                                                 \
        }                                                                                                     \
    } while (false)

namespace {
struct ConvParams {
    const aclTensor* input;
    const aclTensor* weight;
    const aclTensor* bias;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    const bool transposed;
    const aclIntArray* outputPadding;
    const int64_t groups;
    aclTensor* output;
    int8_t cubeMathType;
    uint64_t* workspaceSize;
    aclOpExecutor** executor;
};

// Conv1d, 2d, 3d
constexpr size_t CONV_1D_DIM_SIZE = 3;
constexpr size_t CONV_2D_DIM_SIZE = 4;
constexpr size_t CONV_3D_DIM_SIZE = 5;
constexpr size_t CONST_VALUE_TWO = 2;
static constexpr uint64_t MAX_UINT16 = 65536;

struct TensorMeta {
public:
    op::Format format;
    op::DataType dataType;
    FVector<int64_t> shape;
    op::Shape tensorShape;
    TensorMeta() = default;
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

        // 未定义的shape，默认为1，实际不会被用到，将所哟类型包括一起查询，未找到的为npos
        auto len = shape.size();
        auto npos = formatStr.npos;
        auto index = formatStr.find('N');
        n_ = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('C');
        c_ = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('D');
        d_ = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('H');
        h_ = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('W');
        w_ = (index == npos || index >= len) ? 1 : shape[index];

        index = formatStr.find('L');
        l_ = (index == npos || index >= len) ? 1 : shape[index];

        // formatStr.endswith('C')
        channelLast_ = formatStr.find('C') == formatStr.length() - 1;
        isZeroTensor_ = tensor->IsEmpty();
    }
    explicit TensorMeta(const aclTensor* tensor)
    {
        this->SetFromTensor(tensor);
    }
    // candy access functions
    int64_t N() const
    {
        return n_;
    }
    int64_t C() const
    {
        return c_;
    }
    int64_t D() const
    {
        return d_;
    }
    int64_t H() const
    {
        return h_;
    }
    int64_t W() const
    {
        return w_;
    }
    int64_t L() const
    {
        return l_;
    }
    bool ChannelLast() const
    {
        return channelLast_;
    }
    bool IsZeroTensor() const
    {
        return isZeroTensor_;
    }
    FVector<int64_t> ToFVector(op::Shape& shapeT) const
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

private:
    int64_t n_ = 0;
    int64_t c_ = 0;
    int64_t d_ = 0;
    int64_t h_ = 0;
    int64_t w_ = 0;
    int64_t l_ = 0;
    bool channelLast_ = false;
    bool isZeroTensor_ = false;
};
} // namespace

namespace {

op::DataType GetUpperFloatDataType(op::DataType a, op::DataType b)
{
    OP_LOGD("The input dtype is %s and %s", op::ToString(a).GetString(), op::ToString(b).GetString());
    if (a == op::DataType::DT_DOUBLE || b == op::DataType::DT_DOUBLE) {
        return op::DataType::DT_DOUBLE;
    }
    if (a == op::DataType::DT_FLOAT || b == op::DataType::DT_FLOAT) {
        return op::DataType::DT_FLOAT;
    }
    if (a == op::DataType::DT_BF16 && b == op::DataType::DT_BF16) {
        return op::DataType::DT_BF16;
    }
    if (a == op::DataType::DT_FLOAT16 && b == op::DataType::DT_FLOAT16) {
        return op::DataType::DT_FLOAT16;
    }
    if (a == op::DataType::DT_HIFLOAT8 && b == op::DataType::DT_HIFLOAT8) {
        return op::DataType::DT_HIFLOAT8;
    }
    if (a == op::DataType::DT_FLOAT8_E4M3FN && b == op::DataType::DT_FLOAT8_E4M3FN) {
        return op::DataType::DT_FLOAT8_E4M3FN;
    }

    return op::DataType::DT_FLOAT; // 注意，原则上a,b都是float类型，若不是，则默认用FP32计算
}

struct ConvMeta {
public:
    TensorMeta input;
    TensorMeta weight;
    TensorMeta bias;
    TensorMeta output;
    // stride、dilation 按照空间分布，3维DHW，2维HW，1维L
    FVector<int64_t> stride;
    FVector<int64_t> dilation;
    // padding outputpadding 按照方向维度分布，3维3个值，代表前后、上下、左右，2维度上下、左右，1维度左右
    FVector<int64_t> padding;
    FVector<int64_t> outputPadding;
    op::DataType calculatDataType;
    void FromParams(ConvParams& params)
    {
        input.SetFromTensor(params.input);
        weight.SetFromTensor(params.weight);
        output.SetFromTensor(params.output);
        if (params.bias) {
            bias.format = params.bias->GetViewFormat();
            bias.dataType = params.bias->GetDataType();
            bias.tensorShape = params.bias->GetViewShape();
            bias.shape = bias.ToFVector(bias.tensorShape);
        }

        stride = ToVector(params.stride);
        dilation = ToVector(params.dilation);
        padding = ToVector(params.padding);
        if (params.transposed) {
            outputPadding = ToVector(params.outputPadding);
        }
        calculatDataType = GetUpperFloatDataType(input.dataType, weight.dataType);
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

constexpr size_t CONV_1D_PAD_DIM = 1;
constexpr size_t CONV_2D_PAD_DIM = 2;
constexpr size_t CONV_4D_PAD_DIM = 4;
constexpr size_t PAD_TOP_INDEX = 0;
constexpr size_t PAD_BOTTOM_INDEX = 1;
constexpr size_t PAD_LEFT_INDEX = 2;
constexpr size_t PAD_RIGHT_INDEX = 3;
constexpr size_t CONVTBC_L_INDEX = 2;
constexpr size_t CONVTBC_C_INDEX = 1;

namespace {

// 本函数的目的是给conv1d制造1维的pad数组，给conv2d制造2维的pad数组，其他类型的conv保留原数组不变
static FVector<int64_t> ConstructPad(FVector<int64_t>& oldPad, FVector<int64_t>& inputShape)
{
    FVector<int64_t> newPad;
    if (inputShape.size() == CONV_1D_DIM_SIZE) {
        if (oldPad.size() == 1) {
            newPad = {oldPad[0] + oldPad[0]};
        } else if (oldPad.size() == CONV_2D_PAD_DIM) {
            newPad = {oldPad[0] + oldPad[1]};
        } else {
            newPad = {0};
        }
    } else if (inputShape.size() == CONV_2D_DIM_SIZE) {
        if (oldPad.size() == CONV_2D_PAD_DIM) {
            newPad = {(oldPad[0] + oldPad[0]), (oldPad[1] + oldPad[1])};
        } else if (oldPad.size() == CONV_4D_PAD_DIM) {
            newPad = {
                (oldPad[PAD_TOP_INDEX] + oldPad[PAD_BOTTOM_INDEX]), (oldPad[PAD_LEFT_INDEX] + oldPad[PAD_RIGHT_INDEX])};
        } else {
            newPad = {0, 0};
        }
    } else {
        return oldPad;
    }
    return newPad;
}

struct ConvEngine {
public:
    ConvParams params;
    // 存储输入输出的元数据，可被直接访问，避免多次调用Get函数带来性能损失
    ConvMeta meta;
    explicit ConvEngine(ConvParams& convParams) : params(convParams)
    {
        meta.FromParams(params);
    }
    FVector<int64_t> CalcOutputShape()
    {
        return InferShape();
    }

private:
    FVector<int64_t> InferShape()
    {
        FVector<int64_t> output;
        FVector<int64_t> inputShape = meta.input.shape;
        int64_t inputSpaceDimIndex =
            meta.input.ChannelLast() ? 1 : 2;                   // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
        int64_t inputSpaceDimNum = meta.input.shape.size() - 2; // 空间维度大小，1d卷积时为1，2d为2，3d为3
        FVector<int64_t> weightShape = meta.weight.shape;
        int64_t weightSpaceDimIndex =
            meta.weight.ChannelLast() ? 1 : 2; // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
        // step 1: put nOut in the first place of shape; for conv and transpose mode
        output.push_back(meta.input.N());
        int64_t cOut = meta.weight.N();
        // step 2: calc spaceDim size and push back to shape
        if (!params.transposed) {
            if (inputShape.size() == CONV_1D_DIM_SIZE || inputShape.size() == CONV_2D_DIM_SIZE) {
                auto newPad = ConstructPad(meta.padding, inputShape);
                for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
                    int64_t xOut = (inputShape[i + inputSpaceDimIndex] + newPad[i] -
                                    meta.dilation[i] * (weightShape[i + weightSpaceDimIndex] - 1) - 1) /
                                       meta.stride[i] +
                                   1;
                    output.push_back(xOut);
                }
            } else {
                for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
                    int64_t xOut = (inputShape[i + inputSpaceDimIndex] + CONV_2D_PAD_DIM * meta.padding[i] -
                                    meta.dilation[i] * (weightShape[i + weightSpaceDimIndex] - 1) - 1) /
                                       meta.stride[i] +
                                   1;
                    output.push_back(xOut);
                }
            }
        } else {
            cOut = meta.weight.C() * params.groups;
            if (inputShape.size() == CONV_2D_DIM_SIZE) {
                auto newPad = ConstructPad(meta.padding, inputShape);
                for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
                    int64_t xOut = (inputShape[i + inputSpaceDimIndex] - 1) * meta.stride[i] - newPad[i] +
                                   meta.dilation[i] * (weightShape[i + weightSpaceDimIndex] - 1) +
                                   meta.outputPadding[i] + 1;
                    output.push_back(xOut);
                }
            } else {
                for (int64_t i = 0; i < inputSpaceDimNum; ++i) {
                    int64_t xOut = (inputShape[i + inputSpaceDimIndex] - 1) * meta.stride[i] - 2 * meta.padding[i] +
                                   meta.dilation[i] * (weightShape[i + weightSpaceDimIndex] - 1) +
                                   meta.outputPadding[i] + 1;
                    output.push_back(xOut);
                }
            }
        }
        // last step : put cOut in right place
        if (meta.input.ChannelLast()) {
            output.push_back(cOut);
        } else {
            output.insert(output.begin() + 1, cOut);
        }
        return output;
    }
};

} // namespace

class ConvolutionChecker {
public:
    ConvolutionChecker() = default;
    virtual ~ConvolutionChecker() = default;
    virtual aclnnStatus Check(ConvEngine& engine) = 0;
};

} // namespace

namespace {

static bool CheckConvParamsDtype(
    const DataType& currDtype, const string& dtypeStr, const std::initializer_list<op::DataType>& dtypeSupportList)
{
    for (auto item : dtypeSupportList) {
        if (currDtype == item) {
            return true;
        }
    }
    string dtypeSupportListStr = "[";
    uint32_t cnt = 0;
    for (auto item : dtypeSupportList) {
        dtypeSupportListStr += op::ToString(item).GetString();
        if (cnt != dtypeSupportList.size() - 1) {
            dtypeSupportListStr += ", ";
        } else {
            dtypeSupportListStr += "]";
        }
        cnt++;
    }
    OP_LOGE(
        ACLNN_ERR_PARAM_INVALID, "expected %s %s is not in dtypeSupportList %s.", dtypeStr.c_str(),
        op::ToString(currDtype).GetString(), dtypeSupportListStr.c_str());
    return false;
}

} // namespace

/* --------------------------------------公共check能力end------------------------------------------------ */

static const std::initializer_list<op::DataType>& GetBiasDtypeSupportListBySocVersion()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND310P) {
        return op::BIAS_SUPPORT_LIST_ASCEND310P;
    }

    return op::BIAS_SUPPORT_LIST;
}

static bool CheckPointWise(const aclIntArray* array, int64_t value)
{
    for (uint64_t i = 0; i < array->Size(); ++i) {
        if ((*array)[i] != value) {
            return false;
        }
    }
    return true;
}

static bool NeedPointWiseKernel(
    const aclTensor* weight, const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation,
    const int64_t groups)
{
    if (groups != 1) {
        return false;
    }
    if (!CheckPointWise(dilation, 1) || !CheckPointWise(stride, 1) || !CheckPointWise(padding, 0)) {
        return false;
    }

    auto weightShape = weight->GetViewShape();
    size_t dimNum = weightShape.GetDimNum();
    for (size_t idx = CONST_VALUE_TWO; idx < dimNum; ++idx) {
        if (weightShape.GetDim(idx) != 1) {
            return false;
        }
    }
    return true;
}

static bool PointWiseKernelBeyondLimits(const aclTensor* fmap)
{
    auto fmapShape = fmap->GetViewShape();
    uint64_t dihiwi = 1;
    for (size_t idx = CONST_VALUE_TWO; idx < CONV_3D_DIM_SIZE; ++idx) {
        dihiwi = dihiwi * fmapShape.GetDim(idx);
    }
    return dihiwi >= MAX_UINT16;
}

namespace {

class DimChecker : public ConvolutionChecker {
public:
    DimChecker() = default;
    ~DimChecker() override = default;
    aclnnStatus CheckDim(const string& inStr, size_t inDim) const
    {
        if (inDim != CONV_1D_DIM_SIZE && inDim != CONV_2D_DIM_SIZE && inDim != CONV_3D_DIM_SIZE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "expect %s equals 3 for conv1d, 4 for conv2d or 5 for conv3d, get %s",
                inStr.c_str(), std::to_string(inDim).c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckTensorDimensions(const ConvEngine& engine) const
    {
        size_t inputDim = engine.meta.input.shape.size();
        const string inputDimStr = "inputDim";
        aclnnStatus ret = CheckDim(inputDimStr, inputDim);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        size_t weightDim = engine.meta.weight.shape.size();
        const string weightDimStr = "weightDim";
        ret = CheckDim(weightDimStr, weightDim);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        size_t outputDim = engine.meta.output.shape.size();
        const string outputDimStr = "outputDim";
        ret = CheckDim(outputDimStr, outputDim);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        CHECK_PARAM_EQ(weightDim, inputDim);
        CHECK_PARAM_EQ(outputDim, inputDim);

        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckBiasDimensions(ConvEngine& engine) const
    {
        if (engine.params.bias == nullptr) {
            return ACLNN_SUCCESS;
        }

        size_t biasDim = engine.meta.bias.shape.size();
        size_t biasSize = 0;
        size_t groupsValue = engine.params.groups;
        size_t weightNValue = engine.meta.weight.N();
        size_t weightCValue = engine.meta.weight.C();
        if (biasDim > static_cast<size_t>(0)) {
            biasSize = engine.meta.bias.shape[0];
        }

        // 如果是transpose场景, bias的维度数必须为1维, 维度大小必须为 weight C * groups
        if (engine.params.transposed && (biasDim != 1 || biasSize != weightCValue * groupsValue)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Given transposed=1, weight of size %s, expected bias to be 1-dimensional with %zu elements, "
                "but got bias of size %s instead",
                op::ToString(engine.meta.weight.tensorShape).GetString(), weightCValue * groupsValue,
                op::ToString(engine.meta.bias.tensorShape).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }

        // 如果是非transpose场景, bias的维度数必须为1维, 维度大小必须为 weight N
        if (!engine.params.transposed && (biasDim != static_cast<size_t>(1) || biasSize != weightNValue)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Given weight of size %s, expected bias to be 1-dimensional with %zu elements, "
                "but got bias of size %s instead",
                op::ToString(engine.meta.weight.tensorShape).GetString(), weightNValue,
                op::ToString(engine.meta.bias.tensorShape).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }

        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckParameterSizes(ConvEngine& engine) const
    {
        size_t inputDim = engine.meta.input.shape.size();

        auto strideSize = engine.meta.stride.size();
        CHECK_PARAM_EQ(strideSize, inputDim - 2);

        auto dilationSize = engine.meta.dilation.size();
        CHECK_PARAM_EQ(dilationSize, inputDim - 2);

        auto paddingSize = engine.meta.padding.size();
        if (((inputDim == CONV_1D_DIM_SIZE || inputDim == CONV_2D_DIM_SIZE) && !engine.params.transposed) ||
            (inputDim == CONV_2D_DIM_SIZE && engine.params.transposed)) {
            CHECK_PARAM_EQ_ONE(paddingSize, size_t, inputDim - 2, inputDim * 2 - 4);
        } else {
            CHECK_PARAM_EQ(paddingSize, inputDim - 2);
        }

        if (engine.params.transposed) {
            auto outputPaddingSize = engine.meta.outputPadding.size();
            CHECK_PARAM_EQ(outputPaddingSize, inputDim - 2);
        }

        return ACLNN_SUCCESS;
    }

    aclnnStatus Check(ConvEngine& engine) override
    {
        aclnnStatus ret = CheckTensorDimensions(engine);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = CheckBiasDimensions(engine);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        return CheckParameterSizes(engine);
    };
};

class DimCheckerTbc : public ConvolutionChecker {
public:
    DimCheckerTbc() = default;
    ~DimCheckerTbc() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        size_t inputDim = engine.meta.input.shape.size();
        CHECK_PARAM_EQ_ONE(inputDim, size_t, CONV_1D_DIM_SIZE);

        size_t weightDim = engine.meta.weight.shape.size();
        CHECK_PARAM_EQ_ONE(weightDim, size_t, CONV_1D_DIM_SIZE);

        size_t outputDim = engine.meta.output.shape.size();
        CHECK_PARAM_EQ_ONE(outputDim, size_t, CONV_1D_DIM_SIZE);

        constexpr size_t biasDimAllowTbc = 1;
        size_t biasDim = engine.meta.bias.shape.size();
        CHECK_PARAM_EQ_ONE(biasDim, size_t, biasDimAllowTbc);

        return ACLNN_SUCCESS;
    };
};

class DimCheckerDepthwise2d : public ConvolutionChecker {
public:
    DimCheckerDepthwise2d() = default;
    ~DimCheckerDepthwise2d() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        size_t inputDim = engine.meta.input.shape.size();
        CHECK_PARAM_EQ_ONE(inputDim, size_t, CONV_2D_DIM_SIZE);

        size_t weightDim = engine.meta.weight.shape.size();
        CHECK_PARAM_EQ_ONE(weightDim, size_t, CONV_2D_DIM_SIZE);

        size_t outputDim = engine.meta.output.shape.size();
        CHECK_PARAM_EQ_ONE(outputDim, size_t, CONV_2D_DIM_SIZE);

        if (engine.params.bias != nullptr) {
            size_t biasDim = engine.meta.bias.shape.size();
            constexpr size_t biasDimAllow = 1;
            CHECK_PARAM_EQ_ONE(biasDim, size_t, biasDimAllow);
        }

        auto strideSize = engine.meta.stride.size();
        CHECK_PARAM_EQ(strideSize, inputDim - 2);

        auto dilationSize = engine.meta.dilation.size();
        CHECK_PARAM_EQ(dilationSize, inputDim - 2);

        auto paddingSize = engine.meta.padding.size();
        CHECK_PARAM_EQ(paddingSize, inputDim - 2);

        return ACLNN_SUCCESS;
    };
};

class DtypeChecker : public ConvolutionChecker {
public:
    DtypeChecker() = default;
    ~DtypeChecker() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        DataType inputDtype = engine.meta.input.dataType;
        DataType weightDtype = engine.meta.weight.dataType;
        DataType biasDtype = inputDtype;
        const string inputDtypeStr = "inputDtype";
        const string weightDtypeStr = "weightDtype";
        const string biasDtypeStr = "biasDtype";
        if (engine.params.bias != nullptr) {
            biasDtype = engine.meta.bias.dataType;
            if (!CheckConvParamsDtype(biasDtype, biasDtypeStr, op::BIAS_SUPPORT_LIST)) {
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        DataType outputDtype = engine.meta.output.dataType;
        const string outputDtypeStr = "outputDtype";
        auto dtypeSupportList = GetDtypeSupportListBySocVersion();
        if (engine.params.transposed) {
            dtypeSupportList = GetDtypeSupportListBySocVersion4ConvBackward(!engine.params.transposed);
        }
        if (CheckConvParamsDtype(inputDtype, inputDtypeStr, dtypeSupportList) &&
            CheckConvParamsDtype(weightDtype, weightDtypeStr, dtypeSupportList) &&
            CheckConvParamsDtype(outputDtype, outputDtypeStr, dtypeSupportList)) {
            return ACLNN_SUCCESS;
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
};

class DtypeCheckerTbc : public ConvolutionChecker {
public:
    DtypeCheckerTbc() = default;
    ~DtypeCheckerTbc() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        DataType inputDtype = engine.meta.input.dataType;
        DataType weightDtype = engine.meta.weight.dataType;
        DataType outputDtype = engine.meta.output.dataType;
        DataType biasDtype = engine.meta.bias.dataType;

        if (engine.params.bias != nullptr) {
            biasDtype = engine.meta.bias.dataType;
            if (!CheckConvParamsDtype(biasDtype, "biasDtype", GetBiasDtypeSupportListBySocVersion())) {
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        auto dtypeSupportList = GetDtypeSupportListBySocVersion();
        if (!CheckConvParamsDtype(inputDtype, "inputDtype", dtypeSupportList)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        CHECK_PARAM_EQ(inputDtype, weightDtype);
        CHECK_PARAM_EQ(weightDtype, outputDtype);

        return ACLNN_SUCCESS;
    }
};

class DtypeCheckerDepthwise2d : public ConvolutionChecker {
public:
    DtypeCheckerDepthwise2d() = default;
    ~DtypeCheckerDepthwise2d() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        DataType inputDtype = engine.meta.input.dataType;
        DataType weightDtype = engine.meta.weight.dataType;
        DataType biasDtype = inputDtype;
        if (engine.params.bias != nullptr) {
            biasDtype = engine.meta.bias.dataType;
            if (!CheckConvParamsDtype(biasDtype, "biasDtype", GetBiasDtypeSupportListBySocVersion())) {
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (inputDtype != DataType::DT_HIFLOAT8) {
                CHECK_PARAM_EQ(inputDtype, biasDtype);
            }
        }
        DataType outputDtype = engine.meta.output.dataType;

        auto dtypeSupportList = GetDtypeSupportListBySocVersion();
        if (!CheckConvParamsDtype(inputDtype, "inputDtype", dtypeSupportList) ||
            !CheckConvParamsDtype(weightDtype, "weightDtype", dtypeSupportList) ||
            !CheckConvParamsDtype(outputDtype, "outputDtype", dtypeSupportList)) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        CHECK_PARAM_EQ(inputDtype, weightDtype);
        CHECK_PARAM_EQ(weightDtype, outputDtype);

        return ACLNN_SUCCESS;
    }
};

class NullptrChecker : public ConvolutionChecker {
public:
    NullptrChecker() = default;
    ~NullptrChecker() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        CHECK_RET(engine.params.input != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(engine.params.weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(engine.params.stride != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(engine.params.padding != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(engine.params.dilation != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(engine.params.output != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(engine.params.workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(engine.params.executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        if (engine.params.transposed) {
            CHECK_RET(engine.params.outputPadding != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        }
        return ACLNN_SUCCESS;
    }
};

class FormatChecker : public ConvolutionChecker {
public:
    FormatChecker() = default;
    ~FormatChecker() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        size_t inputDimNum = engine.meta.input.shape.size();
        auto inputFormat = engine.meta.input.format;
        auto weightFormat = engine.meta.weight.format;
        auto outputFormat = engine.meta.output.format;

        switch (inputDimNum) {
            case CONV_1D_DIM_SIZE: {
                // conv1d convtranspose1d，input weight output format都应是NCL
                CHECK_PARAM_ALL_EQ(Format::FORMAT_NCL, op::Format, inputFormat, weightFormat, outputFormat);
                break;
            }
            case CONV_2D_DIM_SIZE: {
                // conv2d input weight output format 支持NCHW. conv2d transpose支持 NCHW
                OP_LOGD("conv2d transpose: [%d]", engine.params.transposed);
                if (!engine.params.transposed) {
                    if (op::IsSupportND()) {
                        CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW);
                    } else {
                        CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW, Format::FORMAT_FRACTAL_Z);
                    }
                    CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCHW);
                    CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCHW);
                } else {
                    CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCHW);
                    CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW);
                    CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCHW);
                }
                break;
            }
            case CONV_3D_DIM_SIZE: {
                if (op::IsSupportND()) {
                    CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCDHW);
                    CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCDHW);
                    CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCDHW);
                    break;
                } else {
                    CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
                    CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
                    CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCDHW, Format::FORMAT_NDHWC);
                    break;
                }
            }
            default:
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invalid input dim: %zu", inputDimNum);
                return ACLNN_ERR_PARAM_INVALID;
        }
        // 输入和输出format要求必须一致
        if (inputFormat != outputFormat) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "expected input format the same as output format. but get input format: %s, output format: %s",
                op::ToString(inputFormat).GetString(), op::ToString(outputFormat).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        // bias format不支持 NHWC
        if (engine.params.bias != nullptr) {
            auto biasFormat = engine.meta.bias.format;
            CHECK_PARAM_EQ_ONE(
                biasFormat, op::Format, Format::FORMAT_NCL, Format::FORMAT_NCHW, Format::FORMAT_NCDHW,
                Format::FORMAT_ND);

            if (engine.params.transposed && biasFormat != Format::FORMAT_ND) {
                OP_LOGW(
                    "Please set bias format to %s, other formats may cause precision issues.",
                    op::ToString(Format::FORMAT_ND).GetString());
            }
        }

        return ACLNN_SUCCESS;
    };
};

class FormatCheckerTbc : public ConvolutionChecker {
public:
    FormatCheckerTbc() = default;
    ~FormatCheckerTbc() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        auto inputFormat = engine.meta.input.format;
        auto weightFormat = engine.meta.weight.format;
        auto outputFormat = engine.meta.output.format;

        // conv_tbc，input weight output format都应是ND或者NCL
        CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_ND, Format::FORMAT_NCL);
        CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_ND, Format::FORMAT_NCL);
        CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_ND, Format::FORMAT_NCL);

        return ACLNN_SUCCESS;
    };
};

class FormatCheckerDepthwise2d : public ConvolutionChecker {
public:
    FormatCheckerDepthwise2d() = default;
    ~FormatCheckerDepthwise2d() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        auto inputFormat = engine.meta.input.format;
        auto weightFormat = engine.meta.weight.format;
        auto outputFormat = engine.meta.output.format;

        CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCHW);
        CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW);
        CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCHW);

        // 输入和输出format要求必须一致
        if (inputFormat != outputFormat) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "expected input format the same as output format. but get input format: %s, output format: %s",
                op::ToString(inputFormat).GetString(), op::ToString(outputFormat).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    };
};

inline void GetSpatialDimInfo(
    const TensorMeta& tensor, bool& channelLast, int64_t& spaceDimIndex, size_t& spaceDimNum, FVector<int64_t>& shape)
{
    shape = tensor.shape;
    channelLast = tensor.ChannelLast();
    spaceDimIndex = channelLast ? 1 : 2; // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
    spaceDimNum = shape.size() - 2;      // 空间维度大小，1d卷积时为1，2d为2，3d为3
}

class ValueChecker : public ConvolutionChecker {
public:
    ValueChecker() = default;
    ~ValueChecker() override = default;

    aclnnStatus Check(ConvEngine& engine) override
    {
        if (CheckShape(engine.meta.input, engine.meta.weight, engine.params.transposed) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check stride
        if (CheckVectorValueGt0(engine.meta.stride) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check dilation
        if (CheckVectorValueGt0(engine.meta.dilation) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check pad
        if (CheckPad(
                engine.meta.input, engine.meta.weight, engine.meta.dilation, engine.meta.padding,
                engine.params.transposed) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pad check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check channel_value (bias, groups)
        if (engine.params.groups <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected groups > 0, get %ld.", engine.params.groups);
            return ACLNN_ERR_PARAM_INVALID;
        }

        // check channel and groups
        int64_t outChannel = -1L;
        if (CheckChannelAndGroups(engine, outChannel) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check channel and groups failed");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (engine.params.bias != nullptr) {
            if (CheckConvBias(engine.meta.bias, engine.meta.input, outChannel) != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check bias failed");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (ACLNN_SUCCESS != CheckEmptyTensor(engine.meta.input, engine.params.transposed)) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        // transposed=true时，空tensor判断
        if (engine.params.transposed && ACLNN_SUCCESS != CheckEmptyTensorTransposed(engine)) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        // 针对 2d transpose error msg is: backprop pad value invalid 提前拦截
        if (!(padBinaryValid(engine))) {
            return ACLNN_ERR_PARAM_INVALID;
        }

        return ACLNN_SUCCESS;
    };

private:
    /** 空tensor判断逻辑
     * input:
     * 在ValueChecker时，保证加上pad后，空间维度也大于0
     * 此处校验针对transpose的情况，仅支持输入的n为0，因此仅需要校验C维度是否为0
     * weight: Cout和K不为0，在ValueChecker已完成校验
     */
    static aclnnStatus CheckEmptyTensor(const TensorMeta& input, bool transposed)
    {
        if (transposed && input.C() == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input c should not be zero in transpose mode");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckEmptyTensorTransposed(ConvEngine& engine)
    {
        if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
            return ACLNN_SUCCESS;
        }

        auto inputFormat = engine.meta.input.format;
        auto weightFormat = engine.meta.weight.format;
        auto outputFormat = engine.meta.output.format;
        CHECK_PARAM_EQ_ONE(inputFormat, op::Format, Format::FORMAT_NCHW, Format::FORMAT_NCL, Format::FORMAT_NCDHW);
        CHECK_PARAM_EQ_ONE(weightFormat, op::Format, Format::FORMAT_NCHW, Format::FORMAT_NCL, Format::FORMAT_NCDHW);
        CHECK_PARAM_EQ_ONE(outputFormat, op::Format, Format::FORMAT_NCHW, Format::FORMAT_NCL, Format::FORMAT_NCDHW);
        if (inputFormat != outputFormat || inputFormat != weightFormat || weightFormat != outputFormat) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected input, output, weight to have the same format,"
                " but get input format: %s, output format: %s, weight format: %s",
                op::ToString(inputFormat).GetString(), op::ToString(outputFormat).GetString(),
                op::ToString(weightFormat).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }

        FVector<int64_t> inputShape = engine.meta.input.shape;
        FVector<int64_t> weightShape = engine.meta.weight.shape;
        FVector<int64_t> outputShape = engine.meta.output.shape;
        // NCL,NCHW,NCDHW
        if (weightShape[0] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Weigt N should not be zero in transpose mode");
            return ACLNN_ERR_PARAM_INVALID;
        }
        for (size_t i = 1; i < inputShape.size(); ++i) {
            // input仅可以N等于0
            if (inputShape[i] <= 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected input %zuth dim > 0, but get %ld", i + 1, inputShape[i]);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // weight: Cin,D,H,W可以为0，仅当output对应维度为0
            if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "expected weight %zuth dim >= 0, but get %ld,"
                    "(0 is only feasible when the %zuth dim of output is also 0)",
                    i + 1, weightShape[i], i + 1);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    // 针对 卷积2d transpose： error msg is: backprop pad value invalid 提前拦截
    // 反向pad:  (weightW - 1) * dilationW + 1 - padLeft/Right <=255   (weightH - 1) * dilationH + 1 - padUp/down
    static bool padBinaryValid(ConvEngine& engine)
    {
        if (!engine.params.transposed) {
            return true;
        }

        // transpose = true， 空tensor场景不校验pad
        if (engine.meta.input.IsZeroTensor() || engine.meta.weight.IsZeroTensor() ||
            engine.meta.output.IsZeroTensor()) {
            return true;
        }

        // 255是目前阈值，二进制不支持该值大于255进行计算
        constexpr int64_t padBinValue = 255;
        int64_t dilationH = engine.meta.dilation[0];
        int64_t dilationW = (engine.meta.dilation.size() == 1) ? engine.meta.dilation[0] : engine.meta.dilation[1];
        int64_t padTop = engine.meta.padding[0];
        int64_t padLeft = (engine.meta.padding.size() == 1) ? engine.meta.padding[0] : engine.meta.padding[1];
        auto weightShape = engine.meta.weight.tensorShape;
        int64_t weightW = engine.meta.weight.W();
        int64_t weightH = engine.meta.weight.H();
        int64_t weightL = engine.meta.weight.L();
        bool padValueValid = false;
        // 3为weight为NCL场景
        if (weightShape.GetDimNum() == 3) {
            padValueValid = ((weightL - 1) * dilationW - padLeft) <= padBinValue;
            OP_CHECK(
                padValueValid,
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Current case is not supported. ((weightL - 1) * dilationW - padLeft)=%ld "
                    "should not be larger than 255.",
                    (weightL - 1) * dilationW - padLeft),
                return false);
        } else if (weightShape.GetDimNum() == 4) { // 4为weight为NCHW / NHWC场景
            padValueValid = (((weightW - 1) * dilationW - padLeft) <= padBinValue) &&
                            (((weightH - 1) * dilationH - padTop) <= padBinValue);
            OP_CHECK(
                padValueValid,
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Current case is not supported. "
                    "((weight_w - 1) * dilation_w - padding_left)=%ld or ((weight_h - 1) * dilation_h - "
                    "padding_top)=%ld "
                    "should not be larger than 255",
                    (weightW - 1) * dilationW - padLeft, (weightH - 1) * dilationH - padTop),
                return false);
        } else if (weightShape.GetDimNum() == CONV_3D_DIM_SIZE) {
            int64_t dilationLast =
                (engine.meta.dilation.size() == 1) ? engine.meta.dilation[0] : engine.meta.dilation[2];
            int64_t padRight = (engine.meta.padding.size() == 1) ? engine.meta.padding[0] : engine.meta.padding[2];
            padValueValid = (((weightW - 1) * dilationLast - padRight) <= padBinValue) &&
                            (((weightW - 1) * dilationLast - padRight) >= 0) &&
                            (((weightH - 1) * dilationW - padLeft) <= padBinValue) &&
                            (((weightH - 1) * dilationW - padLeft) >= 0);
            OP_CHECK(
                padValueValid,
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Current case is not supported. "
                    "((weight_w - 1) * dilation_w - padding_left)=%ld or ((weight_h - 1) * dilation_h - "
                    "padding_top)=%ld or "
                    "should not be larger than 255 or less than 0",
                    (weightW - 1) * dilationLast - padRight, (weightH - 1) * dilationW - padLeft),
                return false);
        }

        return true;
    }

    static inline aclnnStatus CheckVectorValueGt0(FVector<int64_t>& param)
    {
        for (size_t i = 0; i < param.size(); ++i) {
            if (param[i] <= 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %zuth value > 0, but get %ld", i + 1, param[i]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckPad(
        TensorMeta& input, TensorMeta& weight, FVector<int64_t>& dilation,
        FVector<int64_t>& padding, bool transposed)
    {
        FVector<int64_t> inputShape, weightShape;
        bool inputChannleLast, weightChannleLast;
        int64_t inputSpaceDimIndex, weightSpaceDimIndex;
        size_t inputSpaceDimNum, weightSpaceDimNum;

        GetSpatialDimInfo(input, inputChannleLast, inputSpaceDimIndex, inputSpaceDimNum, inputShape);
        GetSpatialDimInfo(weight, weightChannleLast, weightSpaceDimIndex, weightSpaceDimNum, weightShape);

        auto newpad = ConstructPad(padding, inputShape);
        for (size_t i = 0; i < inputSpaceDimNum; ++i) {
            auto inputShapeValue = inputShape[i + inputSpaceDimIndex];
            auto weightShapeValue = weightShape[i + weightSpaceDimIndex];
            auto paddingValueFront =
                (input.shape.size() == CONV_1D_DIM_SIZE || input.shape.size() == CONV_2D_DIM_SIZE) ? newpad[i] :
                                                                                                     padding[i];
            auto dilationValue = dilation[i];

            // check input shape after pad only for conv
            if (!transposed && !input.IsZeroTensor()) {
                int64_t inputShapeValueAfterPad = -1;
                if (input.shape.size() == CONV_1D_DIM_SIZE || input.shape.size() == CONV_2D_DIM_SIZE) {
                    inputShapeValueAfterPad =
                        (inputShapeValue + paddingValueFront - dilationValue * (weightShapeValue - 1L) - 1L);
                } else {
                    inputShapeValueAfterPad =
                        (inputShapeValue + paddingValueFront * CONST_VALUE_TWO -
                         dilationValue * (weightShapeValue - 1L) - 1L);
                }

                if (inputShapeValueAfterPad < 0) {
                    OP_LOGE(
                        ACLNN_ERR_PARAM_INVALID,
                        "after pad and dilation, expect input shape[%zu] should be greater than kernerl shape[%zu], "
                        "(input + pad - dilation * (weight - 1) - 1) should >= 0, actual get: %ld", 
                        i, i, inputShapeValueAfterPad);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }

        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckConvBias(TensorMeta& bias, TensorMeta& input, int64_t outChannel)
    {
        auto biasShape = bias.shape;
        size_t biasDimNum = biasShape.size();

        // the index of C in Bias
        size_t idx_c = 0;
        if (biasDimNum != static_cast<size_t>(1)) {
            std::string str(op::ToString(input.format).GetString());
            idx_c = str.find('C');
        }

        for (size_t i = 0; i < biasDimNum; i++) {
            if (i == idx_c) {
                auto biasCout = biasShape[i];
                if (biasCout != outChannel) {
                    OP_LOGE(
                        ACLNN_ERR_PARAM_INVALID, "expected input bias size of dim_c[%ld] equal to out_channels[%ld].",
                        biasCout, outChannel);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else {
                if (biasShape[i] != 1) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected input bias size of none channel equal to 1.");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }

        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckChannelAndGroups(ConvEngine& engine, int64_t& outChannel)
    {
        int64_t inChannel = engine.meta.input.C();
        if (engine.params.transposed) {
            outChannel = engine.meta.weight.C() * engine.params.groups;
            if (engine.meta.weight.N() != inChannel) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected weight 1st dim equal to input channel in transpose mode");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // output_padding value check  output_padding参数不支持负数
            for (size_t i = 0; i < engine.meta.outputPadding.size(); i++) {
                auto outputPaddingValue = engine.meta.outputPadding[i];
                if (outputPaddingValue >= engine.meta.stride[i] && outputPaddingValue >= engine.meta.dilation[i]) {
                    OP_LOGE(
                        ACLNN_ERR_PARAM_INVALID,
                        "expected outputPadding < dilation or stride, get outputPadding %ld, dilation %ld stride %ld.",
                        outputPaddingValue, engine.meta.dilation[i], engine.meta.stride[i]);
                    return ACLNN_ERR_PARAM_INVALID;
                }

                OP_CHECK(
                    outputPaddingValue >= 0,
                    OP_LOGE(
                        ACLNN_ERR_PARAM_INVALID, "negative output_padding[%ld] is not supported", outputPaddingValue),
                    return ACLNN_ERR_PARAM_INVALID);
            }
        } else {
            outChannel = engine.meta.weight.N();
            if (engine.meta.weight.C() * engine.params.groups != inChannel) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "expected input channel equal to filter channel * groups. "
                    "get input channel %ld, filter channel %ld, groups %ld.",
                    inChannel, engine.meta.weight.C(), engine.params.groups);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (engine.meta.weight.N() % engine.params.groups != 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "expected weight 1st dim divisible by groups (including transpose mode), get weight %ld, groups %ld",
                engine.meta.weight.N(), engine.params.groups);
            return ACLNN_ERR_PARAM_INVALID;
        }

        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckShape(TensorMeta& input, TensorMeta& weight, bool transposed = false)
    {
        // check n c
        int64_t inputShapeN = input.N();
        int64_t inputShapeC = input.C();
        int64_t weightShapeN = weight.N();
        int64_t weightShapeC = weight.C();
        CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC);

        // check space(d h w or l)
        FVector<int64_t> inputShape = input.shape;
        bool inputChannleLast = input.ChannelLast();
        int64_t inputSpaceDimIndex = inputChannleLast ? 1 : 2; // 空间维度在shape中的起始位置，C维度后置时为1，否则为2
        size_t inputSpaceDimNum = input.shape.size() - 2;      // 空间维度大小，1d卷积时为1，2d为2，3d为3

        FVector<int64_t> weightShape = weight.shape;
        bool weightChannleLast = weight.ChannelLast();
        int64_t weightSpaceDimIndex = weightChannleLast ? 1 : 2; // 空间维度在shape中的起始位置，C维度后置时为1，否则为2

        // 假设是NCL，判断L的值。假设是NCHW，判断HW的值
        for (size_t i = 0; i < inputSpaceDimNum; ++i) {
            int64_t inputShapeSpace = inputShape[i + inputSpaceDimIndex]; // 空间维度的值
            if (inputShapeSpace < 0) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "expected input %zuth dim >= 0, but get %ld", i + inputSpaceDimIndex + 1,
                    inputShapeSpace);
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 && transposed &&
                weight.IsZeroTensor()) {
                continue;
            }
            int64_t weightShapeSpace = weightShape[i + weightSpaceDimIndex]; // 空间维度的值
            if (weightShapeSpace <= 0) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "expected weight %zuth dim > 0, but get %ld", i + weightSpaceDimIndex + 1,
                    weightShapeSpace);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }
};

class ValueCheckerTbc : public ConvolutionChecker {
public:
    ValueCheckerTbc() = default;
    ~ValueCheckerTbc() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        if (CheckShapeTbc(engine.meta.input, engine.meta.weight, engine.meta.output) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check conv_tbc shape failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t outChannel = engine.meta.weight.N();
        if (CheckConvBiasTbc(engine.meta.bias, outChannel) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check conv_tbc bias failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t kernelSize = engine.meta.weight.shape[CONVTBC_L_INDEX];
        CHECK_PARAM_GT(kernelSize, 0);
        if (CheckPadTbc(engine.meta.input, engine.meta.weight, engine.meta.padding) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pad check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (engine.meta.weight.shape[CONVTBC_C_INDEX] != engine.meta.input.shape[CONVTBC_C_INDEX]) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "expected input channel equal to filter channel. get input channel %ld, filter channel %ld",
                engine.meta.input.shape[CONVTBC_C_INDEX], engine.meta.weight.shape[CONVTBC_C_INDEX]);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    };

private:
    /*
    input weight output 的shape均瑶大于等于0
    bias（一维）的值要等于channel_out
    */
    aclnnStatus CheckShapeTbc(TensorMeta& input, TensorMeta& weight, TensorMeta& output) const
    {
        // check shape >= 0
        int64_t inputShapeN = input.N();
        int64_t inputShapeC = input.C();
        int64_t inputShapeL = input.L();
        int64_t weightShapeN = weight.N();
        int64_t weightShapeC = weight.C();
        int64_t weightShapeL = weight.L();
        int64_t outputShapeN = output.N();
        int64_t outputShapeC = output.C();
        int64_t outputShapeL = output.L();

        CHECK_PARAM_ALL_GTE(
            0L, int64_t, inputShapeN, inputShapeC, inputShapeL, weightShapeN, weightShapeC, weightShapeL, outputShapeN,
            outputShapeC, outputShapeL);

        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckConvBiasTbc(TensorMeta& bias, int64_t outChannel) const
    {
        auto biasShape = bias.shape;
        if (biasShape[0] != outChannel) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "expected input bias size of dim_c[%ld] equal to out_channels[%ld].",
                biasShape[0], outChannel);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckPadTbc(TensorMeta& input, TensorMeta& weight, FVector<int64_t>& padding)
    {
        FVector<int64_t> inputShape = input.shape;
        FVector<int64_t> weightShape = weight.shape;
        int64_t inputShapeL = inputShape[CONVTBC_L_INDEX];
        int64_t weightShapeL = weightShape[CONVTBC_L_INDEX];
        constexpr int64_t dilationValue = 1;
        if (!input.IsZeroTensor()) {
            int64_t inputShapeValueAfterPad = (inputShapeL + padding[0] - dilationValue * (weightShapeL - 1) - 1);
            CHECK_PARAM_GTE(inputShapeValueAfterPad, 0);
        }
        return ACLNN_SUCCESS;
    }
};

class ValueCheckerDepthwise2d : public ConvolutionChecker {
public:
    ValueCheckerDepthwise2d() = default;
    ~ValueCheckerDepthwise2d() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        if (CheckShapeDepthwise2d(engine.meta.input, engine.meta.weight) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check stride
        if (CheckVectorValueGt0Depthwise2d(engine.meta.stride) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check dilation
        if (CheckVectorValueGt0Depthwise2d(engine.meta.dilation) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check pad
        if (CheckPadDepthwise2d(
                engine.meta.input, engine.meta.weight, engine.meta.dilation, engine.meta.padding,
                engine.params.transposed) != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pad check failed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // check channel
        int64_t inChannel = engine.meta.input.C();
        int64_t outChannel = -1L;
        outChannel = engine.meta.weight.N();
        if (engine.meta.weight.C() != 0L && engine.meta.weight.C() != 1L) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "expected filter channel equal to 1. "
                "get filter channel %ld.",
                engine.meta.weight.C());
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (inChannel != 0 && outChannel % inChannel != 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "expected outChannel divisible by inChannel, get outChannel %ld, inChannel %ld", outChannel, inChannel);
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (engine.params.bias != nullptr) {
            if (CheckConvBiasDepthwise2d(engine.meta.bias, outChannel) != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check bias failed");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        return ACLNN_SUCCESS;
    };

private:
    static aclnnStatus CheckShapeDepthwise2d(TensorMeta& input, TensorMeta& weight)
    {
        // check n c
        int64_t inputShapeN = input.N();
        int64_t inputShapeC = input.C();
        int64_t weightShapeN = weight.N();
        int64_t weightShapeC = weight.C();
        CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC);
        CHECK_PARAM_GT(weightShapeN, 0L);

        // check space(d h w or l)
        FVector<int64_t> inputShape, weightShape;
        bool inputChannleLast, weightChannleLast;
        int64_t inputSpaceDimIndex, weightSpaceDimIndex;
        size_t inputSpaceDimNum, weightSpaceDimNum;

        GetSpatialDimInfo(input, inputChannleLast, inputSpaceDimIndex, inputSpaceDimNum, inputShape);
        GetSpatialDimInfo(weight, weightChannleLast, weightSpaceDimIndex, weightSpaceDimNum, weightShape);
        for (size_t i = 0; i < inputSpaceDimNum; ++i) {
            int64_t inputShapeSpace = inputShape[i + inputSpaceDimIndex]; // 空间维度的值
            if (inputShapeSpace < 0) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "expected input %zuth dim >= 0, but get %ld", i + inputSpaceDimIndex + 1,
                    inputShapeSpace);
                return ACLNN_ERR_PARAM_INVALID;
            }
            int64_t weightShapeSpace = weightShape[i + weightSpaceDimIndex]; // 空间维度的值
            if (weightShapeSpace <= 0) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "expected weight %zuth dim > 0, but get %ld", i + weightSpaceDimIndex + 1,
                    weightShapeSpace);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static inline aclnnStatus CheckVectorValueGt0Depthwise2d(FVector<int64_t>& param)
    {
        for (size_t i = 0; i < param.size(); ++i) {
            if (param[i] <= 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected %zuth value > 0, but get %ld", i + 1, param[i]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckPadDepthwise2d(
        const TensorMeta& input, const TensorMeta& weight, const FVector<int64_t>& dilation,
        const FVector<int64_t>& padding, bool transposed)
    {
        FVector<int64_t> inputShape, weightShape;
        bool inputChannleLast, weightChannleLast;
        int64_t inputSpaceDimIndex, weightSpaceDimIndex;
        size_t inputSpaceDimNum, weightSpaceDimNum;

        GetSpatialDimInfo(input, inputChannleLast, inputSpaceDimIndex, inputSpaceDimNum, inputShape);
        GetSpatialDimInfo(weight, weightChannleLast, weightSpaceDimIndex, weightSpaceDimNum, weightShape);

        for (size_t i = 0; i < inputSpaceDimNum; ++i) {
            auto inputShapeValue = inputShape[i + inputSpaceDimIndex];
            auto weightShapeValue = weightShape[i + weightSpaceDimIndex];
            auto paddingValueFront = padding[i];
            auto dilationValue = dilation[i];

            // check input shape after pad only for conv
            if (!transposed && !input.IsZeroTensor()) {
                int64_t inputShapeValueAfterPad =
                    (inputShapeValue + paddingValueFront * 2 - dilationValue * (weightShapeValue - 1) - 1);
                CHECK_PARAM_GTE(inputShapeValueAfterPad, 0);
            }
        }

        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckConvBiasDepthwise2d(TensorMeta& bias, int64_t outChannel)
    {
        auto biasShape = bias.shape;
        if (biasShape[0] != outChannel) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "expected bias shape %ld equal to out_channels %ld.", biasShape[0],
                outChannel);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
};

class ConvXdCheckerTbc : public ConvolutionChecker {
public:
    ConvXdCheckerTbc() = default;
    ~ConvXdCheckerTbc() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        FVector<int64_t> outputShape = engine.CalcOutputShape();
        for (size_t i = 0; i < outputShape.size(); i++) {
            if (outputShape[i] != engine.meta.output.shape[i]) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "expected output %zuth dim equal %ld, get %ld", i + 1, outputShape[i],
                    engine.meta.output.shape[i]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }
};

class ConvXdChecker : public ConvolutionChecker {
public:
    ConvXdChecker() = default;
    ~ConvXdChecker() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        FVector<int64_t> outputShape = engine.CalcOutputShape();
        for (size_t i = 0; i < outputShape.size(); i++) {
            if (outputShape[i] != engine.meta.output.shape[i]) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "expected output %zuth dim equal %ld, get %ld", i + 1, outputShape[i],
                    engine.meta.output.shape[i]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (engine.meta.input.IsZeroTensor() || engine.meta.weight.IsZeroTensor()) {
            if (!engine.meta.output.IsZeroTensor()) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Zero tensor input and non-zero tensor output are not supported!");
                return ACLNN_ERR_PARAM_INVALID;
            }
            for (size_t i = 0; i < outputShape.size(); i++) {
                if (outputShape[i] < 0) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, output infers negative value!");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }

        return ACLNN_SUCCESS;
    }
};

class HardwareLimitChecker : public ConvolutionChecker {
public:
    HardwareLimitChecker() = default;
    ~HardwareLimitChecker() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        DataType upperDtype = engine.meta.calculatDataType;
        CHECK_RET(CheckCubeMathType(upperDtype, engine.params.cubeMathType), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
};

class HardwareLimitCheckerTbc : public ConvolutionChecker {
public:
    HardwareLimitCheckerTbc() = default;
    ~HardwareLimitCheckerTbc() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        DataType inputDtype = engine.meta.input.dataType; // input和weight应该是一个Dtype
        CHECK_RET(CheckCubeMathType(inputDtype, engine.params.cubeMathType), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
};

class TemporarySoftwareLimitChecker : public ConvolutionChecker {
public:
    TemporarySoftwareLimitChecker() = default;
    ~TemporarySoftwareLimitChecker() override = default;
    aclnnStatus Check(ConvEngine& engine) override
    {
        size_t inputDim = engine.meta.input.shape.size();
        // 除了910A 910B 310P，其余暂不支持
        SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
        switch (socVersion) {
            case SocVersion::ASCEND910:
            case SocVersion::ASCEND910B:
            case SocVersion::ASCEND910_93:
            case SocVersion::ASCEND310P:
            case SocVersion::ASCEND910_95:
                break;
            case SocVersion::ASCEND310B: {
                if (engine.params.transposed || inputDim != CONV_2D_DIM_SIZE) {
                    OP_LOGE(
                        ACLNN_ERR_PARAM_INVALID, "aclnnConvolution only support conv2d for %s.",
                        op::ToString(socVersion).GetString());
                    return ACLNN_ERR_PARAM_INVALID;
                }
                break;
            }
            default: {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }
};

class TemporarySoftwareLimitCheckerTbc : public ConvolutionChecker {
public:
    TemporarySoftwareLimitCheckerTbc() = default;
    ~TemporarySoftwareLimitCheckerTbc() override = default;
    aclnnStatus Check([[maybe_unused]] ConvEngine& engine) override
    { // 3D暂不支持
        // 除了910A 910B 310P，其余暂不支持
        SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
        switch (socVersion) {
            case SocVersion::ASCEND910:
            case SocVersion::ASCEND910B:
            case SocVersion::ASCEND910_93:
            case SocVersion::ASCEND310P:
            case SocVersion::ASCEND910_95:
                break;
            default: {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }
};

} // namespace

namespace {
constexpr int64_t DIM_DHW_NUM = 3;
constexpr int64_t CI_DIM_CO_CI_DHW_INDEX = 1;
constexpr int64_t D_DIM_NCDHW_INDEX = 2;
constexpr int64_t H_DIM_NCDHW_INDEX = 3;
constexpr int64_t W_DIM_NCDHW_INDEX = 4;
constexpr int64_t H_DIM_NCHW_INDEX = 2;
constexpr int64_t W_DIM_NCHW_INDEX = 3;

constexpr int64_t N_DIM_NCL_INDEX = 0;
constexpr int64_t C_DIM_NCL_INDEX = 1;
constexpr int64_t L_DIM_NCL_INDEX = 2;

struct BatchMatmulInput {
    const aclTensor* leftData;
    const aclTensor* rightData;
    const aclTensor* biasData;
    const aclTensor* outputData;
    bool isLeftTranspose;
    bool isRightTranspose;
};
} // namespace

namespace {
static inline bool CheckNotNull(
    const aclTensor* self, const aclTensor* weight, const aclTensor* bias, const aclTensor* output)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

static aclnnStatus CheckConvParams(ConvEngine& engine)
{
    std::vector<unique_ptr<ConvolutionChecker>> checkList;
    // math level check
    // common checkers: nullptr, dims, format
    checkList.push_back(make_unique<NullptrChecker>());
    checkList.push_back(make_unique<DtypeChecker>());
    checkList.push_back(make_unique<DimChecker>());
    checkList.push_back(make_unique<FormatChecker>());
    checkList.push_back(make_unique<ValueChecker>());
    // different conv checkers: infershape and so on
    checkList.push_back(make_unique<ConvXdChecker>());

    // implement level check
    // hardware limit checkers:double conv, fp32 conv in 1980...
    checkList.push_back(make_unique<HardwareLimitChecker>());
    // temporary software limitation checkers: 3d conv
    checkList.push_back(make_unique<TemporarySoftwareLimitChecker>());

    for (auto& checker : checkList) {
        aclnnStatus ret = checker->Check(engine);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckConvTbcParams(ConvEngine& engine)
{
    std::vector<unique_ptr<ConvolutionChecker>> checkList;
    // math level check
    // common checkers: nullptr, dims, format
    checkList.push_back(make_unique<DtypeCheckerTbc>());
    checkList.push_back(make_unique<DimCheckerTbc>());
    checkList.push_back(make_unique<FormatCheckerTbc>());
    checkList.push_back(make_unique<ValueCheckerTbc>());
    // different conv checkers: infershape and so on
    checkList.push_back(make_unique<ConvXdCheckerTbc>());

    // implement level check
    // hardware limit checkers:double conv, fp32 conv in 1980...
    checkList.push_back(make_unique<HardwareLimitCheckerTbc>());
    // temporary software limitation checkers: 3d conv
    checkList.push_back(make_unique<TemporarySoftwareLimitCheckerTbc>());

    for (auto& checker : checkList) {
        aclnnStatus ret = checker->Check(engine);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckConvDepthwise2dKernelSize(ConvEngine& engine, const aclIntArray* kernelSize)
{
    if (engine.meta.weight.format == op::Format::FORMAT_NCL) {
        return ACLNN_SUCCESS;
    }
    int64_t weightH = engine.meta.weight.H();
    int64_t weightW = engine.meta.weight.W();
    int64_t kernelH = static_cast<int64_t>((*kernelSize)[0]);
    int64_t kernelW = static_cast<int64_t>((*kernelSize)[1]);
    if (kernelH != weightH || kernelW != weightW) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expect weight's [H, W] = kernelSize, get weight's [H, W] [%ld, %ld], kernelSize: [%ld, %ld].", weightH,
            weightW, kernelH, kernelW);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckConvDepthwise2dParams(ConvEngine& engine)
{
    std::vector<unique_ptr<ConvolutionChecker>> checkList;
    // math level check
    // common checkers: nullptr, dims, format
    checkList.push_back(make_unique<NullptrChecker>());
    checkList.push_back(make_unique<DtypeCheckerDepthwise2d>());
    checkList.push_back(make_unique<DimCheckerDepthwise2d>());
    checkList.push_back(make_unique<FormatCheckerDepthwise2d>());
    checkList.push_back(make_unique<ValueCheckerDepthwise2d>());
    // different conv checkers: infershape and so on
    checkList.push_back(make_unique<ConvXdChecker>());

    // implement level check
    // hardware limit checkers:double conv, fp32 conv in 1980...
    checkList.push_back(make_unique<HardwareLimitChecker>());
    // temporary software limitation checkers: 3d conv
    checkList.push_back(make_unique<TemporarySoftwareLimitChecker>());

    for (auto& checker : checkList) {
        aclnnStatus ret = checker->Check(engine);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckParamsNullptrTbc(
    const aclTensor* self, const aclTensor* weight, const aclTensor* bias, const aclTensor* output)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, weight, bias, output), ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutputBiasShape(const aclTensor* output, const aclTensor* bias)
{
    size_t outputDimNum = output->GetViewShape().GetDimNum();
    OP_CHECK_WRONG_DIMENSION(output, CONV_1D_DIM_SIZE, return false);

    for (size_t i = 0; i < outputDimNum; i++) {
        if (output->GetViewShape()[i] < 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "output for tbc expected %zuth value > 0, but get [%ld].", i + 1,
                output->GetViewShape()[i]);
            return false;
        }
    }
    OP_CHECK_WRONG_DIMENSION(bias, 1, return false);

    if (bias->GetViewShape()[0] != output->GetViewShape()[L_DIM_NCL_INDEX]) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "bias for tbc [%ld] should be equal to output's last dim [%ld].",
            bias->GetViewShape()[0], output->GetViewShape()[L_DIM_NCL_INDEX]);
        return false;
    }
    return true;
}

static aclnnStatus CheckOutputBiasDtype(const aclTensor* output, const aclTensor* bias)
{
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, dtypeSupportList, return false);
    return true;
}

static aclnnStatus CheckOutputBiasFormat(const aclTensor* output, const aclTensor* bias)
{
    if ((output->GetViewFormat() != op::Format::FORMAT_ND && output->GetViewFormat() != op::Format::FORMAT_NCL)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "output for tbc format should be NCL or ND, actual [%s].",
            op::ToString(output->GetViewFormat()).GetString());
        return false;
    }

    if (bias->GetViewFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "bias for tbc format should be ND, actual [%s].",
            op::ToString(bias->GetViewFormat()).GetString());
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParamsEmpty(const aclTensor* output, const aclTensor* bias)
{
    CHECK_RET(CheckOutputBiasShape(output, bias), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputBiasDtype(output, bias), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputBiasFormat(output, bias), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessBias(
    const aclTensor*& bias, const aclTensor* contiguousBias, const ConvolutionOpInfo& opInfo, bool transposed,
    aclOpExecutor* executor)
{
    if (bias != nullptr) {
        // cast
        auto castBias = l0op::Cast(contiguousBias, opInfo.biasDtype, executor);
        CHECK_RET(castBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // transdata
        if (!transposed) {
            bias = castBias;
        } else {
            bias = l0op::ReFormat(castBias, opInfo.biasFormat);
            CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

// 实现公共数据预处理，将数据准备为L0可接受的形式
static aclnnStatus CommonPreProcess(
    const aclTensor*& input, const aclTensor*& weight, const aclTensor*& bias, const int64_t groups,
    const bool transposed, const ConvolutionOpInfo& opInfo, bool changeFormat, bool contiguous, aclOpExecutor* executor)
{
    // 非连续转连续 + cast + transdata
    // input
    auto contiguousInput = input;
    auto contiguousWeight = weight;
    auto contiguousBias = bias;
    if (contiguous) {
        contiguousInput = l0op::Contiguous(input, executor);
        CHECK_RET(contiguousInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (op::GetPrimaryFormat(weight->GetStorageFormat()) != op::Format::FORMAT_FRACTAL_Z) {
            contiguousWeight = l0op::Contiguous(weight, executor);
            CHECK_RET(contiguousWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        if (bias != nullptr) {
            contiguousBias = l0op::Contiguous(bias, executor);
            CHECK_RET(contiguousBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    // cast
    auto castedInput = l0op::Cast(contiguousInput, opInfo.inputDtype, executor);
    CHECK_RET(castedInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (changeFormat) {
        // input format transdata
        input = l0op::TransData(castedInput, opInfo.inputFormat, groups, executor);
        CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        input = castedInput;
    }
    // weight
    // cast
    auto castedWeight = l0op::Cast(contiguousWeight, opInfo.weightDtype, executor);
    CHECK_RET(castedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (changeFormat) {
        // weight format transdata
        weight = l0op::TransData(castedWeight, opInfo.weightFormat, groups, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        weight = castedWeight;
    }

    // bias
    auto ret = ProcessBias(bias, contiguousBias, opInfo, transposed, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

// 实现公共数据预处理，将数据准备为L0可接受的形式  C04特殊分支
static aclnnStatus CommonPreProcessC04(
    const aclTensor*& input, const aclTensor*& weight, const aclTensor*& bias, const int64_t groups,
    const bool transposed, const ConvolutionOpInfo& opInfo, bool changeFormat, bool contiguous, aclOpExecutor* executor)
{
    auto contiguousInput = input;
    auto contiguousWeight = weight;
    auto contiguousBias = bias;
    if (contiguous) {
        contiguousInput = l0op::Contiguous(input, executor);
        CHECK_RET(contiguousInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        contiguousWeight = l0op::Contiguous(weight, executor);
        CHECK_RET(contiguousWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (bias != nullptr) {
            contiguousBias = l0op::Contiguous(bias, executor);
            CHECK_RET(contiguousBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    auto castedInput = l0op::Cast(contiguousInput, opInfo.inputDtype, executor);
    CHECK_RET(castedInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (changeFormat) {
        // input format transdata
        input = l0op::TransData(castedInput, opInfo.inputFormat, groups, executor);
        CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        input = castedInput;
    }

    // weight特殊操作规避：NCHW(1, 1, 3, 3) -> pad -> NCHW(1, 4, 3, 3) -> transpose -> NHWC(1, 3, 3, 4)
    //                   -> ND(1, 36) -> transdata -> FZ_NZ -> setformat(FZC04)
    auto castedWeight = l0op::Cast(contiguousWeight, opInfo.weightDtype, executor);
    CHECK_RET(castedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("Before c04 weight is %s", weight->ToString().GetString());
    auto initialWeightViewShape = weight->GetViewShape();
    if (changeFormat) {
        // NCHW (a, b, c, d) -> padV3 -> NCHW(a, 4, c, d)  保证C维度pad到4
        if (weight->GetViewShape().GetDim(1) != 4) {
            const_cast<aclTensor*>(weight)->SetStorageShape(weight->GetViewShape());
            const_cast<aclTensor*>(weight)->SetOriginalShape(weight->GetViewShape());
            OP_LOGD("Before padv3 weight is %s", weight->ToString().GetString());
            // [0, 0, 0, 4 - b, 0, 0, 0, 0] padding  padding总长度为8
            int64_t paddingArray[8] = {}; // Already initialized to zeros
            // CO4需求，因此需要补充维度到4。预计维度数是4，padding中的第3位要pad到4，具体取1指代NCHW中的C
            paddingArray[3] = 4 - weight->GetViewShape().GetDim(1);
            aclIntArray* paddingArrayRes = executor->AllocIntArray(paddingArray, 8);
            CHECK_RET(paddingArrayRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto paddingTensor = executor->ConvertToTensor(paddingArrayRes, DataType::DT_INT32);
            auto constantValues = executor->ConvertToTensor(executor->AllocScalar(0), weight->GetDataType());
            weight = l0op::PadV3(weight, paddingTensor, constantValues, op::REFLECTION_MODE, true, executor);
            CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
            OP_LOGD("After padv3 weight is %s", weight->ToString().GetString());
        }

        // NCHW(a, b, c, d) -> transpose -> NHWC(a, c, d, b)  因为NCHW，所以长度为4
        int64_t valuePerm[4] = {0, 1, 2, 3}; // Initialize directly
        // 1表示C，3表示W
        std::swap(valuePerm[1], valuePerm[3]); // b, c, d -> d, c, b
        // 1表示W，2表示H
        std::swap(valuePerm[1], valuePerm[2]); // d, c, b -> c, d, b
        auto perm = executor->AllocIntArray(valuePerm, 4);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        weight = l0op::Transpose(weight, perm, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const_cast<aclTensor*>(weight)->SetViewFormat(Format::FORMAT_NHWC);
        const_cast<aclTensor*>(weight)->SetStorageFormat(Format::FORMAT_NHWC);
        const_cast<aclTensor*>(weight)->SetOriginalFormat(Format::FORMAT_NHWC);
        OP_LOGD("After transpose weight is %s", weight->ToString().GetString());

        // NHWC (a, b, c, d) 转成 ND (a, b * c * d)
        auto weightShape = weight->GetViewShape();
        int64_t newFormatRes = weightShape.GetDim(1) * weightShape.GetDim(2) * weightShape.GetDim(3);
        op::Shape ndShape = op::Shape({weightShape.GetDim(0), newFormatRes});
        const_cast<aclTensor*>(weight)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(weight)->SetOriginalFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(weight)->SetStorageShape(ndShape);
        const_cast<aclTensor*>(weight)->SetOriginalShape(ndShape);
        OP_LOGD("After update format weight is %s", weight->ToString().GetString());

        // transdata: NHWC -> FZ_NZ
        weight = l0op::TransData(weight, FORMAT_FRACTAL_NZ, groups, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("After transdata weight is %s", weight->ToString().GetString());
        const_cast<aclTensor*>(weight)->SetOriginalShape(initialWeightViewShape);
        const_cast<aclTensor*>(weight)->SetOriginalFormat(Format::FORMAT_NCHW);
        auto storageShape = weight->GetStorageShape();

        // reformat
        auto weightFormatC04 = executor->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
        weight = weightFormatC04;
        const_cast<aclTensor*>(weight)->SetStorageFormat(Format::FORMAT_FRACTAL_Z_C04);
        const_cast<aclTensor*>(weight)->SetStorageShape(storageShape);
        const_cast<aclTensor*>(weight)->SetOriginalShape(initialWeightViewShape);
        const_cast<aclTensor*>(weight)->SetOriginalFormat(Format::FORMAT_NCHW);
        OP_LOGD("After reformat weight is %s", weight->ToString().GetString());
    } else {
        weight = castedWeight;
    }

    // bias
    auto ret = ProcessBias(bias, contiguousBias, opInfo, transposed, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

enum class ConvToBmmMode : std::uint8_t
{
    CONV_NO_MM = 0,
    CONV_MM_FEATURE_MAP_EQ_FILTER = 1,
};

enum class ConvTranspose1DToBmmMode : std::uint8_t
{
    CONVTRANSPOSE1D_NO_MM = 0,
    CONVTRANSPOSE1D_MM_FEATURE_MAP_EQ_FILTER = 1,
};

// 实现公共数据后处理，将数据转换为L2输出，但并不做viewcopy
static aclnnStatus CommonPostProcess(
    const int64_t groups, bool changeFormat, const aclTensor* output, const aclTensor*& convOut,
    aclOpExecutor* executor)
{
    // output format transdata
    auto formatOutput = changeFormat ? l0op::TransData(convOut, output->GetStorageFormat(), groups, executor) : convOut;
    CHECK_RET(formatOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // output cast
    auto castedOutput = l0op::Cast(formatOutput, output->GetDataType(), executor);
    CHECK_RET(castedOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    convOut = castedOutput;

    return ACLNN_SUCCESS;
}

static void UpdateOutputDtype(
    const aclTensor* output, struct ConvolutionOpInfo& opInfo, int8_t cubeMathType, DataType& upperDtype,
    const bool transposed)
{
    if (!transposed) {
        opInfo.outputDtype = upperDtype; // 目前conv2d算子底层二进制仅支持输入输出相同，暂不支持16进32出的场景
    } else {
        opInfo.outputDtype = (output->GetDataType() == op::DataType::DT_FLOAT) ? output->GetDataType() : upperDtype;
    }
    // ASCEND910 + ASCEND310P 仅支持fp16的卷积，或者USE_FP16场景必走FP16， 因此必须转为fp16实现
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910 || socVersion == SocVersion::ASCEND310P || cubeMathType == USE_FP16) {
        opInfo.outputDtype = op::DataType::DT_FLOAT16; // 目前底层二进制暂不支持16进32出的场景，故设为FP16运算
    }
    if ((upperDtype == op::DataType::DT_HIFLOAT8) && op::IsSupportND()) {
        opInfo.outputDtype =
            op::DataType::DT_HIFLOAT8; // In conv2d/3d/3dtranspose hif8 case, the output dtype should be hif8.
    }
    if ((upperDtype == op::DataType::DT_FLOAT8_E4M3FN) && op::IsSupportND()) {
        opInfo.outputDtype =
            op::DataType::DT_FLOAT8_E4M3FN; // In conv2d/3d/3dtranspose hif8 case, the output dtype should be hif8.
    }
}

static void UpdateInputDtype(
    const aclTensor* input, const aclTensor* bias, struct ConvolutionOpInfo& opInfo, DataType& upperDtype,
    const bool transposed)
{
    opInfo.inputDtype = upperDtype;
    opInfo.weightDtype = upperDtype;
    if (bias != nullptr) {
        SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
        if (transposed && (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
                           socVersion == SocVersion::ASCEND910_95)) {
            upperDtype = GetUpperFloatDataType(opInfo.outputDtype, bias->GetDataType());
        }
        opInfo.biasDtype = upperDtype;
        // 因为bias二进制不支持为BF16，所以得转成FP32
        if (upperDtype == op::DataType::DT_BF16 &&
            (!(transposed && input->GetViewShape().GetDimNum() == CONV_3D_DIM_SIZE)) && !op::IsSupportND()) {
            OP_LOGD("Since bias does not support BF16, change the dtype of bias to fp32.");
            opInfo.biasDtype = op::DataType::DT_FLOAT;
        }
        if ((upperDtype == op::DataType::DT_HIFLOAT8 || upperDtype == op::DataType::DT_FLOAT8_E4M3FN) &&
            (!transposed) && op::IsSupportND()) {
            OP_LOGD("Bias dtype must be fp32 in hif8 or fp8_e4m3fn scene for conv forward.");
            opInfo.biasDtype = op::DataType::DT_FLOAT;
        }
    }
}

void GetConvolutionOpDtype(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* output,
    struct ConvolutionOpInfo& opInfo, const bool transposed, int8_t cubeMathType)
{
    OP_LOGD("Get into GetConvolutionOpDtype Function.");
    DataType upperDtype = GetUpperFloatDataType(input->GetDataType(), weight->GetDataType());
    if (op::IsSupportND()) {
        upperDtype = CalcPromoteTypeCubeMathTypeNew(upperDtype, cubeMathType);
    } else {
        upperDtype = CalcPromoteTypeCubemathtype(upperDtype, cubeMathType);
    }
    UpdateOutputDtype(output, opInfo, cubeMathType, upperDtype, transposed);
    UpdateInputDtype(input, bias, opInfo, upperDtype, transposed);
}

} // namespace

namespace {

constexpr int STRIDEH_DMA = 63;
constexpr int DILATION_DMA = 255;
constexpr int PAD_DMA = 255;
constexpr int weight_DMA = 511;
static bool isNotDMAFromPad(bool isDMASpec, const aclIntArray* padding)
{
    if (padding->Size() == CONV_2D_PAD_DIM) {
        isDMASpec = isDMASpec || ((*padding)[0] > PAD_DMA) || ((*padding)[1] > PAD_DMA);
    } else if (padding->Size() == CONV_4D_PAD_DIM) {
        isDMASpec = isDMASpec || ((*padding)[0] > PAD_DMA) || ((*padding)[1] > PAD_DMA) ||
                    ((*padding)[PAD_LEFT_INDEX] > PAD_DMA) || ((*padding)[PAD_RIGHT_INDEX] > PAD_DMA);
    }
    return isDMASpec;
}
// padding = [pad_top, pad_bottom, pad_left, pad_right]
// 1. 不满足DMA的规格   2. load3d L1最小切分要在L1能够放下
static bool isNotDMA(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* output,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation)
{
    int64_t inputHeight = (int64_t)input->GetViewShape().GetDim(2);
    int64_t inputWidth = (int64_t)input->GetViewShape().GetDim(3);
    int64_t weightH = (int64_t)weight->GetViewShape().GetDim(2);
    int64_t weightW = (int64_t)weight->GetViewShape().GetDim(3);
    int64_t outputSize = (int64_t)output->GetViewShape().GetDimNum();
    int64_t outputW = (int64_t)output->GetViewShape().GetDim(2);
    if (outputSize == CONV_2D_DIM_SIZE) {
      outputW = static_cast<int64_t>(output->GetViewShape().GetDim(3)); // NCH(W) -> 012(3)
    }

    // CUBE_FP16的M, K
    constexpr int64_t BLK_M = 16;
    constexpr int64_t BLK_K = 16;
    constexpr int64_t BIT_L12BT_MIN_BIAS = 64;

    // 1. 不满足DMA的规格
    int64_t strideH = (*stride)[0];
    int64_t strideW = (*stride)[1];
    int64_t dilationH = (*dilation)[0];
    int64_t dilationW = (*dilation)[1];
    bool alignResult = ((weightH * weightW * 4 + BLK_K - 1) / BLK_K * BLK_K) <= 65535;
    OP_LOGD("alignResult is %d", alignResult);

    // stride wh <=63, dilation wh<=255, padding <=255, weight wh<=511, align(weightH * weightW * 4, BLK_K) <= 65535
    // 以上条件同时满足表示不满足DMA规格
    bool isDMASpec = (strideH > STRIDEH_DMA) || (strideW > STRIDEH_DMA) || (dilationH > DILATION_DMA) ||
                     (dilationW > DILATION_DMA) || (weightH > weight_DMA) || (weightW > weight_DMA) || !alignResult;
    isDMASpec = isNotDMAFromPad(isDMASpec, padding);
    if (isDMASpec) {
        OP_LOGD("Fulfill DMA requirement，return False");
        return false;
    }

    // 2. load3d L1最小切分要在L1能够放下
    int64_t hoNum = BLK_M / outputW + 2;
    int64_t hkDilation = (weightH - 1) * dilationH + 1;
    int64_t hiNum = std::min(((hoNum - 1) * strideH + hkDilation), inputHeight);
    int64_t wiL1 = (int64_t)input->GetViewShape().GetDim(3);
    int64_t hiL1 = hiNum;

    // input_height = 1 & weight_height = 1 & pad_top = 0 & pad bottom = 0
    bool isConv1d = (inputHeight == 1) && (weightH == 1) && ((*padding)[0] == 0) && ((*padding)[1] == 0);
    OP_LOGD("isConv1d is %d", isConv1d);
    if (isConv1d) {
        int64_t woNum = BLK_M;
        int64_t wkDilation = (weightW - 1) * dilationW + 1;
        wiL1 = std::min(((woNum - 1) * strideW + wkDilation), inputWidth);
    }

    // 非Conv1d时width <= 32767才能走C04
    constexpr int64_t WIDTH_THRESSHOLD = 32767;
    if (!isConv1d && inputWidth > WIDTH_THRESSHOLD) {
        OP_LOGD("when not conv1d scene, inputWidth[%ld] > 32767", inputWidth);
        return false;
    }
    int64_t hiwiMul = wiL1 * hiL1;
    constexpr int64_t c0OnL1 = 4;
    uint64_t maxL1Size = static_cast<uint64_t>(hiwiMul * c0OnL1 * 2); // dataTypeToByte(FP16)为2
    maxL1Size = (bias != nullptr) ? maxL1Size + BIT_L12BT_MIN_BIAS : maxL1Size;
    OP_LOGD("maxL1Size is %lu", maxL1Size);

    // hardwareInfo.l1size 910B 为524288
    return maxL1Size <= 524288U;
}

static bool CanSwitchC04InBF16Scene(const struct ConvolutionOpInfo& opInfo)
{
    if (opInfo.weightDtype == op::DataType::DT_BF16 &&
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B)) {
        return true;
    }
    return false;
}

static bool CanSwitchC04InF16Scene(const struct ConvolutionOpInfo& opInfo)
{
    // datatype为float16，且socversion为910B和910_93
    if (opInfo.weightDtype == op::DataType::DT_FLOAT16 && IsCubeSupportFp32()) {
        return true;
    }
    return false;
}

// 1. groups为1  2. Cin<=4  3. dtype为FP16 4. 必须是910B芯片 5. 非DMA场景   同时满足才能走c04
static bool CanSwitchC04(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* output,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, int64_t groups, bool transposed)
{
    // 必须为非transpose场景 + format为NCHW才行
    if (transposed || input->GetViewFormat() != Format::FORMAT_NCHW) {
        OP_LOGD("input is not NCHW or is transposed, thus no C04");
        return false;
    }

    int64_t cin = input->GetViewShape().GetDim(1);
    // groups数量必须为1， 并且C04场景必须Cin为4,非DMA可直接切c04
    if ((groups == 1) && (cin <= op::SMALL_CHANNEL && cin > 0)) {
        return isNotDMA(input, weight, bias, output, stride, padding, dilation);
    }

    OP_LOGD("Not fulfill the requirements for C04");
    return false;
}

// 非C04场景的更新 卷积format
void GetConvolutionOpFormat(
    struct ConvolutionOpInfo& opInfo, const aclTensor* input, const aclTensor* weight, const aclTensor* output)
{
    opInfo.weightFormat = Format::FORMAT_FRACTAL_Z;
    opInfo.inputFormat = Format::FORMAT_NC1HWC0;
    opInfo.outputFormat = Format::FORMAT_NC1HWC0;
    opInfo.biasFormat = Format::FORMAT_ND;
    if (op::IsSupportND()) {
        opInfo.weightFormat =
            weight->GetStorageFormat() == Format::FORMAT_NHWC ? Format::FORMAT_NHWC : Format::FORMAT_NCHW;
        opInfo.inputFormat =
            input->GetStorageFormat() == Format::FORMAT_NHWC ? Format::FORMAT_NHWC : Format::FORMAT_NCHW;
        opInfo.outputFormat =
            output->GetStorageFormat() == Format::FORMAT_NHWC ? Format::FORMAT_NHWC : Format::FORMAT_NCHW;
    }
}

void GetConvolution3dOpFormat(
    struct ConvolutionOpInfo& opInfo, const aclTensor* input, const aclTensor* weight, const aclTensor* output)
{
    opInfo.weightFormat = Format::FORMAT_FRACTAL_Z_3D;
    opInfo.inputFormat = Format::FORMAT_NDC1HWC0;
    opInfo.outputFormat = Format::FORMAT_NDC1HWC0;
    opInfo.biasFormat = Format::FORMAT_ND;
    if (op::IsSupportND()) {
        opInfo.weightFormat =
            weight->GetStorageFormat() == Format::FORMAT_NDHWC ? Format::FORMAT_NDHWC : Format::FORMAT_NCDHW;
        opInfo.inputFormat =
            input->GetStorageFormat() == Format::FORMAT_NDHWC ? Format::FORMAT_NDHWC : Format::FORMAT_NCDHW;
        opInfo.outputFormat =
            output->GetStorageFormat() == Format::FORMAT_NDHWC ? Format::FORMAT_NDHWC : Format::FORMAT_NCDHW;
    }
}

void GetConvolutionOpFormatC04(struct ConvolutionOpInfo& opInfo)
{
    opInfo.weightFormat = Format::FORMAT_FRACTAL_Z_C04;
    opInfo.inputFormat = Format::FORMAT_NC1HWC0;
    opInfo.outputFormat = Format::FORMAT_NC1HWC0;
    opInfo.biasFormat = Format::FORMAT_ND;
}

// 更新convolution所需要的dtype format
void GetConvOpInfo(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* output,
    struct ConvolutionOpInfo& opInfo, const bool transposed, int64_t groups, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int8_t cubeMathType)
{
    GetConvolutionOpDtype(input, weight, bias, output, opInfo, transposed, cubeMathType);
    // 支持C04 + NCHW + 非transposed的场景
    op::Shape inputSpecialShape = op::Shape({320, 3, 224, 224}); // 客户专用场景
    op::Shape weightSpecialShape = op::Shape({768, 3, 32, 32});  // 客户专用场景
    if ((weight->GetViewShape() == weightSpecialShape) && (input->GetViewShape() == inputSpecialShape) &&
        CanSwitchC04InF16Scene(opInfo) &&
        CanSwitchC04(input, weight, bias, output, stride, padding, dilation, groups, transposed)) {
        OP_LOGD("Entering float16 C04 branch");
        GetConvolutionOpFormatC04(opInfo);
    } else if (
        CanSwitchC04InBF16Scene(opInfo) &&
        CanSwitchC04(input, weight, bias, output, stride, padding, dilation, groups, transposed)) {
        OP_LOGD("Entering bfloat16 C04 branch");
        GetConvolutionOpFormatC04(opInfo);
    } else {
        Conv2DSplitWInfo conv2dInfo;
        conv2dInfo.InitConv2DSplitWInfo(input, weight, stride, padding, dilation);
        if (conv2dInfo.CanSwitchSplitW(bias, output, groups, opInfo)) {
            OP_LOGD("Entering splitW branch");
            GetConvolution3dOpFormat(opInfo, input, weight, output);
        } else {
            OP_LOGD("Entering normal C04 branch");
            GetConvolutionOpFormat(opInfo, input, weight, output);
        }
    }
}

void GetConv3dOpInfo(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* output,
    struct ConvolutionOpInfo& opInfo, const bool transposed, int8_t cubeMathType)
{
    GetConvolutionOpDtype(input, weight, bias, output, opInfo, transposed, cubeMathType);
    GetConvolution3dOpFormat(opInfo, input, weight, output);
}

static aclIntArray* ViewConv1dPad1dAs4d(const aclIntArray* intArray, aclOpExecutor* executor)
{
    constexpr uint64_t newDimSize = 4;
    int64_t data[newDimSize];
    data[0] = 0;
    data[1] = 0;
    data[PAD_LEFT_INDEX] = (*intArray)[0];
    data[PAD_RIGHT_INDEX] = (*intArray)[0];
    aclIntArray* newArray = executor->AllocIntArray(data, newDimSize);
    return newArray;
}

static aclIntArray* ViewConv1dPad2dAs4d(const aclIntArray* intArray, aclOpExecutor* executor)
{
    constexpr uint64_t newDimSize = 4;
    int64_t data[newDimSize];
    data[0] = 0;
    data[1] = 0;
    data[PAD_LEFT_INDEX] = (*intArray)[0];
    data[PAD_RIGHT_INDEX] = (*intArray)[1];
    aclIntArray* newArray = executor->AllocIntArray(data, newDimSize);
    return newArray;
}

static aclIntArray* ViewConv2dPad2dAs4d(const aclIntArray* intArray, aclOpExecutor* executor)
{
    constexpr uint64_t newDimSize = 4;
    int64_t data[newDimSize];
    data[0] = (*intArray)[0];
    data[1] = (*intArray)[0];
    data[PAD_LEFT_INDEX] = (*intArray)[1];
    data[PAD_RIGHT_INDEX] = (*intArray)[1];
    aclIntArray* newArray = executor->AllocIntArray(data, newDimSize);
    return newArray;
}

aclIntArray* View1dAs2d(const aclIntArray* intArray, int64_t expendValue, aclOpExecutor* executor)
{
    constexpr uint64_t newDimSize = 2;
    int64_t data[newDimSize];
    uint64_t size = intArray->Size();
    if (size != static_cast<uint64_t>(1)) {
        return nullptr;
    }
    data[0] = expendValue;
    data[1] = (*intArray)[0];
    aclIntArray* newArray = executor->AllocIntArray(data, newDimSize);
    return newArray;
}

aclIntArray* ViewValueAs1d(const int64_t value, aclOpExecutor* executor)
{
    int64_t data[1];
    data[0] = value;
    aclIntArray* newArray = executor->AllocIntArray(data, 1);
    return newArray;
}

const aclTensor* View1dAs4d(const aclTensor* input, aclOpExecutor* executor)
{
    // input NCL->contigious->unsqueeze(2)->reformat NCHW
    // 非连续转连续contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze(2)
    constexpr int64_t appendDim[] = {0, 2, 3};
    aclIntArray* dim = executor->AllocIntArray(appendDim, 3);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);

    // reformat
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* View3dAs4d(const aclTensor* input, aclOpExecutor* executor)
{
    // input NCL->contigious->unsqueeze(2)->reformat NCHW
    // 非连续转连续contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze(2)
    constexpr int64_t appendDim[] = {2};
    aclIntArray* dim = executor->AllocIntArray(appendDim, 1);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);

    // reformat
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* View4dAs3d(const aclTensor* input, aclOpExecutor* executor)
{
    // input NCL->contigious->unsqueeze(2)->reformat NCHW
    // 非连续转连续contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    // unsqeeze(2)
    constexpr int64_t appendDim[] = {2};
    aclIntArray* dim = executor->AllocIntArray(appendDim, 1);
    auto squeezedInput = l0op::SqueezeNd(contiguousInput, dim, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);

    // reformat
    auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCL);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* Permute(const aclTensor* input, FVector<int64_t> dims, aclOpExecutor* executor)
{
    // contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    // Transpose
    auto* perm = executor->AllocIntArray(dims.data(), dims.size());
    CHECK_RET(perm != nullptr, nullptr);

    auto* result = l0op::Transpose(contiguousInput, perm, executor);
    CHECK_RET(result != nullptr, nullptr);

    return result;
}

static inline const aclTensor* ViewWithShape(const aclTensor* tensor, const op::Shape& shape, aclOpExecutor* executor)
{
    if (shape == tensor->GetViewShape() && shape == tensor->GetStorageShape()) {
        return tensor;
    }
    return executor->CreateView(tensor, shape, tensor->GetViewOffset());
}

static aclnnStatus CheckConv2dWithWeightFZ(const aclTensor* input, const aclTensor* weight)
{
    if (weight->GetStorageFormat() != Format::FORMAT_FRACTAL_Z) {
        return ACLNN_SUCCESS;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Current weight format is Internal Format, only support soc 310P! Current soc is %s.",
            op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (input->GetDataType() != weight->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Current weight format is Internal Format, fmap should be same dtype with weight! "
            "input dtype: %s, weight dtype: %s.",
            op::ToString(input->GetDataType()).GetString(), op::ToString(weight->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CommonPostProcessForBmm(const aclTensor* output, const aclTensor*& convOut, aclOpExecutor* executor)
{
    // ND --> NCDHW
    auto viewOutput = ViewWithShape(convOut, output->GetViewShape(), executor);
    CHECK_RET(viewOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // output reformat
    auto formatOutput = l0op::ReFormat(viewOutput, output->GetViewFormat(), executor);
    CHECK_RET(formatOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ViewCopy(formatOutput, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static const aclTensor* ViewWithShapeAndReformatND(
    const aclTensor* tensor, const std::initializer_list<int64_t>& shape, aclOpExecutor* executor)
{
    op::Shape shapeBMN = op::Shape(shape);
    auto tensorBMN = ViewWithShape(tensor, shapeBMN, executor);
    CHECK_RET(tensorBMN != nullptr, nullptr);
    return l0op::ReFormat(tensorBMN, op::Format::FORMAT_ND);
}

static int64_t CalcCountByAxisVec(const op::Shape& dataShape, const vector<int64_t>& axisVec)
{
    int64_t count = 1;
    for (auto axis : axisVec) {
        count *= dataShape[axis];
    }
    return count;
}

static aclnnStatus GetAndCastConvolutionOpDtype(ConvEngine& engine, aclOpExecutor* executor)
{
    ConvolutionOpInfo opInfo = {};
    GetConvolutionOpDtype(
        engine.params.input, engine.params.weight, engine.params.bias, engine.params.output, opInfo,
        engine.params.transposed, engine.params.cubeMathType);
    return CommonPreProcess(
        engine.params.input, engine.params.weight, engine.params.bias, engine.params.groups, engine.params.transposed,
        opInfo, false, true, executor);
}

static aclnnStatus GenInOutByConvToBmm(
    ConvEngine engine, const ConvToBmmMode& convToBmmMode, BatchMatmulInput& bmmInput, aclOpExecutor* executor)
{
    auto ret = GetAndCastConvolutionOpDtype(engine, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    if (convToBmmMode == ConvToBmmMode::CONV_MM_FEATURE_MAP_EQ_FILTER) {
        const vector<int64_t> cidhwIdxUnionVec{
            CI_DIM_CO_CI_DHW_INDEX, D_DIM_NCDHW_INDEX, H_DIM_NCDHW_INDEX, W_DIM_NCDHW_INDEX};
        const vector<int64_t> cihwIdxUnionVec{CI_DIM_CO_CI_DHW_INDEX, H_DIM_NCHW_INDEX, W_DIM_NCHW_INDEX};
        const auto& dimIdxUnionVec =
            (engine.meta.input.format == op::Format::FORMAT_NCHW) ? cihwIdxUnionVec : cidhwIdxUnionVec;
        // weight --> [1, Co, CiDHW]
        std::initializer_list<int64_t> weightShapeVec = {
            1, engine.meta.weight.N(), CalcCountByAxisVec(engine.meta.weight.tensorShape, dimIdxUnionVec)};
        auto weightND = ViewWithShapeAndReformatND(engine.params.weight, weightShapeVec, executor);
        CHECK_RET(weightND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // input --> [1, N, CiDHW]
        std::initializer_list<int64_t> inputShapeVec = {
            1, engine.meta.input.N(), CalcCountByAxisVec(engine.meta.input.tensorShape, dimIdxUnionVec)};
        auto inputND = ViewWithShapeAndReformatND(engine.params.input, inputShapeVec, executor);
        CHECK_RET(inputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto biasND = engine.params.bias;
        if (biasND != nullptr && engine.meta.bias.format != op::Format::FORMAT_ND) {
            biasND = l0op::ReFormat(engine.params.bias, op::Format::FORMAT_ND);
            CHECK_RET(biasND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // output --> [1, N, Co]
        std::initializer_list<int64_t> outputShapeVec = {1, engine.meta.output.N(), engine.meta.output.C()};
        auto outputND = ViewWithShapeAndReformatND(engine.params.output, outputShapeVec, executor);
        CHECK_RET(outputND != nullptr, ACLNN_ERR_INNER_NULLPTR);

        bmmInput.leftData = inputND;
        bmmInput.isLeftTranspose = false;
        bmmInput.rightData = weightND;
        bmmInput.isRightTranspose = true;
        bmmInput.biasData = biasND;
        bmmInput.outputData = outputND;
    }
    OP_LOGD(
        "convolution to batchmatmul op leftDataType: %s, rightDataType: %s.",
        op::ToString(bmmInput.leftData->GetDataType()).GetString(),
        op::ToString(bmmInput.rightData->GetDataType()).GetString());

    return ACLNN_SUCCESS;
}

static aclnnStatus GenInOutByConvTranspose1DToBmm(
    ConvEngine engine, const ConvTranspose1DToBmmMode& convTranspose1DToBmmMode, BatchMatmulInput& bmmInput,
    aclOpExecutor* executor)
{
    auto ret = GetAndCastConvolutionOpDtype(engine, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    if (convTranspose1DToBmmMode == ConvTranspose1DToBmmMode::CONVTRANSPOSE1D_MM_FEATURE_MAP_EQ_FILTER) {
        // The format of 1d is NCL, get n, cin, cout, l
        auto n = engine.meta.input.shape[N_DIM_NCL_INDEX];
        auto inChannels = engine.meta.input.shape[C_DIM_NCL_INDEX];
        auto outChannels = engine.meta.output.shape[C_DIM_NCL_INDEX];
        auto l = engine.meta.weight.shape[L_DIM_NCL_INDEX];
        // input shape [n, cin, 1] reshape to [n, cin]
        auto inputND = ViewWithShapeAndReformatND(engine.params.input, {n, inChannels}, executor);
        CHECK_RET(inputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // weight shape [cin, cout, l] reshape to [cin, cout * l]
        auto weightND = ViewWithShapeAndReformatND(engine.params.weight, {inChannels, outChannels * l}, executor);
        CHECK_RET(weightND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto biasND = engine.params.bias;
        if (biasND != nullptr && engine.meta.bias.format != op::Format::FORMAT_ND) {
            biasND = l0op::ReFormat(engine.params.bias, op::Format::FORMAT_ND);
            CHECK_RET(biasND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // output shape [n, cout, l] reshape to [n, cout * l]
        auto outputND = ViewWithShapeAndReformatND(engine.params.output, {n, outChannels * l}, executor);
        CHECK_RET(outputND != nullptr, ACLNN_ERR_INNER_NULLPTR);

        bmmInput.leftData = inputND;
        bmmInput.isLeftTranspose = false;
        bmmInput.rightData = weightND;
        bmmInput.isRightTranspose = false;
        bmmInput.biasData = biasND;
        bmmInput.outputData = outputND;
    }
    return ACLNN_SUCCESS;
}

static bool IsSupportConvTranspose1DToBmm(ConvEngine engine)
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910_95) {
        return false;
    }
    if (engine.meta.input.format != op::Format::FORMAT_NCL) {
        return false;
    }
    if (IsSupportND() && (engine.meta.input.dataType == op::DataType::DT_HIFLOAT8 ||
                          engine.meta.input.dataType == op::DataType::DT_FLOAT8_E4M3FN)) {
        return false;
    }
    if (!engine.params.transposed || engine.params.groups != 1) {
        return false;
    }
    if (engine.meta.padding[0] != 0 || engine.meta.outputPadding[0] != 0 || engine.meta.dilation[0] != 1) {
        return false;
    }
    return true;
}

static ConvTranspose1DToBmmMode GetConvTranspose1DToBmmMode(ConvEngine engine)
{
    if (!IsSupportConvTranspose1DToBmm(engine)) {
        return ConvTranspose1DToBmmMode::CONVTRANSPOSE1D_NO_MM;
    }
    if (engine.meta.input.shape[L_DIM_NCL_INDEX] == 1 &&
        engine.meta.output.shape[L_DIM_NCL_INDEX] == engine.meta.weight.shape[L_DIM_NCL_INDEX]) {
        return ConvTranspose1DToBmmMode::CONVTRANSPOSE1D_MM_FEATURE_MAP_EQ_FILTER;
    }
    return ConvTranspose1DToBmmMode::CONVTRANSPOSE1D_NO_MM;
}

static bool IsSupportConvToBmm(ConvEngine engine)
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93 &&
        socVersion != SocVersion::ASCEND910_95) {
        return false;
    }

    if ((socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) &&
        engine.meta.input.format != op::Format::FORMAT_NCDHW) {
        return false;
    }

    if (op::IsSupportND()) {
        if (engine.meta.input.format != op::Format::FORMAT_NCDHW &&
            engine.meta.input.format != op::Format::FORMAT_NCHW) {
            return false;
        }
        if (engine.meta.input.dataType == op::DataType::DT_HIFLOAT8) {
            return false;
        }
    }
    // padding
    for (uint64_t paddingIdx = 0; paddingIdx < engine.meta.padding.size(); ++paddingIdx) {
        if (engine.meta.padding[paddingIdx] != 0) {
            return false;
        }
    }
    // dilation
    for (uint64_t dilationIdx = 0; dilationIdx < engine.meta.dilation.size(); ++dilationIdx) {
        if (engine.meta.dilation[dilationIdx] != 1) {
            return false;
        }
    }
    // other attribute
    if (engine.params.transposed || engine.params.groups != 1) {
        return false;
    }
    return true;
}

static ConvToBmmMode GetConvToBmmMode(ConvEngine engine)
{
    if (!IsSupportConvToBmm(engine)) {
        return ConvToBmmMode::CONV_NO_MM;
    }

    bool isFmapEqFilter = true;
    const std::vector<int64_t> dimIdxVecNcdhw{D_DIM_NCDHW_INDEX, H_DIM_NCDHW_INDEX, W_DIM_NCDHW_INDEX};
    const std::vector<int64_t> dimIdxVecNchw{H_DIM_NCHW_INDEX, W_DIM_NCHW_INDEX};
    const auto& dimIdxVec = (engine.meta.input.format == op::Format::FORMAT_NCHW) ? dimIdxVecNchw : dimIdxVecNcdhw;
    for (int64_t dimIdx : dimIdxVec) {
        if (engine.meta.input.shape[dimIdx] != engine.meta.weight.shape[dimIdx]) {
            isFmapEqFilter = false;
            break;
        }
    }
    if (isFmapEqFilter) {
        return ConvToBmmMode::CONV_MM_FEATURE_MAP_EQ_FILTER;
    }
    return ConvToBmmMode::CONV_NO_MM;
}

} // namespace

namespace AclnnConvolution {

static inline void RegisterConv2dL0Functions(std::map<std::string, L0FUNCTION>& l0Functions)
{
    REG_L0_FUNCTION(
        l0Functions, Conv2d5HdFp16, op::DataType::DT_FLOAT16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT16,
        op::Format::FORMAT_NC1HWC0);
    REG_L0_FUNCTION(
        l0Functions, Conv2d5HdFp32, op::DataType::DT_FLOAT, op::Format::FORMAT_NC1HWC0, op::DataType::DT_FLOAT,
        op::Format::FORMAT_NC1HWC0);
    REG_L0_FUNCTION(
        l0Functions, Conv2d5HdFp1625HdFp32, op::DataType::DT_FLOAT16, op::Format::FORMAT_NC1HWC0,
        op::DataType::DT_FLOAT, op::Format::FORMAT_NC1HWC0);
    REG_L0_FUNCTION(
        l0Functions, Conv2d5HdBf16, op::DataType::DT_BF16, op::Format::FORMAT_NC1HWC0, op::DataType::DT_BF16,
        op::Format::FORMAT_NC1HWC0);
    REG_L0_FUNCTION_BY_OPTYPE(l0Functions, Conv2dV2NCHW, "Conv2DV2");
}

static inline aclnnStatus CommonConvImpl(
    std::map<std::string, L0FUNCTION>& l0Functions, ConvolutionOpInfo& opInfo, const aclTensor* input,
    const aclTensor* weight, const aclTensor* bias, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool transposed, const aclIntArray* outputPadding, int64_t groups, bool useHf32,
    aclOpExecutor* executor, const aclTensor*& convOut, const char* errorMsg)
{
    if (op::IsSupportND()) {
        convOut = FUNCTION_CALL_BY_OPTYPE(
            l0Functions, "Conv2DV2", input, weight, bias, opInfo.outputDtype, stride, padding, dilation, transposed,
            outputPadding, groups, useHf32, executor);
    } else {
        convOut = FUNCTION_CALL(
            l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
            useHf32, executor);
    }

    if (convOut == nullptr) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "%s", errorMsg);
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

class ConvolutionImpl {
public:
    virtual aclnnStatus PreProcess() = 0;
    virtual aclnnStatus Impl() = 0;
    virtual aclnnStatus PostProcess() = 0;
    ConvolutionImpl(
        const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,
        const aclIntArray* strideParam, const aclIntArray* paddingParam, const aclIntArray* dilationParam,
        const bool transposedParam, const aclIntArray* outputPaddingParam, const int64_t groupsParam,
        aclTensor* outputParam, bool useHf32Param, int8_t cubeMathTypeParam, aclOpExecutor* executorParam)
        : input(inputParam),
          weight(weightParam),
          bias(biasParam),
          stride(strideParam),
          padding(paddingParam),
          dilation(dilationParam),
          transposed(transposedParam),
          outputPadding(outputPaddingParam),
          groups(groupsParam),
          output(outputParam),
          useHf32(useHf32Param),
          cubeMathType(cubeMathTypeParam),
          executor(executorParam) {};
    virtual ~ConvolutionImpl()
    {
        input = nullptr;
        weight = nullptr;
        bias = nullptr;
        stride = nullptr;
        padding = nullptr;
        dilation = nullptr;
        outputPadding = nullptr;
        output = nullptr;
        executor = nullptr;
    };

protected:
    const aclTensor* input;
    const aclTensor* weight;
    const aclTensor* bias;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    const bool transposed;
    const aclIntArray* outputPadding;
    const int64_t groups;
    aclTensor* output;
    const bool useHf32;
    int8_t cubeMathType;
    uint64_t* workspaceSize = nullptr;
    aclOpExecutor* executor;
    const aclTensor* convOut = nullptr;
    ConvolutionOpInfo opInfo; // 用于提前计算所有前后处理相关的format、dtype等信息
    std::map<std::string, L0FUNCTION> l0Functions;
};

static inline aclnnStatus ExecuteConvImpl(const std::shared_ptr<AclnnConvolution::ConvolutionImpl>& convImpl)
{
    if (convImpl == nullptr) {
        return ACLNN_ERR_INNER;
    }
    aclnnStatus ret = convImpl->PreProcess();
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = convImpl->Impl();
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = convImpl->PostProcess();
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    return ACLNN_SUCCESS;
}

static inline void RegisterTransposedConvL0Functions(
    std::map<std::string, L0FUNCTION>& l0Functions, op::Format dataFormat)
{
    REG_L0_FUNCTION(
        l0Functions, ConvTranspose2d5HdFp16, op::DataType::DT_FLOAT16, dataFormat, op::DataType::DT_FLOAT16,
        dataFormat);
    REG_L0_FUNCTION(
        l0Functions, ConvTranspose2d5HdFp32, op::DataType::DT_FLOAT, dataFormat, op::DataType::DT_FLOAT, dataFormat);
    REG_L0_FUNCTION(
        l0Functions, ConvTranspose2d5HdBf16, op::DataType::DT_BF16, dataFormat, op::DataType::DT_BF16, dataFormat);
    REG_L0_FUNCTION(
        l0Functions, ConvTranspose2d5HdHif8, op::DataType::DT_HIFLOAT8, dataFormat, op::DataType::DT_HIFLOAT8,
        dataFormat);
    REG_L0_FUNCTION(
        l0Functions, ConvTranspose2d5HdF8e4m3fn, op::DataType::DT_FLOAT8_E4M3FN, dataFormat,
        op::DataType::DT_FLOAT8_E4M3FN, dataFormat);
}

#define CONV_CONSTRUCTOR(type)                                                                               \
    CONCAT(Conv##type, Impl)(                                                                                \
        const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,               \
        const aclIntArray* strideParam, const aclIntArray* paddingParam, const aclIntArray* dilationParam,   \
        const bool transposedParam, const aclIntArray* outputPaddingParam, const int64_t groupsParam,        \
        aclTensor* outputParam, bool useHf32Param, int8_t cubeMathTypeParam, aclOpExecutor* executorParam)   \
        : ConvolutionImpl(                                                                                   \
              inputParam, weightParam, biasParam, strideParam, paddingParam, dilationParam, transposedParam, \
              outputPaddingParam, groupsParam, outputParam, useHf32Param, cubeMathTypeParam, executorParam)  \
    {}

#define CONCAT(a, b) a##b

class Conv2dImpl : public ConvolutionImpl {
public:
    CONV_CONSTRUCTOR(2d)

    aclnnStatus PreProcess() override
    {
        RegisterConv2dL0Functions(l0Functions);
        if (padding->Size() != CONV_2D_PAD_DIM && padding->Size() != CONV_4D_PAD_DIM) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv2d pad size is not in 2 or 4");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        if (padding->Size() == CONV_2D_PAD_DIM) {
            padding = ViewConv2dPad2dAs4d(padding, executor);
            CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        auto ret = CheckConv2dWithWeightFZ(input, weight);
        CHECK_RET(ret == ACL_SUCCESS, ret);
        CHECK_RET(!CheckUnSupportDtype(input, weight), ACLNN_ERR_INNER_NULLPTR);
        GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
        OP_LOGD(
            "convolution aclnn op inputDtype: %s, outputDtype: %s, biasDtype: %s, useHf32: %d.",
            op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
            op::ToString(opInfo.biasDtype).GetString(), useHf32);
        // 需要切C04分支卷积
        if (opInfo.weightFormat == Format::FORMAT_FRACTAL_Z_C04 && weight->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_LOGD("Conv2d entering float16 C04 branch");
            return CommonPreProcessC04(input, weight, bias, groups, transposed, opInfo, true, true, executor);
        }
        if (opInfo.weightFormat == Format::FORMAT_FRACTAL_Z_C04 && weight->GetDataType() == op::DataType::DT_BF16) {
            OP_LOGD("Conv2d entering bfloat16 C04 branch");
        } else if (opInfo.inputFormat == Format::FORMAT_NDC1HWC0) {
            OP_LOGD("Conv2d entering splitW branch");
            auto changeRes = ChangeConv2dAttrToConv3d(stride, padding, dilation, executor);
            CHECK_RET(changeRes == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
            changeRes = ChangeConv2dInputToConv3d(input, weight, executor);
            CHECK_RET(changeRes == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
            REG_L0_FUNCTION(
                l0Functions, Conv3dv26HdFp16, op::DataType::DT_FLOAT16, op::Format::FORMAT_NDC1HWC0,
                op::DataType::DT_FLOAT16, op::Format::FORMAT_NDC1HWC0);
            REG_L0_FUNCTION(
                l0Functions, Conv3dv26HdBf16, op::DataType::DT_BF16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_BF16,
                op::Format::FORMAT_NDC1HWC0);
            REG_L0_FUNCTION(
                l0Functions, Conv3dv26HdFp32, op::DataType::DT_FLOAT, op::Format::FORMAT_NDC1HWC0,
                op::DataType::DT_FLOAT, op::Format::FORMAT_NDC1HWC0);
            OP_LOGD(
                "convolution aclnn op inputDtype: %s, outputDtype: %s, biasDtype: %s, useHf32: %d.",
                op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
                op::ToString(opInfo.biasDtype).GetString(), useHf32);
        } else {
            OP_LOGD("Conv2d entering normal branch");
        }
        // 调用静态函数PreProcess
        bool needChangeFormat = !op::IsSupportND();
        return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, needChangeFormat, true, executor);
    };

    aclnnStatus Impl() override
    {
        if (op::IsSupportND()) {
            convOut = FUNCTION_CALL_BY_OPTYPE(
                l0Functions, "Conv2DV2", input, weight, bias, opInfo.outputDtype, stride, padding, dilation, transposed,
                outputPadding, groups, useHf32, executor);
        } else {
            convOut = FUNCTION_CALL(
                l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
                useHf32, executor);
        }
        if (convOut == nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv2d raise an unknown error");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override
    {
        if (opInfo.inputFormat != Format::FORMAT_NDC1HWC0) {
            bool needChangeOutFormat = !op::IsSupportND();
            auto res = CommonPostProcess(groups, needChangeOutFormat, output, convOut, executor);
            CHECK_RET(res == ACLNN_SUCCESS, res);
        } else {
            // splitw模式，会使得conv2d转为conv3d做，所以后处理先按照conv3d的处理方式输出
            auto fakeOutput3d =
                executor->AllocTensor(output->GetDataType(), op::Format::FORMAT_NCDHW, op::Format::FORMAT_NCDHW);
            CHECK_RET(fakeOutput3d != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto res = CommonPostProcess(groups, true, fakeOutput3d, convOut, executor);
            CHECK_RET(res == ACLNN_SUCCESS, res);
            convOut = View5dAs4dForOutput(convOut, executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };
    ~Conv2dImpl() override = default;
    ;
};

class ConvTbcImpl : public ConvolutionImpl {
public:
    CONV_CONSTRUCTOR(Tbc)

    aclnnStatus PreProcess() override
    {
        RegisterConv2dL0Functions(l0Functions);

        // conv1d is implemented by 2d, so first change view of input, weight, bias
        stride = View1dAs2d(stride, 1, executor);
        CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

        padding = View1dAs2d(padding, 0, executor);
        CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

        dilation = View1dAs2d(dilation, 1, executor);
        CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

        input = View3dAs4d(input, executor);
        CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);

        weight = View3dAs4d(weight, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

        bias = View1dAs4d(bias, executor);
        CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(!CheckUnSupportDtype(input, weight), ACLNN_ERR_INNER_NULLPTR);
        GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
        OP_LOGD(
            "convolution aclnn op inputDtype: %s, outputDtype: %s, biasDtype: %s, useHf32: %d.",
            op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
            op::ToString(opInfo.biasDtype).GetString(), useHf32);
        // 调用静态函数PreProcess
        return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, true, false, executor);
    };

    aclnnStatus Impl() override
    {
        // conv1d is implement by conv2d
        return CommonConvImpl(
            l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
            useHf32, executor, convOut, "convTbc raise an unknown error");
    };

    aclnnStatus PostProcess() override
    {
        // 因仅支持NCL格式的conv1d，所以转为conv2d的format默认为HCHW
        auto fakeOutput2d =
            executor->AllocTensor(output->GetDataType(), op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);

        // 调用静态函数PostProcess
        auto res = CommonPostProcess(groups, true, fakeOutput2d, convOut, executor);
        CHECK_RET(res == ACLNN_SUCCESS, res);
        // 现在Conv1d转为conv2d来做，所以需要转换输出
        convOut = View4dAs3d(convOut, executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        // permute to [T, B, C]
        FVector<int64_t> permuteDim{2, 0, 1};
        auto permuteConvTbc = Permute(convOut, permuteDim, executor);
        CHECK_RET(permuteConvTbc != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        // view copy
        auto castConvTbc = l0op::ReFormat(permuteConvTbc, Format::FORMAT_ND);
        auto result = l0op::ViewCopy(castConvTbc, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };
    ~ConvTbcImpl() override = default;
};

class Conv1dImpl : public ConvolutionImpl {
public:
    CONV_CONSTRUCTOR(1d)

    aclnnStatus PreProcess() override
    {
        RegisterConv2dL0Functions(l0Functions);

        // conv1d is implemented by 2d, so first change view of input, weight, bias
        stride = View1dAs2d(stride, 1, executor);
        CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (padding->Size() == 1) {
            padding = ViewConv1dPad1dAs4d(padding, executor);
            CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else if (padding->Size() == CONV_2D_PAD_DIM) {
            padding = ViewConv1dPad2dAs4d(padding, executor);
            CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv1d pad dim not equal to 1 or 2");
            return ACLNN_ERR_INNER_NULLPTR;
        }

        dilation = View1dAs2d(dilation, 1, executor);
        CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

        input = View3dAs4d(input, executor);
        CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);

        weight = View3dAs4d(weight, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (bias != nullptr) {
            if (bias->GetViewShape().GetDimNum() == 3) { // 输入维度为3
                bias = View3dAs4d(bias, executor);
            } else {
                // bias dim = 1, 其他dim在check时候返回
                bias = View1dAs4d(bias, executor);
            }
            CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(!CheckUnSupportDtype(input, weight), ACLNN_ERR_INNER_NULLPTR);
        GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
        OP_LOGD(
            "convolution aclnn op inputDtype: %s, outputDtype: %s, biasDtype: %s, useHf32: %d.",
            op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
            op::ToString(opInfo.biasDtype).GetString(), useHf32);
        specialConv1d = isSpecialConv1d(input, weight, stride, padding, dilation) && (groups == 1);
        // 调用静态函数PreProcess
        bool needChangeFormat = op::IsSupportND() ? false : !specialConv1d;
        return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, needChangeFormat, true, executor);
    };

    aclnnStatus Impl() override
    {
        if (specialConv1d) {
            // assert x and weight format = NCHW, C=H=1
            // x view to shape n*w/s s, the batch dim is fold to the dim 1,
            op::Shape inputShape2d =
                op::Shape({input->GetViewShape()[0] * input->GetViewShape()[3] / (*stride)[1], (*stride)[1]});
            auto input2d = ViewWithShape(input, inputShape2d, executor);
            CHECK_RET(input2d != nullptr, ACLNN_ERR_INNER_NULLPTR);

            // weight reshape to shape Cout s
            op::Shape weightShape2d = op::Shape({weight->GetViewShape()[0], (*stride)[1]});
            auto weight2d = ViewWithShape(weight, weightShape2d, executor);
            CHECK_RET(weight2d != nullptr, ACLNN_ERR_INNER_NULLPTR);

            // weight perpute to shape s Cout
            FVector<int64_t> dims{1, 0};
            auto permutWeight = Permute(weight2d, dims, executor);
            CHECK_RET(permutWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto input2dND = l0op::ReFormat(input2d, op::Format::FORMAT_ND);
            auto permutWeightND = l0op::ReFormat(permutWeight, op::Format::FORMAT_ND);
            // matmul (x,weight) to shape n*w/s Cout
            auto mmOut = ExecMmOp(input2dND, permutWeightND, 0, executor);
            CHECK_RET(mmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            // matmul output reshape to shape n w/s Cout
            op::Shape mmOut3dShape = op::Shape(
                {input->GetViewShape()[0], input->GetViewShape()[3] / (*stride)[1], weight->GetViewShape()[0]});
            auto mmOut3d = ViewWithShape(mmOut, mmOut3dShape, executor);
            CHECK_RET(mmOut3d != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto mmOut3dNCL = l0op::ReFormat(mmOut3d, op::Format::FORMAT_NCL);

            // matmul output contiguous
            auto contiguousMmOut3d = l0op::Contiguous(mmOut3dNCL, executor);
            CHECK_RET(contiguousMmOut3d != nullptr, ACLNN_ERR_INNER_NULLPTR);

            // matmul output permut to shape n Cout w/s
            dims = {0, 2, 1};
            auto permutMmOut3d = Permute(contiguousMmOut3d, dims, executor);
            CHECK_RET(permutMmOut3d != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto output3dNCL = l0op::ReFormat(permutMmOut3d, op::Format::FORMAT_NCL);
            convOut = output3dNCL;

            return ACLNN_SUCCESS;
        }
        // conv1d is implement by conv2d
        return CommonConvImpl(
            l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
            useHf32, executor, convOut, "conv1d raise an unknown error");
    };

    aclnnStatus PostProcess() override
    {
        // conv1d 转换为conv2d做，所以后处理先按照conv2d的处理方式处理输出
        // 因仅支持NCL格式的conv1d，所以转为conv2d的format默认为HCHW
        auto fakeOutput2d =
            executor->AllocTensor(output->GetDataType(), op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);

        // 调用静态函数PostProcess
        auto res = CommonPostProcess(groups, !specialConv1d, fakeOutput2d, convOut, executor);
        CHECK_RET(res == ACLNN_SUCCESS, res);
        // 现在Conv1d转为conv2d来做，所以需要转换输出
        if (!specialConv1d) {
            convOut = View4dAs3d(convOut, executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        }

        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };

    ~Conv1dImpl() override = default;

private:
    bool isSpecialConv1d(
        const aclTensor* inputParam, const aclTensor* weightParam, const aclIntArray* strideParam,
        const aclIntArray* paddingParam, const aclIntArray* dilationParam) const
    {
        if ((*strideParam)[1] > op::specialStride &&
            (*strideParam)[1] == weightParam->GetViewShape()[op::specialChannelIndex] &&
            (*paddingParam)[PAD_LEFT_INDEX] == 0 && (*paddingParam)[PAD_RIGHT_INDEX] == 0 && (*dilationParam)[1] == 1 &&
            inputParam->GetViewShape()[1] == 1) {
            return true;
        } else {
            return false;
        }
    }
    bool specialConv1d = false;
};

class Conv3dImpl : public ConvolutionImpl {
public:
    CONV_CONSTRUCTOR(3d)

    aclnnStatus PreProcess() override
    {
        REG_L0_FUNCTION(
            l0Functions, Conv3d6HdFp16, op::DataType::DT_FLOAT16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT16,
            op::Format::FORMAT_NDC1HWC0);
        REG_L0_FUNCTION(
            l0Functions, Conv3dv26HdFp32, op::DataType::DT_FLOAT, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_FLOAT,
            op::Format::FORMAT_NDC1HWC0);
        REG_L0_FUNCTION(
            l0Functions, Conv3dv26HdBf16, op::DataType::DT_BF16, op::Format::FORMAT_NDC1HWC0, op::DataType::DT_BF16,
            op::Format::FORMAT_NDC1HWC0);
        REG_L0_FUNCTION(
            l0Functions, Conv3dv2NCDHWFp16, op::DataType::DT_FLOAT16, op::Format::FORMAT_NCDHW,
            op::DataType::DT_FLOAT16, op::Format::FORMAT_NCDHW);
        REG_L0_FUNCTION(
            l0Functions, Conv3dv2NCDHWFp32, op::DataType::DT_FLOAT, op::Format::FORMAT_NCDHW, op::DataType::DT_FLOAT,
            op::Format::FORMAT_NCDHW);
        REG_L0_FUNCTION(
            l0Functions, Conv3dv2NCDHWBf16, op::DataType::DT_BF16, op::Format::FORMAT_NCDHW, op::DataType::DT_BF16,
            op::Format::FORMAT_NCDHW);
        REG_L0_FUNCTION(
            l0Functions, Conv3dv2NCDHWHif8, op::DataType::DT_HIFLOAT8, op::Format::FORMAT_NCDHW,
            op::DataType::DT_HIFLOAT8, op::Format::FORMAT_NCDHW);
        CHECK_RET(!CheckUnSupportDtype(input, weight), ACLNN_ERR_INNER_NULLPTR);
        GetConv3dOpInfo(input, weight, bias, output, opInfo, transposed, cubeMathType);

        // 判断是否是PointWise卷积
        isPointWiseKernelFlag = !op::IsSupportND() && IsSupportConv3DToConv3DV2() &&
                                NeedPointWiseKernel(weight, stride, padding, dilation, groups) &&
                                !PointWiseKernelBeyondLimits(input);
        // PointWise卷积，biasDtype只能为FLOAT32
        if (isPointWiseKernelFlag) {
            opInfo.biasDtype = op::DataType::DT_FLOAT;
            opInfo.weightFormat = Format::FORMAT_NCDHW;
            opInfo.inputFormat = Format::FORMAT_NCDHW;
            opInfo.outputFormat = Format::FORMAT_NCDHW;
            OP_LOGD("Entering PointWise branch.");
        }
        OP_LOGD(
            "convolution aclnn op inputDtype: %s, outputDtype: %s, biasDtype: %s, useHf32: %d.",
            op::ToString(opInfo.inputDtype).GetString(), op::ToString(opInfo.outputDtype).GetString(),
            op::ToString(opInfo.biasDtype).GetString(), useHf32);
        // 调用静态函数PreProcess
        bool needChangeFormat = op::IsSupportND() ? false : !isPointWiseKernelFlag;
        return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, needChangeFormat, true, executor);
    };

    aclnnStatus Impl() override
    {
        convOut = FUNCTION_CALL(
            l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
            useHf32, executor);
        if (convOut == nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv3d raise an unknown error");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override
    {
        auto res = CommonPostProcess(groups, !isPointWiseKernelFlag, output, convOut, executor);
        CHECK_RET(res == ACLNN_SUCCESS, res);

        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };
    ~Conv3dImpl() override = default;

private:
    bool isPointWiseKernelFlag = false;
};
class ConvTransposed1dImpl : public ConvolutionImpl {
public:
    CONV_CONSTRUCTOR(Transposed1d)

    aclnnStatus PreProcess() override
    {
        op::Format dataFormat = op::IsSupportND() ? op::Format::FORMAT_NCHW : op::Format::FORMAT_NC1HWC0;
        RegisterTransposedConvL0Functions(l0Functions, dataFormat);

        stride = View1dAs2d(stride, 1, executor);
        CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

        padding = View1dAs2d(padding, 0, executor);
        CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

        dilation = View1dAs2d(dilation, 1, executor);
        CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

        outputPadding = View1dAs2d(outputPadding, 0, executor);
        CHECK_RET(outputPadding != nullptr, ACLNN_ERR_INNER_NULLPTR);

        input = View3dAs4d(input, executor);
        CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);

        weight = View3dAs4d(weight, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

        GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
        // 调用静态函数PreProcess
        if (bias != nullptr &&
            (opInfo.outputDtype == op::DataType::DT_HIFLOAT8 || opInfo.outputDtype == op::DataType::DT_FLOAT8_E4M3FN)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "ConvTranspose1d does not support DataType HIFLOAT8 or FLOAT8_E4M3FN when bias is present.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        bool needChangeFormat = op::IsSupportND() ? false : true;
        return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, needChangeFormat, false, executor);
    };

    aclnnStatus Impl() override
    {
        // conv1d is implement by conv2d
        convOut = FUNCTION_CALL(
            l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
            useHf32, executor);
        if (convOut == nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "conv1d raise an unknown error");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override
    {
        // 按照conv2d 的方式处理bias
        if (bias != nullptr && op::IsSupportND()) {
            int64_t biasLength = bias->GetViewShape().GetDim(0);
            bias = l0op::Reshape(bias, {1, biasLength, 1, 1}, executor);
            CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
            convOut = l0op::Add(convOut, bias, executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // conv1d 转换为conv2d做，所以后处理先按照conv2d的处理方式处理输出
        // 因仅支持NCL格式的conv1d，所以转为conv2d的format默认为NCHW
        auto fakeOutput2d =
            executor->AllocTensor(output->GetDataType(), op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);
        // 调用静态函数PostProcess
        bool needChangeFormat = op::IsSupportND() ? false : true;
        auto res = CommonPostProcess(groups, needChangeFormat, fakeOutput2d, convOut, executor);
        CHECK_RET(res == ACLNN_SUCCESS, res);
        // 现在Conv1d转为conv2d来做，所以需要转换输出
        convOut = View4dAs3d(convOut, executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };
    ~ConvTransposed1dImpl() override = default;
};
class ConvTransposed2dImpl : public ConvolutionImpl {
public:
    CONV_CONSTRUCTOR(Transposed2d)

    aclnnStatus PreProcess() override
    {
        op::Format dataFormat = op::IsSupportND() ? op::Format::FORMAT_NCHW : op::Format::FORMAT_NC1HWC0;
        RegisterTransposedConvL0Functions(l0Functions, dataFormat);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose2d5HdFp16, op::DataType::DT_FLOAT16, op::Format::FORMAT_NHWC,
            op::DataType::DT_FLOAT16, op::Format::FORMAT_NHWC);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose2d5HdFp32, op::DataType::DT_FLOAT, op::Format::FORMAT_NHWC,
            op::DataType::DT_FLOAT, op::Format::FORMAT_NHWC);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose2d5HdBf16, op::DataType::DT_BF16, op::Format::FORMAT_NHWC, op::DataType::DT_BF16,
            op::Format::FORMAT_NHWC);

        GetConvOpInfo(input, weight, bias, output, opInfo, transposed, groups, stride, padding, dilation, cubeMathType);
        // 调用静态函数PreProcess
        if (bias != nullptr &&
            (opInfo.outputDtype == op::DataType::DT_HIFLOAT8 || opInfo.outputDtype == op::DataType::DT_FLOAT8_E4M3FN)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "ConvTranspose2d does not support DataType HIFLOAT8 or FLOAT8_E4M3FN when bias is present.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        bool needChangeFormat = op::IsSupportND() ? false : true;
        return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, needChangeFormat, true, executor);
    };

    aclnnStatus Impl() override
    {
        convOut = FUNCTION_CALL(
            l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
            useHf32, executor);
        if (convOut == nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "convTranspose2d raise an unknown error");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override
    {
        if (bias && op::IsSupportND()) {
            op::Shape biasShape = bias->GetViewShape();
            int64_t biasLength = biasShape.GetDim(0);
            if (output->GetStorageFormat() == op::Format::FORMAT_NHWC) {
                bias = l0op::Reshape(bias, {1, 1, 1, biasLength}, executor);
            } else {
                bias = l0op::Reshape(bias, {1, biasLength, 1, 1}, executor);
            }
            CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);

            convOut = l0op::Add(convOut, bias, executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        bool needChangeFormat = op::IsSupportND() ? false : true;
        auto res = CommonPostProcess(groups, needChangeFormat, output, convOut, executor);
        CHECK_RET(res == ACLNN_SUCCESS, res);

        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };
    ~ConvTransposed2dImpl() override = default;
};
class ConvTransposed3dImpl : public ConvolutionImpl {
public:
    CONV_CONSTRUCTOR(Transposed3d)

    aclnnStatus PreProcess() override
    {
        op::Format dataFormat;
        if (op::IsSupportND()) {
            dataFormat = op::Format::FORMAT_NCDHW;
        } else {
            dataFormat = op::Format::FORMAT_NDC1HWC0;
        }
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdFp16, op::DataType::DT_FLOAT16, dataFormat, op::DataType::DT_FLOAT16,
            dataFormat);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdFp32, op::DataType::DT_FLOAT, dataFormat, op::DataType::DT_FLOAT,
            dataFormat);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdBf16, op::DataType::DT_BF16, dataFormat, op::DataType::DT_BF16, dataFormat);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdHif8, op::DataType::DT_HIFLOAT8, dataFormat, op::DataType::DT_HIFLOAT8,
            dataFormat);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdF8e4m3fn, op::DataType::DT_FLOAT8_E4M3FN, dataFormat,
            op::DataType::DT_FLOAT8_E4M3FN, dataFormat);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdFp16, op::DataType::DT_FLOAT16, op::Format::FORMAT_NDHWC,
            op::DataType::DT_FLOAT16, op::Format::FORMAT_NDHWC);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdFp32, op::DataType::DT_FLOAT, op::Format::FORMAT_NDHWC,
            op::DataType::DT_FLOAT, op::Format::FORMAT_NDHWC);
        REG_L0_FUNCTION(
            l0Functions, ConvTranspose3d6HdBf16, op::DataType::DT_BF16, op::Format::FORMAT_NDHWC, op::DataType::DT_BF16,
            op::Format::FORMAT_NDHWC);
        GetConv3dOpInfo(input, weight, bias, output, opInfo, transposed, cubeMathType);
        // 调用静态函数PreProcess
        if (bias != nullptr &&
            (opInfo.outputDtype == op::DataType::DT_HIFLOAT8 || opInfo.outputDtype == op::DataType::DT_FLOAT8_E4M3FN)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "ConvTranspose3d does not support DataType HIFLOAT8 or FLOAT8_E4M3FN when bias is present.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        bool needChangeFormat = op::IsSupportND() ? false : true;
        return CommonPreProcess(input, weight, bias, groups, transposed, opInfo, needChangeFormat, true, executor);
    };

    aclnnStatus Impl() override
    {
        convOut = FUNCTION_CALL(
            l0Functions, opInfo, input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups,
            useHf32, executor);
        if (convOut == nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "convTranspose3d raise an unknown error");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    };

    aclnnStatus ProcessBias(op::Format dstFormat)
    {
        if (!IsSupportND()) {
            // output format transdata
            convOut = l0op::TransData(convOut, dstFormat, groups, executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if (bias) {
            op::Shape biasShape = bias->GetViewShape();
            int64_t biasLength = biasShape.GetDim(0);
            if (dstFormat == op::Format::FORMAT_NDHWC) {
                bias = l0op::Reshape(bias, {1, 1, 1, 1, biasLength}, executor);
            } else {
                bias = l0op::Reshape(bias, {1, biasLength, 1, 1, 1}, executor);
            }
            CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);

            convOut = l0op::Add(convOut, bias, executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if (!op::IsSupportND()) {
            // output cast
            convOut = l0op::Cast(convOut, output->GetDataType(), executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if (op::IsSupportND()) {
            bool needChangeFormat = op::IsSupportND() ? false : true;
            auto res = CommonPostProcess(groups, needChangeFormat, output, convOut, executor);
            CHECK_RET(res == ACLNN_SUCCESS, res);
        }

        return ACLNN_SUCCESS;
    }

    virtual aclnnStatus ProcessConvOut()
    {
        return ProcessBias(output->GetStorageFormat());
    }

    aclnnStatus PostProcess() override
    {
        auto res = ProcessConvOut();
        CHECK_RET(res == ACLNN_SUCCESS, res);

        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };

    ~ConvTransposed3dImpl() override = default;
};

class ConvTranspose2dTo3dImpl : public ConvTransposed3dImpl {
public:
    ConvTranspose2dTo3dImpl(
        const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,
        const aclIntArray* strideParam, const aclIntArray* paddingParam, const aclIntArray* dilationParam,
        const bool transposedParam, const aclIntArray* outputPaddingParam, const int64_t groupsParam,
        aclTensor* outputParam, bool useHf32Param, int8_t cubeMathTypeParam, aclOpExecutor* executorParam)
        : ConvTransposed3dImpl(
              inputParam, weightParam, biasParam, strideParam, paddingParam, dilationParam, transposedParam,
              outputPaddingParam, groupsParam, outputParam, useHf32Param, cubeMathTypeParam, executorParam)
    {}

    aclnnStatus PreProcess() override
    {
        constexpr int paddingDim = 6; // 3D padding Dim
        std::vector<int64_t> data = {0, 0, (*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1]};
        if (padding->Size() == CONV_4D_PAD_DIM) {
            data = {0, 0, (*padding)[0], (*padding)[1], (*padding)[2], (*padding)[3]};
        }
        padding = executor->AllocIntArray(data.data(), paddingDim);
        CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
        outputPadding = View2dAs3dForAttr(outputPadding, 0, executor, false);
        CHECK_RET(outputPadding != nullptr, ACLNN_ERR_INNER_NULLPTR);
        stride = View2dAs3dForAttr(stride, 1, executor, false);
        CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dilation = View2dAs3dForAttr(dilation, 1, executor, false);
        CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto changeRes = ChangeConv2dInputToConv3d(input, weight, executor);
        CHECK_RET(changeRes == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        return ConvTransposed3dImpl::PreProcess();
    };

    aclnnStatus ProcessConvOut() override
    {
        auto res = ConvTransposed3dImpl::ProcessBias(op::Format::FORMAT_NCDHW);
        CHECK_RET(res == ACLNN_SUCCESS, res);
        convOut = View5dAs4dForOutput(convOut, executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };

    ~ConvTranspose2dTo3dImpl() override = default;
};

class ConvTransposed1dTo3dImpl : public ConvTranspose2dTo3dImpl {
public:
    ConvTransposed1dTo3dImpl(
        const aclTensor* inputParam, const aclTensor* weightParam, const aclTensor* biasParam,
        const aclIntArray* strideParam, const aclIntArray* paddingParam, const aclIntArray* dilationParam,
        const bool transposedParam, const aclIntArray* outputPaddingParam, const int64_t groupsParam,
        aclTensor* outputParam, bool useHf32Param, int8_t cubeMathTypeParam, aclOpExecutor* executorParam)
        : ConvTranspose2dTo3dImpl(
              inputParam, weightParam, biasParam, strideParam, paddingParam, dilationParam, transposedParam,
              outputPaddingParam, groupsParam, outputParam, useHf32Param, cubeMathTypeParam, executorParam)
    {}

    aclnnStatus PreProcess() override
    {
        // 1d transpose: pad仅支持1维度
        outputPadding = View1dAs2d(outputPadding, 0, executor);
        CHECK_RET(outputPadding != nullptr, ACLNN_ERR_INNER_NULLPTR);
        stride = View1dAs2d(stride, 1, executor);
        CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);
        padding = View1dAs2d(padding, 0, executor);
        CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dilation = View1dAs2d(dilation, 1, executor);
        CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);
        input = View3dAs4d(input, executor);
        CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
        weight = View3dAs4d(weight, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ConvTranspose2dTo3dImpl::PreProcess();
    };

    aclnnStatus ProcessConvOut() override
    {
        auto res = ConvTranspose2dTo3dImpl::ProcessConvOut(); // 先降为2D
        CHECK_RET(res == ACLNN_SUCCESS, res);
        convOut = View4dAs3d(convOut, executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        return ACLNN_SUCCESS;
    };

    ~ConvTransposed1dTo3dImpl() override = default;
};

std::shared_ptr<ConvolutionImpl> CreateConvolutionImpl(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, const bool transposed, const bool tbc,
    const aclIntArray* outputPadding, const int64_t groups, int8_t cubeMathType, aclTensor* output,
    aclOpExecutor* executor)
{
    // 存疑：是否按照原来的只看input dtype
    auto promoteType = op::PromoteType(input->GetDataType(), weight->GetDataType());
    // In conv hif8 case, do not consider bias dtype for useHf32
    if ((bias != nullptr) && (promoteType != DataType::DT_HIFLOAT8 && promoteType != DataType::DT_FLOAT8_E4M3FN)) {
        promoteType = op::PromoteType(promoteType, bias->GetDataType());
    }
    bool useHf32 = NeedCubeGoHF32(promoteType, cubeMathType);

    size_t inputDim = input->GetViewShape().GetDimNum();
    if (tbc) {
        return std::make_shared<ConvTbcImpl>(
            input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
            cubeMathType, executor);
    }
    if (!transposed) {
        switch (inputDim) {
            case CONV_1D_DIM_SIZE: {
                return std::make_shared<Conv1dImpl>(
                    input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                    cubeMathType, executor);
            }
            case CONV_2D_DIM_SIZE: {
                return std::make_shared<Conv2dImpl>(
                    input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                    cubeMathType, executor);
            }
            case CONV_3D_DIM_SIZE: {
                return std::make_shared<Conv3dImpl>(
                    input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                    cubeMathType, executor);
            }
            default:
                return nullptr;
        }
    }
    switch (inputDim) {
        case CONV_1D_DIM_SIZE: {
            if (IsSupportConv1DTransposeTo3D()) {
                return std::make_shared<ConvTransposed1dTo3dImpl>(
                    input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                    cubeMathType, executor);
            }
            return std::make_shared<ConvTransposed1dImpl>(
                input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                cubeMathType, executor);
        }
        case CONV_2D_DIM_SIZE: {
            if (IsSupportConv2DTransposeTo3D(
                    input, weight, bias, stride, padding, dilation, outputPadding, groups, output)) {
                return std::make_shared<ConvTranspose2dTo3dImpl>(
                    input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                    cubeMathType, executor);
            }
            return std::make_shared<ConvTransposed2dImpl>(
                input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                cubeMathType, executor);
        }
        case CONV_3D_DIM_SIZE: {
            return std::make_shared<ConvTransposed3dImpl>(
                input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, useHf32,
                cubeMathType, executor);
        }
        default:
            return nullptr;
    }
}

} // namespace AclnnConvolution

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnConvolutionGetWorkspaceSize(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, bool transposed, const aclIntArray* outputPadding,
    const int64_t groups, aclTensor* output, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnConvolution,
        DFX_IN(input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, cubeMathType),
        DFX_OUT(output));
    // construct param and convolution engine
    ConvParams params = {input,         weight, bias,   stride,       padding,       dilation, transposed,
                         outputPadding, groups, output, cubeMathType, workspaceSize, executor};
    ConvEngine convEngine(params);
    // check param
    auto ret = CheckConvParams(convEngine);
    CHECK_RET_CODE(ret, "Check Param failed");

    auto uniqueExecutor = CREATE_EXECUTOR();
    // 创建OpExecutor
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (!(input->IsEmpty() || weight->IsEmpty() || output->IsEmpty())) {
        auto convToBmmMode = GetConvToBmmMode(convEngine);
        auto convTranspose1DToBmmMode = GetConvTranspose1DToBmmMode(convEngine);
        if (convToBmmMode != ConvToBmmMode::CONV_NO_MM) {
            OP_LOGD("Aclnn convolution entering batch matmul branch.");
            BatchMatmulInput bmmInput = {nullptr, nullptr, nullptr, nullptr, false, false};
            ret = GenInOutByConvToBmm(convEngine, convToBmmMode, bmmInput, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);

            auto convOut = ExecBatchMatmulOpWithBiasAndAttrs(
                bmmInput.leftData, bmmInput.rightData, bmmInput.biasData, bmmInput.outputData, bmmInput.isLeftTranspose,
                bmmInput.isRightTranspose, cubeMathType, uniqueExecutor.get());
            OP_CHECK(
                convOut != nullptr, OP_LOGE(ACLNN_ERR_INNER, "The BatchMatmul in Conv Return Nullptr."),
                return ACLNN_ERR_INNER_NULLPTR);
            auto resForBmm = CommonPostProcessForBmm(output, convOut, uniqueExecutor.get());
            CHECK_RET(resForBmm == ACLNN_SUCCESS, resForBmm);
        } else if (convTranspose1DToBmmMode != ConvTranspose1DToBmmMode::CONVTRANSPOSE1D_NO_MM) {
            BatchMatmulInput bmmInput = {nullptr, nullptr, nullptr, nullptr, false, false};
            ret = GenInOutByConvTranspose1DToBmm(convEngine, convTranspose1DToBmmMode, bmmInput, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            auto convTranspose1DOut = ExecBatchMatmulOpWithBiasAndAttrs(
                bmmInput.leftData, bmmInput.rightData, bmmInput.biasData, bmmInput.outputData, bmmInput.isLeftTranspose,
                bmmInput.isRightTranspose, cubeMathType, uniqueExecutor.get());
            OP_CHECK(
                convTranspose1DOut != nullptr,
                OP_LOGE(ACLNN_ERR_INNER, "The BatchMatmul in ConvTranspose1D Return Nullptr."),
                return ACLNN_ERR_INNER_NULLPTR);
            auto resForBmm = CommonPostProcessForBmm(output, convTranspose1DOut, uniqueExecutor.get());
            CHECK_RET(resForBmm == ACLNN_SUCCESS, resForBmm);
        } else {
            std::shared_ptr<AclnnConvolution::ConvolutionImpl> convImpl = AclnnConvolution::CreateConvolutionImpl(
                input, weight, bias, stride, padding, dilation, transposed, false, outputPadding, groups, cubeMathType,
                output, uniqueExecutor.get());
            if (convImpl == nullptr) {
                return ACLNN_ERR_INNER;
            }

            ret = convImpl->PreProcess();
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }

            ret = convImpl->Impl();
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }

            ret = convImpl->PostProcess();
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
    } else {
        OP_LOGD("Input is zero tensor.");
    }

    *workspaceSize = (uniqueExecutor.get())->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvTbcGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weight, const aclTensor* bias, const int64_t pad, aclTensor* output,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnConvTbc, DFX_IN(self, weight, bias, pad, cubeMathType), DFX_OUT(output));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParamsNullptrTbc(self, weight, bias, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        ret = CheckParamsEmpty(output, bias);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    // 创建对应TBC的padding, stride, dilation
    aclIntArray* padding = ViewValueAs1d(pad, uniqueExecutor.get()); // [pad]
    // 1: 单位步长值
    FVector<int64_t> unitValue{1};
    const aclIntArray* strideArray = (uniqueExecutor.get())->AllocIntArray(unitValue.data(), 1);   // [1]
    const aclIntArray* dilationArray = (uniqueExecutor.get())->AllocIntArray(unitValue.data(), 1); // [1]
    // permute & unsqueeze input, weight, output
    FVector<int64_t> permuteDimsAll{1, 2, 0};
    FVector<int64_t> permuteDimsWeight{2, 1, 0};
    op::Shape inputShape = op::Shape({self->GetViewShape()[1], self->GetViewShape()[2], self->GetViewShape()[0]});
    op::Shape outputShape =
        op::Shape({output->GetViewShape()[1], output->GetViewShape()[2], output->GetViewShape()[0]});
    op::Shape weightShape =
        op::Shape({weight->GetViewShape()[2], weight->GetViewShape()[1], weight->GetViewShape()[0]});

    auto* permuteInput = self->IsEmpty() ? ViewWithShape(self, inputShape, uniqueExecutor.get()) :
                                           Permute(self, permuteDimsAll, uniqueExecutor.get());
    auto* permuteOutput = output->IsEmpty() ? ViewWithShape(output, outputShape, uniqueExecutor.get()) :
                                              Permute(output, permuteDimsAll, uniqueExecutor.get());
    auto* permuteWeight = weight->IsEmpty() ? ViewWithShape(weight, weightShape, uniqueExecutor.get()) :
                                              Permute(weight, permuteDimsWeight, uniqueExecutor.get());
    auto permuteOutputT = const_cast<aclTensor*>(permuteOutput);
    // construct param and convolution engine
    ConvParams params = {permuteInput,  permuteWeight, bias,    strideArray, padding,
                         dilationArray, false,         nullptr, 1,           permuteOutputT,
                         cubeMathType,  workspaceSize, executor};
    ConvEngine convEngine(params);
    // conv_tbc param check
    ret = CheckConvTbcParams(convEngine);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (!(self->IsEmpty() || weight->IsEmpty() || output->IsEmpty())) {
        // convTbcImplement
        std::shared_ptr<AclnnConvolution::ConvolutionImpl> convImpl = AclnnConvolution::CreateConvolutionImpl(
            permuteInput, permuteWeight, bias, strideArray, padding, dilationArray, false, true, nullptr, 1,
            cubeMathType, output, uniqueExecutor.get());
        ret = ExecuteConvImpl(convImpl);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    } else if ((self->IsEmpty() || weight->IsEmpty()) && !output->IsEmpty()) {
        OP_LOGD("Input is zero tensor, and output is non-zero tenosr.");
        auto biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
        op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(output->GetViewShape());
        auto shapes = (uniqueExecutor.get())->AllocIntArray(broadcastDims.data(), output->GetViewShape().GetDimNum());
        auto out = l0op::BroadcastTo(biasContiguous, shapes, uniqueExecutor.get());
        auto outCast = l0op::Cast(out, output->GetDataType(), uniqueExecutor.get());
        auto viewCopyOut = l0op::ViewCopy(outCast, output, uniqueExecutor.get());
        CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        OP_LOGD("Output is zero tensor.");
    }

    *workspaceSize = (uniqueExecutor.get())->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvolution(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConvolution);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnConvTbc(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConvTbc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weight, const aclIntArray* kernelSize, const aclTensor* bias,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnConvDepthwise2d, DFX_IN(self, weight, kernelSize, bias, stride, padding, dilation, cubeMathType),
        DFX_OUT(out));
    int64_t groups = 1;
    // construct param and convolution engine
    ConvParams params = {self,    weight, bias, stride,       padding,       dilation, false,
                         nullptr, groups, out,  cubeMathType, workspaceSize, executor};
    ConvEngine convEngine(params);
    // check param
    auto ret = CheckConvDepthwise2dKernelSize(convEngine, kernelSize);
    CHECK_RET_CODE(ret, "Check kernelSize failed");
    ret = CheckConvDepthwise2dParams(convEngine);
    CHECK_RET_CODE(ret, "Check Param failed");

    auto uniqueExecutor = CREATE_EXECUTOR();
    // 创建OpExecutor
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空tensor的情况，由于外部已经将output的shape,type,format设置好，故不需要做任何操作，直接返回
    if (!(self->IsEmpty() || weight->IsEmpty() || out->IsEmpty())) {
        groups = self->GetViewShape().GetDim(1);
        if (self->GetViewFormat() == op::Format::FORMAT_NHWC) {
            groups = self->GetViewShape().GetDim(op::specialChannelIndex);
        }
        std::shared_ptr<AclnnConvolution::ConvolutionImpl> convImpl = AclnnConvolution::CreateConvolutionImpl(
            self, weight, bias, stride, padding, dilation, false, false, nullptr, groups, cubeMathType, out,
            uniqueExecutor.get());
        if (convImpl == nullptr) {
            return ACLNN_ERR_INNER;
        }

        ret = convImpl->PreProcess();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = convImpl->Impl();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = convImpl->PostProcess();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    } else {
        OP_LOGD("Input is zero tensor.");
    }
    *workspaceSize = (uniqueExecutor.get())->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvDepthwise2d(
    void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConvDepthwise2d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif