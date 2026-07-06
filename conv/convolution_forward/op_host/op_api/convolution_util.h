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
 * \file convolution_util.h
 * \brief
 */

#ifndef OP_API_SRC_CONVOLUTION_UTIL_H
#define OP_API_SRC_CONVOLUTION_UTIL_H

#include <map>
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "opdev/fast_vector.h"

namespace Ops {
namespace NN {
const bool NAMESPACE_PLACEHOLDER = true;
}
} // namespace Ops

namespace op {
op::FVector<int64_t> ToFVector(const op::Shape& shapeT);
std::string FVectorToString(const op::FVector<int64_t>& vec);
} // namespace op

namespace ConvolutionUtil {
#define CHECK_PARAM_NULLPTR(entityName, param, paramName)                               \
    do {                                                                                \
        if ((param) == nullptr) {                                                       \
            OP_LOGE_FOR_INVALID_VALUE(entityName, paramName, "nullptr", "not nullptr"); \
            return ACLNN_ERR_PARAM_NULLPTR;                                             \
        }                                                                               \
    } while (0)

template <typename T, typename Converter>
std::string VectorsToString(const std::vector<std::vector<T>>& vecs, Converter converter, size_t skipIdx = -1)
{
    std::string result = "[";

    for (size_t j = 0; j < vecs.size(); ++j) {
        result += "[";
        bool needComma = false;
        for (size_t i = 0; i < vecs[j].size(); ++i) {
            if (i == skipIdx) {
                continue;
            }
            if (needComma) {
                result += ",";
            }
            result += converter(vecs[j][i]);
            needComma = true;
        }
        result += "]";
        if (j < vecs.size() - 1) {
            result += ",";
        }
    }
    result += "]";
    return result;
}

std::string GeFormatToString(const ge::Format& geFormat);

std::string GeDtypeToString(const ge::DataType& geDtype);

template <typename T>
std::string IntToString(const T& intValue)
{
    return std::to_string(intValue);
}

constexpr int64_t RESERVED_SIZE_8K = 8 * 1024;

struct ConvolutionOpInfo {
    op::DataType inputDtype;
    op::Format inputFormat;
    op::DataType weightDtype;
    op::Format weightFormat;
    op::DataType biasDtype;
    op::Format biasFormat;
    op::DataType outputDtype;
    op::Format outputFormat;
};

class Conv2DSplitWInfo {
public:
    void InitConv2DSplitWInfo(const aclTensor* input, const aclTensor* weight, const aclIntArray* stride,
                              const aclIntArray* padding, const aclIntArray* dilation);
    bool CanSwitchSplitW(const aclTensor* bias, aclTensor* output, int64_t groups, const ConvolutionOpInfo& opInfo);

private:
    bool CheckConv2DTbeOptFlag(const ConvolutionOpInfo& opInfo);
    bool CheckConv2DPad() const;
    bool CheckConv2DInput() const;
    bool CheckBasicInfoInSplitW(int64_t groups, const ConvolutionOpInfo& opInfo);
    bool CheckLoad3dIns();
    bool CheckLoadL1InSplitW(const aclTensor* bias, aclTensor* output);

private:
    int64_t hi = 0;
    int64_t wi = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t padU = 0;
    int64_t padD = 0;
    int64_t padL = 0;
    int64_t padR = 0;
    int64_t biasTypeSize = 0;
    int64_t k0 = 0;
};

aclnnStatus ChangeConv2dAttrToConv3d(const aclIntArray*& stride, const aclIntArray*& padding,
                                     const aclIntArray*& dilation, aclOpExecutor* executor);
aclnnStatus ChangeConv2dInputToConv3d(const aclTensor*& input, const aclTensor*& weight, aclOpExecutor* executor);
const aclTensor* View4dAs5dForInput(const aclTensor* input, aclOpExecutor* executor);
const aclTensor* View5dAs4dForOutput(const aclTensor* input, aclOpExecutor* executor);
aclIntArray* View2dAs3dForAttr(const aclIntArray* intArray, int64_t expandValue, aclOpExecutor* executor, bool isPad);
aclIntArray* View2DSwapHWForAttr(const aclIntArray* intArray, aclOpExecutor* executor);
const aclTensor* View4DSwapHWForTensor(const aclTensor* input, aclOpExecutor* executor);
bool CheckDisContinuousStride(const aclTensor* input, const std::vector<int64_t>& newStrides, uint32_t dims);
void GetUbSize();
void GetL1Size();
bool CheckDmaLimits(const struct ConvolutionOpInfo* opInfo, const aclTensor* input, const aclTensor* weight,
                    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation,
                    const aclTensor* bias);
bool CheckL1SizeLimitsDma(uint32_t inputDtypeSize, uint64_t biasL1Size, uint32_t weightDtypeSize, int64_t k0);
uint64_t Conv2DInferHiL1(uint64_t inputHoL1, uint64_t khDilated, uint64_t hi, uint64_t strideH);
uint64_t ConvAlignB(uint64_t a, uint64_t b);
} // namespace ConvolutionUtil

#endif // OP_API_SRC_CONVOLUTION_UTIL_H
