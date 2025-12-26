/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_CONVTBC_BACKWARD_CHECKER_H_
#define OP_API_INC_CONVTBC_BACKWARD_CHECKER_H_

#include "aclnn_convolution_backward.h"

#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "../../common/op_host/op_api/conv_cube_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "runtime/context.h"
namespace Ops {
namespace NN {
namespace Conv {
struct ConvTbcBackwardInput {
    const aclTensor *self;
    const aclTensor *input;
    const aclTensor *weight;
    const aclTensor *bias;
};

struct ConvTbcBackwardOutput {
    const aclTensor *gradInput;
    const aclTensor *gradWeight;
    const aclTensor *gradBias;
};

struct ConvTbcBackwardParams {
    int64_t pad;
    int8_t cubeMathType;
};

class ConvTbcBackwardChecker {
public:
ConvTbcBackwardChecker(const ConvTbcBackwardInput &inputTensor, const ConvTbcBackwardOutput &outputTensor,
                       const ConvTbcBackwardParams &params, const op::SocVersion socVersion):
                       inputTensor_(inputTensor),
                       outputTensor_(outputTensor),
                       params_(params),
                       socVersion_(socVersion){}
public:
/**
 * @brief [self, input, weight, gradInput, gradWeight, gradBias] nullptr 约束
 *
 * 详细约束说明
 *
 * ```
 * 1. [self, input, weight, gradInput, gradWeight, gradBias] 均不能等于 nullptr
 * ```
 */
inline bool CheckTbcNotNull();
/**
 * @brief [self, input, weight, gradInput, gradWeight, gradBias] 的 Dtype约束
 *
 * 详细约束说明
 *
 * ```
 * 1. ?
 * ```
 */
bool CheckTbcDtypeValid(const aclTensor *inputTensor) const;
/**
 * @brief [bias] 的 Dtype 约束
 *
 * 详细约束说明：
 * ```
 * 1. [bias] in [DT_FLOAT, DT_FLOAT16, DT_BF16]
 * ```
 */
bool CheckDtypeValidBf16Allowed(const aclTensor *inputTensor) const;
/**
 * @brief [self, input, weight, gradInput, gradWeight] Format约束
 *
 * 详细约束说明
 *
 * ```
 * 1. [self, input, weight, gradInput, gradWeight] in [FORMAT_ND, FORMAT_NCL]
 * ```
 */
bool CheckTbcFormat(const aclTensor *inputTensor, const string &tensorName) const;
/**
 * @brief [bias, gradBias] Format约束
 *
 * 详细约束说明
 *
 * ```
 * 1. [bias, gradBias] in [FORMAT_ND]
 * ```
 */
bool CheckTbcBiasFormat(const aclTensor *inputTensor, const string &tensorName) const;
/**
 * @brief 输入输出shape约束
 *
 * 详细约束说明
 * ```
 * 约束1: 输入tensor的维度规则：[self, input, weight, bias] == [3, 3, 3, 1]
 * 约束2: 输出tensor的维度规则: [gradInput, gradWeight, gradBias] == [3, 3, 1]
 * 约束3: input[2] == weight[1]
 * 约束4: bias[0] == weight[2]
 * 约束5: pad >= 0
 * 约束6:
 * t = input[0] + 2*pad + 1 - weight[0]
 * b = input[1]
 * c0 = weight[2]
 * t >= 0 && self[0] == t && self[1] == b && self[2] == c0
 * 约束7:
 * input.shape == gradInput.shape &&
 * weight.shape == gradWeight.shape &&
 * bias.shape == gradBias.shape
 * ```
 */
bool CheckTbcShape();
/**
 * @brief [self, input, weight] 是否满足推导规则
 */
bool CheckTbcCubeMathType();
/**
 * @brief tbc参数校验
 */
aclnnStatus CheckTbcParams();

private:
const ConvTbcBackwardInput inputTensor_;
const ConvTbcBackwardOutput outputTensor_;
const ConvTbcBackwardParams params_;
const op::SocVersion socVersion_;
};
} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // OP_API_INC_CONVTBC_BACKWARD_CHECKER_H_