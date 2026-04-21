/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_DEFORMABLE_CONV2D_BACKWARD_CHECKER_H_
#define OP_API_INC_DEFORMABLE_CONV2D_BACKWARD_CHECKER_H_

#include "aclnn_convolution_backward.h"

#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
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
struct DeformableConv2dBackwardInputTensor {
    const aclTensor* gradOutput;
    const aclTensor* input;
    const aclTensor* weight;
    const aclTensor* offset;
    const aclTensor* offsetOut;
};

struct ConvolutionBackwardOutput {
    aclTensor* gradInput;
    aclTensor* gradWeight;
    aclTensor* gradBias;
    aclTensor* gradOffset;
};

struct ConvolutionBackwardResult {
    const aclTensor* gradInput;
    const aclTensor* gradWeight;
    const aclTensor* gradOffset;
    const aclTensor* gradBias;
};

struct DeformableConv2dBackwardParams {
    const aclIntArray* kernelSize;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    const bool modulated;
    const int64_t groups;
    const int64_t deformableGroups;
};

class DeformableConv2dBackwardChecker {
public:
    DeformableConv2dBackwardChecker(const DeformableConv2dBackwardInputTensor& inputTensor,
                                    const ConvolutionBackwardOutput& outputTensor,
                                    const DeformableConv2dBackwardParams& params,
                                    const NpuArch npuArch)
        : inputTensor_(inputTensor),
          outputTensor_(outputTensor),
          params_(params),
          npuArch_(npuArch) {}

    bool CheckNotNull();
    bool CheckDtypeValid();
    bool CheckFormat();
    bool CheckAttrs();
    bool CheckDimension();
    bool CheckShape();
    aclnnStatus CheckParams();

private:
    const DeformableConv2dBackwardInputTensor inputTensor_;
    const ConvolutionBackwardOutput outputTensor_;
    const DeformableConv2dBackwardParams params_;
    const NpuArch npuArch_;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // OP_API_INC_DEFORMABLE_CONV2D_BACKWARD_CHECKER_H_
