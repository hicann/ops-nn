/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_COMMON_AICPU_OP_DEF_H
#define OPS_NN_COMMON_AICPU_OP_DEF_H

#include <string>

namespace ops {
const std::string NN_AICPU_KERNEL_SO = "libnn_aicpu_kernels.so";
const std::string AICPU_OP_KERNEL_LIB = "CUSTAICPUKernel";
const std::string LEGACY_AICPU_OP_KERNEL_LIB = "AICPUKernel";
const std::string OP_INFO_WORKSPACESIZE = "opInfo.workspaceSize";
const std::string DEFAULT_WORKSPACE_SIZE = "100";
const std::string OP_INFO_FORMAT_AGNOSTIC = "opInfo.formatAgnostic";
const std::string FALSE_FORMAT_AGNOSTIC = "False";
const std::string TRUE_FORMAT_AGNOSTIC = "True";
const std::string OP_INFO_OPS_FLAG = "opInfo.opsFlag";
const std::string CLOSE_OPS_FLAG = "OPS_FLAG_CLOSE";
const std::string OPEN_OPS_FLAG = "OPS_FLAG_OPEN";
const std::string OP_INFO_SUB_TYPE_OF_INFERSHAPE = "opInfo.subTypeOfInferShape";
const std::string DEFAULT_SUB_TYPE_OF_INFERSHAPE_1 = "1";
const std::string DEFAULT_SUB_TYPE_OF_INFERSHAPE_3 = "3";

template <typename TOpDef>
inline void ApplyNnAicpuDefaultCfg(TOpDef& opDef)
{
    opDef.AICPU().OpKernelLib(AICPU_OP_KERNEL_LIB.c_str());
    opDef.AICPU().KernelSo(NN_AICPU_KERNEL_SO.c_str());
    opDef.AICPU().UserDefined(true);
    opDef.AICPU().ExtendCfgInfo(OP_INFO_WORKSPACESIZE.c_str(), DEFAULT_WORKSPACE_SIZE.c_str());
    opDef.AICPU().ExtendCfgInfo(OP_INFO_FORMAT_AGNOSTIC.c_str(), FALSE_FORMAT_AGNOSTIC.c_str());
    opDef.AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), CLOSE_OPS_FLAG.c_str());
    opDef.AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), DEFAULT_SUB_TYPE_OF_INFERSHAPE_1.c_str());
}
} // namespace ops

#endif
