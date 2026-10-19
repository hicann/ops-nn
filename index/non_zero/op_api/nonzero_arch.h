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
 * \file nonzero_arch.h
 * \brief NonZero 算子内部的芯片架构判定。
 */
#ifndef OP_API_INC_LEVEL0_OP_NONZERO_ARCH_H_
#define OP_API_INC_LEVEL0_OP_NONZERO_ARCH_H_

#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

namespace NonZeroArch {
// Ascend960 对应 NpuArch DAV_9201(960DT)/DAV_9202(960PR)。
// SocVersion 枚举中 960 排在末尾，无法用 SocVersion 区间覆盖，只能按 NpuArch 显式判定。
inline bool IsArch92(NpuArch npuArch) { return npuArch == NpuArch::DAV_9201 || npuArch == NpuArch::DAV_9202; }

// Ascend960 与 RegBase 芯片(Ascend950/Ascend350)共用同一套 NonZero 实现。
inline bool IsRegbaseOrArch92(NpuArch npuArch) { return Ops::NN::AclnnUtil::IsRegbase(npuArch) || IsArch92(npuArch); }
} // namespace NonZeroArch

#endif // OP_API_INC_LEVEL0_OP_NONZERO_ARCH_H_
