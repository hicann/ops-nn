/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_PCIE_THROUGH_UTIL_H_
#define OP_HOST_PCIE_THROUGH_UTIL_H_

#include "log/log.h"
#include "exe_graph/runtime/tiling_context.h"
#include "version/metadef_version.h"

namespace ops {

/**
 * @brief 判断当前 Tiling context 是否为 PCIe through 场景
 *
 * 需满足：CANN 版本 >= 9.2.0 且 context 的 GetPcieThroughFlag 返回 true
 *
 * @param context Tiling 上下文
 * @return true 表示 PCIe through 场景，false 表示非 PCIe 场景
 */
inline bool IsPcieThrough(const gert::TilingContext* context)
{
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90200000
    bool result = context->GetPcieThroughFlag();
    OP_LOGD(context->GetNodeName(), "IsPcieThrough:%s", (result ? "true" : "false"));
    return result;
#else
    OP_LOGD(context->GetNodeName(), "IsPcieThrough:false");
    return false;
#endif
}

} // namespace ops

#endif // OP_HOST_PCIE_THROUGH_UTIL_H_
