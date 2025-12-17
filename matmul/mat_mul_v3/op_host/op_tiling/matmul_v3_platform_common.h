/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file matmul_v3_platform_common.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__
#define __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__

#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "error_util.h"
namespace optiling {
const std::initializer_list<platform_ascendc::SocVersion> AdvancedSocVersion = {
    platform_ascendc::SocVersion::ASCEND910_95};

template <typename T>
inline typename std::enable_if<
    std::is_same<T, gert::TilingParseContext>::value || std::is_same<T, gert::TilingContext>::value, bool>::type
IsAdvancedSocVersion(T *context) {
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    return std::find(AdvancedSocVersion.begin(), AdvancedSocVersion.end(), socVersion) != AdvancedSocVersion.end();
}
}
#endif // __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__
