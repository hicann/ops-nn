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
 * \file ge_glu_grad_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GE_GLU_GRAD_V2_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GE_GLU_GRAD_V2_TILING_H_
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GeGluGradV2TilingData)
TILING_DATA_FIELD_DEF(int32_t, approximate);
TILING_DATA_FIELD_DEF(int32_t, activateLeft);
TILING_DATA_FIELD_DEF(int64_t, maxProcCount);
TILING_DATA_FIELD_DEF(int64_t, valueN);
TILING_DATA_FIELD_DEF(int64_t, valueM);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, loopNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndex);
TILING_DATA_FIELD_DEF(int64_t, tailUbLoopNum);
TILING_DATA_FIELD_DEF(int64_t, groupNum);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GeGluGradV2, GeGluGradV2TilingData)

struct GeGluGradV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    NpuArch curSocVersion = NpuArch::DAV_2201;
    bool isRegbase{false};
};

enum class GeGluGradV2TilingKey : uint64_t
{
    TILING_KEY_TANH_101 = 101,
    TILING_KEY_TANH_102,
    TILING_KEY_TANH_103,
    TILING_KEY_TANH_201 = 201,
    TILING_KEY_TANH_202,
    TILING_KEY_TANH_203,
    TILING_KEY_TANH_301 = 301,
    TILING_KEY_TANH_302,
    TILING_KEY_TANH_303,
    TILING_KEY_ERF_701 = 701,
    TILING_KEY_ERF_702,
    TILING_KEY_ERF_703,
    TILING_KEY_ERF_801 = 801,
    TILING_KEY_ERF_802,
    TILING_KEY_ERF_803,
    TILING_KEY_ERF_901 = 901,
    TILING_KEY_ERF_902,
    TILING_KEY_ERF_903
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GE_GLU_GRAD_V2_TILING_H_
