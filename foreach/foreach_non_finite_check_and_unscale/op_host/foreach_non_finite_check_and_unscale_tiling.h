/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_non_finite_check_and_unscale_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_

#include <vector>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/log/log.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_COUNT = 256;
constexpr uint16_t MAX_CORE_COUNT = 64;
constexpr uint16_t MAX_CORE_COUNT_REGBASE = 80;
struct ForeachNonFiniteCheckAndUnscaleCompileInfo {
    uint32_t coreNum;
};

BEGIN_TILING_DATA_DEF(ForeachNonFiniteCheckAndUnscaleTilingData)
TILING_DATA_FIELD_DEF(uint32_t, scaledGradsUbSize);
TILING_DATA_FIELD_DEF(uint32_t, reduceTempValUbSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorEndList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorStartOffsetList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachNonFiniteCheckAndUnscale, ForeachNonFiniteCheckAndUnscaleTilingData)

BEGIN_TILING_DATA_DEF(ForeachNonFiniteCheckAndUnscaleRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, scaledGradsUbSize);
TILING_DATA_FIELD_DEF(uint32_t, reduceTempValUbSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT_REGBASE, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT_REGBASE, tensorEndList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT_REGBASE, tensorStartOffsetList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT_REGBASE, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachNonFiniteCheckAndUnscale_100, ForeachNonFiniteCheckAndUnscaleRegbaseTilingData)

class ForeachNonFiniteCheckAndUnscaleBaseClass : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit ForeachNonFiniteCheckAndUnscaleBaseClass(gert::TilingContext* context) : TilingBaseClass(context) {};
    ~ForeachNonFiniteCheckAndUnscaleBaseClass() override = default;
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    NpuArch npuArch = NpuArch::DAV_2201;
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_