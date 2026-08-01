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
 * \file add_rms_norm_dynamic_quant_v2_tiling_arch35.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_V2_TILING_ARCH35_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_V2_TILING_ARCH35_H_

#include "norm/add_rms_norm_dynamic_quant/op_host/arch35/add_rms_norm_dynamic_quant_tiling_arch35.h"

namespace optiling {
struct AddRmsNormDynamicQuantV2CompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND950;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};
} // namespace optiling
#endif // ADD_RMS_NORM_DYNAMIC_QUANT_V2_TILING_ARCH35_H_
