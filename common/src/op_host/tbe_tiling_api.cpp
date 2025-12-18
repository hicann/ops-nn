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
 * \file tbe_tiling_api.cpp
 * \brief
 */
#include "op_host/tbe_tiling_api.h"
#include "log/log.h"
#include "legacy_common_manager.h"

// 兼容opp整包场景：整包不编译本文件，仅子包编译
namespace Ops {
namespace NN {
bool GetTbeTiling(
    gert::TilingContext* context, optiling::Conv3dBackpropV2TBETilingData& tbeTilingForV2,
    const optiling::OpTypeV2 opType)
{
    using FuncType = bool (*)(gert::TilingContext*, optiling::Conv3dBackpropV2TBETilingData&, const optiling::OpTypeV2);
    const char* symbolName = "LegacyGenTbeConvBackwardTiling";
    static FuncType func = Ops::NN::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(context, tbeTilingForV2, opType);
    }
}
} // namespace NN
} // namespace Ops