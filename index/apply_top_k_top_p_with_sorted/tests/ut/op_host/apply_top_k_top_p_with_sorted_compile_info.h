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
 * \file apply_top_k_top_p_with_sorted_compile_info.h
 * \brief
 */
#ifndef __APPLY_TOP_K_TOP_P_WITH_SORTED_COMPILE_INFO_H__
#define __APPLY_TOP_K_TOP_P_WITH_SORTED_COMPILE_INFO_H__

#include "register/tilingdata_base.h"

namespace optiling {

struct TilingForApplyTopKTopPWithSortedCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

}  // namespace optiling
#endif  // __APPLY_TOP_K_TOP_P_WITH_SORTED_COMPILE_INFO_H__