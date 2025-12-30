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
 * \file quant_update_scatter_tpl_def.h
 * \brief quant_update_scatter tpl def file
 */

#ifndef QUANT_UPDATE_SCATTER_TPL_DEF_H_
#define QUANT_UPDATE_SCATTER_TPL_DEF_H_

#define TPL_MODE_LITTLE_ELE_LITTLE_QUANT 1
#define TPL_MODE_LARGE_BATCH 2
#define TPL_MODE_LARGE_ELE_LITTLE_QUANT 3
#define TPL_MODE_LARGE_ELE_LARGE_QUANT 4
#define TPL_MODE_LARGE_BATCH_LITTLE_QUANT 5
#define TPL_MODE_LARGE_BATCH_LARGE_QUANT 6

#define TPL_NONE 0
#define TPL_INT32 1
#define TPL_BF16 2

#define TPL_DIV_MODE_DIV 1
#define TPL_DIV_MODE_MUL 2

#define TPL_ROUND_MODE_RINT 1
#define TPL_ROUND_MODE_ROUND 2
#define TPL_ROUND_MODE_HYBRID 3

#endif  // QUANT_UPDATE_SCATTER_TPL_DEF_H_