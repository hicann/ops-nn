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
 * \file conv3d_v2_instr.h
 * \brief
 */

#ifndef CONV3D_V2_INSTR_H
#define CONV3D_V2_INSTR_H

#if (__CCE_AICORE__ > 300)
    #include "conv3d_v2_instr_impl.h"
    #include "conv3d_v2_instr_dequant_impl.h"
#endif

#endif // CONV3D_V2_INSTR_H