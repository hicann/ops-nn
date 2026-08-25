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
 * \file blaze_kernel_stub.h
 * \brief Compatibility definitions required by Blaze kernel UT in CPU debug mode.
 */

#pragma once

#include <cstdint>

#ifndef __biasbuf__
#define __biasbuf__
#endif

#ifndef POS_LOWEST
constexpr int32_t POS_LOWEST = 0;
#endif

#ifndef POS_HIGHEST
constexpr int32_t POS_HIGHEST = 1;
#endif
