/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_UT_LOG_H
#define KERNEL_UT_LOG_H

#include <cstdio>

/**
 * Kernel UT 独立日志宏定义
 * 完全独立，不依赖 pkg_inc 或 data_utils.h
 */

#define KERNEL_UT_LOG_INFO(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define KERNEL_UT_LOG_WARN(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define KERNEL_UT_LOG_ERROR(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

#endif // KERNEL_UT_LOG_H