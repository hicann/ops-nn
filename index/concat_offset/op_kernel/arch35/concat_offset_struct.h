/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_offset_struct.h
 * \brief
 */

#ifndef CONCAT_OFFSET_STRUCT_H
#define CONCAT_OFFSET_STRUCT_H

class ConcatOffsetTilingData {
public:
    int64_t threadNum;
    int64_t concatDim;
    int64_t perTensorShapeSize;
    int64_t needCalNum;
};

#endif