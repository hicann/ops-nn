/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pp_matmul_default.h
 * \brief
 */
#ifndef __OP_HOST_PP_MATMUL_DEFAULT_H__
#define __OP_HOST_PP_MATMUL_DEFAULT_H__

#include "register/op_def_registry.h"
#include "pp_matmul_info.h"

namespace optiling {
namespace pp_matmul {

class PpMatMulDefault{
public:
    explicit PpMatMulDefault(gert::TilingContext* context) : context_(context) {}
    virtual ~PpMatMulDefault() = default;

    void GetHardwareInfo();
    bool GetMatMulTilingData();
    void PrintTiling();
    gert::TilingContext *context_ = nullptr;
    MatMulInfo matMulInfo_;
    PpMatmulDefaultTilingData ppMatmulDefaultTilingData_;
    HardwareInfo hardwareInfo_;
    uint64_t kernelKey_;
};
}
}
#endif
