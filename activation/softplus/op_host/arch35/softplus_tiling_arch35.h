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
 * \file softplus_tiling_arch35.h
 * \brief
 */
#ifndef _ACTIVATION_HOST_SOFTPLUS_TILING_H_
#define _ACTIVATION_HOST_SOFTPLUS_TILING_H_

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "activation/softplus/op_kernel/arch35/softplus_tilingdata.h"

namespace optiling {

class SoftplusTiling {
public:
    explicit SoftplusTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus SetTilingData();

private:
    ge::graphStatus CheckShape();

    SoftplusTilingData* tiling = nullptr;
    gert::TilingContext* tilingContext_;
    ge::DataType outputDtype;
};

} // namespace optiling
#endif // _ACTIVATION_HOST_SOFTPLUS_TILING_H_
