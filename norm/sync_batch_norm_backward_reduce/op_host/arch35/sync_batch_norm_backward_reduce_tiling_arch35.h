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
 * \file sync_batch_norm_backward_reduce_tiling_arch35.h
 * \brief
 */
#ifndef SYNC_BATCH_NORM_BACKWARD_REDUCE_TILING_ARCH35_H
#define SYNC_BATCH_NORM_BACKWARD_REDUCE_TILING_ARCH35_H

#include "norm/sync_batch_norm_backward_reduce/op_kernel/arch35/sync_batch_norm_backward_reduce_tilingdata.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

namespace optiling {
class SyncBatchNormBackwardReduceTiling
{
public:
    explicit SyncBatchNormBackwardReduceTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunTiling();
    SyncBatchNormBackwardReduceTilingData *tiling;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
};

} // namespace optiling
#endif // SYNC_BATCH_NORM_BACKWARD_REDUCE_TILING_ARCH35_H