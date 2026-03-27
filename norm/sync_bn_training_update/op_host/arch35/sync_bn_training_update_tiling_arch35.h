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
 * \file sync_bn_training_update_tiling_arch35.h
 * \brief
 */
#ifndef SYNC_BN_TRAINING_UPDATE_TILING_H
#define SYNC_BN_TRAINING_UPDATE_TILING_H

#include "norm/sync_bn_training_update/op_kernel/arch35/sync_bn_training_update_tilingdata.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

namespace optiling {
class SyncBNTrainingUpdateTiling {
public:
    explicit SyncBNTrainingUpdateTiling(gert::TilingContext *context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    SyncBNTrainingUpdateTilingData *tiling;

protected:
    ge::graphStatus CheckTensorDtype();
    ge::graphStatus SetTilingData();
    void SetAttr();
    void PrintTilingData();

private:
    const char *opName = "SyncBNTrainingUpdate";
    gert::TilingContext* tilingContext;
    ge::DataType meanDtype;
    ge::DataType runningMeanDtype;
    ge::DataType runningMeanUpdateDtype;
    float momentum_ = 0.0f;

};
};  // namespace optiling
#endif // SYNC_BN_TRAINING_UPDATE_TILING_H