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
 * \file reverse_sequence_simt_tiling.h
 * \brief
 */

#ifndef REVERSE_SEQUENCE_SIMT_TILING_H_
#define REVERSE_SEQUENCE_SIMT_TILING_H_

#pragma once
#include <array>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "reverse_sequence_tiling_common.h"

namespace optiling
{
using Ops::NN::Optiling::TilingBaseClass;

class ReverseSequenceSimtTiling : public TilingBaseClass
{
public:
    explicit ReverseSequenceSimtTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~ReverseSequenceSimtTiling()
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData() const;
    ge::graphStatus DoReverseSequenceSimtTiling();

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;

public:
    ReverseInputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;
    int64_t perCoreHandleNums_ = 0;
    int64_t usedCoreNums_ = 0;
    int64_t tailCoreHandleNums_ = 0;
    int64_t xUbFactor_ = 0;
    int64_t xUbLoop_ = 0;
    int64_t xTailUbLoopSize_ = 0;
    int64_t xTailCoreLoop_ = 0;
    int64_t xTailCoreTailLoopSize_ = 0;
    int64_t addrRange_ = 0;
};
}
#endif