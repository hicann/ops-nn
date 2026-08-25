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
 * \file reverse_sequence_a1sba_tiling.h
 * \brief
 */

#pragma once
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "reverse_sequence_tiling_common.h"

namespace optiling {

using Ops::NN::Optiling::TilingBaseClass;

class ReverseSequenceSBACommonTiling : public TilingBaseClass {
public:
    explicit ReverseSequenceSBACommonTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

    ~ReverseSequenceSBACommonTiling() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void DumpTilingInfo() override;

protected:
    virtual void DoUBTiling();
    virtual void DoUBTilingSingle();
    virtual void DoBlockTiling();
    virtual int64_t CalcBufferSize(int64_t inA1, int64_t inS, int64_t inB, int64_t inA, int64_t splitMode);
    virtual void CalcSplitDimA1();
    virtual void CalcSplitDimA();
    virtual void CalcSplitDimS();
    virtual void CalcSplitDimB();
    virtual void SetTilingData();
    virtual void InitializationVars();

    uint64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t ubFactorA1_{0};
    int64_t ubFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubFactorA_{0};
    int64_t a1Loop_{0};
    int64_t bLoop_{0};
    int64_t sLoop_{0};
    int64_t aLoop_{0};
    int64_t oneBlockNum_{32};
    int64_t availableUb_{0};
    int64_t sbaResvervedNum_{0};
    int64_t inUbSize_{0};
    int64_t usedCoreNum_{0};
    int64_t splitMode_{0};
    int64_t dtypeSize_{0};
    ReverseInputInfo inputData_;
    int64_t addrRange_ = 0;
};

} // namespace optiling
