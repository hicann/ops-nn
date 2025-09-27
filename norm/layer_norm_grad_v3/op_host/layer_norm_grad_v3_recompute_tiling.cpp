/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_v3_recompute_tiling.cc
 * \brief
 */

#include <iostream>
#include <vector>
#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_EIGHT = 8;

constexpr int64_t DEFAULT_GAMMA_BETA_M_FACTOR = 128;
#ifdef DAVID_FPGA
constexpr int64_t DEFAULT_GAMMA_BETA_N_FACTOR = 16;
#else
constexpr int64_t DEFAULT_GAMMA_BETA_N_FACTOR = 64;
#endif

constexpr int64_t DEFAULT_BACKWARD_M_FACTOR = 1;
#ifdef DAVID_FPGA
constexpr int64_t DEFAULT_BACKWARD_N_FACTOR = 1024;
#else
constexpr int64_t DEFAULT_BACKWARD_N_FACTOR = 1024 * 6;
#endif

bool LayerNormGradV3RecomputeTiling::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::GammaBetaKernelTiling()
{
    int64_t gammaBetaMainBlockFactor =
        Ops::Base::CeilDiv(static_cast<int64_t>(commonParams.colSize), static_cast<int64_t>(commonParams.coreNum));
    if (gammaBetaMainBlockFactor < DEFAULT_GAMMA_BETA_N_FACTOR) {
        gammaBetaMainBlockFactor = DEFAULT_GAMMA_BETA_N_FACTOR;
    }
    int64_t gammaBetaBlockDim =  Ops::Base::CeilDiv(static_cast<int64_t>(commonParams.colSize), gammaBetaMainBlockFactor);
    int64_t gammaBetaMainBlockCount = static_cast<int64_t>(commonParams.colSize) / gammaBetaMainBlockFactor;
    int64_t gammaBetaTailBlockCount = gammaBetaBlockDim - gammaBetaMainBlockCount;
    int64_t gammaBetaTailBlockFactor =
        static_cast<int64_t>(commonParams.colSize) - gammaBetaMainBlockCount * gammaBetaMainBlockFactor;
    int64_t gammaBetaMloop = static_cast<int64_t>(commonParams.rowSize) / DEFAULT_GAMMA_BETA_M_FACTOR;
    int64_t gammaBetaMTotalLoop =  Ops::Base::CeilDiv(static_cast<int64_t>(commonParams.rowSize), DEFAULT_GAMMA_BETA_M_FACTOR);
    int64_t gammaBetaMtail = static_cast<int64_t>(commonParams.rowSize) - gammaBetaMloop * DEFAULT_GAMMA_BETA_M_FACTOR;

    int64_t gammaBetaBasicBlockLoop = 0;
    if (gammaBetaMTotalLoop <= CONST_ONE) {
        gammaBetaBasicBlockLoop = CONST_ZERO;
    } else if (gammaBetaMTotalLoop <= CONST_TWO) {
        gammaBetaBasicBlockLoop = CONST_ONE;
    } else if (gammaBetaMTotalLoop <= CONST_FOUR) {
        gammaBetaBasicBlockLoop = CONST_TWO;
    } else {
        int64_t temp_ = gammaBetaMTotalLoop - CONST_ONE;
        int64_t pow_ = static_cast<int64_t>(std::floor(std::log2(temp_)));
        gammaBetaBasicBlockLoop = (CONST_ONE << pow_);
    }
    int64_t gammaBetaMainFoldCount = gammaBetaMloop - gammaBetaBasicBlockLoop;
    int64_t gammaBetaNfactorBlockAligned =  Ops::Base::CeilAlign(DEFAULT_GAMMA_BETA_N_FACTOR, CONST_EIGHT);
    int64_t gammaBetaMfactorBlockAligned =  Ops::Base::CeilAlign(DEFAULT_GAMMA_BETA_M_FACTOR, CONST_EIGHT);

    td_.set_gammaBetaMfactor(DEFAULT_GAMMA_BETA_M_FACTOR);
    td_.set_gammaBetaNfactor(DEFAULT_GAMMA_BETA_N_FACTOR);
    td_.set_gammaBetaMainBlockFactor(gammaBetaMainBlockFactor);
    td_.set_gammaBetaBlockDim(gammaBetaBlockDim);
    td_.set_gammaBetaMainBlockCount(gammaBetaMainBlockCount);
    td_.set_gammaBetaTailBlockCount(gammaBetaTailBlockCount);
    td_.set_gammaBetaTailBlockFactor(gammaBetaTailBlockFactor);
    td_.set_gammaBetaMloop(gammaBetaMloop);
    td_.set_gammaBetaMTotalLoop(gammaBetaMTotalLoop);
    td_.set_gammaBetaMtail(gammaBetaMtail);
    td_.set_gammaBetaBasicBlockLoop(gammaBetaBasicBlockLoop);
    td_.set_gammaBetaMainFoldCount(gammaBetaMainFoldCount);
    td_.set_gammaBetaNfactorBlockAligned(gammaBetaNfactorBlockAligned);
    td_.set_gammaBetaMfactorBlockAligned(gammaBetaMfactorBlockAligned);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::BackwardKernelTiling()
{
    int64_t backwardMainBlockFactor =
         Ops::Base::CeilDiv(static_cast<int64_t>(commonParams.rowSize), static_cast<int64_t>(commonParams.coreNum));
    int64_t backwardBlockDim =  Ops::Base::CeilDiv(static_cast<int64_t>(commonParams.rowSize), backwardMainBlockFactor);

    td_.set_backwardMfactor(DEFAULT_BACKWARD_M_FACTOR);
    td_.set_backwardNfactor(DEFAULT_BACKWARD_N_FACTOR);
    td_.set_backwardMainBlockFactor(backwardMainBlockFactor);
    td_.set_backwardBlockDim(backwardBlockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::DoOpTiling()
{
    td_.set_row(static_cast<int64_t>(commonParams.rowSize));
    td_.set_col(static_cast<int64_t>(commonParams.colSize));
    td_.set_pdxIsRequire(static_cast<int32_t>(commonParams.pdxIsRequire));
    td_.set_pdgammaIsRequire(static_cast<int32_t>(commonParams.pdgammaIsRequire));
    td_.set_pdbetaIsRequire(static_cast<int32_t>(commonParams.pdbetaIsRequire));
    ge::graphStatus statusGammaBeta = GammaBetaKernelTiling();
    ge::graphStatus statusBackward = BackwardKernelTiling();
    return (statusGammaBeta && statusBackward);
}

ge::graphStatus LayerNormGradV3RecomputeTiling::GetWorkspaceSize()
{
    constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::PostTiling()
{
    int64_t blockDim = td_.get_gammaBetaBlockDim() > td_.get_backwardBlockDim() ? td_.get_gammaBetaBlockDim() :
                                                                                   td_.get_backwardBlockDim();
    context_->SetBlockDim(blockDim);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradV3RecomputeTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNGTemplateKey::RECOMPUTE);
    return templateKey * LNG_TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3RecomputeTiling, 7000);
} // namespace optiling
