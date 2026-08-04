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
 * \file sgd_tiling.h
 * \brief
 */
#ifndef RUNTIME_V2_OP_IMPL_SGD_REGBASE_TILING_H_
#define RUNTIME_V2_OP_IMPL_SGD_REGBASE_TILING_H_

#include "atvoss/elewise/elewise_tiling.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/sgd_dag.h"
#include "../../op_kernel/arch35/sgd_tiling_key.h"
#include "../../op_kernel/arch35/sgd_tiling_data.h"

namespace optiling {
using namespace Ops::Base;

class SgdRegbaseTiling {
public:
    explicit SgdRegbaseTiling(gert::TilingContext* context) : tilingContext_(context) {};

    ge::graphStatus RunTiling();
    SgdRegbaseTilingData* tiling_ = nullptr;

protected:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckShapeAndType();
    ge::graphStatus CheckScalarShape(int32_t inputIdx);
    ge::graphStatus CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape);
    ge::graphStatus CheckSameDtype(int32_t inputIdx, const ge::DataType& input0Dtype);
    ge::graphStatus CheckRank(const gert::Shape& input0Shape);
    ge::graphStatus CheckNotEmpty(const gert::Shape& input0Shape);
    ge::graphStatus DoElewiseTiling();
    ge::graphStatus SetTilingData();

    // DoElewiseTiling 的编译期分支展开：按 useNesterov / hasWeightDecay /
    // hasDampening 三个属性分支 × dtype 实例化对应的 OpDag 反解 ubFormer。
    // 【一律用回写 DAG（doWriteback = true）】—— 掩码 DAG 的 BufferNum 严格更小
    // （OutList::Size 3→1，GetLvl12Mte3Count()*2 由 6 降到 2，另有 OpZeroTsr 与两个
    // 降精度 Cast 被裁），按更保守的一套反解不会溢出。且 Host 看不见 momentum 的值，
    // 本来也无从按掩码分支反解。
    template <bool useNesterov, bool hasWeightDecay, bool hasDampening>
    ge::graphStatus DoElewiseTilingByDtype(ElewiseBaseTiling& eleBaseTiling, ge::DataType dtype);

private:
    gert::TilingContext* tilingContext_ = nullptr;
    uint64_t tilingKey_ = 0;
    uint64_t useNesterovKey_ = 0;
    uint64_t hasWeightDecayKey_ = 0;
    uint64_t hasDampeningKey_ = 0;
    bool nesterov_ = false;
    float dampening_ = 0.0f;
    float weightDecay_ = 0.0f;
};
} // namespace optiling

#endif // RUNTIME_V2_OP_IMPL_SGD_REGBASE_TILING_H_
