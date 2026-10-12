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
 * \file add_rms_norm_cast_tiling_reduce_empty_arch35.h
 * \brief 空张量(reduce empty)兜底 tiling 模板类声明。
 *        归一化维元素数 N=0 且行数 M>0 时，把 rstd 当一维 M 个 fp32 做核间切分，
 *        kernel 侧仅按行向 rstd 写 NaN。空归约专属的形状校验(rstd 元素数等于 M)、
 *        切分与序列化在子类实现；Regbase 模板不依赖空归约场景的共享标志。
 *        专用 tiling data 与其注册保留在主头 add_rms_norm_cast_tiling.h
 *        (tiling codegen 仅收集主头，移出会导致 kernel 侧结构缺失)。
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_CAST_REDUCE_EMPTY_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_CAST_REDUCE_EMPTY_H_

#include "add_rms_norm_cast_tiling.h"

namespace optiling {

class AddRmsNormCastReduceEmptyTiling : public AddRmsNormCastRegbaseTiling {
public:
    explicit AddRmsNormCastReduceEmptyTiling(gert::TilingContext* tilingContext)
        : AddRmsNormCastRegbaseTiling(tilingContext)
    {}
    ~AddRmsNormCastReduceEmptyTiling() override = default;

    bool CheckInputShapeValue() override;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    AddRmsNormCastReduceEmptyTilingData reduceEmptyTilingData;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_CAST_REDUCE_EMPTY_H_
