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
 * \file ascend_dequant_def.cpp
 * \brief AscendDequant operator definition.
 *
 *   The DataType / Format support lists of x, deq_scale and y are parallel.
 *   Six combinations are registered:
 *
 *     combo 0: x int32, deq_scale uint64, y float16, format NHWC
 *     combo 1: x int32, deq_scale uint64, y float16, format NCHW
 *     combo 2: x int32, deq_scale uint64, y float16, format NCDHW
 *     combo 3: x int32, deq_scale uint64, y float16, format NDHWC
 *     combo 4: x int32, deq_scale uint64, y float16, format ND
 *     combo 5: x int32, deq_scale uint64, y float32, format ND
 *
 *   deq_scale is uint64 and ND only. y dtype is selected by attr `dtype`
 *   (0 -> float32, 1 -> float16) and must agree with the registered
 *   combination. The format only labels the layout of x/y; the operator itself
 *   is element-wise, so the channel axis that deq_scale binds to is derived
 *   from the layout.
 *
 *   A uint64 deq_scale carries the IEEE-754 bit pattern of the fp32 scale in
 *   its low 32 bits; the high 32 bits are ignored by the kernel.
 *
 *   attrs: sqrt_mode:Bool=false, relu_flag:Bool=false, dtype:Int=1 (float16).
 */
#include "register/op_def_registry.h"
#include "op_common/op_host/util/op_arch_helper.h"

namespace ops {
class AscendDequant : public OpDef {
public:
    explicit AscendDequant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format(
                {ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("deq_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(
                {ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("sqrt_mode").AttrType(OPTIONAL).Bool(false);
        this->Attr("relu_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("dtype").AttrType(OPTIONAL).Int(ge::DT_FLOAT16);

        OpAICoreConfig config_950;
        config_950.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, config_950);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, config_950);
    }
};

OP_ADD(AscendDequant);
} // namespace ops
