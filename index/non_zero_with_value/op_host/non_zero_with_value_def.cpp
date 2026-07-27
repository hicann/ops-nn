/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_with_value_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
// A5(arch35):x/value 支持全 12 类(对齐 GE proto REG_OP 的 TensorType,proto 与实现一致);
// 严格 2D、transpose=true、静态 max-size 输出。value dtype 恒等于 x dtype(两列表同序)。
// VF regbase 覆盖 ≤32bit 类型;8 字节(double/int64/uint64)由非 regbase 传统向量路径兜底(功能正确)。
static const std::vector<ge::DataType> xDataType = {ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8,
                                                    ge::DT_UINT8,  ge::DT_INT16, ge::DT_UINT16,  ge::DT_INT32,
                                                    ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,  ge::DT_BOOL};
static const std::vector<ge::DataType> valueDataType = {ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8,
                                                        ge::DT_UINT8,  ge::DT_INT16, ge::DT_UINT16,  ge::DT_INT32,
                                                        ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,  ge::DT_BOOL};
// 所有输入/输出的 DataType/Format 列表长度必须一致(= bin 数 = 12):x/value 为 12 类,
// index/count 恒 int32 也需按 bin 重复 12 次(每个 dtype bin 配一个 int32),否则框架丢弃 dtype 字段。
static const std::vector<ge::DataType> indexDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
static const std::vector<ge::DataType> countDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
static const std::vector<ge::Format> nzvFormat12 = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class NonZeroWithValue : public OpDef {
public:
    explicit NonZeroWithValue(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDataType).Format(nzvFormat12).UnknownShapeFormat(nzvFormat12);
        // 静态 max-size 输出:value=[numel]/index=[2*numel]/count=[1](由 infershape 静态推导);
        // 非数据依赖 shape,故不加 OutputShapeDependOnCompute。
        this->Output("value")
            .ParamType(REQUIRED)
            .DataType(valueDataType)
            .Format(nzvFormat12)
            .UnknownShapeFormat(nzvFormat12);
        this->Output("index")
            .ParamType(REQUIRED)
            .DataType(indexDataType)
            .Format(nzvFormat12)
            .UnknownShapeFormat(nzvFormat12);
        this->Output("count")
            .ParamType(REQUIRED)
            .DataType(countDataType)
            .Format(nzvFormat12)
            .UnknownShapeFormat(nzvFormat12);
        this->Attr("transpose").AttrType(OPTIONAL).Bool(false);
        this->Attr("dtype").AttrType(OPTIONAL).Int(ge::DT_INT32);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "non_zero_with_value");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(NonZeroWithValue);
} // namespace ops
