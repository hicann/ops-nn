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
 * \file foreach_regbase_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "foreach_tiling_common.h"
#include "foreach_tiling_def.h"

namespace optiling {
class ForeachRegbaseTiling : public ForeachBaseClass
{
public:
    explicit ForeachRegbaseTiling(gert::TilingContext* context) : ForeachBaseClass(context)
    {}
    ~ForeachRegbaseTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        return Ops::NN::OpTiling::IsRegbaseSocVersion(context_);
    }
    // 检查output shape
    ge::graphStatus CheckOutput();
    // Finalize UB split: given per-element UB bytes, compute and store inputsTensorUbSize.
    // Returns GRAPH_FAILED if a divisor (ubSizePerNumber / sizePerElem) is non-positive.
    ge::graphStatus SetInputsTensorUbSize(int64_t ubSizePerNumber);
    // check scalar shape and dtype
    ge::graphStatus CheckScalar(int64_t scalarIdx);
    ge::graphStatus CheckScalarList(int64_t scalarIdx);
    ge::graphStatus CheckScalarListInt(int64_t scalarIdx);
    ge::graphStatus CheckScalarListSameDtype(int64_t scalarIdx);
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;

    ge::DataType dataType_ = ge::DT_UNDEFINED;
    ge::DataType scalarDtype_ = ge::DT_UNDEFINED;
    int64_t numBlocks_ = 0;
    int64_t tensorDataCountList_[MAX_TENSOR_CONT_950] = {0};
    uint16_t tensorStartList_[MAX_CORE_CONT_950] = {0};
    uint16_t tensorEndList_[MAX_CORE_CONT_950] = {0};
    int64_t tensorStartOffsetList_[MAX_CORE_CONT_950] = {0};
    int64_t tensorEndOffsetList_[MAX_CORE_CONT_950] = {0};
    int64_t totalDataCount_ = 0;
    uint16_t totalTensorCount_ = 0;
    ForeachSoloTilingDataRegbase foreachSoloTilingData_;

private:
    void AssignDataToEachCore(int64_t needCoreNum, int64_t elementsPerBlock);
    ge::graphStatus CheckShapeAllPositive(const gert::Shape& shape, uint32_t idx);
    const char* GetFirstTensorName() const;
};

class ForeachRegbaseTilingUnaryScalar : public ForeachRegbaseTiling
{
public:
    explicit ForeachRegbaseTilingUnaryScalar(gert::TilingContext* context) : ForeachRegbaseTiling(context)
    {}
    ~ForeachRegbaseTilingUnaryScalar() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTiling::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
};

class ForeachRegbaseTilingTernaryScalar : public ForeachRegbaseTiling
{
public:
    explicit ForeachRegbaseTilingTernaryScalar(gert::TilingContext* context) : ForeachRegbaseTiling(context)
    {}
    ~ForeachRegbaseTilingTernaryScalar() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTiling::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // scalar 校验：单标量用 CheckScalar；per-tensor 标量列表的派生类重写为 CheckScalarList
    virtual ge::graphStatus CheckScalarParam();

private:
    ge::graphStatus CheckContext();
    ge::graphStatus CheckShape(uint32_t idx);
};

// ternary + per-tensor 标量列表（addcmul_list/addcdiv_list）：复用 TernaryScalar 全部逻辑，仅标量校验换成列表校验
class ForeachRegbaseTilingTernaryScalarList : public ForeachRegbaseTilingTernaryScalar
{
public:
    explicit ForeachRegbaseTilingTernaryScalarList(gert::TilingContext* context)
        : ForeachRegbaseTilingTernaryScalar(context)
    {}
    ~ForeachRegbaseTilingTernaryScalarList() override = default;

protected:
    ge::graphStatus CheckScalarParam() override;
};

class ForeachRegbaseTilingBinaryScalar : public ForeachRegbaseTiling
{
public:
    explicit ForeachRegbaseTilingBinaryScalar(gert::TilingContext* context) : ForeachRegbaseTiling(context)
    {}
    ~ForeachRegbaseTilingBinaryScalar() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTiling::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus CheckContext();
    ge::graphStatus CheckShape(uint32_t idx);
    // check scalar shape and dtype
    ge::graphStatus CheckScalar();
};

// Two tensor-list inputs (x1, x2), no scalar. Used by in-place binary foreach ops
// (mul_list/div_list). UB budget accounts for 2 inputs + 1 output.
class ForeachRegbaseTilingBinary : public ForeachRegbaseTiling
{
public:
    explicit ForeachRegbaseTilingBinary(gert::TilingContext* context) : ForeachRegbaseTiling(context)
    {}
    ~ForeachRegbaseTilingBinary() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTiling::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
};

class ForeachRegbaseTilingUnaryScalarList : public ForeachRegbaseTiling
{
public:
    explicit ForeachRegbaseTilingUnaryScalarList(gert::TilingContext* context) : ForeachRegbaseTiling(context)
    {}
    ~ForeachRegbaseTilingUnaryScalarList() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTiling::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus CheckContext();
    ge::graphStatus CheckShape(uint32_t idx);
};

class ForeachRegbaseTilingUnaryScalarList2 : public ForeachRegbaseTiling
{
public:
    explicit ForeachRegbaseTilingUnaryScalarList2(gert::TilingContext* context) : ForeachRegbaseTiling(context)
    {}
    ~ForeachRegbaseTilingUnaryScalarList2() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTiling::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus CheckScalarList(int64_t scalarIdx);
};

class ForeachRegbaseTilingUnaryScalarListWithInt : public ForeachRegbaseTilingUnaryScalarList
{
public:
    explicit ForeachRegbaseTilingUnaryScalarListWithInt(gert::TilingContext* context)
        : ForeachRegbaseTilingUnaryScalarList(context)
    {}
    ~ForeachRegbaseTilingUnaryScalarListWithInt() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTilingUnaryScalarList::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
};

class ForeachRegbaseTilingUnary : public ForeachRegbaseTilingUnaryScalar
{
public:
    explicit ForeachRegbaseTilingUnary(gert::TilingContext* context) : ForeachRegbaseTilingUnaryScalar(context)
    {}
    ~ForeachRegbaseTilingUnary() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachRegbaseTiling::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_