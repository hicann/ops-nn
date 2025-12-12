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
 * \file foreach_regbase_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
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
        if (socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }
        return true;
    }
    // 检查output shape
    ge::graphStatus CheckOutput();
    // check scalar shape and dtype
    ge::graphStatus CheckScalar(int64_t scalarIdx);
    ge::graphStatus CheckScalarList(int64_t scalarIdx);
    ge::graphStatus CheckScalarListInt(int64_t scalarIdx);
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;

    ge::DataType dataType_ = ge::DT_UNDEFINED;
    ge::DataType scalarDtype_ = ge::DT_UNDEFINED;
    int64_t blockDim_ = 0;
    int64_t tensorDataCountList_[MAX_TENSOR_CONT_910D] = {0};
    uint16_t tensorStartList_[MAX_CORE_CONT_910D] = {0};
    uint16_t tensorEndList_[MAX_CORE_CONT_910D] = {0};
    int64_t tensorStartOffsetList_[MAX_CORE_CONT_910D] = {0};
    int64_t tensorEndOffsetList_[MAX_CORE_CONT_910D] = {0};
    int64_t totalDataCount_ = 0;
    uint16_t totalTensorCount_ = 0;
    ForeachSoloTilingDataRegbase foreachSoloTilingData_;

private:
    void AssignDataToEachCore(int64_t needCoreNum, int64_t elementsPerBlock);
    ge::graphStatus CheckShapeAllPositive(const gert::Shape& shape, uint32_t idx);
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

private:
    ge::graphStatus CheckContext();
    ge::graphStatus CheckShape(uint32_t idx);
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