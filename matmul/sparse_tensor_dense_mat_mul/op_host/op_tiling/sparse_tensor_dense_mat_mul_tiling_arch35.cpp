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
 * \file sparse_tensor_dense_mat_mul_tiling_arch35.cpp
 * \brief
 */
#include <sstream>
#include <string>
#include <algorithm>
#include <unordered_set>
#include "sparse_tensor_dense_mat_mul_tiling.h"

namespace {
constexpr uint64_t TILING_KEY_B16_ADJA_ADJB = 10001;
constexpr uint64_t TILING_KEY_B16_ADJA_NOT_ADJB = 10002;
constexpr uint64_t TILING_KEY_B16_NOT_ADJA_ADJB = 10003;
constexpr uint64_t TILING_KEY_B16_NOT_ADJA_NOT_ADJB = 10004;
constexpr uint64_t TILING_KEY_B32_ADJA_ADJB = 20001;
constexpr uint64_t TILING_KEY_B32_ADJA_NOT_ADJB = 20002;
constexpr uint64_t TILING_KEY_B32_NOT_ADJA_ADJB = 20003;
constexpr uint64_t TILING_KEY_B32_NOT_ADJA_NOT_ADJB = 20004;
constexpr uint64_t TILING_KEY_EMPTY_Y = 30000;
constexpr uint64_t TILING_KEY_ZEROING_Y = 40000;

constexpr int64_t SIMT_MAX_THREAD_NUM = 2048;           // SIMT每核能启动多少线程，目前定为2048
constexpr int64_t SIMT_DCACHE_SIZE = 65536LL;           // SIMT要占用掉UB中的64k空间作为DCache
constexpr int64_t RESERVED_WORKSPACE_SIZE = 16777216LL; // Workspace为AscendC框架要预留16M空间
constexpr int64_t ONE_CORE_MIN_COPY_BYTES =
    16384LL; // 每个Core至少应该DataCopy多少字节数，小于这个数量直接单核拷贝，这里设置为16k
constexpr int64_t SIZEOF_B32 = sizeof(float);
constexpr int64_t ALIGNED_B32_ELEM_NUM = 32LL / SIZEOF_B32; // 每32字节块，有8个B32元素，用于对齐UB内的数据
constexpr int64_t INT32_MAX_INTEGER = 2147483647LL;
constexpr double INT32_MAX_FLOATING = static_cast<double>(INT32_MAX_INTEGER);

constexpr int64_t OUT_TQUE_NUM = 2;   // Output阶段使用yInQue和yOutQue两个TQue（UB上的Buffer），大小相同
constexpr int64_t OUT_BUFFER_NUM = 2; // Output阶段开启DoubleBuffer

// op inputs index
constexpr size_t IDX_INPUT_X1_INDICES = 0;
constexpr size_t IDX_INPUT_X1_VALUES = 1;
constexpr size_t IDX_INPUT_X1_SHAPE = 2;
constexpr size_t IDX_INPUT_X2 = 3;
// op output index
constexpr size_t IDX_OUTPUT_Y = 0;
// op attr index
constexpr size_t IDX_ATTR_ADJOINT_A = 0;
constexpr size_t IDX_ATTR_ADJOINT_B = 1;
// op inputs ranks required
constexpr int64_t RANKS_X1_INDICES = 2;
constexpr int64_t RANKS_X1_VALUES = 1;
constexpr int64_t RANKS_X1_SHAPE = 1;
constexpr int64_t RANKS_X2 = 2;
constexpr int64_t RANKS_Y = 2;
// op inputs dim required
constexpr int64_t DIM1_X1_INDICES = 2;
constexpr int64_t DIM0_x1_SHAPE = 2;
} // namespace

namespace Ops::NN::Optiling {
class SparseTensorDenseMatMulTilingArch35 : public TilingBaseClass {
public:
    explicit SparseTensorDenseMatMulTilingArch35(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override
    {
        OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::IsCapable()");
        return socVersion_ == platform_ascendc::SocVersion::ASCEND910_95;
    }
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    bool isEmptyOutput() const;
    bool isZeroingOutput() const;
    bool isValuesTypeB16() const; // 输入的Values类型是否为B16（目前仅float16）
    void logTilingInfo() const;   // 用LOGI打印tiling计算的各种值的日志

    ge::graphStatus GetValueDependTensorInfo(); // 获取x_shape这个ValueDepend输入携带的x1的shape信息
    ge::graphStatus CheckInputShapes();         // 校验输入tensor的shape信息是否正确
    ge::graphStatus CheckInputDTypes();
    ge::graphStatus GetAndCheckInputParams(); // 获取nnz、m、n、p等输入的shape信息，并做校验
    ge::graphStatus CheckOutputShapes();      // 校验输出tensor的shape信息是否正确
    ge::graphStatus CheckOutputDTypes();

    void tilingForInit();    // 计算init阶段需要的tilingData
    void tilingForCompute(); // 计算compute阶段需要的tilingData
    void tilingForOutput();  // 计算(B16 only)output阶段需要的tilingData
    void setTilingData();    // 写入tilingData
    void setGlobalBlockDim(); // 计算&&设置整个算子要用到的（最大）核数，即context_->SetBlockDim()里面要传的值

    // PlatformInfos
    platform_ascendc::SocVersion socVersion_{platform_ascendc::SocVersion::RESERVED_VERSION};
    int64_t maxThreadNum_{0};
    int64_t aivCoreNum_{0};
    int64_t ubSize_{0};
    // available ub size for SIMD computing
    int64_t availUbSize_{0};
    // BlockDim for whole kernel
    int64_t globalBlockDim_{0};
    // Input shapes&dtypes
    gert::Shape x1IndicesShape_;
    gert::Shape x1ValuesShape_;
    gert::Shape x1ShapeShape_;
    gert::Shape x2Shape_;
    gert::Shape yShape_;
    ge::DataType x1IndicesDType_{ge::DataType::DT_MAX};
    ge::DataType x1ValuesDType_{ge::DataType::DT_MAX};
    ge::DataType x1ShapeDType_{ge::DataType::DT_MAX};
    ge::DataType x2DType_{ge::DataType::DT_MAX};
    ge::DataType yDType_{ge::DataType::DT_MAX};
    // Attr
    bool adjointA_{false};
    bool adjointB_{false};
    // ValueDepend Params
    int64_t x1Row_{0};
    int64_t x1Col_{0};
    // tiling params
    int64_t nnz_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t p_{0};
    int64_t initAndOutElemNum_{0};
    int64_t initAndOutUsedCoreNum_{0};
    int64_t initAndOutFormerCoreElemNum_{0};
    int64_t initAndOutTailCoreElemNum_{0};
    int64_t computeTotalElemNum_{0};
    int64_t computePerCoreThreadNum_{0};
    int64_t computeUsedCoreNum_{0};
    int64_t outFormerCoreUbFactor_{0};
    int64_t outFormerCoreUbTailFactor_{0};
    int64_t outFormerCoreUbLoopTimes_{0};
    int64_t outTailCoreUbFactor_{0};
    int64_t outTailCoreUbTailFactor_{0};
    int64_t outTailCoreUbLoopTimes_{0};

    // TilingData
    SparseTensorDenseMatMulTilingData* tilingDataPtr_{nullptr};
};

ge::graphStatus SparseTensorDenseMatMulTilingArch35::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::GetShapeAttrsInfo()");

    // x1_indices shape
    auto* inputX1IndicesShapePtr = context_->GetInputShape(IDX_INPUT_X1_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1IndicesShapePtr);
    x1IndicesShape_ = inputX1IndicesShapePtr->GetStorageShape();
    // x1_values shape
    auto* inputX1ValuesShapePtr = context_->GetInputShape(IDX_INPUT_X1_VALUES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1ValuesShapePtr);
    x1ValuesShape_ = inputX1ValuesShapePtr->GetStorageShape();
    // x1_shape shape
    auto* inputX1ShapeShapePtr = context_->GetInputShape(IDX_INPUT_X1_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1ShapeShapePtr);
    x1ShapeShape_ = inputX1ShapeShapePtr->GetStorageShape();
    // x2 shape
    auto* inputX2ShapePtr = context_->GetInputShape(IDX_INPUT_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX2ShapePtr);
    x2Shape_ = inputX2ShapePtr->GetStorageShape();
    // y shape
    auto* outputYShapePtr = context_->GetOutputShape(IDX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYShapePtr);
    yShape_ = outputYShapePtr->GetStorageShape();

    // x1_indices dtype
    auto* inputX1IndicesDescPtr = context_->GetInputDesc(IDX_INPUT_X1_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1IndicesDescPtr);
    x1IndicesDType_ = inputX1IndicesDescPtr->GetDataType();
    // x1_values dtype
    auto* inputX1ValuesDescPtr = context_->GetInputDesc(IDX_INPUT_X1_VALUES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1ValuesDescPtr);
    x1ValuesDType_ = inputX1ValuesDescPtr->GetDataType();
    // x1_shape dtype
    auto* inputX1ShapeDescPtr = context_->GetInputDesc(IDX_INPUT_X1_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1ShapeDescPtr);
    x1ShapeDType_ = inputX1ShapeDescPtr->GetDataType();
    // x2 dtype
    auto* inputX2DescPtr = context_->GetInputDesc(IDX_INPUT_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX2DescPtr);
    x2DType_ = inputX2DescPtr->GetDataType();
    // y dtype
    auto* outputYDescPtr = context_->GetOutputDesc(IDX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDescPtr);
    yDType_ = outputYDescPtr->GetDataType();

    // attrs
    auto* attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);
    // adjointA
    auto* attrAdjointAPtr = attrsPtr->GetBool(IDX_ATTR_ADJOINT_A);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAdjointAPtr);
    adjointA_ = static_cast<bool>(*attrAdjointAPtr);
    // adjointB
    auto* attrAdjointBPtr = attrsPtr->GetBool(IDX_ATTR_ADJOINT_B);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAdjointBPtr);
    adjointB_ = static_cast<bool>(*attrAdjointBPtr);

    // ret
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::GetPlatformInfo()
{
    using namespace platform_ascendc;
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::GetPlatformInfo()");

    // aclnn接口等一些通路取不到compileInfo，因此这里选择每次执行tiling时获取平台信息
    auto* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = PlatformAscendC(platformInfoPtr);
    // socVersion
    socVersion_ = ascendcPlatform.GetSocVersion();
    // maxThreadNum（该平台信息参数仅赋值，目前未使用）
    maxThreadNum_ = SIMT_MAX_THREAD_NUM;
    OP_CHECK_IF(
        maxThreadNum_ <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get platform maxThreadNum."),
        return ge::GRAPH_FAILED);
    // aivCoreNum
    aivCoreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        aivCoreNum_ <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get platform aivCoreNum."),
        return ge::GRAPH_FAILED);
    // ubSize
    uint64_t ubsz;
    ascendcPlatform.GetCoreMemSize(CoreMemType::UB, ubsz);
    ubSize_ = static_cast<int64_t>(ubsz);
    OP_CHECK_IF(
        ubSize_ <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get platform ubSize."), return ge::GRAPH_FAILED);
    // 计算实际可用的UB空间（这里仅扣除给SIMT预留的DCACHE大小）
    availUbSize_ = ubSize_ - SIMT_DCACHE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::DoOpTiling()");

    ge::graphStatus ret;
    // 获取ValueDepend的信息（x1_shape)
    if ((ret = GetValueDependTensorInfo()) != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 校验输入tensor的shape合法性
    if ((ret = CheckInputShapes()) != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 校验输入tensor的dtype合法性
    if ((ret = CheckInputDTypes()) != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 获取&&校验nnz、m、n、p等关键参数
    if ((ret = GetAndCheckInputParams()) != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 校验输出tensor的shape合法性
    if ((ret = CheckOutputShapes()) != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 校验输出tensor的dtype合法性
    if ((ret = CheckOutputDTypes()) != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 计算init阶段分核心，由于output阶段复用init阶段的计算，tilingForInit必须在tilingForOutput之前调用
    tilingForInit();
    tilingForCompute();
    tilingForOutput();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::GetWorkspaceSize()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::GetWorkspaceSize()");

    // 计算需要用的ws大小
    int64_t userWsSize = 0;
    if (!(isEmptyOutput() || isZeroingOutput()) && isValuesTypeB16()) {
        // 如果是空Tensor输出，或是全0输出，则不需要workspace中转，直接对output写0即可
        // 如果是B16模板，预留y.numel()*sizeof(B32)的workspace大小，作为yWorkspace，放临时结果
        userWsSize += m_ * p_ * SIZEOF_B32;
    }
    // 获取写ws大小的指针
    auto* workspacePtr = context_->GetWorkspaceSizes(1); // 只需要1个workspace
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspacePtr);
    // 写入ws大小=需要用的大小+为框架保留大小
    workspacePtr[0] = static_cast<size_t>(userWsSize + RESERVED_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::PostTiling()");

    // 写入tilingData
    setTilingData();
    // 计算&&设置好整个算子用到的核数BlockDim
    setGlobalBlockDim();
    // 设置ub可用大小（这里是ub原始大小减去simt用掉的dcache大小）
    context_->SetLocalMemorySize(availUbSize_);
    // 涉及核间同步的算子必须设置schedule_mode为1
    context_->SetScheduleMode(1);
    // 打印TilingInfo
    logTilingInfo();

    return ge::GRAPH_SUCCESS;
}

uint64_t SparseTensorDenseMatMulTilingArch35::GetTilingKey() const
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::GetTilingKey()");

    if (isEmptyOutput()) {
        // (0,n)*(n,p)->(0,p), (m,n)*(n,0)->(m,0)
        return TILING_KEY_EMPTY_Y;
    } else if (isZeroingOutput()) {
        // (m,0)*(0,p)->(m,p), 里面的元素全为0
        return TILING_KEY_ZEROING_Y;
    }

    uint64_t tilingKey = 0ULL;
    if (isValuesTypeB16()) {
        if (adjointA_ && adjointB_) {
            tilingKey = TILING_KEY_B16_ADJA_ADJB;
        } else if (adjointA_) {
            tilingKey = TILING_KEY_B16_ADJA_NOT_ADJB;
        } else if (adjointB_) {
            tilingKey = TILING_KEY_B16_NOT_ADJA_ADJB;
        } else {
            tilingKey = TILING_KEY_B16_NOT_ADJA_NOT_ADJB;
        }
    } else {
        if (adjointA_ && adjointB_) {
            tilingKey = TILING_KEY_B32_ADJA_ADJB;
        } else if (adjointA_) {
            tilingKey = TILING_KEY_B32_ADJA_NOT_ADJB;
        } else if (adjointB_) {
            tilingKey = TILING_KEY_B32_NOT_ADJA_ADJB;
        } else {
            tilingKey = TILING_KEY_B32_NOT_ADJA_NOT_ADJB;
        }
    }

    return tilingKey;
}

inline bool SparseTensorDenseMatMulTilingArch35::isEmptyOutput() const
{
    return m_ == 0 || p_ == 0;
}

inline bool SparseTensorDenseMatMulTilingArch35::isZeroingOutput() const
{
    return n_ == 0;
}

inline bool SparseTensorDenseMatMulTilingArch35::isValuesTypeB16() const
{
    static const std::unordered_set<ge::DataType> SUPPORTED_B16_VALUES_DTYPES = {ge::DataType::DT_FLOAT16};
    return SUPPORTED_B16_VALUES_DTYPES.count(x1ValuesDType_) != 0;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::GetValueDependTensorInfo()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::GetValueDependTensorInfo()");

    auto* tensorX1ShapePtr = context_->GetInputTensor(IDX_INPUT_X1_SHAPE);
    OP_CHECK_IF(
        tensorX1ShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "Failed to get ValueDepend input x1_shape"),
        return ge::GRAPH_FAILED);
    const int64_t* x1ShapeData = tensorX1ShapePtr->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1ShapeData);
    x1Row_ = static_cast<int64_t>(*x1ShapeData);
    x1Col_ = static_cast<int64_t>(*(x1ShapeData + 1));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::CheckInputShapes()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::CheckInputShapes()");

    // x1_indices
    auto x1IndicesRanks = static_cast<int64_t>(x1IndicesShape_.GetDimNum());
    OP_CHECK_IF(
        x1IndicesRanks != RANKS_X1_INDICES,
        OP_LOGE(
            context_->GetNodeName(), "Invalid ranks of input x1_indices: %lld, should be: %lld.", x1IndicesRanks,
            RANKS_X1_INDICES),
        return ge::GRAPH_FAILED);
    auto x1IndicesDim0 = static_cast<int64_t>(x1IndicesShape_.GetDim(0));
    auto x1IndicesDim1 = static_cast<int64_t>(x1IndicesShape_.GetDim(1));
    OP_CHECK_IF(
        x1IndicesDim1 != DIM1_X1_INDICES,
        OP_LOGE(
            context_->GetNodeName(), "Invalid length of input x1_indices' dim1: %lld, should be: %lld.", x1IndicesDim1,
            DIM1_X1_INDICES),
        return ge::GRAPH_FAILED);

    // x1_values
    auto x1ValuesRanks = static_cast<int64_t>(x1ValuesShape_.GetDimNum());
    OP_CHECK_IF(
        x1ValuesRanks != RANKS_X1_VALUES,
        OP_LOGE(
            context_->GetNodeName(), "Invalid ranks of input x1_values: %lld, should be: %lld.", x1ValuesRanks,
            RANKS_X1_VALUES),
        return ge::GRAPH_FAILED);
    auto x1valuesDim0 = static_cast<int64_t>(x1ValuesShape_.GetDim(0));
    OP_CHECK_IF(
        x1valuesDim0 != x1IndicesDim0,
        OP_LOGE(
            context_->GetNodeName(),
            "Mismatch length of input x1_values' dim0: %lld, should equals to x1_indices' dim0: %lld.", x1valuesDim0,
            x1IndicesDim0),
        return ge::GRAPH_FAILED);

    // x1_shape
    auto x1ShapeRanks = static_cast<int64_t>(x1ShapeShape_.GetDimNum());
    OP_CHECK_IF(
        x1ShapeRanks != RANKS_X1_SHAPE,
        OP_LOGE(
            context_->GetNodeName(), "Invalid ranks of input x1_shape: %lld, should be: %lld.", x1ShapeRanks,
            RANKS_X1_SHAPE),
        return ge::GRAPH_FAILED);
    auto x1ShapeDim0 = static_cast<int64_t>(x1ShapeShape_.GetDim(0));
    OP_CHECK_IF(
        x1ShapeDim0 != DIM0_x1_SHAPE,
        OP_LOGE(
            context_->GetNodeName(), "Invalid length of input x1_shape's dim0: %lld, should be: %lld.", x1ShapeDim0,
            DIM0_x1_SHAPE),
        return ge::GRAPH_FAILED);

    // x2
    auto x2Ranks = static_cast<int64_t>(x2Shape_.GetDimNum());
    OP_CHECK_IF(
        x2Ranks != RANKS_X2,
        OP_LOGE(context_->GetNodeName(), "Invalid ranks of input x2: %lld, should be: %lld.", x2Ranks, RANKS_X2),
        return ge::GRAPH_FAILED);
    // 对x2的dim0、dim1的check延后到GetAndCheckInputParams，因为要先取得m、n的值

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::CheckInputDTypes()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::CheckInputDTypes()");

    static const std::unordered_set<ge::DataType> SUPPORTED_INDICES_DTYPES = {
        ge::DataType::DT_INT32, ge::DataType::DT_INT64};
    static const std::unordered_set<ge::DataType> SUPPORTED_VALUES_DTYPES = {
        ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_INT32};

    // x1_indices
    OP_CHECK_IF(
        SUPPORTED_INDICES_DTYPES.count(x1IndicesDType_) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Unsupported dtype of input x1_indices: %d, currently only supports: %d(DT_INT32), %d(DT_INT64).",
            x1IndicesDType_, ge::DataType::DT_INT32, ge::DataType::DT_INT64),
        return ge::GRAPH_FAILED);

    // x1_values
    OP_CHECK_IF(
        SUPPORTED_VALUES_DTYPES.count(x1ValuesDType_) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Unsupported dtype of input x1_values: %d, currently only supports: %d(DT_FLOAT16), %d(DT_FLOAT), "
            "%d(DT_INT32).",
            x1ValuesDType_, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_INT32),
        return ge::GRAPH_FAILED);

    // x1_shape
    OP_CHECK_IF(
        x1ShapeDType_ != ge::DataType::DT_INT64,
        OP_LOGE(
            context_->GetNodeName(), "Unsupported dtype of input x1_shape: %d, currently only supports: %d(DT_INT64).",
            x1ShapeDType_, ge::DataType::DT_INT64),
        return ge::GRAPH_FAILED);

    // x2
    OP_CHECK_IF(
        x2DType_ != x1ValuesDType_,
        OP_LOGE(
            context_->GetNodeName(), "Mismatch dtype of input x2: %d, should equals to dtype of x1_values: %d.",
            x2DType_, x1ValuesDType_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::GetAndCheckInputParams()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::GetAndCheckInputParams()");

    nnz_ = static_cast<int64_t>(x1IndicesShape_.GetDim(0));
    m_ = adjointA_ ? x1Col_ : x1Row_;
    n_ = adjointA_ ? x1Row_ : x1Col_;
    auto x2Row = static_cast<int64_t>(x2Shape_.GetDim(0));
    auto x2Col = static_cast<int64_t>(x2Shape_.GetDim(1));
    if (adjointB_) {
        OP_CHECK_IF(
            x2Col != n_,
            OP_LOGE(
                context_->GetNodeName(), "Invalid col(dim1) of input x2: %lld, should equals to n: %lld", x2Col, n_),
            return ge::GRAPH_FAILED);
        p_ = static_cast<int64_t>(x2Row);
    } else {
        OP_CHECK_IF(
            x2Row != n_,
            OP_LOGE(
                context_->GetNodeName(), "Mismatch row(dim0) of input x2: %lld, should equals to n: %lld", x2Row, n_),
            return ge::GRAPH_FAILED);
        p_ = static_cast<int64_t>(x2Col);
    }

    OP_CHECK_IF(
        (m_ < 0 || n_ < 0 || p_ < 0 || nnz_ < 0),
        OP_LOGE(
            context_->GetNodeName(), "Some of params (m: %lld, n: %lld, p:%lld, nnz:%lld) belows 0.", m_, n_, p_, nnz_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (m_ >= INT32_MAX_INTEGER || n_ >= INT32_MAX_INTEGER || p_ >= INT32_MAX_INTEGER || nnz_ >= INT32_MAX_INTEGER),
        OP_LOGE(
            context_->GetNodeName(), "Some of params (m: %lld, n: %lld, p:%lld, nnz:%lld) exceeds INT32_MAX.", m_, n_,
            p_, nnz_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        p_ > 0 &&
            ((m_ >= INT32_MAX_FLOATING / p_) || (n_ >= INT32_MAX_FLOATING / p_) || (nnz_ >= INT32_MAX_INTEGER / p_)),
        OP_LOGE(
            context_->GetNodeName(),
            "Some multiplications (m*p, n*p, nnz*p) of params (m: %lld, n: %lld, p:%lld, nnz:%lld) is calculated "
            "exceeds INT32_MAX.",
            m_, n_, p_, nnz_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((m_ > 0 && static_cast<double>(nnz_) / m_ > n_) || (n_ > 0 && static_cast<double>(nnz_) / n_ > m_)),
        OP_LOGE(
            context_->GetNodeName(), "Param nnz: %lld should not be greater than m * n (%lld * %lld).", nnz_, m_, n_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::CheckOutputShapes()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::CheckOutputShapes()");

    // y ranks
    auto yRanks = static_cast<int64_t>(yShape_.GetDimNum());
    OP_CHECK_IF(
        yRanks != RANKS_Y,
        OP_LOGE(context_->GetNodeName(), "Invalid ranks of output y: %lld, should be: %lld.", yRanks, RANKS_Y),
        return ge::GRAPH_FAILED);

    // y dims
    auto yDim0 = static_cast<int64_t>(yShape_.GetDim(0));
    OP_CHECK_IF(
        yDim0 != m_,
        OP_LOGE(context_->GetNodeName(), "Mismatch row(dim0) of output y: %lld, should equals to m: %lld.", yDim0, m_),
        return ge::GRAPH_FAILED);

    auto yDim1 = static_cast<int64_t>(yShape_.GetDim(1));
    OP_CHECK_IF(
        yDim1 != p_,
        OP_LOGE(context_->GetNodeName(), "Mismatch col(dim1) of output y: %lld, should equals to p: %lld.", yDim1, p_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseTensorDenseMatMulTilingArch35::CheckOutputDTypes()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::CheckOutputDTypes()");

    OP_CHECK_IF(
        yDType_ != x1ValuesDType_,
        OP_LOGE(
            context_->GetNodeName(), "Mismatch dtype of output y: %d, should equals to dtype of x1_values: %d.",
            yDType_, x1ValuesDType_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SparseTensorDenseMatMulTilingArch35::tilingForInit()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::tilingForInit()");

    // 计算init/output阶段要处理的元素个数，也就是y的元素个数
    initAndOutElemNum_ = m_ * p_;
    if (initAndOutElemNum_ == 0) {
        // 若元素个数为零，那么用到的核数也为0，分配到每个核的元素个数也为0，直接结束计算
        initAndOutUsedCoreNum_ = 0;
        initAndOutFormerCoreElemNum_ = 0;
        initAndOutTailCoreElemNum_ = 0;
        return;
    }

    // 计算使用核心数
    int64_t coreMinElemNum = Ops::Base::CeilDiv(ONE_CORE_MIN_COPY_BYTES, SIZEOF_B32);
    initAndOutUsedCoreNum_ = std::min<int64_t>(Ops::Base::CeilDiv(initAndOutElemNum_, coreMinElemNum), aivCoreNum_);

    // 计算每核处理的元素数
    initAndOutFormerCoreElemNum_ = Ops::Base::CeilDiv(initAndOutElemNum_, initAndOutUsedCoreNum_);
    initAndOutTailCoreElemNum_ = initAndOutElemNum_ - (initAndOutUsedCoreNum_ - 1) * initAndOutFormerCoreElemNum_;
}

void SparseTensorDenseMatMulTilingArch35::tilingForCompute()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::tilingForCompute()");

    // 用元素总数，每核线程数，计算需要启动多少核（SIMT计算用）
    computeTotalElemNum_ = nnz_ * p_;
    computePerCoreThreadNum_ = SIMT_MAX_THREAD_NUM;
    computeUsedCoreNum_ =
        std::min<int64_t>(Ops::Base::CeilDiv(computeTotalElemNum_, computePerCoreThreadNum_), aivCoreNum_);
}

void SparseTensorDenseMatMulTilingArch35::tilingForOutput()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::tilingForOutput()");

    if (initAndOutElemNum_ == 0) {
        // 当init/output阶段的数据量为0，相关tilingData全部为0，无需再计算
        outFormerCoreUbFactor_ = 0;
        outFormerCoreUbLoopTimes_ = 0;
        outFormerCoreUbTailFactor_ = 0;
        outTailCoreUbFactor_ = 0;
        outTailCoreUbLoopTimes_ = 0;
        outTailCoreUbTailFactor_ = 0;
        return;
    }

    // 计算一次ubLoop能处理多少元素：可用ub大小/每个元素大小（32位）/TQUE数目（2个，in和out）/每个TQue的Buffer数目（DB=2）/
    // 元素数目对齐到32字节（这里是FloorDiv）
    int64_t oneUbLoopMaxElemNum =
        availUbSize_ / SIZEOF_B32 / OUT_TQUE_NUM / OUT_BUFFER_NUM / ALIGNED_B32_ELEM_NUM * ALIGNED_B32_ELEM_NUM;

    // 计算非尾核的tiling
    outFormerCoreUbFactor_ = std::min(
        initAndOutFormerCoreElemNum_,
        oneUbLoopMaxElemNum); // 非尾核心一次Loop要么处理所有元素，要么能处理多少处理多少
    outFormerCoreUbLoopTimes_ = Ops::Base::CeilDiv(initAndOutFormerCoreElemNum_, outFormerCoreUbFactor_);
    outFormerCoreUbTailFactor_ =
        initAndOutFormerCoreElemNum_ - (outFormerCoreUbLoopTimes_ - 1) * outFormerCoreUbFactor_;

    // 计算尾核的tiling
    outTailCoreUbFactor_ = std::min(
        initAndOutTailCoreElemNum_,
        oneUbLoopMaxElemNum); // 非尾核心一次Loop要么处理所有元素，要么能处理多少处理多少
    outTailCoreUbLoopTimes_ = Ops::Base::CeilDiv(initAndOutTailCoreElemNum_, outTailCoreUbFactor_);
    outTailCoreUbTailFactor_ = initAndOutTailCoreElemNum_ - (outTailCoreUbLoopTimes_ - 1) * outTailCoreUbFactor_;
}

void SparseTensorDenseMatMulTilingArch35::setTilingData()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::setTilingData()");

    tilingDataPtr_ = context_->GetTilingData<SparseTensorDenseMatMulTilingData>();
    tilingDataPtr_->initAndOutUsedCoreNum = initAndOutUsedCoreNum_;
    tilingDataPtr_->initAndOutFormerCoreElemNum = initAndOutFormerCoreElemNum_;
    tilingDataPtr_->initAndOutTailCoreElemNum = initAndOutTailCoreElemNum_;
    tilingDataPtr_->computeUsedCoreNum = computeUsedCoreNum_;
    tilingDataPtr_->computePerCoreThreadNum = computePerCoreThreadNum_;
    tilingDataPtr_->computeTotalElemNum = computeTotalElemNum_;
    tilingDataPtr_->computeNnz = nnz_;
    tilingDataPtr_->computeM = m_;
    tilingDataPtr_->computeN = n_;
    tilingDataPtr_->computeP = p_;
    tilingDataPtr_->outFormerCoreUbFactor = outFormerCoreUbFactor_;
    tilingDataPtr_->outFormerCoreUbTailFactor = outFormerCoreUbTailFactor_;
    tilingDataPtr_->outFormerCoreUbLoopTimes = outFormerCoreUbLoopTimes_;
    tilingDataPtr_->outTailCoreUbFactor = outTailCoreUbFactor_;
    tilingDataPtr_->outTailCoreUbTailFactor = outTailCoreUbTailFactor_;
    tilingDataPtr_->outTailCoreUbLoopTimes = outTailCoreUbLoopTimes_;
}

void SparseTensorDenseMatMulTilingArch35::setGlobalBlockDim()
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::setGlobalBlockDim()");

    // 取init/compute/out阶段用到的核数的最大值（这些最大值已经设置为小于等于aivCoreNum_了）
    globalBlockDim_ = std::max<int64_t>(initAndOutUsedCoreNum_, computeUsedCoreNum_);
    globalBlockDim_ = std::max<int64_t>(globalBlockDim_, 1LL); // 如果每个阶段都只用0个核的话，也至少启动1个核
    context_->SetBlockDim(globalBlockDim_);
}

void SparseTensorDenseMatMulTilingArch35::logTilingInfo() const
{
    OP_LOGD(context_->GetNodeName(), "In SparseTensorDenseMatMulTilingArch35::logTilingInfo()");

    std::stringstream ss;
    ss << "\n[PlatformInfo]\n";
    ss << "socVersion = " << static_cast<int>(socVersion_) << "\n";
    ss << "maxThreadNum = " << maxThreadNum_ << "\n";
    ss << "aivCoreNum = " << aivCoreNum_ << "\n";
    ss << "ubSize = " << ubSize_ << "\n";
    ss << "availUbSize = " << availUbSize_ << "\n";
    ss << "globalBlockDim = " << globalBlockDim_;
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());

    ss.str("");
    ss.clear();
    ss << "\n[ValueDepend]\n";
    ss << "x1ValuesDType_ = " << x1ValuesDType_ << "\n";
    ss << "x1Row = " << x1Row_ << "\n";
    ss << "x1Col = " << x1Col_;
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());

    ss.str("");
    ss.clear();
    ss << "\n[TilingKey]\n";
    ss << "TilingKey = " << GetTilingKey() << "\n";
    ss << "[TilingData]\n";
    ss << "initAndOutUsedCoreNum = " << tilingDataPtr_->initAndOutUsedCoreNum << "\n";
    ss << "initAndOutFormerCoreElemNum = " << tilingDataPtr_->initAndOutFormerCoreElemNum << "\n";
    ss << "initAndOutTailCoreElemNum = " << tilingDataPtr_->initAndOutTailCoreElemNum << "\n";
    ss << "computeUsedCoreNum = " << tilingDataPtr_->computeUsedCoreNum << "\n";
    ss << "computePerCoreThreadNum = " << tilingDataPtr_->computePerCoreThreadNum << "\n";
    ss << "computeTotalElemNum = " << tilingDataPtr_->computeTotalElemNum << "\n";
    ss << "computeNnz = " << tilingDataPtr_->computeNnz << "\n";
    ss << "computeM = " << tilingDataPtr_->computeM << "\n";
    ss << "computeN = " << tilingDataPtr_->computeN << "\n";
    ss << "computeP = " << tilingDataPtr_->computeP << "\n";
    ss << "outFormerCoreUbFactor = " << tilingDataPtr_->outFormerCoreUbFactor << "\n";
    ss << "outFormerCoreUbTailFactor = " << tilingDataPtr_->outFormerCoreUbTailFactor << "\n";
    ss << "outFormerCoreUbLoopTimes = " << tilingDataPtr_->outFormerCoreUbLoopTimes << "\n";
    ss << "outTailCoreUbFactor = " << tilingDataPtr_->outTailCoreUbFactor << "\n";
    ss << "outTailCoreUbTailFactor = " << tilingDataPtr_->outTailCoreUbTailFactor << "\n";
    ss << "outTailCoreUbLoopTimes = " << tilingDataPtr_->outTailCoreUbLoopTimes;
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(
    SparseTensorDenseMatMul, SparseTensorDenseMatMulTilingArch35, SPARSE_TENSOR_DENSE_MAT_MUL_TILING_ARCH35_PRIORITY);

} // namespace Ops::NN::Optiling
