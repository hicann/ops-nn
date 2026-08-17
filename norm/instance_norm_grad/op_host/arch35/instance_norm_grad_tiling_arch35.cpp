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
 * \file instance_norm_grad_tiling_arch35.cpp
 * \brief RegBase (arch35) tiling for InstanceNormGrad + IMPL_OP_OPTILING dispatch.
 *
 * Geometry: physical [N,D,H,W,C] viewed as logical [N, M, C] (M = D*H*W). Reduce over M keeping C.
 * Core task granularity = (n, cTileIdx): one instance's full-M column reduction over a C sub-range.
 * Cross-N reduction for pd_gamma/pd_beta uses workspace + SyncAll + deterministic stage2.
 */
#include "instance_norm_grad_tiling_arch35.h"
#include "instance_norm_grad_empty_tiling_arch35.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace {
constexpr uint16_t INPUT_IDX_DY = 0;
constexpr uint16_t INPUT_IDX_X = 1;
constexpr uint16_t INPUT_IDX_VARIANCE = 2;
constexpr uint16_t INPUT_IDX_MEAN = 3;
constexpr uint16_t INPUT_IDX_GAMMA = 4;
constexpr uint16_t OUTPUT_IDX_PDX = 0;
constexpr uint16_t OUTPUT_IDX_PDGAMMA = 1;
constexpr uint16_t OUTPUT_IDX_PDBETA = 2;
constexpr uint16_t DIM0 = 0;
constexpr uint16_t MIN_X_DIM = 2;
constexpr uint8_t SCHEDULE_MODE = 1;

enum class INGDtypeKey : int { FLOAT = 1, HALF = 2 };
} // namespace

ge::graphStatus InstanceNormGradRegBaseTiling::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<InstanceNormGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(context_->GetNodeName(), "ubSize should be greater than zero."),
                return ge::GRAPH_FAILED);
    coreNum_ = compileInfo->totalCoreNum;
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(context_->GetNodeName(), "core num should be greater than zero."),
                return ge::GRAPH_FAILED);
    vectorLen_ = compileInfo->vectorLen / sizeof(float);
    OP_CHECK_IF((vectorLen_ <= 0), OP_LOGE(context_->GetNodeName(), "vectorLen should be greater than zero."),
                return ge::GRAPH_FAILED);
    blockSize_ = compileInfo->blockSize;
    OP_CHECK_IF((blockSize_ <= 0), OP_LOGE(context_->GetNodeName(), "blockSize should be greater than zero."),
                return ge::GRAPH_FAILED);
    sysWorkspaceSize_ = compileInfo->sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

uint32_t InstanceNormGradRegBaseTiling::GetTypeSize(ge::DataType dtypeStr) const
{
    switch (dtypeStr) {
        case ge::DT_FLOAT:
            return FLOAT_DTYPE_BYTES;
        case ge::DT_FLOAT16:
            return FLOAT16_DTYPE_BYTES;
        default:
            return 0;
    }
}

ge::graphStatus InstanceNormGradRegBaseTiling::GetShapeAttrsInfo()
{
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dtype_ = dyDesc->GetDataType();
    auto dyShapePtr = context_->GetInputShape(INPUT_IDX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShapePtr);
    auto dyShape = dyShapePtr->GetStorageShape();
    auto dimNum = dyShape.GetDimNum();
    OP_CHECK_IF(
        (dimNum < MIN_X_DIM),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyShape).c_str(),
                                              "dy must have at least 2 dims [N, ..., C]"),
        return ge::GRAPH_FAILED);
    N_ = dyShape.GetDim(DIM0);
    C_ = dyShape.GetDim(dimNum - 1);
    M_ = 1;
    for (size_t i = 1; i + 1 < dimNum; i++) {
        M_ *= dyShape.GetDim(i);
    }
    OP_CHECK_IF(
        (N_ <= 0 || C_ <= 0 || M_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyShape).c_str(),
                                              "N, C and M(=D*H*W) of dy must all be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(InputCheck() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Input check failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParamsCheck() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Params check failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验单个张量的 dtype 与 dy 一致（dtype_ 已在调用前取自 dy）。
ge::graphStatus InstanceNormGradRegBaseTiling::CheckTensorDtype(const gert::CompileTimeTensorDesc* desc,
                                                                const char* name) const
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
    OP_CHECK_IF((desc->GetDataType() != dtype_),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), name,
                                          ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(),
                                          ge::TypeUtils::DataTypeToSerialString(dtype_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradRegBaseTiling::InputCheck()
{
    // dtype must be fp16/fp32 and identical for dy/x/pd_x.
    if (GetTypeSize(dtype_) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                                  "float32 or float16");
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    auto pdxDesc = context_->GetOutputDesc(OUTPUT_IDX_PDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pdxDesc);
    ge::DataType xDtype = xDesc->GetDataType();
    ge::DataType pdxDtype = pdxDesc->GetDataType();
    OP_CHECK_IF((dtype_ != xDtype || xDtype != pdxDtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "dy, x and pd_x",
                                                       (ge::TypeUtils::DataTypeToSerialString(dtype_) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(xDtype) + " and " +
                                                        ge::TypeUtils::DataTypeToSerialString(pdxDtype))
                                                           .c_str(),
                                                       "The dtypes of dy, x and pd_x must be the same"),
                return ge::GRAPH_FAILED);

    // variance/mean/gamma 与 pd_gamma/pd_beta 也必须同 dtype：kernel 按 dy 的 dtype 强转读写这些张量，
    // dtype 不一致会静默读错数据。
    const std::vector<std::pair<const gert::CompileTimeTensorDesc*, const char*>> sameDtypeTensors = {
        {context_->GetInputDesc(INPUT_IDX_VARIANCE), "variance"},
        {context_->GetInputDesc(INPUT_IDX_MEAN), "mean"},
        {context_->GetInputDesc(INPUT_IDX_GAMMA), "gamma"},
        {context_->GetOutputDesc(OUTPUT_IDX_PDGAMMA), "pd_gamma"},
        {context_->GetOutputDesc(OUTPUT_IDX_PDBETA), "pd_beta"}};
    for (const auto& [desc, name] : sameDtypeTensors) {
        OP_CHECK_IF(CheckTensorDtype(desc, name) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Tensor dtype check failed."), return ge::GRAPH_FAILED);
    }
    auto xShapePtr = context_->GetInputShape(INPUT_IDX_X);
    auto pdxShapePtr = context_->GetOutputShape(OUTPUT_IDX_PDX);
    auto dyShapePtr = context_->GetInputShape(INPUT_IDX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pdxShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto pdxShape = pdxShapePtr->GetStorageShape();
    auto dyShape = dyShapePtr->GetStorageShape();
    OP_CHECK_IF(
        (dyShape != xShape || xShape != pdxShape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy, x and pd_x",
                                               (Ops::Base::ToString(dyShape) + ", " + Ops::Base::ToString(xShape) +
                                                " and " + Ops::Base::ToString(pdxShape))
                                                   .c_str(),
                                               "The shapes of dy, x and pd_x must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradRegBaseTiling::ParamsCheck()
{
    auto gammaShapePtr = context_->GetInputShape(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    int64_t gammaSize = 1;
    for (uint32_t i = 0; i < gammaShape.GetDimNum(); i++) {
        gammaSize *= gammaShape.GetDim(i);
    }
    OP_CHECK_IF(
        (gammaSize != C_),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "gamma", Ops::Base::ToString(gammaShape).c_str(),
            ("The size of gamma must equal the C axis (last dim of dy), where C = " + std::to_string(C_)).c_str()),
        return ge::GRAPH_FAILED);

    auto varShapePtr = context_->GetInputShape(INPUT_IDX_VARIANCE);
    auto meanShapePtr = context_->GetInputShape(INPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanShapePtr);
    auto varShape = varShapePtr->GetStorageShape();
    auto meanShape = meanShapePtr->GetStorageShape();
    int64_t varSize = 1;
    int64_t meanSize = 1;
    for (uint32_t i = 0; i < varShape.GetDimNum(); i++) {
        varSize *= varShape.GetDim(i);
    }
    for (uint32_t i = 0; i < meanShape.GetDimNum(); i++) {
        meanSize *= meanShape.GetDim(i);
    }
    OP_CHECK_IF((varSize != N_ * C_ || meanSize != N_ * C_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "variance/mean", Ops::Base::ToString(varShape).c_str(),
                    ("The size of variance and mean must equal N*C (spatial dims are 1), where N*C = " +
                     std::to_string(N_ * C_))
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool InstanceNormGradRegBaseTiling::IsCapable() { return true; }

ge::graphStatus InstanceNormGradRegBaseTiling::BlockTiling()
{
    tTypeBytes_ = GetTypeSize(dtype_);
    OP_CHECK_IF(tTypeBytes_ == 0, OP_LOGE(context_->GetNodeName(), "tTypeBytes_ is zero"), return ge::GRAPH_FAILED);

    // Choose the C-tile so (1) when N is small we split C to occupy idle cores, and
    // (2) the per-task fp32 accumulators (PARAM_BUFFERS length-cTile vectors) fit UB.
    cTileNum_ = 1;
    if (N_ < static_cast<int64_t>(coreNum_)) {
        cTileNum_ = std::min<int64_t>((static_cast<int64_t>(coreNum_) + N_ - 1) / N_, C_);
    }
    cTile_ = Ops::Base::CeilDiv(C_, cTileNum_);

    // 收缩 cTile 直到「参数区 + 一行流水」装得下 UB;记账口径与内核实际分配严格一致。
    while (cTileNum_ <= C_) {
        int64_t oneRowFlow = FlowTileBytes(cTile_, 1) * UB_COPIES_3 * DOUBLE_BUFFER;
        if (Stage1ParamBytes(cTile_) + oneRowFlow <= static_cast<int64_t>(ubSize_) || cTileNum_ >= C_) {
            break;
        }
        cTileNum_ += 1;
        cTile_ = Ops::Base::CeilDiv(C_, cTileNum_);
    }
    cTileNum_ = Ops::Base::CeilDiv(C_, cTile_);
    taskNum_ = N_ * cTileNum_;

    taskNumPerCore_ = Ops::Base::CeilAlign(taskNum_, static_cast<int64_t>(coreNum_)) / coreNum_;
    OP_CHECK_IF(taskNumPerCore_ == 0, OP_LOGE(context_->GetNodeName(), "taskNumPerCore_ is zero"),
                return ge::GRAPH_FAILED);
    stage1CoreUsed_ = (taskNum_ - 1) / taskNumPerCore_ + 1;
    taskNumPerTailCore_ = taskNumPerCore_;
    tailCore_ = stage1CoreUsed_;
    if (taskNum_ % stage1CoreUsed_ != 0) {
        taskNumPerTailCore_ = taskNumPerCore_ - 1;
        tailCore_ = taskNum_ % stage1CoreUsed_;
    }
    return ge::GRAPH_SUCCESS;
}

// stage1 参数区的实际占用。与 op_kernel/arch35/instance_norm_grad_base.h 的 InitStage1Buffers 一一对应:
//   PARAM_BUFFERS 个 fp32 缓冲 + 1 个输入 dtype 的临时缓冲,长度均为 C 对齐到【向量长度】后的值。
// 注意对齐粒度是 vectorLen_ 而非 blockSize:内核用 CeilAlign(cTile, VL_FP32),按 block 对齐会少算。
int64_t InstanceNormGradRegBaseTiling::Stage1ParamBytes(int64_t cTile) const
{
    const int64_t cAlign = Ops::Base::CeilAlign(cTile, static_cast<int64_t>(vectorLen_));
    // tmpParamBuf 各 dtype 均分配:除低精度载入暂存外,N==1 时的 f32->T 落盘(WritePartialOrOutput)也用它。
    const int64_t tmpBytes = cAlign * static_cast<int64_t>(tTypeBytes_);
    return static_cast<int64_t>(PARAM_BUFFERS) * cAlign * FLOAT_DTYPE_BYTES + tmpBytes;
}

// 单个流水缓冲(x/dy/pd_x 之一,未计双缓冲)的实际占用,对应内核的
//   tileBytes = (mUbTile * rowStrideMaxT + VL_FP32) * sizeof(T)
// 末尾那一个向量长度是内核的补边,必须一并计入,否则记账偏小。
int64_t InstanceNormGradRegBaseTiling::FlowTileBytes(int64_t cTile, int64_t mRows) const
{
    const int64_t rowStrideT = Ops::Base::CeilAlign(cTile * static_cast<int64_t>(tTypeBytes_),
                                                    static_cast<int64_t>(blockSize_)) /
                               static_cast<int64_t>(tTypeBytes_);
    return (mRows * rowStrideT + static_cast<int64_t>(vectorLen_)) * static_cast<int64_t>(tTypeBytes_);
}

ge::graphStatus InstanceNormGradRegBaseTiling::UbTiling()
{
    const int64_t paramBytes = Stage1ParamBytes(cTile_);
    OP_CHECK_IF(static_cast<int64_t>(ubSize_) <= paramBytes,
                OP_LOGE(context_->GetNodeName(), "ubSize less than stage1 param space"), return ge::GRAPH_FAILED);

    // 流水缓冲共 UB_COPIES_3 份、每份双缓冲,故单份预算为剩余 UB 的 1/(3*2)。
    const int64_t perTileBudget = (static_cast<int64_t>(ubSize_) - paramBytes) / (UB_COPIES_3 * DOUBLE_BUFFER);
    const int64_t rowStrideT = Ops::Base::CeilAlign(cTile_ * static_cast<int64_t>(tTypeBytes_),
                                                    static_cast<int64_t>(blockSize_)) /
                               static_cast<int64_t>(tTypeBytes_);
    // C==0(空 tensor 的类别维为零)时 rowStrideT 为 0,流水缓冲不占空间,全部 M 行都装得下,
    // 直接按 full_load 处理;否则由 FlowTileBytes 反解行数:
    //   (rows * rowStrideT + vectorLen_) * tTypeBytes_ <= perTileBudget
    int64_t rowsCap = M_;
    if (rowStrideT > 0) {
        rowsCap = (perTileBudget / static_cast<int64_t>(tTypeBytes_) - static_cast<int64_t>(vectorLen_)) / rowStrideT;
        OP_CHECK_IF(rowsCap <= 0, OP_LOGE(context_->GetNodeName(), "UB too small for one M row, cTile too large"),
                    return ge::GRAPH_FAILED);
    }

    if (rowsCap >= M_) {
        modeKey_ = MODE_FULL_LOAD;
        mUbTile_ = static_cast<uint32_t>(M_);
        mUbIterNum_ = 1;
        mUbTailNum_ = static_cast<uint32_t>(M_);
    } else {
        modeKey_ = MODE_RECOMPUTE;
        mUbTile_ = static_cast<uint32_t>(rowsCap);
        mUbIterNum_ = static_cast<uint32_t>((M_ + mUbTile_ - 1) / mUbTile_);
        mUbTailNum_ = static_cast<uint32_t>(M_ - static_cast<int64_t>(mUbIterNum_ - 1) * mUbTile_);
    }

    // 下发给内核的三个缓冲字节数,以及自检:总占用必须真的装得下 UB。
    const int64_t cAlign = Ops::Base::CeilAlign(cTile_, static_cast<int64_t>(vectorLen_));
    paramBufBytes_ = static_cast<uint32_t>(cAlign * FLOAT_DTYPE_BYTES);
    tmpParamBufBytes_ = static_cast<uint32_t>(cAlign * static_cast<int64_t>(tTypeBytes_));
    tileBytes_ = static_cast<uint32_t>(FlowTileBytes(cTile_, static_cast<int64_t>(mUbTile_)));
    const int64_t totalBytes = paramBytes + static_cast<int64_t>(tileBytes_) * UB_COPIES_3 * DOUBLE_BUFFER;
    OP_CHECK_IF(totalBytes > static_cast<int64_t>(ubSize_),
                OP_LOGE(context_->GetNodeName(), "stage1 UB budget %ld exceeds ubSize %lu", totalBytes, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradRegBaseTiling::DoOpTiling()
{
    OP_CHECK_IF(BlockTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "fail to calculate block tiling"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(UbTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "fail to calculate UB tiling"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradRegBaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t InstanceNormGradRegBaseTiling::GetTilingKey() const
{
    uint64_t tilingKey = modeKey_;
    if (dtype_ == ge::DT_FLOAT) {
        tilingKey += static_cast<uint64_t>(INGDtypeKey::FLOAT);
    } else if (dtype_ == ge::DT_FLOAT16) {
        tilingKey += static_cast<uint64_t>(INGDtypeKey::HALF);
    }
    return tilingKey;
}

ge::graphStatus InstanceNormGradRegBaseTiling::GetWorkspaceSize()
{
    reduceNCnt_ = N_;
    workSpaceSize_ = reduceNCnt_ * C_;

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    int64_t usrWorkspaceSize = static_cast<int64_t>(WORKSPACE_COPIES) * workSpaceSize_ * FLOAT_DTYPE_BYTES;
    workspaces[0] = sysWorkspaceSize_ + usrWorkspaceSize;

    // stage2: split C across cores for the deterministic cross-N reduction.
    const int64_t blkF32 = static_cast<int64_t>(blockSize_) / FLOAT_DTYPE_BYTES;
    cBlockFactor_ = Ops::Base::CeilDiv(C_, static_cast<int64_t>(coreNum_));
    cBlockFactor_ = std::max<int64_t>(cBlockFactor_, blkF32);
    stage2CoreUsed_ = static_cast<uint32_t>(Ops::Base::CeilDiv(C_, cBlockFactor_));
    cTailBlockFactor_ = C_ - cBlockFactor_ * (static_cast<int64_t>(stage2CoreUsed_) - 1);

    // stage2 每次处理的通道数由 UB 决定,不能在内核里硬编码:内核 Stage2Process 先 pipe_->Reset(),
    // 之后每通道占 STAGE2_BUFFERS_F32 个 float 缓冲(in 双缓冲 2 + accDg/accDb + 两个 Kahan 补偿)
    // 外加 1 份输出 dtype。改动内核缓冲个数时必须同步 STAGE2_BUFFERS_F32,否则 UB 超限。
    const int64_t stage2BytesPerCh = static_cast<int64_t>(STAGE2_BUFFERS_F32) * FLOAT_DTYPE_BYTES +
                                     static_cast<int64_t>(tTypeBytes_);
    // 向下对齐到一个向量长度:内核直接按此值分配,不再二次向上对齐(向上取整会越过本容量)。
    const int64_t vl = static_cast<int64_t>(vectorLen_);
    int64_t stage2Cap = static_cast<int64_t>(ubSize_) / stage2BytesPerCh / vl * vl;
    OP_CHECK_IF(stage2Cap <= 0,
                OP_LOGE(context_->GetNodeName(), "ubSize %lu is too small for the stage2 reduction.", ubSize_),
                return ge::GRAPH_FAILED);
    // cBlockFactor_ 本身不一定是向量长度整数倍,取整数倍上界即可(不足一个向量时保底一个向量)。
    stage2SubCap_ = static_cast<uint32_t>(std::min<int64_t>(stage2Cap, Ops::Base::CeilDiv(cBlockFactor_, vl) * vl));
    // 与内核 InitStage2Buffers 一一对应:STAGE2_BUFFERS_F32 份 fp32 缓冲 + 1 份输出 dtype 缓冲。
    stage2BufBytes_ = stage2SubCap_ * FLOAT_DTYPE_BYTES;
    stage2OutBufBytes_ = stage2SubCap_ * tTypeBytes_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradRegBaseTiling::PostTiling()
{
    SetTilingData();
    PrintTilingData();
    uint32_t blockDim = stage1CoreUsed_;
    if (N_ > 1) {
        blockDim = std::max(stage1CoreUsed_, stage2CoreUsed_);
    }
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(SCHEDULE_MODE);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void InstanceNormGradRegBaseTiling::SetTilingData()
{
    tilingData.set_N(N_);
    tilingData.set_C(C_);
    tilingData.set_M(M_);
    tilingData.set_cTile(cTile_);
    tilingData.set_cTileNum(cTileNum_);
    tilingData.set_taskNum(taskNum_);
    tilingData.set_taskNumPerCore(taskNumPerCore_);
    tilingData.set_taskNumPerTailCore(taskNumPerTailCore_);
    tilingData.set_tailCore(tailCore_);
    tilingData.set_stage1CoreUsed(stage1CoreUsed_);
    tilingData.set_mUbTile(mUbTile_);
    tilingData.set_mUbIterNum(mUbIterNum_);
    tilingData.set_mUbTailNum(mUbTailNum_);
    tilingData.set_reduceNCnt(reduceNCnt_);
    tilingData.set_workSpaceSize(workSpaceSize_);
    tilingData.set_stage2CoreUsed(stage2CoreUsed_);
    tilingData.set_cBlockFactor(cBlockFactor_);
    tilingData.set_cTailBlockFactor(cTailBlockFactor_);
    tilingData.set_stage2SubCap(stage2SubCap_);
    tilingData.set_paramBufBytes(paramBufBytes_);
    tilingData.set_tmpParamBufBytes(tmpParamBufBytes_);
    tilingData.set_tileBytes(tileBytes_);
    tilingData.set_stage2BufBytes(stage2BufBytes_);
    tilingData.set_stage2OutBufBytes(stage2OutBufBytes_);
}

void InstanceNormGradRegBaseTiling::PrintTilingData() const
{
    OP_LOGD(opName, "N=%ld C=%ld M=%ld cTile=%ld cTileNum=%ld taskNum=%ld", N_, C_, M_, cTile_, cTileNum_, taskNum_);
    OP_LOGD(opName, "mode=%u mUbTile=%u mUbIterNum=%u mUbTailNum=%u", modeKey_, mUbTile_, mUbIterNum_, mUbTailNum_);
    OP_LOGD(opName, "stage1CoreUsed=%u taskNumPerCore=%u taskNumPerTailCore=%u tailCore=%u", stage1CoreUsed_,
            taskNumPerCore_, taskNumPerTailCore_, tailCore_);
    OP_LOGD(opName,
            "reduceNCnt=%ld workSpaceSize=%ld stage2CoreUsed=%u cBlockFactor=%ld cTailBlockFactor=%ld stage2SubCap=%u",
            reduceNCnt_, workSpaceSize_, stage2CoreUsed_, cBlockFactor_, cTailBlockFactor_, stage2SubCap_);
}

// ---- dispatch ----------------------------------------------------------------------------------
static bool TilingShapeEmptyJudge(gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(INPUT_IDX_X);
    if (xShapePtr == nullptr) {
        return false;
    }
    auto xShape = xShapePtr->GetStorageShape();
    int64_t xSize = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        xSize *= xShape.GetDim(i);
    }
    // x 为空即走空 tensor 路径,C 轴为 0(gamma 同样为空)也在内:主 tiling 的 BlockTiling 会算出
    // cTileNum_ = 0 并触发 CeilDiv(C_, 0) 除零。
    return xSize == 0;
}

ge::graphStatus TilingInstanceNormGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("InstanceNormGrad", "Tiling context is nullptr"), return ge::GRAPH_FAILED);
    if (TilingShapeEmptyJudge(context)) {
        InstanceNormGradEmptyTiling tilingObj(context);
        return tilingObj.DoTiling();
    }
    InstanceNormGradRegBaseTiling tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepareForInstanceNormGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("InstanceNormGrad", "TilingParse context is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<InstanceNormGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo->isRegBase = true;
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    compileInfo->vectorLen = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF((compileInfo->vectorLen <= 0), OP_LOGE(context->GetNodeName(), "Failed to get vector length."),
                return ge::GRAPH_FAILED);
    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((compileInfo->blockSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get block size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InstanceNormGrad)
    .Tiling(TilingInstanceNormGrad)
    .TilingParse<InstanceNormGradCompileInfo>(TilingPrepareForInstanceNormGrad);
} // namespace optiling
