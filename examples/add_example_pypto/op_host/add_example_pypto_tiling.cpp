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
 * \file add_example_pypto_tiling.cpp
 * \brief AddExamplePypto算子的tiling实现
 *
 * 注意：本文件显式include PyPTO生成的头文件。enable_pypto_kernel(add_example_pypto)在configure阶段
 * 生成 AddExamplePyptoTilingKey_tilingkey.h 与 AddExamplePyptoTiling_tiling.h（位于build目录
 * PYPTO_GEN_DIR），cmake通过 target_include_directories 将该目录加入tiling目标的include路径
 * （CMake 3.16 兼容，不依赖 3.18 的 -include 强制包含）。因此下面可以直接使用
 * AddExamplePyptoTiling —— 它就是op_kernel/add_example_pypto.py中的同名dataclass，host与kernel
 * 共享同一份tiling布局，无需手写副本。
 */

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
// PyPTO 生成的头文件（顺序：tilingkey在前，tiling在后），路径由 pypto_include_tiling_gen() 注入
#include "AddExamplePyptoTilingKey_tilingkey.h"
#include "AddExamplePyptoTiling_tiling.h"

namespace optiling {

// 向框架登记tiling data的布局与大小。字段与PyPTO生成的AddExamplePyptoTiling一一对应。
BEGIN_TILING_DATA_DEF(AddExamplePyptoTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows)
TILING_DATA_FIELD_DEF(int64_t, columns)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AddExamplePypto, AddExamplePyptoTilingData)

// 编译信息结构体
struct AddExamplePyptoCompileInfo {};

// PyPTO kernel按16x16的tile处理，shape需按tile对齐
constexpr int64_t TILE_ROWS = 16;
constexpr int64_t TILE_COLUMNS = 16;
// 输入要求为2维
constexpr size_t DIMS_LIMIT = 2;
// tiling key bit0：0=add，1=sub。与add_example_pypto.py中AddExamplePyptoTilingKey.Operation对应
constexpr uint64_t OPERATION_ADD = 0;

/*!
 * \brief AddExamplePypto算子的tiling入口
 */
static ge::graphStatus AddExamplePyptoTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台信息，确定可用的AI Core数量
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    // 2、获取并校验shape
    auto inputX1 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX1);
    const auto& shapeX1 = inputX1->GetStorageShape();
    OP_CHECK_IF(shapeX1.GetDimNum() != DIMS_LIMIT,
                OP_LOGE(context, "AddExamplePypto: input dim = %zu, should be equal 2", shapeX1.GetDimNum()),
                return ge::GRAPH_FAILED);

    // kernel 用同一份 rows/columns 构造 x1、x2、y 的寻址，x2 与 x1 必须完全同 shape，
    // 否则 kernel 会按 x1 的范围访问 x2，产生越界 GM 访问。
    auto inputX2 = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX2);
    const auto& shapeX2 = inputX2->GetStorageShape();
    OP_CHECK_IF(shapeX2.GetDimNum() != shapeX1.GetDimNum(),
                OP_LOGE(context, "AddExamplePypto: x2 dim = %zu, should equal x1 dim = %zu", shapeX2.GetDimNum(),
                        shapeX1.GetDimNum()),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < shapeX1.GetDimNum(); i++) {
        OP_CHECK_IF(shapeX2.GetDim(i) != shapeX1.GetDim(i),
                    OP_LOGE(context, "AddExamplePypto: x2 shape[%zu] = %ld, should equal x1 shape[%zu] = %ld", i,
                            shapeX2.GetDim(i), i, shapeX1.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }

    int64_t rows = shapeX1.GetDim(0);
    int64_t columns = shapeX1.GetDim(1);
    OP_CHECK_IF(rows % TILE_ROWS != 0 || columns % TILE_COLUMNS != 0,
                OP_LOGE(context, "AddExamplePypto: shape [%ld, %ld] should be aligned to [%ld, %ld]", rows, columns,
                        TILE_ROWS, TILE_COLUMNS),
                return ge::GRAPH_FAILED);

    // 3、设置workspace，本算子不需要额外workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U;

    // 4、填充tiling data。AddExamplePyptoTiling来自PyPTO kernel(.py dataclass)，经force-include可见
    AddExamplePyptoTiling* tilingData = context->GetTilingData<AddExamplePyptoTiling>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->rows = rows;
    tilingData->columns = columns;

    // 5、设置block dim与tiling key
    context->SetBlockDim(1);
    // TilingKey: bit0=Operation(0=add, 1=sub)
    context->SetTilingKey(OPERATION_ADD);

    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief 解析AddExamplePypto算子的编译信息，本算子无静态tiling信息
 */
static ge::graphStatus TilingParseForAddExamplePypto([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(AddExamplePypto)
    .Tiling(AddExamplePyptoTilingFunc)
    .TilingParse<AddExamplePyptoCompileInfo>(TilingParseForAddExamplePypto);
} // namespace optiling
