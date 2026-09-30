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
 * \file extend_conv2d_op_graph.cpp
 * \brief ExtendConv2D OpSelectFormat实现：算子信息库侧的format、unknownShapeFormat列表置空后，
 *        dtype/format组合全量由本实现下发。按C04 HwSplit模式校验结果，把全量组合按filter格式
 *        （FRACTAL_Z / FRACTAL_Z_C04）过滤后下发，两个分支都覆盖全部dtype组合（各96种），
 *        仅filter格式不同；filter（weight）额外下发sub_format=0（单个值，对全部组合生效）。
 *
 *        下面的12张列表即全量组合（共192种），索引对齐：同一索引i的各列表取值属于同一种组合。
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t NUM_OF_DTYPE = 192;

std::vector<ge::DataType> extendConv2dFmpDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};

std::vector<ge::DataType> extendConv2dWeightDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8};

std::vector<ge::DataType> extendConv2dBiasDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,
    ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,
    ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32};

std::vector<ge::DataType> extendConv2dOffsetWDataTypeList(NUM_OF_DTYPE, ge::DT_INT8);

std::vector<ge::DataType> extendConv2dScaleAttrDataTypeList = {
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,
    ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64,
    ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64,
    ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64,
    ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64};

std::vector<ge::DataType> extendConv2dReluWeightAttrDataTypeList(NUM_OF_DTYPE, ge::DT_FLOAT);

std::vector<ge::DataType> extendConv2dOutput0DataTypeList = {
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8};

std::vector<ge::DataType> extendConv2dOutput1DataTypeList = {
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8};

std::vector<ge::Format> extendConv2dFmapFormatList = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC};

std::vector<ge::Format> extendConv2dWeightFormatList = {
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z,     ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04};

std::vector<ge::Format> extendConv2dNDFormatList(NUM_OF_DTYPE, ge::FORMAT_ND);

std::vector<ge::Format> extendConv2dOutputFormatList = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC};

std::string DtToString(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return "float16";
        case ge::DT_INT8:
            return "int8";
        case ge::DT_INT32:
            return "int32";
        case ge::DT_INT64:
            return "int64";
        case ge::DT_UINT64:
            return "uint64";
        case ge::DT_FLOAT:
            return "float32";
        default:
            return "undefined";
    }
}

std::string FmtToString(ge::Format fmt)
{
    switch (fmt) {
        case ge::FORMAT_NCHW:
            return "NCHW";
        case ge::FORMAT_NHWC:
            return "NHWC";
        case ge::FORMAT_FRACTAL_Z:
            return "FRACTAL_Z";
        case ge::FORMAT_FRACTAL_Z_C04:
            return "FRACTAL_Z_C04";
        case ge::FORMAT_ND:
            return "ND";
        default:
            return "ND";
    }
}

struct TensorSpec {
    std::string classify;
    struct {
        std::string name;
        std::vector<std::string> dtypes;
        std::vector<std::string> formats;
        std::vector<std::string> dynFormats;
        std::vector<std::string> subFormats;
    } desc;
};

template <typename T>
std::string Join(const std::vector<T>& sequence, std::string const& separator)
{
    std::ostringstream result;
    typename std::vector<T>::const_iterator iter = sequence.begin();
    if (iter != sequence.end()) {
        result << *iter++;
    }
    while (iter != sequence.end()) {
        result << separator << *iter++;
    }
    return result.str();
}

// 由nlohmann序列化，字符串转义由库保证；sub_format非空时下发（逗号分隔），
// FE解析后写入ext_dynamic_sub_format作为该tensor支持的subformat列表
inline void ToJson(nlohmann::json& jsonResult, const std::vector<TensorSpec>& result)
{
    for (const auto& op : result) {
        nlohmann::json descJson = {
            {"name", op.desc.name},
            {"dtype", Join(op.desc.dtypes, ",")},
            {"format", Join(op.desc.formats, ",")},
            {"unknownshape_format", Join(op.desc.dynFormats, ",")},
        };
        if (!op.desc.subFormats.empty()) {
            descJson["sub_format"] = Join(op.desc.subFormats, ",");
        }
        jsonResult[op.classify] = descJson;
    }
}

inline TensorSpec GenTensorSpec(const std::string& classify, const std::string& name,
                                const std::vector<ge::DataType>& dtypes, const std::vector<ge::Format>& formats)
{
    std::vector<std::string> dtypeStrs;
    std::vector<std::string> fmtStrs;
    for (size_t i = 0; i < dtypes.size(); i++) {
        dtypeStrs.push_back(DtToString(dtypes[i]));
        fmtStrs.push_back(FmtToString(formats[i]));
    }
    return {.classify = classify,
            .desc = {.name = name, .dtypes = dtypeStrs, .formats = fmtStrs, .dynFormats = fmtStrs}};
}

struct CubeMkn {
    int64_t m;
    int64_t k;
    int64_t n;
};

inline CubeMkn GetCubeMkn(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return {16, 32, 16};
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return {16, 16, 16};
        default:
            return {16, 16, 16};
    }
}

inline int64_t GetBitRatio(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_INT4:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2;
        case ge::DT_INT32:
        case ge::DT_UINT32:
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_INT64:
        case ge::DT_UINT64:
        case ge::DT_DOUBLE:
            return 8;
        default:
            return 1;
    }
}

inline int64_t AlignB(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return ((a + b - 1) / b) * b;
}

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

inline bool CheckC04HwSplitModeL1Valid(const gert::OpCheckContext* context)
{
    const auto* fmapDesc = context->GetInputDesc(0);
    const auto* fmapShape = context->GetInputShape(0);
    const auto* weightDesc = context->GetInputDesc(1);
    const auto* weightShape = context->GetInputShape(1);
    const auto* attrs = context->GetAttrs();
    if (fmapDesc == nullptr || fmapShape == nullptr || weightDesc == nullptr || weightShape == nullptr ||
        attrs == nullptr) {
        OP_LOGD(context->GetNodeName(), "[OpSelectFormat] CheckC04HwSplitModeL1Valid: nullptr input, return false.");
        return false;
    }

    const char* dataFormatPtr = attrs->GetStr(4);
    if (dataFormatPtr == nullptr) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: data_format attr is null, return false.");
        return false;
    }
    std::string dataFormat = dataFormatPtr;
    size_t posH = dataFormat.find('H');
    size_t posW = dataFormat.find('W');
    if (posH == std::string::npos || posW == std::string::npos) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: H/W not found in data_format[%s], return false.",
                dataFormat.c_str());
        return false;
    }

    const auto* stridesList = attrs->GetListInt(0);
    const auto* padsList = attrs->GetListInt(1);
    const auto* dilationsList = attrs->GetListInt(2);
    if (stridesList == nullptr || padsList == nullptr || dilationsList == nullptr) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: strides/pads/dilations attr is null, return false.");
        return false;
    }
    const int64_t* strides = stridesList->GetData();
    const int64_t* pads = padsList->GetData();
    const int64_t* dilations = dilationsList->GetData();
    if (stridesList->GetSize() <= posH || stridesList->GetSize() <= posW || dilationsList->GetSize() <= posH ||
        dilationsList->GetSize() <= posW || padsList->GetSize() < 4) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: strides/dilations/pads size too small, return false.");
        return false;
    }
    int64_t strideH = strides[posH];
    int64_t strideW = strides[posW];
    int64_t dilateH = dilations[posH];
    int64_t dilateW = dilations[posW];
    int64_t padLeft = pads[2];
    int64_t padRight = pads[3];

    std::string fmapOriFormat = ge::TypeUtils::FormatToSerialString(fmapDesc->GetOriginFormat());
    size_t fmapPosH = fmapOriFormat.find('H');
    size_t fmapPosW = fmapOriFormat.find('W');
    if (fmapPosH == std::string::npos || fmapPosW == std::string::npos) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: H/W not found in fmap ori_format[%s], return false.",
                fmapOriFormat.c_str());
        return false;
    }
    const auto& fmapOriShape = fmapShape->GetOriginShape();
    if (fmapOriShape.GetDimNum() <= fmapPosH || fmapOriShape.GetDimNum() <= fmapPosW) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: fmap shape dim_num too small, return false.");
        return false;
    }
    int64_t hIn = fmapOriShape.GetDim(fmapPosH);
    int64_t wIn = fmapOriShape.GetDim(fmapPosW);

    std::string weightOriFormat = ge::TypeUtils::FormatToSerialString(weightDesc->GetOriginFormat());
    size_t weightPosH = weightOriFormat.find('H');
    size_t weightPosW = weightOriFormat.find('W');
    if (weightPosH == std::string::npos || weightPosW == std::string::npos) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: H/W not found in weight ori_format[%s], return false.",
                weightOriFormat.c_str());
        return false;
    }
    const auto& weightOriShape = weightShape->GetOriginShape();
    if (weightOriShape.GetDimNum() <= weightPosH || weightOriShape.GetDimNum() <= weightPosW) {
        OP_LOGD(context->GetNodeName(),
                "[OpSelectFormat] CheckC04HwSplitModeL1Valid: weight shape dim_num too small, return false.");
        return false;
    }
    int64_t hK = weightOriShape.GetDim(weightPosH);
    int64_t wK = weightOriShape.GetDim(weightPosW);

    int64_t hkDilation = (hK - 1) * dilateH + 1;
    int64_t wkDilation = (wK - 1) * dilateW + 1;
    int64_t wOut = (wIn + padLeft + padRight - wkDilation) / strideW + 1;
    if (wOut <= 0) {
        OP_LOGD(context->GetNodeName(), "[OpSelectFormat] CheckC04HwSplitModeL1Valid: wOut=%ld <= 0, return false.",
                wOut);
        return false;
    }

    CubeMkn cube = GetCubeMkn(fmapDesc->GetDataType());
    int64_t m0 = cube.m;
    int64_t k0 = cube.k;
    int64_t n0 = cube.n;
    int64_t wBitRatio = GetBitRatio(weightDesc->GetDataType());
    int64_t fmapBitRatio = GetBitRatio(fmapDesc->GetDataType());

    constexpr int64_t kC04CinValue = 4;
    int64_t kbMin = AlignB(kC04CinValue * hK * wK, k0);
    int64_t weightUsedL1Size = AlignB(kbMin * n0 * wBitRatio, k0);
    int64_t hoAl1Min = (wOut < m0) ? CeilDiv(m0, wOut) : 1;
    int64_t hiAl1Min = std::min(hIn, (hoAl1Min - 1) * strideH + hkDilation);
    int64_t fmapUsedL1Size = AlignB(hiAl1Min * wIn * kC04CinValue * fmapBitRatio, k0);
    int64_t minL1LoadSize = fmapUsedL1Size + weightUsedL1Size;

    constexpr int64_t kL1BufferSize = 1048576; // 1M
    constexpr int64_t kFb0Size = 4096;
    constexpr int64_t kFb1Size = 2048;
    constexpr int64_t kBtSize = 4096;
    int64_t l1BufferSize = kL1BufferSize - kFb0Size - kFb1Size - kBtSize;

    OP_LOGD(context->GetNodeName(),
            "[OpSelectFormat] CheckC04HwSplitModeL1Valid: hIn=%ld, wIn=%ld, hK=%ld, wK=%ld, strideH=%ld, strideW=%ld, "
            "dilateH=%ld, dilateW=%ld, padLeft=%ld, padRight=%ld, wOut=%ld, m0=%ld, k0=%ld, n0=%ld, "
            "wBitRatio=%ld, fmapBitRatio=%ld, kbMin=%ld, weightUsedL1Size=%ld, hoAl1Min=%ld, hiAl1Min=%ld, "
            "fmapUsedL1Size=%ld, minL1LoadSize=%ld, l1BufferSize=%ld.",
            hIn, wIn, hK, wK, strideH, strideW, dilateH, dilateW, padLeft, padRight, wOut, m0, k0, n0, wBitRatio,
            fmapBitRatio, kbMin, weightUsedL1Size, hoAl1Min, hiAl1Min, fmapUsedL1Size, minL1LoadSize, l1BufferSize);

    bool valid = minL1LoadSize <= l1BufferSize;
    OP_LOGD(context->GetNodeName(), "[OpSelectFormat] CheckC04HwSplitModeL1Valid: result=%d.", valid);
    return valid;
}

// 按filter格式过滤：c04EnableFlag为true取FRACTAL_Z_C04组合，否则取FRACTAL_Z组合
template <typename T>
std::vector<T> FilterListByC04Flag(const std::vector<T>& list, bool c04EnableFlag)
{
    std::vector<T> result;
    for (size_t i = 0; i < list.size() && i < extendConv2dWeightFormatList.size(); i++) {
        bool isC04 = (extendConv2dWeightFormatList[i] == ge::FORMAT_FRACTAL_Z_C04);
        if (c04EnableFlag == isC04) {
            result.push_back(list[i]);
        }
    }
    return result;
}
} // namespace

static ge::graphStatus ExtendConv2dOpSelectFormatByFlag(bool c04EnableFlag, ge::AscendString& result);

static ge::graphStatus ExtendConv2dOpSelectFormat(const gert::OpCheckContext* context, ge::AscendString& result)
{
    OP_LOGD(context == nullptr ? "" : context->GetNodeName(),
            "[OpSelectFormat] ExtendConv2dOpSelectFormat enter, context=%p.", context);
    bool c04EnableFlag = false;
    if (context != nullptr) {
        const auto* fmapDesc = context->GetInputDesc(0);
        const auto* fmapShape = context->GetInputShape(0);
        const auto* attrs = context->GetAttrs();
        if (fmapDesc != nullptr && fmapShape != nullptr && attrs != nullptr) {
            int cIndex = 1;
            int wIndex = 3;
            if (fmapDesc->GetOriginFormat() == ge::FORMAT_NHWC) {
                cIndex = 3;
                wIndex = 2;
            }
            const auto& originShape = fmapShape->GetOriginShape();
            if (originShape.GetDimNum() > static_cast<size_t>(wIndex)) {
                int64_t fmapC = originShape.GetDim(static_cast<size_t>(cIndex));
                int64_t fmapW = originShape.GetDim(static_cast<size_t>(wIndex));
                const int64_t* groupsPtr = attrs->GetInt(3);
                if (groupsPtr != nullptr && fmapC > 0 && fmapW > 0) {
                    constexpr int64_t kLoad3dv2WinLimit = 32767;
                    bool load3dv2WinLimitFlag = (fmapW <= kLoad3dv2WinLimit);
                    bool c04LoadL1ValidFlag = CheckC04HwSplitModeL1Valid(context);
                    // weight为int8时不走c04
                    const auto* weightDesc = context->GetInputDesc(1);
                    bool weightNotInt8Flag = (weightDesc != nullptr && weightDesc->GetDataType() != ge::DT_INT8);
                    c04EnableFlag = (fmapC <= 4 && *groupsPtr == 1 && load3dv2WinLimitFlag && c04LoadL1ValidFlag &&
                                     weightNotInt8Flag);
                    OP_LOGD(context->GetNodeName(),
                            "[OpSelectFormat] ExtendConv2dOpSelectFormat: fmapC=%ld, fmapW=%ld, groups=%ld, "
                            "load3dv2WinLimitFlag=%d, c04LoadL1ValidFlag=%d, weightNotInt8Flag=%d, c04EnableFlag=%d.",
                            fmapC, fmapW, *groupsPtr, load3dv2WinLimitFlag, c04LoadL1ValidFlag, weightNotInt8Flag,
                            c04EnableFlag);
                } else {
                    OP_LOGD(context->GetNodeName(),
                            "[OpSelectFormat] ExtendConv2dOpSelectFormat: skip c04 check, groupsPtr=%p, fmapC=%ld, "
                            "fmapW=%ld.",
                            groupsPtr, fmapC, fmapW);
                }
            } else {
                OP_LOGD(
                    context->GetNodeName(),
                    "[OpSelectFormat] ExtendConv2dOpSelectFormat: originShape dim_num=%zu <= wIndex=%d, c04 disabled.",
                    originShape.GetDimNum(), wIndex);
            }
        } else {
            OP_LOGD(context->GetNodeName(),
                    "[OpSelectFormat] ExtendConv2dOpSelectFormat: nullptr desc/shape/attrs, fmapDesc=%p, fmapShape=%p, "
                    "attrs=%p.",
                    fmapDesc, fmapShape, attrs);
        }
    }

    OP_LOGD(context == nullptr ? "" : context->GetNodeName(),
            "[OpSelectFormat] ExtendConv2dOpSelectFormat: c04EnableFlag=%d, filter all lists by weight format.",
            c04EnableFlag);
    ge::graphStatus ret = ExtendConv2dOpSelectFormatByFlag(c04EnableFlag, result);
    OP_LOGD(context == nullptr ? "" : context->GetNodeName(),
            "[OpSelectFormat] ExtendConv2dOpSelectFormat: json result = %s",
            result.GetString() == nullptr ? "(null)" : result.GetString());
    return ret;
}

// 按已判定的c04标志过滤12张列表并拼出算子信息库JSON，与context无关，便于直接驱动两个分支
static ge::graphStatus ExtendConv2dOpSelectFormatByFlag(bool c04EnableFlag, ge::AscendString& result)
{
    auto fmapDataTypeList = FilterListByC04Flag(extendConv2dFmpDataTypeList, c04EnableFlag);
    auto weightDataTypeList = FilterListByC04Flag(extendConv2dWeightDataTypeList, c04EnableFlag);
    auto biasDataTypeList = FilterListByC04Flag(extendConv2dBiasDataTypeList, c04EnableFlag);
    auto offsetWDataTypeList = FilterListByC04Flag(extendConv2dOffsetWDataTypeList, c04EnableFlag);
    auto scaleDataTypeList = FilterListByC04Flag(extendConv2dScaleAttrDataTypeList, c04EnableFlag);
    auto reluWeightDataTypeList = FilterListByC04Flag(extendConv2dReluWeightAttrDataTypeList, c04EnableFlag);
    auto output0DataTypeList = FilterListByC04Flag(extendConv2dOutput0DataTypeList, c04EnableFlag);
    auto output1DataTypeList = FilterListByC04Flag(extendConv2dOutput1DataTypeList, c04EnableFlag);
    auto fmapFormatList = FilterListByC04Flag(extendConv2dFmapFormatList, c04EnableFlag);
    auto weightFormatList = FilterListByC04Flag(extendConv2dWeightFormatList, c04EnableFlag);
    auto ndFormatList = FilterListByC04Flag(extendConv2dNDFormatList, c04EnableFlag);
    auto outputFormatList = FilterListByC04Flag(extendConv2dOutputFormatList, c04EnableFlag);
    OP_LOGD("", "[OpSelectFormat] ExtendConv2dOpSelectFormatByFlag: after filter, list size=%zu (origin=%zu).",
            fmapDataTypeList.size(), extendConv2dFmpDataTypeList.size());

    std::vector<TensorSpec> opInfo;
    opInfo.push_back(GenTensorSpec("input0", "x", fmapDataTypeList, fmapFormatList));
    opInfo.push_back(GenTensorSpec("input1", "filter", weightDataTypeList, weightFormatList));
    // weight的sub_format固定为0：仅下发一个0，对该tensor的全部format组合生效
    opInfo.back().desc.subFormats = {"0"};
    opInfo.push_back(GenTensorSpec("input2", "bias", biasDataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("input3", "offset_w", offsetWDataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("input4", "scale0", scaleDataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("input5", "relu_weight0", reluWeightDataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("input6", "clip_value0", output0DataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("input7", "scale1", scaleDataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("input8", "relu_weight1", reluWeightDataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("input9", "clip_value1", output1DataTypeList, ndFormatList));
    opInfo.push_back(GenTensorSpec("output0", "y0", output0DataTypeList, outputFormatList));
    opInfo.push_back(GenTensorSpec("output1", "y1", output1DataTypeList, outputFormatList));

    nlohmann::json jsonResult = nlohmann::json::object();
    ToJson(jsonResult, opInfo);
    const std::string jsonStr = jsonResult.dump();
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

// OpSelectFormat仅在算子信息库format、unknownShapeFormat为空时生效
IMPL_OP(ExtendConv2D).OpSelectFormat(ExtendConv2dOpSelectFormat);
} // namespace ops
