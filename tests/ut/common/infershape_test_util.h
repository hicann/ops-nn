/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infershape_test_util.h
 */
#ifndef NN_TESTS_UT_COMMON_INFERSHAPE_TEST_UTIL_H_
#define NN_TESTS_UT_COMMON_INFERSHAPE_TEST_UTIL_H_

#include <iostream>

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ge/ge_api_types.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "op_impl_registry.h"

ge::TensorDesc create_desc(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt=ge::DT_FLOAT);

ge::TensorDesc create_desc_with_ori(std::initializer_list<int64_t> shape_dims,
                                    ge::DataType dt=ge::DT_FLOAT,
                                    ge::Format format=ge::FORMAT_ND,
                                    std::initializer_list<int64_t> ori_shape_dims={},
                                    ge::Format ori_format=ge::FORMAT_ND);
ge::TensorDesc create_desc_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);

ge::TensorDesc create_desc_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);
ge::TensorDesc create_desc_with_original_shape(std::initializer_list<int64_t> shape_dims,
                                               ge::DataType dt,
                                               ge::Format format,
                                               std::initializer_list<int64_t> ori_shape_dims,
                                               ge::Format ori_format);
ge::TensorDesc create_desc_shape_and_origin_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);

/*
 * @brief: Create gert::Shape according to std::vector
 * @param shape: std::vector of shape
 * @return gert::Shape: the register function for infer axis type info
 */
gert::Shape CreateShape(const std::vector<int64_t>& shape);

/*
 * @brief: Create gert::StorageShape according to std::vector of ori_shape and shape
 * @param ori_shape: std::vector of original shape
 *            shape: std::vector of storage shape, default is the same as original shape
 * @return gert::Shape: the register function for infer axis type info
 */
gert::StorageShape CreateStorageShape(const std::vector<int64_t>& ori_shape, const std::vector<int64_t>& shape = {});
#endif //NN_TESTS_UT_COMMON_INFERSHAPE_TEST_UTIL_H_
