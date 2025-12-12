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
 * \file infershape_test_util.cpp
 */

#include "infershape_test_util.h"

ge::TensorDesc create_desc(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);
  return tensorDesc;
}

ge::TensorDesc create_desc_with_ori(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt,
                           ge::Format format,
                           std::initializer_list<int64_t> ori_shape_dims,
                           ge::Format ori_format) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  return tensorDesc;
}

ge::TensorDesc create_desc_with_original_shape(std::initializer_list<int64_t> shape_dims,
                                               ge::DataType dt,
                                               ge::Format format,
                                               std::initializer_list<int64_t> ori_shape_dims,
                                               ge::Format ori_format) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

gert::Shape CreateShape(const std::vector<int64_t> &shape) {
  gert::Shape gert_shape;
  gert_shape.SetDimNum(shape.size());
  for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
    gert_shape.SetDim(i, shape[i]);
  }

  return gert_shape;
}

gert::StorageShape CreateStorageShape(const std::vector<int64_t> &ori_shape, const std::vector<int64_t> &shape) {
  gert::StorageShape storage_shape;
  auto &gert_ori_shape = storage_shape.MutableOriginShape();
  auto &gert_storage_shape = storage_shape.MutableStorageShape();

  gert_ori_shape = CreateShape(ori_shape);
  if (shape.empty()) {
    gert_storage_shape = CreateShape(ori_shape);
  } else {
    gert_storage_shape = CreateShape(shape);
  }

  return storage_shape;
}
