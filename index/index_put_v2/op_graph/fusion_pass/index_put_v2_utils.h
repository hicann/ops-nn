/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_INDEX_PUT_WITH_SORT_V2__UTILS_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_INDEX_PUT_WITH_SORT_V2__UTILS_CPP_H_
#include <utility>
#include <vector>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "index_put_v2_utils_c.h"

namespace ge {
namespace es {


inline EsTensorHolder IndexPutV3(
    EsTensorHolder &selfRef,
    EsTensorHolder &values,
    EsTensorHolder &masks,
    const std::vector<EsTensorHolder> &indices,
    bool accumulate = false,
    bool deterministic = false) {
  auto esb_indices = TensorsToEsCTensorHolders(indices);
  auto out = EsIndexPutV3(
      selfRef.GetCTensorHolder(),
      values.GetCTensorHolder(),
      masks.GetCTensorHolder(),
      esb_indices.data(),
      static_cast<int64_t>(esb_indices.size()),
      accumulate,
      deterministic);
  return out;
}

inline EsTensorHolder IndexPutWithSortV2(
    EsTensorHolder &self,
    EsTensorHolder &linear_index,
    EsTensorHolder &pos_idx,
    EsTensorHolder &values,
    const std::vector<int64_t> &indexed_sizes,
    bool accumulate = false) {
  auto out = EsIndexPutWithSortV2(
      self.GetCTensorHolder(),
      linear_index.GetCTensorHolder(),
      pos_idx.GetCTensorHolder(),
      values.GetCTensorHolder(),
      indexed_sizes.data(),
      static_cast<int64_t>(indexed_sizes.size()),
      accumulate);
  return out;
}

inline EsTensorHolder LinearIndexV2(
    const std::vector<EsTensorHolder> &indices_list,
    EsTensorHolder &stride,
    EsTensorHolder &value_size) {
  auto esb_indices_list = TensorsToEsCTensorHolders(indices_list);
  auto out = EsLinearIndexV2(
      esb_indices_list.data(),
      static_cast<int64_t>(esb_indices_list.size()),
      stride.GetCTensorHolder(),
      value_size.GetCTensorHolder());
  return out;
}

}  // namespace es
}  // namespace ge
#endif