/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_group_quant.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_group_quant_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_group_quant_test SetUp" << endl;
  }
  static void TearDownTestCase() {
    cout << "l2_group_quant_test TearDown" << endl;
  }

 public:
  void CommonTest(const vector<int64_t>& xShape, const vector<int64_t>& scaleShape,
                  const vector<int64_t>& groupIndexShape, const vector<int64_t>& offsetShape,
                  const vector<int64_t>& yShape,
                  aclDataType xDtype, aclDataType scaleDtype, aclDataType groupIndexDtype,
                  aclDataType offsetDtype, aclDataType yDtype,
                  aclDataType dstType, bool isOffset, aclnnStatus expectRet) {
    auto x = TensorDesc(xShape, xDtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto scale = TensorDesc(scaleShape, scaleDtype, ACL_FORMAT_ND);
    auto groupIndex = TensorDesc(groupIndexShape, groupIndexDtype, ACL_FORMAT_ND);
    auto y = TensorDesc(yShape, yDtype, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ACLNN_SUCCESS;
    if (isOffset) {
      auto offset = TensorDesc(offsetShape, offsetDtype, ACL_FORMAT_ND);
      auto ut = OP_API_UT(aclnnGroupQuant,
                          INPUT(x, scale, groupIndex, offset, static_cast<int32_t>(dstType)), OUTPUT(y));
      aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else {
      auto ut = OP_API_UT(aclnnGroupQuant,
                          INPUT(x, scale, groupIndex, nullptr, static_cast<int32_t>(dstType)), OUTPUT(y));
      aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
    EXPECT_EQ(aclRet, expectRet);
    // ut.TestPrecision();
  }
};


TEST_F(l2_group_quant_test, ascend910B2_success) {
  // empty
  CommonTest({3, 0}, {2, 0}, {2}, {1}, {3, 0}, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_SUCCESS);
  // data type cases
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_SUCCESS);
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_BF16, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_SUCCESS);
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT16, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_SUCCESS);
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT64, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_SUCCESS);
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT64, ACL_FLOAT, ACL_INT8, ACL_INT8, false, ACLNN_SUCCESS);
  // int4
  CommonTest({3, 8}, {2, 8}, {2}, {1}, {3, 8}, ACL_FLOAT, ACL_FLOAT, ACL_INT64, ACL_FLOAT, ACL_INT4, ACL_INT4, true, ACLNN_SUCCESS);
  CommonTest({3, 8}, {2, 8}, {2}, {1}, {3, 1}, ACL_FLOAT, ACL_FLOAT, ACL_INT64, ACL_FLOAT, ACL_INT32, ACL_INT32, true, ACLNN_SUCCESS);
}

TEST_F(l2_group_quant_test, ascend910B2_param_invalid) {
  // invalid dtype
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_ERR_PARAM_INVALID);
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT8, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_ERR_PARAM_INVALID);
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT4, ACL_INT8, true, ACLNN_ERR_PARAM_INVALID);

  // invalid shape
  CommonTest({3, 6}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_ERR_PARAM_INVALID);
  CommonTest({3, 5}, {2, 5}, {2}, {2}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_INT8, ACL_INT8, true, ACLNN_ERR_PARAM_INVALID);
  // int4
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT64, ACL_FLOAT, ACL_INT4, ACL_INT4, true, ACLNN_ERR_PARAM_INVALID);
  CommonTest({3, 5}, {2, 5}, {2}, {1}, {3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_INT64, ACL_FLOAT, ACL_INT32, ACL_INT32, true, ACLNN_ERR_PARAM_INVALID);
}
