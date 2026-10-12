# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# socVersion 与 npuArch 的映射关系
set(ASCEND_ALL_COMPUTE_UNIT "ascend031" "ascend035" "ascend310b" "ascend310p" "ascend910_93"
                             "ascend950" "ascend350" "ascend910b" "ascend910" "kirinx90" "kirin9030"
                             "mc62" "ascend5162a" "ascend960dt" "ascend960pr")
set(SOC_TO_ARCH_VALUES "dav_3004" "dav_3505" "dav_3002" "dav_2002" "dav_2201" "dav_3510" "dav_3510" "dav_2201"
                      "dav_1001" "dav_3003" "dav_3113" "dav_5102" "dav_5162" "dav_9201" "dav_9202")
# npuArch 集合（由映射值去重得到）
set(NPU_ARCH_LIST ${SOC_TO_ARCH_VALUES})
list(REMOVE_DUPLICATES NPU_ARCH_LIST)
