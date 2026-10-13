# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# AICPU built-in package manifest
# Default is OFF for all AICPU ops; only list ON entries here.
# Key is op name (e.g. "elu", "scatter_nd_min").
# To enable all AICPU ops in built-in package, set FORCE_ALL to ON
# (edit this file, or pass -DAICPU_BUILTIN_PACKAGE_FORCE_ALL=ON to cmake).

if(NOT DEFINED AICPU_BUILTIN_PACKAGE_FORCE_ALL)
  set(AICPU_BUILTIN_PACKAGE_FORCE_ALL OFF)
endif()

# The manifest may shrink between re-configures of the same build dir; drop stale
# AICPU cust obj target names accumulated in the cache by a previous configure run.
unset(AICPU_CUST_OBJ_TARGETS CACHE)

set(AICPU_BUILTIN_PACKAGE_MANIFEST
  # Add entries like "new_op=ON" when new AICPU ops should be packaged in built-in.
)

foreach(_entry ${AICPU_BUILTIN_PACKAGE_MANIFEST})
  if(NOT _entry MATCHES "^[a-z0-9_]+=ON$")
    message(WARNING "Invalid AICPU built-in package manifest entry '${_entry}' (expected '<op_name>=ON', case-sensitive); this entry will be ignored.")
  endif()
endforeach()
