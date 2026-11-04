# Unique

English | [简体中文](./README.md)

This directory provides the global deduplication interfaces `aclnnUnique` and `aclnnUnique2`. Both interfaces return the unique elements in the input tensor and can optionally return the indices from the original input to the deduplication result; `aclnnUnique2` can also return the number of occurrences of each unique element.

This module implements functions by combining existing operators and does not contain an independent kernel. The path supporting AI Core uses operators such as Sort and UniqueConsecutive; other scenarios call the UniqueWithCountsAndSorting AI CPU implementation. For specific support conditions, refer to the interface implementation and the interface documentation.

`op_api/unique_common.cpp` and `op_api/unique_common.h` are the internal implementations shared by the two interfaces. The consecutive deduplication interface `aclnnUniqueConsecutive` and its underlying implementation are still located in [unique_consecutive](../unique_consecutive/README.md).

| Interface | Documentation | Example |
| :--- | :--- | :--- |
| aclnnUnique | [Interface description](docs/aclnnUnique_en.md) | [Invocation example](examples/test_aclnn_unique.cpp) |
| aclnnUnique2 | [Interface description and example](docs/aclnnUnique2_en.md) | See the interface documentation |
