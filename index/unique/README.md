# Unique

本目录提供全局去重接口 `aclnnUnique` 和 `aclnnUnique2`。两者均返回输入张量中的唯一元素，可选择返回原输入到去重结果的索引；`aclnnUnique2` 还可返回每个唯一元素的出现次数。

本模块通过组合已有算子实现功能，不包含独立 Kernel。支持 AI Core 的路径使用 Sort、UniqueConsecutive 等算子；其他场景调用 UniqueWithCountsAndSorting AI CPU 实现。具体支持条件以接口实现和接口文档为准。

`op_api/unique_common.cpp` 和 `op_api/unique_common.h` 是两个接口共用的内部实现。连续去重接口 `aclnnUniqueConsecutive` 及其底层实现仍位于 [unique_consecutive](../unique_consecutive/README.md)。

| 接口 | 文档 | 示例 |
| :--- | :--- | :--- |
| aclnnUnique | [接口说明](docs/aclnnUnique.md) | [调用示例](examples/test_aclnn_unique.cpp) |
| aclnnUnique2 | [接口说明及示例](docs/aclnnUnique2.md) | 见接口文档 |
