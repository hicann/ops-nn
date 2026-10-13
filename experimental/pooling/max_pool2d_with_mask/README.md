# MaxPool2dWithMask

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Atlas 800T A2（Ascend 910B） | √ |
| Atlas 300V Pro（Ascend 310P） | √ |

## 功能说明

使用 Ascend C 实现二维最大池化，输出每个池化窗口的最大值及其通道内平面索引。
相等的最大值保留按行优先顺序扫描到的首个位置，索引为 `ih * W + iw`。

## 参数说明

两段式接口为 `aclnnMaxPool2dWithMaskGetWorkspaceSize` 和 `aclnnMaxPool2dWithMask`。
第一段检查参数并生成执行器及工作空间大小，第二段在指定 stream 上执行。

| 参数 | 输入/输出/属性 | 说明 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| self | 输入 | CHW 或 NCHW，支持非连续存储。 | FLOAT16、FLOAT；A2 还支持 BFLOAT16。 | ND、NCHW |
| kernelSize | 属性 | 长度为 1 或 2，元素大于 0。 | ListInt | - |
| stride | 属性 | 长度为 0、1 或 2；空数组使用 kernelSize。 | ListInt | - |
| padding | 属性 | 长度为 1 或 2，各维取值为 0 至对应窗口大小的一半。 | ListInt | - |
| dilation | 属性 | 长度为 1 或 2，元素为 1。 | ListInt | - |
| ceilMode | 属性 | 控制输出尺寸的向上或向下取整。 | Bool | - |
| out | 输出 | 池化结果，支持非连续存储。 | 与 self 相同。 | ND、NCHW |
| indices | 输出 | 连续容器，保存连续有效 INT32 索引，未使用尾部不作保证。 | INT8 | ND、NCHW |

## 约束说明

输入不包含 NaN 或负无穷。输出保留所选输入值的位模式。

记 `P=Ho*Wo`，四维输入的 indices 形状为 `[N,C,kH*kW,W]`，三维输入省略 N。
当 `kH*kW>=2` 时，`W=(ceil(P/16)+1)*32`；当 `kH*kW=1` 时，
`W=max((ceil(P/16)+1)*32,ceil(P/8)*32)`，保证 1×1 池化也能保存全部 INT32 索引。
容器前 `4*N*C*Ho*Wo` 字节保存连续小端 INT32 索引，其余字节未定义，不应读取或参与正确性比较。
调用方须使用上述形状分配容器；形状不匹配时接口返回参数错误。例如 5×5 输入的 1×1 池化需要 128 字节，
原形状的 96 字节不足以保存 25 个 INT32 索引。
非连续输入通过 `Contiguous` 处理，非连续 out 通过 `ViewCopy` 写回。

## 调用说明

| 调用方式 | 调用样例 | 接口声明 |
| --- | --- | --- |
| aclnn 调用 | [test_aclnn_max_pool2d_with_mask](examples/test_aclnn_max_pool2d_with_mask.cpp) | [aclnn_max_pool2d_with_mask.h](op_api/aclnn_max_pool2d_with_mask.h) |

## 构建与安装

加载 CANN 9.1.0 环境后，在仓库根目录执行：

```bash
bash build.sh --pkg --experimental --soc=ascend910b \
  --ops=max_pool2d_with_mask --vendor_name=customize -j8
LD_PRELOAD="$ASCEND_HOME_PATH/lib64/libacl_rt.so" \
  bash build_out/cann-ops-nn-customize_linux-$(uname -m).run \
  --install-path="$PWD/installed" --quiet
export CUSTOM_OPP_ROOT="$PWD/installed/vendors/customize_nn"
source "$CUSTOM_OPP_ROOT/bin/set_env.bash"
export LD_LIBRARY_PATH="$CUSTOM_OPP_ROOT/op_api/lib:$LD_LIBRARY_PATH"
```

310P 将 `--soc=ascend910b` 替换为 `--soc=ascend310p`。
CANN 9.1.0 的安装检查显式加载 SDK 运行时库；上述 `LD_PRELOAD` 仅作用于安装命令。

## 示例与测试

从仓库根目录运行示例：

```bash
bash build.sh --run_example max_pool2d_with_mask eager cust \
  --experimental --vendor_name=customize --soc=ascend910b
```

310P 将上述 `--soc` 改为 `ascend310p`。示例成功时输出 `PASS` 并返回 0。
示例逐字节校验池化结果与有效 INT32 索引。完整附件、泛化和性能测试步骤见
[tests/st/README.md](tests/st/README.md)。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| wiooo_ | 个人开发者 | MaxPool2dWithMask | 2026/09 | Ascend C 实现、双平台适配及测试。 |
