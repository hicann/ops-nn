|序号|规则(CH)|Rule(EN)|配置参数|语言|支持场景|
|--|--------------------|----------------------|------------|----|----------|
|1  |[G.CMT.01-CPP 注释符与注释内容间留有1个空格](# G.CMT.01-CPP 注释符与注释内容间留有1个空格)|[G.CMT.01-CPP Add a space between the comment character and comment content](# G.CMT.01-CPP 注释符与注释内容间留有1个空格)|-|C++|版本级,门禁级|
|2  |[G.CMT.02-CPP 代码注释置于对应代码的上方或右边](# G.CMT.02-CPP 代码注释置于对应代码的上方或右边)|[G.CMT.02-CPP Place code comments above or to the right of the code](# G.CMT.02-CPP 代码注释置于对应代码的上方或右边)|-|C++|版本级,门禁级|
|3  |[G.CMT.03-CPP 文件头注释包含版权说明](# G.CMT.03-CPP 文件头注释包含版权说明)|[G.CMT.03-CPP File header comments shall contain the copyright notice](# G.CMT.03-CPP 文件头注释包含版权说明)|check_modified_year:false;is_open_source:true;corporations_zh:华为技术有限公司 &#124;海思半导体;corporations_en:Huawei Technologies Co., Ltd. &#124;Hisilicon Technologies Co., Ltd.|C++|版本级,门禁级|
|4  |[G.CMT.04-CPP 不写空有格式的函数头注释](# G.CMT.04-CPP 不写空有格式的函数头注释)|[G.CMT.04-CPP Do not write empty-content function header comments](# G.CMT.04-CPP 不写空有格式的函数头注释)|-|C++|版本级,门禁级|
|5  |[G.FMT.01-CPP 非纯ASCII码源文件使用UTF-8编码](# G.FMT.01-CPP 非纯ASCII码源文件使用UTF-8编码)|[G.FMT.01-CPP Source files involving non-ASCII code should use UTF-8 formatting](# G.FMT.01-CPP 非纯ASCII码源文件使用UTF-8编码)|-|C++|版本级,门禁级|
|6  |[G.FMT.02-CPP 使用空格进行缩进，每次缩进4个空格](# G.FMT.02-CPP 使用空格进行缩进，每次缩进4个空格)|[G.FMT.02-CPP Use spaces to indent and indent four spaces at a time](# G.FMT.02-CPP 使用空格进行缩进，每次缩进4个空格)|-|C++|版本级,门禁级|
|7  |[G.FMT.04-CPP 每个变量单独一行进行声明或赋值](# G.FMT.04-CPP 每个变量单独一行进行声明或赋值)|[G.FMT.04-CPP Each variable is declared or assigned on a separate line](# G.FMT.04-CPP 每个变量单独一行进行声明或赋值)|check_multiple_var_dec:true|C++|版本级,门禁级|
|8  |[G.FMT.05-CPP 行宽不超过120个字符](# G.FMT.05-CPP 行宽不超过120个字符)|[G.FMT.05-CPP Each line contains no more than 120 characters](# G.FMT.05-CPP 行宽不超过120个字符)|column_width:120|C++|版本级,门禁级|
|9  |[G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数调用参数换行](#G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数调用参数换行)|[G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Function call parameter wrapping](#G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数调用参数换行)|c_coding_standard_v5.0:true|C++|版本级,门禁级|
|10  |[G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数声明参数换行](# G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数声明参数换行)|[G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Data initialization wrapping](# G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数声明参数换行)|-|C++|版本级,门禁级|
|11  |[G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--数据初始化换行](# G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--数据初始化换行)|[G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Function declaration parameter wrapping](# G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--数据初始化换行)|-|C++|版本级,门禁级|
|12  |[G.FMT.07-CPP 预处理的"#"统一放在行首，嵌套预处理语句时，"#"可以进行缩进](# G.FMT.07-CPP 预处理的"#"统一放在行首，嵌套预处理语句时，"#"可以进行缩进)|[G.FMT.07-CPP The number sign (#) that starts a preprocessor directive should be placed at the beginning of a line and can be indented in nested](# G.FMT.07-CPP 预处理的"#"统一放在行首，嵌套预处理语句时，"#"可以进行缩进)|-|C++|版本级,门禁级|
|13  |[G.FMT.08-CPP 遵循传统的类成员声明顺序](# G.FMT.08-CPP 遵循传统的类成员声明顺序)|[G.FMT.08-CPP Use a conventional class member declaration order](# G.FMT.08-CPP 遵循传统的类成员声明顺序)|-|C++|版本级,门禁级|
|14  |[G.FMT.09-CPP 构造函数初始化列表放在同一行或按4空格缩进并排多行](# G.FMT.09-CPP 构造函数初始化列表放在同一行或按4空格缩进并排多行)|[G.FMT.09-CPP A constructor initialization list is placed on the same line or on different lines indented by four spaces](# G.FMT.09-CPP 构造函数初始化列表放在同一行或按4空格缩进并排多行)|-|C++|版本级,门禁级|
|15  |[G.FMT.12-CPP 避免将if/else/else if写在同一行](# G.FMT.12-CPP 避免将if/else/else if写在同一行)|[G.FMT.12-CPP Avoid placing if/else/else if on the same line](# G.FMT.12-CPP 避免将if/else/else if写在同一行)|-|C++|版本级,门禁级|
|16  |[G.FMT.13-CPP case/default语句相对switch缩进一层](# G.FMT.13-CPP case/default语句相对switch缩进一层)|[G.FMT.13-CPP Indent the case or default statement in a switch statement](# G.FMT.13-CPP case/default语句相对switch缩进一层)|-|C++|版本级,门禁级|
|17  |[G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--引用类型](# G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--引用类型)|[G.FMT.14-CPP The pointer type "*" and reference type "&" can only be placed close to a variable name or a type--Reference type](# G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--引用类型)|-|C++|版本级,门禁级|
|18  |[G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--指针类型](#G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--指针类型)|[G.FMT.14-CPP The pointer type "*" and reference type "&" can only be placed close to a variable name or a type--pointer type](#G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--指针类型)|c_coding_standard_v5.0:true;pointer_follow_type_or_var:2|C++|版本级,门禁级|
|19  |[G.FMT.16-CPP 用空格突出关键字和重要信息--关键信息](# G.FMT.16-CPP 用空格突出关键字和重要信息--关键信息)|[G.FMT.16-CPP Use spaces to highlight keywords and important information--Key Information](# G.FMT.16-CPP 用空格突出关键字和重要信息--关键信息)|-|C++|版本级,门禁级|
|20  |[G.FMT.17-CPP 合理安排空行](# G.FMT.17-CPP 合理安排空行)|[G.FMT.17-CPP Use blank lines when necessary](# G.FMT.17-CPP 合理安排空行)|-|C++|版本级,门禁级|
|21  |[G.NAM.02-CPP C++文件名和类名保持一致](# G.NAM.02-CPP C++文件名和类名保持一致)|[G.NAM.02-CPP Keep C++ file names and class names consistent](# G.NAM.02-CPP C++文件名和类名保持一致)|-|C++|版本级,门禁级|
|22  |[G.FMT.03-CPP 使用统一的大括号换行风格](# G.FMT.03-CPP 使用统一的大括号换行风格)|[G.FMT.03-CPP Use the consistent brace style for new lines](# G.FMT.03-CPP 使用统一的大括号换行风格)|c_coding_standard_v5.0:true;StyleOfBrace:K&R|C++|版本级,门禁级|

# G.CMT.01-CPP 注释符与注释内容间留有1个空格

## 【描述】

注释符与注释内容之间留有1个空格。 注释符包括： /* */和 //。 注释符的对齐方式保持与下面示例相同：

使用/* */
```
/* 单行注释 */
/*
 * 多行注释
 * 第二行
 */
```
使用//
```
// 单行注释  
// 多行注释
// 第二行
```

在多行注释中为了换行对齐，可以在注释符与注释内容之间存在多个空格：

```
// 这是一段注释符与注释内容之间存在多个空格的例子：
// XX 功能说明
// 提示：提示信息，超出一行行宽时需要换行
//       这是第二行，前面使用空格对齐
```

## 【修复建议】
按要求增加或减少空格。

# G.CMT.02-CPP 代码注释置于对应代码的上方或右边

## 【描述】

针对代码的注释，应该置于对应代码的上方或右边。

- 代码上方的注释：与代码行间无空行，保持与代码一样的缩进
- 代码右边的注释：与代码之间，至少留有1空格

通常右边的注释内容不宜过多，当注释超过行宽时，考虑将注释置于代码上方。

```CPP
// Foo() 函数的注释
int Foo()
{
    int b = 0; // 变量 b 的注释，与代码至少留有1空格
    ...
}
```

在适当的时候，右边的注释上下对齐会更美观：

```CPP
const int A_CONST = 100;       // 此处两行注释属于同类  
const int ANOTHER_CONST = 200; // 可保持左侧对齐  
```

## 【错误示例】

场景1：注释符与代码之间没有空格
错误示例：
```CPP
int x;// 不符合：注释符与代码之间无空格

int y;/* 不符合：注释符与代码之间无空格 */
```

场景2：注释位于代码中间
错误示例：
```CPP
int y /* 注释 */ = 1;       // 不符合：注释位于代码中间
```

## 【正确示例】

场景1：注释符与代码之间没有空格
修复示例：
```CPP
int x; // 符合：注释符与代码之间至少留有1个空格

int y; /* 符合：注释符与代码之间至少留有1个空格 */
```

场景2：注释位于代码中间
修复示例1：
```CPP
int y = 1; // 注释          // 符合：注释位于代码右侧
```

修复示例2：
```CPP
int y = 1; /* 注释 */       // 符合：注释位于代码右侧
```

修复示例3：

```CPP
/*
 * 注释
 * 第二行
 */
int y = 1;                  // 符合：注释位于代码上方
```

## 【修复建议】
按照要求放置代码注释。

# G.CMT.03-CPP 文件头注释包含版权说明

## 【描述】
文件头注释应首先包含版权说明。
如果文件头注释需要增加其他内容，可以后面补充。
比如：文件功能说明，作者、创建日期、注意事项等等。

版权许可内容及格式必须如下：
中文： 版权所有 (c) 华为技术有限公司 2012-2018
或英文：Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.

版权许可内容解释如下：

- 2012-2018 根据实际需要可以修改。
- 2012 是文件首次创建年份，而 2018 是最后文件修改年份。 
- 对文件有重大修改时，必须更新后面年份，如特性扩展，重大重构等。
- 可以只写一个创建年份，后续文件修改则不用更新版权声明。
- 如：版权所有 (c) 华为技术有限公司 2018
- 版权说明可以使用华为子公司，如：
- 中文： 版权所有 (c) 海思半导体 2012-2018
- 或英文：Copyright (c) Hisilicon Technologies Co., Ltd. 2012-2018. All rights reserved.
- 英文中的 'All rights reserved.' 目前不具有强制法律效力，可省略或根据习惯继续沿用。

文件头注释举例：

```CPP
/*
* 版权所有 (c) 华为技术有限公司 2012-2018
* XX 功能实现
* 注意：
* - 注意点 1
* - 注意点 2
*/
```

编写文件头注释应注意：

文件头注释应该从文件顶头开始。
如果包含“关键资产说明”类注释，则应紧随其后。
保持统一格式。
具体格式由项目或更大范围统一制定。格式可参考上面举例。
保持版面工整，若内容过长，超出行宽要求，换行时应注意对齐。
对齐可参考上述例子 '注意点'。
首先包含“版权许可”，然后包含其他可先选内容。
其他选项按需添加，并保持格式统一。
不要空有格式，无内容。
如上述例子，如果 '注意：' 后面无内容。
该规则有可配置参数，具体配置详见规则集中该规则的选项配置。

## 【正确示例】
```CPP
/*
 * 版权所有 (c) 华为技术有限公司 2012-2018
 * XX 功能实现
 * 注意：
 *      - 注意点 1
 *      - 注意点 2
 */
```

## 【修复建议】

正确包含版权声明。

# G.CMT.04-CPP 不写空有格式的函数头注释

## 【描述】
尽量通过函数名自注释其功能，按需写函数头注释。不要写空有格式的函数头注释；不要写无用、信息冗余的函数头注释。

函数头注释分为函数声明处的注释和函数定义处的注释，这两类注释的目标不同，内容也应不同:

- 函数声明处的注释的目的是传递函数原型无法表达却又希望读者知道的信息，描述函数的功能以及如何使用它
- 函数定义处的注释的目的是传递希望读者知道的函数实现细节，例如使用的算法、为什么这样实现等

## 【错误示例】
下面的函数头注释存在两个问题，应避免出现类似的注释：

- 参数、返回值，空有格式没内容
- 注释中的函数名信息冗余

```CPP
/*
 * 函数名：WriteData
 * 功能：写入字符串
 * 参数：
 * 返回值：
 */
int WriteData(const unsigned char* buf, size_t len);
```

## 【正确示例】
函数头注释统一放在函数声明或定义上方，并选择使用如下风格之一：

使用//
```CPP
// 单行函数头
int Fun1();

// 多行函数头
// 第二行
int Fun2();
```

使用 /* */
```CPP
/* 单行函数头 */  
int Fun1();

/*
 * 单行或多行函数头
 * 第二行
 */
int Fun2();
```

函数头注释内容可选，其内容不限于：功能说明、返回值、性能约束、用法、内存约定、可重入的要求等等。

## 【修复建议】
补充内容或者删除空有格式的注释。

# G.FMT.01-CPP 非纯ASCII码源文件使用UTF-8编码

## 【描述】

对于含有非ASCII字符的源文件，使用UTF-8编码。

## 【修复建议】
使用UTF-8编码。


# G.FMT.02-CPP 使用空格进行缩进，每次缩进4个空格

## 【描述】

建议使用空格进行缩进，每次缩进为 4 个空格。避免使用制表符('\t')进行缩进。
当前几乎所有的集成开发环境（IDE）和代码编辑器都支持配置将Tab键自动扩展为 4 空格输入。

## 【修复建议】

按要求进行缩进。


# G.FMT.03-CPP 使用统一的大括号换行风格

## 【描述】

选择并统一使用一种大括号换行风格，避免多种风格并存。

**K&R风格**
换行时，函数（不包括lambda表达式）左大括号另起一行放行首，并独占一行；其他左大括号跟随语句放在行末。
右大括号独占一行，除非后面跟着同一语句的剩余部分，如 do 语句中的 while，或者 if 语句的 else/else if，或者逗号、分号、小括号。

**Allman风格**
换行时，左大括另起并独占一行，保持与上一行相同缩进；
右大括号独占一行，除非后面跟着 do 语句中的 while，或者逗号、分号。

## 【修复建议】

使用相应的大括号换行风格。

# G.FMT.04-CPP 每个变量单独一行进行声明或赋值

## 【描述】

一个变量的声明或者赋值语句应该单独占一行，更加利于阅读和理解代码。
该规则有可配置参数，具体配置详见规则集中该规则的选项配置。

## 【错误示例】
```CPP
int count = 10; bool isCompleted = false; // 不符合：多个变量初始化需要分开放在多行

char* str, ch, arr[10];                   // 不符合：多个变量定义且类型不同，容易产生误解
int a = 0, b = 0;                         // 不符合：多个变量定义需要分开放在多行

int x;
int y;
...
x = 1; y = 2;                             // 不符合：一行中存在多个赋值语句
```

## 【正确示例】
```CPP
int count = 10;
bool isCompleted = false;
int a = 0;
int b = 0;

int x;
int y;
...
x = 1;
y = 2;
```

## 【修复建议】

将语句拆分到多行。

# G.FMT.05-CPP 行宽不超过120个字符

## 【描述】

代码行宽不宜过长，否则不利于阅读。

控制行宽可以间接的引导程序员去缩短函数、变量的命名，减少嵌套的层数，提升代码的可读性。

建议每行字符数不要超过 120 个；除非超过 120 能显著提升可读性，并且不会隐藏信息。

## 【修复建议】

在合适的位置换行。

# G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数调用参数换行

## 【描述】

当语句过长，或者可读性不佳时，需要在合适的地方换行。
换行时将操作符放在行末，表示“未结束，后续还有”。新行缩进一层，或者保持同类对齐。

调用函数的参数列表换行时，左圆括号总是跟函数名，右圆括号总是跟最后一个参数，并进行合理的参数对齐。

一般选择在较低优先级操作符后换行，或者根据表达式层次换行。

## 【错误示例】

场景1：函数调用参数列表换行时未缩进对齐

错误示例：
```CPP
void Fun(int ret)
{
    WriteLog(FWM_ETH_LOG_ERROR, FWM_ETH_CATEGORY_DEFAULT, FWM_ETH_MID_VLAN,
         "[DOWNLOAD] ret:%u", ret); // 不符合：换行时缩进5空格
}
```

## 【正确示例】

场景1：函数调用参数列表换行时未缩进对齐

修复示例1：
```CPP
void Fun(int ret)
{
    WriteLog(FWM_ETH_LOG_ERROR, FWM_ETH_CATEGORY_DEFAULT, FWM_ETH_MID_VLAN,
        "[DOWNLOAD] ret:%u", ret); // 符合：相对上一行缩进4空格
}
```

修复示例2：

```CPP
void Fun(int ret)
{
    WriteLog(FWM_ETH_LOG_ERROR, FWM_ETH_CATEGORY_DEFAULT, FWM_ETH_MID_VLAN,
             "[DOWNLOAD] ret:%u", ret); // 符合：与上方参数对齐
}
```

## 【修复建议】

按要求进行对齐。

# G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数声明参数换行

## 【描述】

当语句过长，或者可读性不佳时，需要在合适的地方换行。
换行时将操作符放在行末，表示“未结束，后续还有”。新行缩进一层，或者保持同类对齐。

调用函数的参数列表换行时，左圆括号总是跟函数名，右圆括号总是跟最后一个参数，并进行合理的参数对齐。

一般选择在较低优先级操作符后换行，或者根据表达式层次换行。

## 【错误示例】

场景1：函数声明时参数列表换行时未缩进对齐

错误示例：
```CPP
VOS_UINT32 Download(VOS_UINT32 type, VOS_CHAR *pData,
 VOS_UINT32 dataLen, VOS_UINT32 dataNum) // 不符合：换行时没有缩进对齐
```

## 【正确示例】

场景1：函数声明时参数列表换行时未缩进对齐

修复示例1：
```CPP
VOS_UINT32 Download(VOS_UINT32 type, VOS_CHAR *pData,
    VOS_UINT32 dataLen, VOS_UINT32 dataNum) // 符合：相对上一行缩进4空格
```

修复示例2：
```CPP
VOS_UINT32 Download(VOS_UINT32 type, VOS_CHAR *pData,
                    VOS_UINT32 dataLen, VOS_UINT32 dataNum) // 符合：与上方参数对齐

```
## 【修复建议】

按要求进行对齐。


# G.FMT.06-CPP 换行时将操作符留在行末，新行缩进一层或进行同类对齐--数据初始化换行


## 【描述】

当语句过长，或者可读性不佳时，需要在合适的地方换行。
换行时将操作符放在行末，表示“未结束，后续还有”。新行缩进一层，或者保持同类对齐。

调用函数的参数列表换行时，左圆括号总是跟函数名，右圆括号总是跟最后一个参数，并进行合理的参数对齐。

一般选择在较低优先级操作符后换行，或者根据表达式层次换行。

## 【错误示例】

场景1：数组初始化时同一层级没有缩进对齐

错误示例：
```CPP
int a[][2] = {
    {1, 2},        // {1, 2}和 {3, 4}、{5, 6}是同一层级
        {3, 4},    // 不符合：同一层级没有缩进对齐
            {5, 6} // 不符合：同一层级没有缩进对齐
};
```
场景2：数组初始化时嵌套层级相对上一层级未缩进4空格

错误示例：
```CPP
int a[][2] = {
{1, 2},        // 不符合：相对上一层级未缩进4空格
{3, 4},
{5, 6}
};
```

## 【正确示例】

场景1：数组初始化时同一层级没有缩进对齐

修复示例：

```CPP
int a[][2] = {
    {1, 2},        // {1, 2}和 {3, 4}、{5, 6}是同一层级
    {3, 4},        // 符合：同一层级缩进对齐
    {5, 6}         // 符合：同一层级缩进对齐
};
```

场景2：数组初始化时嵌套层级相对上一层级未缩进4空格

修复示例：

```CPP
int a[][2] = {
    {1, 2},    // 符合：相对上一层级缩进4空格
    {3, 4},
    {5, 6}
};
```

## 【修复建议】

按要求进行对齐。


# G.FMT.07-CPP 预处理的"#"统一放在行首，嵌套预处理语句时，"#"可以进行缩进

## 【描述】

预处理的#统一放在行首。
函数体中的预处理的#应该放在行首。
预处理语句中嵌套的#`可以按照缩进要求进行缩进对齐，区分层次。

## 【错误示例】

场景1：预处理的#不在行首

错误示例：

```CPP
// 不符合："#"不在行首
    #if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
// 不符合："#"不在行首
    #define ATOMIC_X86_HAS_CMPXCHG16B 1
#else
#define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif
```

场景2：函数体中的预处理的#不在行首。

错误示例：

```CPP
int FunctionName()
{
    if (someThingError) {
        ...
        #ifdef HAS_SYSLOG // 不符合：`#`不在行首
            WriteToSysLog();
        #else             // 不符合：`#`不在行首
            WriteToFileLog();
        #endif            // 不符合：`#`不在行首
    }
}
```

## 【正确示例】

场景1：预处理的#不在行首

修复示例1：将预处理的#统一放在行首。

```CPP
// 符合："#"放在行首
#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
// 符合："#"放在行首
#define ATOMIC_X86_HAS_CMPXCHG16B 1
#else
#define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif
```

修复示例2：预处理语句中嵌套的#可以按照缩进要求进行缩进对齐。

```CPP
// 符合："#"放在行首
#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
    // 符合：区分层次，便于阅读
    #define ATOMIC_X86_HAS_CMPXCHG16B 1
#else
    #define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif
```

场景2：函数体中的预处理的#不在行首。

修复示例1：函数体中的预处理的#应该放在行首。
```CPP
int FunctionName()
{
    if (someThingError) {
        ...
#ifdef HAS_SYSLOG         // 符合：即便在函数内部，"#"也放在行首
        WriteToSysLog();
#else                     // 符合：即便在函数内部，"#"也放在行首
        WriteToFileLog();
#endif                    // 符合：即便在函数内部，"#"也放在行首
    }
}
```

## 【修复建议】

将预处理的#统一放在行首。
预处理语句中嵌套的#可以按照缩进要求进行缩进对齐。

# G.FMT.08-CPP 遵循传统的类成员声明顺序

## 【描述】

类访问控制块的声明顺序默认依次是 public:、protected:、 private:，缩进与 class 关键字对齐。

## 【错误示例】

场景1：类访问控制块顺序错误

错误示例：
```CPP
class Derived : public Base {
private:                                    // 不符合，private应该在public之后
    void Fun();

    int value{0};

public:
    explicit Derived(int var);
    ~Derived() override = default;

    void SetValue(int var) { value = var; }
    int GetValue() const { return value; }
};
```

场景2：public:、protected:、 private:缩进和class关键字没有对齐。

错误示例：
```CPP
class Base {
    private:        // 不符合：没有和`class`对齐
        void Foo();

    protected:      // 不符合：没有和`class`对齐
        void Bar();

    public:         // 不符合：没有和`class`对齐
        void Fun();
};
```

## 【正确示例】

场景1：类访问控制块顺序错误

修复示例：
```CPP
class Derived : public Base {
public:                                     // 符合
    explicit Derived(int var);
    ~Derived() override = default;

    void SetValue(int var) { value = var; }
    int GetValue() const { return value; }

private:
    void Fun();

    int value{0};
};
```

场景2：public:、protected:、 private:缩进和class关键字没有对齐。

修复示例：

```CPP
class Base {
private:            // 符合：和`class`对齐
    void Foo();

protected:          // 符合：和`class`对齐
    void Bar();

public:             // 符合：和`class`对齐
    void Fun();
};
```

## 【修复建议】

类访问控制块按顺序 public:、protected:、 private:声明。
public:、protected:、 private:缩进和class关键字对齐。
在各个访问控制块中，建议将类似的声明放在一起，如果项目组没有制定声明顺序，可参考如下声明顺序：
- 类型 (包括 typedef，using 和嵌套的结构体与类)
- 静态常量
- 工厂函数
- 构造函数
- 赋值操作符
- 析构函数
- 其他成员函数
- 成员变量

# G.FMT.09-CPP 构造函数初始化列表放在同一行或按4空格缩进并排多行

## 【描述】

构造函数初始化列表放在同一行，如果需要换行，则将冒号放置新行，首行缩进4空格，其余多行与首行的成员变量对齐缩进。

## 【错误示例】

场景1：初始化列表换行时格式错误

错误示例1：
```CPP
SomeClass::SomeClass(int var) : // 不符合：冒号没有换行
    someVar(var)
{
    DoSomething();
}
```

错误示例2：
```CPP
SomeClass::SomeClass(int var)
    : someVar(var),
    someOtherVar(var + 1)   // 不符合：没有与上一行的成员变量对齐
{
    DoSomething();
}
```

## 【正确示例】

场景1：初始化列表换行时格式错误

修复示例1：

```CPP
// 符合：如果所有变量能放在同一行:
SomeClass::SomeClass(int var) : someVar(var)
{
    DoSomething();
}
```

修复示例2：

```CPP
// 符合：如果需要换行, 将冒号放置新行, 并缩进4个空格
SomeClass::SomeClass(int var)
    : someVar(var), someOtherVar(var + 1)
{
    DoSomething();
}
```

修复示例3：

```CPP
// 符合：如果初始化列表需要多行, 需要逐行对齐
SomeClass::SomeClass(int var)
    : someVar(var),           // 缩进4个空格
      someOtherVar(var + 1)   // 与上一行的成员变量对齐
{  
    DoSomething();
}
```

## 【修复建议】

构造函数初始化列表放在同一行，如果需要换行，则将冒号放置新行，首行缩进4空格，其余多行与首行的成员变量对齐缩进。
请参考如下修复示例。

# G.FMT.12-CPP 避免将if/else/else if写在同一行

## 【描述】

if语句中，若有多个分支，应该写在不同行。

## 【错误示例】

场景1：else 与 if 在同一行

错误示例：

```CPP
if (someConditions) { ... } else { ... } // 不符合：else 与 if 在同一行
```

场景2：else if 与 if 在同一行

错误示例：
```CPP
if (someConditions) { ... } else if (...) { ... } else { ... } // 不符合：else 与 if 在同一行
```

## 【正确示例】

场景1：else 与 if 在同一行

修复示例：
```CPP
if (someConditions) {
    DoSomething();
    ...
} else {                                 // 符合：else 与 if 在不同行
    ...
}
```

场景2：else if 与 if 在同一行

修复示例：
```CPP
if (someConditions) {
    DoSomething();
    ...
} else if (...) { // 符合：else if 与 if 在不同行
    ...
} else {          // 符合：else 与 if 在不同行                       
    ...
}
```

## 【修复建议】

将else、else if语句拆分到多行。


# G.FMT.13-CPP case/default语句相对switch缩进一层

## 【描述】

case/default语句相对switch缩进一层，case中的语句相对case缩进一层。

## 【错误示例】
```CPP
switch (var) {
case CASE1:             // 不符合：case 未缩进  
    DoSomething1();
    break;
...
default:                // 不符合：default 未缩进  
    break;
}
```

## 【正确示例】

```CPP
switch (var) {
    case CASE1:         // 符合：缩进  
        DoSomething1(); // 符合：缩进  
        break;
    case CASE2: {       // 符合：带大括号格式  
        DoSomething2();
        break;
    }
    ...
    default:
        break;
}
```

## 【修复建议】

按要求进行缩进。

# G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--引用类型

## 【描述】

引用类型“&”可以跟随类型也可以跟随变量名或函数名，编写代码时需要选择使用其中一种风格，项目中保持风格统一。

## 【错误示例】

场景1：选用的风格是跟随变量名或函数名，但是实际开发时跟随类型

错误示例：

```CPP
int& r;                         // 不符合
struct Foo& CreateFoo(void);    // 不符合
```

场景2：选用的风格是跟随类型，但是实际开发时跟随变量名或函数名

错误示例：

```CPP
int &r = i;         // 不符合
int &&rr = i + 10;  // 不符合
int *&rp = p;       // 不符合
Foo &GetFoo();      // 不符合
```

场景3：引用符号左右都有空格或者都没有空格

错误示例：

```CPP
int&r1;   // 不符合：两边都没空格
int & r2; // 不符合：两边都有空格
```

## 【正确示例】

场景1：选用的风格是跟随变量名或函数名，但是实际开发时跟随类型

修复示例：
```CPP
int &r;                         // 符合："&"跟随变量名
struct Foo &CreateFoo(void);    // 符合: "&"跟随函数名
场景2：选用的风格是跟随类型，但是实际开发时跟随变量名或函数名
```

修复示例：
```CPP
int& r = i;         // 符合："&"跟随类型
int&& rr = i + 10;  // 符合："&"跟随类型
int*& rp = p;       // 符合："&"跟随类型
Foo& GetFoo();      // 符合："&"跟随类型
```

场景3：引用符号左右都有空格或者都没有空格

修复示例1：选用跟随变量名或函数名风格的规则并遵从。
```CPP
int &r1; // 符合
int &r2; // 符合
```

修复示例2：选用跟随类型风格的规则并遵从。
```CPP
int& r1; // 符合
int& r2; // 符合
```

## 【修复建议】

确认选用的规则是跟随类型还是跟随变量名或函数名，根据确认结果修改。

可通过格式化工具修复问题，可配置是跟随类型或跟随变量名或函数名。


# G.FMT.14-CPP 指针类型"*"和引用类型"&"只跟随类型或变量名--指针类型

## 【描述】

指针类型的类型修饰符*可以跟随类型也可以跟随变量名或函数名，编写代码时需要选择使用其中一种风格。 按照C++代码的惯例，建议使用跟随类型的风格。

当*与类型、变量、函数名之间有其他关键字而无法跟随时，可以选择向左或向右跟随关键字，或者选择与关键字之间留有1个空格，并保持风格一致。
当*由于其他原因无法跟随类型、变量、函数名时，选择一种跟随风格并保持一致。

类似地，成员指针::* 和引用类型&、&&、*&的风格也应与指针类型*的风格相同。

本条款中列举了一些应该避免出现的风格以及常见的推荐风格。

该规则有可配置参数，具体配置详见规则集中该规则的选项配置。


## 【错误示例】

场景1：选用的风格是跟随变量名或函数名，但是实际开发时跟随类型

错误示例：
```CPP
int* p2;                        // 不符合
struct Foo* CreateFoo(void);    // 不符合
```

场景2：选用的风格是跟随类型，但是实际开发时跟随变量名或函数名

错误示例：
```CPP
int i = 0;
int *p = &i;        // 不符合
Foo *CreateFoo();   // 不符合
```

场景3：指针符号左右都有空格或者都没有空格

错误示例：
```CPP
int*p3;                             // 不符合：两边都没空格
int * p4;                           // 不符合：两边都有空格
const char*const VERSION = "V100";  // 不符合：两边都没空格
```

## 【正确示例】

场景1：选用的风格是跟随变量名或函数名，但是实际开发时跟随类型

修复示例：

```CPP
int *p2;                        // 符合："*"跟随变量名
struct Foo *CreateFoo(void);    // 符合: "*"跟随函数名
```

场景2：选用的风格是跟随类型，但是实际开发时跟随变量名或函数名

修复示例：
```CPP
int i = 0;
int* p = &i;        // 符合："*"跟随类型
Foo* CreateFoo();   // 符合："*"跟随类型
```

场景3：指针符号左右都有空格或者都没有空格

修复示例1：选用跟随变量名或函数名风格的规则并遵从。

```CPP
int *p3; // 符合
int *p4; // 符合
const char *const VERSION = "V100"; // 符合：向右跟随关键字（可选"*"两边都有空格的风格）
```

修复示例2：选用跟随类型风格的规则并遵从。
```CPP
int* p3; // 符合
int* p4; // 符合
const char* const VERSION = "V100"; // 符合：跟随类型（可选"*"两边都有空格的风格）
```

## 【修复建议】

"*"和"&"按要求跟随类型或者名称。



# G.FMT.16-CPP 用空格突出关键字和重要信息--关键信息
## 【描述】

空格应该突出关键字和重要信息。总体建议如下：

- 行末不应加空格
- if, switch, case, do, while, for 等关键字之后加空格；#include之后加空格
- 小括号内部的两侧，不加空格；外部两侧与关键字或重要信息之间加空格
- 无论大括号内部两侧有无空格，左右要保持一致；外部两侧与关键字或重要信息之间加空格
- 使用大括号进行初始化时，左侧大括号跟随类型或变量名时不加空格
- 一元操作符（& * + ‐ ~ !）之后不应加空格
- 二元操作符（= + ‐ < > * / % | & ^ <= >= == !=）两侧都应加空格
- 三元操作符（? :）符号两侧都应加空格
- 前置和后置的自增、自减操作符（++ --）和变量之间不应加空格
- 尾置返回类型语法中的->前后加空格
- 结构体成员操作符（. ->）前后不应加空格
- 结构体中表示位域的冒号，两侧都应加空格
- 函数参数列表的小括号与函数名之间不应加空格
- 类型强制转换的小括号与被转换对象之间不应加空格
- 数组的中括号与数组名之间不应加空格
- 对于模板和类型转换(<>)和类型之间不加空格
- 域操作符(::)前后不加空格
- 类成员指针（::* .* ->*）前后或中间不加空格
- 类继承、构造函数初始化、范围for语句中的冒号(:)前后加空格
- 逗号、分号、冒号（上述规则场景除外）前面不加空格，后面加空格

## 【错误示例】

场景1：行尾有空格

错误示例：
```CPP
// 不符合：行尾有空格
void Foo(int x); 
```

场景2：小括号内两侧有空格

错误示例：
```CPP
void Foo( int x );  // 不符合：小括号内两侧有空格
```

场景3：逗号，分号，冒号前有空格

错误示例：
```CPP
// 不符合：第一个逗号前有多余的空格，第二个逗号后未加空格
void Foo(int x ,int y,int z);
```

场景4：#include 后无空格

错误示例：
```CPP
#include<stdio.h>  // 不符合：#include指令后无空格
```

## 【正确示例】

场景1：行尾有空格

修复示例：

```CPP
// 符合：清除行尾空格
void Foo(int x);
```

场景2：小括号内两侧有空格

修复示例：
```CPP
void Foo(int x);    // 符合：清除小括号内部两侧的空格
```

场景3：逗号，分号，冒号前有空格

修复示例：

```CPP
// 符合：逗号，分号，冒号（不含三元操作符和表示位域的冒号）前无需空格，后需要空格
void Foo(int x, int y, int z);
```

场景4：#include 后无空格

修复示例：
```CPP
#include <stdio.h>  // 符合：#include指令后加上空格
```

## 【修复建议】

按要求增加/减少空格。


# G.FMT.17-CPP 合理安排空行
## 【描述】

使用空行可以凸显出代码块的相关程度，能提升代码的可读性。但是过多的空行并不能带来更多的好处，反而可能降低代码的可读性。可参考如下建议：

- 函数内部、类型定义内部、宏内部、初始化表达式内部，不使用连续空行
- 仅在需要特别凸显代码块逻辑分组时使用连续空行，并且最多连续2个空行
- 大括号内的代码块行首之前和行尾之后不要加空行，但namespace的大括号内不作要求

## 【错误示例】

```CPP
ret = DoSomething();

if (ret != OK) {     // 不符合：返回值判断应该紧跟函数调用  
    return -1;
}
```

如下代码中，Foo与Bar函数体之间的空行多于2个空行

```CPP
int Foo()
{
    ...
}

int Bar()     
{
    ...
}
```

如下代码中，函数体内行首和行尾不应该加入空行。
```CPP
int Foo()
{

    ...

}
```

如下代码中，大括号内的代码块行首和行尾不应该加入空行。
```CPP
if (...) {

    ...

}
```

## 【修复建议】

删除多余空行。

# G.NAM.02-CPP C++文件名和类名保持一致
## 【描述】

C++的头文件和源文件中只有一个类时，C++文件名和类名保持一致。

## 【错误示例】

场景1：头文件名与类名不一致

```CPP
错误示例：
// databaseUtie.h

// 不符合：头文件名 databaseUtie 与类名 DatabaseConnection 不一致
class DatabaseConnection {};
```

场景2：头文件名不符合命名风格

错误示例：
```CPP
// 不符合：头文件名命名风格既不是大驼峰，也不是下划线小写
// databaseconnection.h

class DatabaseConnection {};
```

## 【正确示例】

场景1：头文件名与类名不一致

修复示例：
```CPP
// database_connection.h

// 符合：头文件名 database_connection 与类名 DatabaseConnection 一致
class DatabaseConnection {};
```

场景2：头文件名不符合命名风格

修复示例：文件名可以使用大驼峰或者小写下划线风格。项目组可自主选择风格并保持风格一致。
```CPP
// 符合：使用小写下划线风格，要保持风格一致
// database_connection.h

class DatabaseConnection {};
```

## 【修复建议】

头文件和cpp文件名和类名保持一致，可以使用大驼峰或者下划线小写风格。
项目组可自主选择风格并保持风格一致。









