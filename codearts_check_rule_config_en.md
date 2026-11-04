# Table

|No.|Rule(CH)|Rule(EN)|Configuration parameters|Language|Supported scenarios|
|--|--------------------|----------------------|------------|----|----------|
|1  |G.CMT.01-CPP注释符与注释内容间留有1个空格|[G.CMT.01-CPP Add a space between the comment character and comment content](#gcmt01-cpp-add-a-space-between-the-comment-character-and-comment-content)|-|C++|Version level, gate level|
|2  |G.CMT.02-CPP代码注释置于对应代码的上方或右边|[G.CMT.02-CPP Place code comments above or to the right of the code](#gcmt02-cpp-place-code-comments-above-or-to-the-right-of-the-code)|-|C++|Version level, gate level|
|3  |G.CMT.03-CPP文件头注释包含版权说明|[G.CMT.03-CPP File header comments shall contain the copyright notice](#gcmt03-cpp-file-header-comments-shall-contain-the-copyright-notice)|check_modified_year:false;is_open_source:true;corporations_zh:华为技术有限公司 &#124;海思半导体;corporations_en:Huawei Technologies Co., Ltd. &#124;Hisilicon Technologies Co., Ltd.|C++|Version level, gate level|
|4  |G.CMT.04-CPP不写空有格式的函数头注释|[G.CMT.04-CPP Do not write empty-content function header comments](#gcmt04-cpp-do-not-write-empty-content-function-header-comments)|-|C++|Version level, gate level|
|5  |G.FMT.01-CPP非纯ASCII码源文件使用UTF-8编码|[G.FMT.01-CPP Source files involving non-ASCII code should use UTF-8 formatting](#gfmt01-cpp-source-files-involving-non-ascii-code-should-use-utf-8-formatting)|-|C++|Version level, gate level|
|6  |G.FMT.02-CPP使用空格进行缩进，每次缩进4个空格|[G.FMT.02-CPP Use spaces to indent and indent four spaces at a time](#gfmt02-cpp-use-spaces-to-indent-and-indent-four-spaces-at-a-time)|-|C++|Version level, gate level|
|7  |G.FMT.04-CPP每个变量单独一行进行声明或赋值|[G.FMT.04-CPP Each variable is declared or assigned on a separate line](#gfmt04-cpp-each-variable-is-declared-or-assigned-on-a-separate-line)|check_multiple_var_dec:true|C++|Version level, gate level|
|8  |G.FMT.05-CPP行宽不超过120个字符|[G.FMT.05-CPP Each line contains no more than 120 characters](#gfmt05-cpp-each-line-contains-no-more-than-120-characters)|column_width:120|C++|Version level, gate level|
|9  |G.FMT.06-CPP换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数调用参数换行|[G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Function call parameter wrapping](#gfmt06-cpp-while-line-breaking-leave-the-operators-at-the-end-and-indent-or-align-the-new-line--function-call-parameter-wrapping)|c_coding_standard_v5.0:true|C++|Version level, gate level|
|10  |G.FMT.06-CPP换行时将操作符留在行末，新行缩进一层或进行同类对齐--函数声明参数换行|[G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Function declaration parameter wrapping](#gfmt06-cpp-while-line-breaking-leave-the-operators-at-the-end-and-indent-or-align-the-new-line--function-declaration-parameter-wrapping)|-|C++|Version level, gate level|
|11  |G.FMT.06-CPP换行时将操作符留在行末，新行缩进一层或进行同类对齐--数据初始化换行|[G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Data initialization wrapping](#gfmt06-cpp-while-line-breaking-leave-the-operators-at-the-end-and-indent-or-align-the-new-line--data-initialization-wrapping)|-|C++|Version level, gate level|
|12  |G.FMT.07-CPP预处理的"#"统一放在行首，嵌套预处理语句时，"#"可以进行缩进|G.FMT.07-CPP The number sign (#) that starts a preprocessor directive should be placed at the beginning of a line and can be indented in nested|-|C++|Version level, gate level|
|13  |G.FMT.08-CPP遵循传统的类成员声明顺序|[G.FMT.08-CPP Use a conventional class member declaration order](#gfmt08-cpp-use-a-conventional-class-member-declaration-order)|-|C++|Version level, gate level|
|14  |G.FMT.09-CPP构造函数初始化列表放在同一行或按4空格缩进并排多行|[G.FMT.09-CPP A constructor initialization list is placed on the same line or on different lines indented by four spaces](#gfmt09-cpp-a-constructor-initialization-list-is-placed-on-the-same-line-or-on-different-lines-indented-by-four-spaces)|-|C++|Version level, gate level|
|15  |G.FMT.12-CPP避免将if/else/else if写在同一行|[G.FMT.12-CPP Avoid placing if/else/else if on the same line](#gfmt12-cpp-avoid-placing-ifelseelse-if-on-the-same-line)|-|C++|Version level, gate level|
|16  |G.FMT.13-CPP case/default语句相对switch缩进一层|[G.FMT.13-CPP Indent the case or default statement in a switch statement](#gfmt13-cpp-indent-the-case-or-default-statement-in-a-switch-statement)|-|C++|Version level, gate level|
|17  |G.FMT.14-CPP指针类型"*"和引用类型"&"只跟随类型或变量名--引用类型|G.FMT.14-CPP The pointer type "*" and reference type "&" can only be placed close to a variable name or a type--Reference type|-|C++|Version level, gate level|
|18  |G.FMT.14-CPP指针类型"*"和引用类型"&"只跟随类型或变量名--指针类型|G.FMT.14-CPP The pointer type "*" and reference type "&" can only be placed close to a variable name or a type--pointer type|c_coding_standard_v5.0:true;pointer_follow_type_or_var:2|C++|Version level, gate level|
|19  |G.FMT.16-CPP用空格突出关键字和重要信息--关键信息|[G.FMT.16-CPP Use spaces to highlight keywords and important information--Key Information](#gfmt16-cpp-use-spaces-to-highlight-keywords-and-important-information--key-information)|-|C++|Version level, gate level|
|20  |G.FMT.17-CPP合理安排空行|[G.FMT.17-CPP Use blank lines when necessary](#gfmt17-cpp-use-blank-lines-when-necessary)|-|C++|Version level, gate level|
|21  |G.NAM.02-CPP C++文件名和类名保持一致|[G.NAM.02-CPP Keep C++ file names and class names consistent](#gnam02-cpp-keep-c-file-names-and-class-names-consistent)|-|C++|Version level, gate level|
|22  |G.FMT.03-CPP使用统一的大括号换行风格|[G.FMT.03-CPP Use the consistent brace style for new lines](#gfmt03-cpp-use-the-consistent-brace-style-for-new-lines)|c_coding_standard_v5.0:true;StyleOfBrace:K&R|C++|Version level, gate level|

# G.CMT.01-CPP Add a space between the comment character and comment content

## Description

Leave one space between the comment character and the comment content. Comment characters include /**/ and //. Keep the alignment of comment characters the same as in the following examples:

Use /**/

```text
/* Single-line comment */
/*
 * Multi-line comment
 * Second line
 */
```

Use //

```text
// Single-line comment
// Multi-line comment
// Second line
```

In a multi-line comment, multiple spaces may exist between the comment character and the comment content to align wrapped lines:

```text
// This is an example with multiple spaces between the comment character and the comment content:
// XX function description
// Note: Note information. Wrap the line when the line width of one line is exceeded
//       This is the second line, aligned with spaces at the front
```

## Repair suggestion

Add or remove spaces as required.

# G.CMT.02-CPP Place code comments above or to the right of the code

## Description

Comments on code should be placed above or to the right of the corresponding code.

- A comment above the code: keep no blank line between it and the code line, and keep the same indentation as the code
- A comment to the right of the code: keep at least one space between it and the code

Usually the comment on the right should not be too long. Consider placing the comment above the code when the comment exceeds the line width.

```CPP
// Comment on the Foo() function
int Foo()
{
    int b = 0; // Comment on variable b, at least one space from the code
    ...
}
```

When appropriate, aligning the comments on the right vertically looks better:

```CPP
const int A_CONST = 100;       // The two comments here are of the same type
const int ANOTHER_CONST = 200; // They can keep left alignment
```

## Incorrect example

Scenario 1: No space between the comment character and the code
Incorrect example:

```CPP
int x;// Non-compliant: no space between the comment character and the code

int y;/* Non-compliant: no space between the comment character and the code */
```

Scenario 2: The comment is in the middle of the code
Incorrect example:

```CPP
int y /* comment */ = 1;       // Non-compliant: the comment is in the middle of the code
```

## Correct example

Scenario 1: No space between the comment character and the code
Corrected example:

```CPP
int x; // Compliant: at least one space between the comment character and the code

int y; /* Compliant: at least one space between the comment character and the code */
```

Scenario 2: The comment is in the middle of the code
Corrected example 1:

```CPP
int y = 1; // Comment          // Compliant: the comment is to the right of the code
```

Corrected example 2:

```CPP
int y = 1; /* Comment */       // Compliant: the comment is to the right of the code
```

Corrected example 3:

```CPP
/*
 * Comment
 * Second line
 */
int y = 1;                  // Compliant: the comment is above the code
```

## Repair suggestion

Place code comments as required.

# G.CMT.03-CPP File header comments shall contain the copyright notice

## Description

The file header comment should contain the copyright notice first.
If the file header comment needs other content, you can add it afterwards.
Examples: file function description, author, creation date, precautions, and so on.

The copyright license content and format must be as follows:
Chinese: 版权所有(c)华为技术有限公司2012-2018
Or English: Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.

The copyright license content is explained as follows:

- 2012-2018 can be modified based on actual needs.
- 2012 is the year when the file was first created, and 2018 is the year of the last file modification.
- When the file is significantly modified, update the later year, for example, feature extension or major refactoring.
- You can write only one creation year, and then you do not need to update the copyright notice for subsequent file modifications.
- For example: 版权所有(c)华为技术有限公司2018
- The copyright notice can use a Huawei subsidiary, for example:
- Chinese: 版权所有(c)海思半导体2012-2018
- Or English: Copyright (c) Hisilicon Technologies Co., Ltd. 2012-2018. All rights reserved.
- 'All rights reserved.' in English currently has no mandatory legal effect. You can omit it or continue to use it by convention.

Example of a file header comment:

```CPP
/*
* 版权所有(c)华为技术有限公司2012-2018
* XX function implementation
* Note:
* - Note 1
* - Note 2
*/
```

Note the following when writing a file header comment:

The file header comment should start at the top of the file.
If a comment of the 'critical asset description' type is included, it should follow immediately.
Keep a unified format.
The project or a larger scope defines the specific format in a unified manner. For the format, refer to the preceding example.
Keep the layout tidy. If the content is too long and exceeds the line width requirement, align the wrapped lines.
For alignment, refer to 'Note' in the preceding example.
Include the 'copyright license' first, and then include other optional content.
Add other options as needed, and keep the format unified.
Do not write comments that have format but no content.
For example, in the preceding example, no content follows 'Note:'.
This rule has configurable parameters. For details, see the option configuration of this rule in the rule set.

## Correct example

```CPP
/*
 * 版权所有(c)华为技术有限公司2012-2018
 * XX function implementation
 * Note:
 *      - Note 1
 *      - Note 2
 */
```

## Repair suggestion

Include the copyright notice correctly.

# G.CMT.04-CPP Do not write empty-content function header comments

## Description

Use the function name to self-document its function as much as possible, and write function header comments as needed. Do not write function header comments that have format but no content; do not write useless or information-redundant function header comments.

Function header comments include the comment at the function declaration and the comment at the function definition. These two types of comments have different goals and should have different content:

- The comment at the function declaration aims to convey information that the function prototype cannot express but readers should know. It describes the function's purpose and how to use it
- The comment at the function definition aims to convey function implementation details that readers should know, for example, the algorithm used and why the implementation is designed this way

## Incorrect example

The following function header comment has two problems. Avoid similar comments:

- Parameters and return value have format but no content
- The function name in the comment is redundant

```CPP
/*
 * Function name: WriteData
 * Function: write a string
 * Parameters:
 * Return value:
 */
int WriteData(const unsigned char* buf, size_t len);
```

## Correct example

Place the function header comment uniformly above the function declaration or definition, and choose one of the following styles:

Use //

```CPP
// Single-line function header
int Fun1();

// Multi-line function header
// Second line
int Fun2();
```

Use /**/

```CPP
/* Single-line function header */
int Fun1();

/*
 * Single-line or multi-line function header
 * Second line
 */
int Fun2();
```

The function header comment content is optional. The content is not limited to: function description, return value, performance constraints, usage, memory conventions, re-entrancy requirements, and so on.

## Repair suggestion

Add content or delete comments that have format but no content.

# G.FMT.01-CPP Source files involving non-ASCII code should use UTF-8 formatting

## Description

For source files that contain non-ASCII characters, use UTF-8 encoding.

## Repair suggestion

Use UTF-8 encoding.

# G.FMT.02-CPP Use spaces to indent and indent four spaces at a time

## Description

Use spaces for indentation, four spaces at a time. Avoid using tab characters ('\t') for indentation.
Currently almost all integrated development environments (IDEs) and code editors support configuring the Tab key to automatically expand to four spaces.

## Repair suggestion

Indent as required.

# G.FMT.03-CPP Use the consistent brace style for new lines

## Description

Choose and uniformly use one brace line-breaking style, and avoid mixing multiple styles.

**K&R style**
When breaking lines, place the left brace of a function (excluding lambda expressions) on a new line at the start of the line and on its own line; place other left braces at the end of the line following the statement.
Place the right brace on its own line, unless the rest of the same statement follows it, such as while in a do statement, else/else if in an if statement, or a comma, semicolon, or parenthesis.

**Allman style**
When breaking lines, place the left brace on a new line on its own and keep the same indentation as the previous line.
Place the right brace on its own line, unless while in a do statement, a comma, or a semicolon follows it.

## Repair suggestion

Use the corresponding brace line-breaking style.

# G.FMT.04-CPP Each variable is declared or assigned on a separate line

## Description

A variable declaration or assignment statement should occupy a separate line, which makes the code easier to read and understand.
This rule has configurable parameters. For details, see the option configuration of this rule in the rule set.

## Incorrect example

```CPP
int count = 10; bool isCompleted = false; // Non-compliant: multiple variable initializations need to be placed on separate lines

char* str, ch, arr[10];                   // Non-compliant: multiple variables of different types are defined, which easily causes misunderstanding
int a = 0, b = 0;                         // Non-compliant: multiple variable definitions need to be placed on separate lines

int x;
int y;
...
x = 1; y = 2;                             // Non-compliant: multiple assignment statements exist on one line
```

## Correct example

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

## Repair suggestion

Split the statements into multiple lines.

# G.FMT.05-CPP Each line contains no more than 120 characters

## Description

The code line width should not be too long; otherwise reading becomes difficult.

Controlling the line width indirectly guides programmers to shorten function and variable names, reduce nesting levels, and improve code readability.

Keep no more than 120 characters per line, unless exceeding 120 significantly improves readability and does not hide information.

## Repair suggestion

Break the line at an appropriate position.

# G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Function call parameter wrapping

## Description

When a statement is too long or has poor readability, break the line at an appropriate position.
When breaking a line, place the operator at the end of the line, which means 'not finished, more follows'. Indent the new line by one level, or keep alignment of the same type.

When the parameter list of a function call wraps, the left parenthesis always follows the function name, the right parenthesis always follows the last parameter, and the parameters are aligned properly.

Generally break the line after a lower-precedence operator, or break the line according to expression levels.

## Incorrect example

Scenario 1: The parameter list of a function call is not indented or aligned when wrapped

Incorrect example:

```CPP
void Fun(int ret)
{
    WriteLog(FWM_ETH_LOG_ERROR, FWM_ETH_CATEGORY_DEFAULT, FWM_ETH_MID_VLAN,
         "[DOWNLOAD] ret:%u", ret); // Non-compliant: 5 spaces are used for indentation when wrapping
}
```

## Correct example

Scenario 1: The parameter list of a function call is not indented or aligned when wrapped

Corrected example 1:

```CPP
void Fun(int ret)
{
    WriteLog(FWM_ETH_LOG_ERROR, FWM_ETH_CATEGORY_DEFAULT, FWM_ETH_MID_VLAN,
        "[DOWNLOAD] ret:%u", ret); // Compliant: indented 4 spaces relative to the previous line
}
```

Corrected example 2:

```CPP
void Fun(int ret)
{
    WriteLog(FWM_ETH_LOG_ERROR, FWM_ETH_CATEGORY_DEFAULT, FWM_ETH_MID_VLAN,
             "[DOWNLOAD] ret:%u", ret); // Compliant: aligned with the parameter above
}
```

## Repair suggestion

Align as required.

# G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Function declaration parameter wrapping

## Description

When a statement is too long or has poor readability, break the line at an appropriate position.
When breaking a line, place the operator at the end of the line, which means 'not finished, more follows'. Indent the new line by one level, or keep alignment of the same type.

When the parameter list of a function call wraps, the left parenthesis always follows the function name, the right parenthesis always follows the last parameter, and the parameters are aligned properly.

Generally break the line after a lower-precedence operator, or break the line according to expression levels.

## Incorrect example

Scenario 1: The parameter list is not indented or aligned when the function declaration wraps

Incorrect example:

```CPP
VOS_UINT32 Download(VOS_UINT32 type, VOS_CHAR *pData,
 VOS_UINT32 dataLen, VOS_UINT32 dataNum) // Non-compliant: no indentation or alignment when wrapping
```

## Correct example

Scenario 1: The parameter list is not indented or aligned when the function declaration wraps

Corrected example 1:

```CPP
VOS_UINT32 Download(VOS_UINT32 type, VOS_CHAR *pData,
    VOS_UINT32 dataLen, VOS_UINT32 dataNum) // Compliant: indented 4 spaces relative to the previous line
```

Corrected example 2:

```CPP
VOS_UINT32 Download(VOS_UINT32 type, VOS_CHAR *pData,
                    VOS_UINT32 dataLen, VOS_UINT32 dataNum) // Compliant: aligned with the parameter above

```

## Repair suggestion

Align as required.

# G.FMT.06-CPP While line breaking, leave the operators at the end and indent or align the new line--Data initialization wrapping

## Description

When a statement is too long or has poor readability, break the line at an appropriate position.
When breaking a line, place the operator at the end of the line, which means 'not finished, more follows'. Indent the new line by one level, or keep alignment of the same type.

When the parameter list of a function call wraps, the left parenthesis always follows the function name, the right parenthesis always follows the last parameter, and the parameters are aligned properly.

Generally break the line after a lower-precedence operator, or break the line according to expression levels.

## Incorrect example

Scenario 1: Array initialization has no indentation alignment at the same level

Incorrect example:

```CPP
int a[][2] = {
    {1, 2},        // {1, 2}, {3, 4}, and {5, 6} are at the same level
        {3, 4},    // Non-compliant: no indentation alignment at the same level
            {5, 6} // Non-compliant: no indentation alignment at the same level
};
```

Scenario 2: The nested level of array initialization is not indented 4 spaces relative to the previous level

Incorrect example:

```CPP
int a[][2] = {
{1, 2},        // Non-compliant: not indented 4 spaces relative to the previous level
{3, 4},
{5, 6}
};
```

## Correct example

Scenario 1: Array initialization has no indentation alignment at the same level

Corrected example:

```CPP
int a[][2] = {
    {1, 2},        // {1, 2}, {3, 4}, and {5, 6} are at the same level
    {3, 4},        // Compliant: indented and aligned at the same level
    {5, 6}         // Compliant: indented and aligned at the same level
};
```

Scenario 2: The nested level of array initialization is not indented 4 spaces relative to the previous level

Corrected example:

```CPP
int a[][2] = {
    {1, 2},    // Compliant: indented 4 spaces relative to the previous level
    {3, 4},
    {5, 6}
};
```

## Repair suggestion

Align as required.

# G.FMT.07-CPP The number sign (#) that starts a preprocessor directive should be placed at the beginning of a line and can be indented in nested

## Description

Place the # of preprocessor directives uniformly at the beginning of a line.
The # of preprocessor directives in a function body should be placed at the beginning of a line.
The nested # in preprocessor statements can be indented and aligned as required to distinguish levels.

## Incorrect example

Scenario 1: The # of a preprocessor directive is not at the beginning of a line

Incorrect example:

```CPP
// Non-compliant: "#" is not at the beginning of a line
    #if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
// Non-compliant: "#" is not at the beginning of a line
    #define ATOMIC_X86_HAS_CMPXCHG16B 1
#else
#define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif
```

Scenario 2: The # of a preprocessor directive in a function body is not at the beginning of a line.

Incorrect example:

```CPP
int FunctionName()
{
    if (someThingError) {
        ...
        #ifdef HAS_SYSLOG // Non-compliant: `#` is not at the beginning of a line
            WriteToSysLog();
        #else             // Non-compliant: `#` is not at the beginning of a line
            WriteToFileLog();
        #endif            // Non-compliant: `#` is not at the beginning of a line
    }
}
```

## Correct example

Scenario 1: The # of a preprocessor directive is not at the beginning of a line

Corrected example 1: Place the # of preprocessor directives uniformly at the beginning of a line.

```CPP
// Compliant: "#" is placed at the beginning of a line
#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
// Compliant: "#" is placed at the beginning of a line
#define ATOMIC_X86_HAS_CMPXCHG16B 1
#else
#define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif
```

Corrected example 2: The nested # in preprocessor statements can be indented and aligned as required.

```CPP
// Compliant: "#" is placed at the beginning of a line
#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
    // Compliant: levels are distinguished for easier reading
    #define ATOMIC_X86_HAS_CMPXCHG16B 1
#else
    #define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif
```

Scenario 2: The # of a preprocessor directive in a function body is not at the beginning of a line.

Corrected example 1: The # of preprocessor directives in a function body should be placed at the beginning of a line.

```CPP
int FunctionName()
{
    if (someThingError) {
        ...
#ifdef HAS_SYSLOG         // Compliant: "#" is placed at the beginning of a line even inside a function
        WriteToSysLog();
#else                     // Compliant: "#" is placed at the beginning of a line even inside a function
        WriteToFileLog();
#endif                    // Compliant: "#" is placed at the beginning of a line even inside a function
    }
}
```

## Repair suggestion

Place the # of preprocessor directives uniformly at the beginning of a line.
The nested # in preprocessor statements can be indented and aligned as required.

# G.FMT.08-CPP Use a conventional class member declaration order

## Description

By default, the declaration order of class access control blocks is public:, protected:, and private: in sequence, and the indentation is aligned with the class keyword.

## Incorrect example

Scenario 1: The order of class access control blocks is incorrect

Incorrect example:

```CPP
class Derived : public Base {
private:                                    // Non-compliant: private should come after public
    void Fun();

    int value{0};

public:
    explicit Derived(int var);
    ~Derived() override = default;

    void SetValue(int var) { value = var; }
    int GetValue() const { return value; }
};
```

Scenario 2: The indentation of public:, protected:, and private: is not aligned with the class keyword.

Incorrect example:

```CPP
class Base {
    private:        // Non-compliant: not aligned with `class`
        void Foo();

    protected:      // Non-compliant: not aligned with `class`
        void Bar();

    public:         // Non-compliant: not aligned with `class`
        void Fun();
};
```

## Correct example

Scenario 1: The order of class access control blocks is incorrect

Corrected example:

```CPP
class Derived : public Base {
public:                                     // Compliant
    explicit Derived(int var);
    ~Derived() override = default;

    void SetValue(int var) { value = var; }
    int GetValue() const { return value; }

private:
    void Fun();

    int value{0};
};
```

Scenario 2: The indentation of public:, protected:, and private: is not aligned with the class keyword.

Corrected example:

```CPP
class Base {
private:            // Compliant: aligned with `class`
    void Foo();

protected:          // Compliant: aligned with `class`
    void Bar();

public:             // Compliant: aligned with `class`
    void Fun();
};
```

## Repair suggestion

Declare class access control blocks in the order public:, protected:, private:.
Align the indentation of public:, protected:, and private: with the class keyword.
In each access control block, place similar declarations together. If the project team has not defined a declaration order, refer to the following declaration order:

- Types (including typedef, using, and nested structs and classes)
- Static constants
- Factory functions
- Constructors
- Assignment operators
- Destructors
- Other member functions
- Member variables

# G.FMT.09-CPP A constructor initialization list is placed on the same line or on different lines indented by four spaces

## Description

Place the constructor initialization list on the same line. If wrapping is needed, place the colon on a new line, indent the first line by 4 spaces, and align the other lines with the member variable on the first line.

## Incorrect example

Scenario 1: The initialization list has an incorrect format when wrapped

Incorrect example 1:

```CPP
SomeClass::SomeClass(int var) : // Non-compliant: the colon does not wrap
    someVar(var)
{
    DoSomething();
}
```

Incorrect example 2:

```CPP
SomeClass::SomeClass(int var)
    : someVar(var),
    someOtherVar(var + 1)   // Non-compliant: not aligned with the member variable on the previous line
{
    DoSomething();
}
```

## Correct example

Scenario 1: The initialization list has an incorrect format when wrapped

Corrected example 1:

```CPP
// Compliant: if all variables fit on the same line:
SomeClass::SomeClass(int var) : someVar(var)
{
    DoSomething();
}
```

Corrected example 2:

```CPP
// Compliant: if wrapping is needed, place the colon on a new line and indent 4 spaces
SomeClass::SomeClass(int var)
    : someVar(var), someOtherVar(var + 1)
{
    DoSomething();
}
```

Corrected example 3:

```CPP
// Compliant: if the initialization list needs multiple lines, align them line by line
SomeClass::SomeClass(int var)
    : someVar(var),           // Indent 4 spaces
      someOtherVar(var + 1)   // Aligned with the member variable on the previous line
{
    DoSomething();
}
```

## Repair suggestion

Place the constructor initialization list on the same line. If wrapping is needed, place the colon on a new line, indent the first line by 4 spaces, and align the other lines with the member variable on the first line.
For details, see the following corrected example.

# G.FMT.12-CPP Avoid placing if/else/else if on the same line

## Description

In an if statement, if multiple branches exist, write them on different lines.

## Incorrect example

Scenario 1: else and if are on the same line

Incorrect example:

```CPP
if (someConditions) { ... } else { ... } // Non-compliant: else and if are on the same line
```

Scenario 2: else if and if are on the same line

Incorrect example:

```CPP
if (someConditions) { ... } else if (...) { ... } else { ... } // Non-compliant: else if and if are on the same line
```

## Correct example

Scenario 1: else and if are on the same line

Corrected example:

```CPP
if (someConditions) {
    DoSomething();
    ...
} else {                                 // Compliant: else and if are on different lines
    ...
}
```

Scenario 2: else if and if are on the same line

Corrected example:

```CPP
if (someConditions) {
    DoSomething();
    ...
} else if (...) { // Compliant: else if and if are on different lines
    ...
} else {          // Compliant: else and if are on different lines
    ...
}
```

## Repair suggestion

Split the else and else if statements into multiple lines.

# G.FMT.13-CPP Indent the case or default statement in a switch statement

## Description

Indent the case/default statement one level relative to switch, and indent the statements in case one level relative to case.

## Incorrect example

```CPP
switch (var) {
case CASE1:             // Non-compliant: case is not indented
    DoSomething1();
    break;
...
default:                // Non-compliant: default is not indented
    break;
}
```

## Correct example

```CPP
switch (var) {
    case CASE1:         // Compliant: indented
        DoSomething1(); // Compliant: indented
        break;
    case CASE2: {       // Compliant: format with braces
        DoSomething2();
        break;
    }
    ...
    default:
        break;
}
```

## Repair suggestion

Indent as required.

# G.FMT.14-CPP The pointer type "*" and reference type "&" can only be placed close to a variable name or a type--Reference type

## Description

The reference type "&" can follow either the type or the variable name or function name. When writing code, choose one of the two styles and keep the style unified in the project.

## Incorrect example

Scenario 1: The selected style is following the variable name or function name, but the type is followed in actual development

Incorrect example:

```CPP
int& r;                         // Non-compliant
struct Foo& CreateFoo(void);    // Non-compliant
```

Scenario 2: The selected style is following the type, but the variable name or function name is followed in actual development

Incorrect example:

```CPP
int &r = i;         // Non-compliant
int &&rr = i + 10;  // Non-compliant
int *&rp = p;       // Non-compliant
Foo &GetFoo();      // Non-compliant
```

Scenario 3: The reference symbol has spaces on both sides or no spaces on both sides

Incorrect example:

```CPP
int&r1;   // Non-compliant: no spaces on both sides
int & r2; // Non-compliant: spaces on both sides
```

## Correct example

Scenario 1: The selected style is following the variable name or function name, but the type is followed in actual development

Corrected example:

```CPP
int &r;                         // Compliant: "&" follows the variable name
struct Foo &CreateFoo(void);    // Compliant: "&" follows the function name
```

Scenario 2: The selected style is following the type, but the variable name or function name is followed in actual development

Corrected example:

```CPP
int& r = i;         // Compliant: "&" follows the type
int&& rr = i + 10;  // Compliant: "&" follows the type
int*& rp = p;       // Compliant: "&" follows the type
Foo& GetFoo();      // Compliant: "&" follows the type
```

Scenario 3: The reference symbol has spaces on both sides or no spaces on both sides

Corrected example 1: Select the rule of following the variable name or function name style and follow it.

```CPP
int &r1; // Compliant
int &r2; // Compliant
```

Corrected example 2: Select the rule of following the type style and follow it.

```CPP
int& r1; // Compliant
int& r2; // Compliant
```

## Repair suggestion

Confirm whether the selected rule follows the type or the variable name or function name, and modify the code based on the confirmation result.

You can fix the problem with a formatting tool, which can be configured to follow the type or the variable name or function name.

# G.FMT.14-CPP The pointer type "*" and reference type "&" can only be placed close to a variable name or a type--pointer type

## Description

The type modifier * of the pointer type can follow either the type or the variable name or function name. When writing code, choose one of the two styles. According to the convention of C++ code, use the style of following the type.

When * cannot follow the type, variable, or function name because another keyword exists between them, choose to follow the keyword to the left or right, or leave one space between them, and keep the style consistent.
When * cannot follow the type, variable, or function name for other reasons, choose one following style and keep it consistent.

Similarly, the style of member pointers ::* and reference types &, &&, and *& should be the same as the style of the pointer type *.

This clause lists some styles that should be avoided and common recommended styles.

This rule has configurable parameters. For details, see the option configuration of this rule in the rule set.

## Incorrect example

Scenario 1: The selected style is following the variable name or function name, but the type is followed in actual development

Incorrect example:

```CPP
int* p2;                        // Non-compliant
struct Foo* CreateFoo(void);    // Non-compliant
```

Scenario 2: The selected style is following the type, but the variable name or function name is followed in actual development

Incorrect example:

```CPP
int i = 0;
int *p = &i;        // Non-compliant
Foo *CreateFoo();   // Non-compliant
```

Scenario 3: The pointer symbol has spaces on both sides or no spaces on both sides

Incorrect example:

```CPP
int*p3;                             // Non-compliant: no spaces on both sides
int * p4;                           // Non-compliant: spaces on both sides
const char*const VERSION = "V100";  // Non-compliant: no spaces on both sides
```

## Correct example

Scenario 1: The selected style is following the variable name or function name, but the type is followed in actual development

Corrected example:

```CPP
int *p2;                        // Compliant: "*" follows the variable name
struct Foo *CreateFoo(void);    // Compliant: "*" follows the function name
```

Scenario 2: The selected style is following the type, but the variable name or function name is followed in actual development

Corrected example:

```CPP
int i = 0;
int* p = &i;        // Compliant: "*" follows the type
Foo* CreateFoo();   // Compliant: "*" follows the type
```

Scenario 3: The pointer symbol has spaces on both sides or no spaces on both sides

Corrected example 1: Select the rule of following the variable name or function name style and follow it.

```CPP
int *p3; // Compliant
int *p4; // Compliant
const char *const VERSION = "V100"; // Compliant: follows the keyword to the right (the style with spaces on both sides of "*" is optional)
```

Corrected example 2: Select the rule of following the type style and follow it.

```CPP
int* p3; // Compliant
int* p4; // Compliant
const char* const VERSION = "V100"; // Compliant: follows the type (the style with spaces on both sides of "*" is optional)
```

## Repair suggestion

Have "*" and "&" follow the type or the name as required.

# G.FMT.16-CPP Use spaces to highlight keywords and important information--Key Information

## Description

Spaces should highlight keywords and important information. The overall suggestions are as follows:

- No space should be added at the end of a line
- Add a space after keywords such as if, switch, case, do, while, and for; add a space after #include
- Do not add spaces on the two inner sides of parentheses; add spaces between the two outer sides and keywords or important information
- Whether or not spaces exist on the two inner sides of braces, keep the left and right sides consistent; add spaces between the two outer sides and keywords or important information
- When initializing with braces, do not add a space when the left brace follows the type or variable name
- Do not add a space after unary operators (& * + ‐ ~ !)
- Add spaces on both sides of binary operators (= + ‐ < > * / % | & ^ <= >= == !=)
- Add spaces on both sides of the ternary operator (? :)
- Do not add a space between prefix and postfix increment and decrement operators (++ --) and the variable
- Add spaces before and after -> in the trailing return type syntax
- Do not add spaces before or after the struct member operators (. ->)
- Add spaces on both sides of the colon representing a bit-field in a struct
- Do not add a space between the parentheses of the function parameter list and the function name
- Do not add a space between the parentheses of a type cast and the object to be converted
- Do not add a space between the brackets of an array and the array name
- Do not add a space between templates and type conversions (<>) and the type
- Do not add spaces before or after the scope operator (::)
- Do not add spaces before, after, or between the class member pointer operators (::*.* ->*)
- Add spaces before and after the colon (:) in class inheritance, constructor initialization, and range-based for statements
- Do not add a space before a comma, semicolon, or colon (except in the preceding rule scenarios), and add a space after it

## Incorrect example

Scenario 1: A space exists at the end of a line

Incorrect example:

```CPP
// Non-compliant: a space exists at the end of a line
void Foo(int x);    
```

Scenario 2: Spaces exist on the two inner sides of parentheses

Incorrect example:

```CPP
void Foo( int x );  // Non-compliant: spaces exist on the two inner sides of parentheses
```

Scenario 3: Spaces exist before commas, semicolons, and colons

Incorrect example:

```CPP
// Non-compliant: an extra space exists before the first comma, and no space is added after the second comma
void Foo(int x ,int y,int z);
```

Scenario 4: No space after #include

Incorrect example:

```CPP
#include<stdio.h>  // Non-compliant: no space after the #include directive
```

## Correct example

Scenario 1: A space exists at the end of a line

Corrected example:

```CPP
// Compliant: spaces at the end of the line are removed
void Foo(int x);
```

Scenario 2: Spaces exist on the two inner sides of parentheses

Corrected example:

```CPP
void Foo(int x);    // Compliant: spaces on the two inner sides of the parentheses are removed
```

Scenario 3: Spaces exist before commas, semicolons, and colons

Corrected example:

```CPP
// Compliant: no space is needed before a comma, semicolon, or colon (excluding the ternary operator and the colon representing a bit-field), and a space is needed after it
void Foo(int x, int y, int z);
```

Scenario 4: No space after #include

Corrected example:

```CPP
#include <stdio.h>  // Compliant: a space is added after the #include directive
```

## Repair suggestion

Add/remove spaces as required.

# G.FMT.17-CPP Use blank lines when necessary

## Description

Using blank lines highlights the relevance between code blocks and improves code readability. However, too many blank lines do not bring more benefits and may reduce code readability. Refer to the following suggestions:

- Do not use consecutive blank lines inside functions, type definitions, macros, or initialization expressions
- Use consecutive blank lines only when the logical grouping of code blocks needs to be specially highlighted, and use at most 2 consecutive blank lines
- Do not add blank lines before the start or after the end of a code block inside braces, except inside the braces of a namespace

## Incorrect example

```CPP
ret = DoSomething();

if (ret != OK) {     // Non-compliant: the return value check should immediately follow the function call
    return -1;
}
```

In the following code, more than 2 blank lines exist between the function bodies of Foo and Bar

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

In the following code, blank lines should not be added at the start or end of the function body.

```CPP
int Foo()
{

    ...

}
```

In the following code, blank lines should not be added at the start or end of the code block inside the braces.

```CPP
if (...) {

    ...

}
```

## Repair suggestion

Delete extra blank lines.

# G.NAM.02-CPP Keep C++ file names and class names consistent

## Description

When only one class exists in a C++ header file and source file, keep the C++ file name consistent with the class name.

## Incorrect example

Scenario 1: The header file name is inconsistent with the class name

```CPP
Incorrect example:
// databaseUtie.h

// Non-compliant: the header file name databaseUtie is inconsistent with the class name DatabaseConnection
class DatabaseConnection {};
```

Scenario 2: The header file name does not conform to the naming style

Incorrect example:

```CPP
// Non-compliant: the header file name style is neither UpperCamelCase nor lowercase with underscores
// databaseconnection.h

class DatabaseConnection {};
```

## Correct example

Scenario 1: The header file name is inconsistent with the class name

Corrected example:

```CPP
// database_connection.h

// Compliant: the header file name database_connection is consistent with the class name DatabaseConnection
class DatabaseConnection {};
```

Scenario 2: The header file name does not conform to the naming style

Corrected example: The file name can use the UpperCamelCase style or the lowercase underscore style. The project team can choose a style and keep the style consistent.

```CPP
// Compliant: the lowercase underscore style is used, and the style should be kept consistent
// database_connection.h

class DatabaseConnection {};
```

## Repair suggestion

Keep the header file name and cpp file name consistent with the class name. You can use the UpperCamelCase style or the lowercase underscore style.
The project team can choose a style and keep the style consistent.
