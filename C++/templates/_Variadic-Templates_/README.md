# C++ 可变模板参数与应用

这个小项目讨论一个很常用的现代 C++ 特性：可变模板参数（variadic templates）。它解决的问题是：当函数或类需要接收“任意个类型”或“任意个参数”时，如何在编译期保持类型安全，同时避免写大量重载。

对应代码在 `src/main.cpp`。

## 1. 什么是可变模板参数

普通模板参数只有一个或固定几个：

```cpp
template <typename T>
void print_one(T value);
```

可变模板参数可以表示一组模板参数，叫参数包（parameter pack）：

```cpp
template <typename... Ts>
void print_many(Ts... args);
```

这里有两个包：

- `Ts...` 是类型参数包，表示零个或多个类型。
- `args...` 是函数参数包，表示零个或多个函数实参。

你可以把它理解成：编译器会把这一组参数整体“打包”起来，后续再按规则展开。

## 2. `typename... Ts` 和 `Ts... args` 分别是什么意思

这两个位置很容易混淆，但职责不同：

```cpp
template <typename... Ts>
void f(Ts... args);
```

- `typename... Ts`：声明一个类型参数包。
- `Ts... args`：用前面的类型参数包生成对应的函数参数包。

如果调用：

```cpp
f(42, 3.14, "hello");
```

编译器大致会推导出：

- `Ts = <int, double, const char*>`
- `args = <42, 3.14, "hello">`

也就是说，类型和实参是一一对应成组出现的。

## 3. 参数包本身不能直接用，必须展开

参数包不是一个普通变量，也不是容器。你不能直接把 `args` 当成单个对象操作，必须通过“参数包展开（pack expansion）”把它展开。

常见展开方式：

```cpp
print_one(args...);
```

这表示把 `args` 展开成：

```cpp
print_one(arg1, arg2, arg3);
```

但具体展开成什么形式，要看它所处的语法环境。

## 4. 最经典的展开方式：递归展开

在 C++17 之前，一个常见写法是“取一个参数 + 递归处理剩余参数”：

```cpp
void print_recursive() {
    std::cout << "(end)\n";
}

template <typename First, typename... Rest>
void print_recursive(const First& first, const Rest&... rest) {
    std::cout << first << ' ';
    print_recursive(rest...);
}
```

思路是：

1. 先定义一个“终止函数”处理空参数包。
2. 每次取出第一个参数。
3. 对剩余参数继续展开。

这种方式非常经典，也有助于理解模板递归。

## 5. C++17 更常见：折叠表达式

C++17 引入了 fold expression，可以更直接地对参数包做归约。

例如求和：

```cpp
template <typename... Ts>
auto sum_all(Ts... values) {
    return (values + ... + 0);
}
```

如果传入 `1, 2, 3, 4`，它大致会展开成：

```cpp
(((1 + 2) + 3) + 4) + 0
```

打印多个参数也可以写成：

```cpp
((std::cout << values << ' '), ...);
```

这比手写递归更短，也更符合现代 C++ 风格。

## 6. `sizeof...` 是什么

`sizeof...` 用来获取参数包里元素的个数：

```cpp
template <typename... Ts>
constexpr std::size_t count_types() {
    return sizeof...(Ts);
}
```

或者：

```cpp
template <typename... Ts>
void f(Ts... args) {
    std::cout << sizeof...(Ts) << '\n';
    std::cout << sizeof...(args) << '\n';
}
```

通常这两个值相同，因为类型包和函数参数包是一一对应的。

## 7. 可变模板参数在类模板里怎么用

类模板也可以接收参数包：

```cpp
template <typename... Ts>
struct TypeList {};
```

这类写法在模板元编程里很常见。你可以把它看成“仅在编译期存在的类型列表”。

例如：

```cpp
using MyTypes = TypeList<int, double, std::string>;
```

它不一定存运行时数据，但可以承载类型信息，用于类型计算、匹配、过滤、映射等。

## 8. 常见应用场景

### 场景一：通用打印或日志接口

```cpp
template <typename... Ts>
void log(Ts&&... args);
```

可以接受任意个参数，避免写很多重载版本。

### 场景二：求和、取最大值、逻辑与/或

折叠表达式非常适合做归约：

```cpp
(args + ...)
(... && args)
(... || args)
```

### 场景三：类型约束检查

例如判断一组类型是否都满足某个性质：

```cpp
std::conjunction_v<std::is_integral<Ts>...>
```

这表示“所有 `Ts` 都是整数类型吗”。

### 场景四：完美转发

标准库中很多工厂函数、容器 `emplace_back`、`std::make_unique` 背后都依赖可变模板参数和转发引用，把任意构造参数原样转发给目标对象构造函数。

这也是 variadic templates 在工程中最重要的应用之一。

## 9. 参数包展开时容易踩的坑

- 参数包不能单独使用，必须在可展开的位置写 `...`。
- 递归展开一定要有终止条件，否则无法匹配空参数包。
- 折叠表达式要注意空包情形，有时需要提供初始值，例如 `(args + ... + 0)`。
- 可变模板参数很强大，但不要把接口写得过于“万能”，否则错误信息会变复杂，可读性也会下降。

## 10. 这和 `initializer_list`、重载有什么区别

`initializer_list` 适合一组同类型元素，例如：

```cpp
sum({1, 2, 3, 4});
```

但它要求元素类型统一。

可变模板参数更灵活：

```cpp
print_many(42, 3.14, "hello", std::string{"world"});
```

这里每个参数类型都可以不同，而且编译器仍然能保留完整类型信息。

## 11. 如何运行

如果环境有 CMake：

```bash
cmake -S . -B build
cmake --build build
./build/template_variadic
```

也可以直接用 g++：

```bash
g++ -std=c++17 -Wall -Wextra -pedantic src/main.cpp -o template_variadic
./template_variadic
```
