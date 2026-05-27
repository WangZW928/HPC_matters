# C++ 模板特化与编译期 If-Then-Else

这个项目讨论一个很核心的模板元编程问题：模板如何在编译期根据类型或常量做选择。

对应代码在 `src/main.cpp`。

## 1. 什么是类模板的特化与偏特化

类模板通常先有一个主模板：

```cpp
template <typename T>
struct TypeName {
    static const char* value() {
        return "general type";
    }
};
```

主模板表示“默认情况”。如果没有更合适的版本，编译器就用它。

全特化表示：模板参数完全确定时，提供一个专门版本。

```cpp
template <>
struct TypeName<int> {
    static const char* value() {
        return "int full specialization";
    }
};
```

偏特化表示：只固定一部分模式，仍然保留模板参数。

```cpp
template <typename T>
struct TypeName<T*> {
    static const char* value() {
        return "pointer partial specialization";
    }
};
```

这里不是针对某一个具体指针类型，而是针对所有 `T*` 类型，例如 `int*`、`double*`、`std::string*`。

## 2. 模板如何根据类型执行不同代码

模板通常不会像普通 `if` 那样在运行期判断类型，而是在编译期选择代码。

常见方式有四种：

1. 类模板特化/偏特化
2. 函数重载
3. `if constexpr`
4. 标准库 type traits，例如 `std::is_pointer_v<T>`

例如：

```cpp
template <typename T>
void inspect(const T& value) {
    if constexpr (std::is_pointer_v<T>) {
        // T 是指针时，这个分支会被编译
    } else {
        // T 不是指针时，这个分支会被编译
    }
}
```

`if constexpr` 是 C++17 引入的编译期分支。没被选中的分支不会被实例化，所以它可以写一些只对特定类型合法的代码。

## 3. 类和函数的模板特化

类模板支持：

- 全特化
- 偏特化

函数模板支持：

- 全特化

函数模板不支持偏特化。遇到“我想偏特化一个函数模板”的场景，通常用函数重载来解决。

例如，函数模板全特化：

```cpp
template <typename T>
void print_value(T value);

template <>
void print_value<std::string>(std::string value);
```

但下面这种“函数模板偏特化”的写法不允许：

```cpp
// 错误示意：函数模板不能偏特化
template <typename T>
void print_value<T*>(T* value);
```

应该改成普通重载：

```cpp
template <typename T>
void print_value(T* value);
```

## 4. “模板特化如同是一个函数，只能在编译期间执行”怎么理解

这句话可以这样理解：模板特化像一个“编译期函数”。

普通函数接收运行期参数，返回运行期结果：

```cpp
int f(int x) {
    return x + 1;
}
```

模板元编程接收编译期参数，产生编译期结果：

```cpp
template <bool Condition, typename Then, typename Else>
struct IfThenElse {
    using type = Then;
};

template <typename Then, typename Else>
struct IfThenElse<false, Then, Else> {
    using type = Else;
};
```

当你写：

```cpp
using R = IfThenElse<true, int, double>::type;
```

编译器在编译期间选择 `int`。这就像调用了一个只在编译期运行的函数：

```text
IfThenElse(true, int, double) -> int
```

它不会在程序运行时执行，也不会生成一个真正的运行期分支。

## 5. 如果模板参数匹配不到任何特化，会发生什么

分情况。

如果有主模板，且没有更合适的特化，编译器使用主模板：

```cpp
TypeName<double>  // 没有 double 特化，所以使用 TypeName<T> 主模板
```

如果没有可用主模板，或者主模板只有声明没有定义，实例化时会报错：

```cpp
template <typename T>
struct OnlySpecialized;

template <>
struct OnlySpecialized<int> {};

OnlySpecialized<double> x; // 错误：没有匹配特化，也没有可用主模板定义
```

如果多个偏特化都能匹配，且编译器无法判断哪个更“特化”，会产生二义性错误。

## 6. 其他需要注意的点

特化要在第一次使用前可见。

如果你先让编译器实例化了 `TypeName<int>`，之后才写 `TypeName<int>` 的全特化，程序通常会报错或违反规则。

函数模板全特化不参与普通重载决议的方式和普通重载不完全一样。

因此实践里，如果目标是“根据参数类型选择函数行为”，优先考虑函数重载或 `if constexpr`，它们通常更直观。

偏特化的匹配是按模式做的。

`TypeName<T*>` 可以匹配所有指针类型；`TypeName<const T>` 可以匹配所有顶层 `const` 类型。模板不是运行时判断，而是在实例化时做模式匹配。

全特化写在头文件里时要注意 ODR。

类模板全特化通常可以放头文件，但非模板函数或函数模板全特化如果在头文件中定义，最好加 `inline`，避免多个翻译单元重复定义。这个示例是单文件程序，所以没有这个问题。

不要滥用特化。

很多现代 C++ 场景可以用 `if constexpr`、concepts、重载、标准库 traits 写得更清晰。特化适合用来定义“类型到类型”或“类型到值”的编译期映射。

## 7. 如何运行

如果环境有 CMake：

```bash
cmake -S . -B build
cmake --build build
./build/template_if_then_else
```

也可以直接用 g++：

```bash
g++ -std=c++17 -Wall -Wextra -pedantic src/main.cpp -o template_if_then_else
./template_if_then_else
```
