#include <iostream>
#include <string>
#include <type_traits>

template <bool Condition, typename Then, typename Else>
struct IfThenElse {
    using type = Then;
};

template <typename Then, typename Else>
struct IfThenElse<false, Then, Else> {
    using type = Else;
};

template <bool Condition, typename Then, typename Else>
using IfThenElseT = typename IfThenElse<Condition, Then, Else>::type;

template <typename T>
struct TypeName {
    static const char* value() {
        return "general type";
    }
};

template <>
struct TypeName<int> {
    static const char* value() {
        return "int full specialization";
    }
};

template <typename T>
struct TypeName<T*> {
    static const char* value() {
        return "pointer partial specialization";
    }
};

template <typename T>
struct TypeName<const T> {
    static const char* value() {
        return "const partial specialization";
    }
};

template <typename T>
struct StoragePolicy {
    using type = IfThenElseT<(sizeof(T) <= sizeof(void*)), T, const T&>;
};

template <typename T>
void print_value(T value) {
    std::cout << "function template primary: " << value << '\n';
}

template <>
void print_value<std::string>(std::string value) {
    std::cout << "function template full specialization for std::string: "
              << value << '\n';
}

void handle_type(int) {
    std::cout << "ordinary overload: handle int\n";
}

template <typename T>
void handle_type(T*) {
    std::cout << "ordinary overload: handle pointer\n";
}

template <typename T>
void handle_type(T) {
    std::cout << "ordinary overload: handle general value\n";
}

template <typename T>
void inspect_with_if_constexpr(const T& value) {
    if constexpr (std::is_pointer_v<T>) {
        std::cout << "if constexpr branch: pointer, address = " << value << '\n';
    } else if constexpr (std::is_integral_v<T>) {
        std::cout << "if constexpr branch: integral, value = " << value << '\n';
    } else {
        std::cout << "if constexpr branch: other type\n";
    }
}

int main() {
    std::cout << "=== Class Template Specialization ===\n";
    std::cout << "TypeName<double>::value()     = " << TypeName<double>::value() << '\n';
    std::cout << "TypeName<int>::value()        = " << TypeName<int>::value() << '\n';
    std::cout << "TypeName<int*>::value()       = " << TypeName<int*>::value() << '\n';
    std::cout << "TypeName<const int>::value()  = " << TypeName<const int>::value() << "\n\n";

    std::cout << "=== Compile-time IfThenElse ===\n";
    using SmallStorage = StoragePolicy<int>::type;
    using LargeStorage = StoragePolicy<std::string>::type;
    std::cout << "StoragePolicy<int> is int? "
              << std::boolalpha << std::is_same_v<SmallStorage, int> << '\n';
    std::cout << "StoragePolicy<std::string> is const std::string&? "
              << std::is_same_v<LargeStorage, const std::string&> << "\n\n";

    std::cout << "=== Function Template Specialization ===\n";
    print_value(42);
    print_value(std::string{"hello specialization"});
    std::cout << '\n';

    std::cout << "=== Prefer Overload for Function Template Branching ===\n";
    int x = 7;
    handle_type(x);
    handle_type(&x);
    handle_type(3.14);
    std::cout << '\n';

    std::cout << "=== if constexpr Branching ===\n";
    inspect_with_if_constexpr(123);
    inspect_with_if_constexpr(&x);
    inspect_with_if_constexpr(std::string{"abc"});

    static_assert(std::is_same_v<IfThenElseT<true, int, double>, int>);
    static_assert(std::is_same_v<IfThenElseT<false, int, double>, double>);

    return 0;
}
