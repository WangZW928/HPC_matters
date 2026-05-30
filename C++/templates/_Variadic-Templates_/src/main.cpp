#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

void print_recursive() {
    std::cout << "(end)\n";
}

template <typename First, typename... Rest>
void print_recursive(const First& first, const Rest&... rest) {
    std::cout << first;
    std::cout << " -> ";
    print_recursive(rest...);
}

template <typename... Ts>
void print_with_fold(const Ts&... values) {
    ((std::cout << values << ' '), ...);
    std::cout << '\n';
}

template <typename... Ts>
auto sum_all(Ts... values) {
    return (values + ... + 0);
}

template <typename... Ts>
constexpr std::size_t count_types() {
    return sizeof...(Ts);
}

template <typename... Ts>
struct AllIntegral : std::bool_constant<(std::is_integral_v<Ts> && ...)> {};

template <typename... Ts>
using FirstType = std::tuple_element_t<0, std::tuple<Ts...>>;

template <typename T, typename... Ts>
std::vector<T> make_vector(Ts&&... values) {
    static_assert((std::is_convertible_v<Ts, T> && ...),
                  "all constructor arguments must be convertible to T");
    return {std::forward<Ts>(values)...};
}

int main() {
    std::cout << "=== Recursive Pack Expansion ===\n";
    print_recursive(10, 2.5, "hello", std::string{"template"});
    std::cout << '\n';

    std::cout << "=== Fold Expression ===\n";
    print_with_fold("values:", 1, 2, 3, 4);
    std::cout << "sum_all(1, 2, 3, 4) = " << sum_all(1, 2, 3, 4) << '\n';
    std::cout << "sum_all(1.5, 2.5, 3.0) = " << sum_all(1.5, 2.5, 3.0) << "\n\n";

    std::cout << "=== sizeof... ===\n";
    std::cout << "count_types<>(): " << count_types<>() << '\n';
    std::cout << "count_types<int, double, std::string>(): "
              << count_types<int, double, std::string>() << "\n\n";

    std::cout << "=== Type Trait on Parameter Pack ===\n";
    std::cout << std::boolalpha;
    std::cout << "AllIntegral<int, short, long>::value = "
              << AllIntegral<int, short, long>::value << '\n';
    std::cout << "AllIntegral<int, double, long>::value = "
              << AllIntegral<int, double, long>::value << "\n\n";

    std::cout << "=== Variadic Helper Function ===\n";
    auto vec = make_vector<int>(1, 2, 3, 4, 5);
    std::cout << "make_vector<int>(1, 2, 3, 4, 5): ";
    for (int value : vec) {
        std::cout << value << ' ';
    }
    std::cout << "\n\n";

    static_assert(count_types<char, int, double>() == 3);
    static_assert(AllIntegral<int, unsigned, long long>::value);
    static_assert(!AllIntegral<int, double>::value);
    static_assert(std::is_same_v<FirstType<int, double, std::string>, int>);

    return 0;
}
