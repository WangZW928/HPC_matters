#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <typeinfo>

template <typename T>
T add(T a, T b) {
    return a + b;
}

template <typename T>
T max_value(T a, T b) {
    return a < b ? b : a;
}

template <typename T, int Size>
struct Array {
    T data[Size];

    constexpr int size() const {
        return Size;
    }

    T& operator[](int index) {
        return data[index];
    }

    const T& operator[](int index) const {
        return data[index];
    }
};

template <typename T, std::size_t Size>
struct ModernArray {
    std::array<T, Size> data{};
};

template <typename T>
void print_type_and_value(const T& value) {
    std::cout << "deduced T = " << typeid(T).name()
              << ", value = " << value << '\n';
}

template <typename T, int Size>
void print_array_info(const Array<T, Size>& arr, const char* name) {
    std::cout << name << ":\n";
    std::cout << "  Size template argument = " << Size << '\n';
    std::cout << "  sizeof(Array<T, Size>) = " << sizeof(arr) << " bytes\n";
    std::cout << "  object address         = " << static_cast<const void*>(&arr) << '\n';
    std::cout << "  data address           = " << static_cast<const void*>(arr.data) << '\n';
    std::cout << "  note: data is embedded inside the object, no new/delete is involved.\n";
}

Array<int, 16> global_arr{};

int main() {
    std::cout << "=== Function Template Instantiation ===\n";
    std::cout << "add<int>(1, 2)       = " << add<int>(1, 2) << '\n';
    std::cout << "add(1.5, 2.5)        = " << add(1.5, 2.5) << '\n';
    std::cout << "max_value('a', 'z')  = " << max_value('a', 'z') << "\n\n";

    std::cout << "=== Template Type Deduction ===\n";
    print_type_and_value(42);
    print_type_and_value(3.14);
    print_type_and_value(std::string{"hello template"});
    std::cout << '\n';

    std::cout << "=== Non-type Template Parameter ===\n";
    Array<int, 16> local_arr{};
    local_arr[0] = 100;
    local_arr[15] = 999;

    print_array_info(local_arr, "local_arr");
    std::cout << "  local_arr[0]        = " << local_arr[0] << '\n';
    std::cout << "  local_arr[15]       = " << local_arr[15] << "\n\n";

    print_array_info(global_arr, "global_arr");

    std::cout << "\n=== Modern C++ Style ===\n";
    ModernArray<int, 16> modern{};
    std::cout << "sizeof(ModernArray<int, 16>) = " << sizeof(modern) << " bytes\n";
    std::cout << "std::array also stores elements inside the object.\n";

    return 0;
}
