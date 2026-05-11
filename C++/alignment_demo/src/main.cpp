#include <cstddef>
#include <cstdint>
#include <iostream>

/*
学习笔记：什么是字节对齐
1) 每种类型有一个对齐要求（alignof(T)），对象起始地址通常要是该值的整数倍。
2) 为满足成员对齐，编译器会在结构体成员间插入 padding（填充字节）。
3) 结构体总大小也要是其对齐值的整数倍，因此末尾也可能有 padding。
4) 成员顺序会影响 padding 数量，进而影响 sizeof(结构体)。
*/

// 常见 64 位环境下：
// S1 内存布局通常为：a(1) + pad(3) + b(4) + c(1) + tail pad(3) = 12
struct S1 {
    char a;   // 1 byte
    int b;    // usually 4 bytes, alignment 4
    char c;   // 1 byte
};

// 把 int 放前面通常能减少中间 padding：
// S2 通常为：b(4) + a(1) + c(1) + tail pad(2) = 8
struct S2 {
    int b;
    char a;
    char c;
};

// 手动指定更高对齐（16 字节），常见于 SIMD/高性能数据结构
struct alignas(16) Vec4Like {
    float x;
    float y;
    float z;
    float w;
};

#define PRINT_OFFSET(type, member) \
    std::cout << "  offset(" #member ") = " << offsetof(type, member) << '\n'

void print_s1_ascii() {
    const std::size_t total = sizeof(S1);
    std::cout << "ASCII layout for S1\n";
    std::cout << "offset  : ";
    for (std::size_t i = 0; i < total; ++i) {
        std::cout << '[' << i << ']';
    }
    std::cout << '\n';

    std::cout << "content : ";
    for (std::size_t i = 0; i < total; ++i) {
        if (i == offsetof(S1, a)) {
            std::cout << "[a]";
        } else if (i >= offsetof(S1, b) && i < offsetof(S1, b) + sizeof(int)) {
            std::cout << "[b]";
        } else if (i == offsetof(S1, c)) {
            std::cout << "[c]";
        } else {
            std::cout << "[p]";
        }
    }
    std::cout << "\n(p = padding)\n\n";
}

void print_s2_ascii() {
    const std::size_t total = sizeof(S2);
    std::cout << "ASCII layout for S2\n";
    std::cout << "offset  : ";
    for (std::size_t i = 0; i < total; ++i) {
        std::cout << '[' << i << ']';
    }
    std::cout << '\n';

    std::cout << "content : ";
    for (std::size_t i = 0; i < total; ++i) {
        if (i >= offsetof(S2, b) && i < offsetof(S2, b) + sizeof(int)) {
            std::cout << "[b]";
        } else if (i == offsetof(S2, a)) {
            std::cout << "[a]";
        } else if (i == offsetof(S2, c)) {
            std::cout << "[c]";
        } else {
            std::cout << "[p]";
        }
    }
    std::cout << "\n(p = padding)\n\n";
}

void print_address_line(const char* name, const void* ptr, std::size_t alignment) {
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    std::cout << "  " << name
              << " = " << ptr
              << "  (addr % " << alignment << " = " << (addr % alignment) << ")\n";
}

void print_addresses() {
    S1 s1{};
    S2 s2{};
    Vec4Like v{};

    std::cout << "=== Runtime Addresses ===\n";
    std::cout << "S1 addresses:\n";
    print_address_line("&s1", &s1, alignof(S1));
    print_address_line("&s1.a", &s1.a, alignof(char));
    print_address_line("&s1.b", &s1.b, alignof(int));
    print_address_line("&s1.c", &s1.c, alignof(char));
    std::cout << '\n';

    std::cout << "S2 addresses:\n";
    print_address_line("&s2", &s2, alignof(S2));
    print_address_line("&s2.b", &s2.b, alignof(int));
    print_address_line("&s2.a", &s2.a, alignof(char));
    print_address_line("&s2.c", &s2.c, alignof(char));
    std::cout << '\n';

    std::cout << "Vec4Like addresses:\n";
    print_address_line("&v", &v, alignof(Vec4Like));
    print_address_line("&v.x", &v.x, alignof(float));
    print_address_line("&v.y", &v.y, alignof(float));
    print_address_line("&v.z", &v.z, alignof(float));
    print_address_line("&v.w", &v.w, alignof(float));
    std::cout << "\n";
}

void print_layout() {
    std::cout << "=== Basic Alignment Info ===\n";
    std::cout << "alignof(char) = " << alignof(char) << '\n';
    std::cout << "alignof(int)  = " << alignof(int) << '\n';
    std::cout << "alignof(double) = " << alignof(double) << "\n\n";

    std::cout << "=== Struct S1: {char, int, char} ===\n";
    std::cout << "sizeof(S1) = " << sizeof(S1) << '\n';
    std::cout << "alignof(S1) = " << alignof(S1) << '\n';
    PRINT_OFFSET(S1, a);
    PRINT_OFFSET(S1, b);
    PRINT_OFFSET(S1, c);
    std::cout << '\n';
    print_s1_ascii();

    std::cout << "=== Struct S2: {int, char, char} ===\n";
    std::cout << "sizeof(S2) = " << sizeof(S2) << '\n';
    std::cout << "alignof(S2) = " << alignof(S2) << '\n';
    PRINT_OFFSET(S2, b);
    PRINT_OFFSET(S2, a);
    PRINT_OFFSET(S2, c);
    std::cout << "\n";
    print_s2_ascii();

    std::cout << "=== Custom aligned type (alignas(16)) ===\n";
    std::cout << "sizeof(Vec4Like) = " << sizeof(Vec4Like) << '\n';
    std::cout << "alignof(Vec4Like) = " << alignof(Vec4Like) << "\n\n";
}

int main() {
    print_layout();
    print_addresses();

    std::cout << "Conclusion:\n";
    std::cout << "1) Alignment means an object prefers addresses that are multiples of its alignment.\n";
    std::cout << "2) Padding bytes may be inserted between members and at the end of struct.\n";
    std::cout << "3) Member order can change struct size and memory efficiency.\n";
    std::cout << "4) Proper alignment matters for performance and sometimes correctness on some CPUs.\n";

    return 0;
}
