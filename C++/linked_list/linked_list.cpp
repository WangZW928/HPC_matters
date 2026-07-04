#include <chrono>
#include <cstddef>
#include <forward_list>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ============================================================
// 1. 单向链表 - 使用 std::unique_ptr 管理节点所有权
//    next 指针持有下一个节点，析构自动级联释放
//    注意：长链表的递归析构可能爆栈，析构函数用迭代方式避免
// ============================================================

template <typename T>
class SinglyLinkedList {
private:
    struct Node {
        T data;
        std::unique_ptr<Node> next;

        template <typename... Args>
        explicit Node(Args&&... args)
            : data(std::forward<Args>(args)...), next(nullptr) {}
    };

    std::unique_ptr<Node> head_;
    Node* tail_ = nullptr;  // 裸指针，不拥有所有权
    std::size_t size_ = 0;

    // 迭代销毁，避免 unique_ptr 递归析构爆栈
    static void destroy_chain(std::unique_ptr<Node> node) {
        while (node) {
            auto next = std::move(node->next);
            node = std::move(next);  // node 析构时 next 已被移走，不再递归
        }
    }

    void copy_from(const SinglyLinkedList& other) {
        static_assert(std::is_copy_constructible<T>::value,
                      "SinglyLinkedList<T> copy requires copy-constructible T");
        for (const auto& value : other) {
            push_back(value);
        }
    }

    std::unique_ptr<Node> detach_front_node() {
        auto node = std::move(head_);
        head_ = std::move(node->next);
        node->next = nullptr;
        if (!head_) tail_ = nullptr;
        --size_;
        return node;
    }

    void append_node(std::unique_ptr<Node> node) {
        Node* raw = node.get();
        if (tail_) {
            tail_->next = std::move(node);
        } else {
            head_ = std::move(node);
        }
        tail_ = raw;
        ++size_;
    }

    void append_chain_from(SinglyLinkedList& other) {
        if (other.empty()) return;
        if (empty()) {
            head_ = std::move(other.head_);
            tail_ = other.tail_;
            size_ = other.size_;
        } else {
            tail_->next = std::move(other.head_);
            tail_ = other.tail_;
            size_ += other.size_;
        }
        other.tail_ = nullptr;
        other.size_ = 0;
    }

public:
    SinglyLinkedList() = default;

    ~SinglyLinkedList() {
        clear();
    }

    SinglyLinkedList(const SinglyLinkedList& other) {
        copy_from(other);
    }

    SinglyLinkedList& operator=(const SinglyLinkedList& other) {
        if (this != &other) {
            SinglyLinkedList tmp(other);
            swap(tmp);
        }
        return *this;
    }

    // 移动构造
    SinglyLinkedList(SinglyLinkedList&& other) noexcept
        : head_(std::move(other.head_))
        , tail_(other.tail_)
        , size_(other.size_)
    {
        other.tail_ = nullptr;
        other.size_ = 0;
    }

    SinglyLinkedList& operator=(SinglyLinkedList&& other) noexcept {
        if (this != &other) {
            clear();
            head_ = std::move(other.head_);
            tail_ = other.tail_;
            size_ = other.size_;
            other.tail_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // ---------- 迭代器 ----------

    class Iterator {
    private:
        Node* current_;
        friend class ConstIterator;
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        explicit Iterator(Node* node) : current_(node) {}

        T& operator*() const { return current_->data; }
        T* operator->() const { return &current_->data; }

        Iterator& operator++() {
            if (current_) current_ = current_->next.get();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return current_ == other.current_;
        }
        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    class ConstIterator {
    private:
        const Node* current_;
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        explicit ConstIterator(const Node* node) : current_(node) {}
        ConstIterator(const Iterator& it) : current_(it.current_) {}

        const T& operator*() const { return current_->data; }
        const T* operator->() const { return &current_->data; }

        ConstIterator& operator++() {
            if (current_) current_ = current_->next.get();
            return *this;
        }

        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const ConstIterator& other) const {
            return current_ == other.current_;
        }
        bool operator!=(const ConstIterator& other) const {
            return !(*this == other);
        }
    };

    Iterator begin() { return Iterator(head_.get()); }
    Iterator end()   { return Iterator(nullptr); }
    ConstIterator begin() const { return ConstIterator(head_.get()); }
    ConstIterator end()   const { return ConstIterator(nullptr); }
    ConstIterator cbegin() const { return begin(); }
    ConstIterator cend()   const { return end(); }

    // ---------- 操作 ----------

    void push_front(const T& value) {
        emplace_front(value);
    }

    void push_front(T&& value) {
        emplace_front(std::move(value));
    }

    template <typename... Args>
    T& emplace_front(Args&&... args) {
        auto new_node = std::make_unique<Node>(std::forward<Args>(args)...);
        Node* raw = new_node.get();
        new_node->next = std::move(head_);
        head_ = std::move(new_node);
        if (!tail_) tail_ = head_.get();
        ++size_;
        return raw->data;
    }

    void push_back(const T& value) {
        emplace_back(value);
    }

    void push_back(T&& value) {
        emplace_back(std::move(value));
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        auto new_node = std::make_unique<Node>(std::forward<Args>(args)...);
        Node* raw = new_node.get();
        if (tail_) {
            tail_->next = std::move(new_node);
        } else {
            head_ = std::move(new_node);
        }
        tail_ = raw;
        ++size_;
        return raw->data;
    }

    void clear() noexcept {
        if (head_) destroy_chain(std::move(head_));
        tail_ = nullptr;
        size_ = 0;
    }

    void swap(SinglyLinkedList& other) noexcept {
        using std::swap;
        swap(head_, other.head_);
        swap(tail_, other.tail_);
        swap(size_, other.size_);
    }

    void pop_front() {
        if (!head_) throw std::runtime_error("pop_front on empty list");
        auto next = std::move(head_->next);
        head_ = std::move(next);
        if (!head_) tail_ = nullptr;
        --size_;
    }

    // 删除第一个匹配的元素
    bool remove(const T& value) {
        if (!head_) return false;

        if (head_->data == value) {
            pop_front();
            return true;
        }

        Node* prev = head_.get();
        while (prev->next) {
            if (prev->next->data == value) {
                auto to_delete = std::move(prev->next);
                prev->next = std::move(to_delete->next);
                if (!prev->next) tail_ = prev;
                --size_;
                return true;
            }
            prev = prev->next.get();
        }
        return false;
    }

    void reverse() noexcept {
        tail_ = head_.get();
        std::unique_ptr<Node> previous;
        while (head_) {
            auto next = std::move(head_->next);
            head_->next = std::move(previous);
            previous = std::move(head_);
            head_ = std::move(next);
        }
        head_ = std::move(previous);
    }

    template <typename Compare = std::less<T>>
    static SinglyLinkedList merge_sorted(SinglyLinkedList left,
                                         SinglyLinkedList right,
                                         Compare comp = Compare{}) {
        SinglyLinkedList result;

        while (!left.empty() && !right.empty()) {
            if (comp(right.front(), left.front())) {
                result.append_node(right.detach_front_node());
            } else {
                result.append_node(left.detach_front_node());
            }
        }
        result.append_chain_from(left);
        result.append_chain_from(right);
        return result;
    }

    T& front() {
        if (!head_) throw std::runtime_error("front on empty list");
        return head_->data;
    }

    const T& front() const {
        if (!head_) throw std::runtime_error("front on empty list");
        return head_->data;
    }

    T& back() {
        if (!tail_) throw std::runtime_error("back on empty list");
        return tail_->data;
    }

    const T& back() const {
        if (!tail_) throw std::runtime_error("back on empty list");
        return tail_->data;
    }

    std::size_t size()  const { return size_; }
    bool        empty() const { return size_ == 0; }

    void print() const {
        std::cout << "[";
        const Node* p = head_.get();
        while (p) {
            std::cout << p->data;
            p = p->next.get();
            if (p) std::cout << " -> ";
        }
        std::cout << "]  (size=" << size_ << ")\n";
    }
};


// ============================================================
// 2. 双向链表 - next 用 unique_ptr 拥有下游，prev 用裸指针回指
// ============================================================

template <typename T>
class DoublyLinkedList {
private:
    struct Node {
        T data;
        std::unique_ptr<Node> next;
        Node* prev = nullptr;

        template <typename... Args>
        explicit Node(Args&&... args)
            : data(std::forward<Args>(args)...), next(nullptr), prev(nullptr) {}
    };

    std::unique_ptr<Node> head_;
    Node* tail_ = nullptr;
    std::size_t size_ = 0;

    static void destroy_chain(std::unique_ptr<Node> node) {
        while (node) {
            auto next = std::move(node->next);
            node = std::move(next);
        }
    }

    void copy_from(const DoublyLinkedList& other) {
        static_assert(std::is_copy_constructible<T>::value,
                      "DoublyLinkedList<T> copy requires copy-constructible T");
        for (const auto& value : other) {
            push_back(value);
        }
    }

public:
    DoublyLinkedList() = default;

    ~DoublyLinkedList() {
        clear();
    }

    DoublyLinkedList(const DoublyLinkedList& other) {
        copy_from(other);
    }

    DoublyLinkedList& operator=(const DoublyLinkedList& other) {
        if (this != &other) {
            DoublyLinkedList tmp(other);
            swap(tmp);
        }
        return *this;
    }

    DoublyLinkedList(DoublyLinkedList&& other) noexcept
        : head_(std::move(other.head_))
        , tail_(other.tail_)
        , size_(other.size_)
    {
        other.tail_ = nullptr;
        other.size_ = 0;
    }

    DoublyLinkedList& operator=(DoublyLinkedList&& other) noexcept {
        if (this != &other) {
            clear();
            head_ = std::move(other.head_);
            tail_ = other.tail_;
            size_ = other.size_;
            other.tail_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // ---------- 迭代器（双向） ----------

    class Iterator {
    private:
        Node* current_;
        Node* tail_;
        friend class DoublyLinkedList;
        friend class ConstIterator;
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        Iterator(Node* node, Node* tail) : current_(node), tail_(tail) {}

        T& operator*() const { return current_->data; }
        T* operator->() const { return &current_->data; }

        Iterator& operator++() {
            if (current_) current_ = current_->next.get();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        Iterator& operator--() {
            current_ = current_ ? current_->prev : tail_;
            return *this;
        }

        Iterator operator--(int) {
            Iterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return current_ == other.current_;
        }
        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    class ConstIterator {
    private:
        const Node* current_;
        const Node* tail_;
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        ConstIterator(const Node* node, const Node* tail)
            : current_(node), tail_(tail) {}
        ConstIterator(const Iterator& it)
            : current_(it.current_), tail_(it.tail_) {}

        const T& operator*() const { return current_->data; }
        const T* operator->() const { return &current_->data; }

        ConstIterator& operator++() {
            if (current_) current_ = current_->next.get();
            return *this;
        }

        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        ConstIterator& operator--() {
            current_ = current_ ? current_->prev : tail_;
            return *this;
        }

        ConstIterator operator--(int) {
            ConstIterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const ConstIterator& other) const {
            return current_ == other.current_;
        }
        bool operator!=(const ConstIterator& other) const {
            return !(*this == other);
        }
    };

    using ReverseIterator = std::reverse_iterator<Iterator>;
    using ConstReverseIterator = std::reverse_iterator<ConstIterator>;

    Iterator begin() { return Iterator(head_.get(), tail_); }
    Iterator end()   { return Iterator(nullptr, tail_); }
    ConstIterator begin() const { return ConstIterator(head_.get(), tail_); }
    ConstIterator end()   const { return ConstIterator(nullptr, tail_); }
    ConstIterator cbegin() const { return begin(); }
    ConstIterator cend()   const { return end(); }
    ReverseIterator rbegin() { return ReverseIterator(end()); }
    ReverseIterator rend() { return ReverseIterator(begin()); }
    ConstReverseIterator rbegin() const { return ConstReverseIterator(end()); }
    ConstReverseIterator rend() const { return ConstReverseIterator(begin()); }
    ConstReverseIterator crbegin() const { return rbegin(); }
    ConstReverseIterator crend() const { return rend(); }

    // ---------- 操作 ----------

    void push_front(const T& value) {
        emplace_front(value);
    }

    void push_front(T&& value) {
        emplace_front(std::move(value));
    }

    template <typename... Args>
    T& emplace_front(Args&&... args) {
        auto node = std::make_unique<Node>(std::forward<Args>(args)...);
        Node* raw = node.get();
        if (head_) {
            head_->prev = raw;
            node->next = std::move(head_);
        } else {
            tail_ = raw;
        }
        head_ = std::move(node);
        ++size_;
        return raw->data;
    }

    void push_back(const T& value) {
        emplace_back(value);
    }

    void push_back(T&& value) {
        emplace_back(std::move(value));
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        auto node = std::make_unique<Node>(std::forward<Args>(args)...);
        Node* raw = node.get();
        if (tail_) {
            tail_->next = std::move(node);
            raw->prev = tail_;
        } else {
            head_ = std::move(node);
        }
        tail_ = raw;
        ++size_;
        return raw->data;
    }

    void clear() noexcept {
        if (head_) destroy_chain(std::move(head_));
        tail_ = nullptr;
        size_ = 0;
    }

    void swap(DoublyLinkedList& other) noexcept {
        using std::swap;
        swap(head_, other.head_);
        swap(tail_, other.tail_);
        swap(size_, other.size_);
    }

    void pop_front() {
        if (!head_) throw std::runtime_error("pop_front on empty list");
        auto next = std::move(head_->next);
        head_ = std::move(next);
        if (head_) {
            head_->prev = nullptr;
        } else {
            tail_ = nullptr;
        }
        --size_;
    }

    void pop_back() {
        if (!tail_) throw std::runtime_error("pop_back on empty list");
        if (tail_->prev) {
            // tail 不是头节点
            Node* new_tail = tail_->prev;
            new_tail->next.reset();  // 释放当前 tail
            tail_ = new_tail;
        } else {
            // tail 就是头节点
            head_.reset();
            tail_ = nullptr;
        }
        --size_;
    }

    // 删除任意节点（通过迭代器），O(1)
    // ⚠️ 传入 end() 是未定义行为
    void erase(Iterator it) {
        Node* target = it.current_;
        if (!target) return;

        if (target->prev) {
            // 不是头节点
            auto owned = std::move(target->prev->next);
            target->prev->next = std::move(owned->next);
            if (target->prev->next) {
                target->prev->next->prev = target->prev;
            } else {
                tail_ = target->prev;
            }
        } else {
            // 是头节点
            auto next = std::move(head_->next);
            head_ = std::move(next);
            if (head_) {
                head_->prev = nullptr;
            } else {
                tail_ = nullptr;
            }
        }
        --size_;
    }

    T& front() {
        if (!head_) throw std::runtime_error("front on empty list");
        return head_->data;
    }

    const T& front() const {
        if (!head_) throw std::runtime_error("front on empty list");
        return head_->data;
    }

    T& back() {
        if (!tail_) throw std::runtime_error("back on empty list");
        return tail_->data;
    }

    const T& back() const {
        if (!tail_) throw std::runtime_error("back on empty list");
        return tail_->data;
    }

    std::size_t size()  const { return size_; }
    bool        empty() const { return size_ == 0; }

    void print_forward() const {
        std::cout << "forward: [";
        const Node* p = head_.get();
        while (p) {
            std::cout << p->data;
            p = p->next.get();
            if (p) std::cout << " ⇄ ";
        }
        std::cout << "]  (size=" << size_ << ")\n";
    }

    void print_backward() const {
        std::cout << "backward: [";
        const Node* p = tail_;
        while (p) {
            std::cout << p->data;
            p = p->prev;
            if (p) std::cout << " ⇄ ";
        }
        std::cout << "]  (size=" << size_ << ")\n";
    }
};


// ============================================================
// 3. Intrusive list：节点不拥有数据，只把已有对象串起来
// ============================================================

struct Particle {
    int id;
    double energy_mev;
    Particle* next = nullptr;  // intrusive hook：对象自己携带链表指针
};

class IntrusiveParticleList {
private:
    Particle* head_ = nullptr;  // 不拥有 Particle 的生命周期

public:
    void push_front(Particle& particle) noexcept {
        particle.next = head_;
        head_ = &particle;
    }

    void print() const {
        std::cout << "[";
        const Particle* p = head_;
        while (p) {
            std::cout << "id=" << p->id << ", E=" << p->energy_mev << "MeV";
            p = p->next;
            if (p) std::cout << " -> ";
        }
        std::cout << "]\n";
    }
};


// ============================================================
// 4. 标准库链表对比 & 性能基准
// ============================================================

// 简单的计时辅助
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;
public:
    Timer() : start_(Clock::now()) {}
    double elapsed_ms() const {
        auto dur = Clock::now() - start_;
        return std::chrono::duration<double, std::milli>(dur).count();
    }
};


void benchmark_insert_front(int n) {
    std::cout << "--- 头部插入 " << n << " 个元素 ---\n";

    // vector: 头部插入是 O(n^2) 的灾难
    {
        Timer t;
        std::vector<int> v;
        for (int i = 0; i < n; ++i) {
            v.insert(v.begin(), i);
        }
        std::cout << "  std::vector          : " << t.elapsed_ms() << " ms  (O(n) per insert, total O(n^2))\n";
    }

    {
        Timer t;
        std::forward_list<int> fl;
        for (int i = 0; i < n; ++i) {
            fl.push_front(i);
        }
        std::cout << "  std::forward_list    : " << t.elapsed_ms() << " ms  (O(1) per insert)\n";
    }

    {
        Timer t;
        std::list<int> l;
        for (int i = 0; i < n; ++i) {
            l.push_front(i);
        }
        std::cout << "  std::list            : " << t.elapsed_ms() << " ms  (O(1) per insert)\n";
    }

    {
        Timer t;
        SinglyLinkedList<int> sl;
        for (int i = 0; i < n; ++i) {
            sl.push_front(i);
        }
        std::cout << "  我的 SinglyLinkedList: " << t.elapsed_ms() << " ms\n";
    }

    {
        Timer t;
        DoublyLinkedList<int> dl;
        for (int i = 0; i < n; ++i) {
            dl.push_front(i);
        }
        std::cout << "  我的 DoublyLinkedList: " << t.elapsed_ms() << " ms\n";
    }
    std::cout << '\n';
}


void benchmark_random_access(int n) {
    std::cout << "--- 随机访问 " << n << " 个元素 ---\n";

    std::vector<int> v(n);
    std::list<int> l(n);

    // 填充
    for (int i = 0; i < n; ++i) v[i] = i;
    std::copy(v.begin(), v.end(), l.begin());

    // vector 随机访问
    {
        Timer t;
        volatile long long sum = 0;  // volatile 防止优化掉
        for (int i = 0; i < n; ++i) {
            sum += v[i];
        }
        std::cout << "  std::vector (下标)   : " << t.elapsed_ms() << " ms  (O(1) per access)\n";
    }

    // list 随机访问 (灾难)
    {
        Timer t;
        volatile long long sum = 0;
        for (int i = 0; i < n; ++i) {
            auto it = l.begin();
            std::advance(it, i);
            sum += *it;
        }
        std::cout << "  std::list (advance)  : " << t.elapsed_ms() << " ms  (O(n) per access, total O(n^2))\n";
    }
    std::cout << '\n';
}


// ============================================================
// 4. 主程序
// ============================================================

int main() {
    static_assert(std::is_copy_constructible<SinglyLinkedList<int>>::value,
                  "tutorial list should support deep copy");
    static_assert(std::is_copy_constructible<DoublyLinkedList<int>>::value,
                  "tutorial list should support deep copy");

    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║   链表（Linked List）— 现代 C++ 实现与对比  ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";

    // ---------- 单向链表演示 ----------
    std::cout << "══════ 1. 单向链表 (Singly Linked List) ══════\n\n";

    SinglyLinkedList<std::string> sl;
    std::cout << "空链表: "; sl.print();

    sl.push_back("Apple");
    sl.push_back("Banana");
    sl.push_back("Cherry");
    std::cout << "push_back 三次: "; sl.print();

    sl.push_front("ZERO");
    std::cout << "push_front(\"ZERO\"): "; sl.print();

    std::cout << "front() = " << sl.front() << "\n";
    std::cout << "back()  = " << sl.back() << "\n";

    sl.remove("Banana");
    std::cout << "remove(\"Banana\"): "; sl.print();

    sl.pop_front();
    std::cout << "pop_front(): "; sl.print();

    std::cout << "使用 range-for 遍历: ";
    for (const auto& s : sl) {
        std::cout << s << " ";
    }
    std::cout << "\n\n";

    const auto& const_sl = sl;
    std::cout << "const_iterator 遍历: ";
    for (auto it_const = const_sl.cbegin(); it_const != const_sl.cend(); ++it_const) {
        std::cout << *it_const << " ";
    }
    std::cout << "\n";

    SinglyLinkedList<std::string> sl_copy = sl;  // 深拷贝：节点重新分配，值相同
    sl_copy.push_back("Durian");
    std::cout << "深拷贝后修改副本，原链表: "; sl.print();
    std::cout << "深拷贝后修改副本，副本:   "; sl_copy.print();

    sl.reverse();
    std::cout << "reverse() 后: "; sl.print();

    SinglyLinkedList<int> odd;
    odd.push_back(1);
    odd.push_back(3);
    odd.push_back(5);
    SinglyLinkedList<int> even;
    even.push_back(2);
    even.push_back(4);
    even.push_back(6);
    auto merged = SinglyLinkedList<int>::merge_sorted(std::move(odd), std::move(even));
    std::cout << "merge_sorted([1,3,5], [2,4,6]): ";
    merged.print();
    std::cout << '\n';

    // 移动语义验证
    SinglyLinkedList<std::string> sl2 = std::move(sl);
    std::cout << "移动后 sl  (应为空): "; sl.print();
    std::cout << "移动后 sl2: "; sl2.print();
    std::cout << '\n';

    // ---------- 双向链表演示 ----------
    std::cout << "══════ 2. 双向链表 (Doubly Linked List) ══════\n\n";

    DoublyLinkedList<int> dl;
    dl.push_back(10);
    dl.push_back(20);
    dl.push_back(30);
    dl.push_back(40);
    dl.push_back(50);

    dl.print_forward();
    dl.print_backward();
    std::cout << "front() = " << dl.front() << ", back() = " << dl.back() << '\n';

    DoublyLinkedList<int> dl_copy = dl;
    dl_copy.pop_front();
    dl_copy.push_back(60);
    std::cout << "深拷贝副本改动后: ";
    dl_copy.print_forward();

    std::cout << "std::reverse_iterator 反向遍历: ";
    for (auto rit = dl.rbegin(); rit != dl.rend(); ++rit) {
        std::cout << *rit << " ";
    }
    std::cout << "\n";

    // 删除中间元素
    auto it = dl.begin();
    ++it; ++it;  // 指向 30
    std::cout << "\n删除 *it = " << *it << " (O(1) 删除任意位置)\n";
    dl.erase(it);
    dl.print_forward();
    dl.print_backward();

    // 删除头部
    dl.pop_front();
    std::cout << "\npop_front() 后:\n";
    dl.print_forward();

    // 删除尾部
    dl.pop_back();
    std::cout << "\npop_back() 后:\n";
    dl.print_forward();
    std::cout << '\n';

    // ---------- intrusive list 对比 ----------
    std::cout << "══════ 3. Intrusive List（非拥有式链表） ══════\n\n";

    Particle p1{1, 2.7, nullptr};
    Particle p2{2, 4.1, nullptr};
    Particle p3{3, 8.6, nullptr};
    IntrusiveParticleList particles;
    particles.push_front(p1);
    particles.push_front(p2);
    particles.push_front(p3);
    std::cout << "Particle 对象由外部拥有，链表只保存 next hook:\n";
    particles.print();
    std::cout << "适合对象生命周期已由别处管理、且要避免额外节点分配的场景。\n\n";

    // ---------- 标准库对比 ----------
    std::cout << "══════ 4. 标准库 std::forward_list vs std::list ══════\n\n";

    std::forward_list<int> fl = {1, 2, 3, 4, 5};
    std::cout << "std::forward_list: ";
    for (int x : fl) std::cout << x << " ";
    std::cout << "\n";

    // forward_list 的 insert_after / erase_after
    auto fl_it = fl.begin();
    fl.insert_after(fl_it, 99);  // 在 1 之后插入 99
    std::cout << "insert_after(begin, 99): ";
    for (int x : fl) std::cout << x << " ";
    std::cout << "\n";

    fl.erase_after(fl.begin());  // 删除 begin 之后的元素
    std::cout << "erase_after(begin):     ";
    for (int x : fl) std::cout << x << " ";
    std::cout << "\n\n";

    std::list<int> lst = {10, 20, 30, 40, 50};
    std::cout << "std::list:\n";
    lst.push_front(5);
    lst.push_back(55);
    std::cout << "  push_front(5) + push_back(55): ";
    for (int x : lst) std::cout << x << " ";
    std::cout << "\n";

    // 在中间插入
    auto mid = lst.begin();
    std::advance(mid, 3);
    lst.insert(mid, 999);
    std::cout << "  insert(第4位, 999):           ";
    for (int x : lst) std::cout << x << " ";
    std::cout << "\n";

    // 从后向前遍历
    std::cout << "  反向遍历: ";
    for (auto ri = lst.rbegin(); ri != lst.rend(); ++ri) {
        std::cout << *ri << " ";
    }
    std::cout << "\n\n";

    // ---------- 性能基准 ----------
    std::cout << "══════ 5. 性能基准测试 ══════\n\n";

    benchmark_insert_front(10000);
    benchmark_random_access(5000);

    // ---------- 总结 ----------
    std::cout << "══════ 总结 ══════\n\n";
    std::cout << "链表的核心优势:\n";
    std::cout << "  ✓ 头部/中间插入删除 O(1)（已知位置）\n";
    std::cout << "  ✓ 迭代器在插入删除后保持稳定\n";
    std::cout << "  ✓ 不需要连续内存，无 reallocate 开销\n\n";

    std::cout << "链表的劣势:\n";
    std::cout << "  ✗ 不支持随机访问 O(1)\n";
    std::cout << "  ✗ 缓存局部性差（节点散布内存各处）\n";
    std::cout << "  ✗ 每个节点额外开销（至少一个指针）\n";
    std::cout << "  ✗ unique_ptr 递归析构需特别处理\n\n";

    std::cout << "工程建议:\n";
    std::cout << "  - 默认用 std::vector，需要频繁头部/中间插入时考虑 list\n";
    std::cout << "  - HPC / 数值计算中 vector 几乎总是更好的选择\n";
    std::cout << "  - 实现自己的链表是理解指针 & 所有权语义的绝佳练习\n";
    std::cout << "  - 工业代码优先用 std::forward_list / std::list\n";

    return 0;
}
