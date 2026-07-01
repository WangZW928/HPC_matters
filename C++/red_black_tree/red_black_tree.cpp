#include <functional>
#include <iostream>
#include <vector>

// 红黑树颜色：每个结点只能是红色或黑色。
enum class Color {
    RED,
    BLACK
};

// 一个 key-only 的模板红黑树，行为类似 std::set 的简化版。
template <typename Key, typename Compare = std::less<Key>>
class RedBlackTree {
private:
    struct Node {
        Key key{};
        Color color{Color::BLACK};
        Node* parent{nullptr};
        Node* left{nullptr};
        Node* right{nullptr};

        Node() = default;

        Node(const Key& k, Color c, Node* nil)
            : key(k), color(c), parent(nil), left(nil), right(nil) {}
    };

    Node* root_;
    Node* nil_;
    std::size_t size_;
    Compare comp_;

public:
    RedBlackTree()
        : root_(nullptr), nil_(new Node()), size_(0), comp_(Compare{}) {
        nil_->color = Color::BLACK;
        nil_->parent = nil_;
        nil_->left = nil_;
        nil_->right = nil_;
        root_ = nil_;
    }

    ~RedBlackTree() {
        clear(root_);
        delete nil_;
    }

    RedBlackTree(const RedBlackTree&) = delete;
    RedBlackTree& operator=(const RedBlackTree&) = delete;

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0;
    }

    bool find(const Key& key) const {
        return searchNode(key) != nil_;
    }

    // 插入成功返回 true；如果 key 已存在，则不插入并返回 false。
    bool insert(const Key& key) {
        Node* parent = nil_;
        Node* current = root_;

        while (current != nil_) {
            parent = current;
            if (comp_(key, current->key)) {
                current = current->left;
            } else if (comp_(current->key, key)) {
                current = current->right;
            } else {
                return false;
            }
        }

        Node* node = new Node(key, Color::RED, nil_);
        node->parent = parent;

        if (parent == nil_) {
            root_ = node;
        } else if (comp_(node->key, parent->key)) {
            parent->left = node;
        } else {
            parent->right = node;
        }

        ++size_;
        insertFixup(node);
        return true;
    }

    // 删除成功返回 true；如果 key 不存在，则返回 false。
    bool erase(const Key& key) {
        Node* z = searchNode(key);
        if (z == nil_) {
            return false;
        }

        Node* y = z;
        Color yOriginalColor = y->color;
        Node* x = nil_;

        if (z->left == nil_) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == nil_) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = minimum(z->right);
            yOriginalColor = y->color;
            x = y->right;

            if (y->parent == z) {
                x->parent = y;
            } else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }

            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }

        delete z;
        --size_;

        // 只有实际移走的黑色结点会破坏黑高，需要进入删除修复。
        if (yOriginalColor == Color::BLACK) {
            deleteFixup(x);
        }

        return true;
    }

    std::vector<Key> inorder() const {
        std::vector<Key> result;
        result.reserve(size_);
        inorder(root_, result);
        return result;
    }

private:
    void clear(Node* node) {
        if (node == nil_) {
            return;
        }
        clear(node->left);
        clear(node->right);
        delete node;
    }

    Node* searchNode(const Key& key) const {
        Node* current = root_;
        while (current != nil_) {
            if (comp_(key, current->key)) {
                current = current->left;
            } else if (comp_(current->key, key)) {
                current = current->right;
            } else {
                return current;
            }
        }
        return nil_;
    }

    Node* minimum(Node* node) const {
        while (node->left != nil_) {
            node = node->left;
        }
        return node;
    }

    void inorder(Node* node, std::vector<Key>& result) const {
        if (node == nil_) {
            return;
        }
        inorder(node->left, result);
        result.push_back(node->key);
        inorder(node->right, result);
    }

    // 左旋：把 x 的右孩子 y 提上来，x 成为 y 的左孩子。
    void leftRotate(Node* x) {
        Node* y = x->right;
        x->right = y->left;

        if (y->left != nil_) {
            y->left->parent = x;
        }

        y->parent = x->parent;
        if (x->parent == nil_) {
            root_ = y;
        } else if (x == x->parent->left) {
            x->parent->left = y;
        } else {
            x->parent->right = y;
        }

        y->left = x;
        x->parent = y;
    }

    // 右旋：把 y 的左孩子 x 提上来，y 成为 x 的右孩子。
    void rightRotate(Node* y) {
        Node* x = y->left;
        y->left = x->right;

        if (x->right != nil_) {
            x->right->parent = y;
        }

        x->parent = y->parent;
        if (y->parent == nil_) {
            root_ = x;
        } else if (y == y->parent->right) {
            y->parent->right = x;
        } else {
            y->parent->left = x;
        }

        x->right = y;
        y->parent = x;
    }

    // 插入修复：新结点初始为红色，只可能破坏“不能连续两个红结点”。
    void insertFixup(Node* z) {
        while (z->parent->color == Color::RED) {
            if (z->parent == z->parent->parent->left) {
                Node* uncle = z->parent->parent->right;

                if (uncle->color == Color::RED) {
                    z->parent->color = Color::BLACK;
                    uncle->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        leftRotate(z);
                    }
                    z->parent->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    rightRotate(z->parent->parent);
                }
            } else {
                Node* uncle = z->parent->parent->left;

                if (uncle->color == Color::RED) {
                    z->parent->color = Color::BLACK;
                    uncle->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->left) {
                        z = z->parent;
                        rightRotate(z);
                    }
                    z->parent->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    leftRotate(z->parent->parent);
                }
            }
        }

        root_->color = Color::BLACK;
    }

    // 用 v 替换 u 在树中的位置，不处理 v 的左右孩子。
    void transplant(Node* u, Node* v) {
        if (u->parent == nil_) {
            root_ = v;
        } else if (u == u->parent->left) {
            u->parent->left = v;
        } else {
            u->parent->right = v;
        }
        v->parent = u->parent;
    }

    // 删除修复：x 带着“少一个黑色”的问题向上移动，直到恢复黑高。
    void deleteFixup(Node* x) {
        while (x != root_ && x->color == Color::BLACK) {
            if (x == x->parent->left) {
                Node* w = x->parent->right;

                if (w->color == Color::RED) {
                    w->color = Color::BLACK;
                    x->parent->color = Color::RED;
                    leftRotate(x->parent);
                    w = x->parent->right;
                }

                if (w->left->color == Color::BLACK && w->right->color == Color::BLACK) {
                    w->color = Color::RED;
                    x = x->parent;
                } else {
                    if (w->right->color == Color::BLACK) {
                        w->left->color = Color::BLACK;
                        w->color = Color::RED;
                        rightRotate(w);
                        w = x->parent->right;
                    }

                    w->color = x->parent->color;
                    x->parent->color = Color::BLACK;
                    w->right->color = Color::BLACK;
                    leftRotate(x->parent);
                    x = root_;
                }
            } else {
                Node* w = x->parent->left;

                if (w->color == Color::RED) {
                    w->color = Color::BLACK;
                    x->parent->color = Color::RED;
                    rightRotate(x->parent);
                    w = x->parent->left;
                }

                if (w->right->color == Color::BLACK && w->left->color == Color::BLACK) {
                    w->color = Color::RED;
                    x = x->parent;
                } else {
                    if (w->left->color == Color::BLACK) {
                        w->right->color = Color::BLACK;
                        w->color = Color::RED;
                        leftRotate(w);
                        w = x->parent->left;
                    }

                    w->color = x->parent->color;
                    x->parent->color = Color::BLACK;
                    w->left->color = Color::BLACK;
                    rightRotate(x->parent);
                    x = root_;
                }
            }
        }

        x->color = Color::BLACK;
    }
};

template <typename T>
void printVector(const std::vector<T>& values) {
    for (const auto& value : values) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}

int main() {
    RedBlackTree<int> tree;
    std::vector<int> values{10, 20, 30, 15, 25, 5, 1, 8};

    std::cout << "插入: ";
    printVector(values);

    for (int value : values) {
        tree.insert(value);
    }

    std::cout << "中序遍历: ";
    printVector(tree.inorder());
    std::cout << "结点数量: " << tree.size() << '\n';

    int target = 15;
    std::cout << "查找 " << target << ": "
              << (tree.find(target) ? "存在" : "不存在") << '\n';

    std::cout << "删除 20 和 5\n";
    tree.erase(20);
    tree.erase(5);

    std::cout << "删除后中序遍历: ";
    printVector(tree.inorder());
    std::cout << "结点数量: " << tree.size() << '\n';

    std::cout << "再次查找 20: "
              << (tree.find(20) ? "存在" : "不存在") << '\n';

    return 0;
}
