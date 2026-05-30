#include <iostream>

#define LOG(x) std::cout<<x<<std::endl;

int main(){
    int var = 8;
    void* pointer = &var;
    return 0;
}