
#include <cassert>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    const int a = 15;
    int *g = const_cast<int*>(&a);
    *g = 10;
    
    assert(g == &a);
    std::cout << a << std::endl;
    std::cout << "Hello world!" << std::endl;
    return 0;
}
