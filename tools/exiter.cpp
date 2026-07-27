#include "exiter.hpp"

#include <csignal>
#include <stdexcept>

namespace tools {
    
bool exit_ = false;
bool exiter_inited_ = false;
// 在这里创建的变量是全局变量，它们的生命周期是整个程序运行期间，直到程序结束才会被销毁。
// 其类似于静态变量，但由于没有写在hpp中所以不会被其他文件引用到，避免了多重定义的问题。

Exiter::Exiter() {
    if (exiter_inited_) throw std::runtime_error("Multiple Exiter instances!"); 
    std::signal(SIGINT, [](int) { exit_ = true; });
    exiter_inited_ = true;
}

bool Exiter::exit() const { return exit_; }

} // namespace tools