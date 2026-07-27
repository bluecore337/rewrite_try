#ifndef TOOLS__EXITER_HPP
#define TOOLS__EXITER_HPP

namespace tools {
    
/*  这是一个简单的退出检测器类 Exiter，它会在构造时注册 Ctrl+C 信号处理函数，并在收到 Ctrl+C 后将 exit_ 标记为true。
 *  你可以在主循环中调用 exit() 方法来检查是否需要退出程序。
 *  需要注意的是，这个类会检查请确保在程序中只会有一个 Exiter 实例。*/
class Exiter {
    public:
        Exiter();
        bool exit() const; // 返回exit状态, true时表示退出
};

} // namespace tools

#endif //TOOLS__EXITER_HPP
