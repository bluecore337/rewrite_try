#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP_
#define TOOLS__THREAD_SAFE_QUEUE_HPP_

#include <iostream> // 仅用于cerr
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>

namespace tools {

template <typename T, bool pop_when_full = false>
// pop_when_full默认为false，即丢掉新数据或是使用传入的函数处理
// 将 pop_when_full 放在此处不占用内存，且可以在编译期确定是否需要 pop
// 或许我们可以考虑将 max_size_ 也放在模板参数中，这样可以在编译期确定队列的最大长度，避免运行时的开销

class ThreadSafeQueue { // 线程安全的队列。仅在有数据时才能读取数据，否则等待数据进入
    public:
        ThreadSafeQueue(size_t max_size, std::function<void(void)> full_handler = nullptr)
            : max_size_(max_size), full_handler_(full_handler) {}

        void push(const T & value) {
            std::unique_lock<std::mutex> lock(mutex_);

            if (queue_.size() >= max_size_) {
                if (pop_when_full) {
                    queue_.pop();
                } else {
                    full_handler_();
                    return ;
                }
            }

            queue_.push(value);
            notifier.notify_all(); // 唤醒所有等待的线程
        }

        T pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            notifier.wait(lock, [this] { return !queue_.empty(); }); // 阻塞直到队列不为空
            T value = std::move(queue_.front());
            queue_.pop();
            return std::move(value);
        }
        
        T front(){
            std::unique_lock<std::mutex> lock(mutex_);
            notifier.wait(lock, [this] { return !queue_.empty(); }); // 阻塞直到队列不为空

            return queue_.front();
        }

        T back(){
            std::unique_lock<std::mutex> lock(mutex_);
            notifier.wait(lock, [this] { return !queue_.empty(); }); // 阻塞直到队列不为空

            return queue_.back();
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            while (!queue_.empty()) {
                queue_.pop();
            }
            notifier.notify_all(); // 唤醒所有等待的线程
        }

        void pop(T & value) {
            std::unique_lock<std::mutex> lock(mutex_);
            notifier.wait(lock, [this] { return !queue_.empty(); }); // 阻塞直到队列不为空
            
            if (queue_.empty()) return ;
            value = queue_.front();
            queue_.pop();
        }

        bool pop(T & value, std::chrono::milliseconds timeout) {
            // timeout表示这个pop的等待时间，超过这个时间也会结束阻塞
            // 这个重载会返回是否读取成功。
            std::unique_lock<std::mutex> lock(mutex_);
            notifier.wait_for(lock, timeout, [this] { return !queue_.empty(); });
            
            if (queue_.empty()) return false;
            value = queue_.front();
            queue_.pop();
            return true;
        }
        /*---旧版兼容---*/
        bool pop_for(T & value, std::chrono::milliseconds timeout)
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (!notifier.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
                return false;
            }

            if (queue_.empty()) {
                return false;
            }

            value = queue_.front();
            queue_.pop();
            return true;
        }
        void back(T & value) {
            std::unique_lock<std::mutex> lock(mutex_);

            if (queue_.empty()) {
                std::cerr << "Error: Attempt to access the back of an empty queue." << std::endl;
                return;
            }

            value = queue_.back();
        }
        /*--------------*/

    private:
        std::queue<T> queue_;
        size_t max_size_;
        mutable std::mutex mutex_; // 一个互斥锁，用于保护队列的访问，确保线程安全。
        std::condition_variable notifier; // 一个"广播"，用于在队列为空时阻塞 pop 操作，并在 Push 时唤醒等待的线程。
        std::function<void(void)> full_handler_;
};

}

#endif  // TOOLS__THREAD_SAFE_QUEUE_HPP_