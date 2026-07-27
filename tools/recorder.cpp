#include "recorder.hpp"

#include <fmt/chrono.h>

#include <filesystem>
#include <string>

#include "tools/math_tools.hpp"
#include "tools/logger.hpp"

namespace tools{
    
Recorder::Recorder(double fps) : init_(false), stop_(false), queue_(1), fps_(fps), init_err_count_(0){
    start_time_ = std::chrono::steady_clock::now();
    last_frame_time_ = start_time_;

    // 创建文件夹和记录文件名，不在这里创建文件，原因见下面init()
    auto folder_path = "records";
    std::filesystem::create_directory(folder_path);
    auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
    text_path_ = fmt::format("{}/{}.txt", folder_path, file_name);
    video_path_ = fmt::format("{}/{}.avi", folder_path, file_name);
}

Recorder::~Recorder() {
    stop_ = true;

    // 发送一个空帧数据避免阻塞
    queue_.push(FrameData{cv::Mat::zeros(0, 0, 0), {0, 0, 0, 0}, std::chrono::steady_clock::now()});
    if (writing_thread_.joinable()) writing_thread_.join(); // 等待视频保存线程结束

    if (!init_) return;
    text_writer_.close();
    video_writer_.release();
}

void Recorder::write_to_file() {
    while (!stop_) {
        FrameData frame = queue_.pop();
        if (frame.img.empty()){
            tools::logger()->debug("[Recorder] Received empty img! Skip this frame.");
            continue;
        }
        // 将图像写入视频
        video_writer_.write(frame.img);

        // 将四元数写入文本文件（顺序为wxyz）
        Eigen::Vector4d xyzw = frame.q.coeffs();
        auto elapse = tools::delta_time(start_time_, frame.timestamp);
        text_writer_ << fmt::format("{} {} {} {} {} \n", 
            elapse, xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
    }
}

void Recorder::record(const cv::Mat & img, const Eigen::Quaterniond & q, 
    const std::chrono::steady_clock::time_point & timestamp) {
    if (img.empty()) return;
    if (!init_) init(img);

    auto since_last_frame = tools::delta_time(last_frame_time_, timestamp);
    if (since_last_frame < 1.0 / fps_) return; // 控制帧率

    last_frame_time_ = timestamp;
    queue_.push(FrameData{img.clone(), q, timestamp});
}

void Recorder::init(const cv::Mat & img) {
    // 在这里创建文件有以下两种考虑：
    // 1. 创建文件是可能失败的（比如磁盘不足时），如果在构造函数里失败的话处理比较麻烦
    // 2. 如果没有读取到图像，直接创建空文件没有任何意义
    // 如果创建失败的话，自瞄程序并不会终止，但是所有的写入操作都不会做任何事。
    text_writer_.open(text_path_);
    auto fourcc = cv::VideoWriter::fourcc('M','J','P','G'); // 设置编码器
    video_writer_ = cv::VideoWriter(video_path_, fourcc, fps_, img.size());
    writing_thread_ = std::thread(&Recorder::write_to_file, this); // 创建写入线程
    if (!video_writer_.isOpened() || !text_writer_.is_open()) {
        tools::logger()->error("[Recorder] Recorder init failed!");
        init_err_count_++;
        if(init_err_count_ <= 10) return;
    }
    init_ = true;
}
    
} // namespace tools
