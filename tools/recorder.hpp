#ifndef TOOLS__RECORDER_HPP
#define TOOLS__RECORDER_HPP

#include <chrono>
#include <fstream>
#include <thread>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "tools/thread_safe_queue.hpp"
namespace tools {

/* 录制器类，用于录制图像和记录四元数数据，并将其分别保存为视频和文本文件。*/
class Recorder{ 
    public:
        Recorder(double fps = 30);
        ~Recorder();
        void record(const cv::Mat & img, const Eigen::Quaterniond & q, 
                    const std::chrono::steady_clock::time_point & timestamp);
    private:
        struct FrameData{
            cv::Mat img;
            Eigen::Quaterniond q; // 这是一种专门用于记录四元数的类型
            std::chrono::steady_clock::time_point timestamp;
        };
        tools::ThreadSafeQueue<FrameData> queue_;
 
        bool init_; // 是否初始化完成
        std::atomic<bool> stop_; // 是否停止录制
        double fps_;
        int init_err_count_; // 初始化错误记数
        std::chrono::steady_clock::time_point last_frame_time_;
        std::chrono::steady_clock::time_point start_time_;
        std::string text_path_;
        std::string video_path_;

        std::ofstream text_writer_;
        cv::VideoWriter video_writer_;

        std::thread writing_thread_;

        void init(const cv::Mat &img); // 初始化函数，做一些可能出现异常的初始化操作
        void write_to_file(); // 将数据写入文件

};

} // namespace tools
#endif // TOOLS__RECORDER_HPP