#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP

#include <optional>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace auto_buff {

/* yolo11类，用于调用指定yolo11模型推理图片，并返回识别到的Object
 * 目前只被buff_detector调用 */
class YOLO11_BUFF {
    public:
        struct Object { // YOLO11模型识别的物体
            cv::Rect_<float> rect; // 矩形框
            float confidence = 0.0f; // 置信度
            std::vector<cv::Point2f> kpt; // 关键点
            // 目前，我们的模型关键点有6个：
            // 第一个关键点在靶子圆上、扇叶最外侧，
            // 第二、三、四关键点与一在靶子圆形处逆时针排列。
            // 第五个在圆中心。
            // 第六个在靶子尾部。
        };

        YOLO11_BUFF(const std::string & config_path, const std::string & model_key);
        std::vector<Object> get_candidateboxes(cv::Mat & image);

    private:
        struct OutputLayout {
            int channels = 0;
            int candidates = 0;
            bool channel_first = true;// 是否通道优先
                // 若通道优先，则数据格式为：先存通道a 的所有候选框数据，再存通道b 的所有候选框数据……以此类推。
                // 否则是候选框优先，数据格式为：先存候选框0 的所有通道数据，再存候选框1 的所有通道数据……以此类推。
            int keypoint_stride = 0; // 单个关键点
            const char * format_name = "unknown";
        };
        //=== OpenVINO相关 ===
        ov::Core core; // 创建OpenVINO Runtime Core对象
        std::shared_ptr<ov::Model> model;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request_;
        ov::Tensor input_tensor_;
        std::string model_path_;

        bool is_draw_info_ = true;
        bool output_layout_logged_ = false;
        const int NUM_POINTS = 6;

        std::string get_mode_path(const YAML::Node & yaml, const std::string & model_key); // 获取模型路径

        void convert(const cv::Mat & input, cv::Mat & output); // 对图像归一化+BRG转RGB
        float fillInputTensor(const cv::Mat & input_image); // 缩放图片、向输入层写入，并返回缩放比例

        std::optional<OutputLayout> checkOutputLayout(const ov::Shape & shape); // 检查并返回输出层形状
        float outputAt(const float * data, const OutputLayout & layout, int channel, int candidate) const;
        std::vector<Object> getCandidates(const ov::Tensor & output, float factor); // 读取输出层数据以获取候选目标
        // 调试用函数
        void drawObject(cv::Mat & image, const Object & obj, const cv::Scalar & point_color) const; // 画图
        void printInputAndOutputsInfo(const ov::Model & network);
            // 打印模型信息, 这个函数修改自$${OPENVINO_COMMON}/utils/src/args_helper.cpp的同名函数
    };

}

#endif  // AUTO_BUFF__YOLO11_BUFF_HPP