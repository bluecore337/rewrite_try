#include "yolo11_buff.hpp"

#include <chrono>
#include <sstream>
#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

const double CONFIDENCE_THRESH = 0.40f; // 置信度阈值，低于这个置信度则忽略目标
const double IOU_THRESH = 0.60f; // 交并比(IoU)阈值，高于这个阈值的两个框视为高度重叠

namespace {
std::string shape_to_string(const ov::Shape & shape) {
    std::ostringstream stream;
    stream << shape;
    return stream.str();
}

}  // namespace

namespace auto_buff {
//=== 初始化 ===
std::string YOLO11_BUFF::get_mode_path(const YAML::Node & yaml, const std::string & model_key) {
    // 从参数文件中读取指定参数以获取模型路径，若找不到则使用model参数给出的路径。
    if (yaml[model_key]) return yaml[model_key].as<std::string>();
    if (yaml["mode"]) {
        tools::logger()->warn(
        "[YOLO11_BUFF] {} 未配置，回退使用 model={}", model_key, yaml["model"].as<std::string>());
        return yaml["model"].as<std::string>();
    }
    throw std::runtime_error("[YOLO11_BUFF] yaml 缺少 model 配置");
}

YOLO11_BUFF::YOLO11_BUFF(const std::string & config_path, const std::string & model_key) {
    auto yaml = tools::load(config_path);
    if (yaml["draw_buff"]) is_draw_info_ = yaml["draw_buff"].as<bool>();
    model_path_ = get_mode_path(yaml, model_key);
    
    model = core.read_model(model_path_);
    // printInputAndOutputsInfo(*model);  // 打印模型信息
    compiled_model = core.compile_model(model, "CPU"); // 载入并编译模型
    infer_request_ = compiled_model.create_infer_request(); // 创建推理请求
    input_tensor_ = infer_request_.get_input_tensor(); // 获取模型输入节点
    input_tensor_.set_shape({1, 3, 640, 640});
    tools::logger()->info("[YOLO11_BUFF] loaded model: {}", model_path_);
}

//=== 主程序 ===
std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_candidateboxes(cv::Mat & image) {
    if (image.empty()) {
        tools::logger()->warn("[YOLO11_BUFF] Received empty image!");
        return std::vector<YOLO11_BUFF::Object>();  
    }
    const int64 start = cv::getTickCount();
    //--- 写入、推理、获取输出层 ---
    const float scale_factor = fillInputTensor(image);
    infer_request_.infer();
    const ov::Tensor output = infer_request_.get_output_tensor();

    //--- 读取和筛选目标 ---
    const auto candidates = getCandidates(output, scale_factor);
    std::vector<cv::Rect> boxes; // 候选框列表
    std::vector<float> confidences; // 对应的置信度
    boxes.reserve(candidates.size());
    confidences.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        boxes.push_back(candidate.rect);
        confidences.push_back(candidate.confidence);
    }
    std::vector<int> indexes; // 存储保留的候选框的索引
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESH, IOU_THRESH, indexes);
        // 筛选出置信度大于CONFIDENCE_THRESH的框，并根据交并比去除高度重合的框。

    //--- 写入结果并返回 ---
    std::vector<Object> object_result;
    for (size_t i = 0; i < indexes.size(); ++i) {
        const int index = indexes[i];
        object_result.push_back(candidates[index]);
        drawObject(image, candidates[index], cv::Scalar(255, 0, 0)); // 画图
    }
    const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40),
        cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
    return object_result;
}
//=== 转换图片并写入输入层 ===
void YOLO11_BUFF::convert(const cv::Mat & input, cv::Mat & output){
    input.convertTo(output, CV_32F); // CV_32F其实就是float，这是将像素转为float类型
    output = output / 255.0; // 归一化处理，即将像素BGR数据从0~255变成0~1.0
    cv::cvtColor(output, output, cv::COLOR_BGR2RGB); // opencv存储图像是BGR，而openvino需要RGB图像
}

float YOLO11_BUFF::fillInputTensor(const cv::Mat & input_image) {
    //--- 计算缩放比例 scale ---
    const ov::Shape tensor_shape = input_tensor_.get_shape(); // [1,3,640,640]
    const size_t num_channels = tensor_shape[1]; // 3，即三个颜色通道BGR
    const size_t height = tensor_shape[2];       // 输入图像高度
    const size_t width = tensor_shape[3];        // 输入图像宽度
    const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
        // scale 是缩放倍数的倒数。使用倒数的原因需要看下文和了解 cv::warpAffine(...) 函数。

    //--- 进行图片缩放和转换 ---
    // 由于转换convert中有除法，减少convert中处理的像素数量可以降低算力开支。
    // cv::warpAffine(...) 函数在说明文件中有介绍。
    const cv::Matx23f matrix{
        scale, 0.0, 0.0,
        0.0, scale, 0.0,
    };
    cv::Mat blob_image;
    if (scale < 1.0f) { // 图像缩小时，先缩小再convert。
        cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
        convert(blob_image, blob_image);
    } else { // 图像放大时，先convert再缩小。
        convert(input_image, blob_image);
        cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
    }

    //--- 向input_tensor写入数据 ---
    float * const input_tensor_data = input_tensor_.data<float>();
    for(size_t c = 0; c < num_channels; c++) {
        for(size_t h = 0; c < height; c++) {
            for(size_t w = 0; c < width; c++) {
                input_tensor_data[c * width * height + h * width + w] =
                    blob_image.at<cv::Vec<float, 3>>(h, w)[c];
            }
        }
    }

    return 1 / scale;
}

//=== 读取输出层数据 ===
std::optional<YOLO11_BUFF::OutputLayout> YOLO11_BUFF::checkOutputLayout(const ov::Shape & shape) {
    if (shape.size() != 3 || shape[0] != 1) {
        tools::logger()->warn("[YOLO11_BUFF] unsupported output shape: {}", shape_to_string(shape));
        return std::nullopt;
    }

    OutputLayout layout;
    if (shape[1] == 17 || shape[1] == 23) {
        layout.channels = static_cast<int>(shape[1]);
        layout.candidates = static_cast<int>(shape[2]);
        layout.channel_first = true;
    } else if (shape[2] == 17 || shape[2] == 23) {
        layout.channels = static_cast<int>(shape[2]);
        layout.candidates = static_cast<int>(shape[1]);
        layout.channel_first = false;
    } else {
        tools::logger()->warn(
        "[YOLO11_BUFF] unsupported output shape: {}, expected channels 17 or 23",
        shape_to_string(shape));
        return std::nullopt;
    }

    if (layout.channels == 17) {
        layout.keypoint_stride = 2;
        layout.format_name = "xy";
    } else {
        layout.keypoint_stride = 3;
        layout.format_name = "xy_score";
    }

    if (!output_layout_logged_) {
        tools::logger()->info(
        "[YOLO11_BUFF] output layout: shape={}, channels={}, candidates={}, format={}, order={}",
        shape_to_string(shape), layout.channels, layout.candidates, layout.format_name,
        layout.channel_first ? "channel_first" : "candidate_first");
        output_layout_logged_ = true;
    }

    return layout;
}

float YOLO11_BUFF::outputAt(const float * data, 
    const OutputLayout & layout, int channel, int candidate) const {
  if (layout.channel_first) return data[channel * layout.candidates + candidate];
  return data[candidate * layout.channels + channel];
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::getCandidates(const ov::Tensor & output, float factor) {
    const auto layout = checkOutputLayout(output.get_shape());
    if (!layout.has_value()) return {};

    std::vector<Object> candidates;
    const float * output_buffer = output.data<const float>();
    for (int i = 0; i < layout->candidates; ++i) {
        const float score = outputAt(output_buffer, *layout, 4, i);
        if (score <= CONFIDENCE_THRESH) continue;

        const float cx = outputAt(output_buffer, *layout, 0, i);
        const float cy = outputAt(output_buffer, *layout, 1, i);
        const float ow = outputAt(output_buffer, *layout, 2, i);
        const float oh = outputAt(output_buffer, *layout, 3, i);

        Object candidate;
        candidate.rect.x = static_cast<int>((cx - 0.5F * ow) * factor);
        candidate.rect.y = static_cast<int>((cy - 0.5F * oh) * factor);
        candidate.rect.width = static_cast<int>(ow * factor);
        candidate.rect.height = static_cast<int>(oh * factor);
        candidate.confidence = score;
        candidate.kpt.reserve(NUM_POINTS);
        for (int j = 0; j < NUM_POINTS; ++j) {
            const int base = 5 + j * layout->keypoint_stride;
            const float x = outputAt(output_buffer, *layout, base, i) * factor;
            const float y = outputAt(output_buffer, *layout, base + 1, i) * factor;
            candidate.kpt.emplace_back(x, y);
        }
        candidates.push_back(std::move(candidate));
    }

    return candidates;
}

//=== 调试用：画图 ===
void YOLO11_BUFF::drawObject(cv::Mat & image, 
    const Object & obj, const cv::Scalar & point_color) const {
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
    const std::string label = "buff:" + std::to_string(obj.confidence).substr(0, 4);
    const cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(
        obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
        image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(0, 0, 0));
    for (int j = 0; j < static_cast<int>(obj.kpt.size()); ++j) {
        cv::circle(image, obj.kpt[j], 2, point_color, -1, cv::LINE_AA);
    }
}

//=== 调试用：打印模型信息（从sp_vision复制来的） ===
void YOLO11_BUFF::printInputAndOutputsInfo(const ov::Model & network) {
  std::cout << "model name: " << network.get_friendly_name() << std::endl;

  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> & input : inputs) {
    std::cout << "    inputs" << std::endl;

    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;

    const ov::element::Type type = input.get_element_type();
    std::cout << "        input type: " << type << std::endl;

    const ov::Shape shape = input.get_shape();
    std::cout << "        input shape: " << shape << std::endl;
  }

  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> & output : outputs) {
    std::cout << "    outputs" << std::endl;

    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;

    const ov::element::Type type = output.get_element_type();
    std::cout << "        output type: " << type << std::endl;

    const ov::Shape shape = output.get_shape();
    std::cout << "        output shape: " << shape << std::endl;
  }
}
} // namespace auto_buff