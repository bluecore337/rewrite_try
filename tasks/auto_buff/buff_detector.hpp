#ifndef AUTO_BUFF__DETECTOR_HPP
#define AUTO_BUFF__DETECTOR_HPP

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "yolo11_buff.hpp"

namespace auto_buff
{
enum class BuffColor {
  RED,
  BLUE
};

class Target;
}

const int LOSE_MAX = 60;

namespace auto_buff
{

enum class TrackStatus {
  TRACK,
  TEM_LOSE,
  PREDICT_TRACK,
  LOSE
};

class Buff_Detector
{
public:
  explicit Buff_Detector(const std::string & config);
  Buff_Detector(const std::string & config, const std::string & model_key);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

  void setEnergyType(EnergyType type);
  void setTarget(std::shared_ptr<Target> target_ptr) { target_ = target_ptr; }

private:
  static const char * colorName(BuffColor color);
  static BuffColor oppositeColor(BuffColor color);
  static BuffColor classifyColor(const cv::Mat & bgr_img, const YOLO11_BUFF::Object & result);
  std::vector<YOLO11_BUFF::Object> filterByConfiguredColor(
    const cv::Mat & bgr_img, const std::vector<YOLO11_BUFF::Object> & results) const;

  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);
  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);
  void handle_lose();
  PowerRune constructPredictedPowerRune();
  void updateDetectionInfo(const std::vector<YOLO11_BUFF::Object> & results);
  void fillBoardMetadata(
    PowerRune & powerrune, const std::vector<YOLO11_BUFF::Object> & results) const;

  YOLO11_BUFF MODE_;
  TrackStatus status_;
  int lose_;
  double lastlen_;
  std::optional<PowerRune> last_powerrune_;

  BuffColor configured_energy_color_;
  BuffColor target_energy_color_;
  bool energy_color_configured_;

  std::shared_ptr<Target> target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  PowerRune predicted_powerrune_;
  bool prediction_valid_;

  EnergyType energy_type_;

  std::vector<cv::Point2f> last_detection_centers_;
  std::vector<int> target_track_counts_;
  static constexpr int MAX_TRACK_COUNT = 100;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__DETECTOR_HPP
