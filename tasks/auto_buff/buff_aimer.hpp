#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <vector>

#include "../auto_aim/planner/planner.hpp"
#include "buff_target.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"

namespace auto_buff
{

enum class AimMode {
  RING_PRIORITY,
  ARM_PRIORITY,
  BALANCED,
  AUTO
};

struct RingTarget {
  int ring = 10;
  double weight = 1.0;
};

class Aimer
{
public:
  Aimer(const std::string & config_path);

  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true);

  io::Command aimWithMode(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    AimMode mode = AimMode::RING_PRIORITY, int target_ring = 10, bool to_now = true);

  void setAimMode(AimMode mode) { aim_mode_ = mode; }
  void setTargetRing(int ring) { target_ring_ = std::clamp(ring, 1, 10); }

  double getCurrentAngle() const { return angle_; }
  int getTargetRing() const { return target_ring_; }
  int getLastHitRing() const { return last_hit_ring_; }

private:
  bool getSendAngle(
    Target & target, const double predict_time, const double bullet_speed, const bool to_now,
    double & yaw, double & pitch, AimMode mode, int target_ring);

  Eigen::Vector3d adjustAimPointToRing(
    const Eigen::Vector3d & original_point, const Eigen::Vector3d & center, int ring) const;
  double ringToRadius(int ring) const;
  Eigen::Vector3d getCurrentArmCenter(Target & target) const;

  double yaw_offset_;
  double pitch_offset_;
  double fire_gap_time_;
  double predict_time_;

  double max_radius_mm_;
  int outer_ring_;
  int inner_ring_;

  AimMode aim_mode_;
  int target_ring_;

  int mistake_count_ = 0;
  bool switch_fanblade_;
  double last_yaw_ = 0;
  double last_pitch_ = 0;
  double angle_ = 0;
  int last_hit_ring_ = 0;

  bool first_in_aimer_ = true;
  std::chrono::steady_clock::time_point last_fire_t_;
  std::vector<double> ring_weights_;
  std::deque<int> ring_history_;
};

}  // namespace auto_buff
#endif  // AUTO_AIM__AIMER_HPP
