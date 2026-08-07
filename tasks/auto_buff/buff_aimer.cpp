#include "buff_aimer.hpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{

Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  yaw_offset_ = yaml["yaw_offset-auto_buff"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset-auto_buff"].as<double>() / 57.3;
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();

  max_radius_mm_ = yaml["ring_mapping"]["max_radius_mm"].as<double>(150.0);
  outer_ring_ = yaml["ring_mapping"]["outer_ring"].as<int>(1);
  inner_ring_ = yaml["ring_mapping"]["inner_ring"].as<int>(10);

  aim_mode_ = AimMode::RING_PRIORITY;
  target_ring_ = 10;

  ring_weights_.resize(11, 1.0);
  for (int i = 1; i <= 10; i++) {
    ring_weights_[i] = 1.0;
  }

  last_fire_t_ = std::chrono::steady_clock::now();
  ring_history_.clear();
}

double Aimer::ringToRadius(int ring) const
{
  double t = static_cast<double>(ring - inner_ring_) / (outer_ring_ - inner_ring_);
  return max_radius_mm_ * (1.0 - t);
}

Eigen::Vector3d Aimer::adjustAimPointToRing(
  const Eigen::Vector3d & original_point, const Eigen::Vector3d & center, int ring) const
{
  double radius_m = ringToRadius(ring) / 1000.0;
  Eigen::Vector3d dir = original_point - center;

  if (dir.norm() < 1e-6) {
    return original_point;
  }

  dir.normalize();
  return center + dir * radius_m;
}

Eigen::Vector3d Aimer::getCurrentArmCenter(Target & target) const
{
  return target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
}

bool Aimer::getSendAngle(
  Target & target, const double predict_time, const double bullet_speed, const bool to_now,
  double & yaw, double & pitch, AimMode mode, int target_ring)
{
  const auto snapshot = target.snapshot();
  auto restore_state = [&]() { target.restore(snapshot); };

  target.predict(predict_time);
  const Eigen::VectorXd predicted_state = target.ekf_x();
  if (predicted_state.size() == 0 || !predicted_state.allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：预测后的目标状态非法");
    restore_state();
    return false;
  }
  angle_ = target.ekf_x()[5];

  Eigen::Vector3d center = target.point_buff2world(Eigen::Vector3d(0, 0, 0));
  Eigen::Vector3d aim_in_world = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  if (!center.allFinite() || !aim_in_world.allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：目标点坐标非法");
    restore_state();
    return false;
  }

  if (mode == AimMode::RING_PRIORITY) {
    double avg_recent_ring = 0;
    if (!ring_history_.empty()) {
      for (int r : ring_history_) {
        avg_recent_ring += r;
      }
      avg_recent_ring /= ring_history_.size();
    }

    int dynamic_target = target_ring;
    if (avg_recent_ring > 0) {
      if (avg_recent_ring < 5) {
        dynamic_target = std::min(target_ring + 2, 10);
      } else if (avg_recent_ring > 8) {
        dynamic_target = std::max(target_ring - 1, 1);
      }
    }

    static int explore_counter = 0;
    explore_counter++;
    if (explore_counter % 30 == 0) {
      dynamic_target = (rand() % 9) + 1;
    }

    aim_in_world = adjustAimPointToRing(aim_in_world, center, dynamic_target);
  } else if (mode == AimMode::ARM_PRIORITY) {
    aim_in_world = getCurrentArmCenter(target);
  } else if (mode == AimMode::BALANCED) {
    int last_ring = target.getLastHitRing();
    if (last_ring > 0 && last_ring <= 10) {
      int adjusted_ring = std::min(last_ring + 2, 10);
      aim_in_world = adjustAimPointToRing(aim_in_world, center, adjusted_ring);
    } else {
      aim_in_world = adjustAimPointToRing(aim_in_world, center, 10);
    }
  }

  double d = std::sqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  double h = aim_in_world[2];
  if (!std::isfinite(d) || !std::isfinite(h)) {
    tools::logger()->debug("[Aimer] 跳过本帧：弹道输入非法 d={} h={}", d, h);
    restore_state();
    return false;
  }

  tools::Trajectory trajectory0(bullet_speed, d, h);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d, h);
    restore_state();
    return false;
  }

  target.predict(trajectory0.fly_time);
  if (!target.ekf_x().allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：飞行时间预测后的目标状态非法");
    restore_state();
    return false;
  }
  aim_in_world = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));

  if (mode == AimMode::RING_PRIORITY) {
    aim_in_world = adjustAimPointToRing(aim_in_world, center, target_ring);
  }
  if (!aim_in_world.allFinite()) {
    tools::logger()->debug("[Aimer] 跳过本帧：二次预测后的目标点坐标非法");
    restore_state();
    return false;
  }

  d = std::sqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  h = aim_in_world[2];
  if (!std::isfinite(d) || !std::isfinite(h)) {
    tools::logger()->debug("[Aimer] 跳过本帧：二次弹道输入非法 d={} h={}", d, h);
    restore_state();
    return false;
  }

  tools::Trajectory trajectory1(bullet_speed, d, h);
  if (trajectory1.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory1: {:.2f} {:.2f} {:.2f}", bullet_speed, d, h);
    restore_state();
    return false;
  }

  auto time_error = trajectory1.fly_time - trajectory0.fly_time;
  if (std::abs(time_error) > 0.025) {
    tools::logger()->debug("[Aimer] Large time error: {:.3f}", time_error);
    restore_state();
    return false;
  }

  yaw = std::atan2(aim_in_world[1], aim_in_world[0]) + yaw_offset_;
  pitch = trajectory1.pitch + pitch_offset_;
  if (!std::isfinite(yaw) || !std::isfinite(pitch)) {
    tools::logger()->debug("[Aimer] 跳过本帧：输出角度非法 yaw={} pitch={}", yaw, pitch);
    restore_state();
    return false;
  }

  last_hit_ring_ = target.getLastHitRing();
  if (last_hit_ring_ > 0) {
    ring_history_.push_back(last_hit_ring_);
    if (ring_history_.size() > 10) {
      ring_history_.pop_front();
    }
  }

  restore_state();
  return true;
}

io::Command Aimer::aim(
  Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
  bool to_now)
{
  return aimWithMode(target, timestamp, bullet_speed, aim_mode_, target_ring_, to_now);
}

io::Command Aimer::aimWithMode(
  Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
  AimMode mode, int target_ring, bool to_now)
{
  io::Command command = {false, false, 0, 0};

  if (target.is_unsolve()) return command;

  bullet_speed = 23.2;

  auto now = std::chrono::steady_clock::now();
  auto detect_now_gap = tools::delta_time(now, timestamp);
  double future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;

  double yaw, pitch;

  if (getSendAngle(target, future, bullet_speed, to_now, yaw, pitch, mode, target_ring)) {
    command.yaw = yaw;
    command.pitch = -pitch;

    bool angle_changed =
      std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3;

    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      command.control = true;
    } else if (angle_changed) {
      switch_fanblade_ = true;
      mistake_count_++;
      command.control = false;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      command.control = true;
    }

    last_yaw_ = yaw;
    last_pitch_ = pitch;
  }

  if (switch_fanblade_) {
    command.shoot = false;
    last_fire_t_ = now;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    command.shoot = true;
    last_fire_t_ = now;
  }

  return command;
}

}  // namespace auto_buff
