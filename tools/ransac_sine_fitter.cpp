#include "ransac_sine_fitter.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>

namespace tools {
    
RansacSineFitter::RansacSineFitter(
    int max_iterations, double threshold, double min_omega, double max_omega)
    :   max_iterations_(max_iterations),
        threshold_(threshold),
        min_omega_(min_omega),
        max_omega_(max_omega),
        gen_(std::random_device{}()) {}

void RansacSineFitter::add_data(double t, double v) {
    if (data_.size() > 0 && (t - data_.back().first > 5)) data_.clear();
        // 旧数据晚于5秒视为丢失大符，删除数据（等下不应该是detector发指令吗？）
    data_.emplace_back(std::make_pair(t, v));
}

void RansacSineFitter::fit() {
  if (data_.size() < 3) return;

    std::uniform_real_distribution<double> omega_dist(min_omega_, max_omega_);
        // 搭配std::mt19937使用，生成[min_omega_, max_omega_]范围内的随机数 
    std::vector<size_t> indices(data_.size());
    std::iota(indices.begin(), indices.end(), 0);
        // 生成从0开始的连续递增整数序列，存入indices

    for (int iter = 0; iter < max_iterations_; ++iter) {
        std::shuffle(indices.begin(), indices.end(), gen_); // 打乱indices
        std::vector<std::pair<double, double>> sample;
        for (int i = 0; i < 3; ++i) {
            sample.push_back(data_[indices[i]]);
        }

        double omega = omega_dist(gen_);
        Eigen::Vector3d params;
        if (!fit_partial_model(sample, omega, params)) continue;

        double A1 = params(0);
        double A2 = params(1);
        double C = params(2);

        double A = std::sqrt(A1 * A1 + A2 * A2);
        double phi = std::atan2(A2, A1);

        int inlier_count = evaluate_inliers(A, omega, phi, C);

        if (inlier_count > best_result_.inliers) {
            best_result_.A = A;
            best_result_.omega = omega;
            best_result_.phi = phi;
            best_result_.C = C;
            best_result_.inliers = inlier_count;
        }
    }
    
    if (data_.size() > 150) data_.pop_front();
}
bool RansacSineFitter::fit_partial_model(
    const std::vector<std::pair<double, double>> & sample, 
    double omega, Eigen::Vector3d & params) {
    Eigen::MatrixXd X(sample.size(), 3);
    Eigen::VectorXd Y(sample.size());

    for (size_t i = 0; i < sample.size(); ++i) {
        double t = sample[i].first;
        double y = sample[i].second;
        X(i, 0) = std::sin(omega * t);
        X(i, 1) = std::cos(omega * t);
        X(i, 2) = 1.0;
        Y(i) = y;
    }

    try {
        params = X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
        // 求最小二乘解
        return true;
    } catch (...) {
        return false;
    }
    
}

int RansacSineFitter::evaluate_inliers(
    double A, double omega, double phi, double C) {
    int count = 0;
    for (const auto & p : data_) {
        double t = p.first;
        double v = p.second;
        double f = A * std::sin(omega * t + phi) + C;
        if (std::abs(f - v) < threshold_) ++count;
    }
    return count;
}

} // namespace tools