#pragma once

#include <Eigen/Dense>
#include <deque>
#include <random>
#include <vector>

namespace tools {

class RansacSineFitter {
    public:
        struct SinParameters {
            double A = 0.0;     // 振幅
            double omega = 0.0; // 频率
            double phi = 0.0;   // 相位
            double C = 0.0;     // 偏移量
            int inliers = 0;    // 内点数量
        };
        SinParameters best_result_;

        RansacSineFitter(int max_iterations, double threshold, double min_omega, double max_omega);
        
        void add_data(double t, double v); // 添加数据点

        void fit(); // 主程序，执行拟合

        double sin_f(double t) {
            return best_result_.A * sin(best_result_.omega * t + best_result_.phi) + best_result_.C;
        }

    private:
        int max_iterations_; // 最大迭代次数
        double threshold_; // 预测容差，即预测轨迹与观测数据的容差
        double min_omega_; // 能量机关最小转速
        double max_omega_; // 能量机关最大转速
        std::mt19937 gen_; // 随机数生成器
        std::deque<std::pair<double, double>> data_; // 观测数据

        bool fit_partial_model(const std::vector<std::pair<double, double>> & sample, 
            double omega, Eigen::Vector3d & params); // 给定omega，拟合A, phi, C

        int evaluate_inliers(double A, double omega, double phi, double C); // 找出符合模型的点的数量
};

} // namespace tools