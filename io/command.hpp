#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

namespace io {
struct Command {
    bool control; // 是否使用自瞄指令控制
    bool shoot; // 是否开火

    double yaw;
    double pitch;

    uint8_t armor_name_num; // 当前瞄准的装甲板是哪种类型
        // 0--无, 1--英雄, 2--工程, 3/4--步兵, 5--哨兵
};

struct MPCCommand{
    bool control; // 是否使用自瞄指令控制
    bool shoot; // 是否开火
    
    double yaw;
    double yaw_vel;
    double yaw_acc;
    double pitch;
    double pitch_vel;
    double pitch_acc;

    uint8_t armor_name_num; // 当前瞄准的装甲板是哪种类型
        // 0--无, 1--英雄, 2--工程, 3/4--步兵, 5--哨兵
};

} // namespace io

#endif // IO__COMMAND_HPP