#ifndef G29_DRIVER_H_INCLUDED
#define G29_DRIVER_H_INCLUDED


//headers in ROS
#include <ros/ros.h>

//headers in Autoware
#include <autoware_msgs/VehicleCmd.h>
#include <autoware_msgs/VehicleStatus.h>
#include <steering_wheel_driver/g29_joyinfo.h>
#include <std_msgs/UInt16MultiArray.h>

//headers in SDL
#include <SDL2/SDL.h>

//headers in boost
#include <boost/optional.hpp>

class G29Driver
{
public:
    G29Driver(ros::NodeHandle nh,ros::NodeHandle pnh);
    ~G29Driver();
    void run();
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher cmd_pub_;
    void publishCmd();
    /*
    int setup(void);
    void ActuaterFeedback(const std_msgs::UInt16MultiArray& ActuatorStatus);
    */
    int ffUpdate(SDL_Joystick * joystick , unsigned short center);
    void vehicleStatusFeedback(autoware_msgs::VehicleStatus msg);
    boost::optional<autoware_msgs::VehicleStatus> status_;
    void setup();
    int joystick_id_;
    SDL_Joystick *joy_;
    SDL_Haptic *haptic_;
    //SDL_HapticEffect effect_;
    void changeSteeringAngle(double target_angle);
};

#endif  //G29_DRIVER_H_INCLUDED