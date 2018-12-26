#ifndef G29_DRIVER_H_INCLUDED
#define G29_DRIVER_H_INCLUDED


//headers in ROS
#include <ros/ros.h>

//headers in Autoware
#include <autoware_msgs/VehicleCmd.h>
#include <steering_wheel_driver/g29_joyinfo.h>

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
};

#endif  //G29_DRIVER_H_INCLUDED