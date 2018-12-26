#ifndef G29_DRIVER_H_INCLUDED
#define G29_DRIVER_H_INCLUDED

#include <ros/ros.h>

class g29_driver
{
public:
    g29_driver(ros::NodeHandle nh,ros::NodeHandle pnh);
    ~g29_driver();
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
};

#endif  //G29_DRIVER_H_INCLUDED