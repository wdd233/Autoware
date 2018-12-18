#ifndef G29_PLUGIN_H_INCLUDED
#define G29_PLUGIN_H_INCLUDED

//headers in ROS
#include <ros/ros.h>
#include <pluginlib/class_list_macros.h>

//headers in this package
#include <steering_wheel_driver/base_steering_wheel_driver_plugin.h>

namespace steering_wheel_driver
{
    class g29_plugin
    {
        public:
            g29_plugin();
            ~g29_plugin();
    };
}

#endif  //G29_PLUGIN_H_INCLUDED