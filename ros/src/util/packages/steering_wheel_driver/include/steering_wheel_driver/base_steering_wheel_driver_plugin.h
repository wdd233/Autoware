#ifndef BASE_STEERING_WHEEL_DRIVER_PLUGIN_H_INCLUDED
#define BASE_STEERING_WHEEL_DRIVER_PLUGIN_H_INCLUDED

#include <autoware_msgs/VehicleCmd.h>

namespace steering_wheel_driver
{
    class BaseSteeringWheelDriverPlugin
    {
        public:
            virtual ~BaseSteeringWheelDriverPlugin();
            virtual autoware_msgs::VehicleCmd getVehicleCmd();
        protected:
            BaseSteeringWheelDriverPlugin(){};
    };
}

#endif  //BASE_STEERING_WHEEL_DRIVER_PLUGIN_H_INCLUDED