#include <steering_wheel_driver/g29_driver.h>

g29_driver::g29_driver(ros::NodeHandle nh,ros::NodeHandle pnh)
{
    nh_ = nh;
    pnh_ = pnh;
}

g29_driver::~g29_driver()
{

}