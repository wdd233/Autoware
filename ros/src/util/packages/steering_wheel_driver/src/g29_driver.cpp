#include <steering_wheel_driver/g29_driver.h>

G29Driver::G29Driver(ros::NodeHandle nh,ros::NodeHandle pnh)
{
    nh_ = nh;
    pnh_ = pnh;
}

G29Driver::~G29Driver()
{

}

void G29Driver::run()
{
    return;
}

void G29Driver::publishCmd()
{
    return;
}

/*
int G29Driver::setup(void)
{

}

void G29Driver::ActuaterFeedback(const std_msgs::UInt16MultiArray& ActuatorStatus)
{
    return;
}
*/

int G29Driver::ffUpdate(SDL_Joystick * joystick , unsigned short center)
{
    SDL_Haptic *haptic;
    SDL_HapticEffect effect;
    int effect_id;
    haptic = SDL_HapticOpenFromJoystick( joystick );
    if (haptic == NULL) return -1; // Most likely joystick isn't haptic
}

void steerFeedback(const std_msgs::UInt16MultiArray& FFStatus)
{
    return;
}