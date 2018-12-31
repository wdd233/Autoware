#ifndef PROJECT_PIXEL_CLOUD_FUSION_GPU_H
#define PROJECT_PIXEL_CLOUD_FUSION_GPU_H

#include <vector>
#include <cuda_runtime.h>

#include <tf/tf.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

class PixelCloudFusionGPU {
public:
    PixelCloudFusionGPU(void);
    ~PixelCloudFusionGPU(void);
    void Fusion(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                cv::Mat&                            input_image,
                tf::StampedTransform&               camera_lidar_tf,
                float&                               fx,
                float&                               fy,
                float&                               cx,
                float&                               cy,
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr& output_cloud
                );
protected:
    bool is_device_memory_ready_;
    size_t allocated_points_size_;
    int kNumThreadsPerBlock_ = 256;

    float* host_input_cloud_x_;
    float* host_input_cloud_y_;
    float* host_input_cloud_z_;
    float* dev_input_cloud_x_;
    float* dev_input_cloud_y_;
    float* dev_input_cloud_z_;

    unsigned char* dev_input_image_b_;
    unsigned char* dev_input_image_g_;
    unsigned char* dev_input_image_r_;

    float* host_output_cloud_r_;
    float* host_output_cloud_g_;
    float* host_output_cloud_b_;
    float* dev_output_cloud_r_;
    float* dev_output_cloud_g_;
    float* dev_output_cloud_b_;

    bool* host_valid_points_;
    bool* dev_valid_points_;
};

#endif  // PROJECT_PIXEL_CLOUD_FUSION_GPU_H
