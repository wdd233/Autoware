#include "pixel_cloud_fusion/pixel_cloud_fusion_gpu.hpp"
#include <vector>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) {CudaErrorCheck((err), __FILE__, __LINE__);}

namespace
{
    cudaTextureObject_t gTextureObject_b = 0;
    cudaTextureObject_t gTextureObject_g = 0;
    cudaTextureObject_t gTextureObject_r = 0;

    ////////////////////
    // Helper structures to transform coordinates
    //
    struct HelperVector3 {
        double x;
        double y;
        double z;

        template<typename T>
        __host__ __device__
        HelperVector3(T& in_x, T& in_y, T& in_z) {
            x = static_cast<double>(in_x);
            y = static_cast<double>(in_y);
            z = static_cast<double>(in_z);
        }

        __host__ __device__
        double dot(const HelperVector3& v) {
            return x * v.x + y * v.y + z * v.z;
        }
    };

    struct HelperMatrix3x3 {
        HelperVector3 m_el[3];
    };

    ////////////////////
    // Wrapper function to check CUDA API call status
    //
    inline void CudaErrorCheck(cudaError_t err,
                               const char* file,
                               const int line,
                               bool abort=true
                               )
    {
        if (err != cudaSuccess) {
            std::cerr << "Error occured while CUDA API call: " << std::endl
                      << cudaGetErrorString(err) << std::endl
                      << "@" << file << "(" << line << ")" << std::endl;
            if (abort) exit(EXIT_FAILURE);
        }
    } // inline void CudaErrorCheck()

    ////////////////////
    // Wrapper function to allocate host and device linear memory
    //
    template <typename T>
    void AllocateLinear(T** host, T** device, size_t num_points)
    {
        // Allocate host memory
        *host = new T[num_points];
        std::memset(*host, 0, num_points*sizeof(T));

        // Allocate devine memory
        CUDA_ERROR_CHECK(cudaMalloc(device, num_points*sizeof(T)));
        CUDA_ERROR_CHECK(cudaMemset(*device, 0, num_points*sizeof(T)));
    }  // void AllocateLinear()


    ////////////////////
    // Wrapper function to free host and device linear memory
    //
    template <typename T>
    void FreeLinear(T* host, T* device)
    {
        // Free host memory
        if (host != nullptr) {
            delete host;
        }

        // Free devine memory
        if (device != nullptr) {
            CUDA_ERROR_CHECK(cudaFree(device));
        }
    }  // void FreeLinear()


    ////////////////////
    // GPU function to process each input points in parallel
    //
    __global__
    void FusionKernel(int num_points,
                      int2 image_size,
                      float2 f,
                      float2 c,
                      HelperVector3 transform_origin,
                      HelperMatrix3x3 transform_basis,
                      cudaTextureObject_t tex_b,
                      cudaTextureObject_t tex_g,
                      cudaTextureObject_t tex_r,
                      float* input_cloud_x,
                      float* input_cloud_y,
                      float* input_cloud_z,
                      float* output_cloud_b,
                      float* output_cloud_g,
                      float* output_cloud_r,
                      bool* valid_points)
    {
        // Calculate global index of this thread
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) {
            return;
        }

        // Get the 3D coordinate of this point
        float x = input_cloud_x[idx];
        float y = input_cloud_y[idx];
        float z = input_cloud_z[idx];

        // Transform coordinate system from lidar to camera
        HelperVector3 tf_point(x, y, z);
        double cam_cloud_x = transform_basis.m_el[0].dot(tf_point) + transform_origin.x;
        double cam_cloud_y = transform_basis.m_el[1].dot(tf_point) + transform_origin.y;
        double cam_cloud_z = transform_basis.m_el[2].dot(tf_point) + transform_origin.z;

        // Get coordinate on image plane
        float u = static_cast<float>(cam_cloud_x * f.x / cam_cloud_z + c.x);
        float v = static_cast<float>(cam_cloud_y * f.y / cam_cloud_z + c.y);

        bool condition = (u >= 0 && u < image_size.x && v >= 0 && v < image_size.y && cam_cloud_z > 0);
        valid_points[idx] = condition;
        output_cloud_r[idx] =  tex2D<uchar>(tex_r, u, v);
        output_cloud_g[idx] =  tex2D<uchar>(tex_g, u, v);
        output_cloud_b[idx] =  tex2D<uchar>(tex_b, u, v);

    }  // void FusionKernel()

}  // namespace


PixelCloudFusionGPU::PixelCloudFusionGPU(void):
    is_device_memory_ready_(false),
    allocated_points_size_(0),
    host_input_cloud_x_(nullptr),
    host_input_cloud_y_(nullptr),
    host_input_cloud_z_(nullptr),
    dev_input_cloud_x_(nullptr),
    dev_input_cloud_y_(nullptr),
    dev_input_cloud_z_(nullptr),
    dev_input_image_b_(nullptr),
    dev_input_image_g_(nullptr),
    dev_input_image_r_(nullptr),
    host_output_cloud_r_(nullptr),
    host_output_cloud_g_(nullptr),
    host_output_cloud_b_(nullptr),
    dev_output_cloud_r_(nullptr),
    dev_output_cloud_g_(nullptr),
    dev_output_cloud_b_(nullptr),
    host_valid_points_(nullptr),
    dev_valid_points_(nullptr)
{
}  // PixelCloudFusionGPU::PixelCloudFusionGPU()


PixelCloudFusionGPU::~PixelCloudFusionGPU(void)
{
    if (is_device_memory_ready_) {
        // Release GPU memory
        FreeLinear(host_input_cloud_x_, dev_input_cloud_x_);
        FreeLinear(host_input_cloud_y_, dev_input_cloud_y_);
        FreeLinear(host_input_cloud_z_, dev_input_cloud_z_);
        FreeLinear(host_output_cloud_r_, dev_output_cloud_r_);
        FreeLinear(host_output_cloud_g_, dev_output_cloud_g_);
        FreeLinear(host_output_cloud_b_, dev_output_cloud_b_);

        FreeLinear(host_valid_points_, dev_valid_points_);

        CUDA_ERROR_CHECK(cudaFree(dev_input_image_b_));
        CUDA_ERROR_CHECK(cudaFree(dev_input_image_g_));
        CUDA_ERROR_CHECK(cudaFree(dev_input_image_r_));

        // Destroy texture objects
        CUDA_ERROR_CHECK(cudaDestroyTextureObject(gTextureObject_b));
        CUDA_ERROR_CHECK(cudaDestroyTextureObject(gTextureObject_g));
        CUDA_ERROR_CHECK(cudaDestroyTextureObject(gTextureObject_r));

    }
}  // PixelCloudFusionGPU::~PixelCloudFusionGPU()


void PixelCloudFusionGPU::Fusion(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                                 cv::Mat&                             input_image,
                                 tf::StampedTransform&                camera_lidar_tf,
                                 float&                                fx,
                                 float&                                fy,
                                 float&                                cx,
                                 float&                                cy,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr&     output_cloud)
{
    // Check whether enough amount of GPU memory is allocated
    static size_t dev_pitch = 0;
    if (!is_device_memory_ready_ || allocated_points_size_ < input_cloud->points.size())
        {
            // Free old region
            FreeLinear(host_input_cloud_x_, dev_input_cloud_x_);
            FreeLinear(host_input_cloud_y_, dev_input_cloud_y_);
            FreeLinear(host_input_cloud_z_, dev_input_cloud_z_);
            FreeLinear(host_output_cloud_r_, dev_output_cloud_r_);
            FreeLinear(host_output_cloud_g_, dev_output_cloud_g_);
            FreeLinear(host_output_cloud_b_, dev_output_cloud_b_);
            FreeLinear(host_valid_points_, dev_valid_points_);
            if (dev_input_image_b_ != nullptr) {CUDA_ERROR_CHECK(cudaFree(dev_input_image_b_));}
            if (dev_input_image_g_ != nullptr) {CUDA_ERROR_CHECK(cudaFree(dev_input_image_g_));}
            if (dev_input_image_r_ != nullptr) {CUDA_ERROR_CHECK(cudaFree(dev_input_image_r_));}
            if (gTextureObject_b != 0) {CUDA_ERROR_CHECK(cudaDestroyTextureObject(gTextureObject_b));}
            if (gTextureObject_g != 0) {CUDA_ERROR_CHECK(cudaDestroyTextureObject(gTextureObject_g));}
            if (gTextureObject_r != 0) {CUDA_ERROR_CHECK(cudaDestroyTextureObject(gTextureObject_r));}

            // Allocate linear memory on GPU if needed
            size_t new_size = input_cloud->points.size();
            AllocateLinear(&host_input_cloud_x_, &dev_input_cloud_x_, new_size);
            AllocateLinear(&host_input_cloud_y_, &dev_input_cloud_y_, new_size);
            AllocateLinear(&host_input_cloud_z_, &dev_input_cloud_z_, new_size);

            AllocateLinear(&host_output_cloud_r_, &dev_output_cloud_r_, new_size);
            AllocateLinear(&host_output_cloud_g_, &dev_output_cloud_g_, new_size);
            AllocateLinear(&host_output_cloud_b_, &dev_output_cloud_b_, new_size);

            AllocateLinear(&host_valid_points_, &dev_valid_points_, new_size);

            // Allocate GPU memory for image and create Texture Object
            CUDA_ERROR_CHECK(cudaMallocPitch(&dev_input_image_b_,
                                             &dev_pitch,
                                             input_image.step / input_image.channels(),
                                             input_image.size().height));
            CUDA_ERROR_CHECK(cudaMallocPitch(&dev_input_image_g_,
                                             &dev_pitch,
                                             input_image.step / input_image.channels(),
                                             input_image.size().height));
            CUDA_ERROR_CHECK(cudaMallocPitch(&dev_input_image_r_,
                                             &dev_pitch,
                                             input_image.step / input_image.channels(),
                                             input_image.size().height));

            cudaResourceDesc res_desc_b;
            memset(&res_desc_b, 0, sizeof(res_desc_b));
            res_desc_b.resType = cudaResourceTypePitch2D;
            res_desc_b.res.pitch2D.devPtr = dev_input_image_b_;
            res_desc_b.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();
            res_desc_b.res.pitch2D.width = input_image.size().width;  // width in element
            res_desc_b.res.pitch2D.height = input_image.size().height;  // height in element
            res_desc_b.res.pitch2D.pitchInBytes = dev_pitch;

            cudaTextureDesc tex_desc;
            memset(&tex_desc, 0, sizeof(tex_desc));
            tex_desc.addressMode[0] = cudaAddressModeBorder; // all out of range access will get zero value
            tex_desc.addressMode[1] = cudaAddressModeBorder; // all out of range access will get zero value
            tex_desc.readMode = cudaReadModeElementType;

            cudaResourceViewDesc view_desc;
            memset(&view_desc, 0, sizeof(view_desc));
            view_desc.format = cudaResViewFormatUnsignedChar1;
            view_desc.width = input_image.size().width;
            view_desc.height = input_image.size().height;
            view_desc.depth = input_image.elemSize1();

            CUDA_ERROR_CHECK(cudaCreateTextureObject(&gTextureObject_b, &res_desc_b, &tex_desc, &view_desc));

            cudaResourceDesc res_desc_g = res_desc_b;
            res_desc_g.res.pitch2D.devPtr = dev_input_image_g_;
            CUDA_ERROR_CHECK(cudaCreateTextureObject(&gTextureObject_g, &res_desc_g, &tex_desc, &view_desc));

            cudaResourceDesc res_desc_r = res_desc_b;
            res_desc_r.res.pitch2D.devPtr = dev_input_image_r_;
            CUDA_ERROR_CHECK(cudaCreateTextureObject(&gTextureObject_r, &res_desc_r, &tex_desc, &view_desc));

            // Update allocated information
            is_device_memory_ready_ = true;
            allocated_points_size_ = new_size;

            std::cerr << "Memory initialized" << std::endl;
        }

    // Upload input image to GPU
    std::vector<cv::Mat> channels;
    cv::split(input_image, channels);  // divide into BGR channel planes

    CUDA_ERROR_CHECK(cudaMemcpy2DAsync(dev_input_image_b_, // dst
                                       dev_pitch,          // dpitch
                                       channels[0].data,   // src
                                       channels[0].step,   // spitch
                                       channels[0].step,   // width (in byte)
                                       channels[0].size().height, // height
                                       cudaMemcpyHostToDevice  // kind
                                       ));
    CUDA_ERROR_CHECK(cudaMemcpy2DAsync(dev_input_image_g_, // dst
                                       dev_pitch,          // dpitch
                                       channels[1].data,   // src
                                       channels[1].step,   // spitch
                                       channels[1].step,   // width (in byte)
                                       channels[1].size().height, // height
                                       cudaMemcpyHostToDevice  // kind
                                       ));
    CUDA_ERROR_CHECK(cudaMemcpy2DAsync(dev_input_image_r_, // dst
                                       dev_pitch,          // dpitch
                                       channels[2].data,   // src
                                       channels[2].step,   // spitch
                                       channels[2].step,   // width (in byte)
                                       channels[2].size().height, // height
                                       cudaMemcpyHostToDevice  // kind
                                       ));

    // Upload input cloud to GPU
    for (size_t i = 0; i < input_cloud->points.size(); i++)
        {
            host_input_cloud_x_[i] = input_cloud->points[i].x;
            host_input_cloud_y_[i] = input_cloud->points[i].y;
            host_input_cloud_z_[i] = input_cloud->points[i].z;
        }

    CUDA_ERROR_CHECK(cudaMemcpyAsync(dev_input_cloud_x_,
                                     host_input_cloud_x_,
                                     input_cloud->points.size() * sizeof(float),
                                     cudaMemcpyHostToDevice
                                     ));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(dev_input_cloud_y_,
                                     host_input_cloud_y_,
                                     input_cloud->points.size() * sizeof(float),
                                     cudaMemcpyHostToDevice
                                     ));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(dev_input_cloud_z_,
                                     host_input_cloud_z_,
                                     input_cloud->points.size() * sizeof(float),
                                     cudaMemcpyHostToDevice
                                     ));

    // Call GPU process
    auto DivRoundUp = [](int value, int radix) {
        return (value + radix - 1) / radix;
    };

    dim3 block_dim(kNumThreadsPerBlock_, 1, 1);
    dim3 grid_dim(DivRoundUp(input_cloud->points.size(), block_dim.x), 1, 1);
    int2 image_size = make_int2(input_image.size().width, input_image.size().height);
    float2 focal = make_float2(fx, fy);
    float2 center = make_float2(cx, cy);
    HelperVector3 transform_origin(camera_lidar_tf.getOrigin().x(),
                                   camera_lidar_tf.getOrigin().y(),
                                   camera_lidar_tf.getOrigin().z());
    HelperMatrix3x3 transform_basis = {HelperVector3(camera_lidar_tf.getBasis().getRow(0).x(),
                                                     camera_lidar_tf.getBasis().getRow(0).y(),
                                                     camera_lidar_tf.getBasis().getRow(0).z()),
                                       HelperVector3(camera_lidar_tf.getBasis().getRow(1).x(),
                                                     camera_lidar_tf.getBasis().getRow(1).y(),
                                                     camera_lidar_tf.getBasis().getRow(1).z()),
                                       HelperVector3(camera_lidar_tf.getBasis().getRow(2).x(),
                                                     camera_lidar_tf.getBasis().getRow(2).y(),
                                                     camera_lidar_tf.getBasis().getRow(2).z())};

    FusionKernel<<<block_dim, grid_dim>>>(input_cloud->points.size(),
                                          image_size,
                                          focal,
                                          center,
                                          transform_origin,
                                          transform_basis,
                                          gTextureObject_b,
                                          gTextureObject_g,
                                          gTextureObject_r,
                                          dev_input_cloud_x_,
                                          dev_input_cloud_y_,
                                          dev_input_cloud_z_,
                                          dev_output_cloud_b_,
                                          dev_output_cloud_g_,
                                          dev_output_cloud_r_,
                                          dev_valid_points_
                                          );

    // Wait until GPU process has been done
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Download data from GPU
    CUDA_ERROR_CHECK(cudaMemcpyAsync(host_output_cloud_b_,
                                     dev_output_cloud_b_,
                                     input_cloud->points.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost
                                     ));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(host_output_cloud_g_,
                                     dev_output_cloud_g_,
                                     input_cloud->points.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost
                                     ));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(host_output_cloud_r_,
                                     dev_output_cloud_r_,
                                     input_cloud->points.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost
                                     ));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(host_valid_points_,
                                     dev_valid_points_,
                                     input_cloud->points.size() * sizeof(bool),
                                     cudaMemcpyDeviceToHost
                                     ));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Reshape data for pcl::PointCloud<pcl::PointXYZRGB> format
    pcl::PointXYZRGB colored_3d_point;
    for (size_t i = 0; i < input_cloud->points.size(); i++)
        {
            if (!host_valid_points_[i])
                {
                    continue;
                }
            colored_3d_point.x = input_cloud->points[i].x; // 3D coordinates are just copies from input cloud
            colored_3d_point.y = input_cloud->points[i].y; // 3D coordinates are just copies from input cloud
            colored_3d_point.z = input_cloud->points[i].z; // 3D coordinates are just copies from input cloud
            colored_3d_point.r = host_output_cloud_r_[i];
            colored_3d_point.g = host_output_cloud_g_[i];
            colored_3d_point.b = host_output_cloud_b_[i];
            output_cloud->points.push_back(colored_3d_point);
        }

}  // void PixelCloudFusionGPU::Fusion()
