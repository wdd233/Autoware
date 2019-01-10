#ifndef GPU_EC_TEST_
#define GPU_EC_TEST_

#include <iostream>

class GPUECTest {
public:
	GPUECTest();

	// Sparse graph test
	static void sparseGraphTest();

	// Cluster number variation
	static void clusterNumVariationTest();

	// Load-imbalance test
	static void imbalanceTest();

	// Worst cases of matrix-based
	static void worstCaseMatrixBased();
	// Worst cases of edge-based
	static void worstCaseEdgeBased();
	// Worst cases of vertex-based
	static void worstCaseVertexBased();

	/* Test on various point cloud forms by changing
	 * number of points, number of disjoint components, and
	 * number of joint components in each disjoint component
	 */
	static void pointCloudVariationTest();

private:

	typedef struct {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
		int disjoint_num_;
		int joint_num_;
		int point_distance_;
	} SampleCloud;

	static void sparseGraphTest100();

	static void sparseGraphTest875();

	static void sparseGraphTest75();

	static void sparseGraphTest625();

	static void sparseGraphTest50();

	static void sparseGraphTest375();

	static void sparseGraphTest25();

	static void sparseGraphTest125();

	static void sparseGraphTest0();

	static void clusterNumVariationTest(int cluster_num);

	static void pointCloudVariationTest(int point_num, int disjoint_comp_num, int joint_comp_num);

	static std::string pointCloudVariationTest(int point_num, int disjoint_comp_num, int joint_comp_num, int point_distance);

	static std::string pointDistanceTest(SampleCloud base_cloud, int point_distance);

	static SampleCloud pointCloudGeneration(int point_num, int distjoint_comp_num, int joint_comp_num);

	static void lineTest(int point_num);

};

#ifndef timeDiff
#define timeDiff(start, end) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec))
#endif

#endif
