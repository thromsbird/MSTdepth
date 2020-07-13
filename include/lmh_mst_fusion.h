/******************************************************************************************
\Author:	Li Minghao
\Function:	1. Compute the fused weight by 1 matrix,
			2. Compute the MVS depth by aggregation, and fuse the converted monocular depth in missing regions.
\Reference:	"Image-Based Rendering for Large-Scale Outdoor Scenes With Fusion of Monocular and Multi-View Stereo Depth," in IEEE Access, vol. 8, pp. 117551-117565, 2020
*******************************************************************************************/
#ifndef LMH_MST_FUSION_H
#define LMH_MST_FUSION_H
#include <opencv2/opencv.hpp>
#include "qx_tree_filter.h"

class lmh_mst_fusion
{
public:
	lmh_mst_fusion();
	~lmh_mst_fusion();
	void clean();
	int init(char *filename, double sigma_range = 0.1, double alpha_range = 0.5);
	int init(cv::Mat &color_img, double sigma_range = 0.1, double alpha_range = 0.5);

	int prepare_data(cv::Mat &mvs_depth_mat, cv::Mat &mono_depth_mat);

	int fuse_depth(cv::Mat &out_depth_mat);

private:
	void compute_per_pixel_depth();
	void lmh_detect_missing_area(cv::Mat &in_mat, double **weight_array, unsigned char **missing_array, int h, int w);

private:
	qx_tree_filter m_tf;
	int m_h, m_w, m_nr_plane;
	double m_alpha_on_mono;
	double m_table[256], m_sigma_range;

	double ***m_buf_d2;
	double **m_mvs_weight, **m_mvs_depth, **m_mono_depth, **m_fused_depth, **m_double_backup, **m_double_temp;

	unsigned char ***m_image, ****m_buf_u3;
	unsigned char **m_mask_missing, ***m_buf_u2;
};
#endif