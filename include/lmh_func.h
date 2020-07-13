/*******************************************************
\Author:	Li Minghao
\Function:	advanded functions (using OpenCV).
\Reference:	"Image-Based Rendering for Large-Scale Outdoor Scenes With Fusion of Monocular and Multi-View Stereo Depth," in IEEE Access, vol. 8, pp. 117551-117565, 2020
********************************************************/
#ifndef LMH_FUNC_H
#define LMH_FUNC_H

#include <opencv2/opencv.hpp>

/*refined depth function*/
void fused(cv::Mat &mvs_depthf_mat, cv::Mat &gray_depthf_mat, cv::Mat &img_mat, double m_sigma, double m_alpha);

/* read/write function*/
int read_colmap_depth_bin(std::string bin_name, cv::Mat &depth_mat);
int write_colmap_depth_bin(std::string bin_name, cv::Mat &depth_mat);
void read_depth_yml(std::string yml_name, cv::Mat &depth_mat);
/* operate function */
void read_gray_depth(std::string image_name, cv::Mat &mvs_depth_mat, cv::Mat &gray_depth_mat, const int min_validNum);
void modify_sky_mask(cv::Mat &sky_mask);
void mod_depth(cv::Mat &sky_mask, cv::Mat &depth_mat, const float sky_max_depth = 10.0);
void lmh_median_filter_depth(cv::Mat &depth_mat, const int size = 5, const float low_limit = 0.9, const float up_limit = 1.10);
void lmh_weighted_median_filter_depth(cv::Mat img_mat, cv::Mat &depth_mat, const int small_size = 5, const int large_size = 31, const float low_limit = 0.9, const float up_limit = 1.10);
void lmh_uncertain_filter_depth(cv::Mat &geo_depth_mat, cv::Mat &pho_depth_mat, cv::Mat img_mat, const float differ = 0.05);
void depthmaps_computeNormal(cv::Mat inputdepthmap, cv::Mat &outputnormmap, cv::Mat R1, float cx, float cy, float focallengthx, float focallengthy);

#endif
