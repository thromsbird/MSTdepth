#include "qx_basic.h"
#include "lmh_mst_fusion.h"
#include "lmh_basic.h"
lmh_mst_fusion::lmh_mst_fusion()
{
	m_buf_u2 = NULL;
	m_buf_u3 = NULL;
	m_buf_d2 = NULL;
}
lmh_mst_fusion::~lmh_mst_fusion()
{
	clean();
}
void lmh_mst_fusion::clean()
{
	qx_freeu_3(m_buf_u2);
	m_buf_u2 = NULL;
	qx_freeu_4(m_buf_u3);
	m_buf_u3 = NULL;
	qx_freed_3(m_buf_d2);
	m_buf_d2 = NULL;
}

int lmh_mst_fusion::init(char *filename, double sigma_range, double alpha_range)
{
	clean();
	int h = 0, w = 0;
	lmh_image_size(filename, h, w);
	m_h = h;
	m_w = w;
	m_sigma_range = sigma_range;
	m_alpha_on_mono = alpha_range;

	m_buf_d2 = qx_allocd_3(5, m_h, m_w);
	m_mvs_weight = m_buf_d2[0];
	m_mvs_depth = m_buf_d2[1];
	m_mono_depth = m_buf_d2[2];
	m_fused_depth = m_buf_d2[3];
	m_double_temp = m_buf_d2[4];

	for (int y = 0; y < m_h; y++)
		for (int x = 0; x < m_w; x++)
			m_mvs_weight[y][x] = 1.0;

	m_buf_u3 = qx_allocu_4(1, m_h, m_w, 3);
	m_image = m_buf_u3[0];
	lmh_loadimage(filename, m_image[0][0], m_h, m_w);

	m_buf_u2 = qx_allocu_3(1, m_h, m_w);
	m_mask_missing = m_buf_u2[0];
	for (int i = 0; i < 256; i++)
		m_table[i] = exp(-double(i) / (m_sigma_range * 255));

	m_tf.init(m_h, m_w, 3, m_sigma_range, 4);

	return 0;
}

int lmh_mst_fusion::init(cv::Mat &color_img, double sigma_range, double alpha_range)
{

	m_h = color_img.rows;
	m_w = color_img.cols;
	m_sigma_range = sigma_range;
	m_alpha_on_mono = alpha_range;

	m_buf_d2 = qx_allocd_3(6, m_h, m_w);
	m_mvs_weight = m_buf_d2[0];
	m_mvs_depth = m_buf_d2[1];
	m_mono_depth = m_buf_d2[2];
	m_fused_depth = m_buf_d2[3];
	m_double_backup = m_buf_d2[4];
	m_double_temp = m_buf_d2[5];

	for (int y = 0; y < m_h; y++)
		for (int x = 0; x < m_w; x++)
			m_mvs_weight[y][x] = 1.0;

	m_buf_u3 = qx_allocu_4(1, m_h, m_w, 3);
	m_image = m_buf_u3[0];

	lmh_mat2array_3u(color_img, m_image[0][0], m_h, m_w);

	m_buf_u2 = qx_allocu_3(1, m_h, m_w);
	m_mask_missing = m_buf_u2[0];
	for (int i = 0; i < 256; i++)
		m_table[i] = exp(-double(i) / (m_sigma_range * 255));

	m_tf.init(m_h, m_w, 3, m_sigma_range, 4);

	return 0;
}

int lmh_mst_fusion::prepare_data(cv::Mat &mvs_depth_mat, cv::Mat &mono_depth_mat)
{
	if (mvs_depth_mat.size != mono_depth_mat.size)
	{
		std::cout << "The size of mvs and mono depth map are different!" << std::endl;
		getchar();
		exit(0);
	}
	const int depthHeight = mvs_depth_mat.rows;
	const int depthWidth = mvs_depth_mat.cols;
	if (depthHeight != m_h || depthWidth != m_w)
	{
		std::cout << "The size of epth map and color image are different!" << std::endl;
		getchar();
		exit(0);
	}

	lmh_mat2array_1f(mvs_depth_mat, m_mvs_depth, m_h, m_w);
	lmh_mat2array_1f(mono_depth_mat, m_mono_depth, m_h, m_w);
	lmh_detect_missing_area(mvs_depth_mat, m_mvs_weight, m_mask_missing, m_h, m_w);

	m_tf.build_tree(m_image[0][0]);
	return 0;
}

int lmh_mst_fusion::fuse_depth(cv::Mat &out_depth_mat)
{
	int radius = 3;
	m_tf.filter(m_mvs_weight[0], m_double_temp[0], 1);
	image_copy(m_double_backup, m_mvs_depth, m_h, m_w); //back up m_mvs_depth
	m_tf.filter(m_mvs_depth[0], m_double_temp[0], 1);
	compute_per_pixel_depth();

	cv::Mat fused_depth(m_h, m_w, CV_32F);
	lmh_array2mat_1f(m_fused_depth, fused_depth, m_h, m_w);
	fused_depth.copyTo(out_depth_mat);
	return 0;
}

void lmh_mst_fusion::compute_per_pixel_depth()
{
	for (int i = 0; i < m_h; i++)
	{
		for (int j = 0; j < m_w; j++)
		{
			if (m_mask_missing[i][j])
			{
				double sum_weight = m_alpha_on_mono + m_mvs_weight[i][j];
				double mono_weight = m_alpha_on_mono / sum_weight;
				double aggregate_mvs_depth = m_mvs_depth[i][j] / sum_weight;
				//double temp = m_mono_depth[i][j];
				m_fused_depth[i][j] = mono_weight * m_mono_depth[i][j] + aggregate_mvs_depth;
				//m_fused_depth[i][j] = m_mono_depth[i][j];
			}
			else
			{
				m_fused_depth[i][j] = m_double_backup[i][j];
			}
		}
	}
	return;
}

void lmh_mst_fusion::lmh_detect_missing_area(cv::Mat &in_mat, double **weight_array, unsigned char **missing_array, int h, int w)
{
	for (int i = 0; i < h; i++)
	{
		float *data = in_mat.ptr<float>(i);
		for (int j = 0; j < w; j++)
		{
			if (data[j] > 0)
			{
				weight_array[i][j] = 1.0;
				missing_array[i][j] = 0;
			}
			else
			{
				weight_array[i][j] = 0.0;
				missing_array[i][j] = 255;
			}
		}
	}
	return;
}
