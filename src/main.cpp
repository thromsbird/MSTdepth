/*
 * MSTdepth - Minimum spanning tree (MST)-based nonlocal filter to fuse MVS depth and mono depth.
 * 
 * Copyright (C) 2020  Beijing University of Posts and Telecommunications
 * 
 * Reference: "Image-Based Rendering for Large-Scale Outdoor Scenes With Fusion of Monocular and Multi-View Stereo Depth"
 *            IEEE Access, vol. 8, pp. 117551-117565, 2020.
 * Authors: Liu Shaohua, Li Minghao, Zhang Xiaona, Liu Shuang, Li Zhaoxin, Liu Jing, and Mao Tianlu
 * 
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contact:
 *  Mao Tianlu
 *  ltm@ict.ac.cn
 *  Institute of Computing Technology Chinese Academy of Sciences, Beijing, China
 *  
 *  Liu Shaohua
 *  liushaohua@bupt.edu.cn
 *  Beijing University of Posts and Telecommunications, Beijing, China  
 */

#include "lmh_basic.h"
#include "lmh_mst_fusion.h"
#include "lmh_func.h"

void example(char *filename_mvs_bin, char *filename_mono_png, char *filename_color_jpg, double set_sigma = 0.06, double set_alpha = 0.00001)
{

	cv::Mat mvs_depth_mat;									// MVS depth
	cv::Mat mono_depth_mat;									// Monocular depth
	cv::Mat color_img_mat = cv::imread(filename_color_jpg); // color image

	read_colmap_depth_bin(filename_mvs_bin, mvs_depth_mat);
	lmh_weighted_median_filter_depth(color_img_mat, mvs_depth_mat);
	read_gray_depth(filename_mono_png, mvs_depth_mat, mono_depth_mat, 100);

	fused(mvs_depth_mat, mono_depth_mat, color_img_mat, set_sigma, set_alpha);

	string result_name_png(filename_color_jpg);
	result_name_png += ".fused_" + to_string(set_sigma) + "_" + to_string(set_alpha) + ".png";
	lmh_depthf_mat2image(mvs_depth_mat, result_name_png);
}

int main(int argc, char *argv[])
{
	if (argc != 4)
	{
		printf("Usage:\n");
		printf("*.exe: filename_mvs_bin filename_mono_png filename_color_jpg \n");
		return (-1);
	}

	char *filename_mvs_bin = argv[1];
	char *filename_gray_image = argv[2];
	char *filename_color_image = argv[3];

	double set_sigma = 0.06;
	double set_alpha = 0.00001;

	example(filename_mvs_bin, filename_gray_image, filename_color_image, set_sigma, set_alpha);
	return (0);
}
