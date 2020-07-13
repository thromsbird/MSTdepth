/*******************************************************
\Author:	Li Minghao
\Function:	Basic functions (using OpenCV).
\Reference:	"Image-Based Rendering for Large-Scale Outdoor Scenes With Fusion of Monocular and Multi-View Stereo Depth," in IEEE Access, vol. 8, pp. 117551-117565, 2020
********************************************************/
#ifndef LMH_BASIC_OPENCV_H
#define LMH_BASIC_OPENCV_H
#include <chrono>
#include <opencv2/opencv.hpp>
/*time class*/
class lmh_timer
{
public:
	void start();
	long long get_us();
	long long get_ms();

	void show_ms(std::string str = "");
	std::chrono::system_clock::time_point lmh_time_start;
	std::chrono::system_clock::time_point lmh_time_end;
};

/*memory function*/
inline void image_copy(double **out, double **in, int h, int w) { memcpy(out[0], in[0], sizeof(double) * h * w); }

/*.txt read/write function*/
int output_txt_d1(std::string filename, double **data, int h, int w);
int output_txt_u1(std::string filename, unsigned char **data, int h, int w);

/*template math function*/
template <typename T>
T Percentile(const std::vector<T> &elems, const double p)
{
	//CHECK(!elems.empty());
	const int idx = static_cast<int>(std::round(p / 100 * (elems.size() - 1)));
	const size_t percentile_idx =
		std::max(0, std::min(static_cast<int>(elems.size() - 1), idx));

	std::vector<T> ordered_elems = elems;
	std::nth_element(ordered_elems.begin(),
					 ordered_elems.begin() + percentile_idx, ordered_elems.end());

	return ordered_elems.at(percentile_idx);
}

/*To get the size of the image*/
void lmh_image_size(char *file_name, int &h, int &w);
/*load image*/
int lmh_loadimage(char *file_name, unsigned char *image, int h, int w, int *nr_channel = NULL);
/*save image*/
void lmh_saveimage(char *file_name, unsigned char *image, int h, int w, int channel);
void lmh_saveimage(const char *file_name, unsigned char *image, int h, int w, int channel);
bool lmh_depthf_mat2image(cv::Mat depthf_mat, std::string savepath);

/*change data mode */
void lmh_mat2array_3u(cv::Mat &in_mat, unsigned char *image, int h, int w);
void lmh_mat2array_1f(cv::Mat &in_mat, double **out_array, int h, int w);
void lmh_array2mat_1f(double **in_array, cv::Mat &out_mat, int h, int w);

/**
 * Loads a PFM image stored in little endian and returns the image as an OpenCV Mat.
 * @brief loadPFM
 * @param filePath
 * @return
 */
cv::Mat loadPFM(const std::string filePath);
/**
 * Saves the image as a PFM file.
 * @brief savePFM
 * @param image
 * @param filePath
 * @return
 */
bool savePFM(const cv::Mat image, const std::string filePath);

#endif