#include "lmh_basic.h"
#include "lmh_func.h"
#include "lmh_mst_fusion.h"
#include "colmapfunc.h"
#include <numeric>
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;

void fused(cv::Mat &mvs_depthf_mat, cv::Mat &mono_depthf_mat, cv::Mat &img_mat, double m_sigma, double m_alpha)
{
	lmh_mst_fusion m_nldf;					//non-local depth fusion class
	m_nldf.init(img_mat, m_sigma, m_alpha); //initialization
	m_nldf.prepare_data(mvs_depthf_mat, mono_depthf_mat);
	m_nldf.fuse_depth(mvs_depthf_mat);
}

int read_colmap_depth_bin(string bin_name, cv::Mat &depth_mat)
{
	ColmapDepthMap depthmap;
	depthmap.Read(bin_name);
	vector<float> depthvec = depthmap.Getdepthmap();
	const int depthWidth = depthmap.GetWidth();
	const int depthHeight = depthmap.GetHeight();
	cv::Mat(depthHeight, depthWidth, CV_32F, (float *)depthvec.data()).copyTo(depth_mat);
	return 1;
}
int write_colmap_depth_bin(std::string bin_name, cv::Mat &depth_mat)
{
	const int depthWidth = depth_mat.cols;
	const int depthHeight = depth_mat.rows;
	ColmapDepthMap depthmap(depthWidth, depthHeight, 1, 0.0, 0.0);
	vector<float> depthvec;
	depthvec.resize(depthWidth * depthHeight);
	for (int j = 0; j < depthHeight; j++)
	{
		float *depth_data = depth_mat.ptr<float>(j);
		for (int i = 0; i < depthWidth; i++)
		{
			depthvec[j * depthWidth + i] = depth_data[i];
		}
	}
	depthmap.Setdepthmap(depthvec);
	depthmap.Write(bin_name);
	return 1;
}

void read_depth_yml(string yml_name, cv::Mat &depth_mat)
{
	FileStorage fs(yml_name, FileStorage::READ);
	fs["depthmap"] >> depth_mat;
	if (depth_mat.empty())
		cout << "empty depthmap" << endl;
	return;
}

void read_gray_depth(string image_name, cv::Mat &mvs_depth_mat, cv::Mat &gray_depth_mat, const int min_validNum)
{
	string filename = image_name;
	Mat gray_depth = imread(filename, IMREAD_ANYDEPTH);
	if (gray_depth.empty())
	{
		cout << "The gray monocular depth load failed, name: " << filename << endl;
		return;
	}
	if (gray_depth.size() != mvs_depth_mat.size())
	{
		cv::resize(gray_depth, gray_depth, mvs_depth_mat.size(), cv::INTER_AREA);
	}
	int m_h = mvs_depth_mat.rows;			  //height
	int m_w = mvs_depth_mat.cols;			  //width
	gray_depth.convertTo(gray_depth, CV_16U); //convert to 16 uint
	Mat gray_mat;							  ////backup 16-int data
	gray_depth.copyTo(gray_mat);
	//gray_depth = gray_depth / 256; the '/' operator of OpenCV Lib is rounding off

	for (int j = 0; j < m_h; j++)
	{
		unsigned short *gray_data = gray_depth.ptr<unsigned short>(j);
		for (int i = 0; i < m_w; i++)
		{
			gray_data[i] = gray_data[i] / 256;
		}
	}

	const int Min_validNum = min_validNum;
	int minGray = 255, maxGray = 0;
	int minGrayInvalid = 255, maxGrayInvalid = 0;
	for (int j = 0; j < m_h; j++)
	{
		unsigned short *gray_data = gray_depth.ptr<unsigned short>(j);
		float *depth_data = mvs_depth_mat.ptr<float>(j);
		for (int i = 0; i < m_w; i++)
		{
			if (depth_data[i])
			{
				if (gray_data[i] > maxGray)
				{
					maxGray = gray_data[i];
				}
				if (gray_data[i] < minGray)
				{
					minGray = gray_data[i];
				}
			}
		}
	}
	vector<vector<float>> depthfvec;
	depthfvec.resize(maxGray - minGray + 1);
	for (int j = 0; j < m_h; j++)
	{
		unsigned short *gray_data = gray_depth.ptr<unsigned short>(j);
		float *depth_data = mvs_depth_mat.ptr<float>(j);
		for (int i = 0; i < m_w; i++)
		{
			if (depth_data[i])
			{
				int level = gray_data[i] - minGray;
				if (level >= 0)
				{
					depthfvec[level].push_back(depth_data[i]);
				}
			}
		}
	}

	for (int i = 0; i != (int)depthfvec.size(); i++)
	{
		if ((int)depthfvec[i].size() >= Min_validNum)
		{
			minGrayInvalid = minGray + i;
			break;
		}
	}

	for (int i = (int)depthfvec.size() - 1; i >= 0; i--)
	{
		if ((int)depthfvec[i].size() >= Min_validNum)
		{
			maxGrayInvalid = minGray + i;
			break;
		}
	}
	if (maxGrayInvalid <= minGrayInvalid)
	{
		cout << "no enough depth samples !!!" << endl;
		return;
	}

	vector<float> LevelDepth(256, 0.0); //get each level depth
	for (int i = minGrayInvalid - minGray; i != (int)depthfvec.size(); i++)
	{
		int grayNum = (int)depthfvec[i].size();
		if (grayNum < Min_validNum)
		{
			LevelDepth[minGray + i] = LevelDepth[i + minGray - 1];
		}
		else
		{
			LevelDepth[minGray + i] = Percentile(depthfvec[i], 50.0);
		}
	}

	const int stepNum = 20;
	float minNextStepDepth = (minGrayInvalid + stepNum < maxGrayInvalid) ? LevelDepth[minGrayInvalid + stepNum] : LevelDepth[maxGrayInvalid];
	minNextStepDepth = abs((LevelDepth[minGrayInvalid] - minNextStepDepth) / (256.0 * (float)stepNum));
	float maxNextStepDepth = (maxGrayInvalid - stepNum > minGrayInvalid) ? LevelDepth[maxGrayInvalid - stepNum] : LevelDepth[minGrayInvalid];
	maxNextStepDepth = -abs((LevelDepth[maxGrayInvalid] - maxNextStepDepth) / (256.0 * (float)stepNum));
	float lutData[65536] = {0.0};
	for (int i = minGrayInvalid; i <= maxGrayInvalid; i++)
		lutData[i * 256] = LevelDepth[i];
	for (int i = minGrayInvalid - 1; i >= 0; i--)
	{
		for (int j = 0; j <= 256; j++)
		{
			lutData[i * 256 + j] = lutData[(i + 1) * 256] + (float)(256 - j) * minNextStepDepth;
		}
	}
	for (int i = minGrayInvalid; i < maxGrayInvalid; i++)
	{
		float stepDepth = (LevelDepth[i + 1] - LevelDepth[i]) / 256.0;
		for (int j = 0; j < 256; j++)
		{
			lutData[i * 256 + j] = LevelDepth[i] + (float)i * stepDepth;
		}
	}
	for (int i = maxGrayInvalid; i < 255; i++)
	{
		for (int j = 1; j <= 256; j++)
		{
			lutData[i * 256 + j] = lutData[i * 256] + (float)j * maxNextStepDepth;
			if (lutData[i * 256 - j] < 0.1)
				lutData[i * 256 - j] = 0.1;
		}
	}
	for (int j = 1; j < 256; j++)
	{
		lutData[255 * 256 + j] = lutData[255 * 256] + (float)j * maxNextStepDepth;
		if (lutData[255 * 256 + j] < 0.1)
			lutData[255 * 256 + j] = 0.1;
	}
	// ofstream outFile;
	// outFile.open(image_name + "_data16.csv", ios::out); //
	// outFile << "Level" << ','  << "Mapping depth" << endl;
	// for (int i = 0; i < 65536; i++)
	// 	outFile << i << ',' <<',' << lutData[i] << endl;
	// outFile.close();

	Mat temp_depth_mat(m_h, m_w, CV_32F, 0.0);
	for (int j = 0; j < m_h; j++)
	{
		unsigned short *gray_data = gray_mat.ptr<unsigned short>(j);
		float *depth_data = temp_depth_mat.ptr<float>(j);
		for (int i = 0; i < m_w; i++)
		{
			depth_data[i] = lutData[gray_data[i]];
		}
	}
	temp_depth_mat.copyTo(gray_depth_mat);
	return;
}

void modify_sky_mask(cv::Mat &mask)
{
	if (mask.empty())
	{
		cout << "load sky seg mask failed ... " << endl;
		return;
	}
	Mat Kernel = cv::getStructuringElement(MORPH_RECT, Size(30, 30));
	cv::dilate(mask, mask, Kernel);
	cv::erode(mask, mask, Kernel);
	cv::erode(mask, mask, Kernel);

	unsigned char lutData[256] = {0};
	for (int i = 0; i < 256; i++)
	{
		if (i > 100)
			lutData[i] = 255;
		else
			lutData[i] = 0;
	}
	Mat lut(1, 256, CV_8U, lutData);
	LUT(mask, lut, mask);
	return;
}

void mod_depth(cv::Mat &sky_mask, cv::Mat &depth_mat, const float sky_max_depth)
{
	float sky_depth_float = sky_max_depth;
	if (sky_mask.size() != depth_mat.size())
	{
		cout << "Different size of MVS and sky mask image !" << endl;
		return;
	}
	int m_h = depth_mat.rows; //height
	int m_w = depth_mat.cols; //width
	for (int j = 0; j < m_h; j++)
	{
		uchar *sky_data = sky_mask.ptr<uchar>(j);
		float *depth_data = depth_mat.ptr<float>(j);
		for (int i = 0; i < m_w; i++)
		{
			if ((int)sky_data[i] == 255)
				depth_data[i] = sky_depth_float;
		}
	}
	return;
}

void lmh_median_filter_depth(cv::Mat &depth_mat, const int size, const float low_limit, const float up_limit)
{
	cv::Mat temp_depth_mat = depth_mat.clone();
	cv::medianBlur(temp_depth_mat, temp_depth_mat, size); //median filter
	for (int i = 0; i < (depth_mat.rows); i++)
	{
		float *mvs_data = depth_mat.ptr<float>(i);
		float *median_data = temp_depth_mat.ptr<float>(i);
		for (int j = 1; j < (depth_mat.cols); j++)
		{
			if (mvs_data[j] < low_limit * median_data[j] || mvs_data[j] > up_limit * median_data[j])
				mvs_data[j] = 0.0;
		}
	}
	return;
}
void lmh_weighted_median_filter_depth(cv::Mat img_mat, cv::Mat &depth_mat, const int small_size, const int large_size, const float low_limit, const float up_limit)
{
	const int m_h = depth_mat.rows;
	const int m_w = depth_mat.cols;
	cv::Mat temp_depth_mat = depth_mat.clone();
	cv::medianBlur(temp_depth_mat, temp_depth_mat, small_size);
	for (int i = 0; i < (depth_mat.rows); i++)
	{
		float *mvs_data = depth_mat.ptr<float>(i);
		float *median_data = temp_depth_mat.ptr<float>(i);
		for (int j = 1; j < (depth_mat.cols); j++)
		{
			if (median_data[j] < low_limit * mvs_data[j] || median_data[j] > up_limit * mvs_data[j])
				mvs_data[j] = 0.0;
		}
	}
	//lmh_depthf_mat2image(temp_depth_mat, "COLMAP_median.png");
	temp_depth_mat = depth_mat.clone();
	cv::Mat weighted_depth_mat;
	cv::ximgproc::weightedMedianFilter(img_mat, temp_depth_mat, weighted_depth_mat, large_size, 0.033);
	//lmh_depthf_mat2image(weighted_depth_mat, "COLMAP_wedightedMedian.png");
	for (int i = 0; i < m_h; i++)
	{
		float *mvs_data = depth_mat.ptr<float>(i);
		float *median_data = weighted_depth_mat.ptr<float>(i);
		for (int j = 1; j < m_w; j++)
		{
			if (median_data[j] < low_limit * mvs_data[j] || median_data[j] > up_limit * mvs_data[j])
				mvs_data[j] = 0.0;
		}
	}
	return;
}

void lmh_uncertain_filter_depth(cv::Mat &geo_depth_mat, cv::Mat &pho_depth_mat, cv::Mat img_mat, const float differ)
{
	const int m_h = geo_depth_mat.rows;
	const int m_w = geo_depth_mat.cols;
	if (pho_depth_mat.size() != geo_depth_mat.size())
	{
		cout << "Different size of Photo and Geo depthmap !" << endl;
		return;
	}
	cv::Mat uncertain_mask(m_h, m_w, CV_8UC1);
	uncertain_mask.setTo(Scalar(0));
	for (int i = 0; i < m_h; i++)
	{
		float *geo_mvs_data = geo_depth_mat.ptr<float>(i);
		float *pho_mvs_data = pho_depth_mat.ptr<float>(i);
		uchar *mask_data = uncertain_mask.ptr<uchar>(i);
		for (int j = 1; j < m_w; j++)
		{
			if (abs(geo_mvs_data[j] - pho_mvs_data[j]) > differ * pho_mvs_data[j])
				mask_data[j] = 255;
		}
	}
	imwrite("uncertain_mask.png", uncertain_mask);
	cv::ximgproc::guidedFilter(img_mat, uncertain_mask, uncertain_mask, 10, 100);
	imwrite("guide_uncertain_mask.png", uncertain_mask);
	for (int i = 0; i < m_h; i++)
	{
		float *mvs_data = geo_depth_mat.ptr<float>(i);
		uchar *mask_data = uncertain_mask.ptr<uchar>(i);
		for (int j = 1; j < m_w; j++)
		{
			if (mask_data[j] > 255 * 0.40)
				mvs_data[j] = 0.0;
		}
	}
	return;
}

void depthmaps_computeNormal(cv::Mat inputdepthmap, cv::Mat &outputnormmap, cv::Mat R1, float cx, float cy, float focallengthx, float focallengthy)
{
	cv::Mat depthmap_ref(inputdepthmap.rows, inputdepthmap.cols, CV_32F);
	inputdepthmap.copyTo(depthmap_ref);

	//::GaussianBlur(depthmap_ref, depthmap_ref, Size(7, 7), 5);

	cv::Mat depthmap_normalaux = cv::Mat::zeros(inputdepthmap.rows, inputdepthmap.cols, CV_32FC3);
	cv::Mat depthmap_normalcount = cv::Mat::zeros(inputdepthmap.rows, inputdepthmap.cols, CV_32F);

	cv::Mat onepointmat(3, 1, CV_32F);
	for (int x0 = 0; x0 < depthmap_ref.cols; x0++)
		for (int y0 = 0; y0 < depthmap_ref.rows; y0++)
		{
			if (x0 < 1 || x0 >= depthmap_ref.cols - 1 || y0 < 1 || y0 >= depthmap_ref.rows - 1)
			{
				continue;
			}

			float depth_ref = depthmap_ref.at<float>(y0, x0);
			float depth_left = depthmap_ref.at<float>(y0, x0 - 1);
			float depth_right = depthmap_ref.at<float>(y0, x0 + 1);
			float depth_up = depthmap_ref.at<float>(y0 - 1, x0);
			float depth_down = depthmap_ref.at<float>(y0 + 1, x0);

			if (depth_ref > 0 && depth_left > 0 && depth_up > 0)
			{
				cv::Vec3f t(depth_up * (x0 - cx) / focallengthx, depth_up * (y0 - 1 - cy) / focallengthy, depth_up);	   //5000
				cv::Vec3f l(depth_left * (x0 - 1 - cx) / focallengthx, depth_left * (y0 - cy) / focallengthy, depth_left); //5000
				cv::Vec3f c(depth_ref * (x0 - cx) / focallengthx, depth_ref * (y0 - cy) / focallengthy, depth_ref);		   //5000]
				cv::Vec3f d = (l - c).cross(t - c);
				d = normalize(d);

				cv::Vec3f n;
				float dzdy = (focallengthx / depth_ref) * (depth_right * 1 - depth_left * 1) / 2.0; //dzdx (1050 / depth_ref)*
				float dzdx = (focallengthx / depth_ref) * (depth_down * 1 - depth_up * 1) / 2.0;	//dzdy (1050 / depth_ref)*

				float nx = -dzdx; // n[0];
				float ny = -dzdy; //n[1];
				float nz = 1.0;	  //n[2];

				float magnitude = sqrt(nx * nx + ny * ny + nz * nz);
				//printf("%f\n", magnitude);
				//printf("%f\n", dzdy);
				nx = nx / magnitude;
				ny = ny / magnitude;
				nz = nz / magnitude;
				cv::Vec3f ndd(-dzdx, -dzdy, 1.0);
				ndd = normalize(ndd);

				cv::Mat onenormalmat(3, 1, CV_32F);
				onenormalmat.at<float>(0, 0) = d(0); //
				onenormalmat.at<float>(1, 0) = d(1); //
				onenormalmat.at<float>(2, 0) = d(2); //

				n[0] = onenormalmat.at<float>(0, 0);
				n[1] = onenormalmat.at<float>(1, 0);
				n[2] = onenormalmat.at<float>(2, 0);

				depthmap_normalaux.at<cv::Vec3f>(y0, x0) = n;
				depthmap_normalcount.at<float>(y0, x0) = 1;
			}
		}

	for (int interation = 0; interation < 10; interation++)
	{
		for (int x0 = 0; x0 < depthmap_ref.cols; x0++)
			for (int y0 = 0; y0 < depthmap_ref.rows; y0++)
			{
				float depth_ref = depthmap_ref.at<float>(y0, x0);
				/*if (depth_ref <= 0){
					continue;
				}*/
				int lefxindex = max(x0 - 1, 0);
				int rightindex = min(x0 + 1, depthmap_ref.cols - 1);
				int upindex = max(y0 - 1, 0);
				int downindex = min(y0 + 1, depthmap_ref.rows - 1);

				cv::Vec3f normal;
				int normalcount = depthmap_normalcount.at<float>(y0, x0);
				float nx = 0;
				float ny = 0;
				float nz = 0;
				if (normalcount == 0)
				{

					int left = depthmap_normalcount.at<float>(y0, lefxindex);
					int right = depthmap_normalcount.at<float>(y0, rightindex);
					int up = depthmap_normalcount.at<float>(upindex, x0);
					int down = depthmap_normalcount.at<float>(downindex, x0);
					int count = 0;

					if (left == 1)
					{
						normal = depthmap_normalaux.at<cv::Vec3f>(y0, lefxindex);
						nx = nx + normal[0];
						ny = ny + normal[1];
						nz = nz + normal[2];
						count++;
					}
					if (right == 1)
					{
						normal = depthmap_normalaux.at<cv::Vec3f>(y0, rightindex);
						nx = nx + normal[0];
						ny = ny + normal[1];
						nz = nz + normal[2];
						count++;
					}
					if (up == 1)
					{
						normal = depthmap_normalaux.at<cv::Vec3f>(upindex, x0);
						nx = nx + normal[0];
						ny = ny + normal[1];
						nz = nz + normal[2];
						count++;
					}
					if (down == 1)
					{
						normal = depthmap_normalaux.at<cv::Vec3f>(downindex, x0);
						nx = nx + normal[0];
						ny = ny + normal[1];
						nz = nz + normal[2];
						count++;
					}

					if (count > 0)
					{
						nx = nx / count;
						ny = ny / count;
						nz = nz / count;
						float magnitude = sqrt(nx * nx + ny * ny + nz * nz);
						nx = nx / magnitude;
						ny = ny / magnitude;
						nz = nz / magnitude;
						normal[0] = nx;
						normal[1] = ny;
						normal[2] = nz;
						depthmap_normalaux.at<cv::Vec3f>(y0, x0) = normal;
						depthmap_normalcount.at<float>(y0, x0) = 1;
					}

				} //end if
			}
	}

	for (int x0 = 0; x0 < depthmap_ref.cols; x0 = x0 + 1)
		for (int y0 = 0; y0 < depthmap_ref.rows; y0 = y0 + 1)
		{
			float depth_ref = depthmap_ref.at<float>(y0, x0);
			int count = depthmap_normalcount.at<float>(y0, x0);
			if (depth_ref <= 0 || count == 0)
			{
				continue;
			}
			cv::Vec3f normal;
			normal = depthmap_normalaux.at<cv::Vec3f>(y0, x0);
			cv::Mat onenormalmat(3, 1, CV_32F);
			onenormalmat.at<float>(0, 0) = -normal[0];
			onenormalmat.at<float>(1, 0) = -normal[1];
			onenormalmat.at<float>(2, 0) = -normal[2];

			cv::Vec3f normal2;
			normal2[0] = onenormalmat.at<float>(0, 0);
			normal2[1] = onenormalmat.at<float>(1, 0);
			normal2[2] = onenormalmat.at<float>(2, 0);

			depthmap_normalaux.at<cv::Vec3f>(y0, x0) = normal2;
		}

	depthmap_normalaux.copyTo(outputnormmap);
}
