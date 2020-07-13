#include "lmh_basic.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace cv;
using namespace std;

#define LEN_MAX 256

void lmh_timer::start()
{
	lmh_time_start = std::chrono::system_clock::now();
}
long long lmh_timer::get_us()
{
	lmh_time_end = std::chrono::system_clock::now();
	auto diff = lmh_time_end - lmh_time_start;
	long long count_us = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
	return count_us;
}
long long lmh_timer::get_ms()
{
	lmh_time_end = std::chrono::system_clock::now();
	auto diff = lmh_time_end - lmh_time_start;
	long long count_ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
	return count_ms;
}
void lmh_timer::show_ms(std::string str)
{
	long long time_ms = get_ms();
	cout << "Running time of " << str << " : " << time_ms << " ms." << endl;
}

float Interpolate(const float val, const float y0, const float x0,
				  const float y1, const float x1)
{
	return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}

float Base(const float val)
{
	if (val <= 0.125f)
	{
		return 0.0f;
	}
	else if (val <= 0.375f)
	{
		return Interpolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
	}
	else if (val <= 0.625f)
	{
		return 1.0f;
	}
	else if (val <= 0.87f)
	{
		return Interpolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
	}
	else
	{
		return 0.0f;
	}
}

int output_txt_d1(string filename, double **data, int h, int w)
{
	ofstream fout;
	fout.open(filename, ofstream::out);
	fout << "data: " << endl;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			fout << data[i][j] << "\t";
		}
		fout << "\n";
	}
	fout << endl;
	fout.close();
	return 1;
}

int output_txt_u1(string filename, unsigned char **data, int h, int w)
{
	ofstream fout;
	fout.open(filename, ofstream::out);
	fout << "data: " << endl;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			fout << (int)data[i][j] << "\t";
		}
		fout << "\n";
	}
	fout << endl;
	fout.close();
	return 1;
}

void lmh_image_size(char *file_name, int &h, int &w)
{
	string image_name(file_name);
	Mat image = imread(image_name, IMREAD_UNCHANGED);

	if (image.empty())
	{
		cout << "Please check input file_name: " << image_name << endl;
		getchar();
		exit(0);
	}
	if (image.channels() != 3)
	{
		cout << "Please check input file_name is a color image: " << image_name << endl;
		getchar();
		exit(0);
	}
	h = image.rows; // height
	w = image.cols; // width
	return;
}

int lmh_loadimage(char *filename, unsigned char *image, int h, int w, int *nr_channel)
{
	string image_name(filename);
	Mat imageMat = imread(image_name, IMREAD_UNCHANGED);

	if (imageMat.empty())
	{
		cout << "Please check input file_name: " << image_name << endl;
		getchar();
		exit(0);
	}
	if (imageMat.channels() != 3)
	{
		cout << "Please check input file_name is a color image: " << image_name << endl;
		getchar();
		exit(0);
	}
	memset(image, 0, sizeof(unsigned char) * h * w * 3);
	for (int i = 0; i < h; i++)
	{
		uchar *data = imageMat.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
		{
			int iNo = (i * w + j) * 3;
			int dNo = j * 3;
			image[iNo] = data[dNo + 2];
			image[iNo + 1] = data[dNo + 1];
			image[iNo + 2] = data[dNo];
		}
	}
	return 0;
}

void lmh_saveimage(char *filename, unsigned char *image, int h, int w, int channel)
{
	string savename(filename);
	savename = savename + ".png";
	if (channel == 1)
	{
		Mat grayImage(h, w, CV_8UC1);
		for (int i = 0; i < h; i++)
		{
			uchar *data = grayImage.ptr<uchar>(i);
			for (int j = 0; j < w; j++)
			{
				data[j] = image[i * w + j];
			}
		}
		imwrite(savename, grayImage);
	}
	if (channel == 3)
	{
		Mat colorImage(h, w, CV_8UC3);
		for (int i = 0; i < h; i++)
		{
			uchar *data = colorImage.ptr<uchar>(i);
			for (int j = 0; j < w; j++)
			{
				int iNo = (i * w + j) * 3;
				int dNo = j * 3;
				data[dNo + 2] = image[iNo];
				data[dNo + 1] = image[iNo + 1];
				data[dNo] = image[iNo + 2];
			}
		}
		imwrite(savename, colorImage);
	}
}

void lmh_saveimage(const char *filename, unsigned char *image, int h, int w, int channel)
{
	string savename(filename);
	savename = savename + ".png";
	if (channel == 1)
	{
		Mat grayImage(h, w, CV_8UC1);
		for (int i = 0; i < h; i++)
		{
			uchar *data = grayImage.ptr<uchar>(i);
			for (int j = 0; j < w; j++)
			{
				data[j] = image[i * w + j];
			}
		}
		imwrite(savename, grayImage);
	}
	if (channel == 3)
	{
		Mat colorImage(h, w, CV_8UC3);
		for (int i = 0; i < h; i++)
		{
			uchar *data = colorImage.ptr<uchar>(i);
			for (int j = 0; j < w; j++)
			{
				int iNo = (i * w + j) * 3;
				int dNo = j * 3;
				data[dNo + 2] = image[iNo];
				data[dNo + 1] = image[iNo + 1];
				data[dNo] = image[iNo + 2];
			}
		}
		imwrite(savename, colorImage);
	}
}

bool lmh_depthf_mat2image(cv::Mat depthf_mat, std::string savepath)
{
	double min_percentile = 0.01;
	double max_percentile = 99.1;
	const int depthWidth = depthf_mat.cols;
	const int depthHeight = depthf_mat.rows;
	int nl = depthf_mat.rows; //height
	int nc = depthf_mat.cols; //width

	std::vector<float> depthvec;
	depthvec.reserve(nl * nc);
	for (int j = 0; j < nl; j++)
	{
		float *depth_data = depthf_mat.ptr<float>(j);
		for (int i = 0; i < nc; i++)
		{
			depthvec.push_back(depth_data[i]);
		}
	}
	//cout << "valid_depths " << depthvec.size();
	std::vector<float> valid_depths;
	valid_depths.reserve(depthvec.size());
	for (const float depth : depthvec)
	{
		if (depth > 0)
		{
			valid_depths.push_back(depth);
		}
	}
	const float robust_depth_min = Percentile(valid_depths, min_percentile);
	const float robust_depth_max = Percentile(valid_depths, max_percentile);
	const float robust_depth_range = robust_depth_max - robust_depth_min;
	Mat rgb_pic(depthHeight, depthWidth, CV_8UC3);
	for (int y = 0; y < depthHeight; ++y)
	{
		for (int x = 0; x < depthWidth; ++x)
		{
			const float depth = depthvec.at(y * depthWidth + x);
			if (depth > 0)
			{
				const float robust_depth =
					std::max(robust_depth_min, std::min(robust_depth_max, depth));
				const float gray = (robust_depth - robust_depth_min) / robust_depth_range;
				float Red = Base(gray - 0.25f);
				float Green = Base(gray);
				float Blue = Base(gray + 0.25f);
				unsigned char uRed = (unsigned char)(255 * Red);
				unsigned char uGreen = (unsigned char)(255 * Green);
				unsigned char uBlue = (unsigned char)(255 * Blue);
				Vec3b color = {uRed, uGreen, uBlue};
				rgb_pic.at<Vec3b>(Point((int)x, (int)y)) = color;
			}
			else
			{
				Vec3b color{255, 255, 255};
				rgb_pic.at<Vec3b>(Point((int)x, (int)y)) = color;
			}
		}
	}
	imwrite(savepath, rgb_pic);
	//cout << "SUCCESS: Save depth image !" << endl;
	return 1;
}

void lmh_mat2array_3u(cv::Mat &in_mat, unsigned char *image, int h, int w)
{
	if (in_mat.channels() != 3)
	{
		cout << "Please check input image is a color image: " << endl;
		getchar();
		exit(0);
	}
	memset(image, 0, sizeof(unsigned char) * h * w * 3);
	for (int i = 0; i < h; i++)
	{
		uchar *data = in_mat.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
		{
			int iNo = (i * w + j) * 3;
			int dNo = j * 3;
			image[iNo] = data[dNo + 2];
			image[iNo + 1] = data[dNo + 1];
			image[iNo + 2] = data[dNo];
		}
	}
	return;
}

void lmh_mat2array_1f(cv::Mat &in_mat, double **out_array, int h, int w)
{
	for (int i = 0; i < h; i++)
	{
		float *data = in_mat.ptr<float>(i);
		for (int j = 0; j < w; j++)
		{
			out_array[i][j] = data[j];
		}
	}
	return;
}

void lmh_array2mat_1f(double **in_array, cv::Mat &out_mat, int h, int w)
{
	for (int i = 0; i < h; i++)
	{
		float *data = out_mat.ptr<float>(i);
		for (int j = 0; j < w; j++)
		{
			data[j] = (float)in_array[i][j];
		}
	}
	return;
}

/**
 * Loads a PFM image stored in little endian and returns the image as an OpenCV Mat.
 * @brief loadPFM
 * @param filePath
 * @return
 */
Mat loadPFM(const string filePath)
{
	//Open binary file
	ifstream file(filePath.c_str(), ios::in | ios::binary);
	Mat imagePFM;

	//If file correctly openened
	if (file)
	{
		//Read the type of file plus the 0x0a UNIX return character at the end
		char type[3];
		file.read(type, 3 * sizeof(char));

		//Read the width and height
		unsigned int width(0), height(0);
		file >> width >> height;

		//Read the 0x0a UNIX return character at the end
		char endOfLine;
		file.read(&endOfLine, sizeof(char));

		int numberOfComponents(0);
		//The type gets the number of color channels
		if (type[1] == 'F')
		{
			imagePFM = Mat(height, width, CV_32FC3);
			numberOfComponents = 3;
		}
		else if (type[1] == 'f')
		{
			imagePFM = Mat(height, width, CV_32FC1);
			numberOfComponents = 1;
		}

		//TODO Read correctly depending on the endianness
		//Read the endianness plus the 0x0a UNIX return character at the end
		//Byte Order contains -1.0 or 1.0
		char byteOrder[4];
		file.read(byteOrder, 4 * sizeof(char));

		//Find the last line return 0x0a before the pixels of the image
		char findReturn = ' ';
		while (findReturn != 0x0a)
		{
			file.read(&findReturn, sizeof(char));
		}
		//Read each RGB colors as 3 floats and store it in the image.
		float *color = new float[numberOfComponents];
		for (unsigned int i = 0; i < height; ++i)
		{
			for (unsigned int j = 0; j < width; ++j)
			{
				file.read((char *)color, numberOfComponents * sizeof(float));
				//In the PFM format the image is upside down
				if (numberOfComponents == 3)
				{
					//OpenCV stores the color as BGR
					imagePFM.at<Vec3f>(height - 1 - i, j) = Vec3f(color[2], color[1], color[0]);
				}
				else if (numberOfComponents == 1)
				{
					//OpenCV stores the color as float
					imagePFM.at<float>(height - 1 - i, j) = color[0] == 50000 ? 0.0 : color[0];
				}
			}
		}
		//Close file
		file.close();
	}
	else
	{
		cerr << "Could not open the file : " << filePath << endl;
	}
	return imagePFM;
}

/**
 * Saves the image as a PFM file.
 * @brief savePFM
 * @param image
 * @param filePath
 * @return
 */
bool savePFM(const cv::Mat image, const std::string filePath)
{
	//Open the file as binary!
	ofstream imageFile(filePath.c_str(), ios::out | ios::trunc | ios::binary);
	if (imageFile)
	{
		int width(image.cols), height(image.rows);
		int numberOfComponents(image.channels());

		//Write the type of the PFM file and ends by a line return
		char type[3];
		type[0] = 'P';
		type[2] = 0x0a;

		if (numberOfComponents == 3)
		{
			type[1] = 'F';
		}
		else if (numberOfComponents == 1)
		{
			type[1] = 'f';
		}

		imageFile << type[0] << type[1] << type[2];

		//Write the width and height and ends by a line return
		imageFile << width << " " << height << type[2];

		//Assumes little endian storage and ends with a line return 0x0a
		//Stores the type
		char byteOrder[10];
		byteOrder[0] = '-';
		byteOrder[1] = '1';
		byteOrder[2] = '.';
		byteOrder[3] = '0';
		byteOrder[4] = '0';
		byteOrder[5] = '0';
		byteOrder[6] = '0';
		byteOrder[7] = '0';
		byteOrder[8] = '0';
		byteOrder[9] = 0x0a;

		for (int i = 0; i < 10; ++i)
		{
			imageFile << byteOrder[i];
		}

		//Store the floating points RGB color upside down, left to right
		float *buffer = new float[numberOfComponents];

		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				if (numberOfComponents == 1)
				{
					buffer[0] = image.at<float>(height - 1 - i, j);
				}
				else
				{
					Vec3f color = image.at<Vec3f>(height - 1 - i, j);

					//OpenCV stores as BGR
					buffer[0] = color.val[2];
					buffer[1] = color.val[1];
					buffer[2] = color.val[0];
				}

				//Write the values
				imageFile.write((char *)buffer, numberOfComponents * sizeof(float));
			}
		}

		delete[] buffer;

		imageFile.close();
	}
	else
	{
		cerr << "Could not open the file : " << filePath << endl;
		return false;
	}
	return true;
}
