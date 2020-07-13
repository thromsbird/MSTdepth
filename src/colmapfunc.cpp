#include <string>
#include <algorithm>
#include "colmapfunc.h"
#include <Eigen/Eigen>

// Read data from text or binary file. Prefer binary data if it exists.
//void ReadCamerasText(const std::string& path);
//void ReadImagesText(const std::string& path);
//void ReadPoints3DText(const std::string& path);
//void ReadCamerasBinary(const std::string& path);
//void ReadImagesBinary(const std::string& path);
//void ReadPoints3DBinary(const std::string& path);
bool IsNotWhiteSpace(const int character)
{
	return character != ' ' && character != '\n' && character != '\r' &&
		   character != '\t';
}
void StringLeftTrim(std::string *str)
{
	str->erase(str->begin(),
			   std::find_if(str->begin(), str->end(), IsNotWhiteSpace));
}
void StringRightTrim(std::string *str)
{
	str->erase(std::find_if(str->rbegin(), str->rend(), IsNotWhiteSpace).base(),
			   str->end());
}
void StringTrim(std::string *str)
{
	StringLeftTrim(str);
	StringRightTrim(str);
}
#define CHECK(condition)                                         \
	LOG_IF(FATAL, GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!(condition))) \
		<< "Check failed: " #condition " "
//Normalize the quaternion vector.
Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d &qvec)
{
	const double norm = qvec.norm();
	if (norm == 0)
	{
		// We do not just use (1, 0, 0, 0) because that is a constant and when used
		// for automatic differentiation that would lead to a zero derivative.
		return Eigen::Vector4d(1.0, qvec(1), qvec(2), qvec(3));
	}
	else
	{
		return qvec / norm;
	}
}

Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d &qvec)
{
	const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
	const Eigen::Quaterniond quat(normalized_qvec(0), normalized_qvec(1),
								  normalized_qvec(2), normalized_qvec(3));
	return quat.toRotationMatrix();
}
//ColmapDepthMap
ColmapDepthMap::ColmapDepthMap() : ColmapDepthMap(0, 0, 1, -1.0f, -1.0f) {}
ColmapDepthMap::ColmapDepthMap(const int width, const int height, const size_t depth, const float depth_min, const float depth_max) : width_(width), height_(height), depth_(depth), depth_min_(depth_min), depth_max_(depth_max)
{
	data_.resize(width_ * height_ * depth_, 0);
}
int ColmapDepthMap::GetWidth() const
{
	return width_;
}
int ColmapDepthMap::GetHeight() const
{
	return height_;
}
void ColmapDepthMap::Read(const std::string &path)
{
	std::fstream text_file(path, std::ios::in | std::ios::binary);
	if (!text_file)
	{
		std::cout << "can not open " << path << std::endl;
		exit(1);
	}
	char unused_char;
	text_file >> width_ >> unused_char >> height_ >> unused_char >> depth_ >>
		unused_char;
	std::streampos pos = text_file.tellg();
	text_file.close();

	data_.resize(width_ * height_ * depth_);
	std::fstream binary_file(path, std::ios::in | std::ios::binary);
	binary_file.seekg(pos);
	ReadBinaryLittleEndian<float>(&binary_file, &data_);
	binary_file.close();
}
void ColmapDepthMap::Write(const std::string &path)
{
	std::fstream text_file(path, std::ios::out);
	text_file << width_ << "&" << height_ << "&" << depth_ << "&";
	text_file.close();
	std::fstream binary_file(path, std::ios::out | std::ios::binary | std::ios::app);

	WriteBinaryLittleEndian<float>(&binary_file, data_);
	binary_file.close();
}
float ColmapDepthMap::Get(const int row, const int col) const
{
	return data_.at(row * width_ + col);
}
const std::vector<float> &ColmapDepthMap::Getdepthmap() const
{
	return data_;
}
void ColmapDepthMap::Setdepthmap(std::vector<float> &setData)
{
	data_ = setData;
	return;
}

//ColmapImage
ColmapImage::ColmapImage()
	: image_id_(kInvalidImageId),
	  name_(""),
	  camera_id_(kInvalidCameraId),
	  registered_(false),
	  num_points3D_(0),
	  qvec_(1.0, 0.0, 0.0, 0.0),
	  tvec_(0.0, 0.0, 0.0),
	  R_(3, 3),
	  T_(0.0, 0.0, 0.0) {}
void ColmapImage::NormalizeQvec() { qvec_ = NormalizeQuaternion(qvec_); }
void ColmapImage::RotationMatrix()
{
	R_ = QuaternionToRotationMatrix(qvec_);
}
void ColmapImage::NormalizeT_()
{
	T_(0) = tvec_(0);
	T_(1) = -tvec_(1);
	T_(2) = -tvec_(2);
}
