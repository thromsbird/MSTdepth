/*******************************************************
\Author:	Li Minghao
\Function:	COLMAP classes (using OpenCV).
\Reference:	"Image-Based Rendering for Large-Scale Outdoor Scenes With Fusion of Monocular and Multi-View Stereo Depth," in IEEE Access, vol. 8, pp. 117551-117565, 2020
********************************************************/
#ifndef LMH_COLMAP_FUNC_H
#define LMH_COLMAP_FUNC_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

// Unique identifier for cameras.
typedef uint32_t camera_t;
// Unique identifier for images.
typedef uint32_t image_t;
// Index per image, i.e. determines maximum number of 2D points per image.
typedef uint32_t point2D_t;
// Unique identifier per added 3D point. Since we add many 3D points,
// delete them, and possibly re-add them again, the maximum number of allowed
// unique indices should be large.
typedef uint64_t point3D_t;
// Values for invalid identifiers or indices.
const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
const image_t kInvalidImageId = std::numeric_limits<image_t>::max();

bool IsNotWhiteSpace(const int character);
void StringLeftTrim(std::string *str);
void StringRightTrim(std::string *str);
void StringTrim(std::string *str);

template <typename T>
T LittleEndianToNative(const T x);
template <typename T>
T NativeToLittleEndian(const T x);
template <typename T>
T ReadBinaryLittleEndian(std::istream *stream);
template <typename T>
void ReadBinaryLittleEndian(std::istream *stream, std::vector<T> *data);
template <typename T>
void WriteBinaryLittleEndian(std::ostream *stream, const T &data);
template <typename T>
void WriteBinaryLittleEndian(std::ostream *stream, const std::vector<T> &data);

//Normalize the quaternion vector.
Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d &qvec);
Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d &qvec);

inline bool IsLittleEndian()
{
#ifdef BOOST_BIG_ENDIAN
	return false;
#else
	return true;
#endif
}
template <typename T>
T ReverseBytes(const T &data)
{
	T data_reversed = data;
	std::reverse(reinterpret_cast<char *>(&data_reversed),
				 reinterpret_cast<char *>(&data_reversed) + sizeof(T));
	return data_reversed;
}
template <typename T>
T LittleEndianToNative(const T x)
{
	if (IsLittleEndian())
	{
		return x;
	}
	else
	{
		return ReverseBytes(x);
	}
}
template <typename T>
T NativeToLittleEndian(const T x)
{
	if (IsLittleEndian())
	{
		return x;
	}
	else
	{
		return ReverseBytes(x);
	}
}
template <typename T>
T ReadBinaryLittleEndian(std::istream *stream)
{
	T data_little_endian;
	stream->read(reinterpret_cast<char *>(&data_little_endian), sizeof(T));
	return LittleEndianToNative(data_little_endian);
}
template <typename T>
void ReadBinaryLittleEndian(std::istream *stream, std::vector<T> *data)
{
	for (size_t i = 0; i < data->size(); ++i)
	{
		(*data)[i] = ReadBinaryLittleEndian<T>(stream);
	}
}

template <typename T>
void WriteBinaryLittleEndian(std::ostream *stream, const T &data)
{
	const T data_little_endian = NativeToLittleEndian(data);
	stream->write(reinterpret_cast<const char *>(&data_little_endian), sizeof(T));
}
template <typename T>
void WriteBinaryLittleEndian(std::ostream *stream, const std::vector<T> &data)
{
	for (const auto &elem : data)
	{
		WriteBinaryLittleEndian<T>(stream, elem);
	}
}
//refer file: colmap->src->base->image.cc
class ColmapImage
{
public:
	ColmapImage();

	// Access the unique identifier of the image.
	inline image_t ImageId() const;
	inline void SetImageId(const image_t image_id);
	// Access the name of the image.
	inline const std::string &Name() const;
	inline std::string &Name();
	inline void SetName(const std::string &name);
	// Access the unique identifier of the camera. Note that multiple images
	// might share the same camera.
	inline camera_t CameraId() const;
	inline void SetCameraId(const camera_t camera_id);
	// Check if image is registered.
	inline bool IsRegistered() const;
	inline void SetRegistered(const bool registered);

	// Access quaternion vector as (qw, qx, qy, qz) specifying the rotation of the
	// pose which is defined as the transformation from world to image space.
	inline const Eigen::Vector4d &Qvec() const;
	inline Eigen::Vector4d &Qvec();
	inline double Qvec(const size_t idx) const;
	inline double &Qvec(const size_t idx);
	inline void SetQvec(const Eigen::Vector4d &qvec);
	// Access quaternion vector as (tx, ty, tz) specifying the translation of the
	// pose which is defined as the transformation from world to image space.
	inline const Eigen::Vector3d &Tvec() const;
	inline Eigen::Vector3d &Tvec();
	inline double Tvec(const size_t idx) const;
	inline double &Tvec(const size_t idx);
	inline void SetTvec(const Eigen::Vector3d &tvec);

	inline const Eigen::Matrix3d &R() const;
	inline const Eigen::Vector3d &T() const;
	// Normalize the quaternion vector.
	void NormalizeQvec();
	// Compose rotation matrix from quaternion vector.
	void RotationMatrix();
	// Normalize the translate vector.
	void NormalizeT_();

private:
	// Identifier of the image, if not specified `kInvalidImageId`.
	image_t image_id_;
	// The name of the image, i.e. the relative path.
	std::string name_;
	// The identifier of the associated camera. Note that multiple images might
	// share the same camera. If not specified `kInvalidCameraId`.
	camera_t camera_id_;
	// Whether the image is successfully registered in the reconstruction.
	bool registered_;
	// The number of 3D points the image observes, i.e. the sum of its `points2D`
	// where `point3D_id != kInvalidPoint3DId`.
	point2D_t num_points3D_;

	Eigen::Vector4d qvec_;
	Eigen::Vector3d tvec_;
	Eigen::Matrix3d R_;
	Eigen::Vector3d T_;
};
//refer file: colmap->src->mvs->depth_map.cc ,mat.h
class ColmapDepthMap
{
public:
	ColmapDepthMap();
	ColmapDepthMap(const int width, const int height, const size_t depth, const float depth_min,
				   const float depth_max);
	int GetWidth() const;
	int GetHeight() const;
	void Read(const std::string &path);
	void Write(const std::string &path);
	inline float Get(const int row, const int col) const;
	const std::vector<float> &Getdepthmap() const;
	void Setdepthmap(std::vector<float> &setData);

private:
	int width_ = 0;
	int height_ = 0;
	size_t depth_ = 0;
	float depth_min_ = -1.0f;
	float depth_max_ = -1.0f;
	std::vector<float> data_;
};

image_t ColmapImage::ImageId() const { return image_id_; }
void ColmapImage::SetImageId(const image_t image_id) { image_id_ = image_id; }
const std::string &ColmapImage::Name() const { return name_; }
std::string &ColmapImage::Name() { return name_; }
void ColmapImage::SetName(const std::string &name) { name_ = name; }
inline camera_t ColmapImage::CameraId() const { return camera_id_; }
inline void ColmapImage::SetCameraId(const camera_t camera_id)
{
	//CHECK_NE(camera_id, kInvalidCameraId);
	camera_id_ = camera_id;
}
bool ColmapImage::IsRegistered() const { return registered_; }
void ColmapImage::SetRegistered(const bool registered) { registered_ = registered; }

const Eigen::Vector4d &ColmapImage::Qvec() const { return qvec_; }
Eigen::Vector4d &ColmapImage::Qvec() { return qvec_; }
inline double ColmapImage::Qvec(const size_t idx) const { return qvec_(idx); }
inline double &ColmapImage::Qvec(const size_t idx) { return qvec_(idx); }
void ColmapImage::SetQvec(const Eigen::Vector4d &qvec) { qvec_ = qvec; }

const Eigen::Vector3d &ColmapImage::Tvec() const { return tvec_; }
Eigen::Vector3d &ColmapImage::Tvec() { return tvec_; }
inline double ColmapImage::Tvec(const size_t idx) const { return tvec_(idx); }
inline double &ColmapImage::Tvec(const size_t idx) { return tvec_(idx); }
void ColmapImage::SetTvec(const Eigen::Vector3d &tvec) { tvec_ = tvec; }

const Eigen::Matrix3d &ColmapImage::R() const { return R_; }
const Eigen::Vector3d &ColmapImage::T() const { return T_; }

#endif