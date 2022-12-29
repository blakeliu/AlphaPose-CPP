//
// Created by tf on 22-2-7.
//

#ifndef _TYPES_H
#define _TYPES_H

#include "headers.h"

namespace types
{
	template<typename T=float>
	struct CenterScale {
		typedef T value_type;
		value_type center_x;
		value_type center_y;
		value_type scale_w;
		value_type scale_h;
	};

	template<typename _T1 = float, typename _T2 = float>
	static inline void __assert_type()
	{
		static_assert(std::is_pod<_T1>::value && std::is_pod<_T2>::value
			&& std::is_floating_point<_T2>::value
			&& (std::is_integral<_T1>::value || std::is_floating_point<_T1>::value),
			"not support type.");
	} // only support for some specific types. check at compile-time.

	// bounding box.
	template<typename T1 = float, typename T2 = float>
	struct BoundingBoxType
	{
		typedef T1 value_type;
		typedef T2 score_type;
		value_type x1; //top left x
		value_type y1; //top left y
		value_type x2; //right bottom x
		value_type y2; //right bottom y
		score_type score;
		const char* label_text;
		unsigned int label; // for general object detection.
		bool flag; // future use.
		// convert type.
		template<typename O1, typename O2 = score_type>
		BoundingBoxType<O1, O2> convert_type() const;

		template<typename O1, typename O2 = score_type>
		value_type iou_of(const types::BoundingBoxType<O1, O2>& other) const;

		value_type width() const;

		value_type height() const;

		value_type area() const;

		cv::Rect rect() const;

		cv::Point2i tl() const;

		cv::Point2i rb() const;

		BoundingBoxType() :
			x1(static_cast<value_type>(0)), y1(static_cast<value_type>(0)),
			x2(static_cast<value_type>(0)), y2(static_cast<value_type>(0)),
			score(static_cast<score_type>(0)), label_text(nullptr), label(0),
			flag(false)
		{
			types::__assert_type<value_type, score_type>();
		}
	}; // End BoundingBox.

	typedef BoundingBoxType<int, float> Boxi;
	typedef BoundingBoxType<float, float> Boxf;
	typedef BoundingBoxType<double, double> Boxd;

	//landmark points
	typedef struct LandmarksType {
		std::vector<cv::Point2f> points;
		bool flag;

		LandmarksType() : flag(false)
		{};
	} Landmarks;

	//bounding box with landmark.
	typedef struct BoxfWithLandmarksType
	{
		Boxf box;
		Landmarks landmarks;
		bool flag;

		BoxfWithLandmarksType() : flag(false)
		{};
	} BoxfWithLandmarks;

	//face whether to wear mask.
	typedef struct MaskType {
		float score;        //����
		unsigned int label; //0: none, 1: mask
		const char* text;   //"none" or "mask"

		bool flag;

		MaskType() : flag(false) {};
	} Mask;
}


template<typename T1, typename T2>
template<typename O1, typename O2>
inline types::BoundingBoxType<O1, O2> types::BoundingBoxType<T1, T2>::convert_type() const
{
	typedef O1 other_value_type;
	typedef O2 other_score_type;
	types::__assert_type<other_value_type, other_score_type>();
	types::__assert_type<value_type, score_type>();
	types::BoundingBoxType<other_value_type, other_score_type> other;
	other.x1 = static_cast<other_value_type>(x1);
	other.y1 = static_cast<other_value_type>(y1);
	other.x2 = static_cast<other_value_type>(x2);
	other.y2 = static_cast<other_value_type>(y2);
	other.score = static_cast<other_score_type>(score);
	other.label_text = label_text;
	other.label = label;
	other.flag = flag;
	return other;
}

template<typename T1, typename T2>
template<typename O1, typename O2>
inline typename types::BoundingBoxType<T1, T2>::value_type
types::BoundingBoxType<T1, T2>::iou_of(const types::BoundingBoxType<O1, O2>& other) const
{
	types::BoundingBoxType<value_type, score_type> tbox = \
		other.template convert_type<value_type, score_type>();
	value_type inner_x1 = x1 > tbox.x1 ? x1 : tbox.x1;
	value_type inner_y1 = y1 > tbox.y1 ? y1 : tbox.y1;
	value_type inner_x2 = x2 < tbox.x2 ? x2 : tbox.x2;
	value_type inner_y2 = y2 < tbox.y2 ? y2 : tbox.y2;
	value_type inner_h = inner_y2 - inner_y1 + static_cast<value_type>(1.0f);
	value_type inner_w = inner_x2 - inner_x1 + static_cast<value_type>(1.0f);
	if (inner_h <= static_cast<value_type>(0.f) || inner_w <= static_cast<value_type>(0.f))
		return std::numeric_limits<value_type>::min();
	value_type inner_area = inner_h * inner_w;
	return static_cast<value_type>(inner_area / (area() + tbox.area() - inner_area));
}

template<typename T1, typename T2>
inline cv::Rect types::BoundingBoxType<T1, T2>::rect() const
{
	types::__assert_type<value_type, score_type>();
	auto boxi = this->template convert_type<int>();
	return cv::Rect(boxi.x1, boxi.y1, boxi.width(), boxi.height());
}

template<typename T1, typename T2>
inline cv::Point2i types::BoundingBoxType<T1, T2>::tl() const
{
	types::__assert_type<value_type, score_type>();
	auto boxi = this->template convert_type<int>();
	return cv::Point2i(boxi.x1, boxi.y1);
}

template<typename T1, typename T2>
inline cv::Point2i types::BoundingBoxType<T1, T2>::rb() const
{
	types::__assert_type<value_type, score_type>();
	auto boxi = this->template convert_type<int>();
	return cv::Point2i(boxi.x2, boxi.y2);
}

template<typename T1, typename T2>
inline typename types::BoundingBoxType<T1, T2>::value_type
types::BoundingBoxType<T1, T2>::width() const
{
	types::__assert_type<value_type, score_type>();
	return (x2 - x1 + static_cast<value_type>(1));
}

template<typename T1, typename T2>
inline typename types::BoundingBoxType<T1, T2>::value_type
types::BoundingBoxType<T1, T2>::height() const
{
	types::__assert_type<value_type, score_type>();
	return (y2 - y1 + static_cast<value_type>(1));
}

template<typename T1, typename T2>
inline typename types::BoundingBoxType<T1, T2>::value_type
types::BoundingBoxType<T1, T2>::area() const
{
	types::__assert_type<value_type, score_type>();
	return std::abs<value_type>(width() * height());
}


#endif //_TYPES_H
