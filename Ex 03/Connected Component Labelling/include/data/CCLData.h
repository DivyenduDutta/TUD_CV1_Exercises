#pragma once

#include <map>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"


typedef std::uint16_t u16;

namespace data {

	struct ConnCompData {
		u16 component_id;
		cv::Rect bounding_box;
		std::map<std::string, u16> moments; //moments upto the second order
		cv::Point2d centroid; //centroid of the component
		std::map<std::string, double> central_moments; //central moments upto the second order
		double orientation; //orientation of the component in degrees
		double eccentricity; //how elongated the component is
	};

}
