#define _USE_MATH_DEFINES

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <chrono>
#include <stack>
#include <vector>
#include <map>
#include <cmath>

#include "data/CCLData.h"

typedef std::uint8_t u8;
typedef std::uint16_t u16;

cv::RNG rng(12345);

int ReadImage(int argc, char** argv, const char* image_name, cv::Mat& src) {
	image_name = (argc >= 2) ? argv[1] : image_name;
	src = cv::imread(cv::samples::findFile(image_name), cv::IMREAD_GRAYSCALE);
	if (src.empty()) {
		printf("\n There was a problem reading image : %s", image_name);
		return -1;
	}
	return 0;
}


void SaveImage(const char* name, const cv::Mat& img) {
	int k = cv::waitKey(0);
	if (k == 's')
	{
		cv::imwrite(name, img);
	}
}

//finds the 8 connectivity neighbors
/*
------------
 0 | 0 | 0 |
------------
 0 | X | 0 |
------------
 0 | 0 | 0 |
------------
*/
void ComputeEightConnectivityNeighbors(size_t v, const cv::Mat& src, std::vector<size_t>& neighbors) {
	int i = (v / src.cols), j = (v % src.cols);
	for (int k = -1; k <= 1; k++) {
		for (int l = -1; l <= 1; l++) {
			if (k != 0 || l != 0) {
				if(((i+k) >= 0 && (i+k) < src.rows) &&
					((j+l) >= 0 && (j+l) < src.cols))
					neighbors.push_back(((i + k) * src.cols + (j + l)));
			}
		}
	}
}

void ConvertToBinary(const cv::Mat& src, cv::Mat& bin) {
	cv::threshold(src, bin, 127, 255, cv::THRESH_BINARY);
}

//flood filling algorithm - used depth first search
u16 FindConnectedComponents(const cv::Mat& src, cv::Mat& components, std::vector<data::ConnCompData>& comp_data) {
	size_t rows = src.rows;
	size_t cols = src.cols;

	std::stack<size_t> stack;
	std::vector<size_t> neighbors;

	u16 component = 0; //label of component

	//bounding box parameters
	size_t min_x = 0, min_y = 0, max_x = 0, max_y = 0;
	
	//image moments upto second order
	std::map<std::string, u16> moment_map;
	
	//central moments upto second order
	std::map<std::string, double> central_moment_map;
	
	for (size_t v = 0; v < (rows * cols); v++) {
		//std::cout << "Processing pixel: " << v << "\n";
		size_t i = v / cols, j = v % cols;

		if (src.at<u8>(i,j) == 0 && components.at<u16>(i,j) == 0) {
			component++;
			//std::cout << "Component: " << component << "\n";
			
			min_x = max_x = i;
			min_y = max_y = j;

			components.at<u16>(i, j) = component;
			
			//computing image moments upto the second order
			moment_map["M_00"] = component;
			moment_map["M_01"] = j * component;
			moment_map["M_10"] = i * component;
			moment_map["M_02"] = j * j * component;
			moment_map["M_20"] = i * i * component;
			moment_map["M_11"] = i * j * component;

			stack.push(v);
			while (!stack.empty()) {
				size_t v = stack.top();
				stack.pop();
				ComputeEightConnectivityNeighbors(v, src, neighbors);
				for (auto it = neighbors.begin(); it != neighbors.end(); it++) {
					size_t i = (*it) / cols, j = (*it) % cols;
					if (src.at<u8>(i,j) == 0 && components.at<u16>(i,j) == 0) {
						components.at<u16>(i, j) = component;
					
						//bounding box
						min_x = min_x < i ? min_x : i;
						min_y = min_y < j ? min_y : j;
						max_x = max_x > i ? max_x : i;
						max_y = max_y > j ? max_y : j;
						
						//image moments upto second order
						moment_map["M_00"] += component;
						moment_map["M_01"] += (j * component);
						moment_map["M_10"] += (i * component);
						moment_map["M_02"] += (j * j * component);
						moment_map["M_20"] += (i * i * component);
						moment_map["M_11"] += (i * j * component);	
						
						stack.push(*it);
					}
				}
				neighbors.clear();
			}

			//compute centroid
			double centroid_x = (double)moment_map["M_10"] / moment_map["M_00"];
			double centroid_y = (double)moment_map["M_01"] / moment_map["M_00"];

			//central moments upto second order
			central_moment_map["u_00"] = moment_map["M_00"];
			central_moment_map["u_01"] = 0;
			central_moment_map["u_10"] = 0;
			central_moment_map["u_02"] = moment_map["M_02"] - centroid_y * moment_map["M_01"];
			central_moment_map["u_20"] = moment_map["M_20"] - centroid_x * moment_map["M_10"];
			central_moment_map["u_11"] = moment_map["M_11"] - centroid_x * moment_map["M_01"];

			//computing orienation
			double u_p_20 = central_moment_map["u_20"] / central_moment_map["u_00"];
			double u_p_02 = central_moment_map["u_02"] / central_moment_map["u_00"];
			double u_p_11 = central_moment_map["u_11"] / central_moment_map["u_00"];
			double orientation = (0.5 * (atan((2 * u_p_11) / (u_p_20 - u_p_02))) * (180/M_PI));

			//computing eccentricity
			double pt_1 = (u_p_20 + u_p_02) * 0.5;
			double pt_2 = std::hypot(2 * u_p_11, (u_p_20 - u_p_02)) * 0.5;
			double lambda_1 = pt_1 + pt_2;
			double lambda_2 = pt_1 - pt_2;
			double eccentricity = 0.0;
			if (lambda_2 > lambda_1)
				std::cerr << "Invalid arg to sqrt for component: " << component << "\n";
			else
				eccentricity = sqrt(1 - (lambda_2 / lambda_1));

			//switch x and y to move to cartesian coord system for plotting purposes
			data::ConnCompData data = data::ConnCompData{component,
				cv::Rect_<size_t>{
					min_y, min_x, (max_x - min_x+1), (max_y - min_y+1) // +1 because they are pixels
				}, moment_map, cv::Point2f(centroid_x,centroid_y),
				central_moment_map,
				orientation, eccentricity
			};
			comp_data.push_back(data);
		
			moment_map.clear();
			central_moment_map.clear();
		}
	}
	
	cv::imshow("CCL output", components);
	//SaveImage("ccl_output.jpg", components); //keep this commented to measure exec time
	//std::cout << components << "\n";
	return component;
}

//computes bounding box using opencv functions 
void BoundingBox(cv::Mat& src, cv::Mat& components, std::vector<data::ConnCompData>& component_data) {
	cv::Mat masked_img;
	cv::Mat bounding_box_img;
	cv::cvtColor(src.clone(), bounding_box_img, cv::COLOR_GRAY2RGB);
	cv::Rect bounding_box;
	
	for (size_t i = 1; i <= component_data.size(); i++) {
		//create mask for component
		masked_img = (components == i);

		//get the bounding box of the component
		bounding_box = cv::boundingRect(masked_img);
		component_data[(i - 1)].bounding_box = bounding_box;
		cv::rectangle(bounding_box_img, bounding_box.tl(), bounding_box.br(), cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)), 1);
	}

	cv::imshow("bounding box", bounding_box_img);
	//SaveImage("bounding_box.jpg", bounding_box_img);
	//cv::imshow("mask", masked_img);
}


//draws bounding boxes
void DrawBoundingBox(cv::Mat& src, std::vector<data::ConnCompData>& component_data) {		
	cv::Mat bounding_box_img;
	cv::cvtColor(src.clone(), bounding_box_img, cv::COLOR_GRAY2RGB);
	for (auto it = component_data.begin(); it != component_data.end(); it++) {
		cv::Rect bounding_box = (*it).bounding_box;
		cv::rectangle(bounding_box_img, bounding_box.tl(), bounding_box.br(), cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)), 1);
	}
	cv::imshow("bounding box", bounding_box_img);
	//SaveImage("bounding_box.jpg", bounding_box_img);
}


int main(int argc, char** argv) {

	cv::Mat src, bin, components;

	int image_read_status = ReadImage(argc, argv, "ccl_image.jpg", src);

	if (image_read_status != 0)
		return -1;

	u8 src_image_channels = src.channels();
	components = cv::Mat::zeros(src.rows, src.cols, CV_16UC(src_image_channels));
	bin = cv::Mat::zeros(src.rows, src.cols, CV_8UC(src_image_channels));

	std::vector<data::ConnCompData> component_data; //stores various info about the components

	//test matrix
	/*cv::Mat src = (cv::Mat_<uint8_t>(5, 5) << 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0);
	cv::Mat components = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	*/
	
	//test connectivity code
	/*std::vector<size_t> vec;
	ComputeEightConnectivityNeighbors(18, src, vec);
	for (auto it = vec.begin(); it != vec.end(); it++) {
		std::cout << *it << "\n";
	}*/

	ConvertToBinary(src, bin);
	
	cv::imshow("Binary Image", bin);
	//SaveImage("ccl_binary_image.jpg", bin); //keep this commented to measure exec time

	auto t1 = std::chrono::high_resolution_clock::now();
	u16 number_of_components = FindConnectedComponents(bin, components, component_data);
	auto t2 = std::chrono::high_resolution_clock::now();

	//compute bounding box of components using opencv and draw them
	//BoundingBox(src, components, component_data);

	std::chrono::duration<double, std::milli> time_ms = t2 - t1;
	std::cout << "Execution time: " << time_ms.count() << " ms\n";

	//draw bounding boxes
	DrawBoundingBox(src, component_data);

	/*
	std::cout << "bounding box is: \n";
	std::cout << "height: " << component_data[0].bounding_box.height <<"\n"
		<< "width: " << component_data[0].bounding_box.width <<"\n"
		<< "x: " << component_data[0].bounding_box.x <<"\n"
		<< "y: " << component_data[0].bounding_box.y << "\n";
	*/
	
	/*
	std::cout << "Image moments are: \n";
	std::cout << "M_00: " << component_data[0].moments["M_00"] << "\n"
		<< "M_01: " << component_data[0].moments["M_01"] << "\n"
		<< "M_10: " << component_data[0].moments["M_10"] << "\n"
		<< "M_02: " << component_data[0].moments["M_02"] << "\n"
		<< "M_20: " << component_data[0].moments["M_20"] << "\n"
		<< "M_11: " << component_data[0].moments["M_11"] << "\n"
		<< "centroid: " << component_data[0].centroid.x << " , " << component_data[0].centroid.y << "\n";
	
	std::cout << "Central moments are: \n";
	std::cout << "u_00: " << component_data[0].central_moments["u_00"] << "\n"
		<< "u_01: " << component_data[0].central_moments["u_01"] << "\n"
		<< "u_10: " << component_data[0].central_moments["u_10"] << "\n"
		<< "u_02: " << component_data[0].central_moments["u_02"] << "\n"
		<< "u_20: " << component_data[0].central_moments["u_20"] << "\n"
		<< "u_11: " << component_data[0].central_moments["u_11"] << "\n";

	std::cout << "Orientation is: " << component_data[0].orientation << "\n"
		<< "Eccentricity is: " << component_data[0].eccentricity;
	*/

	cv::waitKey(0);
	return 0;
}


