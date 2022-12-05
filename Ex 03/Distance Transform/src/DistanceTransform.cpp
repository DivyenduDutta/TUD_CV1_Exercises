#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <chrono>

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

int main(int argc, char** argv) {

	cv::Mat src, distance_image;

	int image_read_status = ReadImage(argc, argv, "ccl_image.jpg", src);

	if (image_read_status != 0)
		return -1;


	cv::waitKey(0);
	return 0;
}