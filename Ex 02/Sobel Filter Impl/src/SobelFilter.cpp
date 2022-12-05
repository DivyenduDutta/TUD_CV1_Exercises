#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <cmath>

cv::Mat src, dst;

static int ReadImage(int argc, char** argv, const char* image_name) {
	image_name = (argc >= 2) ? argv[1] : image_name;
	src = cv::imread(cv::samples::findFile(image_name), cv::IMREAD_GRAYSCALE);
	if (src.empty()) {
		printf("\n There was a problem reading image : %s", image_name);
		return -1;
	}
	return 0;
}

static void SobelFilterGradientX(const cv::Mat& gradient_x) {
	
	//temp matrix to store 1D convolutions over rows using [-1, 0, 1]
	cv::Mat row_temp = cv::Mat(src.rows, src.cols, CV_8UC1);
	
	//we do the below because we consider pixel intensities outside image to be 0
	row_temp.col(0) = src.col(1);
	row_temp.col(src.cols - 1) = -src.col(src.cols - 2);

	//process columns 1 to n-2 since we have handled columns 1 and n-1 above
	for (int c = 1; c < (src.cols - 1); c++) {
		row_temp.col(c) = src.col(c + 1) - src.col(c - 1);
	}


	//temp matrix to store 1D convolutions over columns using [0, 1, 1]
	cv::Mat col_temp = cv::Mat(src.rows, src.cols, CV_8UC1);

	//process rows 0 to n-2, not n-1 since there's no row after it
	for (int r = 0; r < src.rows - 1; ++r) {
		col_temp.row(r) = row_temp.row(r) + row_temp.row(r + 1);
	}

	//std::cout << row_temp << "\n";


	//store 1D convolutions over columns using [1, 1, 0]
	//process rows 1 to n-1, not 0 since there's no row before it
	for (int r = 1; r < src.rows; ++r) {
		gradient_x.row(r) = col_temp.row(r-1) + col_temp.row(r);
	}
	
	//std::cout << src << "\n";
	cv::imwrite("gradient_x_sobel.jpg", gradient_x);

}


static void SobelFilterGradientY(const cv::Mat& gradient_y) {

	//temp matrix to store 1D convolutions over rows using [0, 1, 1]
	cv::Mat row_temp_1 = cv::Mat(src.rows, src.cols, CV_8UC1);

	//process cols 0 to m-2, not m-1 since there's no col after it
	for (int c = 0; c < src.cols - 1; ++c) {
		row_temp_1.col(c) = src.col(c) + src.col(c + 1);
	}
	
	//temp matrix to store 1D convolutions over rows using [1, 1, 0]
	cv::Mat row_temp_2 = cv::Mat(src.rows, src.cols, CV_8UC1);

	//process rows 1 to m-1, not 0 since there's no row before it
	for (int c = 1; c < src.cols; ++c) {
		row_temp_2.col(c) = row_temp_1.col(c-1) + row_temp_1.col(c);
	}
	
	//we do the below because we consider pixel intensities outside image to be 0
	gradient_y.row(0) = row_temp_2.row(1);
	gradient_y.row(src.rows - 1) = -row_temp_2.row(src.rows - 2);

	//store 1D convolutions over rows using [-1, 0, 1]
	//process rows 1 to n-2 since we have handled columns 1 and n-1 above
	for (int r = 1; r < (src.rows - 1); r++) {
		gradient_y.row(r) = row_temp_2.row(r + 1) - row_temp_2.row(r - 1);
	}
	
	//std::cout << src << "\n";
	cv::imwrite("gradient_y_sobel.jpg", gradient_y);

}

static void SobelEdgeDetection(const cv::Mat& gradient_x,const cv::Mat& gradient_y, cv::Mat& sobel_final_mag) {
	for (int r = 0; r < src.rows - 1; ++r) {
		for (int c = 0; c < src.cols-1; ++c) {
			sobel_final_mag.at<float>(r, c) = sqrt(
									pow(gradient_x.at<unsigned char>(r,c),2) +
									pow(gradient_y.at<unsigned char>(r,c),2));
		}
	}

	cv::imwrite("sobel_final.jpg", sobel_final_mag);
}


int main(int argc, char** argv) {

	int image_read_status = ReadImage(argc, argv, "sample_sobel_image.jpg");

	if (image_read_status != 0)
		return -1;

	//gradients/derivatives in x direction - 1D convolutions over columns using [1, 1, 0]
	cv::Mat gradient_x = cv::Mat(src.rows, src.cols, CV_8UC1);

	//gradients/derivatives in y direction - 1D convolutions over columns using [-1, 0, 1]
	cv::Mat gradient_y = cv::Mat(src.rows, src.cols, CV_8UC1);

	cv::Mat sobel_final_mag = cv::Mat(src.rows, src.cols, CV_32FC1);

	SobelFilterGradientX(gradient_x);
	SobelFilterGradientY(gradient_y);

	SobelEdgeDetection(gradient_x, gradient_y, sobel_final_mag);

	return 0;
}
