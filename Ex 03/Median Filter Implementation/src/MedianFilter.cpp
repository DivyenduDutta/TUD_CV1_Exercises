#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

typedef std::uint8_t u8;

const int KERNEL_SIZE = 7; //separable into nX1 and 1Xn vectors

int ReadImage(int argc, char** argv, const char* image_name, cv::Mat& src) {
	image_name = (argc >= 2) ? argv[1] : image_name;
	src = cv::imread(cv::samples::findFile(image_name), cv::IMREAD_GRAYSCALE);
	if (src.empty()) {
		printf("\n There was a problem reading image : %s", image_name);
		return -1;
	}
	return 0;
}

//splice/extract elements of Mat row
void ExtractRowOfKernel(const cv::Mat& row_m, u8* row_k, int start) {
	for (int i = start; i < (start+KERNEL_SIZE); i++) {
		row_k[(i-start)] = row_m.at<u8>(0, i);
	}
}

//finds the median of a list of greyscale values
u8 FindMedian(u8* row_k) {
	std::sort(row_k, row_k + KERNEL_SIZE);
	return row_k[static_cast<u8>(KERNEL_SIZE * 0.5)];
}

//Generate image with salt and pepper noise from input image
void GenerateSaltPepperNoise(const cv::Mat& src, cv::Mat& snp_img) {
	cv::Mat salt_pepper_noise = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	cv::randu(salt_pepper_noise, 0, 255);

	const u8 NOISE_THRESHOLD = 20;

	//create white and black masks
	cv::Mat black = salt_pepper_noise < NOISE_THRESHOLD;
	cv::Mat white = salt_pepper_noise > (255 - NOISE_THRESHOLD);

	snp_img.setTo(0, black);
	snp_img.setTo(255, white);
}

void SaveImage(const char* name, const cv::Mat& img) {
	int k = cv::waitKey(0);
	if (k == 's')
	{
		cv::imwrite(name, img);
	}
}

void MedianFilterSeparable(const cv::Mat& src, cv::Mat& dst) {
	
	size_t padding_size = static_cast<size_t>(KERNEL_SIZE * 0.5f);
	size_t rows = src.rows;
	size_t cols = src.cols;
	cv::Mat padded_image;

	//generate salt and pepper noisy image
	cv::Mat salt_pepper_img = src.clone();
	GenerateSaltPepperNoise(src, salt_pepper_img);

	cv::copyMakeBorder(salt_pepper_img, padded_image, 
					   padding_size, padding_size, padding_size, padding_size, 
					   cv::BorderTypes::BORDER_DEFAULT);

	cv::imshow("Source Image", src);
	cv::imshow("Noisy Image", salt_pepper_img);
	//SaveImage("salt_pepper_image.jpg", salt_pepper_img); //keep this commented to measure code execution time

	cv::parallel_for_(cv::Range(0, rows * cols), [cols, &padded_image, &dst](const cv::Range& range) {
		for (size_t r = range.start; r < range.end; r++) {
			size_t i = r / cols, j = r % cols;
			u8 median_of_rows[KERNEL_SIZE]; //array to hold medians of rows (first stage operation)
			for (int k = i; k < (KERNEL_SIZE+i); k++) { //mapping from src image to padded image
				u8 row_of_kernel[KERNEL_SIZE]; //stores each row of kernel
				ExtractRowOfKernel(padded_image.row(k), row_of_kernel, j);
				median_of_rows[(k-i)] = FindMedian(row_of_kernel);
			}

			//now we find the median of medians (second stage operation)
			u8 median_val = FindMedian(median_of_rows);
			dst.at<u8>(i, j) = median_val;
		}
	});

		//sequential execution	
		/*for (int r = 0; r < rows * cols; r++) {
			int i = r / cols, j = r % cols;
			uint8_t median_of_rows[KERNEL_SIZE]; //array to hold medians of rows (first stage operation)
			for (int k = i; k < (KERNEL_SIZE+i); k++) { //mapping from src image to padded image
				uint8_t row_of_kernel[KERNEL_SIZE]; //stores each row of kernel
				ExtractRowOfKernel(padded_image.row(k), row_of_kernel, j);
				median_of_rows[(k-i)] = FindMedian(row_of_kernel);
			}

			//now we find the median of medians (second stage operation)
			uint8_t median_val = FindMedian(median_of_rows);
			dst.at<uint8_t>(i, j) = median_val;
		}*/
	//std::cout << src << "\n\n" <<padded_image << "\n\n" << dst;
	
	cv::imshow("Median Filtered Image", dst);
	//SaveImage("median_filtered_image.jpg", dst); //keep this commented to measure code execution time
}


int main(int argc, char** argv) {

	cv::Mat src, dst;
	
	int image_read_status = ReadImage(argc, argv, "snp_image.jpg", src);

	if (image_read_status != 0)
		return -1;

	u8 src_image_channels = src.channels();
	dst = cv::Mat(src.rows, src.cols, CV_8UC(src_image_channels));

	//test matrix
	/*cv::Mat src = (cv::Mat_<uint8_t>(5, 5) << 1, 2, 0, 1, 2, 3, 2, 4, 9, 3, 1, 4, 5, 5, 1, 2, 3, 0, 0, 5, 1, 2, 1, 1, 3);
	cv::Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
	*/

	auto t1 = std::chrono::high_resolution_clock::now();
	MedianFilterSeparable(src, dst);
	auto t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> time_ms = t2 - t1;
	std::cout << "Execution time: " << time_ms.count() << " ms\n";


	return 0;
}


