#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace std;


int main(int argc, char **argv) {
    // Load the image from a file
    if (argc < 2){
        cerr << "format: to_point_cloud.exe (depth image) (output file name)" << endl;
        return 1;
    }

    // cv::Mat image = cv::imread(string(argv[1]), cv::IMREAD_COLOR);

    // // Check if the image was loaded successfully
    // if (image.empty()) {
    //     std::cerr << "Error: Could not open or find the image" << std::endl;
    //     return 1;
    // }

    // // You can now work with the image as a matrix (cv::Mat)
    // // For example, you can access pixel values, manipulate the image, etc.

    // // Display the image (you may need to configure your OpenCV build for this)
    // cv::imshow("Loaded Image", image);
    // cv::waitKey(0);

    // cout << "Width : " << image.cols << endl;
    // cout << "Height: " << image.rows << endl;

    cv::Mat depth_map = cv::imread(string(argv[1]), cv::IMREAD_ANYDEPTH);
    if (depth_map.empty()) {
        std::cerr << "Error: Could not open or find the depth map" << std::endl;
        return 1;
    }
    cv::imshow("Loaded Depth Map", depth_map);
    cv::waitKey(0);

    cout << "Width : " << depth_map.cols << endl;
    cout << "Height: " << depth_map.rows << endl;

    // open output file
    const string point_cloud_csv = string(argv[2]);
    cout << point_cloud_csv << endl;

    std::ofstream f;
    f.open(point_cloud_csv);
    f << fixed;

    int step = 5;
    for (int i = 0; i < depth_map.rows; i+=step) {
        for (int j = 0; j < depth_map.cols; j+=step) {
            //f << static_cast<int>(image.at<cv::Vec3b>(i, j)[0]) << "," << static_cast<int>(image.at<cv::Vec3b>(i, j)[1]) << "," << static_cast<int>(image.at<cv::Vec3b>(i, j)[2]) << "," << i << "," << j << "," << static_cast<int>(depth_map.at<ushort>(i, j))<< endl;
            f << j << "," << i << "," << static_cast<int>(depth_map.at<ushort>(i, j))<< endl; // x, y, z
        }
    }
    f.close();

    return 0;
}
