#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace std;


int main(int argc, char **argv) {
    // Load the image from a file
    if (argc < 3){
        cerr << "format: to_fp_cloud.exe (fp csv) (depth image) (output file name)" << endl;
        return 1;
    }


    cv::Mat depth_map = cv::imread(string(argv[2]), 1);
    if (depth_map.empty()) {
        std::cerr << "Error: Could not open or find the depth map" << std::endl;
        return 1;
    }
    cv::imshow("Loaded Depth Map", depth_map);
    cv::waitKey(0);

    std::ofstream outputFile; // open to write to 3D coords file
    outputFile.open(string(argv[3]));
    outputFile << fixed;
    std::ifstream inputFile(argv[1]); // open feature point csv to read

    if (inputFile.is_open()) {
        std::string line;

        while (std::getline(inputFile, line)) {
            std::istringstream lineStream(line);
            std::string field;
            std::vector<int> coords;

            while (std::getline(lineStream, field, ',')) {
                coords.push_back((int)std::round(std::stod((field))));
            }
            outputFile << coords[0] << "," << coords[1] << "," << static_cast<int>(depth_map.at<ushort>(coords[1], coords[0])) << endl;
        }
        inputFile.close(); 
        outputFile.close();

    } else {
        std::cerr << "Failed to open the CSV file." << std::endl;
    }

    return 0;
}
