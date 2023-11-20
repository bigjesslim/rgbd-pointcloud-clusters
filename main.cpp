#include <GL/glew.h>
#include <GL/gl.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "dbscan/dbscan.hpp"

#include <iostream>
#include <string>
#include <system_error>
#include <vector>
#include <utility>
#include <fstream>
#include <charconv>
#include <cassert>
#include <tuple>
#include <cstring>
#include <cmath>
#include <time.h>

using namespace std;


template<typename T>
std::vector<std::vector<T>> convertTo2DVector(const std::vector<T>& inputVector, size_t valuesPerSubVector) {
    std::vector<std::vector<T>> outputVector;

    for (size_t i = 0; i < inputVector.size(); i += valuesPerSubVector) {
        std::vector<T> subVector;
        for (size_t j = 0; j < valuesPerSubVector && (i + j) < inputVector.size(); j++) {
            subVector.push_back(inputVector[i + j]);
        }
        outputVector.push_back(subVector);
    }

    return outputVector;
}

template<typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const &vec)
{
    std::vector<T> flattened;
    for (auto const &v: vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

void normalizeColumns(std::vector<std::vector<float>>& data, float& z_weight) {
    // Check if the data is empty or has inconsistent column sizes
    if (data.empty() || data[0].empty()) {
        return; // Handle empty data
    }

    const size_t numColumns = data[0].size();

    for (size_t col = 0; col < numColumns; col++) {
        // Calculate the mean for the current column
        float mean = 0.0;
        for (size_t row = 0; row < data.size(); row++) {
            mean += data[row][col];
        }
        mean /= data.size();

        // Calculate the standard deviation for the current column
        float variance = 0.0;
        for (size_t row = 0; row < data.size(); row++) {
            variance += std::pow(data[row][col] - mean, 2);
        }
        float stddev = std::sqrt(variance / data.size());

        // Normalize the column
        if (col == (numColumns-1)){
            for (size_t row = 0; row < data.size(); row++) {
                data[row][col] = z_weight*((data[row][col] - mean) / (stddev));
            }
        }
        else {
            for (size_t row = 0; row < data.size(); row++) {
                data[row][col] = ((1-z_weight)/2)*((data[row][col] - mean) / (stddev));
            }
        }
    }
}

template<typename T>
auto to_num(const std::string& str)
{
    T value = 0;
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value);

    if(ec != std::errc())
    {
        std::cerr << "Error converting value '" << str << "'\n";
        std::exit(1);
    }
    return value;
}


// noise will be labelled as 0
auto label(const std::vector<std::vector<size_t>>& clusters, size_t n)
{
    auto flat_clusters = std::vector<size_t>(n);

    for(size_t i = 0; i < clusters.size(); i++)
    {
        for(auto p: clusters[i])
        {
            flat_clusters[p] = i + 1;
        }
    }

    return flat_clusters;
}

auto dbscan3d(const std::span<const float>& data, float eps, int min_pts)
{
    auto points = std::vector<point3>(data.size() / 3);

    std::memcpy(points.data(), data.data(), sizeof(float) * data.size());

    auto clusters = dbscan(points, eps, min_pts);
    auto flat     = label(clusters, points.size());
    std::cout << "total number of clusters: " << clusters.size() << endl;
    return flat;
}

int main(int argc, char **argv) {
    if(argc != 5)
    {
        std::cerr << "usage: main <depth and rgb image name> <z weight> <epsilon> <min points> \n";
        return 1;
    }
    //////////////////////////////////////// LOADING IMAGES ////////////////////////////////////////
    // Load depth map into 1D vector (x,y,z)
    string depth_map_path = "../images/depth/" + string(argv[1]) + ".png";
    string rgb_image_path =  "../images/rgb/" + string(argv[1]) + ".png";
    cv::Mat depth_map = cv::imread(depth_map_path, cv::IMREAD_ANYDEPTH);
    if (depth_map.empty()) {
        std::cout << depth_map_path << std::endl;
        std::cerr << "Error: Could not open or find the depth map" << std::endl;
        return 1;
    }

    ////////////////////////////////////// COLLECTING XYZ VALUES /////////////////////////////////
    // Load depth map into 1D vector (x,y,z)
    std::vector<float> values;
    int step = 5; // downsampling by 5
    for (int i = 0; i < depth_map.rows; i+=step) {
        for (int j = 0; j < depth_map.cols; j+=step) {
            int z = static_cast<int>(depth_map.at<ushort>(i, j));
            if (z != 0){
                values.push_back(j); // x
                values.push_back(i); // y
                values.push_back(z); // z
            }
        }
    }
    cout << values.size() << endl;

    //////////////////////////////////////// CLUSTERING ////////////////////////////////////////
    // DBSCAN clustering 
    auto z_weight  = to_num<float>(argv[2]);
    auto epsilon  = to_num<float>(argv[3]);
    auto min_pts  = to_num<int>  (argv[4]);

    std::vector<std::vector<float>> values_2D;
    values_2D = convertTo2DVector(values, 3);
    std::vector<std::vector<float>> points = values_2D;
    normalizeColumns(values_2D, z_weight);
    values = flatten(values_2D);

    clock_t tStart = clock();
    std::vector<size_t> clusters;
    clusters = dbscan3d(values, epsilon, min_pts);
    printf("time taken for clustering: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    /////////////////////////////////////// VISUALISATION //////////////////////////////////////
    // Making and saving 2d visualization
    cv::Mat image = cv::imread(rgb_image_path, cv::IMREAD_UNCHANGED); // loading the PNG image

    if (image.empty()) {
        std::cout << rgb_image_path << std::endl;
        std::cerr << "Error: Unable to load the image." << std::endl;
        return -1;
    }

    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat colorImg;
    cv::cvtColor(grayImage,colorImg,cv::COLOR_GRAY2BGR);

    // Creating cluster colours
    //int max_clusters;
    //max_clusters = std::ceil(std::max_element(clusters.begin(), clusters.end()));
    std::vector<cv::Scalar> cluster_colours;
    srand(5); //  set cluster colours 
    for(int i=0;i<100;i++){
		cv::Scalar colour(rand() % 256, rand() % 256, rand() % 256);
		cluster_colours.push_back(colour);
	}

    // Draw points on the grayscale image
    float font_size = 0.15;
    float font_weight = 0.03;
    cv::Point2f point;
    for (int i=0; i<points.size(); i++) {
        //cv::circle(color_img, points[i], 2, colors[clusters[i]], -1); // White circles
        point = cv::Point2f(points[i][0], points[i][1]);
        cv::putText(colorImg, to_string(int(std::floor(clusters[i]))), point, cv::FONT_HERSHEY_SIMPLEX, font_size, cluster_colours[std::floor(clusters[i])], font_weight);
    }
    //actual_colors[i]
    //colors[clusters[i]]
    cv::imshow("Grayscale Image with Points", colorImg);

    // Saving under the format image_zweight_eps_minpoints
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << z_weight; 
    std::string zweightString = oss.str();
    zweightString.erase(std::remove(zweightString.begin(), zweightString.end(), '.'), zweightString.end());

    oss.str("");
    oss.clear();

    oss << std::fixed << std::setprecision(2) << epsilon; 
    std::string epsilonString = oss.str();
    epsilonString.erase(std::remove(epsilonString.begin(), epsilonString.end(), '.'), epsilonString.end());

    string output_viz_path = "../viz_nobg/" + string(argv[1]) + "_" + zweightString + "_" + epsilonString + "_" + to_string(min_pts)  + ".png";
    cv::imwrite(output_viz_path, colorImg, {cv::IMWRITE_PNG_COMPRESSION, 0});

    /////////////////////////////////////// PREPROCESSING ///////////////////////////////////////
    // Reformat clusters into vector<vector> (using clusters and points)
    std::vector<std::vector<std::vector<int>>> cluster_vectors;
    size_t maxCluster = *std::max_element(clusters.begin(), clusters.end());
    cluster_vectors.resize(maxCluster + 1);

    for (int i = 0; i < clusters.size(); i++){
        std::vector<int> xy_coords = {static_cast<int>(std::round(points[i][0])), static_cast<int>(std::round(points[i][1]))};
        cluster_vectors[clusters[i]].push_back(xy_coords);
    }

    // Create bitmap
    int width = 128;
    int height = 96;

    //cv::Mat binaryImage(height, width, CV_8UC1, cv::Scalar(0)); // CV_8UC1 restricts pixels to 0/1
    cv::Mat grayscaleImage(height, width, CV_8UC1, cv::Scalar(0));
    int num_sections = 0;
    for(int i=1;i<cluster_vectors.size();i++){ // starts from 1 because cluster[0] are the cluster outliers
        int cluster_size = cluster_vectors[i].size();
        if (cluster_size > 50) {
            num_sections ++;
            for (int j=0;j<cluster_size;j++) {
                std::vector<int> xy_coords = cluster_vectors[i][j];
                grayscaleImage.at<uchar>(static_cast<int>(xy_coords[1]/5), static_cast<int>(xy_coords[0]/5)) = 1;
            }

            cv::Mat resizedImage; 
            cv::resize(grayscaleImage, resizedImage, cv::Size(), 5, 5, cv::INTER_NEAREST);

            cv::Mat maskedImage;
            cv::bitwise_and(image, image, maskedImage, resizedImage);

            cv::imshow("Masked image " + to_string(num_sections), maskedImage);

            int key = 0;
            while (true) {
                int key = cv::waitKey(500); 

                if (key == 13) {  // ASCII value for Enter key
                    break;
                }
            }

            maskedImage.setTo(0);
            grayscaleImage.setTo(0);
            resizedImage.setTo(0);
        }
    }
    
    return 0;

}