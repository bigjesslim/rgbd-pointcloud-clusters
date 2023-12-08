#include <GL/glew.h>
#include <GL/gl.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "dbscan/dbscan.hpp"
#include "src/utils.h"

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
#include <chrono>

using namespace std;

// noise will be labelled as 0

int main(int argc, char **argv) {
    if(argc != 5)
    {
        std::cerr << "usage: main <image name> <z weight> <epsilon> <min points> \n";
        return 1;
    }
    //////////////////////////////////////// LOADING IMAGES ////////////////////////////////////////
    // load images into cv::Mat
    string depth_map_path = "../images/depth/" + string(argv[1]) + ".png";
    string rgb_image_path =  "../images/rgb/" + string(argv[1]) + ".png";
    cv::Mat depth_map = cv::imread(depth_map_path, cv::IMREAD_ANYDEPTH);
    if (depth_map.empty()) {
        std::cout << depth_map_path << std::endl;
        std::cerr << "Error: Could not open or find the depth map" << std::endl;
        return 1;
    }

    // 1oad depth map into 1D vector (x,y,z)
    const std::vector<float> values = toXYZ(depth_map, 5);

    // convert depth map into 2D vector (x,y,z) [H,W]
    auto values_2D = convertTo2DVector(values, 3);
    auto points = values_2D;

    // normalize and cluster data (dbscan) 
    std::vector<size_t> clusters = normalizeAndCluster(argv[2], argv[3], argv[4], values_2D);

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
        int min_x = 640; 
        int max_x = 0;
        int min_y = 480;
        int max_y = 0;
        if (cluster_size > 50) {
            num_sections ++;
            for (int j=0;j<cluster_size;j++) {
                std::vector<int> xy_coords = cluster_vectors[i][j];
                int y = xy_coords[1];
                int x = xy_coords[0];

                if (y > max_y){
                    max_y = y;
                }
                if (y < min_y){
                    min_y = y;
                }
                if (x > max_x){
                    max_x = x;
                }
                if(x < min_x){
                    min_x = x;
                }

                grayscaleImage.at<uchar>(static_cast<int>(xy_coords[1]/5), static_cast<int>(xy_coords[0]/5)) = 1;
            }

            // get, show and save masked image
            cv::Mat resizedImage; 
            cv::resize(grayscaleImage, resizedImage, cv::Size(), 5, 5, cv::INTER_NEAREST);
            cv::Mat maskedImage;
            cv::bitwise_and(image, image, maskedImage, resizedImage);
            //cv::imshow("Masked Image " + to_string(num_sections), maskedImage);
            string output_mi_path = "../masked_images/" + string(argv[1]) + "_" + to_string(num_sections)  + ".png";
            //cv::imwrite(output_mi_path, maskedImage,{cv::IMWRITE_PNG_COMPRESSION, 0});

            // get, show and save cropped image
            int height = max_y - min_y;
            int width = max_x - min_x;
            cout << "min_x: " << min_x << ", min_y: " << min_y << endl;
            cout << "width: " << width << ", height: " << height << endl;
            cv::Mat croppedImage = image(cv::Rect(min_x, min_y, width, height));
            // cv::imshow("Cropped Image " + to_string(num_sections), croppedImage);
            string output_ci_path = "../cropped_images/" + string(argv[1]) + "_" + to_string(num_sections)  + ".png";
            //cv::imwrite(output_ci_path, croppedImage,{cv::IMWRITE_PNG_COMPRESSION, 0});

            // get, show and save masked + cropped image
            cv::Mat maskedCroppedImage = maskedImage(cv::Rect(min_x, min_y, width, height));
            //cv::imshow("Masked and Cropped Image " + to_string(num_sections), maskedCroppedImage);
            string output_mci_path = "../mc_images/" + string(argv[1]) + "_" + to_string(num_sections)  + ".png";
            //cv::imwrite(output_mci_path, maskedCroppedImage,{cv::IMWRITE_PNG_COMPRESSION, 0});


            while (true) {
                int key = cv::waitKey(500); 

                if (key == 13) {  // ASCII value for Enter key
                    break;
                }
            }

            // save masked images
            maskedImage.setTo(0);
            grayscaleImage.setTo(0);
            resizedImage.setTo(0);
        }
    }
    
    return 0;

}