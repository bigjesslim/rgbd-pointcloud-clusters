#include <GL/glew.h>
#include <GL/gl.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
using namespace std;


int main(int argc, char **argv) {

    if (argc < 3){
        cerr << "format: viz_cluster.exe (coords csv) (cluster data txt) (rgb image png)" << endl;
        return 1;
    }

    std::string csvname = argv[1];
	std::string txtname = argv[2];
    std::string imgname = argv[3];

    // Load the PNG image
    cv::Mat image = cv::imread(imgname, cv::IMREAD_UNCHANGED);

    if (image.empty()) {
        std::cerr << "Error: Unable to load the image." << std::endl;
        return -1;
    }

    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Get coords from csv and store into dataset var
    std::ifstream file(csvname);
    std::vector<cv::Point2f> points;
    std::string line;
    //std::vector<cv::Scalar> actual_colors;

    while (std::getline(file, line)) {
        std::vector<int> row;
        std::istringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stoi(cell));
        }

        points.push_back(cv::Point2f(row[0], row[1]));
        //cv::Scalar actual_color(row[2], row[1], row[0]);
        //actual_colors.push_back(actual_color);
    }
    file.close();

    // Getting cluster colors
    std::vector<cv::Scalar> colors;
    srand(5); //  set good cluster colours 
    for(int i=0;i<300;i++){
		cv::Scalar eigen_colour(rand() % 256, rand() % 256, rand() % 256);
		colors.push_back(eigen_colour);
	}

    // Getting clusters
    std::ifstream txt_file(txtname);
    std::vector<int> clusters;
    while (std::getline(txt_file, line)) {
        clusters.push_back(std::stoi(line));
    }
    txt_file.close();

    cv::Mat color_img;
    cv::cvtColor(grayImage,color_img,cv::COLOR_GRAY2BGR);

    // Draw points on the grayscale image

    float font_size = 0.2;//Declaring the font size//
    float font_weight = 0.05;
    for (int i=0; i<points.size(); i++) {
        //cv::circle(color_img, points[i], 2, colors[clusters[i]], -1); // White circles
        cv::putText(color_img, to_string(clusters[i]), points[i], cv::FONT_HERSHEY_SIMPLEX, font_size, colors[clusters[i]], font_weight);
    }
    //actual_colors[i]
    //colors[clusters[i]]


    // Show the grayscale image with points
    cv::imshow("Grayscale Image with Points", color_img);
    cv::waitKey(0);
    return 0;
}