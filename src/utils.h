#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include "../dbscan/dbscan.hpp"
#include <torch/torch.h>
#include <torch/script.h>

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

std::vector<float> toXYZ(cv::Mat depth_map, int step=5);

template<typename T> std::vector<T> flatten(std::vector<std::vector<T>> const &vec);

template<typename T> std::vector<std::vector<T>> convertTo2DVector(const std::vector<T>& inputVector, size_t valuesPerSubVector);

void normalizeColumns(std::vector<std::vector<float>>& data, float& z_weight);

template<typename T> auto to_num(const std::string& str);

auto label(const std::vector<std::vector<size_t>>& clusters, size_t n);

auto dbscan3d(const std::span<const float>& data, float eps, int min_pts);

std::vector<size_t> normalizeAndCluster(const std::string& z_weight_str, const std::string& epsilon_str, const std::string& min_points_str, std::vector<std::vector<float>> xyz_values_2D);

cv::Mat paddingAndResize(const cv::Mat& image, const cv::Size& size);

torch::Tensor padTensor(const torch::Tensor& input_tensor);

#endif // UTILS_H