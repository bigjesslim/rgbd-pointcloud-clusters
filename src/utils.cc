#include "utils.h"


template<typename T> std::vector<std::vector<T>> convertTo2DVector(const std::vector<T>& inputVector, size_t valuesPerSubVector) {
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

template std::vector<std::vector<float>> convertTo2DVector(const std::vector<float>& inputVector, size_t valuesPerSubVector);

std::vector<float> toXYZ(cv::Mat depth_map, int step){
    std::vector<float> values;
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
    return values;
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
    std::cout << "total number of clusters: " << clusters.size() << std::endl;
    return flat;
}

std::vector<size_t> normalizeAndCluster(const std::string& z_weight_str, const std::string& epsilon_str, const std::string& min_points_str, std::vector<std::vector<float>> xyz_values_2D)
{
    float z_weight  = to_num<float>(z_weight_str);
    float epsilon  = to_num<float>(epsilon_str);
    int min_pts  = to_num<int>(min_points_str);

    normalizeColumns(xyz_values_2D, z_weight);
    std::vector<float> values = flatten(xyz_values_2D);

    std::vector<size_t> clusters = dbscan3d(values, epsilon, min_pts);

    return clusters;

}

cv::Mat paddingAndResize(const cv::Mat& image, const cv::Size& size) {
    int height = image.size[0];
    int width = image.size[1];

    // std::cout << "height and width." << std::endl;
    // std::cout << std::to_string(image.size[0]) << std::endl;
    // std::cout << std::to_string(image.size[1]) <<std::endl;

    cv::Mat padded_img;

    if (width > height) {
        // pad vertically
        int total_pad_value = width - height;
        int top_pad_value = total_pad_value / 2;
        int bottom_pad_value = total_pad_value - top_pad_value;
        cv::copyMakeBorder(image, padded_img, top_pad_value, bottom_pad_value, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
    } else if (height > width) {
        // pad horizontally
        std::cout << "horizontal padding" << std::endl;
        int total_pad_value = height - width;
        int left_pad_value = total_pad_value / 2;
        int right_pad_value = total_pad_value - left_pad_value;
        cv::copyMakeBorder(image, padded_img, 0, 0, left_pad_value, right_pad_value, cv::BORDER_CONSTANT, cv::Scalar(0));
    } else {
        padded_img = image.clone();
    }
    cv::Mat resized_image;
    cv::resize(padded_img, resized_image, size);

    return resized_image;
}

torch::Tensor padTensor(const torch::Tensor& input_tensor) {
    int original_height = input_tensor.size(2);
    int original_width = input_tensor.size(3);

    // Calculate the padding needed on both sides 
    int padding_updown = (original_width - original_height)/2; // assumes landscape photo

    // Pad the tensor to achieve the target width
    torch::Tensor padded_tensor = torch::nn::functional::pad(
        input_tensor,
        torch::nn::functional::PadFuncOptions({0, 0, padding_updown, padding_updown}).mode(torch::kConstant).value(0)
    );


    torch::Tensor resized_tensor = torch::nn::functional::interpolate(
        padded_tensor, 
        torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64>{240,240}).mode(torch::kNearest)
    );

    return resized_tensor;
}
