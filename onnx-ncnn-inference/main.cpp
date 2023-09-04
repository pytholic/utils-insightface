// #include <algorithm>
#include <stdio.h>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <net.h>
#include <omp.h>
#include "utility/distance.h"
#include "utility/utility.h"

#include "Eigen/core"

int main(int argc, char **argv)
{
  auto image1 = std::vector<std::string>{
      "../test_dataset/Aaron_Peirsol/Aaron_Peirsol_0001.jpg",
      "../test_dataset/Aaron_Sorkin/Aaron_Sorkin_0001.jpg",
      "../test_dataset/Alec_Baldwin/Alec_Baldwin_0001.jpg",
      "../test_dataset/Jake_Gyllenhaal/Jake_Gyllenhaal_0001.jpg",
      "../test_dataset/Jennifer_Garner/Jennifer_Garner_0001.jpg",
  };
  auto image2 = std::vector<std::string>{
      "../test_dataset/Aaron_Peirsol/Aaron_Peirsol_0002.jpg",
      "../test_dataset/Aaron_Sorkin/Aaron_Sorkin_0002.jpg",
      "../test_dataset/Alec_Baldwin/Alec_Baldwin_0002.jpg",
      "../test_dataset/Jake_Gyllenhaal/Jake_Gyllenhaal_0003.jpg",
      "../test_dataset/Jennifer_Garner/Jennifer_Garner_0003.jpg",
  };
  auto features1 = std::vector<FEATURE>{};
  auto features2 = std::vector<FEATURE>{};
  // auto res = std::vector<std::vector<float>>{};

  // Load NCNN model
  ncnn::Net net;
  int ret = net.load_param("../models/ncnn/model_opt.param");
  if (ret)
    std::cerr << "Failed to load model parameters!" << std::endl;
  ret = net.load_model("../models/ncnn/model_opt.bin");
  if (ret)
    std::cerr << "Failed to load model weights!" << std::endl;

  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};

  for (const auto &path : image1)
  {
    // Load image
    cv::Mat orig_img = cv::imread(path, cv::IMREAD_COLOR);
    if (orig_img.empty())
    {
      std::cerr << "Unable to read image file " << path << std::endl;
      return -1;
    }

    auto img = cv::Mat();
    cv::resize(orig_img, img, cv::Size(112, 112));

    // Convert image data to ncnn format
    // Opencv image is bgr, model also expects bgr
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB,
                                             img.cols, img.rows);

    // Preprocess image
    input.substract_mean_normalize(mean_vals, norm_vals);

    // Inference
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(6);
    extractor.input("input", input);
    ncnn::Mat output;
    extractor.extract("output", output);

    // Flatten
    ncnn::Mat outFlattened = output.reshape(output.w * output.h * output.c);
    // std::cout << outFlattened.h;

    // Convert to Eigen matrix
    auto out = ncnn_to_eigen(outFlattened);
    // std::cout << outEigen.rows() << std::endl;

    features1.push_back(out);
  }

  for (const auto &path : image2)
  {
    // Load image
    cv::Mat orig_img = cv::imread(path, cv::IMREAD_COLOR);
    if (orig_img.empty())
    {
      std::cerr << "Unable to read image file " << path << std::endl;
      return -1;
    }

    auto img = cv::Mat();
    cv::resize(orig_img, img, cv::Size(112, 112));

    // Convert image data to ncnn format
    // Opencv image is bgr, model also expects bgr
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB,
                                             img.cols, img.rows);

    // Preprocess image
    input.substract_mean_normalize(mean_vals, norm_vals);

    // Inference
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(6);
    extractor.input("input", input);
    ncnn::Mat output;
    extractor.extract("output", output);

    // Flatten
    ncnn::Mat outFlattened = output.reshape(output.w * output.h * output.c);
    // std::cout << outFlattened.h;

    // Convert to Eigen matrix
    auto out = ncnn_to_eigen(outFlattened);
    // std::cout << outEigen.rows() << std::endl;

    features2.push_back(out);
  }

  for (const auto &feat1 : features1)
  {
    auto tmp = std::vector<float>{};
    for (const auto &feat2 : features2)
    {
      auto distance = Distance{};
      auto similarity = distance.cosine_similarity(feat1, feat2);
      tmp.push_back(similarity);
    }
    for (const float &score : tmp)
    {
      std::cout << score << ", ";
    }
    std::cout << "\n";
  }
}
