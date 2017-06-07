/*
 *  GPUSURFFeatureMatcher.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 6/13/12.
 *
 */

#include "IFeatureMatcher.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

class GPUSURFFeatureMatcher : public IFeatureMatcher {
private:
	cv::Ptr<cv::cuda::SURF_CUDA> extractor;
	std::vector<cv::cuda::GpuMat> descriptors;
	std::vector<cv::cuda::GpuMat> imgs; 
	std::vector<std::vector<cv::KeyPoint> >& imgpts;
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
	bool use_ratio_test;
public:
	//c'tor
	GPUSURFFeatureMatcher(std::vector<cv::Mat>& imgs, std::vector<std::vector<cv::KeyPoint> >& imgpts);
	
	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches = NULL);
	
	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};
