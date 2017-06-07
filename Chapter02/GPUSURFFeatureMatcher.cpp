/*
 *  GPUSURFFeatureMatcher.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 6/13/12.
 *
 */

#include "GPUSURFFeatureMatcher.h"

#include "FindCameraMatrices.h"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <set>

using namespace std;
using namespace cv;

//c'tor
GPUSURFFeatureMatcher::GPUSURFFeatureMatcher(vector<Mat>& imgs_, vector<vector<KeyPoint> >& imgpts_) :
	imgpts(imgpts_), use_ratio_test(true)
{
	// The helper function printShortCudaDeviceInfo() moved between OpenCV v2.3 and v2.4, so might not compile.
	//printShortCudaDeviceInfo(cv::gpu::getDevice());

	cuda::SURF_CUDA extractor;
	matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

	cout << " -------------------- extract feature points for all images (GPU) -------------------\n";
	
	cout << "imgpts has " << imgpts.size() << " points (descriptors " << descriptors.size() << ")" << endl;

	imgpts.resize(imgs_.size());
	descriptors.resize(imgs_.size());

	cout << "imgpts has " << imgpts.size() << " points (descriptors " << descriptors.size() << ")" << endl;

	CV_PROFILE("extract",
	for(int img_i=0; img_i<imgs_.size(); img_i++) {
		cuda::GpuMat _m; _m.upload(imgs_[img_i]);
		extractor(_m, cuda::GpuMat(), imgpts[img_i], descriptors[img_i]);
		cout << ".";
	})
}	

void GPUSURFFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches) {
	
#ifdef __SFM__DEBUG__
	Mat img_1; imgs[idx_i].download(img_1);
	Mat img_2; imgs[idx_j].download(img_2);
#endif
//	const vector<KeyPoint>& imgpts1 = imgpts[idx_i];
//	const vector<KeyPoint>& imgpts2 = imgpts[idx_j];
	const cuda::GpuMat descriptors_1 = descriptors[idx_i];
	const cuda::GpuMat descriptors_2 = descriptors[idx_j];

	cuda::GpuMat d_matches;
    vector<DMatch> matches_, good_matches_, very_good_matches_;
    vector<vector<DMatch> > knn_matches;
    
//	vector<KeyPoint> keypoints_1, keypoints_2;
//	cout << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << endl;
//    cout << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << endl;
	
//	keypoints_1 = imgpts1;
//	keypoints_2 = imgpts2;
	
    cout << "match " << descriptors_1.rows << " vs. " << descriptors_2.rows << " ..." << endl;
	if (descriptors_1.empty()) {
		CV_Error(0, "descriptors_1 is empty");
	}
	if (descriptors_2.empty()) {
		CV_Error(0, "descriptors_2 is empty");
	}
	
	if (matches == NULL) {
		matches = &matches_;
	}
	if (matches->size() == 0) {
		if (use_ratio_test) {
			CV_PROFILE("match",	
                matcher->knnMatchAsync(descriptors_1, descriptors_2, d_matches, 2);
                matcher->knnMatchConvert(d_matches, knn_matches);
            )

			(*matches).clear();

			//ratio test
			for (int i=0; i<knn_matches.size(); i++) {
				if (knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) {
					(*matches).push_back(knn_matches[i][0]);
				}
			}
	        cout << " d_matches = " << d_matches.size() << "," << " knn_matches = " << knn_matches.size() << endl;
			cout << "kept " << (*matches).size() << " features after ratio test"<<endl;
		} else {
			CV_PROFILE("match", matcher->matchAsync(descriptors_1, descriptors_2, *matches);)
		}
    }
}
