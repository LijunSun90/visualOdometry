// visualOdometry.cpp
// Calling convention:
// ./visualOdometry [frames_directory]
// ./visualOdometry ~/data/KITTI_odometry_gray/dataset/

#include <istream>
#include <ostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "ceres/ceres.h"

using namespace std;

void help(char* argv[]){
}

/**
 * @brief Obtain the 3x4 projection matrix after rectification
 *        from the KITTI odometry dataset.
 * @param file_name: the calibration file to be parsed.
 * @param PO: for camera0, the left gray camera.
 * @param P1: for camera1, the right gray camera.
 * @param P2: for camera2, the left color camera.
 * @param P3: for camera3, the right gray camera.
 */
void parserCalib(string file_name , cv::Mat & Q){

    // Obtain the 3x4 projection matrix after rectification.
    // PO: for camera0, the left gray camera.
    // P1: for camera1, the right gray camera.
    // P2: for camera2, the left color camera.
    // P3: for camera3, the right gray camera.
    
    cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P3 = cv::Mat::zeros(3, 4, CV_64F);

    ifstream loaded_projection_matrices(file_name.c_str());
    if(!loaded_projection_matrices.is_open()){
        cout << "ERROR: cannot open the file " + file_name << endl;
        exit(-1);
    }
    string lines, entry;
    istringstream linestream;
    // P0. 0 represents the left grayscale camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            // cout << entry << endl;
            P0.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P0 << endl;
    // P1. 1 represents the right grayscale camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P1.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P1 << endl;
    // P2. 2 represents the left color camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P2.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P2 << endl;
    // P3. 3 represents the right color camera.
    //
    getline(loaded_projection_matrices, lines);
    linestream.clear();
    linestream.str(lines);
    linestream >> entry;
    // cout << entry << endl;
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 4; col++){
            linestream >> entry;
            P3.at<double>(row, col) = atof(entry.c_str());
        }
    }
    // cout << P3 << endl;
    // cout << "P3.type(): " << P3.type() << endl;    
    //
    // Get the camera intrinsics (cx, cy, f)
    // and the extrinsics tx in (tx, ty, tz).
    // Here, we took the baseline between the left and right
    // stereo cameras as tx.
    double cx, cy, f, baseline, baselinePoint0, baselinePoint1;
    
    cx = P0.at<double>(0, 2);
    cy = P0.at<double>(1, 2);
    f = P0.at<double>(0, 0);
    baselinePoint0 = P0.at<double>(0, 3) / (-f);
    baselinePoint1 = P1.at<double>(0, 3) / (-f);
    baseline = baselinePoint1 - baselinePoint0;

    cout << "\ncx: " << cx << "\n"
    << "cy: " << cy << "\n"
    << "f: " << f << "\n"
    << "baselinePoint0: " << baselinePoint0 << "\n"
    << "baselinePoint1: " << baselinePoint1 << "\n"
    << "baseline: " << baseline << endl;

    // Build the reprojection matrix Q.
    Q = cv::Mat::eye(4, 4, CV_64F);
    Q.at<double>(0, 3) = -cx;
    Q.at<double>(1, 3) = -cy;
    Q.at<double>(2, 3) = f;
    Q.at<double>(3, 3) = 0;
    Q.at<double>(3, 2) = -1/baseline;

    cout << "\nQ: \n" << Q << "\n" << endl;

} // END OF void parserCalib().


/**
 * @param image
 * @param keypoints
 * @param descriptors
 */
void calculateFeatures(
    cv::Mat image,
    vector< cv::KeyPoint > & keypoints,
    cv::Mat & descriptors
){
    // Detector definition.
    //    
    // cv::Ptr<cv::Feature2D> f2d = cv::ORB::create(1500);
    // cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    // cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create();
    // cv::Ptr<cv::Feature2D> f2d = cv::BRISK::create();

    // Detect the keypoints & Extract descriptors.
    // f2d->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    // cout << "keypoints: " << keypoints.size() << endl;
    // cout << "descriptors: " << descriptors.type() << endl;

    // cv::Ptr<cv::Feature2D> f2d = cv::FastFeatureDetector::create();
    cv::Ptr<cv::Feature2D> f2d = cv::GFTTDetector::create(
        1000,
        0.01,
        1.0,
        3,
        false);
    f2d->detect(image, keypoints);
    // cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    // cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::FREAK::create();
    extractor->compute(image, keypoints, descriptors);

    // Display.
    //
    cv::Mat image_out;
    cv::drawKeypoints(
        image,
        keypoints,
        image_out,
        cv::Scalar(0, 255, 0),
        cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("The_left_image", image_out);
    cv::moveWindow("The_left_image", 0, 660);
    // cv::imwrite("features.jpg", image_out);
    // if(cv::waitKey(0) == 27) exit(0);

}


/**
 * Correspond the found features in the images.
 * @return
 * @param descriptors_1
 * @param descriptors_2
 * @param matches, correspondence between descriptors_1 and descriptors_2.
*/
void matchFeatures(
    cv::Mat descriptors_1,
    cv::Mat descriptors_2, 
    vector < cv::DMatch > & matches
) {

    cv::Mat image_out;

    // Step 3: Matching descriptor vectors.
    cv::FlannBasedMatcher matcher_flann;

    if(descriptors_1.type()!=CV_32F) {
        descriptors_1.convertTo(descriptors_1, CV_32F);
    }
    if(descriptors_2.type()!=CV_32F) {
        descriptors_2.convertTo(descriptors_2, CV_32F);
    }
    matcher_flann.match(descriptors_1, descriptors_2, matches);
    // cout << "matches queryIdx: " << matches.at(0).queryIdx
    // << "\nmatches trainIdx: " << matches.at(0).trainIdx
    // << "\nmatches imgIdx: " << matches.at(0).imgIdx
    // << endl;

} // End of matchFeatures.


void calculateCorners(
    cv::Mat image,
    vector< cv::Point2f > & keypoints
){

    const int MAX_CORNERS = 4000;
    cv::goodFeaturesToTrack(
        image, // Image to track
        keypoints, // Vector of detected corners (output)
        MAX_CORNERS, // Keep up to this many corners
        0.001, // Quality level (percent of maximum)
        5, // Min distance between corners
        cv::noArray(), // Mask,
        3, // Block size
        false, // true: Harris, false: Shi-Tomasi
        0.04 // method specific parameter
    );

}


void calculateOpticalFlow(
    cv::Mat image_former,
    cv::Mat image_new,
    vector< cv::Point2f > & keypoints_former,
    vector< cv::Point2f > & keypoints_new,
    vector<uchar> & features_found
){
    if(keypoints_former.size() < 3000){
        keypoints_former.clear();
        calculateCorners(image_former, keypoints_former);
    }

    int win_size = 10;
    cv::calcOpticalFlowPyrLK(
        image_former, // Previous image
        image_new, // Next image
        keypoints_former, // Previous set of corners (from imgA)
        keypoints_new, // Next set of corners (from imgB)
        features_found, // Output vector, elements are 1 for tracked
        cv::noArray(), // Output vector, lists errors (optional)
        cv::Size(win_size*2+1, win_size*2+1), // Search window size
        5, // Maximum pyramid level to construct
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
            50, // Maximum number of iterations
            0.3 // Minimum change per iteration
        )
    );

} // END OF calculateOpticalFlow.


void filterCorners(
    vector< cv::Point2f > & keypoints_former,
    vector< cv::Point2f > & keypoints_new,
    vector<uchar> & features_found    
){
    vector< cv::Point2f >::iterator iter_keypoints = keypoints_former.begin();
    for(vector<uchar>::iterator iter = features_found.begin();
        iter < features_found.end(); iter++, iter_keypoints++){
        
        if((*iter) == false){
            keypoints_former.erase(iter_keypoints);
            keypoints_new.erase(iter_keypoints);
            features_found.erase(iter);

            // Update.
            iter--;
            iter_keypoints--;
        }

    } // END OF for().

} // END OF filterCorners().


/**
 * @param image_1, the left RECTIFIED stereo image.
 * @param image_2, the right RECTIFIED stereo image.
 * @param Q, the 4x4 reprojection matrix, which maps the 2-D coordiantes
 *           withe the disparity (x, y, d) to the 3-D 3-D coordinates 
 *           in the physical world (X/W, Y/W, Z/W).
 * @param image_3D, (X/W, Y/W, Z/W),3-D coordinates in the physical world.
 */
void triangulateStereo(
    string image_1_name,
    string image_2_name,
    cv::Mat Q,
    cv::Mat & image_1,
    cv::Mat & image_3D
){

    // cout << "triangulateStereo" << endl;

    cv::Mat image_2;
    image_1 = cv::imread(image_1_name, cv::IMREAD_GRAYSCALE);
    image_2 = cv::imread(image_2_name, cv::IMREAD_GRAYSCALE);

    // Step 1: Disparity.
    //         using the stero correspondence.
    //
    // cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
    //     -64, 128, 11, 0, 0, 12, 
    //     0, 15, 1000, 16,
    //     cv::StereoSGBM::MODE_SGBM // cv::StereoSGBM::MODE_HH
    // );
    //
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0, // minDisparity
        64, // numDisparities, total number of distinct possible disparities
             // to be returned. 
        5, // blockSize, usually 3 or 5 would be enough. Always be odd.
            // the larger the value is, the fewer false matches you 
            // are likely to find. However, not only the computational
            // cost scale with the area of the window, but it implicitly
            // assume that the disparity is actually the same over the
            // area of the windlow. Also, with larger blockSize, it is
            // likely to get vaguer depth maps.
        0, // P1
        0, // P2
           // Leave P1 and P2 as zeros and make the implementation 
           // compute some optimal values for them based on the image
           // resolution and blockSize.
        0, // disp12MaxDiff
        1, // preFilterCap, a positive numeric limit.
        15, // uniquessRatio, typical values: 5 ~ 15.
            // speckleWindowSize and speckleRange work together.
        11, // speckleWindowSize, the size of any small, isolated blobs 
            // that are substantially different from their surrounding values.
        1,  // speckleRange, the largest difference between disparities
            // that will include in the same blob.
            // This value is compared directly to the values of the disparity,
            // then this value will, in effect, be multiplied by 16.
            // Typically, a small number, 1 or 2 work most of the time,
            // though values as large as 4 are not uncommon.
        cv::StereoSGBM::MODE_SGBM // mode
                                  // the five-directional version.
    );
    
    // 
    cv::Mat disparity, vdisparity, disparity_true;
    // Output 16-bit fixed-point disparity map.
    //
    stereo->compute(image_1, image_2, disparity);
    // cout << "disparity1: " <<  disparity.at<double>(0, 0) << endl;
    // cout << "disparity2: " <<  disparity.at<double>(300, 100) << endl;
    // cout << "disparity: " << disparity << endl;

    //Get the true disparity values.
    // cout << "disparity type: " << disparity.type() << endl;
    disparity.convertTo(disparity_true, CV_32F, 1.0/16.0, 0.0);

    // This is for displaying the disparity image only.
    cv::normalize(disparity_true, vdisparity, 0, 256, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("Disparity_map", vdisparity);
    // cv::moveWindow("Disparity_map", 670, 0);
    // cv::imwrite("disparityMap.jpg", vdisparity);
    // if(cv::waitKey(0) == 27) exit(0);

    // Step 2: Depth map.
    cv::reprojectImageTo3D(
        disparity_true,
        image_3D,
        Q,
        true,
        -1
    );

    // Mark the outliers.
    double count = 0;
    for(int row = 0; row < disparity_true.size().height; row++){
        for(int col = 0; col < disparity_true.size().width; col++){
            if(disparity_true.at<double>(row, col) < 0){
                // cout << "Original outlier depth: "
                // << image_3D.at<cv::Vec3f>(row, col)(2) << endl;
                image_3D.at<cv::Vec3f>(row, col)(2) = 10000;
                count++;
            }
        }
    }
    cout << "The ratio of negative disparities is: " 
    << count/(disparity_true.rows * disparity_true.cols) 
    << endl;

    // if(cv::waitKey(1) == 27) exit(0);

} // END OF triangulateStereo().


/**
 * @param keypoints
 * @param image_3D, (X/W, Y/W, Z/W),3-D coordinates in the physical world.
 * @param keypoints3D, the collection of (X/W, Y/W, Z/W) in the order of
 *                     the keypoints.
 */
void getEffective3Dkeypoints(
    vector< cv::KeyPoint > & keypoints,
    cv::Mat & descriptors,
    cv::Mat image_3D,
    vector< cv::Vec3f > & keypoints3D
){
    
    // cout << "keypoints Before: " << keypoints.size() << endl;
    // cout << "descriptors Before: " << descriptors.size() << endl;

    cv::Vec3f point3D;
    int keypoints_sz = keypoints.size(); 
    size_t ix = 0;
    int descriptors_rows = descriptors.size().height;
    int descriptors_cols = descriptors.size().width;

    for(vector< cv::KeyPoint >::iterator iter = keypoints.begin();
        iter < keypoints.end();
        ix++, iter++){
        // Find the corresponding 3D coordinates of the specific
        // keypoints[ix].pt.
        point3D = image_3D.at<cv::Vec3f>(keypoints[ix].pt);
        
        // Filter out the outliers whose depth value equals to 10000.
        //
        if(point3D(2) == 10000){
            // Delete the corresponding row of keypoints.
            //
            keypoints.erase(iter);

            // Delete the corresponding row ofdescriptors.
            //
            cv::Mat descriptors_temp = cv::Mat::zeros(
                descriptors_rows-1, descriptors_cols, CV_32F);
            // cout << "descriptors_temp Before: " << descriptors_temp.at<double>(0) << endl;                
            // Delete the row ix from descripteros.
            descriptors.rowRange(0, ix).copyTo(descriptors_temp.rowRange(0, ix));
            descriptors.rowRange(ix+1, descriptors_rows).copyTo(descriptors_temp.rowRange(ix, descriptors_rows-1));
            // cout << "descriptors_temp: " << descriptors_temp.at<double>(0) << endl;
            
            // Update.
            //
            descriptors_rows--;
            iter--;
            ix--;
            descriptors = descriptors_temp.clone();
            // cout << "descriptors: " << descriptors.at<double>(0) << endl;
        } else{
             keypoints3D.push_back(point3D);
        }
        
    }
    
    // cout << "keypoints: " << keypoints.size() << endl;
    // cout << "descriptors: " << descriptors.size() << endl;
    cout << "The number of effective keypoints3D is: " <<  keypoints3D.size() << endl;

} // END OF get3Dkeypoints().


void getEffective3DCorners(
    vector< cv::Point2f > & keypoints_former,
    vector< cv::Point2f > & keypoints_new,
    cv::Mat image_3D_former,
    cv::Mat image_3D_new,
    vector< cv::Vec3f > & keypoints3D_former,
    vector< cv::Vec3f > & keypoints3D_new
){
    
    cout << "Corners Before: " << keypoints_former.size() << endl;
    // cout << "Corners_new Before: " << keypoints_new.size() << endl;

    cv::Vec3f point3D_former, point3D_new;
    int keypoints_sz = keypoints_former.size(); 
    size_t ix = 0;
    int count = 0;

    for(vector< cv::Point2f >::iterator iter = keypoints_former.begin();
        iter < keypoints_former.end();
        ix++, iter++){
        // Find the corresponding 3D coordinates of the specific
        // keypoints[ix].pt.
        point3D_former = image_3D_former.at<cv::Vec3f>(keypoints_former[ix]);
        point3D_new = image_3D_new.at<cv::Vec3f>(keypoints_new[ix]);
        // cout << "point3D_former: " << point3D_former << endl;
        // cout << "point3D_new: " << point3D_new << endl;
        

        // Filter out the outliers whose depth value equals to 10000.
        // || (point3D_new(2) == 10000) 
        if( (point3D_former(2) == 10000) || (point3D_new(2) == 10000) ) {
            // Delete the corresponding row of keypoints.
            //
            keypoints_former.erase(iter);
            keypoints_new.erase(iter);
            
            // Update.
            //
            iter--;
            ix--;
            
            // cout << "descriptors: " << descriptors.at<double>(0) << endl;
        } else{
             keypoints3D_former.push_back(point3D_former);
             keypoints3D_new.push_back(point3D_new);
        }
        
        // // Update.
        // count++;
        // cout << "count: " << count << endl;

    }
    
    cout << "Corners After: " << keypoints_former.size() << endl;
    // cout << "descriptors: " << descriptors.size() << endl;
    cout << "The number of effective 3Dcorners is: " <<  keypoints3D_former.size() << endl;

} // END OF getEffective3DCorners().


/**
 * Sort keypoints3D_former and keypoints3D_new in the order
 * according to @param matches.
 */
void sortKeypoints3D(
    vector< cv::Vec3f > & keypoints3D_former, 
    vector< cv::Vec3f > & keypoints3D_new,
    vector < cv::DMatch > matches
){
    // cout << "keypoints3D_new Before: " << keypoints3D_new[30] << endl;

    vector< cv::Vec3f > keypoints3D_former_temp, keypoints3D_new_temp;
    for(size_t ix = 0; ix < matches.size(); ix++){

        keypoints3D_former_temp.push_back(keypoints3D_former[matches[ix].queryIdx]);
        keypoints3D_new_temp.push_back(keypoints3D_new[matches[ix].trainIdx]);

    } // END OF for(size_t ix = 0; ix < matches.size(); ix++).

    // Update.
    //
    keypoints3D_former = keypoints3D_former_temp;
    keypoints3D_new = keypoints3D_new_temp;
    // cout << "keypoints3D_temp: " << keypoints3D_temp[30] << endl;
    // cout << "keypoints3D_new: " << keypoints3D_new[30] << endl;

} // END OF sortKeypoints3D().



void getRelativeTransform(
    vector< cv::Vec3f > keypoints3D_former,
    vector< cv::Vec3f > keypoints3D_new,
    vector < cv::DMatch > matches,
    cv::Mat image_3D_former,
    cv::Mat image_3D_new,
    cv::Mat & R_relative,
    cv::Mat & t_relative
){
    
    // Sort keypoints3D_former and keypoints3D_new in the order
    // according to matches.
    //
    sortKeypoints3D(keypoints3D_former, keypoints3D_new, matches);

    // Calculate the relative 3D affine transformation.
    //
    cv::Mat affine_relative_3D;
    cv::estimateAffine3D(keypoints3D_former, keypoints3D_new, affine_relative_3D, cv::noArray(), 1, 0.999);
    cout << "\naffine_relative_3D: \n" << affine_relative_3D << endl;
    R_relative = affine_relative_3D.colRange(0, 3);
    t_relative = affine_relative_3D.col(3);

    // cv::Mat objectPoints = cv::Mat::zeros(keypoints3D_former.size(), 1, CV_8UC3);
    // cv::Mat imagePoints = cv::Mat::zeros(keypoints3D_new.size(), 1, CV_8UC3);
    // for(size_t ix = 0; ix < keypoints3D_new.size(); ix++){
    //     objectPoints.at<cv::Vec3f>(ix) = keypoints3D_former[ix];
    //     imagePoints.at<cv::Vec3f>(ix) = keypoints3D_new[ix];
    // }

    // cv::solvePnPRansac(
    //     objectPoints,
    //     imagePoints,
    //     cv::noArray(),
    //     cv::noArray(),
    //     R_relative,
    //     t_relative
    // );
    

    // cout << "\nR_relative: \n" << R_relative
    // cout << "\nt_relative: \n" << t_relative
    // << endl;

}


/**
 * @param Rs, the collection of the total rotation matrix to the reference.
 * @param ts, the collection of the total translation matrix to the reference.
 */
void visualizeTrajectoryXZ(cv::Mat & trajectory, cv::Mat R, cv::Mat t, int frame = 0){
   
    // trajectory: Horizontal: x; Vertical: z.  
    cv::Point pose_current;

    pose_current = cv::Point(t.at<double>(0) + 300, t.at<double>(2) + 100);
    cv::circle(trajectory, pose_current, 1, cv::Scalar(0, 0, 255), 1, CV_FILLED);
    
    // cout << "Show trajectory" << endl;
    string trajectory_name = 
        "Trajectory: Blue, ground truth poses; Red, estimated poses.";
    cv::imshow(trajectory_name, trajectory);
    cv::moveWindow(trajectory_name, 0, 0);
    if(cv::waitKey(1) == 27){
        cv::imwrite("trajectory.jpg", trajectory);
        exit(0);
    }

} // END OF visualizeTrajectory().



/**
 * @param dir_name, the name of the directory containing the
 *                  RECTIFIED stereo images.
 * @param Q, the 4x4 reprojection matrix, which maps the 2-D coordiantes
 *           withe the disparity (x, y, d) to the 3-D 3-D coordinates 
 *           in the physical world (X/W, Y/W, Z/W).
 * @param Rs, the collection of the total rotation matrix to the reference.
 * @param ts, the collection of the total translation matrix to the reference.
 */
void getTrajectory(
    string dir_name,
    cv::Mat Q,
    vector< cv::Mat > & Rs,
    vector< cv::Mat > & ts
){

    cv::FileStorage fs("poses.xml", cv::FileStorage::WRITE);
    int frame = 0;

    // Open the stereo left and right RECTIFIED images files and check.
    // image_1 correspond to the left RECTIFIED stereo image.
    // image_2 correspond to the right RECTIFIED stereo image.
    //
    string image_1_names_file = dir_name + "/sequences/00/image_0.txt";
    string image_2_names_file = dir_name + "/sequences/00/image_1.txt";

    ifstream image_1_names(image_1_names_file.c_str());
    if(!image_1_names.is_open()){
      cout << "ERROR: Cannot open the file "
        << image_1_names_file << endl;
      exit(-1);
    }
    
    ifstream image_2_names(image_2_names_file.c_str());
    if(!image_2_names.is_open()){
      cout << "ERROR: Cannot open the file "
        << image_2_names_file << endl;
      exit(-1);
    }

    string image_1_name, image_2_name;
    cv::Mat image_1_former, image_1_new;
    cv::Mat image_3D_former, image_3D_new;

    // Find the features, and match them between 
    // the former and the new frames of the left camera.
    // 
    vector< cv::KeyPoint > keypoints_former, keypoints_new;
    // vector<cv::Point2f> keypoints_former, keypoints_new;
    cv::Mat descriptors_former, descriptors_new;
    vector < cv::DMatch > matches;

    // Get the most first frame.
    //
    getline(image_1_names, image_1_name);
    image_1_name =  dir_name + "/sequences/00/" + image_1_name;

    getline(image_2_names, image_2_name);
    image_2_name = dir_name + "/sequences/00/" +   image_2_name;

    triangulateStereo(image_1_name, image_2_name, Q, image_1_former, image_3D_former);
    cout << "Image size: " << image_1_former.size() << endl;

    // Feature detector + descriptor extractor + matcher.
    //
    calculateFeatures(image_1_former, keypoints_former, descriptors_former);
    vector< cv::Vec3f > keypoints3D_former, keypoints3D_new;
    getEffective3Dkeypoints(
        keypoints_former, 
        descriptors_former,
        image_3D_former, 
        keypoints3D_former);

    // Harris-corners + optical Flow.
    //
    // calculateCorners(image_1_former, keypoints_former);

    // Get the total transformation with respect to the reference.
    // 
    cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat R_relative_new, t_relative_new;
    cv::Mat affine_3D = cv::Mat::zeros(3, 4, CV_64F);
    // Save.
    R_total.copyTo(affine_3D.colRange(0, 3));
    t_total.copyTo(affine_3D.col(3));
    // cout << "R_total: \n" << R_total << endl;
    // cout << "t_total: \n" << t_total << endl;
    // cout << "affine_3D: \n" << affine_3D << endl;
    fs << "frame" + to_string(frame) << affine_3D;
    
    // trajectory: Horizontal: x; Vertical: z.
    //
    string trajectory_groundTruth_name = "./trajectory_groundTruth.jpg";
    // cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3); 
    cv::Mat trajectory = cv::imread(trajectory_groundTruth_name);

    while(getline(image_1_names, image_1_name)){
        cout << "Current frame: " << frame++ << endl;

        // Obtain the aboslute data path from the relative path.
        image_1_name =  dir_name + "/sequences/00/" + image_1_name;
        getline(image_2_names, image_2_name);
        image_2_name = dir_name + "sequences/00/" +   image_2_name;
        triangulateStereo(image_1_name, image_2_name, Q, image_1_new, image_3D_new);

        // Feature detector + descriptor extractor + matcher.
        //
        calculateFeatures(image_1_new, keypoints_new, descriptors_new);
        getEffective3Dkeypoints(
            keypoints_new, 
            descriptors_new,
            image_3D_new, 
            keypoints3D_new);
        matchFeatures(descriptors_former, descriptors_new, matches);


        // Harris-corners + optical Flow.
        //
        // vector<uchar> features_found;
        // calculateOpticalFlow(
        //     image_1_former,
        //     image_1_new,
        //     keypoints_former,
        //     keypoints_new,
        //     features_found
        // );
        // filterCorners(keypoints_former, keypoints_new, features_found);
        // getEffective3DCorners(
        //     keypoints_former,
        //     keypoints_new,
        //     image_3D_former,
        //     image_3D_new,
        //     keypoints3D_former,
        //     keypoints3D_new
        // );


        // Calculate the relatvie transformation between the former and the new frames.
        //
        getRelativeTransform(
            keypoints3D_former, 
            keypoints3D_new, 
            matches, 
            image_3D_former, 
            image_3D_new,
            R_relative_new, 
            t_relative_new);
        
        // Obtain the total transformation from the relative transformation.
        //
        R_total = R_relative_new * R_total;
        t_total = R_relative_new * t_total + t_relative_new;  

        // t_total = t_total + t_relative_new;
        // cout << "\nR_relative_former: \n" << R_relative_former << endl;
        // cout << "R_relative_new: \n" << R_relative_new << endl;
        // cout << "R_total: \n" << R_total << endl;
        // cout << "t_relative_former: \n" << t_relative_former << endl;
        // cout << "\nt_relative_new: \n" << t_relative_new << endl;
        cout << "t_total: \n" << t_total << endl;
        
        // Save.
        //
        Rs.push_back(R_total);
        ts.push_back(t_total);
        // Save.
        R_total.copyTo(affine_3D.colRange(0, 3));
        t_total.copyTo(affine_3D.col(3));
        // cout << "affine_3D: \n" << affine_3D << endl;
        fs << "frame" + to_string(frame) << affine_3D;

        // Visualize the trajectory.
        //
        visualizeTrajectoryXZ(trajectory, R_total, t_total, frame);

        // Update.
        //
        // cout << "keypoints_former: " << keypoints_former[0].pt << endl;
        // cout << "keypoints_new: " << keypoints_new[0].pt << endl;
        // cout << "descriptors_former: " << descriptors_former.row(0) << endl;
        // cout << "descriptors_new: " << descriptors_new.row(0) << endl;
        keypoints_former = keypoints_new;
        descriptors_former = descriptors_new.clone();
        image_3D_former = image_3D_new.clone();
        keypoints3D_former = keypoints3D_new;
        keypoints3D_new.clear();

    } // END OF while(getline(image_1_names, image_1_name)).

    // Save the trajectory.
    //
    cv::imwrite("trajectory.jpg", trajectory);

    fs.release();

} // END OF getTrajectory().





int main(int argc, char* argv[]){

    // Read the input parameters.
    if(argc < 2){
        cout << "ERROR: Wrong number of parameters!" << endl;
        help(argv);
    }

    string dir_name = argv[1];

    // Parse the calib.txt file and build the reprojection matrix Q.
    //
    string file_name = dir_name + "/sequences/00/calib.txt";
    // Reprojection matrix Q.
    cv::Mat Q;
    parserCalib(file_name, Q);

    // Get the trajectory.
    vector< cv::Mat > Rs, ts;
    getTrajectory(
        dir_name, 
        Q,
        Rs, 
        ts);

    

    // Over.
    cv::destroyAllWindows();
    return 0;
}
