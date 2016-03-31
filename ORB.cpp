#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<vector>
#include<iostream>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


/**************ZONE OF CLASS*********************/
class Track
{
public:
    Track(){}
    Track(Mat &);
    Track &operator()(Mat &);
    void match(Mat &);
    void process();
    vector<Point2f> Points(vector<KeyPoint> &);
private:
    Mat cap_gray;
    Mat cap_descriptors;
    vector<vector<DMatch>> matches;
    vector<KeyPoint> cap_keypoints;
    const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
    const double ransac_thresh = 2.5f; // RANSAC inlier threshold
};
/************DEFINE FUNCTION**********************/
Track::Track(Mat &cap)
{
   cvtColor(cap,cap_gray,COLOR_BGR2GRAY);
   
   Ptr<ORB> detector=ORB::create();
   detector->detectAndCompute(cap_gray, Mat(), cap_keypoints,cap_descriptors);   
}

void Track::match(Mat &obj_des)
{
    FlannBasedMatcher matcher(new cv::flann::LshIndexParams(6, 12, 1));
    matcher.knnMatch(obj_des,cap_descriptors,matches,2);
    cout<<matches.size()<<endl;
}

Track &Track::operator()(Mat &cap)
{
    cvtColor(cap,cap_gray,COLOR_BGR2GRAY);
   
    Ptr<ORB> detector=ORB::create();
    detector->detectAndCompute(cap_gray, Mat(), cap_keypoints,cap_descriptors);  
    return *this; 
}

vector<Point2f> Track::Points(vector<KeyPoint> &kp)
{
    vector<Point2f> temp;
    for (unsigned i = 0; i < kp.size(); i++)
    {
	temp.push_back(kp[i].pt);
    }
    return temp;
}

void Track::process()
{
    Mat homography;
    vector<vector<DMatch>> matches,inlier_mask;
    vector<KeyPoint> matched1, matched2;
    vector<Point2f> target_corners(4);
    vector<Point2f> obj_corners(4);
    for (size_t i = 0; i < matches.size(); i++)
    {
    	DMatch first = matches[i][0];
	float dist1 = matches[i][0].distance;
	float dist2 = matches[i][1].distance;
	if (dist1 < nn_match_ratio * dist2)
	{
	    matched1.push_back(keypoints_object[first.queryIdx]);
	    matched2.push_back(keypoints_scene[first.trainIdx]);
	}
    }
    if(matched1.size()>4)
    {
	homography=findHomography(Points(matched1), Points(matched2),RANSAC, ransac_thresh, inlier_mask);
    }
    
}
/************ZONE OF CLASS*************************/

int main(int argc,char **argv)
{
    
    Mat target=imread(argv[1],IMREAD_GRAYSCALE);
    if(!target.data)
    {
	cout<<"image reading error"<<endl;
	exit(-1);
    }

    Ptr<ORB> detector=ORB::create();
    vector<KeyPoint> obj_keypoints;
    Mat obj_descriptors;
    detector->detectAndCompute(target,Mat(),obj_keypoints,obj_descriptors);

    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 960);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    if(!cap.isOpened())
    {
	cout<<"Camera error"<<endl;
	return false;
    }
    Mat cap_frame;
    Track track;
    while(true)
    {
	cap>>cap_frame;
	if(!cap_frame.data)
	{
	    cout<<"Capture image error"<<endl;
	    return false;	
	}
	track(cap_frame);  
        track.match(obj_descriptors);
        waitKey(1);
    }
    return 0;
}
