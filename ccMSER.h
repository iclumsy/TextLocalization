#ifndef _CCMSER_H
#define _CCMSER_H
#include "opencv2/opencv.hpp"

#include <vector>
#include <map>
using namespace cv;
struct ccRegion
{
public:
	//! Constructor
	//ccRegion(int level = 256, int pixel = 0, int x = 0, int y = 0);
	//! Destructor
	~ccRegion(){};

	//! seed point and the threshold (max grey-level value)
	//int pixel;
	
	Rect rect;
	int area;
	double var;//variation

	//! pointers preserving the tree structure of the component tree
	ccRegion* parent;
	ccRegion* child;
	ccRegion* next;
	ccRegion* prev;
	int level;
	vector<Point> points;
};
class ccMSER //: public cv::FeatureDetector
{
public:
	//! the full constructor
	 explicit ccMSER( int _delta=5, int _min_area=60, int _max_area=14400,
		double _max_variation=0.25, double _min_diversity=.2,
		int _max_evolution=200, double _area_threshold=1.01,
		double _min_margin=0.003, int _edge_blur_size=5 );

	//! the operator that extracts the MSERs from the image or the specific part of it
	void operator()( const Mat& image,  vector<vector<Point> >& msers,	const Mat& mask=Mat() ) const;
	void operator()( const Mat& image,  vector<vector<Point> >& msers,  vector<Vec4i>& hierarchy, const Mat& mask=Mat() ) const; 
	void operator()( const Mat& image,vector<ccRegion> &regions, const Mat& mask=Mat()  )const;
	void mserTest(const Mat &image, Mat &draw_image);
	cv::AlgorithmInfo* info() const;

protected:
	void detectImpl( const cv::Mat& image, vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;

	int delta;
	int minArea;
	int maxArea;
	double maxVariation;
	double minDiversity;
	int maxEvolution;
	double areaThreshold;
	double minMargin;
	int edgeBlurSize;
	
};


#endif