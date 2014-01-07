#include "opencv.hpp"
#include <vector>
#include <deque>
#include <string>
namespace cv
{

struct CV_EXPORTS ERStat
{
public:
    //! Constructor
    explicit ERStat(int level = 256, int pixel = 0, int x = 0, int y = 0);
    //! Destructor
    ~ERStat(){};

    //! seed point and the threshold (max grey-level value)
    int pixel;
    int level;

    //! incrementally computable features
    int area;
    int perimeter;
    int euler;                 //!< euler number
    Rect rect;
    double raw_moments[2];     //!< order 1 raw moments to derive the centroid
    double central_moments[3]; //!< order 2 central moments to construct the covariance matrix
    std::deque<int> *crossings;//!< horizontal crossings
    float med_crossings;       //!< median of the crossings at three different height levels

	double var;//variation

    //! 2nd stage features
    float hole_area_ratio;
    float convex_hull_ratio;
    float num_inflexion_points;

    // TODO Other features can be added (average color, standard deviation, and such)


    // TODO shall we include the pixel list whenever available (i.e. after 2nd stage) ?
    std::vector<int> *pixels;

    //! probability that the ER belongs to the class we are looking for
    double probability;

    //! pointers preserving the tree structure of the component tree
    ERStat* parent;
    ERStat* child;
    ERStat* next;
    ERStat* prev;

    //! wenever the regions is a local maxima of the probability
    bool local_maxima;
    ERStat* max_probability_ancestor;
    ERStat* min_probability_ancestor;
};

/*!
    Base class for 1st and 2nd stages of Neumann and Matas scene text detection algorithms
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    Extracts the component tree (if needed) and filter the extremal regions (ER's) by using a given classifier.
*/
class CV_EXPORTS ERFilter : public Algorithm
{
public:

    //! callback with the classifier is made a class. By doing it we hide SVM, Boost etc.
    class CV_EXPORTS Callback
    {
    public:
        virtual ~Callback(){};
        //! The classifier must return probability measure for the region.
        virtual double eval(const ERStat& stat) = 0; //const = 0; //TODO why cannot use const = 0 here?
    };

    /*!
        the key method. Takes image on input and returns the selected regions in a vector of ERStat
        only distinctive ERs which correspond to characters are selected by a sequential classifier
        \param image   is the input image
        \param regions is output for the first stage, input/output for the second one.
    */
    virtual void run( InputArray image, std::vector<ERStat>& regions ) = 0;


    //! set/get methods to set the algorithm properties,
    virtual void setCallback(const Ptr<ERFilter::Callback>& cb) = 0;
    virtual void setThresholdDelta(int thresholdDelta) = 0;
    virtual void setMinArea(float minArea) = 0;
    virtual void setMaxArea(float maxArea) = 0;
    virtual void setMinProbability(float minProbability) = 0;
    virtual void setMinProbabilityDiff(float minProbabilityDiff) = 0;
    virtual void setNonMaxSuppression(bool nonMaxSuppression) = 0;
    virtual int  getNumRejected() = 0;
};

// the classe implementing the interface for the 1st and 2nd stages of Neumann and Matas algorithm
class CV_EXPORTS ERFilterNM : public ERFilter
{
public:
	//Constructor
	ERFilterNM();
	//Destructor
	~ERFilterNM() {};

	float minProbability;
	bool  nonMaxSuppression;
	float minProbabilityDiff;

	// the key method. Takes image on input, vector of ERStat is output for the first stage,
	// input/output - for the second one.
	void run( InputArray image, std::vector<ERStat>& regions );

protected:
	int thresholdDelta;
	float maxArea;
	float minArea;

	Ptr<ERFilter::Callback> classifier;

	// count of the rejected/accepted regions
	int num_rejected_regions;
	int num_accepted_regions;

public:

	// set/get methods to set the algorithm properties,
	void setCallback(const Ptr<ERFilter::Callback>& cb);
	void setThresholdDelta(int thresholdDelta);
	void setMinArea(float minArea);
	void setMaxArea(float maxArea);
	void setMinProbability(float minProbability);
	void setMinProbabilityDiff(float minProbabilityDiff);
	void setNonMaxSuppression(bool nonMaxSuppression);
	int  getNumRejected();
	void FilterRegion(vector<ERStat>& _regions);
private:
	// pointer to the input/output regions vector
	std::vector<ERStat> *regions;
	// image mask used for feature calculations
	Mat region_mask;

	// extract the component tree and store all the ER regions
	void er_tree_extract( InputArray image );
	// accumulate a pixel into an ER
	void er_add_pixel( ERStat *parent, int x, int y, int non_boundary_neighbours,
		int non_boundary_neighbours_horiz,
		int d_C1, int d_C2, int d_C3 );
	// merge an ER with its nested parent
	void er_merge( ERStat *parent, ERStat *child );
	// copy extracted regions into the output vector
	ERStat* er_save( ERStat *er, ERStat *parent, ERStat *prev );
	// recursively walk the tree and filter (remove) regions using the callback classifier
	ERStat* er_tree_filter( InputArray image, ERStat *stat, ERStat *parent, ERStat *prev );
	// recursively walk the tree selecting only regions with local maxima probability
	ERStat* er_tree_nonmax_suppression( ERStat *er, ERStat *parent, ERStat *prev );

	
	ERStat* FilterRegionHelper( ERStat * stat, ERStat *parent, ERStat *prev);
};
}