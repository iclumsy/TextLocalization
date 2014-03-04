#include <iostream>
#include <stack>
#include <queue>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"  
#include "ccer.hpp"
#include <fstream>
#include "LogisticRegression.hpp"
#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp" // includes boost/filesystem/path.hpp
#include "boost/filesystem/fstream.hpp"    // ditto
#include "ccMSER.h"
#include <time.h>
#include "ReadConfig.h"
using namespace std;
using namespace cv;
using namespace LogisticRegression;
using namespace boost;
namespace bfs = boost::filesystem;
int cnt=0,err=0;
Mat draw;

void breadth(ccRegion *root){
	queue<ccRegion *> s;
	s.push(root->child);
	int level=1;
	while(!s.empty()){
		ccRegion *tmp=s.front();
		s.pop();
		while(tmp){
			if(tmp->child){
				if(tmp->child->parent!=tmp)err++;//check tree structure
				s.push(tmp->child);
			}
			//imshow("re",draw(tmp->rect));
			//waitKey();
			//destroyWindow("re");
			//printf("%d ",tmp->area);
			cnt++;
			//rectangle(draw,tmp->rect,Scalar(0,0,255),2);
			
			char b[100];
			sprintf(b,"%f",tmp->var);
			string a(b);
			imwrite("e://mserout//"+a+".jpg",draw(tmp->rect));
			//cout<<"xxxx var:"<<tmp->var<<"\tarea:"<<tmp->area<<"\n";
			if(tmp->next&&tmp->next->prev!=tmp)err++;//check tree structure
			tmp=tmp->next;
		}
		//cout<<endl;
	}
}
void Clear(ccRegion *e){
	e->parent=NULL;
	e->child=NULL;
	e->prev=NULL;
	e->next=NULL;
}
ccRegion* LinearReduction(ccRegion *root){
	if(root==NULL){
		return NULL;
	}
	if(root->child==NULL)return root;
	if(root->child->next==NULL){//only one child
		ccRegion *tmp=LinearReduction(root->child);
		if(root->var<tmp->var){
			if(tmp->child){
				tmp->child->parent=root;
			}
			root->child=tmp->child;
			Clear(tmp);
			return root;
		}
		else{
			if(root->parent){
				root->parent->child=tmp;
			}
			tmp->parent=root->parent;
			Clear(root);
			return tmp;
		}
	}
	else{//more than one child
		ccRegion *cur=root->child->next,*pre=LinearReduction(root->child);
		root->child=pre;
		pre->parent=root;
		pre->prev=NULL;
		while(cur){
			ccRegion *next=cur->next,*tmp=LinearReduction(cur);
			//cout<<cur<<"\t"<<next<<endl;
			tmp->parent=root;
			pre->next=tmp;
			tmp->prev=pre;
			pre=tmp;
			cur=next;
		}
		return root;
	}
}
void Union(vector<ccRegion *> &dst,vector<ccRegion *> &src){
	for(size_t i=0;i<src.size();i++){
		dst.push_back(src[i]);
	}
}
vector<ccRegion *> TreeAccumulation(ccRegion *root){
	vector<ccRegion *> res;
	if(root->child!=NULL&&root->child->next!=NULL){//more than one child
		ccRegion *tmp=root->child;
		while(tmp){
			Union(res,TreeAccumulation(tmp));
			tmp=tmp->next;
		}
		for(size_t i=0;i<res.size();i++){
			if(root->var>res[i]->var){
				return res;
			}
		}
	}
	res.clear();
	res.push_back(root);
	return res;
}
void ComputeVariation(ccRegion *root,int delta,double min_aspect_ratio,double max_aspect_ratio,double theta1,double theta2){
	if(root==NULL)return;
	int curlevel=root->level;
	ccRegion *parent=root,*child=root;
	while(parent->parent&&//has parent
		curlevel+delta>parent->parent->level){
		parent=parent->parent;
	}
	while(child->child&&
		curlevel-delta>child->child->level){
		child=child->child;
	}

	//var=(R_{i+delta}-R_{i})/R_{i}
	//there exists other ways to compute variation
	double var=1.0*(parent->area-root->area)/root->area;
	root->var=var;

	//regularization
	double aspect_ratio=root->rect.width*1.0/root->rect.height;
	if(aspect_ratio>max_aspect_ratio){
		var=var+theta1*(aspect_ratio-max_aspect_ratio);
	}
	else if(aspect_ratio<min_aspect_ratio){
		var=var+theta2*(min_aspect_ratio-aspect_ratio);
	}

	//printf("current area:%d var:%f\n",root->area,var);
	child=root->child;
	while(child){
		ComputeVariation(child,delta,min_aspect_ratio,max_aspect_ratio,theta1,theta2);
		child=child->next;
	}
}
void ComputeStrokeWidthHelper(Mat &img,int level,int color,double startx,double starty,double diagonallength,double xincrease,double yincrease,int &strokewidth){
	int currentwidth=0;
	double x=startx,y=starty;
	for(int i=0;i<diagonallength;++i){
		int tx=floor(x+0.5),ty=floor(y+0.5);
		if(tx>=img.cols||ty>=img.rows){
			break;
		}
		//cout<<"tx:"<<tx<<"\tty:"<<ty<<endl;
		if(img.at<uchar>(ty,tx)*color>=level*color){
			currentwidth++;
		}
		else{
			if(currentwidth){
				strokewidth=min(currentwidth,strokewidth);
			}
			currentwidth=0;
		}
		x+=xincrease;
		y+=yincrease;
	}
}
int ComputeStrokeWidth(Mat &img,Rect rect,int level,int color){
	double xincrease,yincrease,diagonallength;
	diagonallength=sqrt(rect.width*rect.width+rect.height*rect.height);
	xincrease=rect.width/diagonallength;
	yincrease=rect.height/diagonallength;
	int strokewidth=INT_MAX;
	ComputeStrokeWidthHelper(img,level,color,rect.x,rect.y,diagonallength,xincrease,yincrease,strokewidth);
	ComputeStrokeWidthHelper(img,level,color,rect.x,rect.y+rect.height,diagonallength,xincrease,-yincrease,strokewidth);
	ComputeStrokeWidthHelper(img,level,color,rect.x+0.5*rect.width,rect.y,diagonallength,0,yincrease,strokewidth);
	ComputeStrokeWidthHelper(img,level,color,rect.x,rect.y+0.5*rect.height,diagonallength,xincrease,0,strokewidth);
	return strokewidth;
}
void ComputeFeature(Mat &img,ccRegion eru,ccRegion erv,vector<double> &featureVector){
	featureVector.clear();
	Rect u=eru.rect,v=erv.rect;
	//spatial distance 
	double spatialdistance=abs(u.x+0.5*u.height-v.x-0.5*u.width)*1.0/max(u.width,v.width);
	featureVector.push_back(spatialdistance);

	//interval
	double interval;
	if(u.x<v.x){
		interval=abs(v.x-u.x-u.width)*1.0/max(u.width,v.width);
	}
	else{
		interval=abs(u.x-v.x-v.width)*1.0/max(u.width,v.width);
	}
	featureVector.push_back(interval);

	//width and height differences
	double widthdifference,heightdifference;
	widthdifference=abs(u.width-v.width)*1.0/max(u.width,v.width);
	heightdifference=abs(u.height-v.height)*1.0/max(u.height,v.height);
	featureVector.push_back(widthdifference);
	featureVector.push_back(heightdifference);

	//top and bottom alignments
	double topalignment,bottomalignment;
	topalignment=atan(abs(u.y-v.y)/abs(u.x+0.5*u.height-v.x-0.5*u.width));
	bottomalignment=atan(abs(u.y-v.y+u.height-v.height)/abs(u.x+0.5*u.height-v.x-0.5*u.width));
	featureVector.push_back(topalignment);
	featureVector.push_back(bottomalignment);

	//color difference

	//stroke width difference
	int ustrokewidth=ComputeStrokeWidth(img,u,eru.level,eru.color);
	int vstrokewidth=ComputeStrokeWidth(img,v,erv.level,erv.color);
	double strokewidthdifference=abs(ustrokewidth-vstrokewidth)*1.0/max(ustrokewidth,vstrokewidth);
	featureVector.push_back(strokewidthdifference);
}
double ComputeDistance(Mat &img,ccRegion eru,ccRegion erv,vector<double> &featureWeights){
	vector<double> featureVector;
	ComputeFeature(img,eru,erv,featureVector);
	//cout<<eru.level<<"\t"<<erv.level<<endl;
	assert(featureVector.size()==featureWeights.size());
	double ret=0;
	for(size_t i=0;i<featureWeights.size();++i){
		ret+=featureWeights[i]*featureVector[i];
	}
	return ret;
}
struct distances{
	double dist;
	int index;
	distances(){
		dist=999999;
		index=-1;
	}
	distances(double d,int i){
		dist=d;
		index=i;
	}
};
// comparing operator for struct distances
struct Cmp{
	bool operator()(const distances d1, const distances d2)const{
		if(d1.dist == d2.dist){
			return d1.index < d2.index;
		}
		return d1.dist < d2.dist;
	}
};

vector<vector<int>> HierarchicalClustering(Mat &img,vector<ccRegion *> &data,vector<double> &featureweights,double threshold) {
	int N=data.size();
	vector<vector<distances> >dist;// 2d vector for storing distances matrix
	vector<multiset<distances, Cmp>> P(N);// multiset for storing sorted distances
	vector<int> active(N,1);// vector for storing flags for marking currently active clusters
	vector<vector<int>> A;// 2d vector for storing lists of titles in clusters
	vector<vector<int>> res;
	for(size_t i=0;i<N;++i){
		vector<int> A_i;
		A_i.push_back(i);
		A.push_back(A_i);
		dist.push_back(vector<distances>(N));
	}
	for(size_t i=0;i<N;++i){
		for(size_t j=i+1;j<N;++j){
			double cur_dist=ComputeDistance(img,*data[i],*data[j],featureweights);
			dist[i][j]=distances(cur_dist,j);
			dist[j][i]=distances(cur_dist,i);
			P[i].insert(dist[i][j]);
			P[j].insert(dist[j][i]);
			//cout<<i<<"\t"<<j<<"\t"<<dist[i][j].dist<<endl;
		}
	}
	while(1){
		double min_dist = INT_MAX;
		int min_index = 0;
		for(int i=0; i<N-1; ++i){
			if(active[i]==1){
				if(P[i].begin()->dist<min_dist){
					min_dist = P[i].begin()->dist;
					min_index = P[i].begin()->index;
				}
			}
		}
		//cout<<"min dist:\t"<<min_dist<<endl;
		if(min_dist>threshold){
			//break;
		}
		// we have minimum distance
		// k1, k2 - indexes of most nearest clusters
		int k1 = min_index;
		int k2 = P[k1].begin()->index;

		Rect tmp=data[k1]->rect|data[k2]->rect;
		if((data[k1]->rect.area()+data[k2]->rect.area())*1.0/tmp.area()<threshold){
			break;
		}

		int N_k1 = A[k1].size();
		int N_k2 = A[k2].size();

		P[k1].clear();
		// add cluster k2 to A[k1] 
		for(int i=0;i<A[k2].size();++i){
			A[k1].push_back(A[k2][i]);
		}

		// clear the second cluster
		active[k2] = 0;
		// O(N*log(N))
		for(int i=0; i<N; ++i){
			// O(log(N)): insert, erase operations
			if((active[i]!=0)&&(i!=k1)){
				P[i].erase(dist[i][k1]);
				P[i].erase(dist[i][k2]);
				dist[i][k1].dist = dist[i][k1].dist<dist[i][k2].dist ? dist[i][k1].dist : dist[i][k2].dist;
				dist[k1][i].dist = dist[i][k1].dist;
				P[i].insert(dist[i][k1]);					
				P[k1].insert(dist[k1][i]);
			}
		}
	}
	int class_num = 0;
	for(int i=0; i<N; ++i)
	{
		if(active[i]==1&&A[i].size()>1)//
		{
			Rect tmp=data[A[i][0]]->rect;
			++class_num;
			vector<int> tmpre;
			//cout<<endl<<"Class number: "<<class_num<<endl<<endl;
			for(int j=0;j<A[i].size();j++){
				tmpre.push_back(A[i][j]);
				tmp=tmp|data[A[i][j]]->rect;
				//char b[100];
				//sprintf(b,"%d---%f",class_num,data[A[i][j]]->var);
				//string a(b);
				//imwrite("e://mserout//"+a+".jpg",draw(data[A[i][j]]->rect));
				//cout<<A[i][j]<<std::endl;
			}
			//rectangle(draw,tmp,Scalar(0,0,255),2);
			res.push_back(tmpre);
		}
	}
	return res;
}
void ComputeStrokeWidthHelper(Mat &img,double startX,double startY,double diagonalLength,double xIncrease,double yIncrease,vector<int> &strokeWidthVector,int thres=10){
	int currentWidth=0;
	double x=startX,y=startY;
	uchar pre=255;
	
	for(int i=0;i<diagonalLength;++i){
		int tx=floor(x+0.5),ty=floor(y+0.5);
		if(tx>=img.cols||ty>=img.rows){
			break;
		}
		//cout<<"tx:"<<tx<<"\tty:"<<ty<<endl;
		int tmpPixel=img.at<uchar>(ty,tx);
		if(abs(tmpPixel-pre)<thres){
			currentWidth++;
		}
		else{
			if(currentWidth>1){
				//strokeWidth=min(currentWidth,strokeWidth);
				strokeWidthVector.push_back(currentWidth);
			}
			currentWidth=0;
			pre=tmpPixel;
		}
		x+=xIncrease;
		y+=yIncrease;
	}
}
int ComputeStrokeWidth(Mat &img,Rect rect){
	double xIncrease,yIncrease,diagonalLength;
	diagonalLength=sqrt(rect.width*rect.width+rect.height*rect.height);
	xIncrease=rect.width/diagonalLength;
	yIncrease=rect.height/diagonalLength;
	int strokeWidth=INT_MAX;
	vector<int> strokeWidthVector;
	ComputeStrokeWidthHelper(img,rect.x,rect.y,diagonalLength,xIncrease,yIncrease,strokeWidthVector);
	ComputeStrokeWidthHelper(img,rect.x,rect.y+rect.height,diagonalLength,xIncrease,-yIncrease,strokeWidthVector);
	ComputeStrokeWidthHelper(img,rect.x+0.5*rect.width,rect.y,diagonalLength,0,yIncrease,strokeWidthVector);
	ComputeStrokeWidthHelper(img,rect.x,rect.y+0.5*rect.height,diagonalLength,xIncrease,0,strokeWidthVector);
	sort(strokeWidthVector.begin(),strokeWidthVector.end());
	return strokeWidthVector.size()>0?strokeWidthVector[strokeWidthVector.size()/2]:0;
}
vector<double> GenerateFeature(Mat &img,Rect rect){
	vector<double> res(4,0);
	
	res[0]=rect.height;
	res[1]=rect.width;
	res[2]=rect.height*1.0/rect.width;

	res[3]=ComputeStrokeWidth(img,rect);
	return res;
}
vector<float> GenerateHogFeature(Mat &img){
	vector<float> hogFeature;
	int width=img.cols/3;
	int height=img.rows/3;
	HOGDescriptor hog;
	hog.winSize=Size(width*3,height*3);
	hog.blockSize=Size(width,height);
	hog.cellSize=Size(width,height);
	hog.blockStride=Size(width,height);
	vector<Point> location;
	// 滑动窗只有一个，指定top-left位置
	location.push_back(Point(0,0));
	//cout << "block dimensions: " << width << " width x " << height << "height" << endl;	
	//cout<<"Calculating the HOG descriptors..."<<endl;
	hog.compute(img,hogFeature,hog.blockSize,Size(0,0),location);
	//cout << "HOG descriptor size is " << hog.getDescriptorSize() << endl;
	//cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
	//cout << "Found " << hogFeature.size() << " descriptor values" << endl;
	return hogFeature;
}
int main2(){
	string path="C:\\Users\\Administrator\\Desktop\\mser\\102.jpg";
	Mat src = imread(path,0);
	draw = imread(path,1);
	ERFilterNM er;
	vector<ccRegion>  regions;
	//er.run(src,regions);
	printf("number of regions:%d\n",regions.size());
	
	//for(int j=0;j<100;j++){
	//	for(int i=0;i<regions.size();i++){
	//		if(regions[i].chi ld!=NULL&&regions[i].child->next==NULL){
	//			regions[i].child=regions[i].child->child;
	//		}
	//	}
	//}
	Mat re=cvCreateMat(1500,3000,draw.type());
	er.setMaxArea(1);
	er.setMinArea(0.001);
	
	//er.FilterRegion(regions);
	printf("number of regions:%d\n",regions.size());
	ccRegion *root=&regions[0],*cur;
	ComputeVariation(root,5,0.3,1.2,0.01,0.35);
	root=LinearReduction(root);

	vector<ccRegion *> accumulatedtree=TreeAccumulation(root);
	printf("number of accumulated tree:%d\n",accumulatedtree.size());
	//for(size_t i=0;i<regions.size();i++){
	//	imshow("re",draw(regions[i].rect));
	//	waitKey();
	//	destroyWindow("re");
	//}
	int maxlevel=-1;
	
	queue<ccRegion *> s;
	s.push(root);
	s.push(NULL);
	int level=1,curlevel=0;
	while(s.empty()){
		ccRegion *tmp=s.front();
		s.pop();
		if(tmp==NULL&&!s.empty()){
			s.push(NULL);
			cout<<level++<<endl;
			curlevel=0;
		}
		while(tmp){
			if(tmp->child!=NULL&&tmp->child->next!=NULL){
				Mat tmpdraw=draw(tmp->rect);
				//printf("%d %d %d\n",cnt,cnt/8*200,cnt%8*100);
				tmpdraw.copyTo(re(Rect(curlevel*200,level*100,tmp->rect.width,tmp->rect.height)));
				curlevel++;
				cnt++;
			}
			if(tmp->child){
				s.push(tmp->child);
			}
			tmp=tmp->next;
		}
	}
	#pragma region test
for(int i=0;i<regions.size();i++){ 
		if(0&&regions[i].parent==NULL){
			cur=&regions[i];
			while(cur){
				if(cur->next){
					ccRegion *tmp=cur;
					while(tmp){
						rectangle(draw,tmp->rect,Scalar(0,0,tmp->level),1);
						tmp=tmp->next;
					}
				}
				printf("current rec: level:%d x:%d y:%d area:%d\n",cur->level,cur->rect.x,cur->rect.y,cur->area);
				cur=cur->child;
			}
		}
		//int curlevel=level(&regions[i]);
		if(0){
			cur=&regions[i];
			while(cur){
				rectangle(draw,cur->rect,Scalar(0,0,cur->level),1);
				printf("current rec: level:%d x:%d y:%d area:%d\n",cur->level,cur->rect.x,cur->rect.y,cur->rect.area());
				cur=cur->parent;
			}
		}
		//maxlevel=curlevel>maxlevel?curlevel:maxlevel;

		if(0&&regions[i].child!=NULL&&regions[i].child->next!=NULL){
			Mat tmp=draw(regions[i].rect);
			printf("%d %d %d\n",cnt,cnt/8*200,cnt%8*100);
			tmp.copyTo(re(Rect(cnt/8*200,cnt%8*100,regions[i].rect.width,regions[i].rect.height)));
			cnt++;
		}
	}
#pragma endregion test

	breadth(root);

	//for(size_t i=0;i<regions.size();i++){
	//	rectangle(draw,regions[i].rect,Scalar(0,0,regions[i].level),2);
	//}
	cout<<cnt<<endl;
	imshow("out",draw);
	//imshow("re",re);
	//imwrite("d:/re.jpg",re);
	waitKey();
	return 0;
}

int lr(int argc, char** argv){

	Mat Data = (Mat_<double>(150, 4)<< 5.1,3.5,1.4,0.2, 4.9,3.0,1.4,0.2, 4.7,3.2,1.3,0.2, 4.6,3.1,1.5,0.2, 5.0,3.6,1.4,0.2, 5.4,3.9,1.7,0.4, 4.6,3.4,1.4,0.3, 5.0,3.4,1.5,0.2, 4.4,2.9,1.4,0.2, 4.9,3.1,1.5,0.1, 5.4,3.7,1.5,0.2, 4.8,3.4,1.6,0.2, 4.8,3.0,1.4,0.1, 4.3,3.0,1.1,0.1, 5.8,4.0,1.2,0.2, 5.7,4.4,1.5,0.4, 5.4,3.9,1.3,0.4, 5.1,3.5,1.4,0.3, 5.7,3.8,1.7,0.3, 5.1,3.8,1.5,0.3, 5.4,3.4,1.7,0.2, 5.1,3.7,1.5,0.4, 4.6,3.6,1.0,0.2, 5.1,3.3,1.7,0.5, 4.8,3.4,1.9,0.2, 5.0,3.0,1.6,0.2, 5.0,3.4,1.6,0.4, 5.2,3.5,1.5,0.2, 5.2,3.4,1.4,0.2, 4.7,3.2,1.6,0.2, 4.8,3.1,1.6,0.2, 5.4,3.4,1.5,0.4, 5.2,4.1,1.5,0.1, 5.5,4.2,1.4,0.2, 4.9,3.1,1.5,0.1, 5.0,3.2,1.2,0.2, 5.5,3.5,1.3,0.2, 4.9,3.1,1.5,0.1, 4.4,3.0,1.3,0.2, 5.1,3.4,1.5,0.2, 5.0,3.5,1.3,0.3, 4.5,2.3,1.3,0.3, 4.4,3.2,1.3,0.2, 5.0,3.5,1.6,0.6, 5.1,3.8,1.9,0.4, 4.8,3.0,1.4,0.3, 5.1,3.8,1.6,0.2, 4.6,3.2,1.4,0.2, 5.3,3.7,1.5,0.2, 5.0,3.3,1.4,0.2, 7.0,3.2,4.7,1.4, 6.4,3.2,4.5,1.5, 6.9,3.1,4.9,1.5, 5.5,2.3,4.0,1.3, 6.5,2.8,4.6,1.5, 5.7,2.8,4.5,1.3, 6.3,3.3,4.7,1.6, 4.9,2.4,3.3,1.0, 6.6,2.9,4.6,1.3, 5.2,2.7,3.9,1.4, 5.0,2.0,3.5,1.0, 5.9,3.0,4.2,1.5, 6.0,2.2,4.0,1.0, 6.1,2.9,4.7,1.4, 5.6,2.9,3.6,1.3, 6.7,3.1,4.4,1.4, 5.6,3.0,4.5,1.5, 5.8,2.7,4.1,1.0, 6.2,2.2,4.5,1.5, 5.6,2.5,3.9,1.1, 5.9,3.2,4.8,1.8, 6.1,2.8,4.0,1.3, 6.3,2.5,4.9,1.5, 6.1,2.8,4.7,1.2, 6.4,2.9,4.3,1.3, 6.6,3.0,4.4,1.4, 6.8,2.8,4.8,1.4, 6.7,3.0,5.0,1.7, 6.0,2.9,4.5,1.5, 5.7,2.6,3.5,1.0, 5.5,2.4,3.8,1.1, 5.5,2.4,3.7,1.0, 5.8,2.7,3.9,1.2, 6.0,2.7,5.1,1.6, 5.4,3.0,4.5,1.5, 6.0,3.4,4.5,1.6, 6.7,3.1,4.7,1.5, 6.3,2.3,4.4,1.3, 5.6,3.0,4.1,1.3, 5.5,2.5,4.0,1.3, 5.5,2.6,4.4,1.2, 6.1,3.0,4.6,1.4, 5.8,2.6,4.0,1.2, 5.0,2.3,3.3,1.0, 5.6,2.7,4.2,1.3, 5.7,3.0,4.2,1.2, 5.7,2.9,4.2,1.3, 6.2,2.9,4.3,1.3, 5.1,2.5,3.0,1.1, 5.7,2.8,4.1,1.3, 6.3,3.3,6.0,2.5, 5.8,2.7,5.1,1.9, 7.1,3.0,5.9,2.1, 6.3,2.9,5.6,1.8, 6.5,3.0,5.8,2.2, 7.6,3.0,6.6,2.1, 4.9,2.5,4.5,1.7, 7.3,2.9,6.3,1.8, 6.7,2.5,5.8,1.8, 7.2,3.6,6.1,2.5, 6.5,3.2,5.1,2.0, 6.4,2.7,5.3,1.9, 6.8,3.0,5.5,2.1, 5.7,2.5,5.0,2.0, 5.8,2.8,5.1,2.4, 6.4,3.2,5.3,2.3, 6.5,3.0,5.5,1.8, 7.7,3.8,6.7,2.2, 7.7,2.6,6.9,2.3, 6.0,2.2,5.0,1.5, 6.9,3.2,5.7,2.3, 5.6,2.8,4.9,2.0, 7.7,2.8,6.7,2.0, 6.3,2.7,4.9,1.8, 6.7,3.3,5.7,2.1, 7.2,3.2,6.0,1.8, 6.2,2.8,4.8,1.8, 6.1,3.0,4.9,1.8, 6.4,2.8,5.6,2.1, 7.2,3.0,5.8,1.6, 7.4,2.8,6.1,1.9, 7.9,3.8,6.4,2.0, 6.4,2.8,5.6,2.2, 6.3,2.8,5.1,1.5, 6.1,2.6,5.6,1.4, 7.7,3.0,6.1,2.3, 6.3,3.4,5.6,2.4, 6.4,3.1,5.5,1.8, 6.0,3.0,4.8,1.8, 6.9,3.1,5.4,2.1, 6.7,3.1,5.6,2.4, 6.9,3.1,5.1,2.3, 5.8,2.7,5.1,1.9, 6.8,3.2,5.9,2.3, 6.7,3.3,5.7,2.5, 6.7,3.0,5.2,2.3, 6.3,2.5,5.0,1.9, 6.5,3.0,5.2,2.0, 6.2,3.4,5.4,2.3, 5.9,3.0,5.1,1.8);
	Mat Labels = (Mat_<int>(150, 1)<< 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3);
	cout<<Labels.type()<<endl;
	cout<<Data.type()<<endl;
	Mat Responses;

	CvLR_TrainParams params = CvLR_TrainParams();


	params.alpha = 1.00;
	params.num_iters = 10000;
	params.normalization = CvLR::REG_L1;
	params.debug = true;
	params.regularized = true;
	params.train_method = CvLR::BATCH;

	CvLR lr_(Data, Labels, params);


	lr_.predict(Data, Responses);

	Mat Result = (Labels == Responses)/255;

	cout<<"[Original Label]\t[Predicted Label]\t[Result]"<<endl;

	for(int i =0;i<Labels.rows;i++)
	{
		cout<<Labels.row(i)<<"\t"<<Responses.row(i)<<"\t"<<Result.row(i)<<endl;
	}

	cout<<"accuracy: "<<((double)cv::sum(Result)[0]/Result.rows)*100<<"%\n";


	Mat NData = (Mat_<double>(1,4)<<5.9,3.0,5.1,1.8);

	cout<<"predicted label of "<<NData<<"is :"<<lr_.predict(NData)<<endl;

	cout<<"done"<<endl;

	lr_.print_learnt_mats();
	 
	CvMLData cvml;


	cvml.read_csv("E:\\LR\\trunk\\src\\digitdata2.txt");

	cvml.set_response_idx(64);

	const CvMat* vs = cvml.get_values();
	cout << "Rows: " << vs->rows << " Cols: " << vs->cols << endl;

	Mat DataMat = vs;
	Mat LLabels, DData;
	cout<<DataMat.rows<<", "<<DataMat.cols<<endl;

	Data = DataMat(Range::all(), Range(0,DataMat.cols-1));
	Data.convertTo(Data, CV_64F);	

	Labels = DataMat(Range::all(), Range(64,65));
	Labels.convertTo(LLabels, CV_32S);
	Labels = LLabels.clone();

	Mat Responses1;

	CvLR_TrainParams params1;

	params1.alpha = 0.37;
	params1.num_iters = 10000;
	params1.normalization = CvLR::REG_L2;
	params1.debug = true;
	params1.regularized = true;
	params1.train_method = CvLR::MINI_BATCH;

	CvLR lr1_(Data, Labels, params1);

	lr1_.predict(Data, Responses1);

	Result = (Labels == Responses1)/255;

	cout<<"[Original Label]\t[Predicted Label]\t[Result]"<<endl;

	 for(int i =0;i<Labels.rows;i++)
	 {
	 	cout<<Labels.row(i)<<"\t"<<Responses1.row(i)<<"\t"<<Result.row(i)<<endl;
	 }

	cout<<"accuracy: "<<((double)cv::sum(Result)[0]/Result.rows)*100<<"%\n";

	return 0;
}

int main78(){
	string p;
	//p="C:\\Users\\Administrator\\Desktop\\mser\\385.jpg";
	string configFile="config.txt";
	map<string, string> config;
	ReadConfig(configFile, config);
	PrintConfig(config);
	p=config["img"];
	double thres=atof(config["thres"].c_str());
	Mat image = imread(p, 0);
	draw = imread(p, 1);
	clock_t start=clock();
	ccMSER t(5, 60, 1440,0.25, 0.2,200, 1.01,0.003, 5);
	//vector<vector<Point> > contours;
	//vector<Vec4i> hierarchy;
	//CvSeq *contours;
	vector<ccRegion> regions;
	t(image,regions);
	//breadth(&regions[0]);
	//int cnt=0;
	vector<ccRegion *> res;
	for(int i=0;i<regions.size();i++){
		//rectangle(draw,regions[i].rect,Scalar(0,255,255),2);
		if(regions[i].parent==NULL){
			//cnt++;
			//cout<<"root var:"<<regions[i].var<<"\tarea:"<<regions[i].area<<"\n";
			//char b[100];
			//sprintf(b,"%f",regions[i].var);
			//string a(b);
			//imwrite("e://mserout//root-"+a+".jpg",draw(regions[i].rect));
			Union(res,TreeAccumulation(LinearReduction(&regions[i])));
			//breadth(&regions[i]);
			//breadth(LinearReduction(&regions[i]));
			//vector<ccRegion *> re;
			//re=TreeAccumulation(LinearReduction(&regions[i]));
		}
	}
	cout<<"after LinearReduction and TreeAccumulation:"<<res.size()<<endl;
	vector<double> featureWeights(7,1);
	string featureWeightsString=config["feature_weights"];
	int send,sbegin=0;
	for(int i=0;i<featureWeights.size();i++){
		send=featureWeightsString.find(",",sbegin);
		featureWeights[i]=atof(featureWeightsString.substr(sbegin,send).c_str());
		sbegin=send+1;
	}
	vector<vector<int>> clusterResult=HierarchicalClustering(image,res,featureWeights,thres);
	cout<<"after HierarchicalClustering:"<<clusterResult.size()<<endl;
	CvSVM SVM;
	SVM.load("e:\\400_svm.xml");
	cout<<"load SVM complete\n";
	Mat predictData(1,81,CV_32FC1);
	Mat duibi=imread(p,1);
	for(int j=0;j<clusterResult.size();++j){
		Rect tmp=res[clusterResult[j][0]]->rect;
		for(int k=1;k<clusterResult[j].size();++k){
			tmp=tmp|res[clusterResult[j][k]]->rect;
		}
		vector<float> t=GenerateHogFeature(image(tmp));
		for(int i=0;i<t.size();++i){
			predictData.at<float>(i)=t[i];
		}
		float response=SVM.predict(predictData);
		if(response>0)rectangle(draw,tmp,Scalar(0,255,0),2);
		else rectangle(draw,tmp,Scalar(0,0,255),2);
	}
	//for(int i=0;i<clusterResult.size();i++){
	//	cout<<i<<":\t";
	//	for(int j=0;j<clusterResult[i].size();j++){
	//		cout<<clusterResult[i][j]<<" ";
	//	}
	//	cout<<endl;
	//}
	//cout<<"err:"<<err<<endl;
	//cout<<cnt<<endl;
	//Seq<CvSeq*> contours;
	//t(image,contours,hierarchy);
	//t(image,contours);
	//t.mserTest(image, draw);
	//CvSeq* tt=contours[0];
	//CvContour* tmp=(CvContour*)*it;
	//imshow("Color mser", draw);
	cout<<"time costs: "<<double(clock()-start)/CLOCKS_PER_SEC<<"s\n";
	imshow("with SVM", draw);
	//imshow("without SVM", duibi);
	waitKey(0);
	return 0;
}
vector<Rect> readGroundTruth(string fileName){
	vector<Rect> res;
	ifstream r(fileName);
	string tmp;
	int a[4];
	while(!r.eof()){
		getline(r,tmp);
		if(tmp.length()<1)break;
		int start=0,end;
		for(int i=0;i<4;i++){
			end=tmp.find(',',start);
			a[i]=atoi(tmp.substr(start,end-start).c_str());
			start=end+1;
			//cout<<a[i]<<"\t";
		}
		//cout<<endl;
		Rect tmpRect(Point(a[0],a[1]),Point(a[2],a[3]));
		res.push_back(tmpRect);
		//rectangle(draw,tmpRect,Scalar(0,0,255),2);
	}
	return res;
}
int main11(){
	string configFile="config.txt";
	map<string, string> config;
	ReadConfig(configFile, config);
	double thres=atof(config["thres"].c_str());
	bfs::path p( "e:\\bizhi" );
	//copy(bfs::directory_iterator(p), bfs::directory_iterator(), // directory_iterator::value_type
		//ostream_iterator<bfs::directory_entry>(cout, "\n")); // is directory_entry, which is
	// converted to a path by the
	// path stream inserter

	ofstream f("C:\\Users\\Administrator\\Desktop\\1.txt");
	bfs::directory_iterator it(p),endIter;
	ccMSER t(5, 60, 1440,0.25, 0.2,200, 1.01,0.003, 5);
	//vector<vector<Point> > contours;
	//vector<Vec4i> hierarchy;
	//CvSeq *contours;
	vector<ccRegion> regions;
	int ii=1;
	for(;it==endIter;it++){
		string path=it->path().string();
		//cout<<path<<endl;
		Mat grayImage=imread(path,0);
		draw=imread(path);
		t(grayImage,regions);

		vector<ccRegion *> res;
		for(int i=0;i<regions.size();i++){
			if(regions[i].parent==NULL){
				Union(res,TreeAccumulation(LinearReduction(&regions[i])));
			}
		}
		cout<<"after LinearReduction and TreeAccumulation:"<<res.size()<<endl;
		int gen=0;
		for(int i=0;i<res.size();i++){
			
			//for(int k=0;k<t.size();++k){
			//	f<<t[k]<<"\t";
			//}
			//f<<""<<endl;

			if(res[i]->area<500||res[i]->area>20000)continue;
			vector<double> t=GenerateFeature(grayImage,res[i]->rect);
			break;
			//imwrite("e:\\negative\\"+lexical_cast <string>(ii)+"-"+lexical_cast <string>(i+1)+".jpg",draw(res[i]->rect));
			//if(++gen>2)break;
		}
		++ii;
		//break;
	}
	for(int i=101;i<=101;++i){
		string p,gt;
		string a=lexical_cast <string>(i);
		cout<<i<<endl;
		p="E:\\test-textloc-gt\\"+a+".jpg";
		gt="E:\\test-textloc-gt\\gt_"+a+".txt"; 
		//draw = imread(p, 1);
		vector<Rect> groundTruthRect=readGroundTruth(gt);
		Mat img=imread(p);
		Mat grayImg=imread(p,0);
		for(int j=0;j<groundTruthRect.size();++j){
			//imwrite("e:\\groundtruth\\"+a+"-"+lexical_cast <string>(j+1)+".jpg",img(groundTruthRect[j]));
			//f<<grayImg(groundTruthRect[8]);
			//imshow("c",grayImg(groundTruthRect[8]));
			//waitKey(0);

			//vector<double> t=GenerateFeature(grayImg,groundTruthRect[j]);
			//for(int k=0;k<t.size();++k){
			//	f<<t[k]<<"\t";
			//}
			//f<<""<<endl;
			
			//imwrite("e:\\positive\\"+lexical_cast <string>(i)+"-"+lexical_cast <string>(j+1)+".jpg",img(groundTruthRect[j]));

			//cout<<"strkoe width:\t"<<t[3]<<endl;
			//break;
		}
		Mat image = imread(p, 0);
		draw=imread(p,0);
		ccMSER t(5, 60, 1440,0.25, 0.2,200, 1.01,0.003, 5);
		vector<ccRegion> regions;
		t(image,regions);
		vector<ccRegion *> res;
		for(int i=0;i<regions.size();i++){
			if(regions[i].parent==NULL){
				Union(res,TreeAccumulation(LinearReduction(&regions[i])));
			}
		}
		cout<<"after LinearReduction and TreeAccumulation:"<<res.size()<<endl;
		vector<double> featureWeights(7,1);
		vector<vector<int>> clusterResult=HierarchicalClustering(image,res,featureWeights,thres);
		cout<<"after HierarchicalClustering:"<<clusterResult.size()<<endl;
		for(int j=0;j<clusterResult.size();++j){
			Rect tmp=res[clusterResult[j][0]]->rect;
			for(int k=1;k<clusterResult[j].size();++k){
				tmp=tmp|res[clusterResult[j][k]]->rect;
			}
			rectangle(draw,tmp,Scalar(0,0,255),2);
		}
	}
	imshow("result.jpg", draw);
	waitKey(0);
	return 0;
}

int svmexample()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[4] = {1.0, -1.0, -1.0, -1.0};
	Mat labelsMat(4, 1, CV_32FC1, labels);

	float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	Vec3b green(0,255,0), blue (255,0,0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1,2) << i,j);
			float response = SVM.predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(j, i)  = green;
			else if (response == -1)
				image.at<Vec3b>(j, i)  = blue;
		}

		// Show the training data
		int thickness = -1;
		int lineType = 8;
		circle(	image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
		circle(	image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
		circle(	image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
		circle(	image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

		// Show support vectors
		thickness = 2;
		lineType  = 8;
		int c     = SVM.get_support_vector_count();

		for (int i = 0; i < c; ++i)
		{
			const float* v = SVM.get_support_vector(i);
			circle(	image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
		}

		//imwrite("result.png", image);        // save the image

		imshow("SVM Simple Example", image); // show it to the user
		waitKey(0);
		return 0;
}
void svmTrain(){
	bfs::path p("C:\\Users\\Administrator\\Desktop\\1101214006_hw3\\text");
	bfs::directory_iterator it(p),endIter;
	vector<vector<float> >trainVector;
	vector<float> trainLabelVector;
	for(;it!=endIter;it++){
		string path=it->path().string();
		//cout<<path<<endl;
		Mat grayImage=imread(path,0);
		vector<float> t=GenerateHogFeature(grayImage);
		trainVector.push_back(t);
		trainLabelVector.push_back(1);
	}
	p=bfs::path("C:\\Users\\Administrator\\Desktop\\1101214006_hw3\\nontext" );
	it=bfs::directory_iterator(p);
	for(;it!=endIter;it++){
		string path=it->path().string();
		//cout<<path<<endl;
		Mat grayImage=imread(path,0);
		vector<float> t=GenerateHogFeature(grayImage);
		trainVector.push_back(t);
		trainLabelVector.push_back(-1);
	}
	Mat trainData=Mat(trainVector.size(),trainVector[0].size(),CV_32FC1);
	for(int i=0;i<trainVector.size();++i){
		for(int j=0;j<trainVector[i].size();++j){
			trainData.at<float>(i,j)=trainVector[i][j];
		}
	}
	Mat trainLabel=Mat(trainLabelVector.size(),1,CV_32FC1,trainLabelVector.data());
	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	// Train the SVM
	CvSVM SVM;
	SVM.train(trainData, trainLabel, Mat(), Mat(), params);
	//SVM.train_auto(trainData,trainLabel,Mat(), Mat(), params);
	cout<<"train SVM complete\n";
	SVM.save("e:\\400_svm.xml");
}
int svm(){

	FileStorage fs("e:\\vocabulary.xml", FileStorage::WRITE);
	//fs<<"truth"<<trainLabel;
	
	//cout<<trainData.rows<<"\t"<<trainData.cols<<endl;
	//cout<<trainLabel.rows<<"\t"<<trainLabel.cols<<endl;
	
	fs.release();
	CvSVM SVM;
	SVM.load("e:\\400_svm.xml");
	cout<<"load SVM complete\n";
	int err=0,testNum=0;
	Mat predictData(1,81,CV_32FC1);
	clock_t start=clock();
	bfs::path p;
	bfs::directory_iterator it,endIter;
	p=bfs::path("E:\\positive" );
	it=bfs::directory_iterator(p);
	for(;it!=endIter;it++){
		string path=it->path().string();
		//cout<<path<<endl;
		Mat grayImage=imread(path,0);
		vector<float> t=GenerateHogFeature(grayImage);
		for(int i=0;i<t.size();++i){
			predictData.at<float>(i)=t[i];
			//cout<<t[i]<<endl;
		}
		//cout<<predictData.rows<<"\t"<<predictData.cols<<endl;
		float response=SVM.predict(predictData);
		if(response<0)++err;
		++testNum;
	}
	//p=bfs::path("C:\\Users\\Administrator\\Desktop\\1101214006_hw3\\nontext_validation" );
	//it=bfs::directory_iterator(p);
	//for(;it!=endIter;it++){
	//	string path=it->path().string();
	//	//cout<<path<<endl;
	//	Mat grayImage=imread(path,0);
	//	vector<float> t=GenerateHogFeature(grayImage);
	//	for(int i=0;i<t.size();++i){
	//		predictData.at<float>(i)=t[i];
	//		//cout<<t[i]<<endl;
	//	}
	//	//cout<<predictData.rows<<"\t"<<predictData.cols<<endl;
	//	float response=SVM.predict(predictData);
	//	if(response>0)++err;
	//	++testNum;
	//}
	cout<<testNum<<"images, time costs: "<<double(clock()-start)/CLOCKS_PER_SEC<<"s\n";
	cout<<"error:"<<err<<endl;
	cout<<"precision:"<<1-err*1.0/testNum<<endl;
	return 0;
}


int cejisuanbixiankuandu(){
	string p;
	p="C:\\Users\\Administrator\\Desktop\\mser\\do.jpg";
	Mat image = imread(p, 0);
	Rect t=Rect(0,0,image.rows,image.cols);
	int strokeWidth=ComputeStrokeWidth(image,t,128,-1);
	//FileStorage fs("C:\\Users\\Administrator\\Desktop\\mser\\t.xml",FileStorage::WRITE);
	//fs<<"t"<<image;
	//fs.release();
	ofstream f("C:\\Users\\Administrator\\Desktop\\mser\\t.txt");
	f<<image;
	f.close();
	cout<<strokeWidth<<endl;
	return 0;
}

int xunlianjulicanshu(){
	ofstream f("e:\\julixunlian.txt");
	vector<map<pair<int,int>,vector<vector<double> > > >feature;
	for(int i=100;i<=105;++i){
		string p,gt;
		string a=lexical_cast <string>(i);
		cout<<i<<endl;
		p="E:\\train-textloc\\"+a+".jpg";
		gt="E:\\train-textloc\\gt_"+a+".txt"; 
		//draw = imread(p, 1);
		vector<Rect> groundTruthRect=readGroundTruth(gt);
		if(groundTruthRect.size()<2){
			continue;
		}
		Mat grayImg=imread(p,0);
		vector<vector<ccRegion> >regions;
		vector<vector<ccRegion *> >resultRegionPtr;
		regions.resize(groundTruthRect.size());
		ccMSER t(5, 60, 1440,0.25, 0.2,200, 1.01,0.003, 5);
		map<pair<int,int>,vector<vector<double> > > imageFeature;
		for(int j=0;j<groundTruthRect.size();++j){
			t(grayImg(groundTruthRect[j]),regions[j]);
			vector<ccRegion *> res;
			for(int k=0;k<regions[j].size();k++){
				if(regions[j][k].parent==NULL){
					Union(res,TreeAccumulation(LinearReduction(&regions[j][k])));
				}
			}
			for(int k=0;k<res.size();++k){
				(*res[k]).rect.x+=groundTruthRect[j].x;
				(*res[k]).rect.y+=groundTruthRect[j].y;
			}
			resultRegionPtr.push_back(res);
		}
		vector<double> featureVector;
		for(int j=0;j<resultRegionPtr.size();++j){
			for(int k=j;k<resultRegionPtr.size();++k){
				vector<vector<double> > mserFeature;
				if(j==k){
					for(int p=0;p<resultRegionPtr[j].size();++p){
						for(int q=p+1;q<resultRegionPtr[j].size();++q){
							ComputeFeature(grayImg,*(resultRegionPtr[j][p]),*(resultRegionPtr[j][q]),featureVector);
							f<<"1";
							for(int r=0;r<featureVector.size();++r){
								f<<"\t"<<featureVector[r];
							}
							f<<"\n";
							mserFeature.push_back(featureVector);
						}
					}
				}
				else{
					for(int p=0;p<resultRegionPtr[j].size();++p){
						for(int q=0;q<resultRegionPtr[k].size();++q){
							ComputeFeature(grayImg,*(resultRegionPtr[j][p]),*(resultRegionPtr[k][q]),featureVector);
							f<<"-1";
							for(int r=0;r<featureVector.size();++r){
								f<<"\t"<<featureVector[r];
							}
							f<<"\n";
							mserFeature.push_back(featureVector);
						}
					}
				}
				imageFeature[make_pair(j,k)]=mserFeature;
			}
		}
		cout<<"map:"<<imageFeature.size()<<endl;
		cout<<"region:"<<resultRegionPtr.size()<<endl;
		feature.push_back(imageFeature);
	}
	cout<<"done\n";
	return 0;
}


int main(){
	//svm();
	xunlianjulicanshu();
	return 0;
}