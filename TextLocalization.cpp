// text.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include <iostream>
#include <stack>
#include <queue>
#include "opencv2/opencv.hpp"
#include "ccer.hpp"

#include "LogisticRegression.hpp"
//#include <assert.h>
#include "ccMSER.h"
using namespace std;
using namespace cv;
using namespace LogisticRegression;
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
void ComputeStrokeWidthHelper(Mat &img,int level,double startx,double starty,double diagonallength,double xincrease,double yincrease,int &strokewidth){
	int currentwidth=0;
	double x=startx,y=starty;
	for(int i=0;i<diagonallength;++i){
		int tx=floor(x+0.5),ty=floor(y+0.5);
		if(tx>=img.cols||ty>=img.rows){
			break;
		}
		//cout<<"tx:"<<tx<<"\tty:"<<ty<<endl;
		if(img.at<uchar>(ty,tx)>=level){
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
int ComputeStrokeWidth(Mat &img,Rect rect,int level){
	double xincrease,yincrease,diagonallength;
	diagonallength=sqrt(rect.width*rect.width+rect.height*rect.height);
	xincrease=rect.width/diagonallength;
	yincrease=rect.height/diagonallength;
	int strokewidth=INT_MAX,currentwidth;
	ComputeStrokeWidthHelper(img,level,rect.x,rect.y,diagonallength,xincrease,yincrease,strokewidth);
	ComputeStrokeWidthHelper(img,level,rect.x,rect.y+rect.height,diagonallength,xincrease,-yincrease,strokewidth);
	ComputeStrokeWidthHelper(img,level,rect.x+0.5*rect.width,rect.y,diagonallength,0,yincrease,strokewidth);
	ComputeStrokeWidthHelper(img,level,rect.x,rect.y+0.5*rect.height,diagonallength,xincrease,0,strokewidth);
	return strokewidth;
}
void ComputeFeature(Mat &img,ccRegion eru,ccRegion erv,vector<double> &featureVector){
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
	int ustrokewidth=1;//ComputeStrokeWidth(img,u,eru.level);
	int vstrokewidth=1;//ComputeStrokeWidth(img,v,erv.level);
	double strokewidthdifference=abs(ustrokewidth-vstrokewidth)/max(ustrokewidth,vstrokewidth);
	featureVector.push_back(strokewidthdifference);
}
double ComputeDistance(Mat &img,ccRegion eru,ccRegion erv,vector<double> &featureWeights){
	vector<double> featureVector;
	ComputeFeature(img,eru,erv,featureVector);
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

vector<vector<int>> HierarchicalClustering(Mat &img,vector<ccRegion *> &data,vector<double> &featureweights,double threshold){
	int N=data.size();
	vector<vector<distances> >dist;// 2d vector for storing distances matrix
	vector<multiset<distances, Cmp>> P(N);// multiset for storing sorted distances
	vector<int> active(N,1);// vector for storing flags for marking currently active clusters
	vector<vector<int>> A;// 2d vector for storing lists of titles in clusters
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
		cout<<"min dist"<<min_dist<<endl;
		if(min_dist>threshold){
			break;
		}
		// we have minimum distance
		// k1, k2 - indexes of most nearest clusters
		int k1 = min_index;
		int k2 = P[k1].begin()->index;

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
		if(active[i]==1&&A[i].size()>1)
		{
			Rect tmp=data[A[i][0]]->rect;
			++class_num;
			cout<<std::endl<<"Class number: "<<class_num<<std::endl<<std::endl;
			for(int j=0;j<A[i].size();j++){
				tmp=tmp|data[A[i][j]]->rect;
				char b[100];
				sprintf(b,"%d---%f",class_num,data[A[i][j]]->var);
				string a(b);
				//imwrite("e://mserout//"+a+".jpg",draw(data[A[i][j]]->rect));
				cout<<A[i][j]<<std::endl;
			}
			rectangle(draw,tmp,Scalar(0,0,255),2);
		}
	}
	return A;
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

int main2(int argc, char** argv){

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

int main(){
	string p="C:\\Users\\Administrator\\Desktop\\mser\\385.jpg";
	Mat image = imread(p, 0);
	draw = imread(p, 1);
	ccMSER t(5, 60, 1440,0.25, .2,200, 1.01,0.003, 5);
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
			char b[100];
			sprintf(b,"%f",regions[i].var);
			string a(b);
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
	vector<vector<int>> clusterResult=HierarchicalClustering(image,res,featureWeights,2.5);
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
	imshow("result.jpg", draw);
	waitKey(0);
}