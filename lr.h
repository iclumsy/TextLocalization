/********************************************************************
* Logistic Regression Classifier V0.10
* Implemented by Rui Xia(rxia@nlpr.ia.ac.cn) , Wang Tao（wangbogong@gmail.com）
* Last updated on 2012-6-12. 
*********************************************************************/
#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <climits>
#include <math.h>
#include <time.h>

# define VERSION       "V0.10"
# define VERSION_DATE  "2012-6-12"

using namespace std;

const long  double min_threshold=1e-300;

struct sparse_feat                       //稀疏特征表示结构
{
	vector<int> id_vec;
	vector<float> value_vec;
};

class LR                                 //logistic regression实现类
{
public:
	vector<sparse_feat> samp_feat_vec;
	vector<int> samp_class_vec;
	int feat_set_size;
	int class_set_size;
	vector< vector<float> > omega; //模型的参数矩阵omega = feat_set_size * class_set_size

public:
	LR();
	~LR();
	void save_model(string model_file);
	void load_model(string model_file);
	void load_training_file(string training_file);
	void init_omega();

	int train_online(int max_loop, double loss_thrd, float learn_rate, float lambda, int avg);    //logistic regression随机梯度优化算法
	vector<float> calc_score(sparse_feat &samp_feat);
	vector<float> score_to_prb(vector<float> &score);
	int score_to_class(vector<float> &score);

	float classify_testing_file(string testing_file, string output_file, int output_format);      //模型分类预测

private:
	void read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec);   //更新函数
	void update_online_ce(int samp_class, sparse_feat &samp_feat, float learn_rate, float lambda);
	void calc_loss_ce(double *loss, float *acc);                                                              //计算损失函数
	float calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec);    
	float sigmoid(float x);
	vector<string> string_split(string terms_str, string spliting_tag);

};