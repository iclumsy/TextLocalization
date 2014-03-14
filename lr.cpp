#include "lr.h"

LR::LR()
{

}

LR::~LR()
{
}

void LR::save_model(string model_file)
{ 
	cout << "Saving model..." << endl;
	ofstream fout(model_file.c_str());
	for (int k = 0; k < feat_set_size; k++) {
		for (int j = 0; j < class_set_size; j++) {
			fout << omega[k][j] << " ";
		}
		fout << endl;
	}
	fout.close();
}


void LR::load_model(string model_file)
{
	cout << "Loading model..." << endl;
	omega.clear();
	ifstream fin(model_file.c_str());
	if(!fin) {
		cerr << "Error opening file: " << model_file << endl;
		exit(0);
	}
	string line_str;
	while (getline(fin, line_str)) {
		vector<string> line_vec = string_split(line_str, " ");
		vector<float>  line_omega;
		for (vector<string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
			float weight = (float)atof(it->c_str());
			line_omega.push_back(weight);
		}
		omega.push_back(line_omega);
	}
	fin.close();
	feat_set_size = (int)omega.size();
	class_set_size = (int)omega[0].size();
}


void LR::read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec) {
	ifstream fin(samp_file.c_str());
	if(!fin) {
		cerr << "Error opening file: " << samp_file << endl;
		exit(0);
	}
	string line_str;
	while (getline(fin, line_str)) 
	{
		size_t class_pos = line_str.find_first_of("\t");
		int class_id = atoi(line_str.substr(0, class_pos).c_str());
		samp_class_vec.push_back(class_id);
		string terms_str = line_str.substr(class_pos+1);
		sparse_feat samp_feat;
		samp_feat.id_vec.push_back(0); // bias
		samp_feat.value_vec.push_back(1); // bias
		if (terms_str != "") 
		{
			vector<string> fv_vec = string_split(terms_str, " ");
			for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) 
			{
				size_t feat_pos = it->find_first_of(":");
				int feat_id = atoi(it->substr(0, feat_pos).c_str());
				float feat_value = (float)atof(it->substr(feat_pos+1).c_str());
				samp_feat.id_vec.push_back(feat_id);
				samp_feat.value_vec.push_back(feat_value);
			}
		}
		samp_feat_vec.push_back(samp_feat);
	}
	fin.close();
}


void LR::load_training_file(string training_file)
{
	cout << "Loading training data..." << endl;
	//read_samp_file(training_file, samp_feat_vec, samp_class_vec);
	feat_set_size = 0;
	class_set_size = 0;
	for (size_t i = 0; i < samp_class_vec.size(); i++) {
		if (samp_class_vec[i] > class_set_size) {
			class_set_size = samp_class_vec[i];
		}
		if (samp_feat_vec[i].id_vec.back() > feat_set_size) {
			feat_set_size = samp_feat_vec[i].id_vec.back();
		}    
	}
	class_set_size += 1;
	feat_set_size += 1;
}

void LR::init_omega()
{
	float init_value = 0.0;
	//float init_value = (float)1/class_set_size;
	for (int i = 0; i < feat_set_size; i++) 
	{
		vector<float> temp_vec(class_set_size, init_value);
		omega.push_back(temp_vec);
	}
}

// Stochastic Gradient Descent (SGD) optimization for the criteria of  Cross Entropy (CE)
int LR::train_online( int max_loop, double loss_thrd, float learn_rate, float lambda, int avg)
{
	int id = 0;
	double loss = 0.0;
	double loss_pre = 0.0;
	vector< vector<float> > omega_pre=omega;
	float acc=0.0;

	vector< vector<float> > omega_sum(omega);

	while (id <= max_loop*(int)samp_class_vec.size()) 
	{


		if (id%samp_class_vec.size() == 0)    // ���һ�ε�����Ԥ��������
		{
			int loop = id/(int)samp_class_vec.size();               //check loss
			loss = 0.0;
			acc = 0.0;

			calc_loss_ce(&loss, &acc);
			cout.setf(ios::left);
			if(loop==100)cout << "Iter: " << setw(8) << loop << "Loss: " << setw(18) << loss << "Acc: " << setw(8) << acc << endl;
			if ((loss_pre - loss) < loss_thrd && loss_pre >= loss && id != 0)
			{
				cout << "Reaching the minimal loss value decrease!" << endl;
				break;
			}
			loss_pre = loss;

			if (id)            //��ʾ��һ�β������������
			{
				for (int i=0;i!=omega_pre.size();i++)
					for (int j=0;j!=omega_pre[i].size();j++)
						omega[i][j]+=omega_pre[i][j]*lambda;
			}

			omega_pre=omega;
		}

		// update omega
		int r = (int)(rand()%samp_class_vec.size());
		sparse_feat samp_feat = samp_feat_vec[r];
		int samp_class = samp_class_vec[r];


		update_online_ce(samp_class, samp_feat, learn_rate, lambda);

		if (avg == 1 && id%samp_class_vec.size() == 0) 
		{
			for (int i = 0; i < feat_set_size; i++) 
			{
				for (int j = 0; j < class_set_size; j++) 
				{
					omega_sum[i][j] += omega[i][j];
				}
			}            
		}
		id++;
	}

	if (avg == 1) 
	{
		for (int i = 0; i < feat_set_size; i++) 
		{
			for (int j = 0; j < class_set_size; j++)
			{
				omega[i][j] = (float)omega_sum[i][j] / id;
			}
		}        
	}
	return 0;
}

void LR::update_online_ce(int samp_class, sparse_feat &samp_feat, float learn_rate, float lambda)
{

	vector<float> score=calc_score(samp_feat);//(W'*X)
	vector<float> softMaxVec(class_set_size);

	float maxScore=*(max_element(score.begin(),score.end()));
	float softMaxSum=0;

	for (int j=0;j<class_set_size;j++)
	{
		softMaxVec[j]=exp(score[j]-maxScore);
		softMaxSum+=softMaxVec[j];                             //ͬʱ������score;
	}
	for (int k=0;k<class_set_size;k++)
		softMaxVec[k]/=softMaxSum;


	for (int i=0;i<class_set_size;i++)                          //����ÿһ����
	{
		float error_term=((int)(i==samp_class)-softMaxVec[i]);
		for (int j=0;j<samp_feat.id_vec.size();j++)             //����ÿ�����е�
		{
			int feat_id=samp_feat.id_vec[j];
			float feat_value=samp_feat.value_vec[j];
			float delt=error_term*feat_value;
			//float regul = lambda * omega[feat_id][i];
			omega[feat_id][i]+=learn_rate*delt;

		}

	}


}

void LR::calc_loss_ce(double *loss, float *acc)
{
	double loss_value = 0.0;
	int err_num = 0;

	for (size_t i = 0; i < samp_class_vec.size(); i++) 
	{
		int samp_class = samp_class_vec[i];
		sparse_feat samp_feat = samp_feat_vec[i];

		vector<float> score = calc_score(samp_feat);
		vector<float> softMaxVec(class_set_size);
		float softMaxSum=0;

		int pred_class = score_to_class(score);
		if (pred_class != samp_class) 
		{
			err_num += 1;
		}

		float maxScore=*(max_element(score.begin(),score.end()));
		for (int k=0;k<class_set_size;k++)
		{
			softMaxVec[k]=exp(score[k]-maxScore);
			softMaxSum+=softMaxVec[k];                     //ͬʱ������score;
		}

		for (int j = 0; j < class_set_size; j++)
		{
			if (j == samp_class) 
			{ 

				double yi=softMaxVec[j]/softMaxSum;
				long double temp=yi<min_threshold ? min_threshold:yi;
				loss_value -= log(temp);                             

			}


		}
	}

	*loss = loss_value ;
	*acc = 1 - (float)err_num / samp_class_vec.size();
}

vector<float> LR::calc_score(sparse_feat &samp_feat)
{
	vector<float> score(class_set_size, 0);
	for (int j = 0; j < class_set_size; j++)
	{
		for (size_t k = 0; k < samp_feat.id_vec.size(); k++) 
		{
			int feat_id = samp_feat.id_vec[k];
			float feat_value = samp_feat.value_vec[k];
			score[j] += omega[feat_id][j] * feat_value;
		}
	}
	return score;
}

vector<float> LR::score_to_prb(vector<float> &score)
{   
	//vector<float> prb(class_set_size, 0);
	//for (int i = 0; i < class_set_size; i++) 
	//{
	//    float delta_prb_sum = 0.0;
	//    for (int j = 0; j < class_set_size; j++) 
	//    {
	//        delta_prb_sum += exp(score[j] - score[i]);
	//    }
	//    prb[i] = 1 / delta_prb_sum;
	//}
	//return prb;

	vector<float> softMaxVec(class_set_size);

	float maxScore=*(max_element(score.begin(),score.end()));
	float softMaxSum=0;

	for (int j=0;j<class_set_size;j++)
	{
		softMaxVec[j]=exp(score[j]-maxScore);
		softMaxSum+=softMaxVec[j];                             //ͬʱ������score;
	}
	for (int k=0;k<class_set_size;k++)
		softMaxVec[k]/=softMaxSum;

	return softMaxVec;

}


int LR::score_to_class(vector<float> &score)
{
	int pred_class = 0;    
	float max_score = score[0];
	for (int j = 1; j < class_set_size; j++) {
		if (score[j] > max_score) {
			max_score = score[j];
			pred_class = j;
		}
	}
	return pred_class;
}

float LR::classify_testing_file(string testing_file, string output_file, int output_format)
{
	cout << "Classifying testing file..." << endl;
	vector<sparse_feat> test_feat_vec;
	vector<int> test_class_vec;
	vector<int> pred_class_vec;
	read_samp_file(testing_file, test_feat_vec, test_class_vec);
	ofstream fout(output_file.c_str());
	for (size_t i = 0; i < test_class_vec.size(); i++) 
	{
		int samp_class = test_class_vec[i];
		sparse_feat samp_feat = test_feat_vec[i];
		vector<float> pred_score = calc_score(samp_feat);            
		int pred_class = score_to_class(pred_score);
		pred_class_vec.push_back(pred_class);
		fout << pred_class << "\t"<<samp_class<<"\t";
		if (output_format == 1) 
		{
			for (int j = 0; j < class_set_size; j++) 
			{
				fout << j << ":" << pred_score[j] << ' '; 
			}        
		}
		else if (output_format == 2) 
		{
			vector<float> pred_prb = score_to_prb(pred_score);
			for (int j = 0; j < class_set_size; j++)
			{
				fout << j << ":" << pred_prb[j] << ' '; 
			}
		}

		fout << endl;        
	}
	fout.close();
	float acc = calc_acc(test_class_vec, pred_class_vec);
	return acc;
}

float LR::calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec)
{
	size_t len = test_class_vec.size();
	if (len != pred_class_vec.size()) {
		cerr << "Error: two vectors should have the same lenght." << endl;
		exit(0);
	}
	int err_num = 0;
	for (size_t id = 0; id != len; id++) {
		if (test_class_vec[id] != pred_class_vec[id]) {
			err_num++;
		}
	}
	return 1 - ((float)err_num) / len;
}

float LR::sigmoid(float x) 
{
	double sgm = 1 / (1+exp(-(double)x));
	return (float)sgm;
}

vector<string> LR::string_split(string terms_str, string spliting_tag)
{
	vector<string> feat_vec;
	size_t term_beg_pos = 0;
	size_t term_end_pos = 0;
	while ((term_end_pos = terms_str.find_first_of(spliting_tag, term_beg_pos)) != string::npos) 
	{
		if (term_end_pos > term_beg_pos)
		{
			string term_str = terms_str.substr(term_beg_pos, term_end_pos - term_beg_pos);
			feat_vec.push_back(term_str);
		}
		term_beg_pos = term_end_pos + 1;
	}
	if (term_beg_pos < terms_str.size())
	{
		string end_str = terms_str.substr(term_beg_pos);
		feat_vec.push_back(end_str);
	}
	return feat_vec;
}