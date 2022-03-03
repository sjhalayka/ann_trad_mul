// https://stackoverflow.com/questions/37500713/opencv-image-recognition-setting-up-ann-mlp

#include <opencv2/opencv.hpp>
using namespace cv;
#pragma comment(lib, "opencv_world3416.lib")

#include <iostream>
#include <iomanip>

using namespace cv;
using namespace ml;
using namespace std;

#include "tmath.h"

float get_one_to_neg_one(void)
{
	float r = rand() / static_cast<float>(RAND_MAX);
	r *= 2;
	r -= 1;

	return r;
}

const size_t num_components = 8;
const float grid_max = 10;
const size_t num_training_samples = 10;
const size_t num_sessions = 10;
const size_t max_iterations = 10000;
const float min_error = 0.0001f;

float inputTrainingDataArray[num_training_samples][num_components * 2];
float outputTrainingDataArray[num_training_samples][num_components];
Mat inputTrainingData;
Mat outputTrainingData;



class input_datum
{
public:
	float input[num_components*2];
};

class output_datum
{
public:
	float output[num_components];
};


void get_train_data(void)
{
	vector<input_datum> input_data;
	vector<output_datum> output_data;

	for (size_t i = 0; i < num_training_samples; i++)
	{
		vertex<float, num_components> in_a;

		for (size_t j = 0; j < in_a.vd.size(); j++)
			in_a.vd[j] = get_one_to_neg_one() * grid_max;

		vertex<float, num_components> in_b;

		for (size_t j = 0; j < in_b.vd.size(); j++)
			in_b.vd[j] = get_one_to_neg_one() * grid_max;

		vertex<float, num_components> answer = traditional_mul(in_a, in_b);

		input_datum id;

		for (size_t j = 0; j < num_components; j++)
			id.input[j] = in_a.vd[j];

		for (size_t j = num_components; j < num_components * 2; j++)
			id.input[j] = in_b.vd[j - num_components];

		input_data.push_back(id);

		output_datum od;

		for (size_t j = 0; j < num_components; j++)
			od.output[j] = answer.vd[j];

		output_data.push_back(od);
	}

	for (size_t i = 0; i < num_training_samples; i++)
		for (size_t j = 0; j < num_components * 2; j++)
			inputTrainingDataArray[i][j] = input_data[i].input[j];

	inputTrainingData = Mat(num_training_samples, num_components * 2, CV_32F, inputTrainingDataArray);

	for (size_t i = 0; i < num_training_samples; i++)
		for (size_t j = 0; j < num_components; j++)
			outputTrainingDataArray[i][j] = output_data[i].output[j];

	outputTrainingData = Mat(num_training_samples, num_components, CV_32F, outputTrainingDataArray);
}



int main(void)
{
	srand(1234);



	Ptr<ANN_MLP> mlp = ANN_MLP::create();

	Mat layersSize = Mat(3, 1, CV_16U);
	layersSize.row(0) = Scalar(num_components * 2);
	layersSize.row(1) = Scalar(static_cast<int>(sqrt(num_components * 2 * num_components)));
	layersSize.row(2) = Scalar(num_components);
	mlp->setLayerSizes(layersSize);

	mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);  

	TermCriteria termCrit = TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, max_iterations, min_error);
	mlp->setTermCriteria(termCrit);

	mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);

	get_train_data();
	Ptr<TrainData> trainingData = TrainData::create(inputTrainingData, SampleTypes::ROW_SAMPLE, outputTrainingData);
	mlp->train(trainingData);

	for (size_t i = 0; i < num_sessions - 1; i++)
	{
		get_train_data();
		trainingData = TrainData::create(inputTrainingData, SampleTypes::ROW_SAMPLE, outputTrainingData);
		mlp->train(trainingData, ANN_MLP::TrainFlags::UPDATE_WEIGHTS | ANN_MLP::TrainFlags::NO_INPUT_SCALE | ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);
	}


	/*cv::FileStorage fs;
	fs.open("mlp.xml", cv::FileStorage::WRITE);
	mlp->write(fs);
	fs.release();

	Ptr<ANN_MLP> mlp2 = ANN_MLP::create();
	fs.open("mlp.xml", cv::FileStorage::READ);
	mlp2->read(fs.root());
	fs.release();*/

	for (int i = 0; i < inputTrainingData.rows; i++)
	{
		Mat sample = Mat(1, inputTrainingData.cols, CV_32F, inputTrainingDataArray[i]);
		Mat result;
		mlp->predict(sample, result);
		cout << sample << " -> " << result << endl;
		//cout << endl;
	}

	return 0;
}
