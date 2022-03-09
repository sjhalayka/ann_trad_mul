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

const size_t num_components = 4; // quaternions
const float grid_max = 8;
const float grid_min = -grid_max;
const size_t res = 25;

const size_t num_sessions = 1;
const size_t max_iterations = 10000;
const float min_error = 0.0001f;

float inputTrainingDataArray[res*res*res*res][num_components * 2];
float outputTrainingDataArray[res*res*res*res][num_components];
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




template<class T, size_t N = 4>
void get_marched_quaternion_vertices(vector<vertex<T, 4>>& grid)
{
	vertex<float, 4> Z;
	grid.clear();

	const float step_size = (grid_max - grid_min) / (res - 1);

	for (size_t i = 0; i < Z.vd.size(); i++)
		Z.vd[i] = grid_min;

	for (size_t i0 = 0; i0 < res; i0++, Z.vd[0] += step_size)
	{
		Z.vd[1] = grid_min;

		for (size_t i1 = 0; i1 < res; i1++, Z.vd[1] += step_size)
		{
			Z.vd[2] = grid_min;

			for (size_t i2 = 0; i2 < res; i2++, Z.vd[2] += step_size)
			{
				Z.vd[3] = grid_min;

				for (size_t i3 = 0; i3 < res; i3++, Z.vd[3] += step_size)
				{
					grid.push_back(Z);
				}
			}
		}
	}
}


void get_train_data(void)
{
	vector<vertex<float, num_components>> grid;
	get_marched_quaternion_vertices(grid);

	vector<input_datum> input_data;
	vector<output_datum> output_data;

	for (size_t i = 0; i < grid.size(); i++)
	{
		for (size_t j = 0; j < grid.size(); j++)
		{
			vertex<float, num_components> in_a = grid[i];
			vertex<float, num_components> in_b = grid[j];

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
	}

	for (size_t i = 0; i < grid.size(); i++)
		for (size_t j = 0; j < num_components * 2; j++)
			inputTrainingDataArray[i][j] = input_data[i].input[j];

	inputTrainingData = Mat(static_cast<int>(grid.size()), num_components * 2, CV_32F, inputTrainingDataArray);

	for (size_t i = 0; i < grid.size(); i++)
		for (size_t j = 0; j < num_components; j++)
			outputTrainingDataArray[i][j] = output_data[i].output[j];

	outputTrainingData = Mat(static_cast<int>(grid.size()), num_components, CV_32F, outputTrainingDataArray);
}

template<class T, size_t N>
vertex<T, N> predict_answer(Ptr<ANN_MLP> &mlp, const vertex<T, N>& in_a, const vertex<T, N>& in_b)
{
	float inputArray[num_components * 2];

	for (size_t i = 0; i < num_components; i++)
		inputArray[i] = in_a.vd[i];

	for (size_t i = num_components; i < num_components * 2; i++)
		inputArray[i] = in_b.vd[i - num_components];	

	Mat sample = Mat(1, num_components * 2, CV_32F, inputArray);
	Mat result;
	mlp->predict(sample, result);

	vertex<T, N> output;

	for (size_t i = 0; i < output.vd.size(); i++)
		output.vd[i] = result.at<float>(0, static_cast<int>(i));

	return output;
}



int main(void)
{
	srand(static_cast<unsigned>(time(0)));

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

	cout << "Session " << 1 << endl;

	get_train_data();
	Ptr<TrainData> trainingData = TrainData::create(inputTrainingData, SampleTypes::ROW_SAMPLE, outputTrainingData);
	mlp->train(trainingData);

	for (size_t i = 1; i < num_sessions; i++)
	{
		cout << "Session " << i + 1 << endl;

		get_train_data();
		trainingData = TrainData::create(inputTrainingData, SampleTypes::ROW_SAMPLE, outputTrainingData);
		mlp->train(trainingData, ANN_MLP::TrainFlags::UPDATE_WEIGHTS | ANN_MLP::TrainFlags::NO_INPUT_SCALE | ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);
	}


	//cv::FileStorage fs;
	//fs.open("mlp.xml", cv::FileStorage::WRITE);
	//mlp->write(fs);
	//fs.release();

	//Ptr<ANN_MLP> mlp2 = ANN_MLP::create();
	//fs.open("mlp.xml", cv::FileStorage::READ);
	//mlp2->read(fs.root());
	//fs.release();

	vertex<float, num_components> in_a, in_b;

	for (size_t i = 0; i < in_a.vd.size(); i++)
		in_a.vd[i] = get_one_to_neg_one() * grid_max;

	for (size_t i = 0; i < in_b.vd.size(); i++)
		in_b.vd[i] = get_one_to_neg_one() * grid_max;

	vertex<float, num_components> answer = predict_answer(mlp, in_a, in_b);

	for (size_t i = 0; i < in_a.vd.size(); i++)
		cout << in_a.vd[i] << " ";

	cout << endl;

	for (size_t i = 0; i < in_b.vd.size(); i++)
		cout << in_b.vd[i] << " ";

	cout << endl;

	for (size_t i = 0; i < answer.vd.size(); i++)
		cout << answer.vd[i] << " ";

	cout << endl;

	vertex<float, num_components> trad_answer = traditional_mul(in_a, in_b);

	for (size_t i = 0; i < trad_answer.vd.size(); i++)
		cout << trad_answer.vd[i] << " ";

	cout << endl;

	return 0;
}
