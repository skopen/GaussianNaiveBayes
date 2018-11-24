#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d,
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/

	vector<double> labeledData[possible_labels.size()][4];

	// separate data by labels
	for (unsigned int i = 0; i < labels.size(); i++)
	{
		string labelStr = labels[i];
		vector<double> d = data[i];

		int label = std::distance(possible_labels.begin(), std::find(possible_labels.begin(), possible_labels.end(), labelStr));

		for (int j = 0; j < 4; j++)
		{
			labeledData[label][j].push_back(d[j]);
		}

		PofCk[label]++;
	}

	// find P(Ck), mu and sigma for the entire data set
	for (unsigned int i = 0; i < possible_labels.size(); i++)
	{
		PofCk[i] /= labels.size();

		for (int j = 0; j < 4; j++)
		{
			mu[i][j] = average(labeledData[i][j]);
			sig[i][j] = stddev(labeledData[i][j], mu[i][j]);

			//cout << "mu[i][j]:" << mu[i][j] << " sig[i][j]: " << sig[i][j] << endl;
		}
	}

	//cout << "P(Ck=0):" << PofCk[0] << " P(Ck=1):" << PofCk[1] << " P(Ck=2):" << PofCk[2] << " TOTAL: " << PofCk[0]+PofCk[1]+PofCk[2] << endl;
}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/
	double prodP[possible_labels.size()];

	for (unsigned int i = 0; i < possible_labels.size(); i++)
	{
		prodP[i] = 1.0;

		for (int j = 0; j < 4; j++)
		{
			prodP[i] *= pdf(mu[i][j], sig[i][j], sample[j]);
		}

		prodP[i] *= PofCk[i];
	}

	int max = distance(prodP, max_element(prodP, prodP + possible_labels.size()));

	return this->possible_labels[max];

}

double GNB::average (vector<double> v)
{
    double sum = 0;
    for(unsigned int i=0; i<v.size(); i++)
           sum += v[i];
    return sum/v.size();
}

double GNB::stddev (vector<double> v, double avg)
{
    double E = 0.0;
    for(unsigned int i=0; i<v.size(); i++)
        E += (v[i] - avg)*(v[i] - avg);
    return sqrt(1.0/v.size() * E);
}

double GNB::pdf (double mu, double sig, double x)
{
	double p = (1.0/sqrt(2 * M_PI * sig*sig)) * exp(-(x-mu)*(x-mu)/(2*sig*sig));

	return p;
}
