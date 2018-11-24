#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

	vector<string> possible_labels = {"left","keep","right"};

	double mu[3][4];
	double sig[3][4];
	double PofCk[3] = {0.0, 0.0, 0.0};

	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double>);

    double average (vector<double> v);

    double stddev (vector<double> v, double avg);

    double pdf (double mu, double sig, double x);
};

#endif



