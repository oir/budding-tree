/* Ozan Irsoy 
 * 
 * Common functions and vector operator overloads
 */

#include <cmath>
#include <vector>
#include <cassert>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include "Eigen/Dense"

#define uint unsigned int

using namespace std;
using namespace Eigen;

double dot(const VectorXd& x, const VectorXd& y) {
  return x.dot(y);
}

uint argmax(const VectorXd& x) {
  double max = x(0);
  uint maxi = 0;
  for (uint i=1; i<x.size(); i++) {
    if (x(i) > max) {
      max = x(i);
      maxi = i;
    }
  }
  return maxi;
}

double str2double(const string& s) {
  istringstream i(s);
  double x;
  if (!(i >> x))
    return 0;
  return x;
}

vector<string> &split(const string &s, char delim, vector<string> &elems) {
  stringstream ss(s);
  string item;
  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  split(s, delim, elems);
  return elems;
}

template <class T>
void shuffle(vector<T>& v) {  // KFY shuffle
  for (uint i=v.size()-1; i>0; i--) {
    uint j = (rand() % i);
    T tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
  }
}

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

MatrixXd sigmoid(const MatrixXd& x) {
  return 1/(1+(-x.array()).exp());
}

MatrixXd sigmoidp(const MatrixXd& y) {
  return y.array() * (1-y.array()); 
}

double logsumexp(const VectorXd &x) {
  double m = x.maxCoeff();
  return log((x.array() - m).exp().sum()) + m;
}

VectorXd softmax(const VectorXd &x) {
  return (x.array() - logsumexp(x)).exp();
}

double urand(double min, double max) {
  return (double(rand())/RAND_MAX)*(max-min) + min;
}

