#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include "BuddingTree.cpp"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

// Regression toy dataset generator
void generateToy(MatrixXd &X, 
                 MatrixXd &Y) {
  default_random_engine gen(123);
  normal_distribution<double> randnorm;
  int d=1; // dimension
  int n=300; // number of instances
  int i,j;

  X = MatrixXd(d,n);
  Y = MatrixXd(1,n);

  for (i=0; i<n; i++) {
    X(0,i) = (((double)rand()/RAND_MAX)*6-3);
    Y(0,i) = (sin(X(0,i)) + randnorm(gen)*0.04);
  }
}

int main() {
  srand(123457);
  MatrixXd X, V, U;
  MatrixXd Y, R, T;
  
  cout.precision(5);
  cout.setf(ios::fixed,ios::floatfield);
  
  generateToy(X, Y); // training set
  generateToy(V, R); // validation set

  double minErr = 1e10;
  double bestEta, bestLambda;

  double eta = 0.4;
  double lambda = 0.00001;
  double lambdaL2 = 0.00006;
  
  srand(122);
  BuddingTree t;
  t.setParams(eta, lambda, lambdaL2);
  t.train(X,Y);
  cout << t.meanSqErr(V,R) << endl;
}
