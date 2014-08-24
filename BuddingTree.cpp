#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "common.cpp"
#include "Eigen/Dense"

#define uint unsigned int

#define MINIBATCH 1
#define MAXEPOCH 200
#define AGGR_W 0.01  // w aggressiveness
#define AGGR_RHO 0.001  // rho aggressiveness
#define INHERIT false // inherit rho, w, w0 from parent

using namespace std;
using namespace Eigen;

class Node {
  public:
    Node(uint, uint, double, double, double lambdaL2 = 0, uint depth=0);
    Node(Node* parent);
    Node(const Node &);
    ~Node();
    void init(uint, uint, double, double, double lambdaL2 = 0, uint depth=0);
    VectorXd evaluate(const VectorXd&);
    VectorXd evaluate(const VectorXd&, uint);
    void computeGrad(const VectorXd&, const VectorXd&);
    void backprop(const VectorXd&, const VectorXd&);
    void update();
    void split();

    uint size();
    double effSize();

    void print(string, bool);
    void printTable(uint, string, ostream& out = cout,
                    double existence = 1, double xcoord=0,
                    double ycoord=0, double width=60, double xparent=0,
                    double yparent=0);
    void printGates(VectorXd);
    void save(ofstream&);
    void load(ifstream&);

  private:
    // params
    VectorXd w;
    MatrixXd rho;
    VectorXd rho0;
    double w0;
    double gamma;

    // hyperparams
    double lambda;    // regularizer for gamma
    double eta;       // learning rate
    double lambdaL2;  // regularizer for w

    // gradient of params for updates
    VectorXd dw;
    MatrixXd drho;
    VectorXd drho0;
    double dw0;
    double dgamma;

    // adagrad histories for params
    VectorXd adaw;
    MatrixXd adarho;
    VectorXd adarho0;
    double adaw0;
    double adagamma;

    // most recent function value,
    // for update purposes
    // (to avoid recomputing)
    VectorXd y;
    VectorXd net;
    double alpha;
    vector<VectorXd > ylist;
    vector<double> alphalist;

    // pointers & stuff
    Node* left;
    Node* right;
    bool isLeaf;
    uint depth;

    friend class BuddingTree;
};

class BuddingTree {
  public:
    BuddingTree();
    void train(const MatrixXd&, const MatrixXd&, bool);
    void train(const MatrixXd&, const MatrixXd&, char, bool);
    void setParams(double, double, double lambdaL2=0);
    VectorXd evaluate(const VectorXd&);
    
    uint size();                  // node count
    double effSize();             // effective size
    void print();                 // prints in a human readable format
    void printTable(ostream& out = cout);  // prints in a machine readable format (useful for plotting)
    void printGates();            // prints gate activations for each leaf
    void save(string filename);   // save tree to file
    void load(string filename);   // load tree from file

    double meanSqErr(const MatrixXd&,    // MSE for regression trees
                     const MatrixXd&);
    double misclassErr(const MatrixXd&,  // % Misclassification Error
                       const MatrixXd&); //    for classification trees

  private:
    Node* root;
    double lambda;
    double eta;
    double lambdaL2;
    uint dimx;
    uint dimy;
    char type;
};

BuddingTree::BuddingTree() {
  // turns out these parameters are
  // very important for regularization / overfitting
  eta = 0.2;
  lambda = 0.001;
  lambdaL2 = 0;
  type = 'r'; // default type is regression
  
  root = NULL;
}

void BuddingTree::train(const MatrixXd &X, const MatrixXd &Y,
                   bool verbose = false) {
  dimx = X.rows();
  dimy = Y.rows();
  vector<uint> perm;
  for (uint i=0; i<X.cols(); i++)
    perm.push_back(i);

  VectorXd yh;
  if(root != NULL)
    delete root;  // reset tree
  root = new Node(dimx, dimy, eta, lambda, lambdaL2);
  for (uint t=0; t<MAXEPOCH; t++) {
    shuffle(perm);
    for (uint j=0; j<Y.cols(); j++) {
      uint i = perm[j];
      yh = evaluate(X.col(i));
      // EXPERIMENTAL: early stopping for correct instances
      /*if (type == 'm' && t >= 200) {
        if (argmax(yh) == argmax(Y[i]))
          continue;
      }*/
      root->backprop(yh-Y.col(i), X.col(i)); // delta_0 = y^estimate - y^true
      if (j % MINIBATCH == MINIBATCH-1 || j == Y.cols()-1)
        root->update();              // params -= (learning rate)*gradient
    }
    if (verbose) {
      double err = misclassErr(X,Y);
      cout << "Epoch " << t << ", Tr Err: " << err << endl;
    }
  }
}

void BuddingTree::train(const MatrixXd &X, const MatrixXd &Y, 
                   char type, bool verbose = false) {
  assert(type == 'r' || type == 'b' || type == 'm');
  this->type = type;
  train(X, Y, verbose);
}

void BuddingTree::setParams(double eta, double lambda, double lambdaL2) {
  this->eta = eta;
  this->lambda = lambda;
  this->lambdaL2 = lambdaL2;
}

VectorXd BuddingTree::evaluate(const VectorXd &x) {
  if (type == 'r') // regression
    return root->evaluate(x);
  else if (type == 'b') // binary classification
    return sigmoid(root->evaluate(x));
  else if (type == 'm') // multiclass classification
    return softmax(root->evaluate(x));
}

double BuddingTree::meanSqErr(const MatrixXd &X, const MatrixXd &Y) {
  assert(Y.rows() == 1 && type == 'r'); // scalar valued regression
  double err = 0;
  for (uint i=0; i<Y.cols(); i++)
    err += pow(Y(0,i) - (double)evaluate(X.col(i))(0),2);
  err /= Y.cols();
  return err;
}

double BuddingTree::misclassErr(const MatrixXd &X, const MatrixXd &Y) {
  assert((Y.rows() == 1 && type == 'b') || type == 'm'); // 2 or k-class
  if (type == 'b') {
    double err = 0;
    for (uint i=0; i<Y.cols(); i++)
      err += abs(Y(0,i)-(int)(evaluate(X.col(i))(0) > 0.5));
    err /= Y.cols();
    return err;
  } else {
    double err = 0;
    for (uint i=0; i<Y.cols(); i++) {
      VectorXd y = evaluate(X.col(i));
      err += (int)( Y(argmax(y),i) != 1 );
    }
    err /= Y.cols();
    return err;
  }
}

uint BuddingTree::size() {
  return root->size();
}

double BuddingTree::effSize() {
  return root->effSize();
}

void BuddingTree::print() {
  cout << "Size: " << size() << endl << endl;
  root->print("", true);
}

void BuddingTree::printTable(ostream& out) {
  root->printTable(0, "1", out);
}

void BuddingTree::printGates() {
  root->printGates(VectorXd::Ones(dimx));
}

void BuddingTree::save(string filename) {
  ofstream store(filename.c_str());
  root->save(store);
  store.close();
}

void BuddingTree::load(string filename) {
  if (root != NULL)
    delete root;
  root = new Node(0,0,0,0);
  ifstream store(filename.c_str());
  root->load(store);
  store.close();
  dimx = root->w.size();
  dimy = root->rho0.size();
}

void Node::init(uint dimx, uint dimy, 
             double eta, double lambda, double lambdaL2, uint depth) {
  w = dw = adaw = VectorXd(dimx);
  for(uint i=0; i<dimx; i++) {
    w(i) = (urand(-AGGR_W,AGGR_W));
    dw(i) = (0.0);
    adaw(i) = (0.001);
  }
  rho = drho = adarho = MatrixXd(dimy,dimx);
  rho0 = drho0 = adarho0 = VectorXd(dimy);
  y = net= VectorXd(dimy);
  for(uint i=0; i<dimy; i++) {
    rho0(i) =(urand(-AGGR_RHO,AGGR_RHO));
    drho0(i) = 0;
    adarho0(i) = 0.001;
    for (uint j=0; j<dimx; j++) {
      rho(i,j) = (urand(-AGGR_RHO,AGGR_RHO));
      drho(i,j) = (0.0);
      adarho(i,j) = (0.001);
    }
    y(i) = (urand(-0.001,0.001));
    net(i) = y(i);
  }
  w0 = urand(-AGGR_W,AGGR_W);
  gamma = 1;
  dw0 = dgamma = 0;
  adaw0 = adagamma = 0.001;
  
  isLeaf = true;
  left = right = NULL;

  this->eta = eta;
  this->lambda = lambda;
  this->lambdaL2 = lambdaL2;
  this->depth = depth;
}

Node::Node(uint dimx, uint dimy, 
             double eta, double lambda, double lambdaL2, uint depth) {
  init(dimx, dimy, eta, lambda, lambdaL2, depth);
}

Node::Node(Node* parent) {
  init(parent->w.size(), parent->rho.rows(), parent->eta,
        parent->lambda, parent->lambdaL2, parent->depth);
  if (INHERIT) {
    rho += parent->rho;
    w += parent->w;
    w0 += parent->w0;
  }
}

Node::~Node() {
  if (left != NULL)
    delete left;
  if (right != NULL)
    delete right;
}

void Node::split() {
  assert(isLeaf == true);
  left = new Node(this);
  right = new Node(this);
  isLeaf = false;
}

VectorXd Node::evaluate(const VectorXd &x) {
  net = rho*x+rho0;
  if (isLeaf || gamma == 1)
    return y = net;
  else {
    alpha = sigmoid(dot(w,x)+w0);
    y = (1-gamma)*(alpha*(left->evaluate(x)) +
                  (1-alpha)*(right->evaluate(x))) +
            gamma*(net);
    return y;
  }
}

VectorXd Node::evaluate(const VectorXd &x, uint depth) {
  net = rho*x+rho0;
  if (isLeaf || gamma == 1)
    return y = net;
  else if (depth == 0)
    return y = net;
  else {
    alpha = sigmoid(dot(w,x)+w0);
    y = (1-gamma)*(alpha*(left->evaluate(x,depth-1)) +
                      (1-alpha)*(right->evaluate(x,depth-1))) +
            gamma*(net);
    return y;
  }
}

void Node::computeGrad(const VectorXd& delta, const VectorXd &x) {
  drho += (delta*gamma)*x.transpose() + lambdaL2*rho;
  drho0 += delta*gamma + lambdaL2*rho0;
  
  if (isLeaf || gamma == 1) {
    dgamma += dot(delta,net) - lambda;
  } else {
    dgamma += dot(delta,(-alpha*(left->y)-(1-alpha)*(right->y) + net)) - lambda;
    for (uint i=0; i<w.size(); i++)
      dw(i) += dot(delta,(1-gamma)*alpha*(1-alpha)*x(i)*(left->y - right->y))
            + lambdaL2*w(i);
    dw0 += dot(delta,(1-gamma)*alpha*(1-alpha)*(left->y - right->y))
        + lambdaL2*w0;
  }
}

void Node::backprop(const VectorXd& delta, const VectorXd &x) {
  computeGrad(delta, x);

  if (!isLeaf) {
    left->backprop(delta*(1-gamma)*alpha, x);
    right->backprop(delta*(1-gamma)*(1-alpha), x);
  }
}

void Node::update() {
  adaw = (adaw.array().square()+dw.array().square()).cwiseSqrt();
  w -= eta*dw.cwiseQuotient(adaw);
  dw.setZero();
  adaw0 = sqrt(pow(adaw0,2)+pow(dw0,2));
  w0 -= eta*dw0 / adaw0;
  dw0 = 0;
  adarho = (adarho.array().square()+drho.array().square()).cwiseSqrt();
  rho -= eta*drho.cwiseQuotient(adarho);
  drho.setZero();
  adarho0 = (adarho0.array().square()+drho0.array().square()).cwiseSqrt();
  rho0 -= eta*drho0.cwiseQuotient(adarho0);
  drho0.setZero();

  bool wasLeaf = isLeaf;
  if (isLeaf && dgamma > 0)
    split();

  adagamma = sqrt(pow(adagamma,2)+pow(dgamma,2));
  gamma -= eta*dgamma / adagamma;
  dgamma = 0;

  if (gamma > 1)
    gamma = 1;
  else if (gamma < 0)
    gamma = 0;

  if (!wasLeaf) {
    left->update();
    right->update();
  }
}

uint Node::size() {
  if (isLeaf || gamma == 1)
    return 1;
  else
    return 1 + left->size() + right->size();
}

double Node::effSize() {
  if (isLeaf || gamma == 1)
    return 1;
  else
    return 1 + (1-gamma)*(left->effSize() + right->effSize());
}

void Node::printGates(VectorXd v) {
  if (!isLeaf) {
    left->printGates(v.cwiseProduct(w)*(1-gamma)+v*gamma);
    right->printGates(v.cwiseProduct(w)*(-1)*(1-gamma)+v*gamma);
  } else {
    for (uint i=0; i<v.size(); i++)
      cout << v[i] << " ";
    cout << endl;
  }
}

void Node::print(string indent, bool rightChild) {
  cout << indent;
  if (rightChild) {
    cout << "\\-";
    indent += " ";
  } else {
    cout << "|-";
    indent += "| ";
  }

  VectorXd ww(w.size()+1);
  ww << w,w0;
  //cout << "w: " << ww << " ";
  //cout << "rho: " << rho << " ";
  cout << "gamma: " << gamma;
  cout << endl;

  if (!isLeaf && gamma != 1) {
    left->print(indent, false);
    right->print(indent, true);
  }
}

// this is mostly for visualization purposes
void Node::printTable(uint depth, string label, ostream& out,
                       double existence,
                       double xcoord, double ycoord, double width,
                       double xparent, double yparent) {
  out << depth << " ";
  for (uint i=0; i<w.size(); i++)
    out << w(i) << " ";
  out << w0 << " ";
  for (uint i=0; i<rho.rows(); i++) {
    out << rho0(i) << " ";
    for (uint j=0; j<rho.cols(); j++)
      out << rho(i,j) << " ";
  }
  out << gamma << " " << label << " ";
  out << existence << " ";
  out << xcoord << " " << ycoord << " ";
  out << xparent << " " << yparent << " ";

  for (uint i=0; i<ylist.size(); i++)
    out << ylist[i](0) << " ";
  out << endl;

  if (!isLeaf && gamma != 1) {
    left->printTable(depth+1, label+"0", out, existence*(1-gamma),
                     xcoord-width,ycoord-10,width/2,
                     xcoord, ycoord);
    right->printTable(depth+1, label+"1", out, existence*(1-gamma),
                      xcoord+width,ycoord-10,width/2,
                      xcoord, ycoord);
  }
}

void Node::save(ofstream& out) {
  out << w.size() << " ";
  for (uint i=0; i<w.size(); i++)
    out << w(i) << " ";
  out << w0 << " ";
  out << rho.rows() << " " << rho.cols() << " ";
  for (uint i=0; i<rho.rows(); i++) {
    out << rho0(i) << " ";
    for (uint j=0; j<rho.cols(); j++)
      out << rho(i,j) << " ";
  }
  out << gamma << " ";
  out << (uint)(isLeaf || gamma == 1) << endl;

  if (!isLeaf && gamma != 1) {
    left->save(out);
    right->save(out);
  } 
}

void Node::load(ifstream& in) {
  uint wsize, rhorows, rhocols, leaf;
  double x;
  in >> wsize;
  w = dw = adaw = VectorXd(wsize);
  for (uint i=0; i<wsize; i++) {
    in >> x;
    w(i) = x;
    dw(i) = 0;
    adaw(i) = 0.01;
  }
  in >> w0;
  dw0 = 0;
  adaw0 = 0.01;
  in >> rhorows >> rhocols;
  rho = drho = adarho = MatrixXd(rhorows, rhocols);
  rho0 = drho0 = adarho0 = VectorXd(rhorows);
  for (uint i=0; i<rhorows; i++) {
    in >> x;
    rho0(i) = x;
    for (uint j=0; j<rhocols; j++) {
      in >> x;
      rho(i,j) = x;
      drho(i,j) = 0;
      adarho(i,j) = 0.01;
    }
  }
  in >> gamma;
  dgamma = 0;
  adagamma = 0.01;
  in >> leaf;
  isLeaf = bool(leaf);
  
  if (!isLeaf) {
    left = new Node(0,0,0,0); // save / load params as well ?!
    right = new Node(0,0,0,0); // necessary for continuing training!!
    left->load(in);
    right->load(in);
  }
}

