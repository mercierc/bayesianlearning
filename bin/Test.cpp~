#include <iostream>
// Includes all relevant components of mlpack.

#include <math.h>
#include <ctime>

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>


#include "bayesridge.hpp"
#include "rvmr.hpp"
#include "utils.hpp"
// Convenience.
using namespace mlpack;
using namespace rvmr;
int main()
{

  std::cout <<"CXX Bayesian regressions.\n"<< std::endl;
  // First, load the data.
  arma::mat X, Xtest;
  arma::rowvec y, ytest;
  double rmsetrain, rmsetest;

  // Comment the dataset that you don't want to use.

  std::cout<< "Corn dataset.\n"
	   << "http://www.eigenvector.com/data/Corn/"
	   << std::endl;
  data::Load("./data/corn_m5_train.csv",X,false,true);  
  data::Load("./data/corn_m5_test.csv",Xtest,false,true);  
  data::Load("./data/corn_y_train.csv",y,false,true);
  data::Load("./data/corn_y_test.csv",ytest,false,true);  

  // std::cout<< "Synthetic dataset.\n" 
  // 	   << "Only the first ten features are non equal to 0."
  // 	   << std::endl;
  // data::Load("./data/synth_train.csv",X,false,true);  
  // data::Load("./data/synth_test.csv",Xtest,false,true);  
  // data::Load("./data/synth_y_train.csv",y,false,true);
  // data::Load("./data/synth_y_test.csv",ytest,false,true);  

  // arma::rowvec predtrain, predtest, stdtrain, stdtest;
  // double rmsetrain, rmsetest;
  
  std::cout << "Dimension of tyhe train set ->"
	    << X.n_rows
	    << "x"
	    << X.n_cols
	    << std::endl;

  
  // Bayesian Ridge regression.
  std::cout << "\nBayesian Ridge regression."<< std::endl;
  auto start = std::chrono::system_clock::now();
  BayesianRidge bayesRidge(true, false);
  bayesRidge.Train(X,y);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "time = " << diff.count() << " s\n";
  
  rmsetrain = bayesRidge.Rmse(X, y); 
  rmsetest = bayesRidge.Rmse(Xtest, ytest);

  std::cout << "rmsetrain = "<< rmsetrain << std::endl;
  std::cout << "rmsetest = "<< rmsetest << "\n"<<  std::endl;

  // RVM ARD regression mode.
  std::cout << "RVM, ARD regression mode "<< std::endl;
  start = std::chrono::system_clock::now();
  RVMR<kernel::LinearKernel> ARDRegression(true, false);
  ARDRegression.Train(X,y);
  end = std::chrono::system_clock::now();
  diff = end - start;
  std::cout << "getBeta = " << ARDRegression.getBeta() << std::endl;
  std::cout << "time = " << diff.count() << " s\n";

  rmsetrain = ARDRegression.Rmse(X,y);
  rmsetest = ARDRegression.Rmse(Xtest,ytest);
    
  std::cout << "rmsetrain = "<< rmsetrain << std::endl;
  std::cout << "rmsetest = "<< rmsetest << "\n"<< std::endl;

  

  // RVM Linear kernel.
  std::cout << "RVM Regression Linear kernel"<< std::endl;
  start = std::chrono::system_clock::now();
  kernel::LinearKernel kernelLin;
  RVMR<kernel::LinearKernel> RVMLinear(kernelLin, true, false);
  RVMLinear.Train(X,y);
  end = std::chrono::system_clock::now();
  diff = end - start;
  std::cout << "getBeta = " << RVMLinear.getBeta() << std::endl;
  std::cout << "time = " << diff.count() << " s\n";

  rmsetrain = RVMLinear.Rmse(X,y);
  rmsetest = RVMLinear.Rmse(Xtest,ytest);
    
  std::cout << "rmsetrain = "<< rmsetrain << std::endl;
  std::cout << "rmsetest = "<< rmsetest << "\n"<<  std::endl;



  // RVM Gaussian kernel.
  std::cout << "RVM Regression Gaussian kernel"<< std::endl;
  start = std::chrono::system_clock::now();
  kernel::GaussianKernel kernelGaussian;
  RVMR<kernel::GaussianKernel> RVMrbf(kernelGaussian, true, false);
  RVMrbf.Train(X,y);
  end = std::chrono::system_clock::now();
  diff = end - start;
  std::cout << "time = " << diff.count() << " s\n";

  rmsetrain = RVMrbf.Rmse(X,y);
  rmsetest = RVMrbf.Rmse(Xtest,ytest);

  std::cout << "rmsetrain = "<< rmsetrain << std::endl;
  std::cout << "rmsetest = "<< rmsetest << "\n"<< std::endl;

  std::cout << "end " << std::endl;
}
