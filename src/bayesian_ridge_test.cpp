#include <iostream>
// Includes all relevant components of mlpack.

#include <math.h>
#include <ctime>

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>


#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Mytest
#include <boost/test/unit_test.hpp>

#include "bayesridge.hpp"
#include "rvmr.hpp"
#include "utils.hpp"
// Convenience.
using namespace mlpack;
using namespace rvmr;


BOOST_AUTO_TEST_CASE(BayesianRidgeRegressionTest)
{
  // First, load the data.
  arma::mat Xtrain, Xtest;
  arma::rowvec ytrain, ytest;
  double RMSETRAIN = 4.64366e-06, RMSETEST = 0.00991352;
  
  std::cout<< "Corn dataset.\n"
	   << "http://www.eigenvector.com/data/Corn/"
	   << std::endl;
  data::Load("./data/corn_m5_train.csv",Xtrain,false,true);  
  data::Load("./data/corn_m5_test.csv",Xtest,false,true);  
  data::Load("./data/corn_y_train.csv",ytrain,false,true);
  data::Load("./data/corn_y_test.csv",ytest,false,true);

  // Instanciate and train the estimator
  BayesianRidge estimator(true, false);
  estimator.Train(Xtrain, ytrain);

  // Check if the RMSE are still equal to the previously fixed values
  BOOST_REQUIRE_SMALL(estimator.Rmse(Xtrain,ytrain) - RMSETRAIN, 0.05);
  BOOST_REQUIRE_SMALL(estimator.Rmse(Xtest,ytest) - RMSETEST, 0.05);

}


