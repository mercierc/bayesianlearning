/**
 * @file bayesridge.cpp
 * @author Clement Mercier Reicrem
 *
 * Implementation of Bayesian Ridge regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "bayesridge.hpp"
#include "utils.hpp"

using namespace mlpack;


BayesianRidge::BayesianRidge(const bool fitIntercept,
			     const bool normalize) :
  fitIntercept(fitIntercept),
  normalize(normalize){

  std::cout << "Baysian Ridge regression(fitIntercept="
  	    << this->fitIntercept
  	    <<", normalize="
  	    <<this->normalize
  	    <<")"
  	    <<std::endl;
  }

void BayesianRidge::Train(const arma::mat& data,
			  const arma::rowvec& responses){

  arma::mat phi;
  arma::rowvec t;
  arma::colvec vecphitT;
  arma::mat phiphiT;
  arma::colvec eigval;
  arma::mat eigvec;
  arma::colvec eigvali;
    
  // Preprocess the data. Center and normalize.
  preprocess_data(data,
		  responses,
		  this->fitIntercept,
		  this->normalize,
		  phi,
		  t,
		  this->data_offset,
		  this->data_scale,
		  this->responses_offset);

  // Transpose to end up with the machine learning standard. 
  vecphitT = phi * t.t();
  phiphiT =  phi * phi.t();

  // Compute the eigenvalues only once.
  arma::eig_sym(eigval, eigvec, phiphiT);
   
  unsigned short p = data.n_rows, n = data.n_cols;
  std::cout << "pxn = " << p <<"x"<< n << std::endl;
  // Initialize the hyperparameters and
  // begin with an infinitely broad prior.
  this->alpha = 1e-6;
  this->beta =  1 / (var(t) * 0.1);

  double tol = 1e-3; 
  unsigned short nIterMax = 50;
  unsigned short i = 0;
  double deltaAlpha = 1, deltaBeta = 1, crit = 1;
  arma::mat matA = arma::eye<arma::mat>(p, p);
  arma::rowvec temp;

  while ((crit > tol) && (i < nIterMax))
    {
      deltaAlpha = -this->alpha;
      deltaBeta = -this->beta;

      // Compute the posterior statistics.
      for(size_t k=0; k<p; k++) {matA(k,k) = this->alpha;}
      // inv is used instead of solve beacause we need matCovariance to
      // compute the prediction uncertainties. If solve is used, matCovariance
      // must be comptuted at the end of the loop.
      this->matCovariance = inv(matA + phiphiT * this->beta);
      this->omega = (this->matCovariance * vecphitT) * this->beta;

      // Update alpha.
      eigvali = eigval * this->beta;
      gamma = sum(eigvali / (this->alpha + eigvali));
      std::cout << "gamma  = "<<gamma << std::endl;
      this->alpha = gamma / dot(this->omega.t(), this->omega);

      // Update beta.
      temp = t - this->omega.t() * phi;
      this->beta = (n - gamma) / dot(temp, temp);

      // Comptute the stopping criterion.
      deltaAlpha += this->alpha;
      deltaBeta += this->beta;
      crit = abs(deltaAlpha/this->alpha + deltaBeta/this->beta);
      i++;
      std::cout << i << std::endl;
    }
}

void BayesianRidge::Predict(const arma::mat& points,
			    arma::rowvec& predictions) const
{
  arma::mat X;
  X = points;

  //Center and normalize the points before applying the model
  X.each_col() -= this->data_offset;
  X.each_col() /= this->data_scale;
  predictions = this->omega.t() * X + this->responses_offset; 
}


void BayesianRidge::Predict(const arma::mat& points,
			    arma::rowvec& predictions,
			    arma::rowvec& std) const
{
  arma::mat X;
  X = points;

  //Center and normalize the points before applying the model
  X.each_col() -= this->data_offset;
  X.each_col() /= this->data_scale;
  predictions = this->omega.t() * X + this->responses_offset;
  
  //Compute the standard deviation of each prediction
  std = arma::zeros<arma::rowvec>(X.n_cols);
  double var = 1.0 / this->beta;
  arma::colvec phi(X.n_rows);
  for (size_t i=0; i < X.n_cols; i++)
    { phi = X.col(i);
      std[i] = sqrt(var + dot(phi.t() * this->matCovariance,  phi));
    }
}

float BayesianRidge::Rmse(const arma::mat& data,
			  const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  this->Predict(data, predictions);
  return sqrt(mean(square(responses - predictions)));
}


