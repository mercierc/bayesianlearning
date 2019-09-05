/**
 * @file utils.cpp
 * @author _____
 *
 * Implementation of some usefull functions for proprocess the data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "utils.hpp"
#include <stdexcept>

using namespace arma;

void preprocess_data(const mat& data,
		     const rowvec& responses,
		     bool fit_intercept,
		     bool normalize,
		     mat& data_proc,
		     rowvec& responses_proc,
		     colvec& data_offset,
		     colvec& data_scale,
		     double& responses_offset) {

  // initialize the offsets to their neutral forms.
  data_offset = zeros<colvec>(data.n_rows);
  data_scale = ones<colvec>(data.n_rows);
  responses_offset = 0.0;

  if (fit_intercept)
    {
      data_offset = mean(data, 1);
      responses_offset = mean(responses);
    }
  if (normalize)
    {
      data_scale = stddev(data, 0, 1);
    }

  // Copy data and response before the processing.
  data_proc = data;
  responses_proc = responses;
  // Center.
  data_proc.each_col() -= data_offset;
  // Scale.
  data_proc.each_col() /= data_scale;

  responses_proc -= responses_offset;
}
    

void gramMatrix(const mat& X,
		const mat& Y,
		mat& gramMatrix,
		double (*kernelFunction)(colvec&, colvec&, double),
		double gamma){

  unsigned int n1 = X.n_cols;
  unsigned int n2 = Y.n_cols;
  unsigned int p = X.n_rows;

  // Check if the dimensions are consistent.
  if (p != Y.n_rows)
    {
      std::cout << "error gramm" << std::endl;
      throw std::invalid_argument("Number of features not consistent");
    }
  
  gramMatrix = zeros<mat>(n1, n2);
  colvec xi = zeros<colvec>(p);
  colvec yj = zeros<colvec>(p);
  for (size_t i=0; i < n1; i++)
    {
      xi = X.col(i);
      for (size_t j=0; j < n2; j++)
	{
	  yj = Y.col(j);
	  gramMatrix(i,j) = kernelFunction(xi, yj, gamma);
	}
    }
}


double rbf(colvec& x,
	   colvec& y,
	   double gamma){
  
  // Set gamma at 1.0 / n_features by default
  if (gamma == 0.0)
    gamma = 1.0 / x.size();

  return exp(-gamma * pow(norm(x-y, 2),2));
}


double linear(colvec& x,
	      colvec& y,
	      double gamma){
  return dot(x,y);
}


