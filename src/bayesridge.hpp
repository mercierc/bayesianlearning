/**
 * @file bayesridge.hpp
 * @ Clement Mercier
 *
 * Definition of the BayesianRidge class, which performs the 
 * bayesian linear regression
**/
#ifndef TATON_BAYESRIDGE_HPP 
#define  TATON_BAYESRIDGE_HPP

#include <mlpack/prereqs.hpp>

class BayesianRidge
{
public:
  /**
   * Set the parameters of Bayesian Ridge regression object. The
   *    regulariation parameter is automaticaly set to its optimal value by 
   *    maximmization of the marginal likelihood.
   *
   * @param fitIntercept Whether or not center the data according to the *
   *      examples.
   * @param normalize Whether or to normalize the data according to the 
   * standard deviation of each feature.
   **/
  BayesianRidge(const bool fitIntercept = true,
		const bool normalize = false);

  /** 
   * Run BayesianRidge regression. The input matrix (like all mlpack matrices) 
   * should be
   * column-major -- each column is an observation and each row is a dimension.
   * 
   * @param data Column-major input data 
   * @param responses A vector of targets.
   **/
  void Train(const arma::mat& data,
	     const arma::rowvec& responses);

  /**
   * Predict \f$y_{i}\f$ for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge model.
   *
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   **/
  void Predict(const arma::mat& points,
               arma::rowvec& predictions) const;

  /**
   * Predict \f$y_{i}\f$ and the standard deviation of the predictive posterior 
   * distribution for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge estimator.
   *
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   * @param std Standard deviations of the predictions.
   * @param rowMajor Should be true if the data points matrix is row-major and
   *     false otherwise.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions,
	       arma::rowvec& std) const;

  /**
   * Compute the Root Mean Square Error
   * between the predictions returned by the model
   * and the true repsonses
   * @param Points Data points to predict
   * @param responses A vector of targets.
   * @return RMSE
   **/
  double Rmse(const arma::mat& data,
	     const arma::rowvec& responses) const;

  /**
   * Get the solution vector
   * @return omega Solution vector.
   **/
  inline arma::colvec getCoefs() const{return this->omega;}


  /**
   * Get the precesion (or inverse variance) beta of the model.
   * @return \f$ \beta \f$ 
   **/
  inline double getBeta() const {return this->beta;} 
  
  /**
   * Get the estimated variance.
   * @return 1.0 / \f$ \beta \f$
   **/
  inline double getVariance() const {return 1.0 / this->getBeta();}


private:
  //! Center the data if true
  bool fitIntercept;
  //! Scale the data by standard deviations if true
  bool normalize;
  //! Mean vector computed over the points
  arma::colvec data_offset;
  //! Std vector computed over the points
  arma::colvec data_scale;
  //! Mean of the response vector computed over the points
  double responses_offset;
  //! Precision of the prio pdf (gaussian)
  double alpha;
  //! Noise inverse variance
  double beta;
  //! Effective number of parameters
  double gamma;
  //! Solution vector
  arma::colvec omega;
  //! Coavriance matrix of the solution vector omega
  arma::mat matCovariance;
};


#endif

  
