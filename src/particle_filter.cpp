/**
 * particle_filter.cpp
 *
 * Created on: July
 * Author: Rajath Rao
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // Set the number of particles
  particles.resize(num_particles); // adjust to the set number of particles
  weights.resize(num_particles); // should equal number of particles

  double std_x,std_y,std_theta;
  // extracting stds for easy readability
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // creating a Gaussian distribution for the three measurements
  std::default_random_engine gen;
  std::normal_distribution<double>dist_x(x,std_x);
  std::normal_distribution<double>dist_y(y,std_y);
  std::normal_distribution<double>angle_theta(theta,std_theta);

  // initializing values for all particles
  for (int i = 0; i<num_particles; ++i){
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = angle_theta(gen);
    particles[i].weight = 1.0;
    
    weights[i] = 1.0; // initializing all weights to 1.0
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  double x0,y0,theta0;
  double pred_x,pred_y,pred_theta;

  for (int i = 0; i < num_particles; ++i){
    x0 = particles[i].x;
    y0 = particles[i].y;
    theta0 = particles[i].theta;

    if (fabs(yaw_rate) > 0.001){
      pred_x = x0 + (velocity/yaw_rate)*(sin(theta0 + yaw_rate*delta_t) - sin(theta0));
      pred_y = y0 + (velocity/yaw_rate)*(cos(theta0) - cos(theta0 + yaw_rate*delta_t));
      pred_theta = theta0 + yaw_rate*delta_t;
    }
    else
    {
      pred_x = x0 + velocity * cos(theta0) * delta_t;
      pred_y = y0 + velocity * sin(theta0) * delta_t;
      pred_theta = theta0 + yaw_rate*delta_t;
    }

    std::normal_distribution<double>dist_x(pred_x,std_pos[0]);
    std::normal_distribution<double>dist_y(pred_y,std_pos[1]);
    std::normal_distribution<double>angle_theta(pred_theta,std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = angle_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  // using nearest neighbor algorithm for associating observations to landmarks
  for (int i = 0; i < observations.size(); ++i){
    double lowest_dist = 1000; // high initial guess
    int nearest_landmark_id = -1; // initial guess should not be part of list
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;

    for (int j = 0; j < predicted.size(); ++j){
      int pred_id = predicted[j].id;
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;
      double new_dist = dist(obs_x,obs_y,pred_x,pred_y);
    
      if(new_dist < lowest_dist){
        lowest_dist = new_dist;
        nearest_landmark_id = pred_id;
      }
    }
    observations[i].id = nearest_landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */


  // variable for normalizing all weights at the end
  double sum_weights = 0.0;

  for (int i = 0; i < num_particles; ++i){
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // get landmark locations within sensor range for each particle
    // create a vector to hold these
    vector<LandmarkObs> preds_obs;

    for (int j = 0; j< map_landmarks.landmark_list.size(); ++j){
      double ex_x = map_landmarks.landmark_list[j].x_f;
      double ex_y = map_landmarks.landmark_list[j].y_f;
      int ex_id = map_landmarks.landmark_list[j].id_i;

      // taking into consideration only those landmarks that fall within the sensor range
      if ((fabs(ex_x - p_x) <= sensor_range) && (fabs(ex_y - p_y) <= sensor_range)){
        preds_obs.push_back(LandmarkObs{ex_id,ex_x,ex_y});
      }
    }
    // create vector to hold the transformed landmark observations from vehicle to 
    // map coordinates
    vector<LandmarkObs> tranformed_obs;
    for (int k = 0; k < observations.size(); ++k) {
      double t_x = cos(p_theta)*observations[k].x - sin(p_theta)*observations[k].y + p_x;
      double t_y = sin(p_theta)*observations[k].x + cos(p_theta)*observations[k].y + p_y;
      tranformed_obs.push_back(LandmarkObs {observations[k].id,t_x,t_y});
    }

    // finding nearest landmark for each particle using data association
    dataAssociation(preds_obs,tranformed_obs);

    // reinitializing all particle weights to 1.0
    particles[i].weight = 1.0;

    for (int j = 0; j < tranformed_obs.size(); ++j){
      // placeholders for the coordinates
      double os_x = tranformed_obs[j].x;
      double os_y = tranformed_obs[j].y;
      double pr_x, pr_y;

      int associated_landmark = tranformed_obs[j].id;

      for (int k = 0; k < preds_obs.size(); ++k){
        if (preds_obs[k].id == associated_landmark){
          pr_x = preds_obs[k].x;
          pr_y = preds_obs[k].y;
        }
      }
    
    // for nearest landmark, calculating weight by comparing predictions and observations using
    // multivariate Gaussian distribution
    
    // stddevs in x and y directions
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    // calculating updated weight
    double new_w = 1/(2*M_PI*std_x*std_y) * exp(- (pow(pr_x-os_x,2)/(2*pow(std_x,2))) + (pow(pr_y-os_y,2)/(2*pow(std_y,2))));
    
    // updating the particle weight with new weight
    particles[i].weight *= new_w;
    }
    // summing all weights
    sum_weights += particles[i].weight;
  }
  // normalizing all weights
  for (int i = 0; i < particles.size(); ++i){
    particles[i].weight /= sum_weights;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	
  // vector holding resampled new particles
  vector<Particle> new_particles;

  std::default_random_engine gen;
  
  // generate random starting index for resampling wheel
  std::uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // get max weight fromt the list
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // resampling step to obtain new particles
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}