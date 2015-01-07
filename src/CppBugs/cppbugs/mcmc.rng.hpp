///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2011 Whit Armstrong                                     //
//                                                                       //
// This program is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by  //
// the Free Software Foundation, either version 3 of the License, or     //
// (at your option) any later version.                                   //
//                                                                       //
// This program is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
// GNU General Public License for more details.                          //
//                                                                       //
// You should have received a copy of the GNU General Public License     //
// along with this program.  If not, see <http://www.gnu.org/licenses/>. //
///////////////////////////////////////////////////////////////////////////

#ifndef MCMC_RNG_HPP
#define MCMC_RNG_HPP

#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cppbugs/mcmc.rng.base.hpp>

namespace cppbugs {

  template<typename T>
  class SpecializedRng : public RngBase {
    T generator_;
    boost::normal_distribution<double> normal_rng_dist_;
    boost::uniform_real<double> uniform_rng_dist_;
    boost::variate_generator<T&, boost::normal_distribution<double> > normal_rng_;
    boost::variate_generator<T&, boost::uniform_real<double> > uniform_rng_;
    double next_norm_;
  public:
    SpecializedRng(): RngBase(),
                      normal_rng_dist_(0, 1), uniform_rng_dist_(0, 1),
                      normal_rng_(generator_, normal_rng_dist_),
                      uniform_rng_(generator_, uniform_rng_dist_) {
      next_norm_ = NAN;
    }

    double normal() {
      if(next_norm_ != next_norm_) {
        double x, y, s;
        do {
          x = uniform_rng_()-0.5;
          y = uniform_rng_()-0.5;
          s = x*x+y*y;
        } while(s > 0.25 || s == 0.);
        double coef = sqrt(-2*log_approx(s*4)/s);
        next_norm_ = coef*y;
        return coef*x;
      } else {
        double r = next_norm_;
        next_norm_ = NAN;
        return r;
      }
    }
    double uniform() { return uniform_rng_(); }
  };

} // namespace cppbugs
#endif // MCMC_RNG_HPP
