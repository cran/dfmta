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

#ifndef MCMC_MODEL_HPP
#define MCMC_MODEL_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <exception>
#include <cppbugs/mcmc.rng.hpp>
#include <cppbugs/mcmc.object.hpp>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {
  typedef std::map<void*,MCMCObject*> vmc_map;

  template<class RNG>
  class MCModel {
  private:
    double accepted_,rejected_,logp_value_,old_logp_value_;
    SpecializedRng<RNG> rng_;
    std::vector<MCMCObject*> mcmcObjects, jumping_nodes, dynamic_nodes;
    std::vector<Likelihiood*> logp_functors;
    std::function<void ()> update;
    vmc_map data_node_map;

    void jump() { for(auto v : jumping_nodes) { v->jump(rng_); } }
    void preserve() { for(auto v : dynamic_nodes) { v->preserve(); } }
    void revert() { for(auto v : dynamic_nodes) { v->revert(); } }
    void set_scale(const double scale) { for(auto v : jumping_nodes) { v->setScale(scale); } }
    void tally() { for(auto v : dynamic_nodes) { v->tally(); } }
    static bool bad_logp(const double value) { return std::isnan(value) || value == -std::numeric_limits<double>::infinity() ? true : false; }
  public:
    MCModel(std::function<void ()> update_, long seed = 42):
      accepted_(0), rejected_(0),
      logp_value_(-std::numeric_limits<double>::infinity()),
      old_logp_value_(-std::numeric_limits<double>::infinity()),
      rng_(seed), update(update_) {}
    ~MCModel() {
      // use data_node_map as delete list
      // only objects allocated by this class are inserted thre
      // addNode allows user allocated objects to enter the mcmcObjects vector
      for(auto m : data_node_map) {
        delete m.second;
      }
    }

    void addStochcasticNode(MCMCObject* node) {
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      // FIXME: this should throw if sp->getLikelihoodFunctor() returns null
      if(sp && sp->getLikelihoodFunctor() ) { logp_functors.push_back(sp->getLikelihoodFunctor()); }
    }

    void initChain() {
      logp_functors.clear();
      jumping_nodes.clear();

      for(auto node : mcmcObjects) {
        addStochcasticNode(node);

        if(node->isStochastic() && !node->isObserved()) {
          jumping_nodes.push_back(node);
        }

        if(!node->isObserved()) {
          dynamic_nodes.push_back(node);
        }
      }
    }

    double acceptance_ratio() const {
      return accepted_ / (accepted_ + rejected_);
    }

    bool reject(const double value, const double old_logp) {
      return bad_logp(value) || log(rng_.uniform()) > (value - old_logp) ? true : false;
    }

    double logp() const {
      double ans(0);
      update();
      for(auto f : logp_functors) {
        ans += f->calc();
      }
      return ans;
    }

    void resetAcceptanceRatio() {
      accepted_ = 0;
      rejected_ = 0;
    }

    void tune(int iterations, int tuning_step) {
      double logp_value,old_logp_value;
      logp_value  = -std::numeric_limits<double>::infinity();
      old_logp_value = -std::numeric_limits<double>::infinity();

      for(int i = 1; i <= iterations; i++) {
	for(auto it : jumping_nodes) {
          old_logp_value = logp_value;
          it->preserve();
          it->jump(rng_);
          logp_value = logp();
          if(reject(logp_value, old_logp_value)) {
            it->revert();
            logp_value = old_logp_value;
            it->reject();
          } else {
            it->accept();
          }
	}
	if(i % tuning_step == 0) {
          //std::cout << "tuning at step: " << i << std::endl;
	  for(auto it : jumping_nodes) {
	    it->tune();
	  }
	}
      }
    }

    void step() {
      old_logp_value_ = logp_value_;
      preserve();
      jump();
      logp_value_ = logp();
      if(reject(logp_value_, old_logp_value_)) {
        revert();
        logp_value_ = old_logp_value_;
        rejected_ += 1;
      } else {
        accepted_ += 1;
      }
    }

    void tune_global(int iterations, int tuning_step) {
      logp_value_ = logp();

      const double thresh = 0.1;
      // FIXME: this should possibly related to the overall size/dimension
      // of the parmaeters to be estimtated, as there is somewhat of a leverage effect
      // via the number of parameters
      const double dilution = 0.10;
      double total_size = 0;

      for(size_t i = 0; i < dynamic_nodes.size(); i++) {
        if(dynamic_nodes[i]->isStochastic()) {
          total_size += dynamic_nodes[i]->size();
        }
      }
      double target_ar = std::max(1/log2(total_size + 3), 0.234);
      for(int i = 1; i <= iterations; i++) {
        step();
        if(i % tuning_step == 0) {
          double diff = acceptance_ratio() - target_ar;
          resetAcceptanceRatio();
          if(std::abs(diff) > thresh) {
            double adj_factor = (1.0 + diff * dilution);
            for(size_t i = 0; i < dynamic_nodes.size(); i++) {
              dynamic_nodes[i]->setScale(dynamic_nodes[i]->getScale() * adj_factor);
            }
          }
        }
      }
    }

    void run(int iterations, int burn, int thin) {
      logp_value_ = logp();
      if(logp_value_==-std::numeric_limits<double>::infinity()) {
        throw std::logic_error("ERROR: cannot start from a logp of -Inf.");
      }

      for(int i = 1; i <= (iterations + burn); i++) {
        step();
        if(i > burn && (i % thin == 0)) {
          tally();
        }
      }
    }

    void sample(int iterations, int burn, int adapt, int thin) {
      if(iterations % thin) {
        throw std::logic_error("ERROR: iterations not a multiple of thin.");
      }

      // setup logp's etc.
      initChain();

      if(logp()==-std::numeric_limits<double>::infinity()) {
        throw std::logic_error("ERROR: cannot start from a logp of -Inf.");
      }

      // tuning phase
      tune(adapt,static_cast<int>(adapt/100));
      if(true) { tune_global(adapt,static_cast<int>(adapt/100)); }

      // sampling
      run(iterations, burn, thin);
    }

    template<template<typename> class MCTYPE, typename T>
    MCTYPE<T>& track(T&& x) {
      MCTYPE<T> *node = new MCTYPE<T>(std::forward<T>(x));
      mcmcObjects.push_back(node);
      data_node_map[std::is_lvalue_reference<T>::value ? (void*)(&x) : (void*)this] = node;
      return *node;
    }

    // allows node to be added without being put on the delete list
    // for those who want full control of their memory...
    void track(MCMCObject* node) {
      mcmcObjects.push_back(node);
    }

    template<typename T>
    Dynamic<T&>& getNode(const T& x) {
      auto iter = data_node_map.find((void*)(&x));
      if(iter == data_node_map.end()) {
        throw std::logic_error("node not found.");
      }
      Dynamic<T&>* ans = dynamic_cast<Dynamic<T&>*>(iter->second);
      if(ans == nullptr) {
        throw std::logic_error("invalid node conversion.");
      }
      return *ans;
    }
  };
} // namespace cppbugs
#endif // MCMC_MODEL_HPP
