#ifndef __ADAM_H__
#define __ADAM_H__

/*
 Copyright (c) 2019
 
 Fahira Afzal Maken and Lionel Ott,
 The University of Sydney, Australia.
 
 All rights reserved.
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 
 1. Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in
 the documentation and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 */


#include "abstract_sgd_optimizer.h"


/**
 * \brief Implementation of the ADAM method.
 *
 * Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
 * arXiv preprint arXiv:1412.6980 (2014).
 */
class Adam : public AbstractSgdOptimizer
{
    public:
        /**
         * \brief Creates a new optimizer instance.
         *
         * \param initial_values initial values of the parameters
         * \param step_size base gradient step size, \alpha
         * \param decay_rate_a decay rate for first moment estimates, \beta_1
         * \param decay_rate_b decay rate for second moment estimates, \beta_2
         */
        Adam(
            std::vector<double>         initial_values,
            double                      step_size,
            double                      decay_rate_a,
            double                      decay_rate_b
        );


    protected:
        /**
         * \see AbstractSgdOptimizer::do_perform_update
         */
        void do_perform_update(std::vector<double> const& gradients) override;


    private:
        //! Base step size for updates
        double                          m_step_size;
        //! Decay rate of the first moment estimates
        double                          m_decay_rate_a;
        //! Decay rate of the second moment estimates
        double                          m_decay_rate_b;

        //! Iteration step counter
        int                             m_timestep;
        //! Accumulated first moment decay rate
        double                          m_decay_rate_a_t;
        //! Accumulated second moment decay rate
        double                          m_decay_rate_b_t;

        //! First moment values for each parameter
        std::vector<double>             m_first_moment;
        //! Second moment values for each parameter
        std::vector<double>             m_second_moment;
};


#endif /* __ADAM_H__ */
