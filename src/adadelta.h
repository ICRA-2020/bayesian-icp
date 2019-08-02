#ifndef __ADADELTA_H__
#define __ADADELTA_H__


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
 * \brief Implementation of the AdaDelta method.
 *
 * Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method."
 * arXiv preprint arXiv:1212.5701 (2012).
 */
class AdaDelta : public AbstractSgdOptimizer
{
    public:
        /**
         * \brief Creates a new optimizer instance.
         *
         * \param initial_values initial values of the parameters
         * \param decay_rate gradient accumulation decay rate, \rho
         * \param preconditioner conditioner for RMS computation, \epsilon
         */
        AdaDelta(
                std::vector<double>     initial_values,
                double                  decay_rate,
                double                  preconditioner
        );


    protected:
        /**
         * \see AbstractSgdOptimizer::do_perform_update
         */
        void do_perform_update(std::vector<double> const& gradients) override;


    private:
        //! Exponential decay rate used for average accumulation
        double                          m_decay_rate;
        //! Preconditioner to ensure numerical stability of RMSE computation
        double                          m_preconditioner;

        //! Accumulated gradient information
        std::vector<double>             m_acc_gradient;
        //! Accumulated update information
        std::vector<double>             m_acc_updates;
};


#endif /* __ADADELTA_H__ */
