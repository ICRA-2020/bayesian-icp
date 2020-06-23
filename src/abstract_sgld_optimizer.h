#ifndef __ABSTRACT_SGLD_OPTIMIZER_H__
#define __ABSTRACT_SGLD_OPTIMIZER_H__

//#include "sgdicp.h"
#include <vector>
#include <random>
#include <types.h>

class AbstractSgldOptimizer
{
    public:
        AbstractSgldOptimizer(std::vector<double> initial_values);
        virtual ~AbstractSgldOptimizer();

        Array6_d update_parameters(
                std::vector<double> const& gradients
        );

        Array6_d get_parameters() const;
  double sample_distribution (float mean, float sigma);
    protected:
        virtual
        void do_perform_update(std::vector<double> const& gradients) = 0;

    protected:
        //! Parameters to optimise
        Array6_d             m_parameters;
    private:
   // std::unique_ptr<SGDICP> m_sgdicp;
    
    std::mt19937                      m_generator;
    std::default_random_engine       m_default_generator;
    
};



#endif /* __ABSTRACT_SGLD_OPTIMIZER_H__ */
