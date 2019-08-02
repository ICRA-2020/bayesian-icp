#ifndef __ABSTRACT_SGD_OPTIMIZER_H__
#define __ABSTRACT_SGD_OPTIMIZER_H__

//#include "sgdicp.h"
#include <vector>
#include <random>

class AbstractSgdOptimizer
{
    public:
        AbstractSgdOptimizer(std::vector<double> initial_values);
        virtual ~AbstractSgdOptimizer();

        std::vector<double> update_parameters(
                std::vector<double> const& gradients
        );

        std::vector<double> get_parameters() const;
  double sample_distribution (float mean, float sigma);
    protected:
        virtual
        void do_perform_update(std::vector<double> const& gradients) = 0;

    protected:
        //! Parameters to optimise
        std::vector<double>             m_parameters;
    private:
   // std::unique_ptr<SGDICP> m_sgdicp;
    
    std::mt19937                      m_generator;
    std::default_random_engine       m_default_generator;
    
};



#endif /* __ABSTRACT_SGD_OPTIMIZER_H__ */
