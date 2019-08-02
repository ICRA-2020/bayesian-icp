
#ifndef __SGLD_Preconditioned_H__
#define __SGLD_Preconditioned_H__


#include "abstract_sgd_optimizer.h"


class SGLD_Preconditioned : public AbstractSgdOptimizer
{
public:
    SGLD_Preconditioned(
                        std::vector<double>         initial_values,
                        double                      step_size,
                        double                      decay_rate_a,
                        double                      max_range,
                        int                         open_mp_id,
                        double                      cloud_size,
                        std::vector<double>         prior_mean,
                        float                       prior_variance
            );
    
protected:
    void do_perform_update(std::vector<double> const& gradients) override;
    
private:
    double                          m_step_size;
    double                          m_decay_rate_a;
    double                          m_max_range;
    int                             m_open_mp_id;
    double                          m_cloud_size;
    std::vector<double>             m_prior_mean;
    float                           m_prior_variance;
    

    
    std::vector<double>             m_second_moment;
    int                             m_iteration_count=0;


private:
  double normalizeAngle(double radians);

};


#endif /* __SGLD_Preconditioned_H__ */
