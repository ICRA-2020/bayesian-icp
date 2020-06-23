
#ifndef __SGLD_Preconditioned_2_H__
#define __SGLD_Preconditioned_2_H__


#include "abstract_sgld_optimizer.h"


class SGLD_Preconditioned2 : public AbstractSgldOptimizer
{
public:
    SGLD_Preconditioned2(
                        std::vector<double>         initial_values,
                        double                      step_size,
                        double                      decay_rate_a,
                        double                      max_range,
                        int                         open_mp_id,               
                        float			    adjust_noise
            );
    
protected:
    void do_perform_update(std::vector<double> const& gradients) override;
    
private:
    double                          m_step_size;
    double                          m_decay_rate_a;
    double                          m_max_range;
    int                             m_open_mp_id;
    float			    m_adjust_noise;
    

    
    std::vector<double>             m_second_moment;
    int                             m_iteration_count=0;


private:
  double normalizeAngle(double radians);

};


#endif /* __SGLD_Preconditioned_H__ */
