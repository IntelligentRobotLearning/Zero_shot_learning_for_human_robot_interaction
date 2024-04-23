#ifndef __MASS_TORQUEREWARD_H__
#define __MASS_TORQUEREWARD_H__
#include <Eigen/Dense>
#include "Reward.h"

namespace MASS
{

class torqueReward : public Reward
{
public:
	torqueReward();
    ~torqueReward();

    virtual double GetReward();
    virtual void ReadFromStream(std::iostream& inp);
    
    static Reward* CreateReward();
    double GetWeight(){return weight;}

private:
    std::vector<std::string> contact_segs;
    double weight;

};


}

#endif
