#ifndef __MASS_COMREWARD_H__
#define __MASS_COMREWARD_H__
#include <Eigen/Dense>
#include "Reward.h"

namespace MASS
{

class COMReward : public Reward
{
public:
	COMReward();
    ~COMReward();

    virtual double GetReward();
    virtual void ReadFromStream(std::iostream& inp);
    
    static Reward* CreateReward();
    double GetWeight(){return weight;}

private:
    std::vector<std::string> contact_segs;

    Eigen::Vector3d unitV;
    double COP_Y_fixed; 
    Eigen::Vector3d COP_left;
    Eigen::Vector3d COP_right;
    Eigen::Vector3d COP; 
    Eigen::Vector3d COP_target; 
    double weight;
};


}

#endif
