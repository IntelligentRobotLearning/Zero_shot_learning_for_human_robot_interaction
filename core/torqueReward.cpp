#include "torqueReward.h"
#include <iostream>
#include "Environment.h"
#include <Eigen/Dense>

namespace MASS
{
    torqueReward::torqueReward()
     :Reward()
    {
    }

    torqueReward::~torqueReward()
    {
    }

    Reward* torqueReward::CreateReward()
    {
        return new torqueReward;
    }

    void torqueReward::ReadFromStream(std::iostream& strm)
    {
        Reward::ReadFromStream(strm);
        strm >> weight;
    }


    double torqueReward::GetReward()
    {
        double r = 0;
        Eigen::VectorXd torque = mEnv->GetDesiredTorques();
        r = exp(-0.001*torque.squaredNorm()); ;  // train
        return r;
    }

}
