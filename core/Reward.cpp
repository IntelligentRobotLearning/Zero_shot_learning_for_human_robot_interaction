#include "Reward.h"
#include "COPReward.h"
#include "COMReward.h"
#include "torqueReward.h"

namespace MASS
{
    Reward::Reward()
    {      
    }

    Reward::~Reward()
    {      
    }

    void Reward::ReadFromStream(std::iostream& inp)
    {
        inp >> mName;
    }

    std::map<std::string, RewardFactory::Creator> RewardFactory::mRewards;
    // int RewardFactory::n_Reward;

    Reward* RewardFactory::CreateReward(const std::string& name) {
        if(mRewards.empty()) {
            mRewards.emplace("COPReward", &COPReward::CreateReward);
            mRewards.emplace("COMReward", &COMReward::CreateReward);
            mRewards.emplace("torqueReward", &torqueReward::CreateReward);
        }

        auto it = mRewards.find(name);
        if(it == mRewards.end()) return nullptr;

        return it->second();

    }
}
