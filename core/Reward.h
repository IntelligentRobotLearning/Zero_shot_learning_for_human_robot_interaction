#ifndef __MASS_REWARD_H__
#define __MASS_REWARD_H__

#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <map>
//#include <memory> //shared_ptr



namespace MASS
{

class Environment;

class Reward
{
public:
	Reward();
    virtual ~Reward();

    const std::string& GetName() const {return mName;}
    void SetName(const std::string& name) {mName = name;}

    virtual double GetReward() {return 0.0;};
    virtual double GetWeight() {return 0.0;};
    virtual void ReadFromStream(std::iostream& inp);

    void SetEnvironment(Environment& env) {mEnv = &env;}

public:
	std::string mName;

    Environment* mEnv;

};

class RewardFactory
{
public:
	static Reward* CreateReward(const std::string &reward);
    // static int n_Reward; 
// private:
	RewardFactory();
    typedef Reward* (*Creator)();
	static std::map<std::string, Creator> mRewards;
};


}

#endif
