#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "NumPyHelper.h"
class EnvManager
{
public:
	EnvManager(std::string meta_file,int num_envs);

	int GetNumState();
	int GetNumAction();
	int GetNumFullObservation();
	int GetNumFutureTargetmotions();
	int GetNumStateHistory();
	int GetNumRootInfo();

	int GetSimulationHz();
	int GetControlHz();
	int GetNumSteps();
	bool UseMuscle();
	bool UseSymmetry();

	void Step(int id);
	void Reset(bool RSI,int id);
	bool IsEndOfEpisode(int id);
	np::ndarray GetState(int id);
	void SetAction(np::ndarray np_array, int id);
	
	double GetReward(int id);
	np::ndarray GetAction(int id); 
	np::ndarray GetFullObservation(int id);

	void Steps(int num);
	void StepsAtOnce();
	void Resets(bool RSI);
	np::ndarray IsEndOfEpisodes();
	np::ndarray GetStates();
	np::ndarray GetFullObservations();
	np::ndarray GetTargetObservations();
	np::ndarray GetActions();

	void SetActions(np::ndarray np_array);
	np::ndarray GetRewards();
	void UpdateStateBuffers();
	void UpdateActionBuffers(np::ndarray np_array);

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs(){return mEnvs[0]->GetNumTotalRelatedDofs();};
	int GetNumMuscles(){return mEnvs[0]->GetCharacter()->GetMuscles().size();}
	np::ndarray GetMuscleTorques();
	np::ndarray GetDesiredTorques();
	void SetActivationLevels(np::ndarray np_array);
	
	p::list GetMuscleTuples();
private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
};

#endif