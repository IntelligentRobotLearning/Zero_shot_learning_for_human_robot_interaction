#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
#include "Reward.h"
#include "COPReward.h"
#include "COMReward.h"
#include "torqueReward.h"
#include "Fixedeque.h"

#include <queue>
#include <deque>
#include <iostream>

#define HISTORY_BUFFER_LEN 3
#define STATE_HISTORY_BUFFER_LEN 4

namespace MASS
{

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::MatrixXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};

struct HuntCrosslyContact
{
	HuntCrosslyContact();

	double stiffness;
	double dissipation;
	double staticFriction;
	double dynamicFriction;
	double viscousFriction;
	double radius;
	double transitionVelocity; //transition velocity from static friction to dynamic friction
	//double order;
};

class Environment
{
public:
	Environment();

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetUseMuscleNN(bool use_muscleNetWork){mUseMuscleNN = use_muscleNetWork;}
	void SetUseHuntContactForce(bool use_Huntcontact){mUseHuntContactForce = use_Huntcontact;}
	void SetSymmetry(bool symmetry){mSymmetry = symmetry;}
	void SetUseCOP(bool use_COP){mUseCOP = use_COP;}
    void SetCOMindependent(bool COM_independent){mCOMindependent = COM_independent;}
	void Settargetmotion_visual(bool target_motion_visualization){mUsetarget_visual = target_motion_visualization;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}
	void SetTerminalTime (double terminal_time) {mterminal_time = terminal_time;}
	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}
	void SetWalkingSkill(bool walking) {walk_skill = walking;}
	void SetSquattingSkill(bool squatting) {squat_skill = squatting;}
	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com,double w_torque,double w_root){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;this->w_torque = w_torque;this->w_root=w_root;}
	void SetPDParameters(double kp) {this->kp=kp;}
	void SetSmoothRewardParameters(double w_sroot, double w_saction, double w_storque,double w_sjoint_vel){this->w_sroot = w_sroot;this->w_saction = w_saction;this->w_storque = w_storque;this->w_sjoint_vel = w_sjoint_vel;}
	void SetFootClearanceRewardParameter(double w_footclr) {this->W_footclr = W_footclr;}
	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);
    void SetJointRewardParameters(double w_hip,double w_knee,double w_ankle){this->w_hip = w_hip;this->w_knee = w_knee;this->w_ankle = w_ankle;}
	void SetFootClearance(double a, double b) {high_clearance = a; low_clearance =b; }
	void SetFootTolerances(double a) {foot_tolerance = a;}
public:
	void ProcessAction(int j, int num); 
	void Step();
	void Step_test(); 
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	Eigen::VectorXd GetState(); // get p and v
	Eigen::VectorXd GetControlState(); // get delayed states 
	Eigen::VectorXd GetFullObservation(); // get p and v and action and target pos
	Eigen::VectorXd GetControlCOP(); 
	Eigen::VectorXd GetRootInfo() {return mroot_relInfo;}
	int GetNumRootInfo() {return mNumRootInfo;}
	void SetAction(const Eigen::VectorXd& a);
	double GetReward();
	std::tuple<double,double,double,double,double,double,double,Eigen::VectorXd,double,double,double,double,double,double,double,double,double,double> GetRenderReward_Error();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	Character* GetCharacter(){return mCharacter;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};
	int GetNumState(){return mNumState;}
	int GetNumFullObservation(){return mNumFullObservation;}
	int GetNumFutureTargetmotions() {return mNumFutureTargetmotions;}
	int GetNumStateHistory() {return mNumStatehistory;}
	int GetNumAction(){return mNumActiveDof;}
	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	bool GetUseMuscle(){return mUseMuscle;}
	bool GetWalkSkill() {return walk_skill;}
	bool GetSquatSkill() {return squat_skill;}
	bool GetUseMuscleNN(){return mUseMuscleNN;}
	bool GetUseHuntContactForce()  {return mUseHuntContactForce;}
	bool GetUseSymmetry(){return mSymmetry;}
	bool GetUsetargetvisual() {return mUsetarget_visual;}
	std::map<std::string,Eigen::Vector3d> Get_mTargetEE_pos(){return mTargetEE_pos;}
	Eigen::Vector3d geo_center_target, geo_center_target_left, geo_center_target_right;
	std::map<std::string, Reward*> GetmReward(){return mReward;}
    Eigen::VectorXd GetTargetObservations();
	Eigen::VectorXd GetCOPRelative();
	Eigen::VectorXd GetAction() {return mAction;}
	void randomize_masses(double lower_bound, double upper_bound);
	void randomize_inertial(double lower_bound, double upper_bound);
	void randomize_motorstrength(double lower_bound, double upper_bound);
	void randomize_controllatency(double lower_bound, double upper_bound);
	void randomize_friction(double lower_bound, double upper_bound);
	void randomize_centerofmass(double lower_bound, double upper_bound);
    std::tuple<std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>> getHuntContactInfo() \
		{return std::make_tuple(contact_pos_left, contact_force_left, contact_pos_right,contact_force_right);}

	void applyContactForce(); 
 	// void applyWeldJointconstraint();

	void UpdateStateBuffer();
	void UpdateActionBuffer(Eigen::VectorXd action);
	void UpdateTorqueBuffer(Eigen::VectorXd torque);

private:
	
	std::vector<Eigen::Vector3d> contact_pos_left; 
	std::vector<Eigen::Vector3d> contact_pos_right; 
	std::vector<Eigen::Vector3d> contact_force_left; 
	std::vector<Eigen::Vector3d> contact_force_right;
	FixedQueue<Eigen::VectorXd> history_buffer_true_state;
	FixedQueue<Eigen::VectorXd> history_buffer_control_state;
	FixedQueue<Eigen::VectorXd> history_buffer_true_COP;
	Eigen::VectorXd control_COP; 
	FixedQueue<Eigen::VectorXd> history_buffer_action;
	FixedQueue<Eigen::VectorXd> history_buffer_torque;
	FixedQueue<Eigen::VectorXd>  history_buffer_root;
    dart::constraint::WeldJointConstraintPtr mWeldJoint;
	std::map<std::string, Reward*> mReward;
	dart::simulation::WorldPtr mWorld;
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	bool mUseMuscleNN;
	bool mUseHuntContactForce;
	bool mSymmetry;
	bool mUseCOP;
	bool mUsehuman;
	bool mCOMindependent;
	bool mUsejointconstraint;
	bool mUsetarget_visual;
	bool squat_skill, walk_skill;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	Eigen::VectorXd mCurrentAction, mPrevAction; 
	Eigen::VectorXd dynamic_torque;
	Eigen::VectorXd mAction;
	Eigen::VectorXd mTargetPositions,mTargetVelocities,mHuman_initial;
	Eigen::VectorXd mroot_relInfo; 
	std::map<std::string,Eigen::Vector3d> mTargetEE_pos;
    Eigen::VectorXd Initial_masses;
	Eigen::MatrixXd Initial_inertia;
	Eigen::VectorXd Initial_centerofmass;
	int mNumState;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumFullObservation;
	int mNumFutureTargetmotions;
	int mNumStatehistory;
	int cnt_step;
	int mNumRootInfo;
	Eigen::VectorXd randomized_strength_ratios;
	double randomized_latency;
	double observation_latency;
	double mterminal_time;
	Eigen::VectorXd mActivationLevels;
	Eigen::VectorXd mAverageActivationLevels;
	Eigen::VectorXd mDesiredTorque;
	Eigen::VectorXd mLastDesiredTorque;
	Eigen::VectorXd mcur_joint_vel;
	Eigen::VectorXd mlast_joint_vel;
	std::vector<MuscleTuple> mMuscleTuples;
	MuscleTuple mCurrentMuscleTuple;
	int mSimCount;
	int mRandomSampleIndex;
	double lastUpdateTimeStamp; 
	double w_q,w_v,w_ee,w_com,w_COP,w_torque,w_root;
	double w_sroot,w_saction, w_storque,w_sjoint_vel;
	double W_footclr;
	double w_hip, w_knee, w_ankle;
	double kp,kv;
	double high_clearance, low_clearance;
	double foot_tolerance;
    std::vector<size_t> const_index;
	HuntCrosslyContact mHct;
};
};

#endif