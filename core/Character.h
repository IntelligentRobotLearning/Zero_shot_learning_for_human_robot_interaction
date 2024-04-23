#ifndef __MASS_CHARACTER_H__
#define __MASS_CHARACTER_H__
#include "dart/dart.hpp"

namespace MASS
{
class BVH;
class Muscle;
class Force;
class Environment;


class Character
{
public:
	Character();

	void LoadSkeleton(const std::string& path,bool create_obj = false);
	void MergeHuman(const std::string& path,bool create_obj = false);
	void LoadMap(const std::string& path);
	void LoadMuscles(const std::string& path);
	void LoadBVH(const std::string& path,bool cyclic=true);
	void LoadHumanInitialState(const std::string& path);
	void LoadSkeletonInitialState(const std::string& path);
	void LoadModelComponents(const std::string& path);
	void LoadConstraintComponents(const std::string& path);
	void LoadHumanforce(const std::string& path);
	void Reset();	
	void SetPDParameters(double kp, double kv);
	void AddEndEffector(const std::string& body_name){mEndEffectors.push_back(mSkeleton->getBodyNode(body_name));}
	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired);
    Eigen::VectorXd GetHumanInitialState(){return mHumanInitialState;}
	Eigen::VectorXd GetSkeletonInitialState(){return mSkelInitialState;}
	std::tuple<Eigen::VectorXd, std::map<std::string,Eigen::Vector3d>> GetTargetPositions(double t,double dt);
	std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> GetTargetPosAndVel(double t,double dt);
	std::vector< std::tuple<std::string,std::string, Eigen::Vector3d, Eigen::Vector3d>> GetJointConstraints() {return jointConstraint;}
	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<Force*>& GetForces() {return mForces;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	const std::map<std::string, Eigen::Vector3d>& GetEndEffectorsOffset(){return mEndEffectorsOffset;}
	const std::vector<std::string>& GetEndEffectorsName(){return mEndEffectorsName;}
	BVH* GetBVH(){return mBVH;}
	// Eigen::VectorXd GetTargetEEPos(double t); 
	std::map<std::string,Eigen::Vector3d> targetEE_pos; 
	void SetEnvironment(Environment& env) {mEnv = &env;} 
    int Gethumandof() {return mhumandof;}
	int Getskeletondof() {return mskeletondof;}
	void ChangeHumanframe();
	Eigen::Isometry3d GetmTc() {return mTc;}
	void SetmTc(Eigen::Isometry3d a) {mTc = a;}
public:
	Eigen::Isometry3d humanInitialRelativeTransform; 
	Environment* mEnv;
	dart::dynamics::SkeletonPtr mSkeleton;
	BVH* mBVH;
	Eigen::Isometry3d mTc;
	Eigen::Isometry3d mTc_EE;
	std::vector<Muscle*> mMuscles;
	std::vector<Force*> mForces;
	std::vector<std::string> mEndEffectorsName;
	std::vector<std::string> mEndEffectorsBVHName;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;
	std::map<std::string, Eigen::Vector3d> mEndEffectorsOffset;
	std::map<std::string, Eigen::Vector3d> mBVHEE_offset;
	Eigen::VectorXd mHumanInitialState;
	Eigen::VectorXd mSkelInitialState;
	Eigen::VectorXd mKp, mKv;
	int mhumandof;
	int mskeletondof;
	int mskeletonJoints;
	dart::constraint::WeldJointConstraintPtr mWeldJoint;
	std::vector< std::tuple<std::string,std::string, Eigen::Vector3d, Eigen::Vector3d> > jointConstraint;
	bool walk_skill,squat_skill;
	double mhuman_rotation;
	Eigen::Vector3d mhuman_translation;
};
};

#endif
