#ifndef __MASS_BVH_H__
#define __MASS_BVH_H__
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <initializer_list>
#include "dart/dart.hpp"
 
namespace MASS
{
Eigen::Matrix3d R_x(double x);
Eigen::Matrix3d R_y(double y);
Eigen::Matrix3d R_z(double z);
class BVHNode
{
public:
	enum CHANNEL
	{
		Xpos=0,
		Ypos=1,
		Zpos=2,
		Xrot=3,
		Yrot=4,
		Zrot=5
	}; 
	static std::map<std::string,MASS::BVHNode::CHANNEL> CHANNEL_NAME;


	BVHNode(const std::string& name,BVHNode* parent);
	void SetChannel(int c_offset,std::vector<std::string>& c_name);
	void Set(const Eigen::VectorXd& m_t);
	void Set(const Eigen::Matrix3d& R_t);
	void SetEE(const Eigen::VectorXd& m_t);
	void SetEE_Matrix(const Eigen::Matrix3d& R_t);
	Eigen::Matrix3d Get();
	Eigen::Matrix3d GetEE(){return mR_EE;}
	

	void AddChild(BVHNode* child);
	BVHNode* GetNode(const std::string& name);
	int GetmChannelOffset(){return mChannelOffset;}
	std::string GetName(){return mName;}
	std::vector<BVHNode*> GetChildren(){return mChildren;}
	Eigen::Isometry3d T_Global;
	void SetOffset(double x, double y, double z);
	Eigen::Vector3d GetOffset(){return offset;} 
	void SetEndEffectorFlag(bool flag){isEndEffector = flag;}
	bool isEndEffector;

private:
     
	BVHNode* mParent;
	std::vector<BVHNode*> mChildren;

	Eigen::Matrix3d mR_EE; 
	Eigen::Matrix3d mR;
	std::string mName;

	int mChannelOffset;
	int mNumChannels;
	std::vector<BVHNode::CHANNEL> mChannel;
	// offset: the translation (x,y,z) from parent to current node.
	Eigen::Vector3d offset = Eigen::VectorXd::Zero(3);
};

class BVH
{
public:
	BVH(const dart::dynamics::SkeletonPtr& skel,const std::map<std::string,std::string>& bvh_map, const std::vector<std::string>& ee_names, const std::map<std::string, Eigen::Vector3d>& EndEffectorsOffset, const std::map<std::string, Eigen::Vector3d>& BVHEEOffset);

	std::tuple<Eigen::VectorXd, BVHNode*> GetMotion(double t);

	Eigen::Matrix3d Get(const std::string& bvh_node);

	double GetMaxTime(){return (mNumTotalFrames)*mTimeStep;}
	double GetTimeStep(){return mTimeStep;}
	void Parse(const std::string& file,bool cyclic=true);
	
	const std::map<std::string,std::string>& GetBVHMap(){return mBVHMap;}
	const std::map<std::string, Eigen::Vector3d>& GetEEOffsetMap(){return mEEOffsetMap;}
	const Eigen::Isometry3d& GetT0(){return T0;}
	const Eigen::Isometry3d& GetT1(){return T1;}
	const Eigen::Isometry3d& GetT0_EE(){return T0_EE;}
	const Eigen::Isometry3d& GetT1_EE(){return T1_EE;}
	bool IsCyclic(){return mCyclic;}
	void SetGlobal(BVHNode* bn, std::map<std::string,Eigen::Vector3d>& mEEPosMap);
	void setBVHNodeEndEffectorFlag(BVHNode* bn);

    std::map<std::string, int> BVHNodeOffset;
	double unit_scale = 1;
	double translation_scale = 0;
	bool walk_skill,squat_skill;
	
private:
	bool mCyclic;
	std::vector<Eigen::VectorXd> mMotions;
	std::map<std::string,BVHNode*> mMap;
	double mTimeStep;
	int mNumTotalChannels;
	int mNumTotalFrames;

	BVHNode* mRoot;
	
	dart::dynamics::SkeletonPtr mSkeleton;
	std::map<std::string,std::string> mBVHMap;
	std::vector<std::string> mBVHEE;
	std::map<std::string, Eigen::Vector3d> mEEOffsetMap;
	std::map<std::string, Eigen::Vector3d> mBVHEEOffsetMap;

	Eigen::Isometry3d T0,T1,T0_EE,T1_EE;   
	BVHNode* ReadHierarchy(BVHNode* parent,const std::string& name,int& channel_offset,std::ifstream& is);
	
};

};

#endif