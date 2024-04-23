#include "Character.h"
#include "BVH.h"
#include "DARTHelper.h"
#include "Muscle.h"
#include "Force.h"
#include <tinyxml.h>
#include <stdio.h>
#include <dart/gui/gui.hpp>
#include <dart/utils/urdf/urdf.hpp>
#include <dart/utils/sdf/SdfParser.hpp>

using namespace dart;
using namespace dart::dynamics;
using namespace MASS;
Character::
Character()
	:mSkeleton(nullptr),mBVH(nullptr),mTc(Eigen::Isometry3d::Identity())
{

}

void
Character::
LoadSkeleton(const std::string& path,bool create_obj)
{
	if(path.substr(path.size()-5,5) == ".urdf") {
		dart::utils::DartLoader loader; // load .urdf file
		mSkeleton = loader.parseSkeleton(path); //the name shall be loaded inside
		mSkeleton->setName("NJIT_BME_EXO_Model");
		std::cout << "skel_name:  " << mSkeleton->getName() << std::endl;    
		BodyNode* bn1 = mSkeleton->getBodyNode("r_foot_ground");
		BodyNode* bn2 = mSkeleton->getBodyNode("l_foot_ground");
		bn1->setFrictionCoeff(1.0);
		bn2->setFrictionCoeff(1.0);	
		}
	else if(path.substr(path.size()-4,4) == ".sdf") {
		mSkeleton = dart::utils::SdfParser::readSkeleton(path); //the name shall be loaded inside
	}
	else {
		throw std::runtime_error("do not know how to load skeleton " + path);
	}
    mskeletondof = mSkeleton->getNumDofs();
	mskeletonJoints = mSkeleton->getNumJoints();
	mhumandof = 0;
}

void
Character::
MergeHuman(const std::string& path, bool create_obj)
{
	BuildFromFile_Merge(mSkeleton,path,create_obj);   // merge human skeleton
    mhumandof = mSkeleton->getNumDofs()-mskeletondof;
	// std::cout << "humandof:\n" << mhumandof << std::endl;
}


void
Character::
LoadMap(const std::string& path)
{
    
	std::map<std::string,std::string> bvh_map;
	TiXmlDocument doc;
	doc.LoadFile(path);

	TiXmlElement *skel_elem = doc.FirstChildElement("Skeleton");  // Skeleton 元素

	for(TiXmlElement* node = skel_elem->FirstChildElement("Node"); node != nullptr; node = node->NextSiblingElement("Node"))
	{
		TiXmlElement* joint_elem = node->FirstChildElement("Joint");
		if(joint_elem->Attribute("bvh")!=nullptr)
		{
			bvh_map.insert(std::make_pair(node->Attribute("name"),joint_elem->Attribute("bvh")));
		}

		if(node->Attribute("endeffector")!=nullptr)
		{
			
			std::string ee =node->Attribute("endeffector");
			if(ee == "True")
			{
				mEndEffectors.push_back(mSkeleton->getBodyNode(std::string(node->Attribute("name"))));
				mEndEffectorsName.push_back(std::string(node->Attribute("name"))); 

				if(joint_elem->Attribute("bvh")!=nullptr)
				{
					mEndEffectorsBVHName.push_back(std::string(joint_elem->Attribute("bvh")));
					Eigen::Vector3d offset;
					Eigen::Vector3d BVHendeffector_offset;
					if(node->Attribute("endeffector_offset")!=nullptr)
						offset = string_to_vector3d(node->Attribute("endeffector_offset"));
					else
						offset = Eigen::VectorXd::Zero(3); 
					// 	std::map<std::string, Eigen::Vector3d> mEndEffectorsOffset;
					mEndEffectorsOffset.insert(std::make_pair(std::string(node->Attribute("name")), offset));
					// std::cout<< "offset" << offset <<std::endl;
                    if(joint_elem->Attribute("BVHEE_offset")!=nullptr)
					   	BVHendeffector_offset = string_to_vector3d(joint_elem->Attribute("BVHEE_offset"));
					else
						BVHendeffector_offset = Eigen::VectorXd::Zero(3); 
					mBVHEE_offset.insert(std::make_pair(std::string(joint_elem->Attribute("bvh")), BVHendeffector_offset));
				}
				else{
					std::cout<< "-----------------------" <<std::endl;
					std::cout<< "Endeffector should have a BVH name!!" <<std::endl;
					std::cout<< "-----------------------" <<std::endl;
				}

			}

		}

	}
    
	mBVH = new BVH(mSkeleton,bvh_map,mEndEffectorsBVHName,mEndEffectorsOffset,mBVHEE_offset);

}

void
Character::
LoadHumanInitialState(const std::string& path)
{

	std::ifstream is(path);
	if(!is)
	{
		std::cout<<"Can't open file " << path <<std::endl;
		return;
	}
	double val;
	int cnt;
	is >> cnt;
	mHumanInitialState.resize(cnt);
    // std::map<std::string, int> IndexOffset;
	int numhumanjoints= mSkeleton->getNumJoints()-mskeletonJoints; // remove skeleton joints
    std::vector<std::string> JointName;
	std::string name; 
	// Eigen::VectorXd index;
	char buffer[1000];
	for(int i=0; i<numhumanjoints; i++)
	{
         is >> name;
	}
	for(int i=0; i<cnt; i++)
	{
		is >> val;
        mHumanInitialState[i]= val;
	} 
	double angle;
	is >> buffer;
	is >> buffer;
	if(!strcmp(buffer,"rotation"))
	{
		is >> angle;
		mhuman_rotation = angle; 
	}
	is >> buffer;
	if(!strcmp(buffer,"translation"))
	{
		is >> buffer;
		double mhuman_translation_x = atof(buffer);
		is >> buffer;
		double mhuman_translation_y = atof(buffer);
		is >> buffer;
		double mhuman_translation_z = atof(buffer);
		mhuman_translation << mhuman_translation_x, mhuman_translation_y, mhuman_translation_z;
	}
}

void
Character::
LoadSkeletonInitialState(const std::string& path)
{

	std::ifstream is(path);
	if(!is)
	{
		std::cout<<"Can't open file " << path <<std::endl;
		return;
	}
	double val;
	int cnt;
	is >> cnt;
	mSkelInitialState.resize(mSkeleton->getNumDofs());
	std::string name; 
	char buffer[1000];

	for(int i=0; i<mSkeleton->getNumDofs(); i++)
	{
         is >> name;
	}
	for(int i=0; i<cnt; i++)
	{
		is >> val;
        mSkelInitialState[i]= val;
		
	} 
	std::cout << mSkelInitialState << std::endl;
}


void
Character::
LoadMuscles(const std::string& path)
{	
	// mSkeleton // robot 

	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
		return;
	}
	
	TiXmlElement *muscledoc = doc.FirstChildElement("Muscle");
	for(TiXmlElement* unit = muscledoc->FirstChildElement("Unit");unit!=nullptr;unit = unit->NextSiblingElement("Unit"))
	{
		std::string name = unit->Attribute("name");
		double f0 = std::stod(unit->Attribute("f0"));
		double lm = std::stod(unit->Attribute("lm"));
		double lt = std::stod(unit->Attribute("lt"));
		double pa = std::stod(unit->Attribute("pen_angle"));
		double lmax = std::stod(unit->Attribute("lmax"));
		mMuscles.push_back(new Muscle(name,f0,lm,lt,pa,lmax));
		int num_waypoints = 0;
		bool hasTibia = false; 
		for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))
		{	
			num_waypoints++;
			std::string body = waypoint->Attribute("body");
			if (( body=="TibiaL") ||  (body=="TibiaR")){
				hasTibia = true;
			}
		}
		int i = 0;
		for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))	
		{
			std::string body = waypoint->Attribute("body");
			Eigen::Vector3d p_pos = string_to_vector3d(waypoint->Attribute("p"));
			if(hasTibia){
				 p_pos[1]+=0.11;
			}
			Eigen::Vector3d glob_pos = p_pos;
			if(i==0||i==num_waypoints-1)
			// if(true)
				mMuscles.back()->AddAnchor(mSkeleton->getBodyNode(body),glob_pos);
			else
				mMuscles.back()->AddAnchor(mSkeleton,mSkeleton->getBodyNode(body),glob_pos,2);
			i++;
		}
	}
	
}



void
Character::
ChangeHumanframe()    // change the frame and position for the human or muscle. 
{
	Joint* parentJoint =mSkeleton->getBodyNode("Pelvis")->getParentJoint();
	Eigen::Isometry3d T = parentJoint->getTransformFromParentBodyNode();
	T.linear() = T.linear()* R_y(mhuman_rotation);
	T.translation()(0) = T.translation()(0)+mhuman_translation(0);
	T.translation()(1) = T.translation()(1)+mhuman_translation(1);
	T.translation()(2) = T.translation()(2)+mhuman_translation(2);
	parentJoint->setTransformFromParentBodyNode(T);         
}




void
Character::
LoadBVH(const std::string& path,bool cyclic)
{
	if(mBVH ==nullptr){
		std::cout<<"Initialize BVH class first"<<std::endl;
		return;
	}
	if(walk_skill==true)
	{
		std::cout << "-----walking imitation-----" << std::endl;
		mBVH->walk_skill = true;
	}
	else if(squat_skill==true)
	{
		std::cout << "-----squatting imitation-----" << std::endl;
		mBVH->squat_skill = true;
	}
	mBVH->Parse(path,cyclic);
}

void
Character::
LoadModelComponents(const std::string& path)
{
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
		return;
	}
	TiXmlElement *forcedoc = doc.FirstChildElement("Forces");
	for(TiXmlElement* forceelm = forcedoc->FirstChildElement();forceelm!=nullptr;forceelm = forceelm->NextSiblingElement())
	{
		std::string forcetype = forceelm->Value();
		auto force = ForceFactory::CreateForce(forcetype);
		if(!force) {
			std::cout << "Can't create force type " << forcetype << std::endl;
			continue;
		}
		force->ReadFromXml(*forceelm);
		force->SetBodyNode(mSkeleton); 
		force->SetEnvironment(*mEnv); 
		force->Update();     // random pos offset 
		force->UpdatePos();  // set mPos 
		// std::cout << force->GetName() << std::endl; 
		mForces.push_back(force);
	}

}

void
Character::
LoadConstraintComponents(const std::string& path)
{
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
		return;
	}
	TiXmlElement *constraintdoc = doc.FirstChildElement("JointConstraint");
	for(TiXmlElement* Constraint = constraintdoc->FirstChildElement("Constraint");Constraint!=nullptr;Constraint = Constraint->NextSiblingElement("Constraint"))
	{
		std::string jointname = Constraint->Attribute("jointname");
		std::string connectedjointname = Constraint->Attribute("connectedjointname");
		Eigen::Vector3d JointPos = string_to_vector3d(Constraint->Attribute("JointPos"));
		Eigen::Vector3d connectedJointPos = string_to_vector3d(Constraint->Attribute("connectedJointPos"));
		jointConstraint.push_back(std::make_tuple(jointname, connectedjointname, JointPos, connectedJointPos));
	}
}


void
Character::
LoadHumanforce(const std::string& path)
{
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
		return;
	}
	// std::cout<< "spring force here"  <<  std::endl;
	TiXmlElement *forcedoc = doc.FirstChildElement("Forces");
	for(TiXmlElement* forceelm = forcedoc->FirstChildElement();forceelm!=nullptr;forceelm = forceelm->NextSiblingElement())
	{
		std::string forcetype = forceelm->Value();
		auto force = ForceFactory::CreateForce(forcetype);
		if(!force) {
			std::cout << "Can't create force type " << forcetype << std::endl;
			continue;
		}
		force->ReadFromXml(*forceelm);

		force->SetBodyNode(mSkeleton);
		force->SetEnvironment(*mEnv); 
		force->Update();     // random pos offset 

		force->UpdatePos();  // set mPos 
		mForces.push_back(force);
	}
}


void
Character::
Reset()
{

	mTc = mBVH->GetT0();
	if(walk_skill==true)
		mTc.translation()[1] = -0.08;
	else if(squat_skill==true)
	{
		mTc.translation()[0] = 0;
		mTc.translation()[1] = 0;
		mTc.translation()[2] = 0.0;
	}

	mTc_EE = mBVH->GetT0_EE();
	mTc_EE.translation()[0] = 0.0;
	mTc_EE.translation()[1] = 0.1;
	mTc_EE.translation()[2] = 0.0;
	
	for(auto force : mForces)
	{
		force->Reset();
	}
}

void
Character::
SetPDParameters(double kp, double kv)
{
	int dof = mSkeleton->getNumDofs();
	mKp = Eigen::VectorXd::Constant(dof,kp);	//
	mKv = Eigen::VectorXd::Constant(dof,kv);	//
}
Eigen::VectorXd
Character::        // use Stable PD control method
GetSPDForces(const Eigen::VectorXd& p_desired)
{
	Eigen::VectorXd q = mSkeleton->getPositions();
	Eigen::VectorXd dq = mSkeleton->getVelocities();
	double dt = mSkeleton->getTimeStep();
	Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();
	Eigen::VectorXd qdqdt = q + dq*dt;
	Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt,p_desired));
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq);
	Eigen::VectorXd ddq = M_inv*(-mSkeleton->getCoriolisAndGravityForces()+p_diff+v_diff+mSkeleton->getConstraintForces());
	Eigen::VectorXd tau = p_diff + v_diff - dt*mKv.cwiseProduct(ddq);
	tau.head<6>().setZero();
	tau.tail(mhumandof).setZero();

	return tau;
}

std::tuple<Eigen::VectorXd, std::map<std::string,Eigen::Vector3d>>
Character::
GetTargetPositions(double t,double dt)
{
	std::tuple<Eigen::VectorXd, BVHNode*> tmp = mBVH->GetMotion(t);
   
	Eigen::VectorXd p = std::get<0>(tmp);
	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(p.head<6>());  // Convert a FreeJoint-style 6D vector P. head (6) into a transform.
	T_current = mBVH->GetT0().inverse()*T_current;

	Eigen::Isometry3d T_head = mTc*T_current;         // How to calculate mTc ?  mTc is the TO (bvh root relative to world ?), but the translation in Y-axis is zero.
	                                                  // mTc*mBVH->GetT0().inverse() is the identity matrix 
													  // rotation is an identity matrix, the translation[1] has no change
	Eigen::Vector6d p_head = dart::dynamics::FreeJoint::convertToPositions(T_head);  // Convert a transform into a 6D vector that can be used to set the positions of a FreeJoint.
	p.head<6>() = p_head;   //the six dof for root joint 
	if(walk_skill==true)
	{
		if(mBVH->IsCyclic())
		{
			double t_mod = std::fmod(t, mBVH->GetMaxTime());
			t_mod = t_mod/mBVH->GetMaxTime();
			
			double r = 0.95;
			if(t_mod>r)
			{
				double ratio = 1.0/(r-1.0)*t_mod - 1.0/(r-1.0);
				Eigen::Isometry3d T01 = mBVH->GetT1()*(mBVH->GetT0().inverse());
				double delta = T01.translation()[1];
				delta *= ratio;
				p[5] += 0;

			}

			double tdt_mod = std::fmod(t+dt, mBVH->GetMaxTime());
			if(tdt_mod-dt<0.0){
				Eigen::Isometry3d T01 = mBVH->GetT1()*(mBVH->GetT0().inverse());
				Eigen::Vector3d p01 = dart::math::logMap(T01.linear());
				p01[0] =0.0;
				p01[2] =0.0;
				T01.linear() = dart::math::expMapRot(p01);
				mTc = T01*mTc;
				mTc.translation()[1] =-0.08;
			}
		}
	}

	// caculate the EE target pos based on mTc_EE and mRoot. 
	std::map<std::string,Eigen::Vector3d> tmp_mEEPosMap;

	return std::make_tuple(p, tmp_mEEPosMap);
}


std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>>
Character::
GetTargetPosAndVel(double t,double dt)
{	

	std::tuple<Eigen::VectorXd, std::map<std::string,Eigen::Vector3d>> tmp = this->GetTargetPositions(t,dt);
	Eigen::VectorXd p = std::get<0>(tmp);
	std::map<std::string,Eigen::Vector3d> mEEPosMap = std::get<1>(tmp);

	Eigen::Isometry3d Tc = mTc;
	Eigen::Isometry3d Tc_EE = mTc_EE;
	tmp = this->GetTargetPositions(t+dt,dt);
	Eigen::VectorXd p1 = std::get<0>(tmp);
	mTc = Tc;
	mTc_EE = Tc_EE;

	return std::make_tuple(p,(p1-p)/dt, mEEPosMap);
}
