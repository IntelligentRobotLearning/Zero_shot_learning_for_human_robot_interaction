#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "Force.h"
#include "BodyForce.h"
#include "SpringForce.h"
#include "Reward.h"
#include "COPReward.h"
#include "COMReward.h"
#include "torqueReward.h"
#include<cmath>
#include <algorithm> 
#include <random>
//#include "Fixedeque.h"

#include "dart/collision/bullet/bullet.hpp"
#include "dart/constraint/ContactConstraint.hpp"
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;

/**
 * A HuntCrossleyContact formulation with power law
 * The force in the normal direction is based on a model due to Hunt & Crossley: 
 * "Coefficient of Restitution Interpreted as Damping in Vibroimpact,"
 * ASME Journal of Applied Mechanics, pp. 440-445, June 1975
 * The friction force is based on a model by Michael Hollars:
 * f = fn*[min(vs/vt,1)*(ud+2(us-ud)/(1+(vs/vt)^2))+uv*vs]
 * see https://github.com/simbody/simbody/blob/master/Simbody/src/HuntCrossleyForce.cpp
 * 
 */

HuntCrosslyContact::HuntCrosslyContact()
{
	stiffness = 1800;
	dissipation = 0.0;
	staticFriction = 0.8;
	dynamicFriction = 0.6;
	viscousFriction =0.0;
	radius =  1e0; //1m
	transitionVelocity = 8.0e-2;
	//order = 2;
}

Environment::
Environment()
	:mControlHz(30),mSimulationHz(900),mterminal_time(10.0),mWorld(std::make_shared<World>()),mUseMuscle(true),mUseMuscleNN(true),mUseHuntContactForce(false), mSymmetry(true), mCOMindependent(true), mUsehuman(false), observation_latency(0.04), w_q(0.7),w_v(0.1),w_ee(0.5),w_com(0.4),w_torque(0.00),w_root(0.4),w_hip(0.3),w_knee(0.7),w_ankle(0.3),w_sroot(0.0),w_saction(0.0),w_storque(0.0),w_sjoint_vel(0.0),W_footclr(0.0)
{
	// mRewards = RewardFactory::mRewards;	
	history_buffer_true_state.setMaxLen(STATE_HISTORY_BUFFER_LEN);
	history_buffer_control_state.setMaxLen(STATE_HISTORY_BUFFER_LEN);
	history_buffer_true_COP.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_action.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_torque.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_root.setMaxLen(STATE_HISTORY_BUFFER_LEN);

}
// initialize ： read metal_file，load use_muscle, con_hz, sim_hz, skel_file, map_file, bvh_file and reward_param
void
Environment::
Initialize(const std::string& meta_file,bool load_obj)
{	
	std::ifstream ifs(meta_file);
	if(!(ifs.is_open()))
	{
		std::cout<<"Can't read file "<<meta_file<<std::endl;
		return;
	}
	std::string str;        // define str
	std::string index;      // define index
	std::stringstream ss;   // define ss
	MASS::Character* character = new MASS::Character();   //
	character ->SetEnvironment(*this);

	while(!ifs.eof())
	{
		str.clear();
		index.clear();
		ss.clear();

		std::getline(ifs,str);
		ss.str(str);
		ss>>index;
		if(!index.compare("skill"))
		{	
			std::string str2;
			ss>>str2;    
			if(!str2.compare("walking")) 
			{                       
				this->SetWalkingSkill(true);
				character->walk_skill=true;
			}
			else if(!str2.compare("squatting"))  
			{
				this->SetSquattingSkill(true);
				character->squat_skill=true;
			}
		}
		if(!index.compare("use_muscle"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseMuscle(true);           
			else                                     //mUseMuscle = true
				this->SetUseMuscle(false);
		}
		if(!index.compare("use_muscleNetWork"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseMuscleNN(true);           
			else                                     //mUseMuscle = true
				this->SetUseMuscleNN(false);
		}
       
		else if(!index.compare("use_Huntcontact"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseHuntContactForce(true);           
			else                                     //mUseHuntContactForce = true
				this->SetUseHuntContactForce(false);
		}
		else if(!index.compare("contact_model"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("Hunt_Crossley")) { //input 7 numbers
				ss >> mHct.stiffness;
				ss >> mHct.dissipation;
				ss >> mHct.staticFriction;
				ss >> mHct.dynamicFriction;
				ss >> mHct.viscousFriction;
				ss >> mHct.radius; 
				ss >> mHct.transitionVelocity;
			}
			else {

			}
		}
		else if(!index.compare("symmetry"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetSymmetry(true);           
			else                                     //mSymmetry = true
				this->SetSymmetry(false);
		}
		else if(!index.compare("use_COP"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseCOP(true);           
			else                                     //mUseCOP = true
				this->SetUseCOP(false);
		}
		else if(!index.compare("com_independent"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetCOMindependent(true);           
			else                                     //mCOMindependent = true
				this->SetCOMindependent(false);
		}
		else if(!index.compare("con_hz")){
			int hz;
			ss>>hz;                              // control_hz = 30hz = mControlHz
			this->SetControlHz(hz);
		}
		else if(!index.compare("sim_hz")){
			int hz;
			ss>>hz;                             // sim_hz = 600hz  = mSimulationHz
			this->SetSimulationHz(hz);
		}
		else if(!index.compare("terminal_time")){	
			double terminal_time;	
			ss>>terminal_time;                            	
			this->SetTerminalTime(terminal_time);	
		}
		else if(!index.compare("PD_param")){             // read PD control parameter
			double a;
			ss>>a;
			this->SetPDParameters(a);  
		}
		else if(!index.compare("foot_clearance_termination")){
			double a,b;
			ss>>a>>b;
			this->SetFootClearance(a,b); 
		}
		else if(!index.compare("foot_tolerance_termination")){
			double a;
			ss>>a;
			this->SetFootTolerances(a);  
		}
		else if(!index.compare("target_motion_visualization"))	
		{		
			std::string str2;	
			ss>>str2;	
			if(!str2.compare("true"))	
				this->Settargetmotion_visual(true);           	
			else                                    	
				this->Settargetmotion_visual(false);	
		}
		else if(!index.compare("skel_file")){
			std::string str2;                                                               //read skel_file
			ss>>str2;
			character->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2,load_obj);

		}
		else if(!index.compare("human_file")){
			std::string str2,str3;                                                               //read human_file
			ss>>str2>>str3;
			if(!str3.compare("true"))
			{
				character->MergeHuman(std::string(MASS_ROOT_DIR)+str2,load_obj);
				mUsehuman = true;
			}
		}
		else if(!index.compare("muscle_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseMuscle())
				character->LoadMuscles(std::string(MASS_ROOT_DIR)+str2);
			if(mUsehuman)
				character->ChangeHumanframe();
		}
		else if(!index.compare("model_component_file")){
			std::string str2,str3;
			ss>>str2>>str3;
			if(!str3.compare("true"))
				character->LoadModelComponents(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("Human_spring_force_file")){
			std::string str2,str3;    //str2： path, str3: true
			ss>>str2>>str3;
			if(!str3.compare("true"))
				character->LoadHumanforce(std::string(MASS_ROOT_DIR)+str2);
		}		
		else if(!index.compare("map_file")){
			std::string str2;                                                             //read map_file
			ss>>str2;
			character->LoadMap(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("human_squatting_initial_state")){ 
			std::string str2;                                                             //read human initial state for squatting
			ss>>str2;
			if (character->squat_skill)
				character->LoadHumanInitialState(std::string(MASS_ROOT_DIR)+str2);
         }
		else if(!index.compare("human_walking_initial_state")){ 
			std::string str2;                                                             //read human initial state for walking
			ss>>str2;
			if (character->walk_skill)
				character->LoadHumanInitialState(std::string(MASS_ROOT_DIR)+str2);
         }
		 else if(!index.compare("Joint_constraint_file")){
			std::string str2,str3;
			ss>>str2>>str3;
			if(!str3.compare("true"))
				character->LoadConstraintComponents(std::string(MASS_ROOT_DIR)+str2);
	
		}
		// else if(!index.compare("bvh_file")){                                              //read bvh_file
		// 	std::string str2,str3;    //str2： path, str3: true
		// 	ss>>str2>>str3;
		// 	bool cyclic = false;
		// 	if(!str3.compare("true"))
		// 		cyclic = true;    
		// 	character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		// }
		else if(!index.compare("motion_file")){                                              //read target motion file
			std::string str2,str3;    //str2： path, str3: true
			ss>>str2>>str3;
			bool cyclic = false;
			if(!str3.compare("true"))
				cyclic = true;    
			character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		}

		else if(!index.compare("reward_param")){
			double a,b,c,d,e,f;
			ss>>a>>b>>c>>d>>e>>f;
			this->SetRewardParameters(a,b,c,d,e,f);     // reward parameters w_q, w_v, w_ee, w_com, w_torque
		}												// 					0.75  0.2  0.45   0.0 0.0 0.4
		else if(!index.compare("smooth_reward_param")){	
			double a,b,c,d;	
			ss>>a>>b>>c>>d;	
			this->SetSmoothRewardParameters(a,b,c,d);   // smoothreward parameters w_sroot, w_saction, w_storque	
		}
		else if(!index.compare("foot_clearance_reward")){	
			double a;	
			ss>>a;	
			this->SetFootClearanceRewardParameter(a);  // foot clearance parameter
		}
		else if(!index.compare("observation_latency")){
			double a;
			ss>>a;
			this->observation_latency = a;
		}
		else if(!index.compare("reward")){
			std::string str2;
			ss >> str2; 
			auto reward = RewardFactory::CreateReward(str2);
			reward->SetEnvironment(*this); 
			reward->ReadFromStream(ss); 
			mReward.insert(std::pair<std::string, Reward*>(reward->GetName(),reward)); 
		}

        else if(!index.compare("joint_reward_weight")){
			double a,b,c;
			ss>>a>>b>>c;
			this->SetJointRewardParameters(a,b,c);
		}
	}
	ifs.close();     
	
	character->SetPDParameters(kp,sqrt(2.0*kp));   // PD parameters： kp=300, kv=sqrt(2*kp)
	this->SetCharacter(character);
	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));
	this->Initialize();
	mUsejointconstraint = false;
}


void
Environment::
Initialize()   // define the related dofs
{
	if(mCharacter->GetSkeleton()==nullptr){
		std::cout<<"Initialize character First"<<std::endl;
		exit(0);
	}
	if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")   //FreeJoint: 6 degree
		mRootJointDof = 6;
	else if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint") //PlanarJoint:
		mRootJointDof = 3;	
	else
		mRootJointDof = 0;
	mNumActiveDof = mCharacter->GetSkeleton()->getNumDofs()-mRootJointDof-mCharacter->Gethumandof();  // remove the root joint dof and human dof
	if(mUseMuscle)
	{
		int num_total_related_dofs = 0;
		for(auto m : mCharacter->GetMuscles()){
			m->Update();    //
			num_total_related_dofs += m->GetNumRelatedDofs();   //num_total_related_dofts = m + num_related_dofs
		}
		mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(num_total_related_dofs);  // define vector JtA
		mCurrentMuscleTuple.L = Eigen::MatrixXd::Zero(mNumActiveDof,mCharacter->GetMuscles().size()); //define Matrix L
		mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof); // define vector b
		mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof); // define vector tau_dex
		mActivationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size()); // define activation levels a
	}
	
	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));  // set gravity
	mWorld->setTimeStep(1.0/mSimulationHz);     // set time step
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->addSkeleton(mCharacter->GetSkeleton());  //
	mWorld->addSkeleton(mGround);


	// if(mUseHuntContactForce) {
	// 	mWorld->getConstraintSolver()->setContactAsConstraints(false); //uncomment after changing dart code
	// }
	
	mAction = Eigen::VectorXd::Zero(mNumActiveDof);  // define mAction vector
	mCurrentAction = Eigen::VectorXd::Zero(mNumActiveDof);  // define mAction vector
	mPrevAction = Eigen::VectorXd::Zero(mNumActiveDof);  // define mAction vector
	dynamic_torque = Eigen::VectorXd::Zero(mNumActiveDof);
	mDesiredTorque = Eigen::VectorXd::Zero(mCharacter->GetSkeleton()->getNumDofs());	
	mcur_joint_vel = Eigen::VectorXd::Zero(mCharacter->GetSkeleton()->getNumDofs()-mRootJointDof);

    for (int i=0;i<mCharacter->Getskeletondof();i++)
		const_index.push_back(i);

	int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	Initial_masses = Eigen::VectorXd::Zero(Numbodynodes);
	Initial_inertia = Eigen::MatrixXd::Zero(3*Numbodynodes,3);
	Initial_centerofmass = Eigen::VectorXd::Zero(Numbodynodes*3);
	for (int i=0;i<Numbodynodes;i++)
	{
       Initial_masses(i) = mCharacter->GetSkeleton()->getBodyNode(i)->getMass();
	//    std::cout << Initial_masses(i) << "\n" << std::endl;
	}
	for (int i=0;i<Numbodynodes;i++)
	{
		const Inertia& iner= mCharacter->GetSkeleton()->getBodyNode(i)->getInertia();
		Initial_inertia.block(i*3,0,3,3) = iner.getMoment();
		// std::cout << Initial_inertia.block(i*3,0,3,3) << "\n" << std::endl;
	}

	for (int i=0;i<Numbodynodes;i++)
	{
		const Inertia& iner= mCharacter->GetSkeleton()->getBodyNode(i)->getInertia();
		Initial_centerofmass.segment<3>(i*3) = iner.getLocalCOM();
	}

	Reset(false);
	
	mNumState = GetState().rows();             // p.rows +v.rows +1
	mNumFullObservation = GetFullObservation().rows();
	mNumFutureTargetmotions = GetTargetObservations().rows();
	mNumStatehistory = mNumState*(STATE_HISTORY_BUFFER_LEN-1);
	mNumRootInfo = GetRootInfo().rows()*(STATE_HISTORY_BUFFER_LEN-1);

	if (mSymmetry)
		std::cout << "Onput of NN:  " << mAction.rows()/2 << std::endl;
	else
		std::cout << "Onput of NN:  " << mAction.rows() << std::endl;
    if (mUseMuscle)
		std::cout << "--use Muscle--" << std::endl;
	else
		std::cout << "--not use Muscle--"  << std::endl;

}
void 
Environment::
Reset(bool RSI)                                       // execute the env.reset() after the terminal state is true
{
	mWorld->reset();                                   // set time = 0, Frame = 0, clear last Collision result in the world

	mCharacter->GetSkeleton()->clearConstraintImpulses();  
	mCharacter->GetSkeleton()->clearInternalForces();
	mCharacter->GetSkeleton()->clearExternalForces();
	double t = 0.0;

	if(RSI)
		t = dart::math::random(0.0,mCharacter->GetBVH()->GetMaxTime()*0.9);  //
	
	mWorld->setTime(t);   // set time in the world 
	mCharacter->Reset();  // reset mT0 = mBVH->GetT0();	 mTc.translation[1] =0;
	
	dynamic_torque.setZero(); 
	mAction.setZero();
	mCurrentAction.setZero();
	mPrevAction.setZero();
	mroot_relInfo = Eigen::VectorXd::Zero(3);

	std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = std::get<0>(pv);
	mTargetVelocities = std::get<1>(pv);
	mTargetEE_pos = std::get<2>(pv);
	if (mUsehuman)
	{
		std::cout << " --Use Human Model--" << std::endl;
		mHuman_initial.resize(mCharacter->Gethumandof());
		mHuman_initial = mCharacter->GetHumanInitialState();
		// std::cout << mCharacter->Gethumandof() << "\n" << mHuman_initial.size() << "\n" << mHuman_initial << std::endl;
		mTargetPositions.tail(mCharacter->Gethumandof()) = mHuman_initial;
	}

    cnt_step =0;
    
    // testing

    randomized_latency = 0;
	randomized_strength_ratios.setOnes(mNumActiveDof);

	/////////randomization
	// randomize_masses(0.8,1.0); 
	// randomize_inertial(0.8,1.5); 
	// // randomize_centerofmass(0.9,1.2); 
	// randomize_motorstrength(0.8,1.2); 
    // randomize_friction(0.9,1.6); 
    // randomize_controllatency(0, observation_latency);
	mCharacter->GetSkeleton()->setPositions(mTargetPositions);
	mCharacter->GetSkeleton()->setVelocities(mTargetVelocities); //set velocities
	mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);

	for(int i=0; i<STATE_HISTORY_BUFFER_LEN; i++)
	{
		history_buffer_true_state.push_back(this->GetState());
		history_buffer_control_state.push_back(this->GetState());
		history_buffer_root.push_back(this->GetRootInfo());

	}

	for(int i=0; i<HISTORY_BUFFER_LEN; i++)
	{
		history_buffer_true_COP.push_back(this->GetCOPRelative()); 
		history_buffer_action.push_back(this->GetAction());
		history_buffer_torque.push_back(this->GetDesiredTorques());
	}


	control_COP = GetControlCOP(); 
	lastUpdateTimeStamp = 0.0; 

	if (mUsehuman)
	{
		if (!mUsejointconstraint)
	     {
			for (auto ss : mCharacter->GetJointConstraints())
			{
				BodyNode* bn1 = mCharacter->GetSkeleton()->getBodyNode(std::get<0>(ss));
				BodyNode* bn2 = mCharacter->GetSkeleton()->getBodyNode(std::get<1>(ss));
				mWeldJoint = std::make_shared<constraint::WeldJointConstraint>(bn1, bn2);
				Eigen::Isometry3d t1 = bn1->getTransform();
				Eigen::Isometry3d t2 = bn2->getTransform();
				t1.translation() = t1*std::get<2>(ss);
				t2.translation() = t2*std::get<3>(ss);
				std::cout << "rel ---------------------------" << std::endl;
				mWeldJoint->setRelativeTransform(t2.inverse()*t1);
				mWorld->getConstraintSolver()->addConstraint(mWeldJoint);
				mUsejointconstraint = true;
			}
	    }
   	}
}


void 
Environment::
randomize_masses(double lower_bound, double upper_bound)
{

    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
    Eigen::VectorXd sample = Eigen::VectorXd::Zero(Numbodynodes);
	Eigen::VectorXd randomized_mass_ratios = Eigen::VectorXd::Zero(Numbodynodes);
	Eigen::VectorXd randomized_mass = Eigen::VectorXd::Zero(Numbodynodes);
    for (int i=0;i<Numbodynodes;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_mass_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}

	randomized_mass = randomized_mass_ratios.cwiseProduct(Initial_masses);
	if (mSymmetry)
	{
		mCharacter->GetSkeleton()->getBodyNode(0)->setMass(randomized_mass(0));
		mCharacter->GetSkeleton()->getBodyNode(1)->setMass(randomized_mass(1));
        for (int i=2;i<Numbodynodes;i++) 
			mCharacter->GetSkeleton()->getBodyNode(i)->setMass(randomized_mass_ratios(2)*Initial_masses(i));
	}
	// else
	// {
	// 	for (int i=1;i<Numbodynodes;i++)
	// 		// std::cout << "mass \n"  <<  Initial_masses(i) << std::endl;
	// 		mCharacter->GetSkeleton()->getBodyNode(i)->setMass(randomized_mass(i));
 	// }
}


void 
Environment::
randomize_inertial(double lower_bound, double upper_bound)
{

    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;

    Eigen::VectorXd sample = Eigen::VectorXd::Zero(Numbodynodes);
	Eigen::VectorXd randomized_inertial_ratios = Eigen::VectorXd::Zero(Numbodynodes);

    for (int i=0;i<Numbodynodes;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_inertial_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}

   Eigen::Matrix3d randomized_inertial = Eigen::Matrix3d::Zero(3,3);
	for (int i=0;i<Numbodynodes;i++)
	{
		randomized_inertial = randomized_inertial_ratios(i) * Initial_inertia.block(i*3,0,3,3);  
		// Set moment of inertia defined around the center of mass.  
		mCharacter->GetSkeleton()->getBodyNode(i)->setMomentOfInertia(randomized_inertial(0,0),randomized_inertial(1,1),randomized_inertial(2,2), randomized_inertial(0,1),randomized_inertial(0,2),randomized_inertial(1,2)); 
	}
}

void 
Environment::
randomize_centerofmass(double lower_bound, double upper_bound)
{

    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
	int cnt =0;
	if (mCOMindependent)
		cnt = 3;
	else
		cnt = 1;

	Eigen::VectorXd sample = Eigen::VectorXd::Zero(cnt*Numbodynodes);
	Eigen::VectorXd randomized_com_ratios = Eigen::VectorXd::Zero(cnt*Numbodynodes);
	for (int i=0;i<cnt*Numbodynodes;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_com_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}
	Eigen::Vector3d randomized_centerofmass;
	randomized_centerofmass.setZero();
	for (int i=0;i<Numbodynodes;i++)
	{
		randomized_centerofmass = randomized_com_ratios.segment(i*cnt,cnt).cwiseProduct(Initial_centerofmass.segment(i*cnt,cnt));  
		mCharacter->GetSkeleton()->getBodyNode(i)->setLocalCOM(randomized_centerofmass); 
	}
}


void 
Environment::
randomize_motorstrength(double lower_bound, double upper_bound)
{
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
 
    Eigen::VectorXd sample= Eigen::VectorXd::Zero(mNumActiveDof);
	// randomized_strength_ratios = Eigen::VectorXd::Zero(mNumActiveDof);
    for (int i=0;i<mNumActiveDof;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_strength_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}
}



void 
Environment::
randomize_controllatency(double lower_bound, double upper_bound)
{
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
 
    double sample = 0; 
	randomized_latency = 0;

	sample = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
	randomized_latency = (sample-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
}



void 
Environment::
randomize_friction(double lower_bound, double upper_bound)
{
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
 
    double sample = 0; 
	double randomized_friction = 0;

    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	sample = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
	randomized_friction = (sample-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	for (int i=0;i<Numbodynodes;i++)
		mCharacter->GetSkeleton()->getBodyNode(i)->setFrictionCoeff(randomized_friction);
}


bool isSoftContact(const collision::Contact& contact)
{
  auto shapeNode1 = contact.collisionObject1->getShapeFrame()->asShapeNode();
  auto shapeNode2 = contact.collisionObject2->getShapeFrame()->asShapeNode();
  assert(shapeNode1);
  assert(shapeNode2);

  auto bodyNode1 = shapeNode1->getBodyNodePtr().get();
  auto bodyNode2 = shapeNode2->getBodyNodePtr().get();

  auto bodyNode1IsSoft =
      dynamic_cast<const dynamics::SoftBodyNode*>(bodyNode1) != nullptr;

  auto bodyNode2IsSoft =
      dynamic_cast<const dynamics::SoftBodyNode*>(bodyNode2) != nullptr;

  return bodyNode1IsSoft || bodyNode2IsSoft;
}


void
Environment::
Step()  
{	
	if(mUseMuscle && mUseMuscleNN)
	{
		int count = 0;
		for(auto muscle : mCharacter->GetMuscles())
		{
			muscle->activation = mActivationLevels[count++];
			muscle->Update();
			muscle->ApplyForceToBody();
		}                                                     // apply muscle force to body
		if(mSimCount == mRandomSampleIndex)
		{
			auto& skel = mCharacter->GetSkeleton();
			auto& muscles = mCharacter->GetMuscles();

			int n = skel->getNumDofs();
			int m = muscles.size();
			Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n,m);
			Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);

			for(int i=0;i<muscles.size();i++)
			{
				auto muscle = muscles[i];
				// muscle->Update();
				Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
				auto Ap = muscle->GetForceJacobianAndPassive();

				JtA.block(0,i,n,1) = Jt*Ap.first;  //matrix.block(i,j,p,q) :from matrix(i, j)，each row has p elements, each column has q elements.
				Jtp += Jt*Ap.second;
			}

			mCurrentMuscleTuple.JtA = GetMuscleTorques();
			mCurrentMuscleTuple.L = JtA.block(mRootJointDof,0,n-mRootJointDof,m);
			mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof,n-mRootJointDof);
			mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
			mMuscleTuples.push_back(mCurrentMuscleTuple);
		}
	}
	else
	{
		if(mUseMuscle && !mUseMuscleNN){
			int count = 0;
			for(auto muscle : mCharacter->GetMuscles())
			{
				muscle->activation = mActivationLevels[count++];
				muscle->Update();
				muscle->ApplyForceToBody();
			} 
		}
		bool toUpdate = false;
		for(auto force : mCharacter->GetForces())
		{
			// std::cout << "name "  << force->GetName() << std::endl;		
			if (mWorld->getTime()-lastUpdateTimeStamp >= 0.3)
			{
				force->Update();
				// std::cout << "name "  << force->GetName() << std::endl;			
				toUpdate = true; 
			}
			// update spring force in the real time 
			if(force->GetName().find("springforce") != std::string::npos)
			{
				force->Update();
			}
			force->UpdatePos();
			force->ApplyForceToBody();
		}
		if(toUpdate)
		{
			lastUpdateTimeStamp = mWorld->getTime(); 
			toUpdate = false; 
		}

		GetDesiredTorques();
		mCharacter->GetSkeleton()->setForces(mDesiredTorque);
		UpdateTorqueBuffer(mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof));
	
	}
    
	// if(mUseHuntContactForce) {
	// 	applyContactForce();
	// }

	mWorld->step();           
	mSimCount++;

}


// void
// Environment::
// applyContactForce()
// {
// 	auto mCollisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
// 	auto mCollisionOption = mWorld->getConstraintSolver()->getCollisionOption();
// 	collision::CollisionResult mCollisionResult;
	
// 	std::vector<constraint::ContactConstraintPtr>  mContactConstraints;
//   	std::vector<constraint::SoftContactConstraintPtr> mSoftContactConstraints;
// 	contact_pos_left.clear(); contact_pos_right.clear(); contact_force_left.clear(); contact_force_right.clear();

// 	mCollisionResult.clear();
// 	mCollisionGroup->collide(mCollisionOption, &mCollisionResult);
// 	// Destroy previous contact constraints
// 	mContactConstraints.clear();
// 	// Destroy previous soft contact constraints
// 	mSoftContactConstraints.clear();
// 	// Create new contact constraints
// 	for (auto i = 0; i < mCollisionResult.getNumContacts(); ++i)
// 	{
// 		// std::cout << mCollisionResult.getNumContacts() << "num" << std::endl;
// 		auto& ct = mCollisionResult.getContact(i);
// 		// Set colliding bodies
// 		auto shapeFrame1 = const_cast<dynamics::ShapeFrame*>(
// 			ct.collisionObject1->getShapeFrame());
// 		auto shapeFrame2 = const_cast<dynamics::ShapeFrame*>(
// 			ct.collisionObject2->getShapeFrame());
// 	DART_SUPPRESS_DEPRECATED_BEGIN
// 		shapeFrame1->asShapeNode()->getBodyNodePtr()->setColliding(true);
// 		shapeFrame2->asShapeNode()->getBodyNodePtr()->setColliding(true);
// 	DART_SUPPRESS_DEPRECATED_END
	
// 		if (isSoftContact(ct))
// 		{
// 		mSoftContactConstraints.push_back(
// 				std::make_shared<constraint::SoftContactConstraint>(ct, mWorld->getTimeStep()));
// 		}
// 		else
// 		{
// 		mContactConstraints.push_back(
// 				std::make_shared<constraint::ContactConstraint>(ct, mWorld->getTimeStep()));
// 		}
// 	}
// 	// Add the new contact constraints to dynamic constraint list
// 	for (const auto& contactConstraint : mContactConstraints)
// 	{
// 		//contactConstraint->update();

// 		const std::vector<collision::Contact*>& cts = contactConstraint->getContacts();  //uncomment after changing dart code
// 		//std::vector<collision::Contact*> cts;  //comment after changing dart code
// 		assert(cts.size() == 1);
// 		auto ct = const_cast<collision::Contact*>(cts[0]);
		
// 		Eigen::Vector3d v = contactConstraint->getRelVelocity(); //check direction! //uncomment after changing dart code
// 		// std::cout << "velcity:\n" << v << std::endl;
// 		//Eigen::Vector3d v;//comment after changing dart code
		
// 		double depth = ct->penetrationDepth;
// 		// std::cout << "depth:\n" << depth << std::endl;
// 		Eigen::Vector3d normal = ct->normal; //normal direction
// 		Eigen::Vector3d pos = ct->point; //contact position relative to world frame
// 		// std::cout << contactConstraint->getContactBodynode2()->getName() << std::endl;

// 		//use Hunt-Crossley(Hertz) force, F = ( k * pdep ^ n + (b pdep^n) * rvel ) * norm;  check the velocity and norm signs
// 		// std::cout << "normal:\n" << normal << std::endl;
//  		// see https://github.com/simbody/simbody/blob/master/Simbody/src/HuntCrossleyForce.cpp
// 		auto mHct2 = mHct; //assume contact between two materials, since there is only one (from input)		
// 		double s1 = mHct.stiffness/(mHct.stiffness+mHct2.stiffness);
// 		double s2 = 1-s1;
// 		Eigen::Vector3d location = pos+(depth*(double(0.5)-s1))*normal;
		
// 		// Calculate the Hertz force.
// 		double k = mHct.stiffness*s1;
// 		double c = mHct.dissipation*s1 + mHct2.dissipation*s2;	
// 		double radius = std::sqrt(mHct.radius*mHct2.radius);
// 		if (depth <=0)
// 		    depth=0;
// 		double fH = 4./3.*k*depth*std::sqrt(radius*k*depth);
// 		double pe = 2./5.*fH*depth;
// 		// std::cout << "k:\n"  << k << "depth:\n" << depth << "radius:\n" << radius << std::endl;
// 		// Calculate the relative velocity of the two bodies at the contact point.
		
// 		double vnormal = v.dot(normal);
// 		Eigen::Vector3d vtangent = v-vnormal*normal; //tangent velocity
// 		// std::cout << "vtangent:\n" << vtangent << std::endl;
// 		// Calculate the Hunt-Crossley force.
		
// 		double f = fH*(1+1.5*c*vnormal);
// 		// std::cout << "vnormal:\n" << vnormal << std::endl;
// 		// std::cout << "fH:\n" << fH  << std::endl;
// 		if (f <= 0) 
// 			return;
// 		Eigen::Vector3d  force = f*normal;
// 		// std::cout << "force without friction:\n" << force << "f:\n"  << f  << std::endl;
// 		// Calculate the friction force.	
// 		const double vslip = vtangent.norm();
// 		if (vslip != 0) {
// 			bool hasStatic = (mHct.staticFriction != 0 || mHct2.staticFriction != 0);
// 			bool hasDynamic= (mHct.dynamicFriction != 0 || mHct2.dynamicFriction != 0);
// 			bool hasViscous = (mHct.viscousFriction != 0 || mHct2.viscousFriction != 0);
// 			double us = hasStatic ? 2*mHct.staticFriction*mHct2.staticFriction/(mHct.staticFriction+mHct2.staticFriction) : 0;
// 			double ud = hasDynamic ? 2*mHct.dynamicFriction*mHct2.dynamicFriction/(mHct.dynamicFriction+mHct2.dynamicFriction) : 0;
// 			double uv = hasViscous ? 2*mHct.viscousFriction*mHct2.viscousFriction/(mHct.viscousFriction+mHct2.viscousFriction) : 0;
// 			double vrel = vslip/mHct.transitionVelocity;
// 			double ffriction = f*(std::min(vrel, double(1))*(ud+2*(us-ud)/(1+vrel*vrel))+uv*vslip);
// 			force += ffriction*vtangent/vslip;
// 		}
		
// 		//apply force to two contact bodies	
// 		// Set colliding bodies	
// 		auto shapeFrame1 = const_cast<dynamics::ShapeFrame*>(	
// 		ct->collisionObject1->getShapeFrame());	
// 		auto shapeFrame2 = const_cast<dynamics::ShapeFrame*>(	
// 		ct->collisionObject2->getShapeFrame());	
// 	DART_SUPPRESS_DEPRECATED_BEGIN	
// 		auto body1 = shapeFrame1->asShapeNode()->getBodyNodePtr();	
// 		auto body2 = shapeFrame2->asShapeNode()->getBodyNodePtr();	
// 	DART_SUPPRESS_DEPRECATED_END	
// 		if(body1->getName() == "l_foot_ground")	
// 		{	
// 			contact_pos_left.push_back(pos);	
// 			contact_force_left.push_back(force);	
// 		}	
// 		else if(body1->getName() == "r_foot_ground")	
// 		{	
// 			contact_pos_right.push_back(pos);	
// 			contact_force_right.push_back(force);	
// 		}	
//         else	
// 		{	
// 			std::cout << body1->getName() << std::endl;	
// 			std::cout << "-----Warning: contact force not on foot-------" << std::endl;	
// 		}
// 		body1->addExtForce(force,pos,false,false);  //positve force?
// 		body2->addExtForce(-force,pos,false,false); //negative?

// 	}
// 	// Add the new soft contact constraints to dynamic constraint list
// 	for (const auto& softContactConstraint : mSoftContactConstraints)
// 	{
// 		//softContactConstraint->update();
// 		//std::cout << "Does nothing with soft body contact yet!"
// 	}
// 	return; 
// }



Eigen::VectorXd clamp(Eigen::VectorXd x, double lo, double hi)
{
	for(int i=0; i<x.rows(); i++)
	{
		x[i] = (x[i] < lo) ? lo : (hi < x[i]) ? hi : x[i];
	}
	return x; 
}


void
Environment::
ProcessAction(int substep_count, int num)
{
    double lerp = double(substep_count + 1) / num;              //substep_count: the step count should be between [0, num_action_repeat).
    mAction = mPrevAction + lerp * (mCurrentAction - mPrevAction);
    // return proc_action;
}

Eigen::VectorXd
Environment::
GetDesiredTorques()
{
	mLastDesiredTorque = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof); 
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.segment(mRootJointDof,mNumActiveDof) = mAction;    // x.tail(n) = x(end-n+1 : end)
	mDesiredTorque = mCharacter->GetSPDForces(p_des); 
	// mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof) = randomized_strength_ratios.cwiseProduct(mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof));
	// mDesiredTorque.segment(mRootJointDof,mNumActiveDof) = clamp(mDesiredTorque.segment(mRootJointDof,mNumActiveDof), -100, 100);  //clamp torques
	return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}

Eigen::VectorXd
Environment::
GetTargetObservations()
{
	Eigen::Isometry3d mTc = mCharacter->GetmTc();
	int dof = mCharacter->Getskeletondof();
    double time0 = mWorld->getTime();
	double dt = 1.0/mControlHz;
	Eigen::MatrixXd tar_poses(6,dof);
	Eigen::VectorXd tar_position; 
	std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(time0,dt);
	Eigen::VectorXd root_info = mCharacter->GetSkeleton()->getPositions().head(mRootJointDof);  //get current root global position
	
	double root_x_rotation = root_info(0);
	double root_y_rotation = root_info(1);
	double root_z_rotation = root_info(2);

	mroot_relInfo << root_x_rotation, root_y_rotation,root_z_rotation;
	for(int step=0; step < 6; step++)
	{
      	double time = time0 + (step+1)* dt;
		std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(time,dt);
		tar_position = std::get<0>(pv);
		Eigen::VectorXd root_rel = root_info-tar_position.head(mRootJointDof);
		tar_position.head(mRootJointDof) = root_rel;
		tar_poses.row(step) = tar_position.head(dof);
	}
	mCharacter->SetmTc(mTc);
	tar_poses.transposeInPlace();   // transpose
	Eigen::VectorXd tar_poses_v = Eigen::Map<const Eigen::VectorXd>(tar_poses.data(), tar_poses.size());  // matrix to vector
    return tar_poses_v;
}


Eigen::VectorXd
Environment::
GetMuscleTorques()
{
	int index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for(auto muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(index,JtA_i.rows()) = JtA_i;
		index += JtA_i.rows();
	}
	
	return mCurrentMuscleTuple.JtA;
}

double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
	return exp(-w*vec.squaredNorm());   //L2 squareNorm()
}
double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
	return exp(-w*vec.squaredNorm());  ////L2 squareNorm()
}
double exp_of_squared(double val,double w)
{
	return exp(-w*val*val);
}


bool
Environment::
IsEndOfEpisode()    //
{
	bool isTerminal = false;
	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();
	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];
    Eigen::Vector3d pos_foot_r = mCharacter->GetSkeleton()->getBodyNode("r_foot")->getCOM();
	Eigen::Vector3d pos_foot_l = mCharacter->GetSkeleton()->getBodyNode("l_foot")->getCOM();
	double foot_l =  mCharacter->GetSkeleton()->getBodyNode("l_foot_ground")->getCOM()(1);
	double foot_r =  mCharacter->GetSkeleton()->getBodyNode("r_foot_ground")->getCOM()(1);
    Eigen::Vector6d root_pos = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	Eigen::Isometry3d cur_root_inv = mCharacter->GetSkeleton()->getRootBodyNode()->getWorldTransform().inverse();

	Eigen::Vector3d root_v = mCharacter->GetSkeleton()->getBodyNode(0)->getCOMLinearVelocity();
	double root_v_norm = root_v.norm();
	Eigen::Vector6d root_pos_diff = mTargetPositions.segment<6>(0) - root_pos;
	if(walk_skill==true)
	{
		if (root_y<1.2 || root_y >1.4)      //prevent falling down
			isTerminal =true;
		else if (dart::math::isNan(p) || dart::math::isNan(v))	
			isTerminal =true;	
		else if(foot_l>high_clearance && foot_r>low_clearance)	
			isTerminal =true;	
		else if(foot_r>high_clearance && foot_l>low_clearance)	
			isTerminal =true;	
		else if(mWorld->getTime()>mterminal_time)     // input 	
			isTerminal =true;	
		else if(foot_r>low_clearance && foot_l>low_clearance)	
			isTerminal =true;
	}	
	else if(squat_skill=true)
	{
		if (root_y<1.19 || root_y >1.42)            //prevent falling down
			isTerminal =true;
		else if (dart::math::isNan(p) || dart::math::isNan(v))
			isTerminal =true;
		else if(mWorld->getTime()>mterminal_time)      
			isTerminal =true;
		else if (foot_l > foot_tolerance)
			isTerminal =true;
		else if (foot_r > foot_tolerance)
			isTerminal =true;
	}
	return isTerminal;
}



// states of the skeleton model include the position, velocity and phrase variable which represents the
                                     // normalized time elapsed in the reference motion. 
Eigen::VectorXd 
Environment::   
GetState()   
{
	auto& skel = mCharacter->GetSkeleton();     
	dart::dynamics::BodyNode* root = skel->getBodyNode(0);   // get root
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	p_save = p_save.head(mCharacter->Getskeletondof());
	v_save = v_save.head(mCharacter->Getskeletondof());
    // current joint positions and velocities
	Eigen::VectorXd p_cur, v_cur;
	// remove global transform of the root
	p_cur.resize(p_save.rows()-6);
	p_cur = p_save.tail(p_save.rows()-6);
	v_cur = v_save/10.0;
	Eigen::VectorXd state(p_cur.rows()+v_cur.rows()); //+tar_poses.rows());
	state<<p_cur,v_cur; //tar_poses;
	return state;
}

void 
Environment:: 
UpdateStateBuffer()
{
	history_buffer_true_state.push_back(this->GetState());
	history_buffer_true_COP.push_back(this->GetCOPRelative());
	// store the delayed observation 
	history_buffer_control_state.push_back(this->GetControlState());
	control_COP = GetControlCOP(); 
	history_buffer_root.push_back(this->GetRootInfo());
}




Eigen::VectorXd 
Environment:: 
GetControlCOP()
{
	double dt = 1.0/mControlHz;
	Eigen::VectorXd _control_COP; 
	if((randomized_latency <= 0) || (history_buffer_true_COP.size() == 1))
	{
		_control_COP = history_buffer_true_COP.get(HISTORY_BUFFER_LEN-1);
	}
	else
	{
		int n_steps_ago = int(randomized_latency / dt);
		if(n_steps_ago + 1 >= history_buffer_true_COP.size())
		{
			_control_COP = history_buffer_true_COP.get(HISTORY_BUFFER_LEN-1);
		}
		else
		{
			double remaining_latency = randomized_latency - n_steps_ago * dt; 
			double blend_alpha = remaining_latency / dt; 
			_control_COP = (
				(1.0 - blend_alpha) * history_buffer_true_COP.get(HISTORY_BUFFER_LEN - n_steps_ago - 1)
				+ blend_alpha * history_buffer_true_COP.get(HISTORY_BUFFER_LEN - n_steps_ago - 2)); 
			if(dart::math::isNan(_control_COP)){
				std::cout << "_control_COP:/n"<<_control_COP << std::endl; 
				std::cout << "n_steps_ago:  " << n_steps_ago << std::endl; 
				std::cout << "blend_alpha:  " << blend_alpha << std::endl; 
			}
		}
	}


    return _control_COP; 
}

Eigen::VectorXd 
Environment:: 
GetControlState()
{
	double dt = 1.0/mControlHz;
	Eigen::VectorXd observation; 
	if((randomized_latency <= 0) || (history_buffer_true_state.size() == 1)){
    	observation = history_buffer_true_state.get(STATE_HISTORY_BUFFER_LEN-1);
	}else{
		int n_steps_ago = int(randomized_latency / dt);
		if(n_steps_ago + 1 >= history_buffer_true_state.size()){
			observation = history_buffer_true_state.get(STATE_HISTORY_BUFFER_LEN-1);
		}else{
			double remaining_latency = randomized_latency - n_steps_ago * dt; 
			double blend_alpha = remaining_latency / dt; 
			observation = (
				(1.0 - blend_alpha) * history_buffer_true_state.get(STATE_HISTORY_BUFFER_LEN - n_steps_ago - 1)
				+ blend_alpha * history_buffer_true_state.get(STATE_HISTORY_BUFFER_LEN - n_steps_ago - 2)); 
		}
	}

    return observation; 

}


void 
Environment:: 
UpdateActionBuffer(Eigen::VectorXd action)
{
	history_buffer_action.push_back(action); 
}


void 
Environment:: 
UpdateTorqueBuffer(Eigen::VectorXd torque)
{
	history_buffer_torque.push_back(torque); 
}


Eigen::VectorXd 
Environment::  
GetCOPRelative()
{
	//use environment mEnv to calculate COP_error
	Eigen::Vector3d pos_foot_r = mCharacter->GetSkeleton()->getBodyNode("r_foot")->getCOM();
	Eigen::Vector3d pos_foot_l = mCharacter->GetSkeleton()->getBodyNode("l_foot")->getCOM();
	pos_foot_l(1) = -0.895985;
	pos_foot_r(1) = -0.895985;
	Eigen::Vector3d COP_target_left = pos_foot_l;
	Eigen::Vector3d COP_target_right = pos_foot_r;

	auto& results = mWorld->getConstraintSolver()->getLastCollisionResult();
    std::vector<constraint::ContactConstraintPtr> mContactConstraints;

	double COP_Y_fixed_left = COP_target_left(1);
	double COP_Y_fixed_right = COP_target_right(1);


	std::vector<Eigen::Vector3d> all_pos_left;
	std::vector<Eigen::Vector3d> all_pos_right;
	std::vector<Eigen::Vector3d> all_force_left;
	std::vector<Eigen::Vector3d> all_force_right;

    Eigen::Vector3d COP_left, COP_right; 

	for(int i = 0; i < results.getNumContacts(); ++i)   // store all contact forces 
	{
		auto& contact = results.getContact(i);
		mContactConstraints.clear();
		mContactConstraints.push_back(
				std::make_shared<constraint::ContactConstraint>(contact, mWorld->getTimeStep()));
		auto pos = contact.point;
		auto force = contact.force;
		// all_pos.push_back(pos);
		// all_force.push_back(force);
		auto shapeFrame1 = const_cast<dynamics::ShapeFrame*>(
			contact.collisionObject1->getShapeFrame());
		auto shapeFrame2 = const_cast<dynamics::ShapeFrame*>(
			contact.collisionObject2->getShapeFrame());
	DART_SUPPRESS_DEPRECATED_BEGIN
		auto body1 = shapeFrame1->asShapeNode()->getBodyNodePtr();
		auto body2 = shapeFrame2->asShapeNode()->getBodyNodePtr();
	DART_SUPPRESS_DEPRECATED_END

		
		for (auto& contactConstraint : mContactConstraints)
		{
			if(body1->getName() == "l_foot_ground")
			{
				all_pos_left.push_back(pos);
				all_force_left.push_back(force);
			}
			else if(body1->getName() == "r_foot_ground"){
				all_pos_right.push_back(pos);
				all_force_right.push_back(force);
			}
			else
			{
				std::cout << body1->getName() << std::endl;
				std::cout << "-----Warning: contact force not on foot-------" << std::endl;
			}
		}
	}

	Eigen::Vector3d p_cross_f_left;
	double f_sum_left = 0; 
	p_cross_f_left.setZero();
	Eigen::Vector3d p;

	Eigen::Vector3d unitV;
	unitV << 0, 1, 0;    // unit normal vector  

	for(int i=0; i<all_pos_left.size(); i++){
		p = all_pos_left[i];
		double f_scalar_left = all_force_left[i].dot(unitV);
		f_sum_left += f_scalar_left; 
		p_cross_f_left += p.cross(f_scalar_left * unitV);
	}
	if (f_sum_left==0)
		COP_left.setZero();
	else
	{
		COP_left = -p_cross_f_left.cross(unitV)/f_sum_left;
		COP_left(1) = COP_target_left(1);
	}
   
	//
	Eigen::Vector3d p_cross_f_right;
	double f_sum_right = 0; 
	p_cross_f_right.setZero();
	for(int i=0; i<all_pos_right.size(); i++){
		p = all_pos_right[i];
		double f_scalar_right = all_force_right[i].dot(unitV);
		f_sum_right += f_scalar_right; 
		p_cross_f_right += p.cross(f_scalar_right * unitV);
	}
    if (f_sum_right==0)
		COP_right.setZero();
	else
	{
		COP_right = -p_cross_f_right.cross(unitV)/f_sum_right;
		COP_right(1) = COP_target_right(1);
	}

	Eigen::Vector3d COP_left_rel = COP_target_left - COP_left;
	Eigen::Vector3d COP_right_rel = COP_target_right - COP_right;
	Eigen::VectorXd COP_rel(COP_left_rel.rows()+COP_right_rel.rows()); 
    COP_rel << 0.2*COP_left_rel, 0.2*COP_right_rel;
    return COP_rel;

}

Eigen::VectorXd 
Environment::  
GetFullObservation()
{
	Eigen::VectorXd tar_poses = this->GetTargetObservations();

	// --------- flatten state history 
	Eigen::MatrixXd states(mNumState, STATE_HISTORY_BUFFER_LEN);
	Eigen::MatrixXd rootInfo(3, STATE_HISTORY_BUFFER_LEN-1);
	for(int i=0; i<STATE_HISTORY_BUFFER_LEN; i++)
	{
		states.col(i) =  history_buffer_control_state.get(i);
	}
	for(int i=0; i<STATE_HISTORY_BUFFER_LEN-1; i++)
	{
		rootInfo.col(i) =  history_buffer_root.get(i);
	}
    
	// a.transposeInPlace();
	// matrix to vector
	Eigen::VectorXd states_v = Eigen::Map<const Eigen::VectorXd>(states.data(), states.size());
	Eigen::VectorXd rootInfo_v = Eigen::Map<const Eigen::VectorXd>(rootInfo.data(), rootInfo.size());
	// --------- flatten action history 
	Eigen::MatrixXd actions(mNumActiveDof, HISTORY_BUFFER_LEN);
	for(int i=0; i<HISTORY_BUFFER_LEN; i++)
		actions.col(i) =  history_buffer_action.get(i);
	// matrix to vector
	Eigen::VectorXd actions_v = Eigen::Map<const Eigen::VectorXd>(actions.data(), actions.size());

    Eigen::VectorXd COP_rel = this->GetCOPRelative();
	Eigen::VectorXd observation(rootInfo_v.rows()+states_v.rows() + actions_v.rows()+tar_poses.rows());
	observation << rootInfo_v,states_v, actions_v,tar_poses;
	if(!mUseCOP)
	{
		Eigen::VectorXd observation(rootInfo_v.rows()+states_v.rows() + actions_v.rows()+tar_poses.rows());
		observation << rootInfo_v,states_v, actions_v,tar_poses;
	}

	return observation;
}




void 
Environment::
SetAction(const Eigen::VectorXd& a)           // execute the env.SecAction() in the GenerateTransitions process
{
	mPrevAction = mCurrentAction; 
	mCurrentAction = a*1; 
	double t = mWorld->getTime();
	std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = std::get<0>(pv);
	mTargetVelocities = std::get<1>(pv);
	mTargetEE_pos = std::get<2>(pv);
	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
	mAverageActivationLevels.setZero();
}

double
Environment::
GetReward()
{
	auto& skel = mCharacter->GetSkeleton();
    mlast_joint_vel = mcur_joint_vel; 
	mcur_joint_vel = skel->getVelocities().tail(mCharacter->Getskeletondof()-mRootJointDof);
	Eigen::VectorXd cur_pos = skel->getPositions();
	Eigen::VectorXd cur_vel = skel->getVelocities();
	Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
	               //Return the difference of two generalized positions which are measured in the configuration space of this Skeleton.
                   //If the configuration space is Euclidean space, this function returns mTargetPositions - cur_pos. 
	Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);

	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
    
	const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();
   	std::map<std::string, Eigen::Vector3d> mEEOffsetMap = mCharacter->GetBVH()->GetEEOffsetMap();
	
    auto ees = mCharacter->GetEndEffectors();

	auto ees_offset = mCharacter->GetEndEffectorsOffset();

	Eigen::VectorXd ee_diff_new(ees.size()*3);
	Eigen::VectorXd foot_angle_d(ees.size()*1);
    int cnt = 0; 
    BodyNode* root = mCharacter->GetSkeleton()->getRootBodyNode();
	
    // calculate position error 
    for(auto ss : mCharacter->GetBVH()->BVHNodeOffset)
	{   
		if(ss.first == root->getName())
		{
			Joint* jn = mCharacter->GetSkeleton()->getJoint("rootJoint");
			int idx = jn->getIndexInSkeleton(0);
			// p_diff.segment<6>(idx) = p_diff_all.segment<6>(idx);
			// v_diff.segment<6>(idx) = v_diff_all.segment<6>(idx);
			continue;
		}

		Joint* jn = mCharacter->GetSkeleton()->getJoint(ss.first);
		if(jn == NULL)
		{
			continue;
		}			
		int idx = jn->getIndexInSkeleton(0);                
		if(jn->getType()=="RevoluteJoint")
		{
			if (jn->getName()=="exo_knee_r" || jn->getName()=="exo_knee_l"){
				p_diff[idx] = w_knee*p_diff_all[idx];
				v_diff[idx] = w_knee*v_diff_all[idx];
				}
			else if (jn->getName()=="exo_hip_r" || jn->getName()=="exo_hip_l"){
				p_diff[idx] = w_hip*p_diff_all[idx];
				v_diff[idx] = w_hip*v_diff_all[idx];
				}
			else if (jn->getName()=="r_ankle_coupling" || jn->getName()=="l_ankle_coupling"){
				p_diff[idx] = w_ankle*p_diff_all[idx];
				v_diff[idx] = w_ankle*v_diff_all[idx];
				}
		    else{
				p_diff[idx] = p_diff_all[idx];
				v_diff[idx] = v_diff_all[idx];
			}

		}
	}

	Eigen::VectorXd ee_diff(ees.size()*3);
	std::vector<Eigen::Isometry3d> ee_transforms;
	Eigen::VectorXd com_diff;
    Eigen::Vector3d root_rotation = p_diff_all.segment<3>(0);
	Eigen::Vector2d root_XZ_position;
	root_XZ_position << p_diff_all[3],p_diff_all[5];
	Eigen::VectorXd root_diff = Eigen::VectorXd::Zero(5);
	root_diff <<root_rotation,root_XZ_position;
	
	// calucuate end_effector error
	for(int i =0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();  
	// for(int i = 0; i < ees.size(); i++){
	// 	ee_transforms.emplace_back(ees[i]->getWorldTransform());
	// }

	com_diff = skel->getCOM();

	skel->setPositions(const_index, mTargetPositions.head(mCharacter->Getskeletondof()));
	skel->computeForwardKinematics(true,false,false);

	com_diff -= skel->getCOM();                     // the error of the skeletion's COM (between the current and target)
	for(int i=0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) -= ees[i]-> getCOM();  
        
	// for(int i = 0; i < ees.size(); i++){
	// 	Eigen::Isometry3d diff = ee_transforms[i].inverse()*ees[i]->getWorldTransform();
	// 	ee_diff.segment<3>(3*i) = diff.translation();
	// }

	skel->setPositions(const_index, cur_pos.head(mCharacter->Getskeletondof()));
	skel->computeForwardKinematics(true,false,false);

	//  COP_reward and Zero_COM_momentum reward from RewardFactory
	double r_COP_ZCM_torque = 0;
	for(auto reward: mReward){
		r_COP_ZCM_torque += reward.second->GetReward()*reward.second->GetWeight();  
		// std::cout << reward.first << "   Reward Weight:   " << reward.second->GetWeight() << std::endl;     
	}
    Eigen::VectorXd pos_diff = p_diff.head(mCharacter->Getskeletondof());
	Eigen::VectorXd vel_diff = v_diff.head(mCharacter->Getskeletondof());

	ee_diff[2]=0;
	ee_diff[5]=0;
	double r_q = exp_of_squared(p_diff,2.0);
	double r_v = exp_of_squared(v_diff,1.0);
	double r_ee = exp_of_squared(ee_diff,15.0);
	double r_com = exp_of_squared(com_diff,0.2);
	double distance = root_XZ_position.squaredNorm();

	double r_root = exp_of_squared(root_diff,10);
    
    // Eigen::VectorXd pelvis_rot_vel = cur_vel.head(3);
    // smooth action
    Eigen::VectorXd action_diff = (history_buffer_action.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_action.get(HISTORY_BUFFER_LEN-2)+history_buffer_action.get(HISTORY_BUFFER_LEN-3)); 
    double r_action_smooth = exp_of_squared(action_diff,10.0);
    Eigen::VectorXd pelvis_acc = mCharacter->GetSkeleton()->getJoint("rootJoint")->getAccelerations(); 
	double r_pelvis_acc = exp(-10*(root_rotation.squaredNorm()+pelvis_acc.squaredNorm())); 
	Eigen::VectorXd torque_diff = (history_buffer_torque.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_torque.get(HISTORY_BUFFER_LEN-2)+history_buffer_torque.get(HISTORY_BUFFER_LEN-3)); 
	double r_torque_smooth = exp_of_squared(torque_diff,15.0);
	// Eigen::VectorXd joint_vel_diff = mcur_joint_vel.tail(mCharacter->Getskeletondof()-mRootJointDof)-mlast_joint_vel.tail(mCharacter->Getskeletondof()-mRootJointDof);
    // double r_joint_vel_smooth = exp_of_squared(joint_vel_diff,15.0);

	Eigen::VectorXd foot_ground_rot_l = mCharacter->GetSkeleton()->getBodyNode("l_foot_ground")->getTransform().linear().eulerAngles(2, 1, 0);
	Eigen::VectorXd foot_ground_rot_r = mCharacter->GetSkeleton()->getBodyNode("r_foot_ground")->getTransform().linear().eulerAngles(2, 1, 0);
	Eigen::VectorXd foot_rot_XZ = Eigen::VectorXd::Zero(4);
	foot_rot_XZ << sin(foot_ground_rot_l(0)),sin(foot_ground_rot_l(2)),sin(foot_ground_rot_r(0)),sin(foot_ground_rot_r(2));
	double r_footclearance = exp_of_squared(foot_rot_XZ,40.0);

    double r_smooth = w_sroot*r_pelvis_acc + w_saction*r_action_smooth + w_storque*r_torque_smooth; 

	double r = w_q*r_q*r_ee + w_root*r_root + r_smooth +r_COP_ZCM_torque + W_footclr*r_footclearance;

	return r; 

}


std::tuple<double,double,double,double,double,double,double,Eigen::VectorXd,double,double,double,double,double,double,double,double,double,double>
Environment::
GetRenderReward_Error()
{
	auto& skel = mCharacter->GetSkeleton();
	Eigen::VectorXd cur_pos = skel->getPositions();
	Eigen::VectorXd cur_vel = skel->getVelocities();

	Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
	               //Return the difference of two generalized positions which are measured in the configuration space of this Skeleton.
                   //If the configuration space is Euclidean space, this function returns mTargetPositions - cur_pos. 
	Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);
	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());

    int idx_hip = mCharacter->GetSkeleton()->getJoint("exo_hip_l")->getIndexInSkeleton(0);
    int idx_knee = mCharacter->GetSkeleton()->getJoint("exo_knee_l")->getIndexInSkeleton(0);
	int idx_ankle = mCharacter->GetSkeleton()->getJoint("l_ankle_coupling")->getIndexInSkeleton(0);
	int idx_foot;
	if(walk_skill)
	{
		int idx_foot = mCharacter->GetSkeleton()->getJoint("l_foot")->getIndexInSkeleton(0);
	}

	double p_tar_hip_l = mTargetPositions[idx_hip];
    double p_tar_knee_l = mTargetPositions[idx_knee];
	double p_tar_ankle_l = mTargetPositions[idx_ankle];
    double p_tar_foot_l=0;
	double p_cur_foot_l=0;
	if(walk_skill)
	{
		p_tar_foot_l = mTargetPositions[idx_foot];
		p_cur_foot_l = cur_pos[idx_foot];
	}
	double p_cur_hip_l = cur_pos[idx_hip];
	double p_cur_knee_l = cur_pos[idx_knee];
	double p_cur_ankle_l = cur_pos[idx_ankle];



	const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();
   	std::map<std::string, Eigen::Vector3d> mEEOffsetMap = mCharacter->GetBVH()->GetEEOffsetMap();
	
    auto ees = mCharacter->GetEndEffectors();

	auto ees_offset = mCharacter->GetEndEffectorsOffset();

	Eigen::VectorXd ee_diff_new(ees.size()*3);
	Eigen::VectorXd foot_angle_d(ees.size()*1);
    int cnt = 0; 
    BodyNode* root = mCharacter->GetSkeleton()->getRootBodyNode();
	
    // calculate position error 
    for(auto ss : mCharacter->GetBVH()->BVHNodeOffset)
	{   
		if(ss.first == root->getName())
		{
			Joint* jn = mCharacter->GetSkeleton()->getJoint("rootJoint");
			int idx = jn->getIndexInSkeleton(0);

			// p_diff.segment<6>(idx) = p_diff_all.segment<6>(idx);
			// v_diff.segment<6>(idx) = v_diff_all.segment<6>(idx);
			continue;
		}

		Joint* jn = mCharacter->GetSkeleton()->getJoint(ss.first);
		if(jn == NULL)
		{
			continue;
		}			
		int idx = jn->getIndexInSkeleton(0);                
		if(jn->getType()=="RevoluteJoint")
		{
			if (jn->getName()=="exo_knee_r" || jn->getName()=="exo_knee_l"){
				p_diff[idx] = w_knee*p_diff_all[idx];
				v_diff[idx] = w_knee*v_diff_all[idx];
				}
			else if (jn->getName()=="exo_hip_r" || jn->getName()=="exo_hip_l"){
				p_diff[idx] = w_hip*p_diff_all[idx];
				v_diff[idx] = w_hip*v_diff_all[idx];
				}
			else if (jn->getName()=="r_ankle_coupling" || jn->getName()=="l_ankle_coupling"){
				p_diff[idx] = w_ankle*p_diff_all[idx];
				v_diff[idx] = w_ankle*v_diff_all[idx];
				}
		   	else{
				p_diff[idx] = p_diff_all[idx];
				v_diff[idx] = v_diff_all[idx];
			}
		}

	}

	Eigen::VectorXd ee_diff(ees.size()*3);
	std::vector<Eigen::Isometry3d> ee_transforms;
	Eigen::VectorXd com_diff;
    Eigen::Vector3d root_rotation = p_diff_all.segment<3>(0);
	Eigen::Vector2d root_XZ_position;
	root_XZ_position << p_diff_all[3],p_diff_all[5];
	Eigen::VectorXd root_diff = Eigen::VectorXd::Zero(3);
	root_diff <<root_rotation;
    
	// calucuate end_effector error
	for(int i =0;i<ees.size();i++)
	{
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();  
		
	}
	com_diff = skel->getCOM();

	skel->setPositions(const_index, mTargetPositions.head(mCharacter->Getskeletondof()));
	skel->computeForwardKinematics(true,false,false);

	com_diff -= skel->getCOM();                     // the error of the skeletion's COM (between the current and target)
	

	for(int i=0;i<ees.size();i++)
	{
		ee_diff.segment<3>(i*3) -= ees[i]-> getCOM();  
		// std::cout << "ee_diff:\n" <<  ee_diff.segment<3>(i*3) << std::endl;
	}

	skel->setPositions(const_index, cur_pos.head(mCharacter->Getskeletondof()));
	skel->computeForwardKinematics(true,false,false);

	//  COP_reward and Zero_COM_momentum reward from RewardFactory
	double r_COP_ZCM_torque = 0;
	// for(auto reward: mReward){
	// 	r_COP_ZCM_torque += reward.second->GetReward()*reward.second->GetWeight();  
	// 	// std::cout << reward.first << "   Reward Weight:   " << reward.second->GetWeight() << std::endl;     
	// }

    double cop_left_reward = mReward["left_cop"]->GetReward();
    double cop_right_reward  = mReward["right_cop"]->GetReward();
	double torque_reward  = mReward["mimum_torque"]->GetReward();
	double ZCM_reward = mReward["zero_com_momentum"]->GetReward();

    Eigen::VectorXd pos_diff = p_diff.head(mCharacter->Getskeletondof());
	Eigen::VectorXd vel_diff = v_diff.head(mCharacter->Getskeletondof());

	ee_diff[2]=0;
	ee_diff[5]=0;
	double r_q = exp_of_squared(p_diff,2.0);
	double r_v = exp_of_squared(v_diff,1.0);
	double r_ee = exp_of_squared(ee_diff,10.0);
	double r_com = exp_of_squared(com_diff,0.2);
	double distance = root_XZ_position.squaredNorm();

	double r_root = exp_of_squared(root_diff,10);

    double pos_error = pos_diff.squaredNorm();
	double ee_error =  ee_diff.squaredNorm();
    double cop_left_error = log(exp(1/cop_left_reward))/40;
	double cop_right_error = log(exp(1/cop_right_reward))/40;
    Eigen::VectorXd torque = this->GetDesiredTorques();


	Eigen::VectorXd foot_ground_rot_l = mCharacter->GetSkeleton()->getBodyNode("l_foot_ground")->getTransform().linear().eulerAngles(2, 1, 0);
	Eigen::VectorXd foot_ground_rot_r = mCharacter->GetSkeleton()->getBodyNode("r_foot_ground")->getTransform().linear().eulerAngles(2, 1, 0);
	Eigen::VectorXd foot_rot_XZ = Eigen::VectorXd::Zero(4);
	foot_rot_XZ << sin(foot_ground_rot_l(0)),sin(foot_ground_rot_l(2)),sin(foot_ground_rot_r(0)),sin(foot_ground_rot_r(2));

	double r_fcl = exp_of_squared(foot_rot_XZ,40.0);     // foot clearance reward: try to make the foot of the robot parallel to the ground
    // std::cout << "r_fcl:  " << r_fcl << std::endl;


	return std::make_tuple(r_q, r_v, r_ee, r_root, r_fcl, cop_left_reward, cop_right_reward, torque, p_tar_hip_l, p_tar_knee_l, p_tar_ankle_l, p_tar_foot_l, p_cur_hip_l, p_cur_knee_l, p_cur_ankle_l, p_cur_foot_l, cop_left_error, cop_right_error);


}
