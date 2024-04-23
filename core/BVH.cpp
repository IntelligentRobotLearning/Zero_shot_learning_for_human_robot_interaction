#include "BVH.h"
#include <iostream>
#include <algorithm>
#include <Eigen/Geometry>
#include "dart/dart.hpp"
using namespace dart::dynamics;
namespace MASS
{
Eigen::Matrix3d
R_x(double x)
{
	double cosa = cos(x*3.141592/180.0);
	double sina = sin(x*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	1,0		,0	  ,
		0,cosa	,-sina,
		0,sina	,cosa ;
	return R;
}
Eigen::Matrix3d R_y(double y)
{
	double cosa = cos(y*3.141592/180.0);
	double sina = sin(y*3.141592/180.0);
	Eigen::Matrix3d R;
	R <<cosa ,0,sina,
		0    ,1,   0,
		-sina,0,cosa;
	return R;	
}
Eigen::Matrix3d R_z(double z)
{
	double cosa = cos(z*3.141592/180.0);
	double sina = sin(z*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	cosa,-sina,0,
		sina,cosa ,0,
		0   ,0    ,1;
	return R;		
}
BVHNode::
BVHNode(const std::string& name,BVHNode* parent)
	:mParent(parent),mName(name),mChannelOffset(0),mNumChannels(0)
{

}

// offset: the translation (x,y,z) from parent to current node.
void 
BVHNode::
SetOffset(double x, double y, double z)
{
	this->offset[0] = x;
	this->offset[1] = y;
	this->offset[2] = z;
}

void
BVHNode::                                                              //
SetChannel(int c_offset,std::vector<std::string>& c_name)
{
	mChannelOffset = c_offset;
	mNumChannels = c_name.size();
	for(const auto& cn : c_name)                 // c_name = [Xposition Yposition Zposition Zrotation Xrotation Yrotation]
		mChannel.push_back(CHANNEL_NAME[cn]);    // mChannel store  [Xpos  Ypos  Zpos  Zrot  Xrot  Yrot]
}                                     


void
BVHNode::
Set(const Eigen::VectorXd& m_t)
{
	mR.setIdentity();
	for(int i=0;i<mNumChannels;i++)
	{
		switch(mChannel[i])
		{
		case Xpos:break;
		case Ypos:break;
		case Zpos:break;
		case Xrot:mR = mR*R_x(m_t[mChannelOffset+i]);break;
		case Yrot:mR = mR*R_y(m_t[mChannelOffset+i]);break;
		case Zrot:mR = mR*R_z(m_t[mChannelOffset+i]);break;
		default:break;
		}
	}
}

void
BVHNode::
SetEE(const Eigen::VectorXd& m_t)
{
	mR_EE.setIdentity();
	for(int i=0;i<mNumChannels;i++)
	{
		switch(mChannel[i])
		{
		case Xpos:break;
		case Ypos:break;
		case Zpos:break;
		case Xrot:mR_EE = mR_EE*R_x(m_t[mChannelOffset+i]);break;
		case Yrot:mR_EE = mR_EE*R_y(m_t[mChannelOffset+i]);break;
		case Zrot:mR_EE = mR_EE*R_z(m_t[mChannelOffset+i]);break;
		default:break;
		}
	}
}


void 
BVH::
SetGlobal(BVHNode* bn, std::map<std::string,Eigen::Vector3d>& mEEPosMap)
{	
   	if(bn->isEndEffector)
	{
		//mEEPosMap.insert(std::make_pair(bn->GetName(), bn->T_Global.linear()*mEEOffsetMap[bn->GetName()]+bn->T_Global.translation()));
		mEEPosMap.insert(std::make_pair(bn->GetName(), bn->T_Global*mBVHEEOffsetMap[bn->GetName()]));
	}
	std::vector<BVHNode*> ss = bn->GetChildren(); 
	for(auto& cbn : ss){

		Eigen::Isometry3d local; 
		local.linear() = cbn->GetEE();
		if(cbn->isEndEffector)
		{
			local.translation() = unit_scale*cbn->GetOffset();
			local.translation()[0] = local.translation()[0]/2.0;
		}
        else
			local.translation() = unit_scale*cbn->GetOffset();
		// global T of parent * local T 
		cbn->T_Global = bn->T_Global * local;
		SetGlobal(cbn, mEEPosMap);
	}
}

void
BVHNode::
SetEE_Matrix(const Eigen::Matrix3d& R_t)
{
	mR_EE = R_t;
}

void
BVHNode::
Set(const Eigen::Matrix3d& R_t)
{
	mR = R_t;
}
Eigen::Matrix3d
BVHNode::
Get()
{
	return mR;
}


void
BVHNode::
AddChild(BVHNode* child)
{
	mChildren.push_back(child);
}
BVHNode*
BVHNode::
GetNode(const std::string& name)
{
	if(!mName.compare(name))
		return this;

	for(auto& c : mChildren)
	{
		BVHNode* bn = c->GetNode(name);
		if(bn!=nullptr)
			return bn;
	}

	return nullptr;
}


BVH::
BVH(const dart::dynamics::SkeletonPtr& skel,const std::map<std::string,std::string>& bvh_map ,const std::vector<std::string>& ee_names, const std::map<std::string, Eigen::Vector3d>& EndEffectorsOffset, const std::map<std::string, Eigen::Vector3d>& BVHEEOffset)
	:mSkeleton(skel),mBVHMap(bvh_map),mBVHEE(ee_names),mEEOffsetMap(EndEffectorsOffset), mBVHEEOffsetMap(BVHEEOffset), mCyclic(true)
{
   walk_skill=false;
   squat_skill=false;
}


std::tuple<Eigen::VectorXd, BVHNode*>
BVH::
GetMotion(double t)
{   

	int k = ((int)std::floor(t/mTimeStep));   // std::floor -->向下取整
	if(mCyclic)
		k %= mNumTotalFrames;
	k = std::max(0,std::min(k,mNumTotalFrames-1)); //
	double dt = t/mTimeStep - std::floor(t/mTimeStep);
	Eigen::VectorXd m_t = mMotions[k];          // m_t is the each row of motion data (correspond to the BVH motion data)
	if(walk_skill==true)
		m_t[2]=0;


	// for(auto& bn: mMap)
	// {
	// 	bn.second->Set(m_t);                
	// 	bn.second->SetEE(m_t);
	// }
    
	Eigen::Matrix3d root_mr; 
	root_mr.setIdentity();
	
	BodyNode* root = mSkeleton->getRootBodyNode();
	// std::string root_bvh_name = mBVHMap[root->getName()]; 


	// and calculate the ee position target 	
	// std::map<std::string,Eigen::Vector3d> mEEPosMap;

	int dof = mSkeleton->getNumDofs();
	Eigen::VectorXd p = Eigen::VectorXd::Zero(dof);



	for(auto ss : BVHNodeOffset)
	{   
		// root joint is free joint 

		if(ss.first == root->getName())
		{
			Eigen::Isometry3d T;                          ////T: 4＊4
			Eigen::Vector3d _p = Eigen::VectorXd::Zero(3);

			if(walk_skill==true)
			{
				_p.segment<3>(0) = unit_scale*m_t.segment<3>(0);
				T.translation() = _p; 
				Eigen::Quaterniond Q(m_t.segment<4>(3));
				Eigen::Matrix3d rotationMatrix = Q.toRotationMatrix();
				T.linear() = rotationMatrix;
				T.linear().setIdentity();
			}
			else if(squat_skill==true)
			{
				_p.segment<2>(0) = unit_scale*m_t.segment<2>(0);
				T.translation() = _p; 
				T.linear() = R_z(m_t[2]);
			}
			Joint* jn = mSkeleton->getJoint("rootJoint");
			int idx = jn->getIndexInSkeleton(0);
			p.segment<6>(idx) = FreeJoint::convertToPositions(T); 
			continue;  

		}

		Joint* jn = mSkeleton->getJoint(ss.first);
		if(jn == NULL)
		{
			continue;
		}			
		
		// this->Get(ss.second);            					// get the rotation matrix R based on .bvh file motion
		//Joint* jn = bn->getParentJoint();                    //Return the parent Joint of this BodyNode
		int idx = jn->getIndexInSkeleton(0);                 //Return the index of this BodyNode within its Skeleton.
		std::string s = jn->getType();

		if(jn->getType()=="BallJoint"){
			// p.segment<3>(idx) = BallJoint::convertToPositions(R);  //Convert a transform into a 3D vector that can be used to set the positions of a Balljoint.
		}
		else if(jn->getType()=="RevoluteJoint")
		{
			Eigen::Vector3d u =dynamic_cast<RevoluteJoint*>(jn)->getAxis();  // get revolute axis of the joint from the model 
			// Eigen::Vector3d aa = BallJoint::convertToPositions(R);  
			double val = m_t[ss.second]/180 * M_PI;
			if(walk_skill==true)
            	val = m_t[ss.second];
			if(val>M_PI)
			{
				val -= 2*M_PI;
			}
			else if(val<-M_PI){
				val += 2*M_PI;
			}
			p[idx] = val;
		}
	}
	//calculate the gobal T of bvh root
	// mRoot->T_Global.translation() =unit_scale*m_t.segment<3>(0);  //+unit_scale*mRoot->GetOffset()
	// mRoot->T_Global.linear() = R_y(90) * mMap[root_bvh_name]->GetEE(); 
	return std::make_tuple(p, mRoot);
}

Eigen::Matrix3d
BVH::
Get(const std::string& bvh_node)
{
	return mMap[bvh_node]->Get();
}
void
BVH::
Parse(const std::string& file,bool cyclic)
{
	mCyclic = cyclic;
	std::ifstream is(file);

	char buffer[256];
	if(!is)
	{
		std::cout<<"Can't open file " << file <<std::endl;
		return;
	}

	if(walk_skill==true)
		mTimeStep =	0.01;
	else if(squat_skill==true)
		mTimeStep = 0.0100251; 

	while(is>>buffer)
	{
		if(!strcmp(buffer,"datarows"))
			break;
	} 
	
	is>>buffer; //num_frames   //400
	mNumTotalFrames = atoi(buffer);   //integer
	is>>buffer;
	is>>buffer; 
	mNumTotalChannels = atoi(buffer) - 1;  //integer
	is.getline(buffer,100); 
	is.getline(buffer,100);  // range 0  2
	is.getline(buffer,100);  //endheader
	
	int idx=0;
	BVHNodeOffset[mSkeleton->getRootBodyNode()->getName()] = 0; 
	if(walk_skill==true)
	{
		is>>buffer;
		is>>buffer;
		is>>buffer;
		is>>buffer;
		is>>buffer;
		is>>buffer;
		is>>buffer;
		is>>buffer;
		idx = 7;  
	}
	else if(squat_skill==true)
	{
		is>>buffer;
		is>>buffer;
		is>>buffer;
		is>>buffer;
		idx = 3;  
	}
	while(idx<mNumTotalChannels)
	{
		is>>buffer;     
		std::string s(buffer); 
		BVHNodeOffset[s] = idx; 
		idx += 1;
	}
	// for(auto ss: BVHNodeOffset)
	// 	std::cout << ss.first << ":   " << ss.second << std::endl; 

	is.getline(buffer,100);  // time
	mMotions.resize(mNumTotalFrames);  //mMotions store all the joint information in each frame
	for(auto& m_t : mMotions)
		m_t = Eigen::VectorXd::Zero(mNumTotalChannels);

	double val;
	for(int i=0;i<mNumTotalFrames;i++)
	{		
		is >> val; 		// discard the time 
		for(int j=0;j<mNumTotalChannels;j++)
		{
			is>>val;
			mMotions[i][j]=val;
		}
	}
    std::cout << "total frames in target motion file:   " << mNumTotalFrames << std::endl; 

	// while(is>>buffer)
	// {
	// 	if(!strcmp(buffer,"Units"))     
	// 	{
	// 		is>>buffer;//are    
	// 		is>>buffer;//S.I. 
	// 		is>>buffer;//units   
	// 		is>>buffer;//S.I. 
	// 		is>>buffer;//are    
	// 		is>>buffer;//S.I. 
	// 		int c_offset = 0;
	// 		mRoot = ReadHierarchy(nullptr,buffer,c_offset,is);
	// 		mNumTotalChannels = c_offset;
	// 	}
                                                     //read bvh

		// if(!strcmp(buffer,"HIERARCHY"))  // compare buffer and Hierarchy
		// {
		// 	is>>buffer;//Root    
		// 	is>>buffer;//Name          // Character1_Hips
		// 	int c_offset = 0;
		// 	mRoot = ReadHierarchy(nullptr,buffer,c_offset,is);
		// 	mNumTotalChannels = c_offset;
		// }
		// else if(!strcmp(buffer,"MOTION"))
		// {
		// 	is>>buffer; //Frames:   
		// 	is>>buffer; //num_frames   //132
		// 	mNumTotalFrames = atoi(buffer);   //integer
		// 	is>>buffer; //Frame
		// 	is>>buffer; //Time:
		// 	is>>buffer; //time step    //0.0833333
		// 	mTimeStep = atof(buffer);
		// 	mMotions.resize(mNumTotalFrames);  //mMotions store all the joint information in each frame
		// 	for(auto& m_t : mMotions)
		// 		m_t = Eigen::VectorXd::Zero(mNumTotalChannels);
		// 	double val;
		// 	for(int i=0;i<mNumTotalFrames;i++)
		// 	{
		// 		for(int j=0;j<mNumTotalChannels;j++)
		// 		{
		// 			is>>val;
		// 			mMotions[i][j]=val;
		// 		}
		// 	}
		// }
	
	is.close();

	// for(auto ss : mBVHMap)
	// {std::cout << ss.first <<  "  " << ss.second << std::endl;
	// }

	BodyNode* root = mSkeleton->getRootBodyNode();
	// std::string root_bvh_name = mBVHMap[root->getName()];  //root_bvh_name 
	std::string root_bvh_name = root->getName();

	Eigen::VectorXd m = mMotions[0];
	if(walk_skill==true)
		m[2]=0;


	Eigen::Vector3d _p = Eigen::VectorXd::Zero(3);
	if(walk_skill==true)
	{
		_p.segment<3>(0) = unit_scale*m.segment<3>(0);
		Eigen::Quaterniond Q0(m.segment<4>(3));
		Eigen::Matrix3d rotationMatrix0 = Q0.toRotationMatrix();
		T0.linear() = rotationMatrix0;
		T0.linear().setIdentity();
		T0.translation() = _p;
	}
	else if(squat_skill)
	{
		_p.segment<2>(0) = unit_scale*m.segment<2>(0);
		T0.linear() = R_z(m[2]);
		T0.translation() = _p;
	}


	m = mMotions[mNumTotalFrames-1];

	_p = Eigen::VectorXd::Zero(3);
	if(walk_skill==true)
	{
		_p.segment<3>(0) = unit_scale*m.segment<3>(0);
		Eigen::Quaterniond Q1(m.segment<4>(3));
		Eigen::Matrix3d rotationMatrix1 = Q1.toRotationMatrix();
		T1.linear() = rotationMatrix1;
		T1.translation() = _p;
		T1.linear().setIdentity();
	}
	else if(squat_skill==true)
	{
		_p.segment<2>(0) = unit_scale*m.segment<2>(0);
		T1.linear() = R_z(m[2]);
		T1.translation() = _p;
	}


	// setBVHNodeEndEffectorFlag(mRoot); 

}

void
BVH::
setBVHNodeEndEffectorFlag(BVHNode* bn)
{
	bool flag = false; 
	if (std::find(mBVHEE.begin(), mBVHEE.end(), bn->GetName()) != mBVHEE.end())
	{
		flag = true;
	}
	bn->SetEndEffectorFlag(flag); 
	
	std::vector<BVHNode*> ss = bn->GetChildren(); 
	for(auto& cbn : ss){
		setBVHNodeEndEffectorFlag(cbn);
	}
}



BVHNode*
BVH::
ReadHierarchy(BVHNode* parent,const std::string& name,int& channel_offset,std::ifstream& is)
{
	char buffer[256];                                                
	double offset[3];
	std::vector<std::string> c_name;                                                    

	BVHNode* new_node = new BVHNode(name,parent);
	mMap.insert(std::make_pair(name,new_node));

	is>>buffer; //{

	while(is>>buffer)
	{
		if(!strcmp(buffer,"}"))
			break;
		if(!strcmp(buffer,"OFFSET"))
		{
			//Ignore
			double x,y,z;

			is>>x;
			is>>y;
			is>>z;
			// new_node->SetOffset(-z,y,-x); 
			new_node->SetOffset(x,y,z); 
		}                           // offset value x, y,z 
		else if(!strcmp(buffer,"CHANNELS"))
		{

			is>>buffer;
			int n;             // n=6
			n= atoi(buffer);           // convert string to int
			
			for(int i=0;i<n;i++)
			{
				is>>buffer;
				c_name.push_back(std::string(buffer));             // c_name store the channels information
			}
			
			new_node->SetChannel(channel_offset,c_name);
			
			channel_offset+=n;                                 //Channel_offset =   Channel_offset     
		}
		else if(!strcmp(buffer,"JOINT"))
		{
			is>>buffer;
			BVHNode* child = ReadHierarchy(new_node,std::string(buffer),channel_offset,is);
			new_node->AddChild(child);
		}
		else if(!strcmp(buffer,"End"))
		{
			is>>buffer;
			BVHNode* child = ReadHierarchy(new_node,std::string("EndEffector"),channel_offset,is);
			new_node->AddChild(child);
		}
	}
	
	return new_node;
}



std::map<std::string,MASS::BVHNode::CHANNEL> BVHNode::CHANNEL_NAME =
{
	{"Xposition",Xpos},
	{"XPOSITION",Xpos},
	{"Yposition",Ypos},
	{"YPOSITION",Ypos},
	{"Zposition",Zpos},
	{"ZPOSITION",Zpos},
	{"Xrotation",Xrot},
	{"XROTATION",Xrot},
	{"Yrotation",Yrot},
	{"YROTATION",Yrot},
	{"Zrotation",Zrot},
	{"ZROTATION",Zrot}
};
};

