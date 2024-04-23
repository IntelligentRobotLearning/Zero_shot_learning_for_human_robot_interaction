#include "BodyForce.h"
#include <iostream>
#include "Environment.h"
#include <Eigen/Dense>
#include "DARTHelper.h"

namespace MASS
{
    BodyForce::BodyForce()
     :Force(), mIsForceLocal(false), mIsPosLocal(false),Constant_force_dir(Eigen::Vector3d (0,-1,0))
    {
    }

    BodyForce::~BodyForce()
    {
    }

    Force* BodyForce::CreateForce()
    {
        return new BodyForce;
    }

	void BodyForce::Update() 
    {
        Eigen::Vector2d _a; 
        _a << (float)rand()/RAND_MAX*2-1, (float)rand()/RAND_MAX*2-1;
        Forcedir << _a(0), -7, _a(1); 
        Constant_force_dir = Constant_force_dir + 0.001*100*(Forcedir-Constant_force_dir);
        // std::cout << "Constant_force_dir"  << Constant_force_dir  << std::endl;
        Constant_force_dir.normalize();
        mForceMag = forcemag*(1-exp(-mEnv->GetWorld()->getTime())); 
        // std::cout << forcemag << "mForceMag"  << std::endl;
        mForce= mForceMag*Constant_force_dir;
        Force::Update(); 
    };
    
    void BodyForce::ApplyForceToBody() 
    {   
        // std::cout << "mName " << mName <<std::endl;
        // std::cout << "mForce\n " << mForce <<std::endl; 
        // std::cout << "mPos\n " << mPos <<std::endl; 
        mBodynode->addExtForce(mForce, mPos, mIsForceLocal, mIsPosLocal);
    };

    void BodyForce::ReadFromXml(TiXmlElement& inp) 
    {
        Force::ReadFromXml(inp);
        mBodyNodeName = inp.Attribute("body");

        if(inp.Attribute("pos")!=nullptr)
            mPosOffset  = string_to_vector3d(inp.Attribute("pos"));
        if(inp.Attribute("force")!=nullptr)
        {
            mForce = string_to_vector3d(inp.Attribute("force"));
            mForceMag = mForce.norm();
            forcemag = mForceMag;
        }
        
    };
    void BodyForce::SetBodyNode(dart::dynamics::SkeletonPtr mSkeleton)
    {
        mBodynode = mSkeleton->getBodyNode(mBodyNodeName);   
        // std::cout << "SetBodyNode: " << mBodynode << std::endl;
    }; 

   RandomBodyForce::RandomBodyForce()
     :BodyForce(), mForceMin(0), mForceMax(1.0), 
     mIsForceMagRandom(true), mIsForceDirRandom(true),
     mIsPosRandom(true), dynamic_force_dir(Eigen::Vector3d::Zero()), dynamic_force_mag(0)
    {
    }

    RandomBodyForce::~RandomBodyForce()
    {
    }

    Force* RandomBodyForce::CreateForce()
    {
        return new RandomBodyForce;
    }
    
    void RandomBodyForce::Update() 
    { 
        //generate random force and update mForce, mPos
        //mForceMag = 0; //use random
        //mForceDir;
        // std::default_random_engine e(time(0));
    
        // Eigen::Vector3d _posOffset; 
        // std::uniform_real_distribution<> param_bounds0 {-0.05, 0.05};  
        // float _r = 0.03; //[-0.05, 0.05]
        // _posOffset << (float)rand()/RAND_MAX *_r*2 - _r, (float)rand()/RAND_MAX *_r*2 - _r, (float)rand()/RAND_MAX *_r*2 - _r;

        // SetPosOffset(_posOffset);
        Eigen::Vector3d _force; 
        std::uniform_real_distribution<> param_bounds2 {mForceMin, mForceMax};  
        mForceMag = (float)rand()/RAND_MAX * (mForceMax - mForceMin) + mForceMin;
        // range random[-0.5, 0.5]

        mForceDir << (float)rand()/RAND_MAX-0.5, (float)rand()/RAND_MAX-0.5, (float)rand()/RAND_MAX-0.5; 
        mForceDir.normalize();
  

        // force delay process
	    dynamic_force_mag = dynamic_force_mag + 0.001*100*(mForceMag-dynamic_force_mag);
        dynamic_force_dir = dynamic_force_dir + 0.001*100*(mForceDir-dynamic_force_dir);

        SetForce(dynamic_force_dir*dynamic_force_mag); 
        // std::cout << "mForceMag: " << dynamic_force_mag << std::endl; 
        // std::cout << "mForceDir: \n" << mForceDir << std::endl; 
        // BodyForce::Update(); 
    };

    void RandomBodyForce::ApplyForceToBody() 
    {
        BodyForce::ApplyForceToBody();
    };

    void RandomBodyForce::UpdatePos() 
    {
        BodyForce::UpdatePos();
    };

    void RandomBodyForce::ReadFromXml(TiXmlElement& inp) 
    {
        BodyForce::ReadFromXml(inp);
        mForceMin = split_to_double(inp.Attribute("min"), 1)[0];
        mForceMax = split_to_double(inp.Attribute("max"), 1)[0];


    };

}
