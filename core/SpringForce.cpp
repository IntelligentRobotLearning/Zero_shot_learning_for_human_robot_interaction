#include "SpringForce.h"
#include <iostream>
#include "Environment.h"
#include <Eigen/Dense>
#include "DARTHelper.h"

namespace MASS
{
    SpringForce::SpringForce()
     :Force(), mIsForceLocal(false), mIsPosLocal(false)
    {
    }

    SpringForce::~SpringForce()
    {
    }


    Force* SpringForce::CreateForce()
    {
        return new SpringForce;
    }

    // update:
    // mCachedAnchorPositions: current anchor pos
	void SpringForce::Update() 
    {
        for(int i=0; i<mbodyNodes.size(); i++){
            mCachedAnchorPositions[i] = mbodyNodes[i]->getTransform()*mPosOffsets[i]; 
            // if (mbodyNodes[i]->getName()=="TalusL")
            //    std::cout << "TalusL should be: \n" << (mbodyNodes[1]->getTransform().inverse())*mbodyNodes[0]->getTransform()*mPosOffsets[0] << std::endl;
        }
        Force::Update(); 
    };

    void SpringForce::UpdatePos() 
    {
         for(int i=0; i<mbodyNodes.size(); i++){
            mCachedAnchorPositions[i] = mbodyNodes[i]->getTransform()*mPosOffsets[i]; 
        }

    };

    Eigen::Vector3d SpringForce::GetForce()
    { 
        
        double distance = (mCachedAnchorPositions[0] - mCachedAnchorPositions[1]).norm();
        double f = (distance - initial_length) * stiffness; 
        Eigen::Vector3d dir = mCachedAnchorPositions[1]-mCachedAnchorPositions[0];
        dir.normalize();
        return f*dir; 
    }

    void SpringForce::ApplyForceToBody() 
    {   

        Eigen::Vector3d f = GetForce();
        mbodyNodes[0]->addExtForce(f, mCachedAnchorPositions[0], mIsForceLocal, mIsPosLocal);
        mbodyNodes[1]->addExtForce(-1.0*f, mCachedAnchorPositions[1], mIsForceLocal, mIsPosLocal);
    };

    void SpringForce::ReadFromXml(TiXmlElement& inp) 
    {   

        Force::ReadFromXml(inp);
        stiffness     = std::stod(inp.Attribute("stiffness"));
        damping       = std::stod(inp.Attribute("damping"));
        if(inp.Attribute("initial_length")!=nullptr)
            initial_length = std::stod(inp.Attribute("initial_length"));
        else
            initial_length =-1;


        for(TiXmlElement* waypoint = inp.FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))
        {
            mbodyNodeNames.push_back(waypoint->Attribute("body"));
			mPosOffsets.push_back(string_to_vector3d(waypoint->Attribute("p")));
        }

        mCachedAnchorPositions.resize(mbodyNodeNames.size());
    };

    void SpringForce::SetBodyNode(dart::dynamics::SkeletonPtr mSkeleton)
    {
        for(int i=0; i<mbodyNodeNames.size();i++){
            mbodyNodes.push_back(mSkeleton->getBodyNode(mbodyNodeNames[i]));
        }
        Update();
        if (initial_length ==-1)
            initial_length = (mCachedAnchorPositions[0]- mCachedAnchorPositions[1]).norm(); 
        for(int i=0; i<mbodyNodes.size(); i++){
           if ((mbodyNodes[i]->getName()=="exo_pelvis") || (mbodyNodes[i]->getName()=="Pelvis"))
              initial_length -=0.02;
        }
}
}