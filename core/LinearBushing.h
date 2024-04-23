#ifndef __MASS_LinearBushing_H__
#define __MASS_LinearBushing_H__
#include <Eigen/Dense>
#include "Force.h"
#include "dart/dart.hpp"

namespace MASS
{

class LinearBushing : public Force
{
public:
	LinearBushing();
    ~LinearBushing();

	virtual void Update();

	virtual void ApplyForceToBody();

    virtual void ReadFromXml(TiXmlElement& inp);
    
    static Force* CreateForce();
    void SetBodyNode(dart::dynamics::SkeletonPtr mSkeleton); 
    
    virtual Eigen::Vector3d GetForce(); 
    virtual Eigen::Vector3d GetTorque(); 

private:
    dart::dynamics::SkeletonPtr mSkeleton; 

    std::vector<std::string>  mbodyNodeNames; 

    dart::dynamics::BodyNode* mBody1;
    dart::dynamics::BodyNode* mBody2;

    Eigen::Isometry3d mFrame1;
    Eigen::Isometry3d mFrame2;

    Eigen::Vector3d mTranslationStiffness;
    Eigen::Vector3d mRotationStiffness;

    Eigen::Vector3d mTranslationDamping;
    Eigen::Vector3d mRotationDamping;

    Eigen::Vector3d mForce1; //force on the local frame of body 1
    Eigen::Vector3d mTorque1;//torque ...

    Eigen::Vector3d mForce2;//force on the local frame of body 2
    Eigen::Vector3d mTorque2;//torque ...
};

}

#endif
