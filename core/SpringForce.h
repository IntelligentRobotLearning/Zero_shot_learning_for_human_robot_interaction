#ifndef __MASS_SPRINGFORCE_H__
#define __MASS_SPRINGFORCE_H__
#include <Eigen/Dense>
#include "Force.h"
#include "dart/dart.hpp"

namespace MASS
{

class SpringForce : public Force
{
public:
	SpringForce();
    ~SpringForce();

	virtual void Update();
    virtual void UpdatePos();
	virtual void ApplyForceToBody();

    virtual void ReadFromXml(TiXmlElement& inp);
    
    static Force* CreateForce();
    void SetBodyNode(dart::dynamics::SkeletonPtr mSkeleton); 
    
    std::vector<Eigen::Vector3d> GetPoint(){return mCachedAnchorPositions;}
    // virtual Eigen::Vector3d GetForce(){return mForce;}
    // dart::dynamics::BodyNode* GetBodyNode(){return mBodynode;}
    // dart::dynamics::BodyNode* GetConnectedBodyNode(){return mConnectedBodynode;}
    // virtual void Reset() {Constant_force_dir(0)=0; Constant_force_dir(1)=-1;Constant_force_dir(2)=0;}   
    virtual Eigen::Vector3d GetForce(); 
private:
    dart::dynamics::SkeletonPtr mSkeleton; 

    std::vector<std::string>  mbodyNodeNames; 
    std::vector<dart::dynamics::BodyNode*> mbodyNodes;
	std::vector<Eigen::Vector3d>  mPosOffsets;
    std::vector<Eigen::Vector3d> mCachedAnchorPositions; // current position;

    double stiffness;
    double damping;
    double mForceMag;
    double forcemag;
    bool mIsForceLocal;
    bool mIsPosLocal;
    double initial_length;

};

}

#endif
