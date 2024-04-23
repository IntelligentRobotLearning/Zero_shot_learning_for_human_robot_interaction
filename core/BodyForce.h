#ifndef __MASS_BODYFORCE_H__
#define __MASS_BODYFORCE_H__
#include <Eigen/Dense>
#include "Force.h"
#include "dart/dart.hpp"

namespace MASS
{

class BodyForce : public Force
{
public:
	BodyForce();
    ~BodyForce();

	virtual void Update();
	virtual void ApplyForceToBody();

    virtual void ReadFromXml(TiXmlElement& inp);
    
    static Force* CreateForce();
    void SetBodyNode(dart::dynamics::SkeletonPtr mSkeleton);
    void SetForce(Eigen::Vector3d f) {mForce = f;}
    void SetPosOffset(Eigen::Vector3d p){mPosOffset = p;}
    void UpdatePos() {
        mPos = mBodynode->getCOM()+mPosOffset;
        } 
    virtual Eigen::Vector3d GetPos(){return mPos;}
    virtual Eigen::Vector3d GetForce(){return mForce;}
    dart::dynamics::BodyNode* GetBodyNode(){return mBodynode;}
    virtual void Reset() {Constant_force_dir(0)=0; Constant_force_dir(1)=-1;Constant_force_dir(2)=0;}   
     
private:
    dart::dynamics::BodyNode* mBodynode;
    std::string mBodyNodeName; 
    Eigen::Vector3d mForce;
    Eigen::Vector3d mPos;
    Eigen::Vector3d mPosOffset;
    Eigen::Vector3d Forcedir;
    Eigen::Vector3d Constant_force_dir;

    double mForceMag;
    double forcemag;
    bool mIsForceLocal;
    bool mIsPosLocal;

};


class RandomBodyForce : public BodyForce
{
public:
	RandomBodyForce();
    ~RandomBodyForce();

	virtual void Update();
	virtual void ApplyForceToBody();
    virtual void ReadFromXml(TiXmlElement& inp);
    virtual void UpdatePos(); 
    static Force* CreateForce();
    virtual void Reset() {dynamic_force_dir.setZero(); dynamic_force_mag=0;}

private:

    double mForceMag;
    Eigen::Vector3d mForceDir;

    double mForceMin, mForceMax;  
    bool mIsForceMagRandom; //is force magnitude random?
    bool mIsForceDirRandom; //is force direction random?
    bool mIsPosRandom;
    Eigen::Vector3d dynamic_force_dir;
    double dynamic_force_mag;

};

}

#endif
