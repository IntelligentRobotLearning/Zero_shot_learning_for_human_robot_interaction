#include "LinearBushing.h"
#include <iostream>
#include "Environment.h"
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include "DARTHelper.h"

namespace MASS
{
    LinearBushing::LinearBushing()
     :Force()
    {
        //In the Master theis by Pujals 2017
        //Simulation of the assistance of an exoskeleton on lower limbs joints using Opensim
        //https://upcommons.upc.edu/handle/2117/110512
        //the following parameters were used
        //Vec3 transStiffness(10000), rotStiffness(1000), transDamping(0.1), rotDamping(0);
        mTranslationStiffness.setConstant(10000);
        mRotationStiffness.setConstant(1000);
        mTranslationDamping.setConstant(0.1);
        mRotationDamping.setZero();

        //default to zero so no force will be produced
        //mTranslationStiffness.setZero();
        //mRotationStiffness.setZero();
        //mTranslationDamping.setZero();
        //mRotationDamping.setZero();
    }

    LinearBushing::~LinearBushing()
    {
    }


    Force* LinearBushing::CreateForce()
    {
        return new LinearBushing;
    }

    // update:
	void LinearBushing::Update() 
    {
        Eigen::Isometry3d trf1 = mBody1->getTransform() * mFrame1;
        Eigen::Isometry3d trf2 = mBody2->getTransform() * mFrame2;
        Eigen::Isometry3d rel = trf1.inverse() * trf2;
        Eigen::Vector3d tr = rel.translation();
        Eigen::Vector3d ea = rel.rotation().eulerAngles(0,1,2);

        mForce1  = mTranslationStiffness.array() * tr.array();
        mTorque1 = mRotationStiffness.array() * ea.array();

        Eigen::Vector3d gv1 = mBody1->getLinearVelocity(mFrame1.translation());
        Eigen::Vector3d gv2 = mBody2->getLinearVelocity(mFrame2.translation());
        Eigen::Vector3d grelv = gv2- gv1;
        Eigen::Vector3d relv = trf1.inverse() * grelv;
        Eigen::Vector3d frelv = mTranslationDamping.array() * relv.array();
        mForce1 += frelv;

        //need to think how to do the torque damping
        Eigen::Vector3d gr1= mBody1->getAngularVelocity();
        Eigen::Vector3d gr2= mBody2->getAngularVelocity();
        Eigen::Vector3d grelr = gr2- gr1;
        Eigen::Vector3d relr = trf1.inverse() * grelr;
        Eigen::Vector3d trelr = mRotationDamping.array() * relr.array();
        //is the consistent with euler angle velocity?
        mTorque1 += trelr;

        Eigen::Vector3d gforce = trf1 * mForce1;
        Eigen::Vector3d gtorque = trf1 * mTorque1;
        //convert the global force and torque to the local frame
        mForce2 = -(trf2.inverse() * gforce);
        mTorque2 = -(trf2.inverse() * gtorque);

        Force::Update(); 
    };

    Eigen::Vector3d LinearBushing::GetForce()
    {
        auto trf1 = mBody1->getTransform() * mFrame1;
        Eigen::Vector3d gforce = trf1 * mForce1;
        return gforce;
    }

    Eigen::Vector3d LinearBushing::GetTorque()
    {
        auto trf1 = mBody1->getTransform() * mFrame1;
        Eigen::Vector3d gtorque = trf1 * mTorque1;
        return gtorque;
    }

    void LinearBushing::ApplyForceToBody() 
    {   
        //update(); //make sure the force is computed

        bool isPosLocal = true; 
        bool isForceLocal = true;

        mBody1->addExtForce(mFrame1 * mForce1, mFrame1.translation(), isForceLocal, isPosLocal);
        mBody1->addExtTorque(mFrame1 * mTorque1, isForceLocal);

        mBody2->addExtForce(mFrame2 * mForce2, mFrame2.translation(), isForceLocal, isPosLocal);
        mBody2->addExtTorque(mFrame2 * mTorque2, isForceLocal);
    };

    void LinearBushing::ReadFromXml(TiXmlElement& inp) 
    {   
        Force::ReadFromXml(inp);
        mTranslationStiffness  = string_to_vector3d(inp.Attribute("translation_stiffness"));
        mTranslationDamping    = string_to_vector3d(inp.Attribute("translation_damping"));
        mRotationStiffness     = string_to_vector3d(inp.Attribute("rotation_stiffness"));
        mRotationDamping       = string_to_vector3d(inp.Attribute("rotation_damping"));

        mbodyNodeNames.push_back(inp.Attribute("body1"));
        mbodyNodeNames.push_back(inp.Attribute("body2"));

        //3 translations followed by 3 Euler angles (X/Y/Z)
        Eigen::Vector6d frm1 = string_to_vectorXd(inp.Attribute("frame1"), 6);
        Eigen::Vector6d frm2 = string_to_vectorXd(inp.Attribute("frame2"), 6);

        mFrame1 = Eigen::AngleAxisd(frm1[3], Eigen::Vector3d::UnitX())
             * Eigen::AngleAxisd(frm1[4], Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(frm1[5], Eigen::Vector3d::UnitZ()); 
        mFrame1.translation() = frm1.head(3);

        mFrame2 = Eigen::AngleAxisd(frm2[3], Eigen::Vector3d::UnitX())
             * Eigen::AngleAxisd(frm2[4], Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(frm2[5], Eigen::Vector3d::UnitZ()); 
        mFrame2.translation() = frm2.head(3);

    };

    void LinearBushing::SetBodyNode(dart::dynamics::SkeletonPtr mSkeleton)
    {
        mBody1 = mSkeleton->getBodyNode(mbodyNodeNames[0]);
        mBody2 = mSkeleton->getBodyNode(mbodyNodeNames[1]);

        Update();
    }
}