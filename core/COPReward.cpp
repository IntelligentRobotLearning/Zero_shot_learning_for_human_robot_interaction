#include "COPReward.h"
#include <iostream>
#include "Environment.h"
#include <Eigen/Dense>
#include "dart/constraint/ContactConstraint.hpp"
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
namespace MASS
{
    COPReward::COPReward()
     :Reward()
    {
    }

    COPReward::~COPReward()
    {
    }

    Reward* COPReward::CreateReward()
    {
        return new COPReward;
    }

    void COPReward::ReadFromStream(std::iostream& strm)
    {
        Reward::ReadFromStream(strm);

        int nseg = 0;
        strm >> nseg; 
        
        // [left_foot]
        for(int i = 0; i < nseg; ++i) {
            std::string foot;
            strm >> foot;
            if(!foot.empty()) {
                contact_segs.push_back(foot);
            }
        }
        strm >> weight;
        // std::cout << "weight : -------------" << weight << std::endl; 

        unitV << 0, 1, 0; 
    }


    double COPReward::GetReward()
    {

        double r = 0;
        //use environment mEnv to calculate COP_error
	    Eigen::Vector3d pos_foot_r = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("r_foot")->getCOM();
	    Eigen::Vector3d pos_foot_l = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("l_foot")->getCOM();
	 	pos_foot_l(1) = -0.895985;
	    pos_foot_r(1) = -0.895985;	    
        geo_center_target_left = pos_foot_l;
	    geo_center_target_right = pos_foot_r;
        geo_center_target = (pos_foot_l + pos_foot_r)/2;
        
        auto& results = mEnv->GetWorld()->getConstraintSolver()->getLastCollisionResult();
        std::vector<constraint::ContactConstraintPtr>  mContactConstraints;


        COP_Y_fixed = geo_center_target_left(1);

        Eigen::Vector3d p;
        std::vector<Eigen::Vector3d> Pos, Force; 
        std::vector<Eigen::Vector3d> all_pos;
        std::vector<Eigen::Vector3d> all_force;
        std::vector<Eigen::Vector3d> all_pos_left;
        std::vector<Eigen::Vector3d> all_pos_right;
        std::vector<Eigen::Vector3d> all_force_left;
        std::vector<Eigen::Vector3d> all_force_right;
        if(!mEnv->GetUseHuntContactForce()){
            for(int i = 0; i < results.getNumContacts(); ++i)   // store all contact forces 
            {
                auto& contact = results.getContact(i);
                mContactConstraints.clear();
		        mContactConstraints.push_back(std::make_shared<constraint::ContactConstraint>(contact, mEnv->GetWorld()->getTimeStep()));
                auto pos = contact.point;
                auto force = contact.force;
                all_pos.push_back(pos);
                all_force.push_back(force);

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
                    if(body1->getName() =="l_foot_ground")                   
                    {
                        all_pos_left.push_back(pos);
                        all_force_left.push_back(force);
                    }
                    else if(body1->getName() == "r_foot_ground")
                    {
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
        }
        else{
            auto huntConactInfo = mEnv->getHuntContactInfo();
            all_pos_left =  std::get<0>(huntConactInfo); 
            all_force_left =  std::get<1>(huntConactInfo); 
            all_pos_right =  std::get<2>(huntConactInfo); 
            all_force_right =  std::get<3>(huntConactInfo);
        }

        for (int _i =0; _i < contact_segs.size(); _i++){
            if(contact_segs[_i] == "left_foot"){
                Pos.insert(Pos.end(), all_pos_left.begin(), all_pos_left.end());
                Force.insert(Force.end(), all_force_left.begin(), all_force_left.end());
                
            }else if (contact_segs[_i] == "right_foot"){
                Pos.insert(Pos.end(), all_pos_right.begin(), all_pos_right.end());
                Force.insert(Force.end(), all_force_right.begin(), all_force_right.end());
            }        
        }

        if(contact_segs.size()==1){
            // COP = (contact_segs[0] == "left_foot") ? COP_left : COP_right; 
            COP_target = (contact_segs[0] == "left_foot") ? geo_center_target_left : geo_center_target_right;  
        }else if (contact_segs.size()==2)
        {
            COP_target =  geo_center_target; 
        }

        Eigen::Vector3d p_cross_f;
        double f_sum = 0; 
        p_cross_f.setZero();

        for(int i=0; i<Pos.size(); i++){
            p = Pos[i];
            double f_scalar = Force[i].dot(unitV);
            f_sum += f_scalar; 
            p_cross_f += p.cross(f_scalar * unitV);
        }
        if(f_sum!=0){
            COP = -p_cross_f.cross(unitV) / f_sum;
            COP(1) = COP_Y_fixed;
            Eigen::Vector3d COP_error = COP_target - COP;
            double distance = COP_error.squaredNorm();
            double d_error = std::max(distance-0.005, 0.00);
            r = exp(-40*fabs(d_error));  // train
        }
        else{
            r = 1;
        }
        // std::cout << "COPReward   " << r << std::endl;
        return r;
    }

}