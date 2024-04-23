#include "COMReward.h"
#include <iostream>
#include "Environment.h"
#include <Eigen/Dense>

namespace MASS
{
    COMReward::COMReward()
     :Reward()
    {
    }

    COMReward::~COMReward()
    {
    }

    Reward* COMReward::CreateReward()
    {
        return new COMReward;
    }

    void COMReward::ReadFromStream(std::iostream& strm)
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
                //std::cout << foot << std::endl;
            }
        }
        strm >> weight;
        //read additional parameters
        
        //initialize data 
        unitV << 0, 1, 0; 

    }


    double COMReward::GetReward()
    {

        double r_ZCM = 0;
        //use environment mEnv to calculate COP_error
        const auto& results = mEnv->GetWorld()->getConstraintSolver()->getLastCollisionResult();
        COP_left = mEnv->geo_center_target_left;
        COP_right = mEnv->geo_center_target_right;
        COP_Y_fixed = COP_right(1);
        COP = mEnv->geo_center_target;
        Eigen::Vector3d p;
        std::vector<Eigen::Vector3d> Pos, Force; 
        std::vector<Eigen::Vector3d> all_pos;
        std::vector<Eigen::Vector3d> all_force;
        std::vector<Eigen::Vector3d> all_pos_left;
        std::vector<Eigen::Vector3d> all_pos_right;
        std::vector<Eigen::Vector3d> all_force_left;
        std::vector<Eigen::Vector3d> all_force_right;
        Eigen::Vector3d combined_force;
        combined_force.setZero();
        for(int i = 0; i < results.getNumContacts(); ++i)   // store all contact forces 
        {
            auto& contact = results.getContact(i);
            auto pos = contact.point;
            auto force = contact.force;
            combined_force += force; 
            // all_pos.push_back(pos);
            // all_force.push_back(force);
            if(pos(2) < 0){
                all_pos_left.push_back(pos);
                all_force_left.push_back(force);
            }else{
                all_pos_right.push_back(pos);
                all_force_right.push_back(force);
            }
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
            COP = (contact_segs[0] == "left_foot") ? COP_left : COP_right; 
            COP_target = (contact_segs[0] == "left_foot") ? mEnv->geo_center_target_left : mEnv->geo_center_target_right;  
        }else{
            COP_target =  mEnv->geo_center_target; 
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
        }
        COP(1) = COP_Y_fixed;

        Eigen::Vector3d COP_COM = mEnv->GetCharacter()->GetSkeleton()->getCOM()- COP;
        double cos_distance; 
        if (COP_COM.norm()*combined_force.norm() != 0)
           cos_distance = (combined_force.dot(COP_COM))/(COP_COM.norm()*combined_force.norm());
        else
           cos_distance = 0;

        r_ZCM = exp(-10*acos(cos_distance));   // train 
        //r_ZCM = acos(cos_distance);        // ZCM error
        //std::cout << "angle_distance   "  <<  r_ZCM << std::endl;
        return r_ZCM;
    }

}
