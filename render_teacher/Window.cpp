
#include "Window.h"
#include "Environment.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <cmath>
#include "matplotlibcpp.h"
#include "Force.h"
#include "BodyForce.h"
namespace plt = matplotlibcpp;
#include <fstream>
#include <iterator>


using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

Window::
Window(Environment* env)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawOBJ(false),mDrawCollision(false),mDrawContact(true), mDrawShape(true), mDrawMuscles(true),
	mDrawSpringforce(false), mDrawEndEffectors(false), mDrawEndEffectorTargets(false), mDrawShadow(true),mMuscleNNLoaded(false),mDrawFigure(false),mDrawCompositionforces(false)
{

	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 1.0;
	mBackground[3] = 1.0;
	SetFocusing();
	mZoom = 0.25;	
	mFocus = false;
	mNNLoaded = false;
	if(mEnv->GetUseSymmetry())
		for(int i; i<mEnv->GetNumAction()/2; i++)
			torque_vectors.push_back(new std::vector<double>());
	else
	    for(int i; i<mEnv->GetNumAction(); i++)
			torque_vectors.push_back(new std::vector<double>());
	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");
	t = 0; 
	
	p::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();

	sys_module.attr("path").attr("insert")(1, module_dir);
	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from TeacherPolicy import *",mns);

}
Window::
Window(Environment* env,const std::string& nn_path)
	:Window(env)
{
	mNNLoaded = true;
	boost::python::str str = ("num_targets = "+std::to_string(mEnv->GetNumFutureTargetmotions())).c_str();
	p::exec(str,mns);

	str = ("num_state = "+std::to_string(mEnv->GetNumFullObservation()-mEnv->GetNumFutureTargetmotions()-mEnv->GetNumStateHistory()-mEnv->GetNumRootInfo())).c_str();
	p::exec(str,mns);
	if(mEnv->GetUseSymmetry())
		str = ("num_action = "+std::to_string(mEnv->GetNumAction()/2)).c_str();
	else
		str = ("num_action = "+std::to_string(mEnv->GetNumAction())).c_str();
	p::exec(str,mns);
	nn_module = p::eval("TeacherNN(num_targets, num_state,num_action)",mns);
	p::object load = nn_module.attr("load");
	load(nn_path);

}
Window::
Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path)
	:Window(env,nn_path)
{
	mMuscleNNLoaded = true;

	boost::python::str str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetNumTotalRelatedDofs())).c_str();
	p::exec(str,mns);
	str = ("num_actions = "+std::to_string(mEnv->GetNumAction())).c_str();
	p::exec(str,mns);
	str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetMuscles().size())).c_str();
	p::exec(str,mns);

	muscle_nn_module = p::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);

	p::object load = muscle_nn_module.attr("load");
	load(muscle_nn_path);
	std::cout << "here load" << std::endl;
}
void
Window::
draw()
{	
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];
	mViewMatrix.linear() = A;
	mViewMatrix.translation() = b;

	auto ground = mEnv->GetGround();
	float y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;


	auto& ctresults = mEnv->GetWorld()->getConstraintSolver()->getLastCollisionResult();

	DrawGround(y);

	if(mDrawContact) DrawContactForces(ctresults);
	if(mDrawEndEffectors) DrawEndEffectors();
    DrawExternalforce();
	if(mDrawMuscles) DrawMuscles(mEnv->GetCharacter()->GetMuscles());
	DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());
	if(mDrawSpringforce)DrawSpringforce();
	// Eigen::Quaterniond q = mTrackBall.getCurrQuat();
	// q.x() = 0.0;
	// q.z() = 0.0;
	// q.normalize();
	// mTrackBall.setQuaternion(q);
	SetFocusing();
}
void
Window::
keyboard(unsigned char _key, int _x, int _y)
{
	switch (_key)
	{
	case 's': this->Step();break;
	case 'f': mFocus = !mFocus;break;
	case 'r': this->Reset();break;
	case ' ': mSimulating = !mSimulating;break;
	case 'm': mDrawMuscles = !mDrawMuscles; break;
	case 'd': mDrawSpringforce =!mDrawSpringforce; break;
	case 'p': mDrawShape = !mDrawShape; break;
	case 'o': mDrawOBJ = !mDrawOBJ;break;
	case 'n': mDrawCollision = !mDrawCollision;break;    // add collision shape key
	case 'c': mDrawContact = !mDrawContact; break;
	case 'e': mDrawEndEffectors = !mDrawEndEffectors; break;
	case 't': mDrawEndEffectorTargets = !mDrawEndEffectorTargets; break;
	case 'k': mDrawFigure = !mDrawFigure; break;
	case 'w': mDrawCompositionforces = !mDrawCompositionforces; break;
	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}

}
void
Window::
displayTimer(int _val)
{
	if(mSimulating)
		Step();
	glutPostRedisplay();
	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
}

void
Window::
Step()
{	
	if (!mEnv->GetUsetargetvisual())
	{ 
		plt::ion();
		int num = mEnv->GetSimulationHz()/mEnv->GetControlHz();
		Eigen::VectorXd action;
		if(mNNLoaded)
			action = GetActionFromNN();
		else
			action = Eigen::VectorXd::Zero(mEnv->GetNumAction());
		Eigen::VectorXd action_full = Eigen::VectorXd::Zero(action.rows()); 
		if(mEnv->GetUseSymmetry())
		{
			action_full.resize(action.rows()*2);
			action_full << action, action; 
		}
		else
		{
			action_full << action;
		} 
		mEnv->SetAction(action_full);
		mEnv->UpdateActionBuffer(action_full);
		if (mEnv->GetWalkSkill())
			action_foot_vector.push_back(action_full(3));
		action_ankle_vector.push_back(action_full(2));
		action_knee_vector.push_back(action_full(1));
		action_hip_vector.push_back(action_full(0));

		if(mEnv->GetUseMuscle() && mEnv->GetUseMuscleNN())
		{
			int inference_per_sim = 2;
			for(int i=0;i<num;i+=inference_per_sim){
				Eigen::VectorXd mt = mEnv->GetMuscleTorques();
				if (mEnv->GetUseMuscleNN())
				{
					mEnv->SetActivationLevels(GetActivationFromNN(mt));
					for(int j=0;j<inference_per_sim;j++)
							mEnv->Step();
				}
			}
		}
		else
		{   
			for(int i=0;i<num;i++)
			{   
				// set mAction as the interpolation of PrevAction, Current Action; 
				mEnv->ProcessAction(i, num);
	
				mEnv->Step();
			}	

			t = mEnv->GetWorld()->getTime();
			Character* mCharacter = mEnv->GetCharacter();
			double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1]-mEnv->GetGround()->getRootBodyNode()->getCOM()[1];
			Eigen::Vector6d root_pos = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
			Eigen::Isometry3d cur_root_inv = mCharacter->GetSkeleton()->getRootBodyNode()->getWorldTransform().inverse();

			Eigen::Vector3d root_v = mCharacter->GetSkeleton()->getBodyNode(0)->getCOMLinearVelocity();
			double root_v_norm = root_v.norm();
	
			Eigen::Vector3d foot_l =  mCharacter->GetSkeleton()->getBodyNode("l_foot")->getWorldTransform().translation();
			Eigen::Vector3d foot_r =  mCharacter->GetSkeleton()->getBodyNode("r_foot")->getWorldTransform().translation();
			double pos_foot_l =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("l_foot_ground")->getCOM()(1);
			double pos_foot_r =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("r_foot_ground")->getCOM()(1);
			foot_left_Forward_vector.push_back(foot_l(0));
			foot_left_Height_vector.push_back(foot_l(1));

			foot_right_Forward_vector.push_back(foot_r(0));
			foot_right_Height_vector.push_back(foot_r(1));


			//////////////////////plot the error 
			double plot_time = mEnv->GetWorld()->getTime();
            Eigen::Vector3d skel_COM = mCharacter->GetSkeleton()->getCOM();
			Eigen::VectorXd pos0 = mEnv->GetCharacter()->GetSkeleton()->getPositions();
			const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
			if (forces.size()!= 0)
			{
				double hip_force = forces[0]->GetForce().norm();
				
				double femur_force_l = sqrt(pow(forces[1]->GetForce().norm(),2) +pow(forces[2]->GetForce().norm(),2));
				double femur_force_r = sqrt(pow(forces[3]->GetForce().norm(),2) +pow(forces[4]->GetForce().norm(),2));
				double tibia_force_l = (forces[5]->GetForce()).norm();
				double tibia_force_r = (forces[6]->GetForce()).norm();
				hip_force_vector.push_back(hip_force);
				
				femur_force_vector_l.push_back(femur_force_l);
				femur_force_vector_r.push_back(femur_force_r);
				tibia_force_vector_l.push_back(tibia_force_l);
				tibia_force_vector_r.push_back(tibia_force_r);
			}


			std::tuple<double,double,double,double,double,double,double,Eigen::VectorXd,double,double,double,double,double,double,double,double,double,double> tmp = mEnv->GetRenderReward_Error();
			double pos_reward = std::get<0>(tmp);
			double vel_reward = std::get<1>(tmp);
			double ee_reward = std::get<2>(tmp);
			double root_reward = std::get<3>(tmp);
			double parallel_reward = std::get<4>(tmp);
			double cop_left_reward = std::get<5>(tmp);
			double cop_right_reward = std::get<6>(tmp);

			Eigen::VectorXd torque =  std::get<7>(tmp);

			double hip_l_tar = std::get<8>(tmp);
			double knee_l_tar = std::get<9>(tmp);
			double ankle_l_tar = std::get<10>(tmp);
			double foot_l_tar = std::get<11>(tmp);

			double hip_l_cur = std::get<12>(tmp);
			double knee_l_cur = std::get<13>(tmp);
			double ankle_l_cur = std::get<14>(tmp);
			double foot_l_cur = std::get<15>(tmp);
			double cop_left_error = std::get<16>(tmp);
			double cop_right_error = std::get<17>(tmp);

			pos_reward_vector.push_back(pos_reward); 
			vel_reward_vector.push_back(vel_reward); 
			ee_reward_vector.push_back(ee_reward); 
			root_reward_vector.push_back(root_reward); 
			foot_clearance_reward_vector.push_back(parallel_reward);
		
			cop_left_reward_vector.push_back(cop_left_reward); 
			cop_right_reward_vector.push_back(cop_right_reward); 


			skel_COM_Forward_vector.push_back(skel_COM(0));
			skel_COM_Height_vector.push_back(skel_COM(1));
			skel_COM_Lateral_vector.push_back(skel_COM(2));


			if(mEnv->GetUseSymmetry()){
				for(int i; i<mEnv->GetNumAction()/2; i++)
					torque_vectors[i]->push_back(torque[i]); 
			}
			else{
				for(int i; i<mEnv->GetNumAction(); i++)
					torque_vectors[i]->push_back(torque[i]); 
			}

			
			hip_l_tar_vector.push_back(hip_l_tar);
			knee_l_tar_vector.push_back(knee_l_tar);
			ankle_l_tar_vector.push_back(ankle_l_tar);
			foot_l_tar_vector.push_back(foot_l_tar);

			hip_l_cur_vector.push_back(hip_l_cur);
			knee_l_cur_vector.push_back(knee_l_cur);
			ankle_l_cur_vector.push_back(ankle_l_cur);
			foot_l_cur_vector.push_back(foot_l_cur);

			cop_left_error_vector.push_back(cop_left_error); 
			cop_right_error_vector.push_back(cop_right_error); 
			time_vector.push_back(plot_time);

			/////////////////////////////////////// save data to .txt 2/22///////////////////////////////////////////////////////
			
			// pos_reward
			// Open a file for writing
    		std::ofstream pos_reward_txt("pos_reward_vector.txt");

			// Check if the file is open
			if (pos_reward_txt.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : pos_reward_vector) {
					pos_reward_txt << element << " ";
				}

				// Add a newline at the end
				pos_reward_txt << "\n";

				// Close the file
				pos_reward_txt.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}

			//----------------------------time vector----------------------------------
			// Open a file for writing
    		std::ofstream outputFile("time_vector.txt");

			// Check if the file is open
			if (outputFile.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : time_vector) {
					outputFile << element << " ";
				}

				// Add a newline at the end
				outputFile << "\n";

				// Close the file
				outputFile.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}

			//------------------ee_reward----------------------
			// pos_reward
			// Open a file for writing
    		std::ofstream ee_reward_txt("ee_reward_vector.txt");

			// Check if the file is open
			if (ee_reward_txt.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : ee_reward_vector) {
					ee_reward_txt << element << " ";
				}

				// Add a newline at the end
				ee_reward_txt << "\n";

				// Close the file
				ee_reward_txt.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}

				/////////////////////////////////////// torque_vectors ///////////////////////////////////////////////////////
			
			// hip_torque
			// Open a file for writing
    		std::ofstream torque_hip_txt("torque_hip.txt");

			// Check if the file is open
			if (torque_hip_txt.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : *(torque_vectors[0])) {
					torque_hip_txt << element << " ";
				}

				// Add a newline at the end
				torque_hip_txt << "\n";

				// Close the file
				torque_hip_txt.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}



			// knee_torque
			// Open a file for writing
    		std::ofstream torque_knee_txt("torque_knee.txt");

			// Check if the file is open
			if (torque_knee_txt.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : *(torque_vectors[1])) {
					torque_knee_txt << element << " ";
				}

				// Add a newline at the end
				torque_knee_txt << "\n";

				// Close the file
				torque_knee_txt.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}



			// ankle_torque
			// Open a file for writing
    		std::ofstream torque_ankle_txt("torque_ankle.txt");

			// Check if the file is open
			if (torque_ankle_txt.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : *(torque_vectors[2])) {
					torque_ankle_txt << element << " ";
				}

				// Add a newline at the end
				torque_ankle_txt << "\n";

				// Close the file
				torque_ankle_txt.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}




			// foot_torque
			// Open a file for writing
    		std::ofstream torque_foot_txt("torque_foot.txt");

			// Check if the file is open
			if (torque_foot_txt.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : *(torque_vectors[3])) {
					torque_foot_txt << element << " ";
				}

				// Add a newline at the end
				torque_foot_txt << "\n";

				// Close the file
				torque_foot_txt.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}

		/////////////////////////////////////////////////////////////////////////////////////////////
			
			std::map<std::string, std::string> a0 = {{"color","black"}, {"linestyle","--"},{"label","pos_reward"}};
			std::map<std::string, std::string> a1 = {{"color","magenta"}, {"linestyle",":"},{"label","ee_reward"}};
			std::map<std::string, std::string> a2 = {{"color","yellow"}, {"label","root_reward"}};
			std::map<std::string, std::string> a3 = {{"color","red"},{"marker","+"}, {"label","CoP_left_reward"}};
			std::map<std::string, std::string> a4 = {{"color","green"}, {"label","CoP_right_reward"}};

			std::map<std::string, std::string> a5 = {{"color","black"}, {"label","hip_l_tar"}};
			std::map<std::string, std::string> a6 = {{"color","blue"}, {"label","knee_l_tar"}};
			std::map<std::string, std::string> a7 = {{"color","red"}, {"label","ankle_l_tar"}};
			std::map<std::string, std::string> a8 = {{"color","magenta"}, {"label","foot_l_tar"}};

			std::map<std::string, std::string> a9 = {{"color","black"}, {"linestyle","--"},{"label","hip_l_cur"}};
			std::map<std::string, std::string> a10 = {{"color","blue"}, {"linestyle","--"},{"label","knee_l_cur"}};
			std::map<std::string, std::string> a11 = {{"color","red"}, {"linestyle","--"},{"label","ankle_l_cur"}};
            std::map<std::string, std::string> a12 = {{"color","magenta"}, {"linestyle","--"},{"label","foot_l_cur"}};

			std::map<std::string, std::string> a13 = {{"color","magenta"}, {"label","COP_left_error"}};
			std::map<std::string, std::string> a14 = {{"color","yellow"}, {"label","COP_right_error"}};

			std::map<std::string, std::string> a15 = {{"color","black"}, {"label","hip torque"}};
			std::map<std::string, std::string> a16 = {{"color","blue"}, {"linestyle","-."}, {"label","knee torque"}};
			std::map<std::string, std::string> a17 = {{"color","red"}, {"linestyle","--"}, {"label","ankle torque"}};
			std::map<std::string, std::string> a18 = {{"color","magenta"}, {"linestyle","--"}, {"label","foot torque"}};

			std::map<std::string, std::string> a19 = {{"color","black"},{"label","action-hip"}};
			std::map<std::string, std::string> a20 = {{"color","blue"}, {"linestyle","-."},{"label","action-knee"}};
			std::map<std::string, std::string> a21 = {{"color","red"}, {"linestyle","--"}, {"label","action-ankle"}};
			std::map<std::string, std::string> a22 = {{"color","magenta"}, {"linestyle","--"}, {"label","action-foot"}};
			

			std::map<std::string, std::string> a23 = {{"color","blue"}, {"linewidth","2"},{"label","hip_force"}};
			std::map<std::string, std::string> a24 = {{"color","red"},{"linewidth","2"},{"linestyle","-."},{"label","femur_force"}};
			std::map<std::string, std::string> a25 = {{"color","red"}, {"linewidth","2"},{"linestyle","-."},{"label","femur_force_r"}};
			std::map<std::string, std::string> a26 = {{"color","black"}, {"linewidth","2"}, {"linestyle","--"},{"label","tibia_force"}};
			std::map<std::string, std::string> a27 = {{"color","black"}, {"linewidth","2"}, {"linestyle","--"},{"label","tibia_force_r"}};

			std::map<std::string, std::string> a28 = {{"color","red"}, {"linewidth","1"},{"label","skel_COM_XY"}};
			std::map<std::string, std::string> a29 = {{"color","black"}, {"linewidth","1"},{"label","skel_COM_height"}};
			std::map<std::string, std::string> a30 = {{"color","blue"}, {"linewidth","1"},{"label","skel_COM_lateral"}};
			
			std::map<std::string, std::string> a_p = {{"color","green"}, {"linewidth","1"},{"label","foot_clearance_reward"}};



			if (mDrawFigure)
			{ 
				// numpy.savetxt("try_pos_reward_stu.csv", a, delimiter=",")

				plt::figure(0);
				plt::clf();
				plt::subplot(2,1,1);
				plt::title("Real-time Sub-Reward");
				// plt::xlabel("Time/s");
				plt::ylabel("Real-time Reward");
				// plt::plot(time_vector,pos_reward_vector, a0);//position tracking reward -- strong tracking  of the control system
				// plt::plot(time_vector,ee_reward_vector,a1);//end-effector reward-- strong tracking of the control system
				plt::plot(time_vector,root_reward_vector,a2);//Root Reward-- track the task root joint motion, position & rotation
				plt::plot(time_vector,foot_clearance_reward_vector,a_p);//Foot clearance Reward--encourage the foot to stay parallel with the ground and create more foot clearance to avoid tripping.
				std::map<std::string, std::string> loc = {{"loc","upper left"}};
				plt::legend(loc);
				// plt::pause(0.001); 


				plt::subplot(2,1,2);
				// plt::title("CoP Reward");
				plt::xlabel("Time/s");
				plt::ylabel("CoP Reward");//system stability and balance
				plt::plot(time_vector,cop_left_reward_vector,a3);
				plt::plot(time_vector,cop_right_reward_vector,a4);



				// plt::xlabel("Time/s");
				// plt::ylabel("Foot clearance Reward");
	


				plt::legend(loc);




				plt::figure(1);
				plt::clf();
				plt::subplot(2,1,1);
				// plt::title("Joint torque");
				// plt::xlabel("Time/s");
				plt::ylabel("Torque/N*m");
				plt::plot(time_vector,*(torque_vectors[0]),a15);//reduce energy consumption and to improve efficiency
				plt::plot(time_vector,*(torque_vectors[1]), a16);
				plt::plot(time_vector,*(torque_vectors[2]), a17);
				// plt::plot(time_vector,*(torque_vectors[3]), a12);
				plt::legend(loc);
			

				plt::subplot(2,1,2);
				// plt::title("Action from NN");
				plt::xlabel("Time/s");
				plt::ylabel("Action from NN/rad");
				plt::plot(time_vector, action_hip_vector, a19);//Prediction Action from NN
				plt::plot(time_vector, action_knee_vector, a20);
				plt::plot(time_vector, action_ankle_vector, a21);
				if (mEnv->GetWalkSkill())
					plt::plot(time_vector, action_foot_vector, a22);
				plt::legend(loc);


				plt::figure(2);
				plt::clf();
				plt::title("Imitation Reward: model VS reference");
				plt::xlabel("Time/s");
				plt::ylabel("Reward");
				plt::plot(time_vector,pos_reward_vector, a0);//position tracking reward -- strong tracking  of the control system
				plt::plot(time_vector,ee_reward_vector,a1);//end-effector reward-- strong tracking of the control system
				plt::legend(loc);

				// plt::show()
				// plt::pause(0.00001); 


/////////////////////////////////////////////////////////////////////////////////////////////

			if (forces.size()!= 0)
			{
					plt::figure(1);
					plt::clf();
					// plt::subplot(3,1,1);
					plt::xlabel("Time/s");
					plt::ylabel("Human perburbation force/N");
					plt::plot(time_vector,hip_force_vector,a23);
					// plt::legend(loc);

					// plt::subplot(3,1,2);
					plt::plot(time_vector,femur_force_vector_l,a24);
					// plt::plot(time_vector,femur_force_vector_r,a17);
					// plt::legend(loc);
					// plt::subplot(3,1,3);
					plt::plot(time_vector,tibia_force_vector_l,a26);
					// plt::plot(time_vector,tibia_force_vector_l,a19);
					plt::legend(loc);
			}
				// plt::figure(2);
				// plt::clf();
				// // plt::subplot(3,1,1);
				// plt::title("COM position");
				// plt::xlabel("X/m");
				// plt::ylabel("Y/m");
				// plt::plot(skel_COM_Forward_vector,skel_COM_Height_vector,a26);
				// // plt::plot(time_vector,skel_COM_Height_vector,a27);
				// // plt::plot(time_vector,skel_COM_Lateral_vector,a28);
				// plt::legend(loc);

				plt::show();
				plt::pause(0.00001); 
		

			}


			
		}

		mEnv->UpdateStateBuffer();
	}
	else
	{
		plt::ion();
		Character* mCharacter = mEnv->GetCharacter();
		// t = mEnv->GetWorld()->getTime();
		// std::tuple<Eigen::VectorXd, Eigen::Vector3d> tmp = mCharacter->GetBVH()->GetMotion(t);
		std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mEnv->GetControlHz());
		std::cout << "time is " << t << std::endl;
		time_vector.push_back(t);
		auto targetPositions = std::get<0>(pv);
		// std::cout << "target root :\n" << targetPositions.head(6) << std::endl;
		auto targetVelocities = std::get<1>(pv);
		auto targetEE_pos = std::get<2>(pv);
		mCharacter->targetEE_pos = targetEE_pos; 
		mCharacter->GetSkeleton()->setPositions(targetPositions); // set position
		mCharacter->GetSkeleton()->setVelocities(targetVelocities); //set velocities
		mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);
		// double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1]+1.4;
		double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mEnv->GetGround()->getRootBodyNode()->getCOM()[1];

		Eigen::VectorXd com_diff = mCharacter->GetSkeleton()->getCOM();
		Eigen::Vector3d skel_COM = mCharacter->GetSkeleton()->getCOM();
		skel_COM_Forward_vector.push_back(skel_COM(0));
		skel_COM_Height_vector.push_back(skel_COM(1));
		skel_COM_Lateral_vector.push_back(skel_COM(2));
		com_info.push_back(com_diff);
    	auto ees = mCharacter->GetEndEffectors();
		Eigen::VectorXd ee_diff(ees.size()*3);
		// for(int i =0;i<ees.size();i++)
		// {
		// 	ee_diff.segment<3>(i*3) = ees[i]->getCOM();  
		// 	std::cout <<  ees[i]->getCOM() << std::endl;
		// }
		Eigen::Vector3d foot_l =  mCharacter->GetSkeleton()->getBodyNode("l_foot")->getWorldTransform().translation();
		Eigen::Vector3d foot_r =  mCharacter->GetSkeleton()->getBodyNode("r_foot")->getWorldTransform().translation();

		foot_left_Forward_vector.push_back(foot_l(0));
		foot_left_Height_vector.push_back(foot_l(1));

		foot_right_Forward_vector.push_back(foot_r(0));
		foot_right_Height_vector.push_back(foot_r(1));



		std::map<std::string, std::string> loc = {{"loc","upper left"}};
		std::map<std::string, std::string> a26 = {{"color","red"}, {"linewidth","1"},{"label","skel_COM_XY"}};
		std::map<std::string, std::string> a27 = {{"color","black"}, {"linewidth","1"},{"label","skel_COM_height"}};
		std::map<std::string, std::string> a28 = {{"color","blue"}, {"linewidth","1"},{"label","skel_COM_lateral"}};

		std::map<std::string, std::string> a29 = {{"color","red"}, {"linewidth","1"},{"label","COP_left"}};
		std::map<std::string, std::string> a30 = {{"color","blue"}, {"linewidth","1"},{"label","COP_right"}};

		std::map<std::string, std::string> a31 = {{"color","red"}, {"linewidth","1"},{"label","foot_left"}};
		std::map<std::string, std::string> a32 = {{"color","blue"}, {"linewidth","1"},{"label","foot_right"}};

		if (mDrawFigure)
		{ 
			plt::figure(0);
			plt::clf();
			plt::title("COM position");
			plt::xlabel("X/m");
			plt::ylabel("Y/m");
			plt::plot(skel_COM_Forward_vector,skel_COM_Height_vector,a26);
			// plt::plot(time_vector,skel_COM_Height_vector,a27);
			// plt::plot(time_vector,skel_COM_Lateral_vector,a28);
			plt::legend(loc);


			plt::figure(1);
			plt::clf();
			plt::title("Foot position");
			plt::xlabel("X/m");
			plt::ylabel("Y/m");
			plt::plot(foot_left_Forward_vector,foot_left_Height_vector,a31);
			plt::plot(foot_right_Forward_vector,foot_right_Height_vector,a32);
			plt::legend(loc);
			plt::show();
			plt::pause(0.00001); 

		}
		t += 1.0/mEnv->GetControlHz(); 
	}
	
	
}

void
Window::
DrawExternalforce()
{
	
	const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
	
	double radius = 0.005; //2cm radius 0.005
	double heightN = 5.0e-3;//1mm per N 5.0e-3
	double coneHt = 2.0e-2; //cone height
	// std::cout << forces.size() << std::endl; 
	// std::cout << forces[0]->GetName() << forces[0]->GetForce() << std::endl; 
	// std::cout << forces[1]->GetName() << forces[1]->GetForce() << std::endl; 
	// std::cout << forces[i]->GetName() << std::endl; 
	for(int i = 0; i < forces.size(); ++i) {
		auto& _force = forces[i];
		// _force->Update(); // random Offset
		// _force->UpdatePos();
		if (_force->GetName().find("springforce")!=std::string::npos)
			continue;
		auto pos = _force->GetPos();
		auto force = _force->GetForce();
		// std::cout << "force :" << i << "\n" << force << std::endl;
		// std::cout << "pos  :" << i << "\n" << pos << std::endl;
		Eigen::Vector4d color; 
		color << 0.678431, 0.478431, 0.478431,1.0;  //red

		mRI->setPenColor(color);
		mRI->pushMatrix();
		mRI->translate(pos);
		mRI->drawSphere(radius);
		mRI->popMatrix();

		Eigen::Vector3d pos2 = pos + force * heightN;
		Eigen::Vector3d u(0, 0, 1);
		Eigen::Vector3d v = pos2 - pos;
		Eigen::Vector3d mid = 0.5 * (pos + pos2);
		double len = v.norm();
		v /= len;
		Eigen::Isometry3d T;
		T.setIdentity();
		Eigen::Vector3d axis = u.cross(v);
		axis.normalize();
		double angle = acos(u.dot(v));
		Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
		w_bracket(0, 1) = -axis(2);
		w_bracket(1, 0) = axis(2);
		w_bracket(0, 2) = axis(1);
		w_bracket(2, 0) = -axis(1);
		w_bracket(1, 2) = -axis(0);
		w_bracket(2, 1) = axis(0);

		Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(angle)) * w_bracket + (1.0 - cos(angle)) * w_bracket * w_bracket;
		T.linear() = R;
		T.translation() = mid;
		mRI->pushMatrix();
		mRI->transform(T);
		mRI->drawCylinder(radius, len);
		mRI->popMatrix();


		T.translation() = pos2;
		mRI->pushMatrix();
		mRI->transform(T);
		mRI->drawCone(2* radius, coneHt);
		mRI->popMatrix();
	}


}

void
Window::
DrawSpringforce()
{
	const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
	int count =0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	

	for(int i=0; i<forces.size(); ++i)
	{
		auto& _force = forces[i];
		if (_force->GetName().find("springforce")==std::string::npos)
		{
			continue;
		}
		double a = 0;
		Eigen::Vector4d color(1,0.0,1.0,1.0);//
		mRI->setPenColor(color);
		auto aps =_force->GetPoint();
		for(int i=0;i<aps.size();i++)
		{
			Eigen::Vector3d p = aps[i];
			mRI->pushMatrix();
			mRI->translate(p);
			mRI->drawSphere(0.01*sqrt(1000/1000.0));
			mRI->popMatrix();
		}
		// std::cout << _force->GetForce() << std::endl;
		///////////////////////////draw force
	    for(int i=0;i<aps.size()-1;i++)
		{	
			if ((_force->GetName()=="foot_springforce_l") || (_force->GetName()=="foot_springforce_r"))
			{
				double radius = 0.005; //2cm radius 0.005
				double heightN = 5.0e-3;//1mm per N 5.0e-3
				double coneHt = 2.0e-2; //cone height
				Eigen::Vector3d pos = aps[i]; 
				Eigen::Vector4d force_color; 
				force_color << 0.678431, 0.478431, 0.478431,1.0;  //red

				mRI->setPenColor(force_color);
				mRI->pushMatrix();
				mRI->translate(pos);
				mRI->drawSphere(radius);
				mRI->popMatrix();
				std::cout << "_forcename:\n" << _force->GetName() << std::endl;
				std::cout << _force->GetForce() << std::endl;
				Eigen::Vector3d pos2 = pos + _force->GetForce()* heightN;
				Eigen::Vector3d u(0, 0, 1);
				Eigen::Vector3d v = pos2 - pos;
				Eigen::Vector3d mid = 0.5 * (pos + pos2);
				double len = v.norm();
				v /= len;
				Eigen::Isometry3d T;
				T.setIdentity();
				Eigen::Vector3d axis = u.cross(v);
				axis.normalize();
				double angle = acos(u.dot(v));
				Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
				w_bracket(0, 1) = -axis(2);
				w_bracket(1, 0) = axis(2);
				w_bracket(0, 2) = axis(1);
				w_bracket(2, 0) = -axis(1);
				w_bracket(1, 2) = -axis(0);
				w_bracket(2, 1) = axis(0);

				Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(angle)) * w_bracket + (1.0 - cos(angle)) * w_bracket * w_bracket;
				T.linear() = R;
				T.translation() = mid;
				mRI->pushMatrix();
				mRI->transform(T);
				mRI->drawCylinder(radius, len);
				mRI->popMatrix();

				T.translation() = pos2;
				mRI->pushMatrix();
				mRI->transform(T);
				mRI->drawCone(2* radius, coneHt);
				mRI->popMatrix();
			}
		}

		for(int i=0;i<aps.size()-1;i++)
		{
			Eigen::Vector3d p = aps[i];
			Eigen::Vector3d p1 = aps[i+1];

			Eigen::Vector3d u(0,0,1);
			Eigen::Vector3d v = p-p1;
			Eigen::Vector3d mid = 0.5*(p+p1);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) =  axis(2);
			w_bracket(0, 2) =  axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) =  axis(0);

			
			Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(0.01*sqrt(1000/1000.0),len);
			mRI->popMatrix();
		}
		
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}

void
Window::
Reset()
{
	mEnv->Reset();
}
void
Window::
SetFocusing()
{
	if(mFocus)
	{
		//if(mEnv->GetWorld()->getNumSkeletons() == 0) return;
		if(mEnv->GetWorld()->getNumSkeletons() == 1) mTrans = -mEnv->GetWorld()->getSkeleton(0)->getRootBodyNode()->getCOM();
		else if(mEnv->GetWorld()->getNumSkeletons() > 1){
			std::string name = mEnv->GetWorld()->getSkeleton(0)->getName();
			boost::to_lower(name);
			if(name!= "ground") mTrans = -mEnv->GetWorld()->getSkeleton(0)->getRootBodyNode()->getCOM();
			else mTrans = -mEnv->GetWorld()->getSkeleton(1)->getRootBodyNode()->getCOM();
		}

		//mTrans = -mEnv->GetWorld()->getSkeleton("NJIT_BME_EXO_Model")->getRootBodyNode()->getCOM();   // load NJIT_BME_EXO_Model
		//mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();              // load Human MASS model

		mTrans[1] -= 0.3;

		mTrans *=1000.0;
		
	}
}

np::ndarray toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	for(int i =0;i<n;i++)
	{
		dest[i] = vec[i];
	}

	return array;
}


Eigen::VectorXd
Window::
GetActionFromNN()
{
	p::object get_action;
	get_action= nn_module.attr("get_action");
	int numstates = mEnv->GetNumFullObservation();
	int numtargets = mEnv->GetNumFutureTargetmotions(); 
	int numcurstate = mEnv->GetNumState();	
	int numstatehistory =  mEnv->GetNumStateHistory()+mEnv->GetNumRootInfo(); 

	Eigen::VectorXd _state = mEnv->GetFullObservation().head(numstates-numtargets);
	Eigen::VectorXd  state = _state.tail(_state.rows()-numstatehistory);

	Eigen::VectorXd target = mEnv->GetFullObservation().tail(numtargets);

	p::tuple shape1 = p::make_tuple(state.rows());
	np::dtype dtype1 = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape1,dtype1);
	float* dest1 = reinterpret_cast<float*>(state_np.get_data());
	for(int i =0;i<state.rows();i++)
		dest1[i] = state[i];

	p::tuple shape2 = p::make_tuple(target.rows());
	np::dtype dtype2 = np::dtype::get_builtin<float>();
	np::ndarray target_np = np::empty(shape2,dtype2);
	float* dest2 = reinterpret_cast<float*>(target_np.get_data());
	for(int i =0;i<target.rows();i++)
		dest2[i] = target[i];

	p::object temp = get_action(target_np,state_np);
	np::ndarray action_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	Eigen::VectorXd action(mEnv->GetNumAction());
	if(mEnv->GetUseSymmetry()){
		action.resize(mEnv->GetNumAction()/2);
	}
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];
	return action;
}

Eigen::VectorXd
Window::
GetActivationFromNN(const Eigen::VectorXd& mt)
{
	if(!mMuscleNNLoaded)
	{
		mEnv->GetDesiredTorques();
		return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
	}
	p::object get_activation = muscle_nn_module.attr("get_activation");
	Eigen::VectorXd dt = mEnv->GetDesiredTorques();
	np::ndarray mt_np = toNumPyArray(mt);
	np::ndarray dt_np = toNumPyArray(dt);

	p::object temp = get_activation(mt_np,dt_np);
	np::ndarray activation_np = np::from_object(temp);

	Eigen::VectorXd activation(mEnv->GetCharacter()->GetMuscles().size());
	float* srcs = reinterpret_cast<float*>(activation_np.get_data());
	for(int i=0;i<activation.rows();i++)
		activation[i] = srcs[i];

	return activation;
}

void
Window::
DrawEntity(const Entity* entity)
{
	if (!entity)
		return;
	const auto& bn = dynamic_cast<const BodyNode*>(entity);
	if(bn)
	{
		DrawBodyNode(bn);
		return;
	}

	const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
	if(sf)
	{
		DrawShapeFrame(sf);
		return;
	}
}
void
Window::
DrawBodyNode(const BodyNode* bn)
{	
	if(!bn)
		return;
	if(!mRI)
		return;

	mRI->pushMatrix();
	mRI->transform(bn->getRelativeTransform());
	// std::cout << bn->getName() << "  " << bn->getRelativeTransform().translation() << std::endl; 
	auto sns = bn->getShapeNodesWith<VisualAspect>();
	for(const auto& sn : sns)
		DrawShapeFrame(sn);

	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	mRI->popMatrix();

}
void
Window::
DrawSkeleton(const SkeletonPtr& skel)
{	
	DrawBodyNode(skel->getRootBodyNode());
}
void
Window::
DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	if(!mRI)
		return;

	const auto& va = sf->getVisualAspect();

	if(va && !va->isHidden()){
		mRI->pushMatrix();
		mRI->transform(sf->getRelativeTransform());
		if(mDrawShape) DrawShape(sf->getShape().get(),va->getRGBA());
		mRI->popMatrix();
	}

	const auto& ca = sf->getCollisionAspect();
	if(ca){
		mRI->pushMatrix();
		mRI->transform(sf->getRelativeTransform());
		Eigen::Vector4d color(1.0,0.0,1.0,1.0);
		if(mDrawCollision) DrawCollisionShape(sf->getShape().get(), color);
		mRI->popMatrix();
	}
			
	if(sf->getName() == "r_foot_ground_ShapeNode_0"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_0.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "r_foot_ground_ShapeNode_1"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_1.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "r_foot_ground_ShapeNode_2"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_2.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "r_foot_ground_ShapeNode_3"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_3.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_0"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_0.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_1"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_1.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_2"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_2.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_3"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_3.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}


}

void
Window::
DrawCollisionShape(const Shape* shape,const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	mRI->setPenColor(color);

	if (shape->is<SphereShape>())
	{
		const auto* sphere = static_cast<const SphereShape*>(shape);
		mRI->drawSphere(sphere->getRadius());
	}
	if (shape->is<BoxShape>())
	{
		const auto* box = static_cast<const BoxShape*>(shape);
		mRI->drawCube(box->getSize());
	}
	glDisable(GL_COLOR_MATERIAL);
}

void
Window::
DrawShape(const Shape* shape,const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	mRI->setPenColor(color);
	
	// if(mDrawOBJ == false)
	// {
	// 	if (shape->is<BoxShape>())
	// 	{
	// 			const auto* box = static_cast<const BoxShape*>(shape);
	// 			mRI->drawCube(box->getSize());
	// 	}
	// 	else if (shape->is<CylinderShape>())
	// 	{
	// 			const auto* cylinder = static_cast<const CylinderShape*>(shape);
	// 			mRI->drawCylinder(cylinder->getRadius(), cylinder->getHeight());
	// 	}	
	// }
	// else
	// {   
	// 	if(shape->is<MeshShape>())
	// 	{
	// 		const auto& mesh = static_cast<const MeshShape*>(shape);
	// 		glDisable(GL_COLOR_MATERIAL);
	// 		mRI->drawMesh(mesh->getScale(), mesh->getMesh());
	// 		float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	// 		this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
	// 	}
	// 	if(mDrawCollision==true)
	//     {
	// 		if (shape->is<SphereShape>())
	// 		{
	// 			const auto* sphere = static_cast<const SphereShape*>(shape);
	// 			mRI->drawSphere(sphere->getRadius());
	// 		}
			
	//     }
	// }
	if(mDrawOBJ == false)
	{
		if(shape->is<MeshShape>())
		{
			const auto& mesh = static_cast<const MeshShape*>(shape);
			glDisable(GL_COLOR_MATERIAL);
			mRI->drawMesh(mesh->getScale(), mesh->getMesh());
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			
			this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
		}
		else{
			
			if (shape->is<BoxShape>())
			{
					const auto* box = static_cast<const BoxShape*>(shape);
					mRI->drawCube(box->getSize());
			}
			else if (shape->is<CylinderShape>())
			{
					const auto* cylinder = static_cast<const CylinderShape*>(shape);
					mRI->drawCylinder(cylinder->getRadius(), cylinder->getHeight());
			}	
		}

	}
     
	glDisable(GL_COLOR_MATERIAL);
}


void 
Window::
DrawArrow(Eigen::Vector3d pos, Eigen::Vector3d force, Eigen::Vector4d color, double radius,double heightN, double coneHt)
{
	mRI->setPenColor(color);
	mRI->pushMatrix();
	mRI->translate(pos);
	mRI->drawSphere(radius);
	mRI->popMatrix();

	Eigen::Vector3d pos2 = pos + force * heightN;
	Eigen::Vector3d u(0, 0, 1);
	Eigen::Vector3d v = pos2 - pos;
	Eigen::Vector3d mid = 0.5 * (pos + pos2);
	double len = v.norm();
	v /= len;
	Eigen::Isometry3d T;
	T.setIdentity();
	Eigen::Vector3d axis = u.cross(v);
	axis.normalize();
	double angle = acos(u.dot(v));
	Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
	w_bracket(0, 1) = -axis(2);
	w_bracket(1, 0) = axis(2);
	w_bracket(0, 2) = axis(1);
	w_bracket(2, 0) = -axis(1);
	w_bracket(1, 2) = -axis(0);
	w_bracket(2, 1) = axis(0);

	Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(angle)) * w_bracket + (1.0 - cos(angle)) * w_bracket * w_bracket;
	T.linear() = R;
	T.translation() = mid;
	mRI->pushMatrix();
	mRI->transform(T);
	mRI->drawCylinder(radius, len);
	mRI->popMatrix();


	T.translation() = pos2;
	mRI->pushMatrix();
	mRI->transform(T);
	mRI->drawCone(2* radius, coneHt);
	mRI->popMatrix();
}

// draw contact forces
void 
Window::
DrawContactForces(collision::CollisionResult& results)
{
	plt::ion();
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
    auto& ees = mEnv->GetCharacter()->GetEndEffectors();
	double radius = 0.005; //2cm radius
	double heightN = 3.0e-3;//1mm per N
	double coneHt = 2.0e-2; //cone height
    // Eigen::Vector3d pos_Root = mEnv->GetCharacter()->GetSkeleton()->getRootBodyNode()->getCOM();
	Eigen::Vector3d pos_foot_r = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("r_foot")->getCOM();
	Eigen::Vector3d pos_foot_l = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("l_foot")->getCOM();
	pos_foot_l(1) =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("l_foot_ground")->getCOM()(1);
	pos_foot_r(1) =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("r_foot_ground")->getCOM()(1);

	Eigen::Vector3d geo_center_target_left = pos_foot_l;
	Eigen::Vector3d geo_center_target_right = pos_foot_r;
	Eigen::Vector3d geo_center_target = (pos_foot_l + pos_foot_r)/2;
    Eigen::Vector4d color1(1.0,0.0,1.0,1.0); 
	mRI->setPenColor(color1);
	mRI->pushMatrix();
	mRI->translate(geo_center_target_left);
	mRI->drawSphere(radius*2);
	mRI->popMatrix();


	mRI->setPenColor(color1);
	mRI->pushMatrix();
	mRI->translate(geo_center_target_right);
	mRI->drawSphere(radius*2);
	mRI->popMatrix();


	Eigen::Vector3d pos = Eigen::Vector3d::Zero();
	Eigen::Vector3d force = Eigen::Vector3d::Zero();
    std::vector<constraint::ContactConstraintPtr>  mContactConstraints;

	// store all the pos and force 
	std::vector<Eigen::Vector3d> all_pos;
	std::vector<Eigen::Vector3d> all_force;
	std::vector<Eigen::Vector3d> all_pos_left;
	std::vector<Eigen::Vector3d> all_pos_right;
	std::vector<Eigen::Vector3d> all_force_left;
	std::vector<Eigen::Vector3d> all_force_right;

	Eigen::Vector4d color; 
	Eigen::Vector3d left_pos, left_force, right_pos, right_force;
	left_pos.setZero(); left_force.setZero(); right_pos.setZero(); right_force.setZero(); 
	for(int i = 0; i < results.getNumContacts(); ++i) 
	{
		auto& contact = results.getContact(i);
		mContactConstraints.clear();
		mContactConstraints.push_back(
				std::make_shared<constraint::ContactConstraint>(contact, mEnv->GetWorld()->getTimeStep()));
		auto pos = contact.point;
		auto force = contact.force;

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
		all_pos.push_back(pos);
		all_force.push_back(force);
		for (const auto& contactConstraint : mContactConstraints)
		{
			if(body1->getName() == "l_foot_ground")
			{
				color << 1.0,0.0,0.0,1.0;  //red
				left_pos += pos;
				left_force += force;
			}
			else if(body1->getName() == "r_foot_ground")
			{
				color << 0.0,1.0,0.0,1.0;  //
				right_pos += pos;
				right_force += force;
			}
			else
			{
				std::cout << body1->getName() << std::endl;
				std::cout << "-----Warning: contact force not on foot-------" << std::endl;
			}
		}
		if (!mDrawCompositionforces)
		   DrawArrow(pos, force, color, radius, heightN, coneHt);
        
	}

//////////////////////////////////////////////// calculate the COP(center of pressure)
	Eigen::Vector3d unitV;
	unitV << 0, 1, 0;    // unit normal vector  
	Eigen::Matrix3f A;
	Eigen::Vector3f b;
	Eigen::Vector3d p, f;

    //////////////////////////////////////////// first method  -- calculate COP of both foot 
	Eigen::Vector3d COP;
	// COP = geo_center_target;
	Eigen::Vector3d p_cross_f;
	double f_sum = 0; 
	p_cross_f.setZero();

	// for(int i=0; i<all_pos.size(); i++){
	// 	p = all_pos[i];
	// 	double f_scalar = all_force[i].dot(unitV);
	// 	f_sum += f_scalar; 
	// 	p_cross_f += p.cross(f_scalar * unitV);
	// }
	// if(all_pos.size()!= 0){
	// 	COP = -p_cross_f.cross(unitV) / f_sum;
	for(int i=0; i<all_pos.size(); i++){
		p = all_pos[i];
		double f_scalar = all_force[i].dot(unitV);
		f_sum += f_scalar; 
		p_cross_f += p.cross(f_scalar * unitV);
	}
	if(all_pos.size() != 0){
		COP = -p_cross_f.cross(unitV) / f_sum;
		COP(1) = geo_center_target(1);	
	
		Eigen::Vector4d color4(0.0,0.0,1.0,1.0);  
		mRI->setPenColor(color4);
		mRI->pushMatrix();
		mRI->translate(COP);
		mRI->drawSphere(radius*2);
		mRI->popMatrix();
	}

	//////////////////////////////////////////// first method  -- calculate COP of each foot
	Eigen::Vector3d COP_left;
	COP_left = geo_center_target_left;
	Eigen::Vector3d p_cross_f_left;
	double f_sum_left = 0; 
	p_cross_f_left.setZero();

	for(int i=0; i<all_pos_left.size(); i++){
		p = all_pos_left[i];
		double f_scalar_left = all_force_left[i].dot(unitV);
		f_sum_left += f_scalar_left; 
		p_cross_f_left += p.cross(f_scalar_left * unitV);
	}
	if(all_pos_left.size()!= 0){
		COP_left = -p_cross_f_left.cross(unitV) / f_sum_left;
		COP_left(1) = geo_center_target_left(1);		
		Eigen::Vector4d color4(0.0,1.0,0.0,1.0);  //red
		mRI->setPenColor(color4);
		mRI->pushMatrix();
		mRI->translate(COP_left);
		mRI->drawSphere(radius*2);
		mRI->popMatrix();
	}
	else 
		COP_left.setZero();

	Eigen::Vector3d COP_right;
	COP_right = geo_center_target_right;
	Eigen::Vector3d p_cross_f_right;
	double f_sum_right = 0; 
	p_cross_f_right.setZero();

	for(int i=0; i<all_pos_right.size(); i++){
		p = all_pos_right[i];
		double f_scalar_right = all_force_right[i].dot(unitV);
		f_sum_right += f_scalar_right; 
		p_cross_f_right += p.cross(f_scalar_right * unitV);
	}
	if(all_pos_right.size() != 0){
		COP_right = -p_cross_f_right.cross(unitV) / f_sum_right;
		COP_right(1) = geo_center_target_right(1);	

		Eigen::Vector4d color4(1.0,0.0,0.0,1.0);  
		mRI->setPenColor(color4);
		mRI->pushMatrix();
		mRI->translate(COP_right);
		mRI->drawSphere(radius*2);
		mRI->popMatrix();
	}
	else
		COP_right.setZero();


	if (COP_left(0)!=0)
		cop_left_Forward_vector.push_back(COP_left(0)); 
	else
		cop_left_Forward_vector.push_back(nan("")); 
	
	if (COP_left(1)!=0)	
		cop_left_Height_vector.push_back(COP_left(1));
	else
		cop_left_Height_vector.push_back(nan("")); 

	if (COP_left(2)!=0)	
		cop_left_Lateral_vector.push_back(COP_left(2));
	else
		cop_left_Lateral_vector.push_back(nan("")); 


	if (COP_right(0)!=0)
		cop_right_Forward_vector.push_back(COP_right(0));
	else
		cop_right_Forward_vector.push_back(nan("")); 


	if (COP_right(1)!=0)	 
		cop_right_Height_vector.push_back(COP_right(1));
	else
		cop_right_Height_vector.push_back(nan("")); 

	if (COP_right(2)!=0)	
		cop_right_Lateral_vector.push_back(COP_right(2));
	else
		cop_right_Lateral_vector.push_back(nan("")); 



	if (mDrawCompositionforces)
	{
		left_pos /= 4;
		right_pos /= 4;
		color << 1.0,0.0,0.0,1.0;  //red
		DrawArrow(COP_left, left_force, color, radius, heightN, coneHt);
		color << 0.0,1.0,0.0,1.0;  //green
		DrawArrow(COP_right, right_force, color, radius, heightN, coneHt);
	}

	double plot_time = mEnv->GetWorld()->getTime();
	t_vector.push_back(plot_time);

	std::map<std::string, std::string> loc = {{"loc","upper left"}};
	std::map<std::string, std::string> a0_0 = {{"color","blue"},{"label","contact_force_forward"}};
	std::map<std::string, std::string> a0_1 = {{"color","red"},{"label","contact_force_height"}};
	std::map<std::string, std::string> a0_2 = {{"color","black"},{"label","contact_force_lateral"}};

	std::map<std::string, std::string> a1_0 = {{"color","red"}, {"linewidth","1"},{"label","COP_left"}};
	std::map<std::string, std::string> a1_1 = {{"color","green"}, {"linewidth","1"},{"label","COP_right"}};

	contact_force_vector_l_forward.push_back(left_force(0));
	contact_force_vector_l_height.push_back(left_force(1));
	contact_force_vector_l_lateral.push_back(left_force(2));
	contact_force_vector_r_forward.push_back(right_force(0));
	contact_force_vector_r_height.push_back(right_force(1));
	contact_force_vector_r_lateral.push_back(right_force(2));

	/////////////////////////////////////// save data to .txt 2/22///////////////////////////////////////////////////////
			
			// pos_reward
			// Open a file for writing
    		std::ofstream contact_force_vector_l_height_txt("contact_force_vector_l_height.txt");

			// Check if the file is open
			if (contact_force_vector_l_height_txt.is_open()) {
				// Iterate through the vector and write each element to the file
				for (const auto &element : contact_force_vector_l_height) {
					contact_force_vector_l_height_txt << element << " ";
				}

				// Add a newline at the end
				contact_force_vector_l_height_txt << "\n";

				// Close the file
				contact_force_vector_l_height_txt.close();

				// std::cout << "Vector has been saved to 'output.txt'" << std::endl;
			} else {
				std::cerr << "Unable to open the file." << std::endl;
			}

			
		/////////////////////////////////////////////////////////////////////////////////////////////
			

	if (mDrawFigure)
	{ 

		plt::figure(3);
		plt::clf();
		plt::title("COP position");
		plt::xlabel("X/m");
		plt::ylabel("Z/m");
		plt::plot(cop_left_Forward_vector,cop_left_Lateral_vector,a1_0);
		plt::plot(cop_right_Forward_vector,cop_right_Lateral_vector,a1_1);
		//plt::plot(time_vector,skel_COM_Height_vector,a27);
		// plt::plot(time_vector,skel_COM_Lateral_vector,a28);
		plt::legend(loc);


		plt::figure(4);
		plt::clf();
		plt::subplot(2,1,1);
		plt::title("Contact force_left_foot");
		// plt::xlabel("Time/s");
		plt::ylabel("Force/N");
		plt::plot(t_vector,contact_force_vector_l_forward,a0_0);
		plt::plot(t_vector,contact_force_vector_l_height,a0_1);
		plt::plot(t_vector,contact_force_vector_l_lateral,a0_2);
		plt::legend(loc);

		plt::subplot(2,1,2);
		plt::title("Contact force_right_foot");
		plt::xlabel("Time/s");
		plt::ylabel("Force/N");
		plt::plot(t_vector,contact_force_vector_r_forward,a0_0);
		plt::plot(t_vector,contact_force_vector_r_height,a0_1);
		plt::plot(t_vector,contact_force_vector_r_lateral,a0_2);
		plt::legend(loc);



		plt::show();
		plt::pause(0.00001); 

	}

	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

   //////////////////////////////////////////// second method
	// A.setZero();
	// b.setZero();
	// for(int i=0; i<all_pos_left.size(); i++){
	// 	p = all_pos_left[i];
	// 	f = all_force_left[i].dot(unitV) * unitV;

	// 	A(0, 0) += 0;
	// 	A(0, 1) += -f(2);
	// 	A(0, 2) += f(1);

	// 	A(1, 0) += -f(1);
	// 	A(1, 1) += f(0);
	// 	A(1, 2) += 0;

	// 	A(2, 0) += unitV(0);
	// 	A(2, 1) += unitV(1);
	// 	A(2, 2) += unitV(2);

	// 	b(0) += p(2) * f(1) - p(1)* f(2);
	// 	b(1) += p(1) * f(0) - p(0)* f(1);
	// 	b(2) += 0; //p.dot(unitV); 
	// }

	// if(all_pos_left.size() != 0)
	// {
	// 	Eigen::Vector3f COP_left1 = A.colPivHouseholderQr().solve(b);
	// 	std::cout << "COP_left1:\n" << COP_left1 << std::endl; 
	// 	COP_left1(1) = -0.88;
	// 	Eigen::Vector4d color4(0.0,0.8,0.0,1.0);  //red
	// 	mRI->setPenColor(color4);
	// 	mRI->pushMatrix();
	// 	mRI->translate(COP_left1.cast<double>());
	// 	mRI->drawSphere(radius*2);
	// 	mRI->popMatrix();
	// }


	//  Draw Hunt-Crossly contact model force
	 if(mEnv->GetUseHuntContactForce())
    {
		std::vector<Eigen::Vector3d> hunt_contact_pos_left;
		std::vector<Eigen::Vector3d> hunt_contact_pos_right;
		std::vector<Eigen::Vector3d> hunt_contact_force_left;
		std::vector<Eigen::Vector3d> hunt_contact_force_right;
		auto huntConactInfo = mEnv->getHuntContactInfo();
		hunt_contact_pos_left =  std::get<0>(huntConactInfo); 
		hunt_contact_force_left =  std::get<1>(huntConactInfo); 
		hunt_contact_pos_right =  std::get<2>(huntConactInfo); 
		hunt_contact_force_right =  std::get<3>(huntConactInfo); 
		color << 0.0,1.0,1.0,1.0;  //red 
        for (int i=0; i<hunt_contact_pos_left.size(); i++)
		   DrawArrow(hunt_contact_pos_left[i], hunt_contact_force_left[i], color, radius, heightN, coneHt);
	     for (int i=0; i<hunt_contact_pos_right.size(); i++)
		 {
		   color << 0.0,1.0,1.0,1.0;  // green
		   DrawArrow(hunt_contact_pos_right[i], hunt_contact_force_right[i], color, radius, heightN, coneHt);
		 }
	}



}


void Window::
DrawEndEffectors()
{
	auto bvh = mEnv->GetCharacter()->GetBVH();
	auto& ees = mEnv->GetCharacter()->GetEndEffectors();

	Eigen::VectorXd ee_diff(ees.size()*3);
	Eigen::VectorXd com_diff;

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	double radius = 0.015; //2cm radius
	Eigen::Vector4d color(0.0,1.0,0.0,1.0);//green

	for(int i =0;i<ees.size();i++) {

		mRI->setPenColor(color);
		mRI->pushMatrix();
		mRI->translate(ees[i]->getCOM());
		mRI->drawSphere(radius);
		mRI->popMatrix();
	}

	if(mDrawEndEffectorTargets) {

		Eigen::Vector4d color2(0.0,0.0,1.0,1.0);//blue
		auto skel = mEnv->GetCharacter()->GetSkeleton();
		com_diff = skel->getCOM();
 
		double t = mEnv->GetWorld()->getTime();

		// ee posotion based on BVH
		Character* mCharacter = mEnv->GetCharacter();
        auto pos0 = skel->getPositions();
		std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mEnv->GetControlHz());
		auto targetPositions = std::get<0>(pv);
		auto targetVelocities = std::get<1>(pv);
		auto targetEE_pos = std::get<2>(pv);
		mCharacter->targetEE_pos = targetEE_pos; 
		mCharacter->GetSkeleton()->setPositions(targetPositions); // set position
		mCharacter->GetSkeleton()->setVelocities(targetVelocities); //set velocities
		mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);

	    // auto pos0 = skel->getPositions();

        auto ees = mCharacter->GetEndEffectors();

		for(int i =0;i<ees.size();i++) {

			mRI->setPenColor(color2);
			mRI->pushMatrix();
			mRI->translate(ees[i]->getCOM());
			mRI->drawSphere(radius);
			mRI->popMatrix();
		}

		skel->setPositions(pos0); //changed the state back
		skel->computeForwardKinematics(true,false,false);

	}

	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}


void
Window::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
	int count =0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
	for(auto muscle : muscles)
	{
		auto aps = muscle->GetAnchors();
		bool lower_body = true;
		double a = muscle->activation;
		// Eigen::Vector3d color(0.7*(3.0*a),0.2,0.7*(1.0-3.0*a));
		Eigen::Vector4d color(0.4+(2.0*a),0.4,0.6,1.0);//0.7*(1.0-3.0*a));
		// glColor3f(1.0,0.0,0.362);
		// glColor3f(0.0,0.0,0.0);
		mRI->setPenColor(color);
		for(int i=0;i<aps.size();i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			mRI->pushMatrix();
			mRI->translate(p);
			mRI->drawSphere(0.005*sqrt(muscle->f0/1000.0));
			mRI->popMatrix();
		}
			
		for(int i=0;i<aps.size()-1;i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			Eigen::Vector3d p1 = aps[i+1]->GetPoint();

			Eigen::Vector3d u(0,0,1);
			Eigen::Vector3d v = p-p1;
			Eigen::Vector3d mid = 0.5*(p+p1);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) =  axis(2);
			w_bracket(0, 2) =  axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) =  axis(0);

			
			Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(0.005*sqrt(muscle->f0/1000.0),len);
			mRI->popMatrix();
		}
		
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}
void
Window::
DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y) 
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(scale[0],scale[1],scale[2]);
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];

	Eigen::Affine3d M;
	M.linear() = A;
	M.translation() = b;
	M = (mViewMatrix.inverse()) * M;

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(mViewMatrix.data());
	DrawAiMesh(mesh,mesh->mRootNode,M,y);
	glPopMatrix();
	glPopMatrix();
	glEnable(GL_LIGHTING);
}
void
Window::
DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
	unsigned int i;
    unsigned int n = 0, t;
    Eigen::Vector3d v;
    Eigen::Vector3d dir(0.4,0,-0.4);
    glColor3f(0.3,0.3,0.3);
    
    // update transform

    // draw all meshes assigned to this node
    for (; n < nd->mNumMeshes; ++n) {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
            GLenum face_mode;

            switch(face->mNumIndices) {
                case 1: face_mode = GL_POINTS; break;
                case 2: face_mode = GL_LINES; break;
                case 3: face_mode = GL_TRIANGLES; break;
                default: face_mode = GL_POLYGON; break;
            }
            glBegin(face_mode);
        	for (i = 0; i < face->mNumIndices; i++)
        	{
        		int index = face->mIndices[i];

        		v[0] = (&mesh->mVertices[index].x)[0];
        		v[1] = (&mesh->mVertices[index].x)[1];
        		v[2] = (&mesh->mVertices[index].x)[2];
        		v = M*v;
        		double h = v[1]-y;
        		
        		v += h*dir;
        		
        		v[1] = y+0.001;
        		glVertex3f(v[0],v[1],v[2]);
        	}
            glEnd();
        }

    }

    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n) {
        DrawAiMesh(sc, nd->mChildren[n],M,y);
    }

}
void
Window::
DrawGround(double y)
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_LIGHTING);
	double width = 0.005;
	int count = 0;
	glBegin(GL_QUADS);
	for(double x = -100.0;x<100.01;x+=1.0)
	{
		for(double z = -100.0;z<100.01;z+=1.0)
		{
			if(count%2==0)
				glColor3f(216.0/255.0,211.0/255.0,204.0/255.0);			
			else
				glColor3f(216.0/255.0-0.1,211.0/255.0-0.1,204.0/255.0-0.1);
			count++;
			glVertex3f(x,y,z);
			glVertex3f(x+1.0,y,z);
			glVertex3f(x+1.0,y,z+1.0);
			glVertex3f(x,y,z+1.0);
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);
}
