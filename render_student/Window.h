#ifndef __MASS_WINDOW_H__
#define __MASS_WINDOW_H__
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace MASS
{
class Environment;
class Muscle;
class Window : public dart::gui::Win3D
{
public:
	Window(Environment* env);
	Window(Environment* env,const std::string& nn_path);
	Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path);

	void draw() override;
	void keyboard(unsigned char _key, int _x, int _y) override;
	void displayTimer(int _val) override;
	Eigen::Vector3d geo_center_target, geo_center_target_left, geo_center_target_right;

private:
	void SetFocusing();

	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawCollisionShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawContactForces(dart::collision::CollisionResult& results);
	void DrawArrow(Eigen::Vector3d pos, Eigen::Vector3d force, Eigen::Vector4d color, double radius,double heightN, double coneHt); 
	void DrawExternalforce();
	void DrawSpringforce();
	void DrawEndEffectors();
	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	void DrawGround(double y);
	void Step();
	void Reset();
	

	Eigen::VectorXd GetActionFromNN();
	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt);

	p::object mm,mns,sys_module,nn_module,muscle_nn_module;
	float t; 

	Environment* mEnv;
	bool mFocus;
	bool mSimulating;
	bool mDrawMuscles;
	bool mDrawShape;
	bool mDrawOBJ;
	bool mDrawCollision;
	bool mDrawContact;
	bool mDrawEndEffectors;
	bool mDrawEndEffectorTargets;
	bool mDrawShadow;
	bool mNNLoaded;
	bool mMuscleNNLoaded;
	bool mDrawSpringforce;
	bool mDrawFigure;
	bool mDrawCompositionforces;
	Eigen::Affine3d mViewMatrix;
	std::vector<double> pos_reward_vector;
	std::vector<double> vel_reward_vector;
	std::vector<double> ee_reward_vector;
	std::vector<double> root_reward_vector;
	std::vector<double> foot_clearance_reward_vector;
	std::vector<double> cop_left_reward_vector;
	std::vector<double> cop_right_reward_vector;
	std::vector<double> hip_l_tar_vector;
	std::vector<double> knee_l_tar_vector;
	std::vector<double> ankle_l_tar_vector;
	std::vector<double> foot_l_tar_vector;
	std::vector<double> hip_l_cur_vector;
	std::vector<double> knee_l_cur_vector;
	std::vector<double> ankle_l_cur_vector;
	std::vector<double> foot_l_cur_vector;

	std::vector<double> cop_left_error_vector;
	std::vector<double> hip_force_vector;
	std::vector<double> femur_force_vector_l;
	std::vector<double> femur_force_vector_r;
	std::vector<double> tibia_force_vector_l;
	std::vector<double> tibia_force_vector_r;

	std::vector<double> cop_right_error_vector;
	std::vector<double> time_vector;
	std::vector<double> t_vector;
	std::vector<double> tt;

	std::vector<double> action_ankle_vector;
	std::vector<double> action_foot_vector;
	std::vector<double> action_knee_vector;
	std::vector<double> action_hip_vector;
	std::vector< std::vector<double>* > torque_vectors;
    std::vector<Eigen::Vector3d> com_info;
	std::vector<Eigen::Vector3d> foot1_com;
	std::vector<Eigen::Vector3d> foot2_com;
	std::vector<double> cop_left_Forward_vector;
	std::vector<double> cop_left_Height_vector;
	std::vector<double> cop_left_Lateral_vector;

	std::vector<double> cop_right_Forward_vector;
	std::vector<double> cop_right_Height_vector;
	std::vector<double> cop_right_Lateral_vector;

	std::vector<double> skel_COM_Height_vector;
	std::vector<double> skel_COM_Forward_vector;
	std::vector<double> skel_COM_Lateral_vector;
	std::vector<double>	foot_left_Forward_vector;
	std::vector<double> foot_left_Height_vector;
	std::vector<double>	foot_right_Forward_vector;
	std::vector<double>	foot_right_Height_vector;
	std::vector<double> contact_force_vector_l_forward;
	std::vector<double> contact_force_vector_l_height;
	std::vector<double> contact_force_vector_l_lateral;
	std::vector<double> contact_force_vector_r_forward;
	std::vector<double> contact_force_vector_r_height;
	std::vector<double> contact_force_vector_r_lateral;


};
};


#endif
