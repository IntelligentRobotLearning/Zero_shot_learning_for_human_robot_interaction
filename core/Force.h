#ifndef __MASS_FORCE_H__
#define __MASS_FORCE_H__

#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <map>
#include <tinyxml.h>
#include "dart/dart.hpp"

namespace MASS
{

class Environment;

class Force
{
public:
	Force();
    virtual ~Force();

    const std::string& GetName() const {return mName;}
    void SetName(const std::string& name) {mName = name;}
	virtual void Reset() {};
	virtual void Update() {};
	virtual void ApplyForceToBody() {};
	virtual Eigen::Vector3d GetPos(){return Eigen::VectorXd::Zero(3);}
    virtual Eigen::Vector3d GetForce(){return Eigen::VectorXd::Zero(3);}
	virtual void UpdatePos() {}; 
    virtual void ReadFromXml(TiXmlElement& inp);
	virtual void SetBodyNode(dart::dynamics::SkeletonPtr) {}; 
    void SetEnvironment(Environment& env) {mEnv = &env;}
	virtual std::vector<Eigen::Vector3d> GetPoint() {};

public:

	std::string mName;
	Environment* mEnv;

};

class ForceFactory
{
public:
	static Force* CreateForce(const std::string &Force);

// private:
	ForceFactory();
    typedef Force* (*Creator)();
	static std::map<std::string, Creator> mForces;
};


}

#endif
