#include "Force.h"
#include "BodyForce.h"
#include "SpringForce.h"

namespace MASS
{
    Force::Force()
    {      
    }

    Force::~Force()
    {      
    }

    void Force::ReadFromXml(TiXmlElement& inp)
    {
     	mName = inp.Attribute("name");
    }

    std::map<std::string, ForceFactory::Creator> ForceFactory::mForces;

    Force* ForceFactory::CreateForce(const std::string& name) {
        if(mForces.empty()) {
            mForces.emplace("BodyForce", &BodyForce::CreateForce);
            mForces.emplace("RandomBodyForce", &RandomBodyForce::CreateForce);
            mForces.emplace("SpringForce",     &SpringForce::CreateForce);
        }

        auto it = mForces.find(name);
        if(it == mForces.end()) return nullptr;

        return it->second();

    }
}
