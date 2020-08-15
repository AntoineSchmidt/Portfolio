///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef EXTENSIONRULE_H
#define EXTENSIONRULE_H

#pragma once

#include <string>
#include <fstream>
#include "dataAccess/extern/File.h"
#include "library/tinyxml2.h"

using namespace std;

/**
* Interface Implementierung fuer das Parsen von Externen Files mit .rule Extension
*/
class ExtensionRule : public File
{
public:
    ExtensionRule(const char* filen);
    ~ExtensionRule();

	bool buildRulesNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc);
};

#endif // EXTENSIONRLE_H