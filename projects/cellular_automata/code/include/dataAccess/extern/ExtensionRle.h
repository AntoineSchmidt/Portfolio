///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef EXTENSIONRLE_H
#define EXTENSIONRLE_H

#pragma once

#include <string>
#include <fstream>
#include "dataAccess/extern/File.h"
#include "library/tinyxml2.h"

using namespace std;

/**
* Interface Implementierung fuer das Parsen von Externen Files mit .rle Extension
*/
class ExtensionRle : public File
{
public:
    ExtensionRle(const char* filen);
    ~ExtensionRle();

	bool buildWorldNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc);
};

#endif // EXTENSIONRLE_H