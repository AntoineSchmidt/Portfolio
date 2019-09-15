///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include "dataAccess/extern/File.h"
#include "controller/Console.h"

File::File(const char* filen)
{
	filename = filen;
}

File::~File()
{
	filestream.close();
}

bool File::openFile()
{
	try
	{
		filestream.open(filename, ios::in);
		return true;
	}
	catch(...)
	{
		Console::Stream << "Error Opening File: " << filename << endl;
		Console::writeAsException();
		return false;
	}
}

bool File::closeFile()
{
	try
	{
		filestream.close();
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool File::nextLine(string &line)
{
	if(getline(filestream, line))
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool File::checkExtension(const string ext)
{
	string file(filename);
	if(file.find(ext, file.length() - ext.length()) != string::npos)
	{
		return true;
	}
	else
	{
		Console::Stream << "Bad File Ending: " << *filename << " expected: " << ext << endl;
		Console::writeAsError();
		return false;
	}
}

bool File::buildRulesNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc)
{
	Console::Stream << "Rules not Supported in this FileType" << endl;
	Console::writeAsError();
	return false;
}

bool File::buildWorldNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc)
{
	Console::Stream << "World not Supported in this FileType" << endl;
	Console::writeAsError();
	return false;
}