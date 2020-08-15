///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef FILE_H
#define FILE_H

#pragma once

#include <string>
#include <fstream>

namespace tinyxml2
{
class XMLElement;
class XMLDocument;
}


using namespace std;

/**
* Interface fuer das Parsen von Externen Files
*/
class File
{
public:
    /**
    * Parsed die Externe Datei hinsichtlich Rules und baut das uebergebene XMLElement auf
    * @param XMLElement* das aufzubauende XMLElement
    * @param XMLDocument* das dazugehoerige XMLDocument
    * @return bool true wenn alles funktioniert hat
    */
	virtual bool buildRulesNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc);

    /**
    * Parsed die Externe Datei hinsichtlich World und baut das uebergebene XMLElement auf
    * @param XMLElement* das aufzubauende XMLElement
    * @param XMLDocument* das dazugehoerige XMLDocument
    * @return bool true wenn alles funktioniert hat
    */
	virtual bool buildWorldNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc);

protected:
    /**
    * Der Konstruktor
    * @param const char* die zu parsende Datei
    */
	File(const char* filen);

	/**
	* Der Dekonstruktor
	*/
	~File();

	/**
	* Oeffnet die in filename gestzte datei im filestream
	* @return true wenn dei datei geoffnet werden konnte
	*/
	bool openFile();

    /**
	* schliesst den filestream
	* @return true wenn dei datei geschlossen werden konnte
	*/
	bool closeFile();

	/**
	* Liest die naechste zeile aus dem filestream
	* @param string in diesen string wird die naechste zeile geschrieben
	* @return true wenn die naechste zeile gelesen werden konnte
	*/
	bool nextLine(string &line);

	/**
	* Die zu parsende Datei
	*/
	const char* filename;

	/**
	* Prueft auf richtigen file typ von filename
	* @param string erwarteter file typ
	* @return true wenn file typ strimmt
	*/
	bool checkExtension(const string ext);

private:
    /**
    * Der Filestream der zu parsenden Datei
    */
	ifstream filestream;
};

#endif // FILE_H