///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef XMLHANDLER_H
#define XMLHANDLER_H

#pragma once

#include <vector>
#include <list>
#include "dataAccess/IData.h"
#include "library/tinyxml2.h"
#include "view/View.h"
#include "controller/Settings.h"

using namespace std;

/**
* Interface Implementierung von IData fuer das Daten Lesen aus einer Xml Datei
*/
class XmlHandler : public IData
{
public:
    /**
    * Der Konstruktor
    */
    XmlHandler();

    /**
    * Der Destruktor
    */
    virtual ~XmlHandler();

    /**
    * Laden des XML-Dokuments und initialisieren der "RootNode" Variable
    * @param const char* Pfad zur Xml Datei
    */
    bool openConnection(const char* Pfad);

    /**
    * Schliessen der Xml Datei
    */
    void closeConnection();

    /**
    * Holen des Rules Node mit Passender Id aus der Xml Datei und aufrufen der Rules Parse Methode
    * @param const char* Rules Node ID
    */
    void loadRules(const char* id);

    /**
    * Speichert Daten der Rules Klasse in einen neuen Rules Node mit Id in die Xml Datei
    * @param const char* ID fuer neue Rules Node
    */
    bool saveRules(const char* id = NULL);

    /**
    * Holen des World Node mit Passender Id aus der Xml Datei und aufrufen der World Parse Methode
    * @param const char* World Node ID
    */
    void loadWorld(const char* id);

    /**
    * Speichert aktuellen Stand der Welt neuen World Node mit Id in die Xml Datei
    * @param const char* Neue World ID
    */
    bool saveWorld(const char* id = NULL);

    /**
    * Holen der View Node mit Passender Id aus der Xml Datei und aufrufen der View Parse Methode
    * @param const char* View Node ID
    */
    void loadView(const char* viewID);

    /**
    * Holen der Settings Node mit Passender Id aus der Xml Datei und aufrufen der Settings Parse Methode
    * @param const char* View Node ID
    * @return Settings* geparste Settings Klasse
    */
    Settings* getSettings(const char* Id);

	/**
	* Gibt alle vorhandenen Settings aus der Xml Datei aus
	*/
	void viewAllSettings();

	/**
    * Holen der Darwin Engines mit Passender Id aus der Xml Datei und aufrufen der Darwin Engine Parse Methode
    * @param const char* Darwin Engine Node ID
    * @return Settings* geparste Darwin Engines Klasse
    */
	vector<IDarwinEngine*>* getDarwinEngines(const char* Id);

private:
    /**
    * Der Xml Document Handle
    */
    tinyxml2::XMLDocument OpenedDocument;

    /**
    * Der Root Node des Xml Documents
    */
    tinyxml2::XMLElement* RootNode;

    /**
    * Pfad zur gefoeffneten Xml Datei
    */
    const char* OpenedFilePath;

    /**
    * World Node Id der geladenen World
    */
    const char* LoadedWorldId;

    /**
    * Rules Node Id der geladenen Rules
    */
    const char* LoadedRulesId;
};

#endif // XMLHANDLER_H