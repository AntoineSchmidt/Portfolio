///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef IDATA_H
#define IDATA_H

#pragma once

#include <string>
#include <set>
#include <vector>

class IDarwinEngine;
class Settings;

/**
* Interface Klasse fuer das Daten Lesen und aufrufen der gewuenschten parse Methoden
*/
using namespace std;
class IData
{
public:
    /**
    * Oeffnet die Verbindung zur Datenquelle
    * @param const char* Werte fuer den Verbindungsaufbau
    */
    virtual bool openConnection(const char* Werte) = 0;

    /**
    * Schliesst die Verbindung zur Datenquelle
    */
    virtual void closeConnection() = 0;

    /**
    * Laed Rules Datensatz mit passender ID und ruft parse funktion in Rules auf
    * @param const char* ID des Rules
    */
    virtual void loadRules(const char* id) = 0;

    /**
    * Speichert Rules Datensatz unter ID
    * @param const char* ID des Rules
    */
    virtual bool saveRules(const char* id = NULL) = 0;

    /**
    * Laed World Datensatz mit passender ID und ruft parse funktion in World auf
    * @param const char* ID des World
    */
    virtual void loadWorld(const char* id) = 0;

    /**
    * Speichert World Datensatz unter ID
    * @param const char* ID der World
    */
    virtual bool saveWorld(const char* id = NULL) = 0;

    /**
    * Laed View Datensatz mit passender ID und ruft parse funktion in View auf
    * @param const char* ID der View
    */
    virtual void loadView(const char* id) = 0;

    /**
    * Laed Rules Datensatz mit passender ID und ruft parse funktion in Rules auf
    * @param const char* ID des Rules
    */
    virtual Settings* getSettings(const char* id) = 0;

    /**
	* Gibt alle vorhandenen Settings aus der Xml Datei aus
	*/
	virtual void viewAllSettings() = 0;

	/**
	* Laed DarwinEngines mit passender ID und ruft parse funktion in entsprechender DarwinEngine auf
	* @param const char* ID der DarwinEngine
	* @return vector<IDarwinEngine*>* Vector der die im Xml File angegeben DarwinEngines enthält
	*/
	virtual vector<IDarwinEngine*>* getDarwinEngines(const char* Id) = 0;

protected:
    /**
    * Der geschuetzte Konstruktor da Klasse als Interface Implementiert
    */
    IData() {}

};

#endif // IDATA_H