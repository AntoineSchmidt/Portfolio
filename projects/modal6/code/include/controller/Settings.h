///////////////////////////////////////////////////////////
//  Settings.h
//  Implementation of the Class Settings
//  Created on:      07-Apr-2016 10:59:27
//  Author: Michell Mueller
///////////////////////////////////////////////////////////

#ifndef SETTINGS_H
#define SETTINGS_H

#pragma once
#include <string>
#include <iostream>
#include "library/tinyxml2.h"

using namespace std;

/**
* Settings-Klasse:
* Die Settings-Klasse dient dazu die variablen Parameter einer Simulation vorzugeben. Dazu liest sie den gewuenschten Settings-Teil anhand der Settings-ID aus der XML aus.
* Die dort in der XML vorgegebenen Parameter werden dann in den jeweiligen Variablen der Klasse gespeichert.
* Anschließend wird diese Klasse als zentraler Zugriffspunkt der anderen Komponenten verwendet, damit diese wissen, welche Vorgaben zur Simulation bestehen.
*/
class Settings
{
private:
	/**
	* Definiert die Anzahl an Steps, die in dieser Simulation durchgefuehrt werden sollen.
	*/
	int stepCount;

	/**
	* Name des Settings zur gewaehlten Setting-ID.
	*/
	string settingsName;

	/**
	* Gibt an, welche ViewID in der Simulation verwendet wird.
	*/
	const char* viewID;

	/**
	* Gibt an, welche worldID in der Simulation verwendet wird.
	*/
	const char* worldID;

	/**
	* Gibt an, welche Rules in der Simulation verwendet wird.
	*/
	const char* rulesID;

	/**
	* Gibt an, welche DarwinEngines in der Simulation verwendet werden.
	*/
	const char* darwinEnginesID;

	/**
	* Tagname der World aus der XML-Datei.
	*/
	static const string worldTag;

	/**
	* Tagname der View aus der XML-Datei.
	*/
	static const string viewTag;

	/**
	* Tagname der Rules aus der XML-Datei.
	*/
	static const string rulesTag;

	/**
	* Tagname der DarwinEngines aus der XML-Datei.
	*/
	static const string darwinEnginesTag;

	/**
	* Tagname der Steps aus der XML-Datei.
	*/
	static const string stepTag;

	/**
	* Tagname des DebugOutputs aus der XML-Datei.
	*/
	static const string debugOutputTag;

	/**
	* Attributname des SettingsName aus der XML-Datei.
	*/
	static const char* settingsNameTag;

public:
	/**
	* Konstruktor.
	*/
    Settings();

	/**
	* Destruktor.
	*/
    ~Settings();

	/**
	* Parst den Setting-Abschnitt aus der XML-Datei und setzt die Parameter anhand der Werte, die in der XML-Datei definiert sind.
	* @param tinyxml2::XMLElement* XML-Node mit den Setting-Tags
	* @return Settings* Geparste Settings-Instanz mit den Simulationsparametern
	*/
    static Settings* parse(const tinyxml2::XMLElement* node);

	/**
	* Liefert die Anzahl an zu simulierenden Steps zurueck.
	* @return int Anzahl Steps, die zu simulieren sind
	*/
    int getStepCount();

	/**
	* Liefert die ViewID zurueck.
	* @return const char* ViewID
	*/
    const char* getViewID();

	/**
	* Liefert die WorldID zurueck.
	* @return const char* WorldID
	*/
    const char* getWorldID();

	/**
	* Liefert die RulesID zurueck.
	* @return const char* RulesID
	*/
    const char* getRulesID();

	/**
	* Liefert die DarwinEnginesID zurueck.
	* @return const char* DarwinEnginesID
	*/
	const char* getDarwinEnginesID();

	/**
	* Liefert den definierten Namen (XML-Datei) der Settings-ID zurueck.
	* @return const string SettingsName
	*/
	const string getSettingsName();
};

#endif // SETTINGS_H