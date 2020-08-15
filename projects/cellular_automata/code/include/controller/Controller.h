///////////////////////////////////////////////////////////
//  Controller.h
//  Implementation of the Class Controller
//  Created on:      07-Apr-2016 10:59:25
//  Author: Michell Mueller
///////////////////////////////////////////////////////////

#ifndef CONTROLLER_H
#define CONTROLLER_H

#pragma once
#include <string>
#include <iostream>
#include "controller/Settings.h"
#include "view/View.h"
#include "dataAccess/IData.h"
#include "interpreter/IInterpreter.h"
#include "model/Rules.h" //Speed Test
#include "darwinEngine/NuklearDarwin.h"
#include <vector>

using namespace std;

/**
* Controller-Klasse:
* Die Controller-Klasse ist der Einstiegspunkt der Simulationsumgebung und steuert zugleich den weiteren Programmablauf.
* Durch die init-Methode werden alle Simulationskomponenten anhand der XML-Datei initialisiert.
* Anschließend sorgt sich der Controller um den Ablauf der Simulationsschritte, indem er mit dem Interpreter und den View-Komponenten kommuniziert.
*/
class Controller
{
private:
	/**
	* Konstruktor:
	* Privater Konstruktor, da die Klasse "Controller" als Singleton modelliert ist.
	*/
	Controller();

	/**
	* Die Instanz der Singleton-Klasse "Controller".
	*/
	static Controller* instance;

	/**
	* Die Instanz der Klasse "Settings" mit den Simulationsparametern.
	*/
	Settings* settings;

	/**
	* Die Instanz des Interpreters, der fuer die Logik verantwortlich ist.
	*/
	IInterpreter* interpreter;

	/**
	* Die aktuelle Anzahl an durchgefuehrten Simulationsschritten.
	*/
	int stepCount;

	/**
	* Die verschiedenen Darwin Engines die aktiv sein koennen
	*/
	vector<IDarwinEngine*>* darwins;


	/**
	* Die Instanz des DataAccess
	*/
	IData* data;

public:
	/**
	* Destruktor.
	*/
	~Controller();

	/**
	* Liefert die Controller-Singleton-Instanz zurueck.
	* @return Controller* Controller-Instanz
	*/
    static Controller* getInstance();

	/**
	* Fuehrt einen Simulationsschritt aus.
	*/
    void doStep();

	/**
	* Fuehrt die definierte Anzahl (XML-Datei) an Simulationsschritten aus.
	*/
    void doSteps();

	/**
	* Liest die Config-File und gibt die vorhandenen Settings aus.
	* @param const char* Pfad zur XML-Datei
	* @return bool Einlesen erfolgreich/fehlgeschlagen
	*/
	static bool viewConfig(const char* file);

	/**
	* Einstiegspunkt der Initialisierung der Simulationsumgebung.
	* @param const char* Pfad zur XML-Datei
	* @param const char* Gewuenschte SettingID aus XML-Datei
	* @return bool Init erfolgreich/fehlgeschlagen
	*/
    static bool init(const char* file, const char* settingID);

	/**
	* Liefert die aktuelle Anzahl an durchgefuehrten Simulationsschritten zurueck.
	* @return int StepCount
	*/
    int getStepCount();

	/**
	* Liefert die Settings-Instanz zurueck.
	* @return Settings* Setting-Instanz
	*/
    Settings* getSettings();
};

#endif // CONTROLLER_H