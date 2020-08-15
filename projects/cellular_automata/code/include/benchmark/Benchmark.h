///////////////////////////////////////////////////////////
//  Created on:      21-Apr-2016 14:43:27
//  Original author: Michell Mueller
///////////////////////////////////////////////////////////

#pragma once
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <boost/chrono.hpp>
#include <boost/regex.hpp>
#include "library/utils.h"

using namespace std;

/**
* Benchmark-Klasse:
* Die Benchmark-Klasse dient zur Ermittlung von Laufzeiten beliebiger Code-Abschnitte.
* Die start-Methode beginnt das Stoppen der Zeit und die stop-Methode beendet die Zeitstoppung.
* Anschließend werden alle Messungen im aktuellen Projektverzeichnis in einer eindeutigen Textdatei ausgegeben.
* Dadurch laesst sich die Performance unterschiedlicher Implementierungen messen oder Engpaesse an nicht optimalen Code erkennen.
*/
class Benchmark
{
private:
	/**
	* Konstruktor:
	* Privater Konstruktor, da die Klasse "Benchmark" als Singleton modelliert ist.
	*/
	Benchmark();

	/**
	* Die Instanz der Singleton-Klasse "Benchmark".
	*/
	static Benchmark* instance;

	/**
	* Map des Benchmarks.
	* Enthaelt den Operationen-Namen (string) als Key und die Startzeit der zu messenden Operation als Schluesselwert.
	*/
	map<string, boost::chrono::time_point<boost::chrono::steady_clock> > benchmarkMap;

	/**
	* String-Variable zur Bestimmung des Dateipfads und -namens.
	*/
	string file;

	/**
	* Stream fuer die notwendigen Dateioperationen.
	*/
	ofstream stream;

public:
	/**
	* Destruktor.
	*/
	~Benchmark();

	/**
	* Liefert die Benchmark-Singleton-Instanz zurueck.
	* @return Benchmark* Benchmark-Instanz
	*/
	static Benchmark* getInstance();

	/**
	* Startet die Benchmark-Messung zum uebergebenden OperationName.
	* @param string Name unter dem die Messung laufen soll
	*/
	void start(string operationName);

	// stopt die Benchmark-Messung mit dem angegebenen OperationName
	/**
	* Stoppt die Benchmark-Messung zum uebergebenden OperationName.
	* @param string Name unter dem die Messung gelaufen ist
	* @param string Zusaetzlicher Kommentar, der zur Messung protokolliert werden soll
	*/
	void stop(string operationName, string comment = "");
};