///////////////////////////////////////////////////////////
//  Original author: Bonifaz Stuhr 
///////////////////////////////////////////////////////////

#pragma once
#include <string>
#include <vector>
#include <model/World.h>
#include <fstream>


using namespace std;

/**
*Erleichtert das Debuging und das suchen nach Fehlern in der Welt
*Informationen werden in Files ausgegeben.
**/
class StepTracer
{
private:
	//PoinNodes die Beobachtet werden sollem
	map<const uint_fast64_t,PointNode*> observedNodes;

	//Altern Werte für späteren vergleich
	map<const uint_fast64_t,PointNode> oldNodes;

	//Stream fuer Dateioperationen
	ofstream fstream;

	//Singelton
	static StepTracer* instance;

	//Filename
	string file;

	//Letzte Ausführung der printChanges Methode
	int oldStep;

public:
	~StepTracer();

	/**
	*Melde eine PointNode an, die Observiert werden soll.
	*@param PointNode*, der Pointer auf die zu observierende PointNode:
	*/
	void observePointNode(PointNode* Node);

	/**
	*Melde eine PointNode ab, die bislang Observiert wurde.
	*@param PointNode*, der Pointer auf die abzumeldende PointNode.
	*/
	void removePointNode(PointNode* Node);

	/**
	*SChreibe alle Aenderungen in ein File, die seit dem letzten Aufruf von printChances() geschehen sind.
	**/
	void printChanges();

	/**
	*Methode um den Pointer auf Singleton Instanz zu erhalten.
	*@return static WorldCheck*, der Pointer auf des Singelton Objekt.
	*/
	static StepTracer* getInstance();

private:
	/**
	*Konstruktor ist private da die Klasse Singelton ist.
	**/
	StepTracer();

	/**
	*Schreibt alle Informationen über die Node in ein File
	*@param PointNode*, Pointer auf die PointNode die in File geschrieben werden soll
	**/
	void printNodeToFile(PointNode* node);

	/**
	*Positions Vectoren wird in einen Ausgabestring verwandelt.
	*@param vector<uint_fast32_t>*, der Positionsvector.
	*@return string, der string zur Ausgabe.
	**/
	string posToString(vector<uint_fast32_t>* pos);

};