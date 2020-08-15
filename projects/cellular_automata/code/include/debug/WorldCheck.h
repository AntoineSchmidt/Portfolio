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
*Erleichtert das Debuging und das suchen nach Fehlern in der Welt.
*Informationen werden in Files ausgegeben.
**/
class WorldCheck
{
private:
	//Matrix des vorherigen Steps
	vector<PointNode> oldMatrix;

	//Chunk des vorherigen Steps
	vector<PointNode> oldChunk;

	//Alte Dimension
	uint_fast8_t oldDimension;

	//AlteGröße
	vector<uint_fast32_t> oldSize;

	//Stream fuer Dateioperationen
	ofstream fstream;

	//Singelton
	static WorldCheck* instance;

	//Filename
	string file;

public:
	~WorldCheck();

	/**
	*Ueberprueft die gesammte Welt nach Vernetzungsfehler und Koordinatenfehler.
	*Schreibt Fehler in ein File, liefert true wenn keine Fehler in der Welt sind.
	*@param string, Code für Filename Index.
	*@return bool, keine Fehler in der Welt?
	*/
	bool checkWorld(string fileIndex);

	/**
	*Ueberprueft die gesammte Matrix nach Vernetzungsfehler und Koordinatenfehler.
	*Schreibt Fehler in ein File, liefert true wenn keine Fehler in der Welt sind.
	*@param string, Code für Filename Index.
	*@return bool, keine Fehler in der Matrix?
	**/
	bool checkMatrix(string fileIndex);

	/**
	*Ueberprueft den gesammten Chunk nach Vernetzungsfehler und Koordinatenfehler
	*Schreibt Fehler in ein File, liefert true wenn keine Fehler in der Welt sind.
	*@param string, Code für Filename Index.
	*@return bool, keine Fehler im Chunk?
	*/
	bool checkChunk(string fileIndex);

	/**
	*Vergleicht die neue Welt mit der Welt des vorherigen Steps und gibt Unterschiede aus.
	*Gibt die Anzahl der Unterschiede zurueck und schreibt die konkreten Unteschiede in ein File.
	*@param string, Code für Filename Index.
	*@return uint_fast64_t, Anzal der Unterschiede.
	*/
	uint_fast64_t compareWorldWithOld(string fileIndex);

	/**
	*Vergleicht die neue MAtrix mit der Matrix des vorherigen Steps und gibt Unterschiede aus.
	*Gibt die Anzahl der Unterschiede zurueck und schreibt die konkreten Unteschiede in ein File.
	*@param string, Code für Filename Index.
	*@return uint_fast64_t, Anzal der Unterschiede.
	*/
	uint_fast64_t compareMatrixWithOld(string fileIndex);

	/**
	*Vergleicht den neuen Chunk mit dem Chunk des vorherigen Steps und gibt Unterschiede aus.
	*Gibt die Anzahl der Unterschiede zurueck und schreibt die konkreten Unteschiede in ein File.
	*@param string, Code für Filename Index.
	*@return uint_fast64_t, Anzal der Unterschiede.
	*/
	uint_fast64_t compareChunkWithOld(string fileIndex);

	/**
	*Schreibt alle Informationen ueber die Welt in ein File.
	*@param string, Code für Filename Index.
	**/
	void printWorld(string fileIndex);

	/**
	*Schreibt alle Informationen ueber die Matrix in ein File.
	*@param string, Code für Filename Index.
	**/
	void printMatrix(string fileIndex);

	/**
	*Schreibt alle Informationen ueber den Chunk in ein File.
	*@param string, Code für Filename Index.
	**/
	void printChunk(string fileIndex);

	/**
	*Ueberprueft eine PointNode ob sie im Kontext der Welt richtige Werte besitzt.
	*Gibt true zurück wenn sie richtig liegt
	*@return bool, sind die Werte korrekt? 
	*/
	bool checkPointNode(PointNode* node, string& error);

	/**
	*Methode um den Pointer auf Singleton Instanz zu erhalten.
	*@return static WorldCheck*, der Pointer auf des Singelton Objekt.
	*/
	static WorldCheck* getInstance();

private:
	/**
	*Konstruktor ist private da die Klasse Singelton ist.
	**/
	WorldCheck();

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