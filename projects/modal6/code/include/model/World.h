///////////////////////////////////////////////////////////
//  Alternativ World Data Struktur
//  Created on:      14-Apr-2016 16:59:28
//  Author:          Bonifaz Stuhr, Antoine Schmidt und Tobias Eigler
///////////////////////////////////////////////////////////

#ifndef WORLD_H
#define WORLD_H

#pragma once

#include <vector>
#include <map>
#include "model/IWorld.h"
#include "model/PointNode.h"
#include "library/tinyxml2.h"

using namespace std;

class World : public IWorld
{
private:
	/*
	* Konstruktor von World.
	* Privat, da die Klasse World Singleton ist.
	*/
	World();

	/*
	* Vector Pointer der PointNodes Pointer enthaelt.
	* Stellt das Chunk dar, welches von Interpretern(zur effizienten Berechnung) genutzt werden kann.
	*/
	vector<PointNode*>* chunk;

	/*
	* Vector Pointer der PointNodes Pointer enthaelt.
	* Stellt die World dar, die alle "Punkte" enthaelt.
	*/
	vector<PointNode*>* pointNodesMatrix;

	/*
	* Groeße der Welt. Speichert die jeweiligen Werte der Deimensionen.
	* Bsp 2D: x=200; y=300;
	*/
	vector<uint_fast32_t>* size;

	//Singelton Instance
	static World* instance;

	//Boolwert fuer die Grenzen der Welt. 
	bool bhasBorder;

public:
    /*
	* Erzeugt die Singleton Instanz der World Klasse und parst die Daten aus dem XML-Dokument.
	*@param const tinyxml2::XMLElement*, node World XML-Knoten aus dem XML-File.
    */
    static void parse(const tinyxml2::XMLElement* node);

    /*
	* Methode um die Singleton Instanz zu bekommen.
	* @return World*, Liefert Pointer auf die Instanz der Worldklasse.
    */
	static World* getInstance();

	/**
	* Gibt einen Pointer auf den Chunk der Welt, der alle einen Wert enthaltenden und relevanten PointNodes enthaelt, zurueck.
	* @return vector<PointNode*>*, der Chunk-Pointer.
	*/
    vector<PointNode*>* getChunk() const;

	/**
	* Liefert einen Pointer auf die PointNodesMatrix, die alle PointNodes der Welt enthaelt.
	* @return vector<PointNode*>*, der PointNodeMatrix-Pointer.
	*/
    vector<PointNode*>* getPointNodesMatrix() const;

	/**
	* Liefert einen Pointer auf einen Vektor mit Integern, der die Groeße der Welt darstellt.
	* vector<uint_fast32_t>*, der Size-Pointer.
	*/
    vector<uint_fast32_t>* getSize() const;


	/**
	* Liefert einen uint_fast8_t Integern, der die Dimension der Welt darstellt.
	* uint_fast8_t, die Dimension.
	*/
	uint_fast8_t getDimensions() const;

	/*
	* Liefert einen Boolwert, der true ist wenn die Welt grenzen hat und false wenn sie keine besitzt.
	* bool, hat die Welt Grenzen?
	* @return bool, liefert True/False je nachdem ob die Welt grenzen besitzt.
	*/
	bool hasBorder() const;

    //Destruktor
    ~World();

};

#endif // WORLD_H