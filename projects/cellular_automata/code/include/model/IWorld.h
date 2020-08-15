///////////////////////////////////////////////////////////
//  Created on:      19-Apr-2016
//  Original author: Tobias Eigler, Bonifaz Stuhr
///////////////////////////////////////////////////////////

#ifndef IWORLD_H
#define IWORLD_H

#pragma once

#include <iostream>
#include <vector>

using namespace std;

class PointNode;

/**
*Schnittstelle für Welten. Welten enthalten die in der Eingabe spezifizierten Punkte in PointNode Form.
**/
class IWorld
{
public:
	/**
	* Gibt einen Pointer auf den Chunk der Welt, der alle einen Wert enthaltenden und relevanten PointNodes enthaelt, zurueck.
	* getChunk() ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @return vector<PointNode*>*, der Chunk-Pointer.
	*/
    virtual vector<PointNode*>* getChunk() const = 0;

	/**
	* Liefert einen Pointer auf die PointNodesMatrix, die alle PointNodes der Welt enthaelt.
	* getPointNodeMatrix() ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @return vector<PointNode*>*, der PointNodeMatrix-Pointer.
	*/
    virtual vector<PointNode*>* getPointNodesMatrix() const = 0;

	/**
	* Liefert einen Pointer auf einen Vektor mit Integern, der die Groeße der Welt darstellt.
	* getSize() ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* vector<uint_fast32_t>*, der Size-Pointer.
	*/
    virtual vector<uint_fast32_t>* getSize() const = 0;

	/**
	* Liefert einen uint_fast8_t Integern, der die Dimension der Welt darstellt.
	* getDimensions() ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* uint_fast8_t, die Dimension.
	*/
    virtual uint_fast8_t getDimensions() const = 0;

	/**
	* Liefert einen Boolwert, der true ist wenn die Welt grenzen hat und false wenn sie keine besitz.
	* hasBorder() ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* bool, hat die Welt Grenzen?
	*/
    virtual bool hasBorder() const = 0;

protected:
	/**
	* Konstruktor
	*/
    IWorld() {}

};

#endif // IWORLD_H