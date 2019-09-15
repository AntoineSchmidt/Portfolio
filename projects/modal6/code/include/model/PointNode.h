///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:27
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef POINTNODE_H
#define POINTNODE_H

#pragma once

#include <vector>
#include <cstdint>
#include "model/Point.h"
#include "controller/Controller.h"

/**
* Wrapper Klasse fuer Point Klasse die es ermoeglicht eine Point Matrix auzuspanne
*/
class PointNode
{
private:
	/**
	* Veraendert den Changed Wert und NeighbourChanged Wert der PointNode und seiner Nachbarn
	* @param bool der zu Setzende tempChanged Wert fuer die PointNode
	* @param bool wenn true werden die NeighbourChanged Werte der Nachbarn gesetzt
	* @param bool wenn true wird direkt der Changed Wert geaendert
	* @param bool wenn true wird bei den PointNode Nachbarn direkt changed gesetzt
	* @param uint_fast8_t die Rekursive Tiefe der Funktion
	*/
	void setChanged(bool value, bool neighboursToo = false, bool instant = false, bool InstantNeighbour = false, uint_fast8_t deep = 0);

	/**
	* Der Punkt des PointNodes der den Punkt Wert haelt
	*/
	Point* myPoint;

	/**
	* Die PointNode Position in der World Matrix
	*/
	vector<uint_fast32_t>* position;

	/**
	* Die Nachbarn des PointNodes
	*/
	vector<PointNode*>* myNeighbours;

	/**
	* Der Cache vector fuer die Nachbar Werte
	*/
	vector<uint_fast16_t> valueWithNeighbours;

	/**
	* Updated Changed auf basis von Neighbour Changed und Temp Changed
	*/
	void updateChanged();

	/**
	* Der Punkt Wert hat sich geaendert
	*/
	bool changed;

	/**
	* Der Punkt Wert hat sich bei einem der Nachbarn geaendert
	*/
	bool neighbourChanged;

	/**
	* Der Punkt Wert hat sich in diesem Schritt geaendert
	*/
	bool tempChanged;

	/**
	* Der Globale Private geschuetzte Id Counter haelt die Id fuer den naechsten PointNode
	*/
	static uint_fast64_t idCounter;

	/**
	* Die eindeutige ID des PoitNode
	*/
	const uint_fast64_t  id;

	/**
	* Sagt aus ob dieser PointNode im WorldChunk ist
	*/
	bool inChunk;

public:
    /**
    * Leere Konstruktor
    */
    PointNode();

    /**
    * Konstruktor mit Position
    * @param vector<uint_fast32_t>* die Position des PointNodes in der Matrix
    */
    PointNode(vector<uint_fast32_t>* Position);

    /**
    * Konstruktor mit Position und seinen Nachbarn
    * @param vector<uint_fast32_t>* die Position des PointNodes in der Matrix
    * @param vector<PointNode*>* die Nachbarn der PointNode
    */
    PointNode(vector<uint_fast32_t>* Position, vector<PointNode*>* MyNeighbours);

    /**
    * Konstruktor mit Position und seinen Nachbarn und den Wert des Punktes
    * @param vector<uint_fast32_t>* die Position des PointNodes in der Matrix
    * @param vector<PointNode*>* die Nachbarn der PointNode
    * @param uint_fast16_t der Wert des Punktes der PointNode
    */
    PointNode(vector<uint_fast32_t>* Position, vector<PointNode*>* MyNeighbours, const uint_fast16_t Value);

    /**
    * Setzt den Temporaeren Punkt Wert
    * @param uint_fast16_t der Temporaere Wert
    */
    void setTempValue(const uint_fast16_t Value);

    /**
    * Gibt den Temporaeren Punkt Wert zurueck
    * @return uint_fast16_t der Temporaere Wert
    */
	uint_fast16_t getTempValue() const;

    /**
    * Setzt den Aktiven Punkt Wert
    * @param uint_fast16_t der Aktive Wert
    */
    void setActiveValue(const uint_fast16_t Value);

    /**
    * Gibt den Aktive Punkt Wert zurueck
    * @return uint_fast16_t der Aktive Wert
    */
	uint_fast16_t getActiveValue() const;

	/**
	* Setzte die Position der PointNode in der Matrix
	* @param vector<uint_fast32_t>* die Position
	*/
    void setPosition(vector<uint_fast32_t>* Position);

    /**
	* Gibt die Position der PointNode in der Matrix zureuck
	* @return vector<uint_fast32_t>* die Position
	*/
    vector<uint_fast32_t>* getPosition() const;

    /**
	* Gibt die Werte des PointNodes und seiner Nachbarn zureuck die Reihenfolge wird durck den Order Vektor in Rules definiert
	* @return vector<uint_fast16_t>* die Rule Key Value
	*/
    vector<uint_fast16_t>* getValueWithNeighbours();

    /**
    * Updatet den PointNode den Temporaeren Punkt Wert wird auf Aktiv gesetzt und die PointNode bool Werte verarbeitet
    * @return bool true wenn sich der Punkt Wert von EmptyValue auf einen anderen Wert geaendert hat
    */
    bool update();

    /**
    * Rueft gleich Namige Funktion bei Point auf
    * @param uint_fast16_t der aktuelle Wert
    * @return bool true wenn der Wert von value auf anderen wechselt
    */
	bool changingOnUpdate(const uint_fast16_t value) const;

    /**
    * Gibt den Werte der bool Changed zurueck ist true wenn sich beim letzten Step der Punkt Wert geaendert hat oder einer seiner Nachbarn
    * @return bool gibt Changed zurueck
    */
    bool getChanged() const;

    /**
    * Setzt den bool InChunk fuer Geschwindigkeits optimiertes festellen ob PointNode bereits in World Chunk
    * @param bool der zu setzende Wert
    */
    void setInChunk(bool Value);

    /**
    * True wenn der PointNode bereits in World Chunk
    * @return bool InChunk
    */
    bool getInChunk() const;

    /**
    * Gibt die ID der PointNode zurueck fuer eindeutige identifizierung
    * @return uint_fast64_t ID
    */
	const uint_fast64_t getId();

	/**
	* Setzt einen Nachbarn der PointNode
	* @param uint_fast16_t die Position des Nachbarn im Nachbar Vektor
	* @param PointNode* der entsprechende Nachbar
	*/
    void setNeighbour(const uint_fast16_t position, PointNode* me);

    /**
    * Gibt einen Nachbarn der PointNode zurueck
    * @param uint_fast16_t die Position des Nachbars im Nachbar Vektor
    * @return PointNode* der gefunden Nachbar
    */
    PointNode* getNeighbour(const uint_fast16_t position) const;

    /**
    * Gibt alle Nachbarn der PointNode zurueck
    * @return vector<PointNode*>* der Nachbarn Vektor der PointNode
    */
    vector<PointNode*>* getNeighbours() const;

};
bool operator==(PointNode& n1, PointNode& n2);
bool operator!=(PointNode& n1, PointNode& n2);

#endif // POINTNODE_H