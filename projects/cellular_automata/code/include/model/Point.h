///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:27
//  Original author: Bonifaz Stuhr & Tobias Eigler
///////////////////////////////////////////////////////////

#ifndef POINT_H
#define POINT_H

#pragma once

#include <cstdint>

using namespace std;

/*
Repraesentiert einen Punkt auf der Welt
*/
class Point
{
friend class PointNode;

public:
	//Konstruktor
	//Per Default "leeren" Punkt erstellen
	Point();
	Point(uint_fast16_t value);
	//Destruktor
	~Point();

	//Liefert einen uint_fast16_t-Wert, der den aktuellen Wert des Punktes beinhaltet.
	uint_fast16_t getActiveValue() const;


	//Ermoeglicht das setzten des temporaeren Wertes.
	void setTempValue(const uint_fast16_t value);
	uint_fast16_t getTempValue() const;

	/**
	* Prueft ob sich TempValue von AktivValue unterscheidet
	* @return bool true wenn sich der Wert unterscheidet
	*/
	bool changingOnUpdate() const;

    /**
    * Prueft auf changingOnUpdate und prueft ob aktueller Wert Value entspricht
    * @param uint_fast16_t der aktuelle Wert
    * @return bool true wenn der Wert von value auf anderen wechselt
    */
	bool changingOnUpdate(const uint_fast16_t value) const;

	//Setzt den temporaeren Wert als aktuellen Wert.
	//True on change from 0
	bool update();

	static const uint_fast16_t emptyValue;

private:
    /*
    "activeValue" beinhaltet den aktuellen Wert des Punktes.
    Dieser Wird durch die Methode update aktuallisiert.
    */
	uint_fast16_t activeValue;

    /*
    "tempValue" beinhaltet den temporaeren Wert des Punktes der durch die Berechung entsteht.
    Variable wird also fuer die Berechnung im Chunk benoetigt und beinhaltet den neuen Wert, der nach der Berechnung in
    "activeValue" geschrieben wird.
    */
    uint_fast16_t tempValue;

    void setAllValue(const uint_fast16_t Value);

};

#endif // POINT_H