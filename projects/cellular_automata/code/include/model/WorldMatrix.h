///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt, Michell Mueller, Michael Warschat, Bonifaz Stuhr, Tobias Eigler
///////////////////////////////////////////////////////////

#ifndef WORLDMATRIX_H
#define WORLDMATRIX_H

#pragma once

#include <vector>
#include <cstdint>
#include <map>
#include "model/PointNode.h"

/**
* Ist zustaendig fuer die Vernetzung der World Matrix
*/
class WorldMatrix
{
public:
    /**
    * Vereinfachter und klarer Vernetzungs Algorithmus
    * @param vector<PointNode*>* Alle PointNodes aus denen die World Matrix besteht
    * @param vector<uint_fast32_t>* Die groesse der Welt
    * @param bool true wenn die Welt grenzen hat ohne grenzen wird die Welt mit der gegenueberliegenden Seite vernetzt
    * @return bool true wenn alles passt
    */
	static bool connect(const vector<PointNode*>* Content, const vector<uint_fast32_t>* Size, const bool Borders);

	/**
    * Im Team entwickelter Vernetzungs Algorithmus
    * Vermutlich schneller, Border nicht funktionsfaehig, die Reihenfolge der PointNodes ist entscheident
    * @param vector<PointNode*>* Alle PointNodes aus denen die World Matrix besteht
    * @param vector<uint_fast32_t>* Die groesse der Welt
    * @param bool true wenn die Welt grenzen hat ohne grenzen wird die Welt mit der gegenueberliegenden Seite vernetzt
    * @return bool true wenn alles passt
    */
	static bool connectQuick(const vector<PointNode*>* Content, const vector<uint_fast32_t>* Size, const bool Borders);

private:
    /**
    * Private Konstruktor da Statische Klasse
    */
	WorldMatrix();

};

#endif // WORLDMATRIX_H