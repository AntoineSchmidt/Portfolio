///////////////////////////////////////////////////////////
//  Interpreter der sowohl PointNode Changed als auch Chunk Verwendet
//  Author: Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef COMBIINTERPRETER_H
#define COMBIINTERPRETER_H

#pragma once

#include <vector>
#include "IInterpreter.h"
#include "model/World.h"
#include "model/PointNode.h"

using namespace std;

/**
*Interpreter wenden die Rules auf die World an und fuehren einen Zeitschritt aus. Die Welt ändert sich.
*/
class CombiInterpreter : public IInterpreter
{
private:

public:
	/**
	*Kunstruktor.
	**/
    CombiInterpreter();

	/**
	*Destruktor.
	**/
    ~CombiInterpreter();

	/**
	* Wendet die Rules auf die World an und fuehrt einen Zeitschritt aus. Die Welt ändert sich.
	* Itteriert über den gesammte Chunk und pflegt diesen, beachtet nur die PointNodes der Welt die sich im letzten Zeitschritt geaendert haben.
	*/
    void update();

};

#endif // COMBIINTERPRETER_H