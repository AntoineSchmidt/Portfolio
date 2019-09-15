///////////////////////////////////////////////////////////
//  DefaultInterpreter.h
//  Implementation of the Class PointNodeInterpreter
//  Created on:      07-Apr-2016 10:59:26
//  Original author: Bonifaz Stuhr
///////////////////////////////////////////////////////////

#ifndef POINTNODEINTERPRETER_H
#define POINTNODEINTERPRETER_H

#pragma once

#include "IInterpreter.h"
#include <vector>
#include "model/World.h"
#include "model/PointNode.h"

using namespace std;

/**
* Interpreter wenden die Rules auf die World an und fuehren einen Zeitschritt aus. Die Welt ändert sich.
*/
class PointNodeInterpreter : public IInterpreter
{
private:

public:
	/**
	*Kunstruktor.
	**/
    PointNodeInterpreter();

	/**
	*Destruktor.
	**/
    ~PointNodeInterpreter();

	/**
	* Wendet die Rules auf die World an und fuehrt einen Zeitschritt aus. Die Welt ändert sich.
	* Itteriert über den gesammte Chunk und pflegt diesen.
	*/
    void update();

};

#endif // POINTNODEINTERPRETER_H