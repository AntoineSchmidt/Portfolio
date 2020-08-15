///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Original author: Bonifaz Stuhr
///////////////////////////////////////////////////////////

#ifndef DUMBINTERPRETER_H
#define DUMBINTERPRETER_H

#pragma once

#include "IInterpreter.h"

using namespace std;

/**
*Interpreter wenden die Rules auf die World an und fuehren einen Zeitschritt aus. Die Welt ändert sich.
*/
class DumbInterpreter : public IInterpreter
{
public:
	/**
	*Kunstruktor.
	**/
	DumbInterpreter();

	/**
	*Destruktor.
	**/
	~DumbInterpreter();

	/**
	* Wendet die Rules auf die World an und fuehrt einen Zeitschritt aus. Die Welt ändert sich.
	* Itteriert über die gesammte Welt.
	*/
	void update();

};

#endif // DUMBINTERPRETER_H