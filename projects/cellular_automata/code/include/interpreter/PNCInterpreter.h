///////////////////////////////////////////////////////////
//  Point Node Changed Interpreter
//  Itteriert ueber komplette welt und prueft auf Changed
//  Created on:      07-Apr-2016 10:59:26
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef PNCINTERPRETER_H
#define PNCINTERPRETER_H

#pragma once

#include <vector>
#include <set>
#include <boost/thread.hpp>
#include "IInterpreter.h"
#include "model/World.h"
#include "model/Rules.h"

/**
* Interpreter wenden die Rules auf die World an und fuehren einen Zeitschritt aus. Die Welt ändert sich.
*/
class PNCInterpreter : public IInterpreter
{
public:
	/**
	*Kunstruktor.
	**/
	PNCInterpreter();

	/**
	*Destruktor.
	**/
	~PNCInterpreter();

	/**
	* Wendet die Rules auf die World an und fuehrt einen Zeitschritt aus. Die Welt ändert sich.
	* Itteriert über die PointNodes der Welt die sich im letzten Zeitschritt geaendert haben.
	*/
	void update();

	/**
	* Wendet die Rules auf die World an und fuehrt einen Zeitschitt aus. Die Welt ändert sich.
	* Itteriert über die PointNodes der Welt die sich im letzten Zeitschritt geaendert haben.
	* Die zu interpretierende Welt wir in Threads aufgeteilt.
	@param unsigned int ThreadCount benoentigt Anzahl der Threads.
	*/
	void updatethreaded(unsigned int ThreadCount);

private:
	/**
	*Wendet die Rules auf den übergebenen Vector von PointNodes an.
	*@param const vector <PointNode*>*, Pointer auf den Vector der die PointNodes enthaelt auf die dei Regel angewendet werden sollen.
	*@param const IRules* R, Pointer auf die Rules, die die zu verwendenden Regeln enthalten.
	*@param const unsigned int Start, Startkoordinate des uebergebenen Vectors, der die PointNodes enthaelt.
	*@param const unsigned int End, Endkoordinate des uebergebenen Vectors, der die PointNodes enthaelt.
	*/
	void applyRule(const vector <PointNode*>* M, const IRules* R, const unsigned int Start, const unsigned int End);

	//Thread Container
    boost::thread_group tg;
};

#endif // PNCINTERPRETER_H