///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Original author: Bonifaz Stuhr & Tobias Eigler
///////////////////////////////////////////////////////////

#ifndef IINTERPRETER_H
#define IINTERPRETER_H

#pragma once

/**
* Interpreter wenden die Rules auf die World an und fuehren einen Zeitschritt aus. Die Welt ändert sich.
*/
class IInterpreter
{
public:
	/**
	* Wendet die Rules auf die World an und fuehrt einen Zeitschritt aus. Die Welt ändert sich.
	* update() ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	*/
    virtual void update() = 0;

protected:
	/**
	*Kunstruktor.
	**/
    IInterpreter() {}

};

#endif // IINTERPRETER_H