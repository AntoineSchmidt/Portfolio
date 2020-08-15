///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Original author: Bonifaz Stuhr
///////////////////////////////////////////////////////////

#include "interpreter/DumbInterpreter.h"
#include "model/World.h"

DumbInterpreter::DumbInterpreter()
{

}

DumbInterpreter::~DumbInterpreter()
{

}

void DumbInterpreter::update()
{
	IWorld* W = World::getInstance();
	IRules* R = Rules::getInstance();
	vector<PointNode*>* M = W->getPointNodesMatrix();
	vector<PointNode*>::iterator MatrixIter;

	//Iteriere ueber die gesamte Welt
	for (MatrixIter = M->begin(); MatrixIter != M->end(); MatrixIter++)
	{
		/*
		1. getPointValueAndNeighbours: Hole anhand der Koordinatne den Punkt-Wert mit seinen Nachbar-Werten (string) z.B. 2D: "00000"
		2. getRuleValue: Der obige Punkt+Nachbar-Wert string dient als Key und wir erhalten somit den neuen Wert des Punktes z.b. 00000->0
		3. setPointValue: Der neue Wert wird in seinem urspruenglichen Punkt an ursuerungliche Koordiante asl Temp Wert gespeichet.
		Es wird erst ein Temp wert gesetzt damit ein Zeitschritt korrekt ausgefuehrt wird. Man sieht sich die Welt an veraendert dann alle Punkte
		Ohen den Tempwert wuerde man sich die Welt nach jeder Iteration erneut ansehen
		*/
		(*MatrixIter)->setTempValue(R->getRuleValue((*MatrixIter)->getValueWithNeighbours()));

	}

	for (MatrixIter = M->begin(); MatrixIter != M->end(); MatrixIter++)
	{
		(*MatrixIter)->update();
	}
}