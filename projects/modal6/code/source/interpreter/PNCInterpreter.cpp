///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <vector>
#include "interpreter/PNCInterpreter.h"
#include "model/World.h"
#include "model/PointNode.h"
#include "controller/Console.h"
#include "model/Rules.h"

PNCInterpreter::PNCInterpreter()
{

}

PNCInterpreter::~PNCInterpreter()
{

}

void PNCInterpreter::update()
{
	IWorld* W;
	IRules* R;
	//Pruefung auf Valid Pointer
	if((W = World::getInstance()) && (R = Rules::getInstance()))
	{
		vector <PointNode*>* M = W->getPointNodesMatrix();
		vector <PointNode*>::iterator MatrixIter;

		applyRule(M, R, 0, M->size());

		//Update Point Values
		for (MatrixIter = M->begin(); MatrixIter != M->end(); ++MatrixIter)
		{
			(*MatrixIter)->update();
		}
	}
}

void PNCInterpreter::updatethreaded(unsigned int ThreadCount)
{
	Console::Stream << "Multi-Threaded Update in Progress" << endl;
	Console::writeAsMessage();

	IWorld* W;
	IRules* R;
	//Pruefung auf Valid Pointer
	if((W = World::getInstance()) && (R = Rules::getInstance()))
	{
		vector <PointNode*>* M = W->getPointNodesMatrix();
		vector <PointNode*>::iterator MatrixIter;

		//Minimum of one Thread per PointNode
		if(ThreadCount > M->size())
		{
			ThreadCount = M->size();
		}

		unsigned int Parts = M->size() / ThreadCount;

		for(unsigned int i = 0; i < ThreadCount; i++)
		{
			if((i + 1) >= ThreadCount)
			{
			    //Last Planned Thread gets Rest
				tg.create_thread(boost::bind(&PNCInterpreter::applyRule, this, M, R, i*Parts, i*Parts + Parts + (M->size() % ThreadCount)));
			}
			else
			{
				//Creating Thread Instant Start
				tg.create_thread(boost::bind(&PNCInterpreter::applyRule, this, M, R, i*Parts, i*Parts + Parts));
			}
		}

		//Waiting for all Threads to Finish
		tg.join_all();

		//Update Point Values
		for (MatrixIter = M->begin(); MatrixIter != M->end(); ++MatrixIter)
		{
			(*MatrixIter)->update();
		}
	}
}

void PNCInterpreter::applyRule(const vector <PointNode*>* M, const IRules* R, const unsigned int Start, const unsigned int End)
{
	vector <PointNode*>::const_iterator MatrixIter;

	//Pruefung auf Valides End Value
	if((M->begin() + End) <= M->end() && Start <= End)
	{
		//Setzte Entsprechende Temp Point Values
		for (MatrixIter = ( M->begin() + Start ); MatrixIter != ( M->begin() + End ); ++MatrixIter)
		{
			//Test auf notwendigkeit des Updates
			if((*MatrixIter)->getChanged())
			{
				(*MatrixIter)->setTempValue(R->getRuleValue((*MatrixIter)->getValueWithNeighbours()));
			}
		}
	}
	else
	{
		Console::Stream << "ApplyRule bad Values" << endl;
		Console::writeAsError();
		return;
	}
}