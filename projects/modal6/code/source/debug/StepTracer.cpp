///////////////////////////////////////////////////////////
//  Original author: Bonifaz Stuhr
///////////////////////////////////////////////////////////

#include "debug/StepTracer.h"
//#include <regex>
#include "controller/Controller.h"

//GCC and Stoi
#include "library/utils.h"

using namespace std;

StepTracer* StepTracer::instance = NULL;

StepTracer::StepTracer()
{
	this->file = "StepTracer_";
}

StepTracer::~StepTracer()
{
	fstream.close();
}

void StepTracer::observePointNode(PointNode* n)
{
	if (n) {
		if (this->observedNodes.find(n->getId()) != this->observedNodes.end())
		{
			this->observedNodes.insert(pair<const uint_fast64_t, PointNode*>(n->getId(), n));
			this->oldNodes.insert(pair<const uint_fast64_t, PointNode>(n->getId(), *n));
		}
	}
}

void StepTracer::removePointNode(PointNode* n)
{
	if (n)
	{
		this->observedNodes.erase(n->getId());
		this->oldNodes.erase(n->getId());
	}
}

void StepTracer::printChanges()
{
	Controller* C = Controller::getInstance();
	this->fstream.open(this->file + utils::to_string(C->getStepCount()), ios::out);
	if (C) {
		map<const uint_fast64_t, PointNode*>::iterator iter;
		fstream << "Changes from Step " << this->oldStep << " to " << utils::to_string(C->getStepCount()) << endl;

		for (iter = this->observedNodes.begin(); iter != this->observedNodes.end(); iter++)
		{
			PointNode* oldNode = &(*oldNodes.find((*iter).first)).second;
			if (*(*iter).second != *oldNode)
			{
				printNodeToFile(oldNode);
				fstream << "to" << endl;
				printNodeToFile((*iter).second);
				oldNode = (*iter).second;
			}
		}
	}
	this->oldStep = C->getStepCount();
	fstream.close();
}

/*
Schreibt alle Informationen über die Node
*/
void StepTracer::printNodeToFile(PointNode* n)
{
	if (n)
	{
		fstream << "Id: " << (n->getId() ? utils::to_string(n->getId()) : "NULL") << endl
			<< " Cord: " << (n->getPosition() ? posToString(n->getPosition()) : "NULL")
			<< " TempValue: " << (n->getTempValue() ? utils::to_string(n->getTempValue()) : "NULL")
			<< " ActiveValue: " << (n->getActiveValue() ? utils::to_string(n->getActiveValue()) : "NULL")
			<< " Neighbours: " << (n->getNeighbours()->size() ? utils::to_string(n->getNeighbours()->size()) : "NULL") << endl;
		if (n->getNeighbours())
		{
			fstream << "Neighbours: " << endl;
			vector<PointNode*>::iterator nIter;
			uint_fast8_t pos = 0;
			for (nIter = n->getNeighbours()->begin(); nIter != n->getNeighbours()->end(); nIter++)
			{
				fstream << "Pos: " << utils::to_string(pos) << " Id: "
					<< ((*nIter)->getId() ? utils::to_string((*nIter)->getId()) : "NULL") << endl
					<< " Cord: " << ((*nIter)->getPosition() ? posToString((*nIter)->getPosition()) : "NULL")
					<< " TempValue: " << ((*nIter)->getTempValue() ? utils::to_string((*nIter)->getTempValue()) : "NULL")
					<< " ActiveValue: " << ((*nIter)->getActiveValue() ? utils::to_string((*nIter)->getActiveValue()) : "NULL")
					<< " Neighbours: " << ((*nIter)->getNeighbours()->size() ? utils::to_string((*nIter)->getNeighbours()->size()) : "NULL") << endl;
				pos++;
			}
		}
		fstream << "-------------------------------------------------------------------------------------------";
	}
	else
	{
		fstream << " PointNode is Null" << endl;
	}
	fstream << endl;
}

string StepTracer::posToString(vector<uint_fast32_t>* v)
{
	string s = "(" + utils::to_string(v->at(0));
	for (uint_fast8_t i = 1; i < v->size(); i++)
	{
		s += ("," + (utils::to_string(v->at(i))));
	}
	s += ")";
	return s;
}

StepTracer* StepTracer::getInstance()
{
	if (!instance)
	{
		instance = new StepTracer();

			//// Aktuelle Zeit ermitteln
			//time_t timeStamp;
			//timeStamp = time(0);
			//tm* now = localtime(&timeStamp);

			//// Neuen Dateinamen erstellen
			//string timeSet = utils::to_string(now->tm_mday) + "_" +
			//	utils::to_string(now->tm_mon + 1) + "_" +
			//	utils::to_string(now->tm_year + 1900) + "  " +
			//	utils::to_string(now->tm_hour) + "-" +
			//	utils::to_string(now->tm_min) + "-" +
			//	utils::to_string(now->tm_sec);

			//instance->file = instance->file + "_" + timeSet + "_";

			//// Datum und Uhrzeit mit gewohnten Zeichen trennen (Als Dateiname sind diese nicht zulaessig)
			//// "_" wird zu "." und "-" wird zu ":"
			//string reg[2][2] = { { "_", "." },{ "-", ":" } };
			//for (int i = 0, j = 0; i < 2; i++)
			//{
			//	regex expr(reg[i][j]);
			//	timeSet = regex_replace(timeSet, expr, reg[i][j + 1]);
			//}
	}
	return instance;
}