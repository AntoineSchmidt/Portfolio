///////////////////////////////////////////////////////////
//  Created on:      02-June-2016
//  Original author: Bonifaz Stuhr und Tobias Eigler
///////////////////////////////////////////////////////////

#include <time.h>
#include "darwinEngine/NuklearDarwin.h"
#include "model/World.h"
#include "library/utils.h"
#include "controller/Console.h"

NuklearDarwin::NuklearDarwin()
{
	this->mutationChanceDenominator = 1000000;
	this->mutationChanceCounter = 1;
	this->startAtStep = 0;
	this->stepsActive = 0;
	this->mutateAfterXSteps = 0;
	this->intervallCounter = 0;
	this->stepCounter = 0;
	this->file = "NuklearDarwin_Mutations.txt";
	this->stream.open(file, ios::out);
}

NuklearDarwin::~NuklearDarwin()
{
	stream.close();
}

void NuklearDarwin::doStep()
{
	if (((this->stepCounter > this->stepsActive) && (this->stepsActive != 0)) || (this->stepCounter <= this->startAtStep))
	{
		this->stepCounter++;
		return;
	}

	srand(time(NULL));

	IWorld* W;
	//Pruefung auf Valid Pointer
	if ((W = World::getInstance()))
	{
		vector <PointNode*>* M = W->getPointNodesMatrix();
		vector <PointNode*>::iterator matrixIter;

		if (this->intervallCounter == this->mutateAfterXSteps)
		{
			//Fahre über gesammte Welt und verstrahle sie zufällig
			for (matrixIter = M->begin(); matrixIter != M->end(); matrixIter++)
			{
				if (mutationChanceCounter >= (rand() % mutationChanceDenominator + 1))
				{
					uint_fast16_t mutateValue = this->mutationValues.at(rand() % this->mutationValues.size());
					if (this->mutationTargets.find((*matrixIter)->getActiveValue()) != this->mutationTargets.end())
					{
						Console::Stream << "Mutate " << (*matrixIter)->getActiveValue() << " to " << mutateValue << " at Step: " << stepCounter << endl;
						Console::writeAsMessage();

						stream << "Mutate " << (*matrixIter)->getActiveValue() << " to " << mutateValue << " at Step: " << stepCounter << endl;
						(*matrixIter)->setActiveValue(mutateValue);
					}
				}
			}
			intervallCounter = 0;
		}
		this->intervallCounter++;
		this->stepCounter++;
	}
}

void NuklearDarwin::parametrize(uint_fast64_t mutationChanceCounter, uint_fast64_t mutationChanceDenominator, vector<uint_fast16_t> mutationValues, set<uint_fast16_t> mutationTargets, uint_fast64_t startAtStep = 0, uint_fast64_t stepsActive = 0, uint_fast32_t mutateAfterXSteps = 0)
{
	this->mutationChanceDenominator = mutationChanceDenominator;
	this->mutationChanceCounter = mutationChanceCounter;
	this->startAtStep = startAtStep;
	this->stepsActive = stepsActive;
	this->mutateAfterXSteps = mutateAfterXSteps;
	this->mutationValues = mutationValues;
	this->mutationTargets = mutationTargets;

	stream << "NuklearDarwin Parameter: MutationChance: " << mutationChanceCounter << "/" << mutationChanceDenominator <<
		"/100000, startAtStep: " << startAtStep << " stepsActive: " << stepsActive << " mutateAfterXSteps: " << mutateAfterXSteps <<
		" mutationValues: ";

	vector<uint_fast16_t>::iterator iter;
	for (iter = mutationValues.begin(); iter != mutationValues.end(); iter++)
	{
		stream << *iter << " ";
	}

	stream << " ,mutationTargets: ";
	set<uint_fast16_t>::iterator iter2;
	for (iter2 = mutationTargets.begin(); iter2 != mutationTargets.end(); iter2++)
	{
		stream << *iter2 << " ";
	}
}

IDarwinEngine* NuklearDarwin::parse(const tinyxml2::XMLElement* node)
{
	NuklearDarwin* darwin = new NuklearDarwin();
	string Segment;


	for (const tinyxml2::XMLElement* e = node->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
	{
		if (strcmp(e->Value(), "OutputDirectory") == 0)
		{
			if(e->GetText())
				darwin->file = e->GetText() + darwin->file;
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"OutputDirectory\" given, setting to projektdirectory" << endl;
			Console::writeAsError();
		}
		if (strcmp(e->Value(), "FileName") == 0)
		{
			if (e->GetText())
				darwin->file = e->GetText() + darwin->file;
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"FileName\" given, setting to NuklearDarwin_Mutations" << endl;
			Console::writeAsError();
		}
		if (strcmp(e->Value(), "MutationTargets") == 0)
		{
			if (e->GetText())
			{
				stringstream Stream(e->GetText());

				//Auftrennen des Strings am "/"
				while (getline(Stream, Segment, ','))
				{
					darwin->mutationTargets.insert(utils::stoi(Segment));
				}
			}
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"MutationChance\" given, setting to 1/1000000" << endl;
			Console::writeAsError();
		}
		if (strcmp(e->Value(), "StartAtStep") == 0)
		{
			if (e->GetText())
				darwin->startAtStep = utils::stoi(e->GetText());
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"StartAtStep\" given, setting to 0" << endl;
			Console::writeAsError();
		}
		if (strcmp(e->Value(), "StepsActive") == 0)
		{ 
			if (e->GetText())
				darwin->stepsActive = utils::stoi(e->GetText());
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"StepsActive\" given, setting to 0" << endl;
			Console::writeAsError();
		}
		if (strcmp(e->Value(), "MutateAfterXSteps") == 0)
		{
			if (e->GetText())
				darwin->mutateAfterXSteps = utils::stoi(e->GetText());
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"MutateAfterXSteps\" given, setting to 0" << endl;
			Console::writeAsError();
		}
		if (strcmp(e->Value(), "MutationChance") == 0)
		{
			if (e->GetText())
			{
				stringstream Stream(e->GetText());
				vector<string> seg;
				string str;

				//Auftrennen des Strings am "Komma"
				while (getline(Stream, str, '/'))
				{
					seg.push_back(str);
				}
				darwin->mutationChanceCounter = utils::stoi(seg.at(0));
				darwin->mutationChanceDenominator = 1;
				for (unsigned int j = 1; j < seg.size(); j++)
				{
					darwin->mutationChanceDenominator *= utils::stoi(seg.at(j));
				}
			}
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"MutationTargets\" given, setting to 0" << endl;
			Console::writeAsError();
		}
		if (strcmp(e->Value(), "MutationValues") == 0)
		{
			if (e->GetText())
			{
				stringstream Stream(e->GetText());

				//Auftrennen des Strings am "Komma"
				while (getline(Stream, Segment, ','))
				{
					darwin->mutationValues.push_back(utils::stoi(Segment));
				}
			}
		}
		else
		{
			Console::Stream << "XML(NuklearDarwin): No \"MutationValues\" given, setting to 0" << endl;
			Console::writeAsError();
		}
	}
	return darwin;
}