///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Author: Michell Mueller
///////////////////////////////////////////////////////////

#include <algorithm>
#include "controller/Controller.h"
#include "controller/Console.h"
#include "debug/WorldCheck.h"
#include "interpreter/DumbInterpreter.h"
#include "interpreter/PointNodeInterpreter.h"
#include "interpreter/PNCInterpreter.h"
#include "interpreter/CombiInterpreter.h"
#include "dataAccess/XmlHandler.h"

#ifdef __linux__
#include "library/conio.h"
#else
#include <conio.h>
#endif

Controller* Controller::instance = NULL;

Controller::Controller()
{
	stepCount = 0;
	data = NULL;
	darwins = NULL;
}

Controller::~Controller()
{

}

Controller* Controller::getInstance()
{
	if (!instance)
	{
		Console::Stream << "Controller Instance Null" << endl;
		Console::writeAsException();
	}
	return instance;
}

void Controller::doSteps()
{
	// Wenn steps = 0 fuehre das Programm aus bis es abgebrochen wird
	if(settings->getStepCount() == 0)
	{
		while(true)
		{
			doStep();
		}
	}
	// Anzahl der Simulationsschritte durchfuehren:
	else
	{
		for (int i = 0; i < settings->getStepCount(); i++)
		{
			doStep();
		}
	}
}

void Controller::doStep()
{
	if (kbhit())
	{
		int key = getch();
		if (key == 32) // Leertaste
		{
			Console::Stream << "Simulation gestoppt!" << endl;
			Console::writeAsInformation();

			Console::Stream << "Press 's' to Save World" << endl;
			Console::writeAsInformation();

			Console::Stream << "Press Space to Continue" << endl;
			Console::writeAsInformation();

			while(true)
			{
				key = getch();
				if(key == 32)
				{
					Console::Stream << "Proceeding with Simulation!" << endl;
					Console::writeAsInformation();
					break;
				}
				else if (key == 's' || key == 'S')
				{
					data->saveWorld();

					Console::Stream << "Press Space to Continue" << endl;
					Console::writeAsInformation();
				}
			}
		}
	}

	// Boni WorldCheck
	//WorldCheck* worldCheck  = WorldCheck::getInstance();

	stepCount++;

	//Test "Permutation"
	if (stepCount == 10)
	{
		//worldCheck->compareWorldWithOld("1");
		//cout << "Manipulating Order Vector" << endl;
		//vector<uint_fast8_t>* Order = Rules::getInstance()->getRuleOrder();
		//rotate(Order->begin() + 2, Order->begin() + 1, Order->end()); //Going Crazy, Setting 8
		//rotate(Order->end(), Order->begin(), Order->begin() + 2); //Diagonal Moving, Setting 2
		//rotate(Order->end(), Order->begin(), Order->begin() + 1); //Eye Cancer, Setting 2; Ship, Setting 4

		//m_data->saveWorld();
		//getchar();
	}

	Console::Stream << "Processing Step: " << stepCount << endl;
	Console::writeAsMessage();

	// Uebergabe an den Interpreter
	interpreter->update();

	if (darwins)
	{
		for (unsigned int i = 0; i < darwins->size(); i++)
		{
			darwins->at(i)->doStep();
		}
	}

	// Ubergabe an die View
	View::getInstance()->update();
}

bool Controller::viewConfig(const char* file)
{
	// XMLHandler erzeugen:
	XmlHandler* Temp = new XmlHandler();

	// Pfad der zu oeffnenden XML-Datei zum Oeffnen uebergeben:
	if (!Temp->openConnection(file))
	{
		getchar();
		return false; // Fehler beim Oeffnen der angegebenen Datei
	}
	else
	{
		Temp->viewAllSettings();
		return true;
	}

	delete Temp;
}

bool Controller::init(const char* file, const char* settingID)
{
	instance = new Controller();
	instance->data = new XmlHandler();

	if(!instance->data->openConnection(file))
	{
		getchar();
		return false;
	}

	//// Boni NuklearDarwin
	//instance->m_nuklearDarwin = new NuklearDarwin();
	//vector<int> mutationVector = { 0,3 };
	//set<int> targets = { 2 };
	//instance->m_nuklearDarwin->parametrize(1, 1000000, mutationVector, targets,200,0,5);

	//WordCheck
	//WorldCheck* worldCheck = WorldCheck::getInstance();
	//worldCheck->checkWorld("1");
	//worldCheck->printWorld("1");

	// Allgemeine Einstellungen aus XML holen:
	if((instance->settings = instance->data->getSettings(settingID)))
	{
		// Weitere Einstellungen aus XML holen:
		instance->data->loadWorld(instance->settings->getWorldID());
		instance->data->loadRules(instance->settings->getRulesID());
		instance->data->loadView(instance->settings->getViewID());

		if (strcmp(instance->settings->getDarwinEnginesID(), "X") != 0)
		{
			instance->darwins = instance->data->getDarwinEngines(instance->settings->getDarwinEnginesID());
		}
		// Connection vom Handler schlie�en
		//instance->data->closeConnection();

		// Interpreter fuer die Logik erstellen:
		//instance->interpreter = new DumbInterpreter();
		//instance->interpreter = new PointNodeInterpreter();
		//instance->interpreter = new PNCInterpreter();
		instance->interpreter = new CombiInterpreter();

		// Anzeigen der Init World
		View::getInstance()->update();
		return true;
	}
	else
	{
		Console::Stream << "Controller::Init Settings Null" << endl;
		Console::writeAsException();
		return false;
	}
}

int Controller::getStepCount()
{
	return stepCount;
}

Settings* Controller::getSettings()
{
	return instance->settings;
}