///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <vector>
#include <list>
#include "dataAccess/XmlHandler.h"
#include "model/World.h"
#include "view/View.h"
#include "model/IRules.h"
#include "model/Rules.h"
#include "controller/Settings.h"
#include "controller/Controller.h"
#include "darwinEngine/NuklearDarwin.h"
#include "darwinEngine/IDarwinEngine.h"
#include "controller/Console.h"
#include "library/utils.h"
#include "library/tinyxml2.h"
#include "dataAccess/extern/ExtensionRule.h"
#include "dataAccess/extern/ExtensionRle.h"
#include "info.h"

XmlHandler::XmlHandler()
{

}

XmlHandler::~XmlHandler()
{

}

//oeffnet datei und holt rootnode
bool XmlHandler::openConnection(const char* Pfad)
{
	try
	{
		if(OpenedDocument.LoadFile(Pfad) != tinyxml2::XML_SUCCESS)
		{
			Console::Stream <<"!openConnection, Error" << endl;
			Console::writeAsError();
			return false;
		}
		OpenedFilePath = Pfad;
		RootNode = OpenedDocument.RootElement();

		//Version Check
		const char* Att = NULL;
		if((Att = RootNode->Attribute("version")))
		{
			if(strcmp(Att, Version) != 0)
			{
				Console::Stream << "Programm Version: " << Version << " not matching with Xml Version: " << Att << endl;
				Console::writeAsError();
			}
		}

		return true;
	}
	catch (...)
	{
		Console::Stream << "!openConnection, Exception" << endl;
		Console::writeAsException();
		return false;
	}
}

//schließt datei keine entsprechende funktion in tinyxml
void XmlHandler::closeConnection()
{
	try
	{
		//delete RootNode;
		//OpenedDocument.cl
	}
	catch (...)
	{

	}
}

//geht alle nodes der rootnode durch und uebergibt rules parser node mit passender rules id
void XmlHandler::loadRules(const char* id)
{
	if(RootNode)
	{
		for (tinyxml2::XMLElement* e = RootNode->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
		{
			if (strcmp("Rules", e->Value()) == 0 && strcmp(e->Attribute("id"), id) == 0)
			{
				LoadedRulesId = id;
				Rules::parse(e);
				return;
			}
		}
		//Behandle als File
		ExtensionRule* Rul = new ExtensionRule(id);
		tinyxml2::XMLElement* e = OpenedDocument.NewElement("Rules");
		if(Rul->buildRulesNode(e, &OpenedDocument))
		{
			//Save Parsed File to Xml
			RootNode->LinkEndChild(e);
			OpenedDocument.SaveFile(OpenedFilePath);

			LoadedRulesId = id;
			Rules::parse(e);
			return;
		}
		else
		{
			Console::Stream << "XML(File): Rules with ID " << id << " not Found" << endl;
			Console::writeAsError();
		}
	}
	else
	{
		Console::Stream << "XmlHandler RootNode Null, Missing File?" << endl;
		Console::writeAsError();
	}

}

bool XmlHandler::saveRules(const char* id)
{
	return false;
}

//geht alle nodes der rootnode durch und uebergibt world parser node mit passender world id
//gibt erstellen chunk von world zurueck
void XmlHandler::loadWorld(const char* id)
{
	if(RootNode)
	{
		for (tinyxml2::XMLElement* e = RootNode->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
		{
			if (strcmp("World", e->Value()) == 0 && strcmp(e->Attribute("id"), id) == 0)
			{
				LoadedWorldId = id;
				World::parse(e);
				return;
			}
		}
		//Behandle als File
		ExtensionRle* Rle = new ExtensionRle(id);
		tinyxml2::XMLElement* e = OpenedDocument.NewElement("World");
		if(Rle->buildWorldNode(e, &OpenedDocument))
		{
			//Save Parsed File to Xml
			RootNode->LinkEndChild(e);
			OpenedDocument.SaveFile(OpenedFilePath);

			LoadedWorldId = id;
			World::parse(e);
			return;
		}
		else
		{
			Console::Stream << "XML(File): World with ID " << id << " not Found" << endl;
			Console::writeAsError();
		}
	}
	else
	{
		Console::Stream << "XmlHandler RootNode Null, Missing File?" << endl;
		Console::writeAsError();
	}
}

bool XmlHandler::saveWorld(const char* id)
{
	if(!id)
	{
		string Title(LoadedRulesId);
		Title += "-" + utils::to_string(Controller::getInstance()->getStepCount());
		id = Title.c_str();
	}

	World* W;
	if(RootNode)
	{
		if((W = World::getInstance()))
		{
			//Prueft auf bereits existiernde Welt id
			for (tinyxml2::XMLElement* e = RootNode->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
			{
				if (strcmp("World", e->Value()) == 0 && strcmp(e->Attribute("id"), id) == 0)
				{
					Console::Stream << "XmlHandler Save World ID existiert bereits" << endl;
					Console::writeAsError();
					return false;
				}
			}

			//Erstelle alle Elemente
			tinyxml2::XMLElement* element = OpenedDocument.NewElement("World");
			element->SetAttribute("id", id);

			//Creating Size String
			vector<uint_fast32_t>* WorldSize = NULL;
			string siz = "";
			if((WorldSize = W->getSize()))
			{
				for(unsigned int i = 0; i < WorldSize->size(); i++)
				{
					siz += utils::to_string(WorldSize->at(i));

					//Pruefen ob letzte Koordinate
					if((i + 1) < WorldSize->size())
					{
						siz += ",";
					}
				}
			}
			element->SetAttribute("size", siz.c_str());

			//Setting Border Attribute
			element->SetAttribute("border", W->hasBorder() == true? "true" : "false");

			//Gehe alle Punkte der Welt druch und haenge sie an
			vector <PointNode*>* M = W->getPointNodesMatrix();
			vector <PointNode*>::iterator MatrixIter;
			for (MatrixIter = M->begin(); MatrixIter != M->end(); ++MatrixIter)
			{
				//Ueberspringe wenn Punkt Wert gleich Emtpy Value
				if((*MatrixIter)->getActiveValue() != Point::emptyValue)
				{
					//Erstelle neues Element
					tinyxml2::XMLElement* worldpoint = OpenedDocument.NewElement("Point");

					//Erstelle Coordinaten String
					vector<uint_fast32_t>* cordvector = NULL;
					string cordvalues = "";
					if((cordvector = (*MatrixIter)->getPosition()))
					{
						for(unsigned int i = 0; i < cordvector->size(); i++)
						{
							cordvalues += utils::to_string(cordvector->at(i));

							//Pruefen ob letzte Koordinate
							if((i + 1) < cordvector->size())
							{
								cordvalues += ",";
							}
						}
					}
					worldpoint->SetAttribute("position", cordvalues.c_str());
					worldpoint->SetText(utils::to_string((*MatrixIter)->getActiveValue()).c_str());

					//Haenge neues Element an
					element->LinkEndChild(worldpoint);
				}
			}

			//Fuege neues World Node in RootNode ein
			RootNode->LinkEndChild(element);

			//Abspeichern der Datei
			OpenedDocument.SaveFile(OpenedFilePath);

			return true;
		}
		else
		{
			Console::Stream << "No World Instance" << endl;
			Console::writeAsError();
			return false;
		}
	}
	else
	{
		Console::Stream << "XmlHandler RootNode Null, Missing/Closed File?" << endl;
		Console::writeAsError();

		return false;
	}
	return false;
}

//geht alle nodes der rootnode durch und uebergibt view parser node mit passender view id
void XmlHandler::loadView(const char* viewID)
{
	if(RootNode)
	{
		for (tinyxml2::XMLElement* e = RootNode->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
		{
			if (strcmp("View", e->Value()) == 0 && strcmp(e->Attribute("id"), viewID) == 0)
			{
				View::parse(e);
				return;
			}
		}
		Console::Stream << "XML(File): View with ID " << viewID << " not Found" << endl;
		Console::writeAsError();
	}
	else
	{
		Console::Stream << "XmlHandler RootNode Null, Missing File?" << endl;
		Console::writeAsError();
	}
}

//geht alle nodes der rootnode durch und uebergibt settings parser node mit passender settings id
//gibt pointer auf erstellten setting zurueck
Settings* XmlHandler::getSettings(const char* Id)
{
	if(RootNode)
	{
		for (tinyxml2::XMLElement* e = RootNode->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
		{
			if (strcmp("Settings", e->Value()) == 0 && strcmp(e->Attribute("id"), Id) == 0)
			{
				return Settings::parse(e);
			}
		}
		Console::Stream << "XML(File): Settings with ID " << Id << " not Found" << endl;
		Console::writeAsError();
	}
	else
	{
		Console::Stream << "XmlHandler RootNode Null, Missing File?" << endl;
		Console::writeAsError();
	}
	return NULL;
}

//geht alle nodes der rootnode durch und uebergibt settings parser node mit passender settings id
//gibt pointer auf erstellten setting zurueck
void XmlHandler::viewAllSettings()
{
	if (RootNode)
	{
		for (tinyxml2::XMLElement* e = RootNode->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
		{
			if (strcmp("Settings", e->Value()) == 0)
			{
				Console::Stream << "ID " << e->Attribute("id") << " : " << e->Attribute("name") << endl;
				Console::writeAsInformation();
			}
		}
	}
	else
	{
		Console::Stream << "XmlHandler RootNode Null, Missing File?" << endl;
		Console::writeAsError();
	}

}

//geht alle nodes der rootnode durch und uebergibt settings parser node mit passender settings id
//gibt pointer auf erstellten setting zurueck
vector<IDarwinEngine*>* XmlHandler::getDarwinEngines(const char* Id)
{
	vector<IDarwinEngine*>* ret = new vector<IDarwinEngine*>();
	if (RootNode)
	{
		for (tinyxml2::XMLElement* node = RootNode->FirstChildElement(); node != NULL; node = node->NextSiblingElement())
		{
			if (strcmp("DarwinEngines", node->Value()) == 0 && strcmp(node->Attribute("id"), Id) == 0)
			{
				for (const tinyxml2::XMLElement* e = node->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
				{
					if (e)
					{
						if (strcmp("NuklearDarwin", e->Value()) == 0)
						{
							ret->push_back(NuklearDarwin::parse(e));

						}
						//Hier können weitere Darwins geparst werden
					}

				}
			}
		}
		return ret;
		Console::Stream << "XML(File): DarwinEngines with ID " << Id << " not Found" << endl;
		Console::writeAsError();
	}
	else
	{
		Console::Stream << "XmlHandler RootNode Null, Missing File?" << endl;
		Console::writeAsError();
	}
	return NULL;
}