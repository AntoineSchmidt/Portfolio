///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:27
//  Author: Michell Mueller
///////////////////////////////////////////////////////////

#include "controller/Settings.h"
#include "controller/Console.h"

using namespace std;

// Entspricht den Tagnamen aus der XML-Datei:
const string Settings::worldTag = "WorldID";
const string Settings::viewTag = "ViewID";
const string Settings::rulesTag = "RuleID";
const string Settings::darwinEnginesTag = "DarwinEnginesID";
const string Settings::stepTag = "Steps";
const string Settings::debugOutputTag = "Output";
const char* Settings::settingsNameTag = "name";

Settings::Settings()
{
	// Da Darwin Optional ist wird "X" als "leeren-Wert" verwendent
	darwinEnginesID = "X";
}

Settings::~Settings()
{

}

Settings* Settings::parse(const tinyxml2::XMLElement* node)
{
    Settings* instance = new Settings();

    // Name des Settings, welches in der XML vergeben wurde
    instance->settingsName = node->Attribute(settingsNameTag);

    for (const tinyxml2::XMLElement* e = node->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
    {
        if (instance->worldTag.compare(e->Value()) == 0)
        {
            instance->worldID = e->GetText();
        }
        else if (instance->rulesTag.compare(e->Value()) == 0)
        {
            instance->rulesID = e->GetText();
        }
        else if (instance->viewTag.compare(e->Value()) == 0)
        {
            instance->viewID = e->GetText();
        }
		else if (instance->darwinEnginesTag.compare(e->Value()) == 0)
		{
			instance->darwinEnginesID = e->GetText();
		}
        else if (instance->stepTag.compare(e->Value()) == 0)
        {
            instance->stepCount = atoi(e->GetText());
        }
        else if (instance->debugOutputTag.compare(e->Value()) == 0)
        {
            const char* Att = NULL;
            if((Att = e->Attribute("messages")))
            {
                if (strcmp(Att, "true") == 0)
                {
                    Console::writeMessages = true;
                }
                else if (strcmp(Att, "false") == 0)
                {
                    Console::writeMessages = false;
                }
                else
                {
                    Console::Stream << "Settings Output Messages Attribut: " << *Att << endl;
                    Console::writeAsError();
                }
            }
            Att = NULL;
            if((Att = e->Attribute("errors")))
            {
                if (strcmp(Att, "true") == 0)
                {
                    Console::writeErrors = true;
                }
                else if (strcmp(Att, "false") == 0)
                {
                    Console::writeErrors = false;
                }
                else
                {
                    Console::Stream << "Settings Output Errors Attribut: " << *Att << endl;
                    Console::writeAsError();
                }
            }
            Att = NULL;
            if((Att = e->Attribute("exceptions")))
            {
                if (strcmp(Att, "true") == 0)
                {
                    Console::writeExceptions = true;
                }
                else if (strcmp(Att, "false") == 0)
                {
                    Console::writeExceptions = false;
                }
                else
                {
                    Console::Stream << "Settings Output Exceptions Attribut: " << *Att << endl;
                    Console::writeAsError();
                }
            }
            Att = NULL;
            if((Att = e->GetText()))
            {
                if (strcmp(Att, "true") == 0)
                {
                    Console::mute = true;
                }
                else if (strcmp(Att, "false") == 0)
                {
                    Console::mute = false;
                }
                else
                {
                    Console::Stream << "Settings Output Mute Attribut: " << *Att << endl;
                    Console::writeAsError();
                }
            }
        }
        else
        {
            Console::Stream << "Settings Parse Not Registered: " << *e->Value() << endl;
            Console::writeAsError();
        }
    }

    Console::Stream << "Settings Parse Registered: " << endl;
    Console::Stream << "    " << instance->settingsName << endl;
    Console::Stream << "    " << instance->worldTag << ": " << instance->worldID << endl;
    Console::Stream << "    " << instance->rulesTag << ": " << instance->rulesID << endl;
    Console::Stream << "    " << instance->viewTag << ": " << instance->viewID << endl;
	Console::Stream << "    " << instance->darwinEnginesTag << ": " << instance->darwinEnginesID << endl;
    Console::Stream << "    " << instance->stepTag << ": " << instance->stepCount << endl;
    Console::writeAsInformation();

    return instance;
}

int Settings::getStepCount()
{
    return stepCount;
}

const char* Settings::getViewID()
{
    return viewID;
}

const char* Settings::getWorldID()
{
    return worldID;
}

const char* Settings::getRulesID()
{
    return rulesID;
}

const char* Settings::getDarwinEnginesID()
{
	return darwinEnginesID;
}

const string Settings::getSettingsName()
{
    return settingsName;
}