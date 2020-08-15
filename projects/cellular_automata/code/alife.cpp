#include <iostream>
#include "controller/Controller.h"
#include "controller/Console.h"
#include "info.h"

#ifdef __linux__
#include "library/conio.h"
#endif

/**
* Artificial Life Simulator
* @param Xml File
* @param Xml Settings ID
*/
int main(int ParamSize, char* Param[])
{
#ifdef __linux__
    init();
#endif

    // Start Parameter und ihre Default Werte
    const char* path = "data.xml";
    const char* setting = "0";

    //Cache fuer Terminal Eingabe
    string pathTemp;
    string settingTemp;

    try
    {
        if(ParamSize == 3)
        {
            //Holen von Path Argument 1
            path = Param[1];

            //Holen von SettingsID Argument 2
            setting = Param[2];
        }
        else if(ParamSize == 2 && strcmp(Param[1], "?") == 0)
        {
            cout << "ALife v" << Version << endl;
            cout << "ALife.exe Path/To/Xml/File XmlSettingID" << endl;
            cout << endl;
            cout << Team << endl;
            return 0;
        }
        else
        {
            throw "Bad Parameters";
        }
    }
    catch(...)
    {
        // Fehlerhafte Argument line eingabe
        // Eingabe der Setting-ID und Pfad zur XML
        Console::Stream << "Pfad zur Konfigurations-XML angeben:" << endl;
        Console::writeAsInformation();

        // Lese Eingabe
        getline(cin, pathTemp);

        // Wurde etwas eingegeben?
        if (pathTemp != "")
        {
            path = pathTemp.c_str();
        }

        if(!Controller::viewConfig(path))
        {
            Console::Stream << "Xml Datei nicht gefunden" << endl;
            Console::writeAsInformation();
            return 0;
        }

        Console::Stream << "Welche Setting-ID soll verwendet werden?" << endl;
        Console::writeAsInformation();

        // Lese Eingabe
        getline(cin, settingTemp);

        // Wurde etwas eingegeben?
        if (settingTemp != "")
        {
            setting = settingTemp.c_str();
        }
    }

    cout << "ALife v" << Version << endl;
    cout << "Start Parameter" << endl;
    cout << "Xml File = " << path << endl;
    cout << "Setting-ID = " << setting << endl;

    // Controller initialsieren
    if(Controller::init(path, setting))
    {
        Console::Stream << "Init erfolgreich, Enter druecken!" << endl;
        Console::writeAsInformation();

        getchar();
        Controller::getInstance()->doSteps();
    }
    else
    {
        Console::Stream << "Init Fehlgeschlagen going to exit" << endl;
        Console::writeAsInformation();
        getchar();
    }

#ifdef __linux__
    endwin();
#endif

    return 0;
}