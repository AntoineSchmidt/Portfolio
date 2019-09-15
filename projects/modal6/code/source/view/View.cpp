///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <map>
#include <vector>
#include <sstream>
#include "view/View.h"
#include "library/tinyxml2.h"
#include "view/viewComponents/ViewComponent.h"
#include "view/viewComponents/ViewBmp.h"
#include "view/viewComponents/ViewConsole.h"
#include "view/viewComponents/ViewOGL.h"
#include "view/viewComponents/ViewOGL1D.h"
#include "controller/Console.h"
#include "library/utils.h"

View* View::Instance = 0;

//View initialisierung
View::View()
{
    viewComponents = new list <ViewComponent*>();
    Colors = new map<uint_fast16_t, string*>();
    RGBColors = new map<uint_fast16_t, vector<unsigned char>*>();
    RGBGridColor = new vector<unsigned char>();

    GridSize = 0;
    PixelSize = 1;
}

View::~View()
{
    //entsorgen der viewcomponent liste und farben map
    delete viewComponents;
    delete Colors;
    delete RGBColors;
    delete RGBGridColor;
}

//geht alle gespeicherten viewcomponenten durch und ruft deren update auf
void View::update()
{
    list <ViewComponent*>::iterator iter;
    for (iter = viewComponents->begin(); iter != viewComponents->end(); ++iter)
    {
        if(*iter)
        {
            (*iter)->update();
        }
        else
        {
            Console::Stream << "ViewComponent Null" << endl;
            Console::writeAsException();
        }
    }
}

//gibt die singleton instance von view zurueck
View* View::getInstance()
{
    if(!Instance)
    {
        Console::Stream << "View Instance Null" << endl;
        Console::writeAsException();
    }
    return Instance;
}

//Ruft Parse funktion von entsprechender viewcomponent classen auf
void View::parse(const tinyxml2::XMLElement* node)
{
    Instance = new View();
    for(const tinyxml2::XMLElement* e = node->FirstChildElement(); e !=  NULL; e = e->NextSiblingElement())
    {
        if(strcmp("ViewBMP", e->Value()) == 0)
        {
            Instance->viewComponents->push_back(ViewBmp::parse(e));
        }
        else if(strcmp("ViewConsole", e->Value()) == 0)
        {
            Instance->viewComponents->push_back(ViewConsole::parse(e));
        }
        else if(strcmp("ViewOGL", e->Value()) == 0)
        {
            Instance->viewComponents->push_back(ViewOGL::parse(e));
        }
		else if (strcmp("ViewOGL1D", e->Value()) == 0)
		{
			Instance->viewComponents->push_back(ViewOGL1D::parse(e));
		}
        else if(strcmp("Colors", e->Value()) == 0)
        {
            Instance->parseColors(e);
        }
        else if(strcmp("PixelSize", e->Value()) == 0)
        {
            Instance->PixelSize = atoi(e->GetText());
        }
        else if(strcmp("GridSize", e->Value()) == 0)
        {
            Instance->RGBGridColor = utils::to_rgb(e->Attribute("color"));
            Instance->GridSize = atoi(e->GetText());
        }
        else
        {
            Console::Stream << "View Parse Not Registered" << e->Value() << endl;
            Console::writeAsError();
        }
    }
}

//Parsed Farben definitionen aus dem Farben Node
void View::parseColors(const tinyxml2::XMLElement* node)
{
    for(const tinyxml2::XMLElement* e = node->FirstChildElement(); e !=  NULL; e = e->NextSiblingElement())
    {
        if(strcmp("Color", e->Value()) == 0)
        {
            Console::Stream << "View::ParseColors " << e->Attribute("id") << ": " << e->GetText() << endl;
            Console::writeAsMessage();
            Colors->insert(pair<uint_fast16_t, string*>(utils::stoi(e->Attribute("id")), new string(e->GetText())));
        }
        else
        {
            Console::Stream << "View Parse Colors Wrong Tag" << e->Value() << endl;
            Console::writeAsError();
        }
    }
}

//bekommt point value und gibt dazu gespeicherte hex farbe ab oder generiert hex farbe
string* View::getColor(const uint_fast16_t ColorID) const
{
    map<uint_fast16_t, string*>::iterator iter;
    iter = Colors->find(ColorID);
    if (iter != Colors->end())
    {
        return iter->second;
    }

    Console::Stream << "View getColor Not in Map" << endl;
    Console::writeAsError();

    return new string();
}

//bekommt point value und gibt dazu gespeicherte hex farbe ab oder generiert hex farbe als int
int View::getColorInt(const uint_fast16_t ColorID) const
{
    return utils::stoi(*getColor(ColorID));
}

//Returns Vector RGB Value
vector<unsigned char>* View::getColorRGBPixel(const uint_fast16_t ColorID)
{
    //Look if Already in Cache
    map<uint_fast16_t, vector<unsigned char>*>::iterator iter;
    iter = RGBColors->find(ColorID);
    if (iter != RGBColors->end())
    {
        return iter->second;
    }
    //Not in Cache
    else
    {
        vector<unsigned char>* Answer;
        if((Answer = utils::to_rgb(*getColor(ColorID))))
        {
            //Add to Cache
            RGBColors->insert(pair<uint16_t, vector<unsigned char>*>(ColorID, Answer));
        }
        return Answer;
    }
}

uint_fast8_t View::getPixelSize() const
{
    return PixelSize;
}

vector<unsigned char>* View::getColorRGBGrid() const
{
    return RGBGridColor;
}

uint_fast8_t View::getGridSize() const
{
    return GridSize;
}

//geht alle gespeicherten viewcomponenten durch und ruft deren finish auf
void View::finish()
{
    list <ViewComponent*>::iterator iter;
    for (iter = viewComponents->begin(); iter != viewComponents->end(); ++iter)
    {
        (*iter)->finish();
    }

    //entsorgen der viewcomponent liste und farben map
    delete viewComponents;
    delete Colors;
    delete RGBColors;
    delete RGBGridColor;
}