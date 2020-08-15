///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include "view/viewComponents/ViewConsole.h"
#include "library/tinyxml2.h"
#include "library/utils.h"
#include "model/World.h"
#include "controller/Console.h"
#include "model/PointNode.h"

ViewConsole::ViewConsole()
{

}

ViewConsole::~ViewConsole()
{

}

//erstellt ViewConsole
ViewComponent* ViewConsole::parse(const tinyxml2::XMLElement* node)
{
    ViewConsole* comp = new ViewConsole();
    for(const tinyxml2::XMLElement* e = node->FirstChildElement(); e !=  NULL; e = e->NextSiblingElement())
    {
        if(strcmp("StepsPerImage", e->Value()) == 0)
        {
            comp->UpdatePerFrame = atoi(e->GetText());
        }
        else
        {
            Console::Stream << "ViewConsole Parse Not Registered" << e->Value() << endl;
            Console::writeAsError();
        }
    }
    return comp;
}

//executeupdated bmp wird nur dann ausgefuehrt wenn stepsperimage erreicht, bekommt stepscount
void ViewConsole::executeupdate()
{
    Console::Stream << "View Console" << endl;
    Console::writeAsMessage();

    World* W;
    if((W = World::getInstance()))
    {
        vector<PointNode*>::iterator iter;
        for (iter = W->getPointNodesMatrix()->begin(); iter < W->getPointNodesMatrix()->end(); iter++)
        {
            if(*iter)
            {
                if ((*iter)->getPosition()->at(0) > W->getSize()->at(0))
                {
                    cout << endl;
                }
                if ((*iter)->getActiveValue() != 0)
                {
                    cout << (*iter)->getActiveValue() << ";";
                }
                else
                {
                    cout << " ;";
                }
            }
            else
            {
                Console::Stream << "PointNode Null" << endl;
                Console::writeAsException();
            }
        }
        cout << endl;
    }
}

//entsorgt und befreit alle ressourcen
void ViewConsole::finish()
{

}