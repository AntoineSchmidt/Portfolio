///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <vector>
#include "view/viewComponents/ViewOGL.h"
#include "library/tinyxml2.h"
#include "controller/Console.h"

ViewOGL::ViewOGL()
{
    window = NULL;
    createWindow = true;
    UpdatePerFrame = 1;
}

ViewOGL::~ViewOGL()
{
    delete window;
}

ViewComponent* ViewOGL::parse(const tinyxml2::XMLElement* node)
{
    ViewOGL* s = new ViewOGL();
    s->window = new ViewOGLWindow();
    for(const tinyxml2::XMLElement* e = node->FirstChildElement(); e !=  NULL; e = e->NextSiblingElement())
    {
        if(strcmp("windowwidth", e->Value()) == 0)
        {
            s->window->setWidth(atoi(e->GetText()));
        }
        else if(strcmp("windowheight", e->Value()) == 0)
        {
            s->window->setHeight(atoi(e->GetText()));
        }
        else if(strcmp("windowtitle", e->Value()) == 0)
        {
            s->window->setTitle(string(e->GetText()));
        }
        else if (strcmp("stepsperimage", e->Value()) == 0)
        {
            s->UpdatePerFrame = atoi(e->GetText());
        }
        else if (strcmp("fps", e->Value()) == 0)
        {
            const char* Att = NULL;
            if((Att = e->Attribute("enable")))
            {
                if (strcmp(Att, "true") == 0)
                {
                    s->window->setUseFps(true);
                }
                else if (strcmp(Att, "false") == 0)
                {
                    s->window->setUseFps(false);
                }
                else
                {
                    Console::Stream << "Parsing ViewOGL enable FPS Attribut: " << *Att << endl;
                    Console::writeAsError();
                }
            }
            Att = NULL;
            if((Att = e->GetText()))
            {
                s->window->setFps(atoi(Att));
            }
        }
        else
        {
            Console::Stream << "OGL Parse Not Registered" << e->Value() << endl;
            Console::writeAsError();
        }
    }
    return s;
}

void ViewOGL::executeupdate()
{
    //Iterate throught world and call drawrectangle
    Console::Stream << "View OpenGL" << endl;
    Console::writeAsMessage();

    if(createWindow)
    {
        if(!window)
        {
            window = new ViewOGLWindow();
        }
        tg.create_thread(boost::bind(&ViewOGLWindow::run, window));
        createWindow = false;
    }
    window->update();
}

void ViewOGL::finish()
{
    delete window;
}