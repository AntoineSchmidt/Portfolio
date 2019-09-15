///////////////////////////////////////////////////////////
//  Author:          Bonifaz Stuhr
///////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include "view/viewComponents/ViewOGL1D.h"
#include "library/tinyxml2.h"
#include "model/World.h"
#include "controller/Console.h"
#include "view/View.h"

#ifdef __linux__
// OGL-Anzeige fuer Linux:
#include <GL/gl.h>
#include <GL/glut.h>
#else
// OGL-Anzeige fuer Windows:
#include "library/freeglut/glut.h"
#endif

ViewOGL1D::ViewOGL1D()
{
    windowNumber = 0;

    //Fill Default Values
    windowWidth = 1000;
    windowX = 10;
    windowHeight = 800;
    windowY = 10;
    windowTitle = "ALife";
    UpdatePerFrame = 1;
	worldBufferSize = 5;

    //Should be Overwritten
    coordinateMaxX = 1;
    coordinateMaxY = 1;
}

ViewOGL1D::~ViewOGL1D()
{
    if(windowNumber != 0)
    {
        glutDestroyWindow(windowNumber);
    }
}

ViewComponent* ViewOGL1D::parse(const tinyxml2::XMLElement* node)
{
	ViewOGL1D* s = new ViewOGL1D();
    for(const tinyxml2::XMLElement* e = node->FirstChildElement(); e !=  NULL; e = e->NextSiblingElement())
    {
        if(strcmp("windowwidth", e->Value()) == 0)
        {
            s->windowWidth = atoi(e->GetText());
        }
        else if(strcmp("windowheight", e->Value()) == 0)
        {
            s->windowHeight = atoi(e->GetText());
        }
        else if(strcmp("windowtitle", e->Value()) == 0)
        {
            s->windowTitle = string(e->GetText());
        }
		else if (strcmp("worldbuffersize", e->Value()) == 0)
		{
			s->worldBufferSize = atoi(e->GetText());
		}
        else
        {
            Console::Stream << "OGL Parse Not Registered" << e->Value() << endl;
            Console::writeAsError();
        }
    }
    return s;
}

void ViewOGL1D::executeupdate()
{
	//Iterate throught world and call drawrectangle
	Console::Stream << "View OpenGL1D" << endl;
	Console::writeAsMessage();

	IWorld* W;
	View* V;
	if ((W = World::getInstance()) && (V = View::getInstance()))
	{
		//Erstelle permanente Kopie der Welt
		vector<PointNode*>* matrix = W->getPointNodesMatrix();
		vector<PointNode> worldCopy;
		for (unsigned int i = 0; i < matrix->size(); i++)
		{
			worldCopy.push_back(*matrix->at(i));
		}

		//Ist der Buffer voll so werfe die älterste Welt weg
		if (matrixBuffer.size() > worldBufferSize)
		{
			matrixBuffer.pop_front();
		}
		matrixBuffer.push_back(worldCopy);

		//Get Sizes
		uint_fast8_t PixelSize = V->getPixelSize();
		uint_fast8_t GridSize = V->getGridSize();

		//Create Window If no Exists
		if (windowNumber == 0)
		{
			if (!createwindow())
			{
				//Unable to Create Window
				return;
			}

			// Set Background color to GridColor
			vector<unsigned char>* GridColor = V->getColorRGBGrid();
			if (GridColor)
			{
				glClearColor(GridColor->at(0), GridColor->at(1), GridColor->at(2), 0.0f);
			}
			// Set Background to Black
			else
			{
				glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			}

			//Calculate Coordinate Max Sizes
			vector<uint_fast32_t>* Size = W->getSize();
			if (Size)
			{
				coordinateMaxX = Size->at(0) * (PixelSize + GridSize) - GridSize;
				coordinateMaxY = worldBufferSize * (PixelSize + GridSize) - GridSize;
			}
			else
			{
				Console::Stream << "World Size Null" << endl;
				Console::writeAsError();
			}

			glPushMatrix();

			//glViewport(0, 0, CoordinateMaxX, CoordinateMaxY);

			//Move 0,0 to -1,-1
			glTranslated(-1, -1, 0);

			//Scale Drawing to Fit in Window
			glScaled(2.0 / coordinateMaxX, 2.0 / coordinateMaxY, 1);
		}

		//Initialisiere neuen Frame
		glClear(GL_COLOR_BUFFER_BIT);
			
		glPushMatrix();

		//Set Draw Mode
		glBegin(GL_QUADS);

		//Itteriere ueber gesammte Welt
		deque<vector<PointNode>>::iterator bufferIter;
		vector<PointNode>::iterator iter;
		uint_fast32_t matrixCounter = 0;

		//Zeichne alle Welten im Buffer
		for (bufferIter = matrixBuffer.begin(); bufferIter < matrixBuffer.end(); bufferIter++, matrixCounter++)
		{
			for (iter = bufferIter->begin(); iter < bufferIter->end(); iter++)
			{
				uint_fast32_t cord = iter->getPosition()->at(0);
				vector<unsigned char>* PointColor;
				if ((PointColor = V->getColorRGBPixel(iter->getActiveValue())))
				{
					glColor3f(PointColor->at(0), PointColor->at(1), PointColor->at(2));
					//Zeichnen des Rechtecks in Image

					//Aktualisieren des PointCache
					uint_fast32_t BaseX = cord * (PixelSize + GridSize);
					uint_fast32_t BaseY = matrixCounter * (PixelSize + GridSize);

					//Left Top
					glVertex2i(BaseX, BaseY);

					//Left Bottom
					glVertex2i(BaseX, BaseY + PixelSize);

					//Right Bottom
					glVertex2i(BaseX + PixelSize, BaseY + PixelSize);

					//Right Top
					glVertex2i(BaseX + PixelSize, BaseY);
				}
			}
		}
	}

	//Aktualisiere Anzeige
	glEnd();
	glPopMatrix();
	glutSwapBuffers();

	//Sleep(100);
}

void ViewOGL1D::finish()
{
    if(windowNumber != 0)
    {
        glutDestroyWindow(windowNumber);
    }
}

bool ViewOGL1D::createwindow()
{
    try
    {
        //GlutInit need the main Args
        //Cheat Some
        char* argv = "E";
        int argc = 1;

        glutInit(&argc, &argv);
        glutInitDisplayMode(GLUT_RGBA);
        glutInitWindowSize(windowWidth, windowHeight);
        glutInitWindowPosition(windowX, windowY);
        windowNumber = glutCreateWindow(windowTitle.c_str());

        return true;
    }
    catch (...)
    {
        return false;
    }
}