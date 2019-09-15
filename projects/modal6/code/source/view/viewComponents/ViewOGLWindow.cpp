///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include "view/viewComponents/ViewOGLWindow.h"
#include "controller/Console.h"

#ifdef __linux__
// OGL-Anzeige fuer Linux:
#include <GL/gl.h>
#include <GL/glut.h>
#else
// OGL-Anzeige fuer Windows:
#include "library/freeglut/glut.h"
#endif

bool ViewOGLWindow::renderWorld = false;
bool ViewOGLWindow::inFullscreen = false;
unsigned int ViewOGLWindow::fps = 60;

ViewOGLWindow::ViewOGLWindow()
{
	title = "ALife";

	x = 10;
	y = 10;
	height = 800;
	width = 1000;

	useFps = false;
}

ViewOGLWindow::~ViewOGLWindow()
{
	if(windowNumber != 0)
	{
		glutDestroyWindow(windowNumber);
	}
}

bool ViewOGLWindow::createWindow()
{
	try
	{
		//GlutInit need the main Args
		//Cheat Some
		char* argv = "E";
		int argc = 1;

		glutInit(&argc, &argv);
		glutInitDisplayMode(GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutInitWindowPosition(x, y);
		windowNumber = glutCreateWindow(title.c_str());

		return true;
	}
	catch (...)
	{
		windowNumber = 0;
		return false;
	}
}

bool ViewOGLWindow::initView(const IWorld* W, const View* V)
{
	//Get Sizes
	uint_fast8_t PixelSize = V->getPixelSize();
	uint_fast8_t GridSize = V->getGridSize();

	// Set Background color to GridColor
	vector<unsigned char>* GridColor = V->getColorRGBGrid();
	if(GridColor)
	{
		glClearColor(GridColor->at(0), GridColor->at(1), GridColor->at(2), 0.0f);
	}
	// Set Background to Black
	else
	{
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}

	//Calculate Coordinate Max Sizes
	double coordinateMaxX;
	double coordinateMaxY;
	vector<uint_fast32_t>* Size = W->getSize();

	if(Size)
	{
		coordinateMaxX = Size->at(0) * (PixelSize + GridSize) - GridSize;
		coordinateMaxY = Size->at(1) * (PixelSize + GridSize) - GridSize;
	}
	else
	{
		Console::Stream << "World Size Null" << endl;
		Console::writeAsError();
		return false;
	}

	glPushMatrix();

	//glViewport(0, 0, CoordinateMaxX, CoordinateMaxY);

	//Move 0,0 to -1,-1
	glTranslated(-1, -1, 0);

	//Scale Drawing to Fit in Window
	glScaled(2.0/coordinateMaxX, 2.0/coordinateMaxY, 1);

	return true;
}

void ViewOGLWindow::setTitle(string title)
{
	this->title = title;
}

void ViewOGLWindow::setX(int x)
{
	this->x = x;
}

void ViewOGLWindow::setY(int y)
{
	this->y = y;
}

void ViewOGLWindow::setWidth(int width)
{
	this->width = width;
}

void ViewOGLWindow::setHeight(int height)
{
	this->height = height;
}

void ViewOGLWindow::setFps(unsigned int fps)
{
	if(fps != 0)
	{
		this->fps = fps;
	}
}

void ViewOGLWindow::setUseFps(bool use)
{
	this->useFps = use;
}

void ViewOGLWindow::run()
{
	IWorld* W;
	View* V;
	if ((W = World::getInstance()) && (V = View::getInstance()))
	{
		if(!createWindow())
		{
			Console::Stream << "OpenGl Error Opening Window" << endl;
			Console::writeAsError();
			return;
		}

		if(!initView(W, V))
		{
			Console::Stream << "OpenGl Error init View" << endl;
			Console::writeAsError();
			return;
		}

		glutDisplayFunc(render);
		glutKeyboardFunc(keyboard);
		glutSpecialFunc(keyboardSpecial);
		if(useFps)
		{
			timer(0);
		}
		else
		{
			glutIdleFunc(render);
		}
		glutMainLoop();
	}
}

void ViewOGLWindow::update()
{
	renderWorld = true;
}

void ViewOGLWindow::timer(int Value)
{
	glutPostRedisplay();
	glutTimerFunc(1000/fps, timer, 0);
}

void ViewOGLWindow::render()
{
	if(renderWorld)
	{
		IWorld* W;
		View* V;
		if ((W = World::getInstance()) && (V = View::getInstance()))
		{
			//Get Sizes
			uint_fast8_t PixelSize = V->getPixelSize();
			uint_fast8_t GridSize = V->getGridSize();

			//Initialisiere neuen Frame
			glClear(GL_COLOR_BUFFER_BIT);

			glPushMatrix();

			//Set Draw Mode
			glBegin(GL_QUADS);

			//Itteriere ueber gesammte Welt
			vector<PointNode*>::iterator iter;
			for (iter = W->getPointNodesMatrix()->begin(); iter < W->getPointNodesMatrix()->end(); iter++)
			{
				if (*iter)
				{
					vector<uint_fast32_t>* cord = (*iter)->getPosition();
					vector<unsigned char>* PointColor;
					if ((PointColor = V->getColorRGBPixel((*iter)->getActiveValue())))
					{
						glColor3f(PointColor->at(0), PointColor->at(1), PointColor->at(2));
						//Zeichnen des Rechtecks in Image
						if (cord->size() == 2)
						{
							//Aktualisieren des PointCache
							uint_fast32_t BaseX = cord->at(0) * (PixelSize + GridSize);
							uint_fast32_t BaseY = cord->at(1) * (PixelSize + GridSize);

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

			renderWorld = false;
		}
	}
}

void ViewOGLWindow::reshape(int Width, int Height)
{
	/*
		glViewport(0, 0, (GLsizei)width, (GLsizei)height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0);
		glMatrixMode(GL_MODELVIEW);
		*/
}

void ViewOGLWindow::keyboard(unsigned char Key, int MouseX, int Mousey)
{
	if(Key == 'f')
	{
		changeDisplayMode();
	}
	else if(Key == 27) //ESC
	{
		//exit(0);
	}
}

void ViewOGLWindow::keyboardSpecial(int Key, int MouseX, int Mousey)
{
	if(Key == GLUT_KEY_F11)
	{
		changeDisplayMode();
	}
}

void ViewOGLWindow::changeDisplayMode()
{
	if(inFullscreen)
	{
		glutLeaveGameMode();
		inFullscreen = false;
	}
	else
	{
		glutEnterGameMode();
		inFullscreen = true;
	}
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(keyboardSpecial);
}