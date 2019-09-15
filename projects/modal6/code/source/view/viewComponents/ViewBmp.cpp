///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include "view/viewComponents/ViewBmp.h"
#include "library/tinyxml2.h"
#include "library/utils.h"
#include "library/bitmap_image.hpp"
#include "model/World.h"
#include "view/View.h"
#include "model/Point.h"
#include "controller/Controller.h"
#include "controller/Console.h"

ViewBmp::ViewBmp()
{
	filename = string("alife-%STEP%.bmp");
	this->image = NULL;
}

ViewBmp::~ViewBmp()
{
	delete image;
}

//erstellt bmp view von bmp node in view node und gibt instanz zurueck
ViewComponent* ViewBmp::parse(const tinyxml2::XMLElement* node)
{
	ViewBmp* comp = new ViewBmp();
	for (const tinyxml2::XMLElement* e = node->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
	{
		if (strcmp("StepsPerImage", e->Value()) == 0)
		{
			comp->UpdatePerFrame = atoi(e->GetText());
		}
		else if (strcmp("OutputDirectory", e->Value()) == 0)
		{
			if (e->GetText())
			{
				comp->filepath = string(e->GetText());
			}
		}
		else if (strcmp("FileName", e->Value()) == 0)
		{
			if (e->GetText())
			{
				comp->filename = string(e->GetText());
			}
		}
		else
		{
			Console::Stream << "BMP Parse Not Registered" << e->Value() << endl;
			Console::writeAsError();
		}
	}
	return comp;
}

//executeupdated bmp wird nur dann ausgefuehrt wenn stepsperimage erreicht, bekommt stepscount
void ViewBmp::executeupdate()
{
	string CurrentFileName = getFile();

	Console::Stream << "View BMP " << CurrentFileName << endl;
	Console::writeAsMessage();

	World* W;
	View* V;
	if ((W = World::getInstance()) && (V = View::getInstance()))
	{
	    //Get Sizes
	    int PixelSize = V->getPixelSize();
	    int GridSize = V->getGridSize();

		//Initialisiere Bild wenn nicht vorhanden
		if (!this->image)
		{
			try
			{
				//Allocate
				vector<uint_fast32_t>* wsize = W->getSize();
				if (wsize && wsize->size() == 2)
				{
					uint_fast16_t Width = wsize->at(0) * (PixelSize + GridSize) - GridSize;
					uint_fast16_t Height = wsize->at(1) * (PixelSize + GridSize) - GridSize;
					this->image = new bitmap_image(Width, Height);
					if (GridSize > 0)
					{
						//Fill Image with Grid Color
						vector<unsigned char>* GridColor;
						if ((GridColor = V->getColorRGBGrid()))
						{
							image->set_region(0, 0, Width, Height, GridColor->at(0), GridColor->at(1), GridColor->at(2));
						}
						//Fill Image with Empty Value Pixels
						vector<unsigned char>* PixelColor;
						if ((PixelColor = V->getColorRGBPixel(Point::emptyValue)))
						{
							for (uint16_t X = 0; X < wsize->at(0); X++)
							{
								uint_fast32_t Xa = X * (PixelSize + GridSize);
								for (uint16_t Y = 0; Y < wsize->at(1); Y++)
								{
									uint_fast32_t Ya = Y * (PixelSize + GridSize);
									image->set_region(Xa, Ya, PixelSize, PixelSize, PixelColor->at(0), PixelColor->at(1), PixelColor->at(2));
								}
							}
						}
					}
				}
				else
				{
					Console::Stream << "Not Supported Dimension Count" << endl;
					Console::writeAsError();
					return;
				}
			}
			catch (bad_alloc)
			{
				Console::Stream << "Cant allocate, Image to Big" << endl;
				Console::writeAsException();
				return;
			}
		}

		//Warte bis BMP Speicher Thread Abgeschlossen hat bevor bild manipuliert wird
		tg.join_all();

		//Itteriere ueber gesammte Welt
		vector<PointNode*>::iterator iter;
		for (iter = W->getPointNodesMatrix()->begin(); iter != W->getPointNodesMatrix()->end(); iter++)
		{
			//If UpdatePerFrame > 1 Image isnt Synchronised
			if ((*iter) && ((*iter)->getChanged() || UpdatePerFrame > 1))
			{
				vector<uint_fast32_t>* cord = (*iter)->getPosition();
				uint_fast16_t PointValue = (*iter)->getActiveValue();
				vector<unsigned char>* PointColor;
				if ((PointColor = V->getColorRGBPixel(PointValue)))
				{
					//Zeichnen des Rechtecks in Image
					if (cord->size() == 2)
					{
						uint_fast32_t X = cord->at(0) * (PixelSize + GridSize);
						uint_fast32_t Y = cord->at(1) * (PixelSize + GridSize);
						image->set_region(X, Y, PixelSize, PixelSize, PointColor->at(0), PointColor->at(1), PointColor->at(2));
					}
				}
			}
		}
		//Lagere Speicher Prozess in anderen Thread
		tg.create_thread(boost::bind(&bitmap_image::save_image, image, CurrentFileName));
		//image->save_image(CurrentFileName);
	}
}

//gibt completen filepath mit manipuliertem file name zurueck
string ViewBmp::getFile()
{
	string temp_str = utils::to_string(Controller::getInstance()->getStepCount());
	return filepath + utils::replaceAll(filename, "%STEP%", temp_str);
}

//entsorgt und befreit alle ressourcen
void ViewBmp::finish()
{
	delete image;
}