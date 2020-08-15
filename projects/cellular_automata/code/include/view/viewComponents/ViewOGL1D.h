///////////////////////////////////////////////////////////
//  Uses OpenGl / Glut
//  Programmed with FreeGlut
//  Compiler Flags -lopengl32 -lglu32 -lgdi32
//  Author:         Bonifaz Stuhr
///////////////////////////////////////////////////////////

#ifndef VIEWOGL1D_H
#define VIEWOGL1D_H

#pragma once

#include <vector>
#include "view/viewComponents/ViewComponent.h"
#include "library/tinyxml2.h"
#include <deque>
#include "model/PointNode.h"

using namespace std;

class ViewOGL1D : public ViewComponent
{
private:
	ViewOGL1D();
	bool createwindow();

	string windowTitle;
	int windowWidth;
	int windowX;
	int windowHeight;
	int windowY;
	int windowNumber;
	unsigned int worldBufferSize;
	deque<vector<PointNode>> matrixBuffer;

	double coordinateMaxX;
	double coordinateMaxY;

	bool bcreatewindow;

public:
    ~ViewOGL1D();

    static ViewComponent* parse(const tinyxml2::XMLElement* node);
    void executeupdate();
    void finish();

};

#endif // VIEWOGL1D_H