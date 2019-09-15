///////////////////////////////////////////////////////////
//  Uses OpenGl / Glut
//  Programmed with FreeGlut
//  Compiler Flags -lopengl32 -lglu32 -lgdi32
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef VIEWOGL_H
#define VIEWOGL_H

#pragma once

#include <vector>
#include <boost/thread.hpp>
#include "view/viewComponents/ViewComponent.h"
#include "view/viewComponents/ViewOGLWindow.h"
#include "library/tinyxml2.h"
#include "view/View.h"
#include "model/World.h"

using namespace std;

/**
* Klasse fuer die Welt-Ausgabe mit OpenGl
*/
class ViewOGL : public ViewComponent
{
private:
	/**
	* Privater Konstruktor Instanziieren mit Parse Funktion
	*/
	ViewOGL();

	/**
	* Das OpenGl Fenster
	*/
	ViewOGLWindow* window;

	/**
	* Soll ein Fenster erstellt werden
	*/
	bool createWindow;

	/**
	* Der Container fuer den glutMainLoop Thread;
	*/
	boost::thread_group tg;

public:
	/**
	* Der Destruktor
	*/
	virtual ~ViewOGL();

	/**
	* Initialisieren der Klasse mit Daten der Xml
	* @param XMLElement* Xml Node mit Daten
	* @return ViewComponent* Die erstellte Instanz
	*/
	static ViewComponent* parse(const tinyxml2::XMLElement* node);

	/**
	* fuehrt tatsaechliche funktion der ViewComponente aus
	*/
	void executeupdate();

	/**
	* Befreit alle Genutzten Resourcen
	*/
	void finish();

};

#endif // VIEWOGL_H