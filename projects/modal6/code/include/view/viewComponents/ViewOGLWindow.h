///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef VIEWOGLWINDOW_H
#define VIEWOGLWINDOW_H

#pragma once

#include <string>
#include "model/World.h"
#include "view/View.h"

using namespace std;

/**
* Klasse fuer die glutMainLoop das OpenGl Fenster
*/
class ViewOGLWindow
{
public:
	/**
	* Der Konstruktor
	*/
	ViewOGLWindow();

	/**
	* Der Destruktor
	*/
	virtual ~ViewOGLWindow();

	/**
	* Startet die MainLoop
	*/
	void run();

	/**
	* Update die Anzeige
	*/
	void update();

	/**
	* Setzt den Fenster Title
	* @param string title
	*/
	void setTitle(string title);

	/**
	* Setzt die X Position
	* @param int x
	*/
	void setX(int x);

	/**
	* Setzt die Y Position
	* @param int y
	*/
	void setY(int y);

	/**
	* Setzt die breite
	* @param int width
	*/
	void setWidth(int width);

	/**
	* Setzt die hoehe
	* @param int height
	*/
	void setHeight(int height);

	/**
	* Setzt die Fps
	* @param unsigned int fps
	*/
	void setFps(unsigned int fps);

	/**
	* Ob Fps Limit benutzt werden soll
	* @param bool use
	*/
	void setUseFps(bool use);

private:
	/**
	* Der Fenster Titel
	*/
	string title;

	/**
	* Die Fenster Bildschirm Position X
	*/
	int x;

	/**
	* Die Fenster Bildschirm Position Y
	*/
	int y;

	/**
	* Die Fenster Breite
	*/
	int width;

	/**
	* Die Fenster Hoehe
	*/
	int height;

	/**
	* Updatet die Anzeige
	*/
	static void render();

	/**
	* FPS Manager
	*/
    static void timer(int Value);

	/**
	* Updated ViewPort bei Fenster Resize
	*/
	static void reshape(int Width, int Height);

	/**
	* Haendelt die Keyboard eingabe
	*/
	static void keyboard(unsigned char Key, int MouseX, int Mousey);

	/**
	* Haendelt die Keyboard Funktionstasten eingabe
	*/
	static void keyboardSpecial(int Key, int MouseX, int Mousey);

	/**
	* Render wird ausgefuehrt
	*/
	static bool renderWorld;

	/**
	* Frames per Second
	*/
	static unsigned int fps;

	/**
	* Use fps Limiter
	*/
	bool useFps;

	/**
	* In Fullscreen
	*/
	static bool inFullscreen;

    /**
	* Erstellt Fenster
	*/
	bool createWindow();

	/**
	* Initialisiert Sicht
	*/
	bool initView(const IWorld* W, const View* V);

	/**
    * Wechselt den Anzeige Modus
    */
    static void changeDisplayMode();

	/**
	* Die Nummer des erstellten Fensters
	*/
	int windowNumber;

};

#endif // VIEWOGLWINDOW_H