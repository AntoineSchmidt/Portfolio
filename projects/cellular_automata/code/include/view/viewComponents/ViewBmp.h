///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef VIEWBMP_H
#define VIEWBMP_H

#pragma once

#include <boost/thread.hpp>
#include "ViewComponent.h"
#include "library/tinyxml2.h"
#include "library/bitmap_image.hpp"

/**
* Klasse fuer die Welt-Ausgabe als Bmp Datei
*/
class ViewBmp : public ViewComponent
{
public:
    /**
    * Der Destruktor
    */
    virtual ~ViewBmp();

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

private:
    /**
    * Privater Konstruktor Instanziieren mit Parse Funktion
    */
    ViewBmp();

    /**
    * Gibt aktuellen Datei Name aus
    * @return string Der Datei Name
    */
    string getFile();

    /**
    * Der Pfad in den die Datei gespeichert werden soll
    */
    string filepath;

    /**
    * Der Datei Name unter den die Datei gespeichert werden soll
    */
    string filename;

    /**
    * Der Bmp Cache
    */
    bitmap_image* image;

    /**
    * Der Container fuer den Bmp Speicher Thread
    */
    boost::thread_group tg;

};

#endif // VIEWCOMPONENT_H