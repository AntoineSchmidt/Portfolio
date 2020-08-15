///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef VIEWCONSOLE_H
#define VIEWCONSOLE_H

#pragma once

#include "ViewComponent.h"
#include "library/tinyxml2.h"

/**
* Klasse fuer die Welt-Ausgabe in der Console
*/
class ViewConsole : public ViewComponent
{
public:
    /**
    * Der Destruktor
    */
    virtual ~ViewConsole();

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
    ViewConsole();

};

#endif // VIEWCONSOLE_H