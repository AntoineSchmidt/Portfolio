///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:28
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef VIEWCOMPONENT_H
#define VIEWCOMPONENT_H

#pragma once

/**
* ViewComponent
* Funktionalitaet fuer verschiedene View Methoden
*/
class ViewComponent
{
public:
    /**
    * Prueft auf notwendigkeit des viewcomponentupdate durch UpdatePerFrame
    * bei jedem Aufruf wird executeEveryStep aufgerufen
    * Ruft bei notwendigkeit executeupdate Auf
    */
    void update();

    /**
    * fuehrt tatsaechliche funktion der ViewComponente aus
    */
    virtual void executeupdate() =0;

    /**
    *wird jeden schritt ausgefuehrt
    */
    virtual void executeEveryStep();

    /**
    * Befreit alle Genutzten Resourcen
    */
    virtual void finish() =0;

protected:
    /**
    * Der geschuetzte Konstruktor da Klasse als Interface Implementiert
    */
    ViewComponent();

    /**
    * Der geschuetzte Destruktor da Klasse als Interface Implementiert
    */
    virtual ~ViewComponent();

    /**
    * Die Anzahl wie oft executeupdate aufgerufen wird wird durch jede Interface Implementierung aus der Xml geparst
    * Default 1
    */
    int UpdatePerFrame;

};

#endif // VIEWCOMPONENT_H