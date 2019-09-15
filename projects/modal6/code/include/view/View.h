///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:28
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef VIEW_H
#define VIEW_H

#pragma once

#include <map>
#include <list>
#include <vector>
#include <iostream>
#include "library/tinyxml2.h"
#include "view/viewComponents/ViewComponent.h"
#include "controller/Settings.h"

using namespace std;

/**
* View-Klasse:
* Hat die Liste aller Initialisierten View-Komponenten
* Fuehrt bei ihnen die Update Funktion aus
*/
class View
{
public:
    /**
    * Der Dekonstruktor loescht Werte der Klasse
    */
    virtual ~View();

    /**
    * Geht die View-Komponenten liste duch und ruft von ihnen update auf
    * Wird nach jedem Schritt von der Controller Klasse Aufgerufen
    */
    void update();

    /**
    * Gibt View Singleton Instance zurueck
    */
    static View* getInstance();

    /**
    * Parsed Xmlnode und initialisiert entsprechende View Componenten durch das aufrufen der entsprechenden parse methode
    * und fuegt diese zu ViewComponents liste hinzu
    * Aufruf durch XmlHandler
    * @param XmlElement* Die View-Node der Xml
    */
    static void parse(const tinyxml2::XMLElement* node);

    /**
    * Gibt zur ColorID gespeicherten hex wert zurueck
    * @param uint_fast16_t Die Color ID, der Wert des Point
    * @return string* Color String aus der Xml gelesen
    */
    string* getColor(const uint_fast16_t ColorID) const;

    /**
    * Ruft getColor auf und gibt wert als int zurueck
    * @param uint_fast16_t die Color ID, der Wert des Point
    * @return int Color String aus der Xml in Int konvertiert
    */
    int getColorInt(const uint_fast16_t ColorID) const;

    /**
    * Ruft getColor auf und konvertiert den String in einen RGB Vector
    * dieser wird zusaetzlich in den RGB Cache gespeichert
    * @param uint_fast16_t die Color ID, der Wert des Point
    * @return vector<unsigned char>* der Xml String wert aus getColor in RGB
    */
    vector<unsigned char>* getColorRGBPixel(const uint_fast16_t ColorID);

    /**
    * Gibt den PixelSize geparst aus der Xml zurueck
    * @return uint_fast8_t Die Pixel Groesse
    */
    uint_fast8_t getPixelSize() const;

    /**
    * Gibt den PixelSize geparst aus der Xml
    * @return uint_fast8_t Die Pixel Groesse
    */
    vector<unsigned char>* getColorRGBGrid() const;

    /**
    * Gibt den GridSize geparst aus der Xml zureuck
    * @return uint_fast8_t Die Grid Groesse
    */
    uint_fast8_t getGridSize() const;

    /**
    * Befreit genutzte Ressourcen der Klasse
    * Ruft die finish funktion bei allen View-Komponenten auf
    */
    void finish();

private:
    /**
    * Privater Konstruktor, erstellen der Instanz mit parse
    */
    View();

    /**
    * Wird von Parse aufgerufen und Parsed die Farben
    * @param XmlElement* Color Node der Xml
    */
    void parseColors(const tinyxml2::XMLElement* node);

    /**
    * Die Liste aller geparster View-Komponenten
    */
    list <ViewComponent*>* viewComponents;

    /**
    * Alle Farben aus der Xml als String
    */
    map<uint_fast16_t, string*>* Colors;

    /**
    * Alle Farben aus der Xml als RGB Vectoren
    */
    map<uint_fast16_t, vector<unsigned char>*>* RGBColors;

    /**
    * geparste Grid Color als RGB Vector aus der Xml
    */
    vector<unsigned char>* RGBGridColor;

    /**
    * geparste Grid Size aus der Xml
    */
    uint_fast8_t GridSize;

    /**
    * geparste Pixel Size aus der Xml
    */
    uint_fast8_t PixelSize;

    /**
    * Die Singleton Instanz der View
    */
    static View* Instance;

};

#endif // VIEW_H