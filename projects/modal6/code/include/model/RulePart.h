///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef RULEPART_H
#define RULEPART_H

#pragma once

#include <vector>
#include <cstdint>
#include <map>
#include "model/Rule.h"

/**
* Ersatz fuer map mit vector als Key schneller
*/
class RulePart
{
private:
	/**
	* Die Translations Werte und dazu entsprechender RulePart der Kind-Knoten
	*/
	map<uint_fast16_t, RulePart*>* ruleParts;

	/**
	* Die Rule Klasse gesetzt bei End-Knoten
	*/
	Rule* outCome;

public:
    /**
    * Der Konstruktor
    */
    RulePart();

    /**
    * Sucht Values in RuleParts
    * @param vector<uint_fast16_t>* Values die zu suchenden Werte
    * @param uint_fast8_t die Rekursive Position Default 0 nur intern zu verwenden
    * @return uint_fast16_t der gefundene Rule Wert
    */
    uint_fast16_t search(const vector<uint_fast16_t>* Values, uint_fast8_t Position = 0);

    /**
    * Fuegt Values in RuleParts ein
    * @param vector<uint_fast16_t>* Values die zu einfuegenden Werte
    * @param Rule* Die Rule zu Values Translation
    * @param uint_fast8_t die Rekursive Position Default 0 nur intern zu verwenden
    * @return bool true bei erfolgreichem einfuegen
    */
    bool insert(const vector<uint_fast16_t>* values, Rule* result, uint_fast8_t position = 0);

    /**
    * Setzt einzelnen Wert als Rule Value
    * @param vector<uint_fast16_t>* Values die translations Werte
    * @param uint_fast16_t der neue Rule Wert
    * @param uint_fast8_t die Rekursive Position Default 0 nur intern zu verwenden
    * @return bool true bei erfolgreichem setzen
    */
    bool setSingleValue(const vector<uint_fast16_t>* Values, const uint_fast16_t Result, uint_fast8_t Position = 0);

    /**
    * Setzt vector Wert als Rule Value
    * @param vector<uint_fast16_t>* Values die translations Werte
    * @param vector<uint_fast16_t>* der neue Rule Wert
    * @param uint_fast8_t die Rekursive Position Default 0 nur intern zu verwenden
    * @return bool true bei erfolgreichem setzen
    */
    bool setValue(const vector<uint_fast16_t>* Values, vector<uint_fast16_t>* Result, uint_fast8_t Position = 0);

};

#endif // RULEPART_H