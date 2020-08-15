///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt, Michell Mueller, Bonifaz Stuhr, Tobias Eigler
///////////////////////////////////////////////////////////

#ifndef RULES_H
#define RULES_H

#pragma once

#include <map>
#include <iostream>
#include <string>
#include <cstdint>
#include "model/IRules.h"
#include "model/Rule.h"
#include "model/RulePart.h"
#include "library/tinyxml2.h"

using namespace std;

class Rules : public IRules
{
private:
	//Konstruktor
	Rules();

	//Die Singleton Instanz der Klasse.
	static Rules* instance;

	//Symetrie angabe des jeweiligen Regelsatzes. z.B: "rotate4" fuer Langton.
	string* symmetries;

	//Regeln fuer die Translation.
	RulePart* rulesByValues;
	
	//Regeln fuer ID
	map<string, Rule*>* rulesById;

	//Alle Regeln.
	vector<Rule*>* allRules;

	//Die Anordnung der Translation.
	vector<uint_fast8_t>* ruleOrder;

public:
	//Destruktor.
    ~Rules();

	/*
	* Erzeugt die Singleton Instanz der Rule Klasse und parst die Daten aus dem XML-Dokument.
	*@param const tinyxml2::XMLElement*, node Rule XML-Knoten aus dem XML-File.
	*/
    static void parse(const tinyxml2::XMLElement* node);

	/*
	* Methode um die Singleton Instanz zu bekommen.
	* @return Rule*, Liefert Pointer auf die Instanz der Ruleklasse.
	*/
    static Rules* getInstance();

	/*
	* Liefert einen uint_fast16_t Integern, welcher das Ergebniss aus den Regeln beinhaltet.
	* Bsp: Regel:	00001 ->Ergebniss: 2
	* @param const vector<uint_fast16_t>* Values, Werte der Regeln.
	* @return uint_fast16_t, Ergebniss.
	*/
    uint_fast16_t getRuleValue(const vector<uint_fast16_t>* values) const;

	/*
	* Liefert einen vektor der die Anordnung der Translation enthaelt, die bei der Eingabe definiert wurde oder die Standard Order.
	* @return vector<uint_fast8_t>*, Vektorpointer auf die RuleOrder.
	*/
    vector<uint_fast8_t>* getRuleOrder() const;

	/*
	* Liefert einen Pointer auf eine Rule, ueber die Rule ID.
	* @param const string* Id, ID der Rule aus dem XML-File.
	* @return Rule*, Pointer auf eine Rule.
	*/
    Rule* getRuleById(const string* id) const;

	/*
	* Dient zur Aktivierung einer Rule, mittels der Rule ID.
	* @param const string* Id, ID der Rule.
	* @return bool, Bool Wert, ob die Aktivierung erfolgreich war.
	*/
    bool activateRuleById(const string* id);

	/*
	* Dient zur Deaktivierung einer Rule, mittels der Rule ID.
	* @param const string* Id, ID der Rule.
	* @return bool, Bool Wert, ob die Deaktivierung erfolgreich war.
	*/
    bool deactivateRuleById(const string* id);

	/*
	* Setzen einer Regel unter Verwendung dieser Regel als Key.
	* @param const vector<uint_fast16_t>* Values, Die Regeln.
	* @param const uint_fast16_t Value, Ergebnis der Regel.
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    bool setRuleByValues(const vector<uint_fast16_t>* values, const uint_fast16_t value);

	/*
	* Setzen einer Regel unter Verwendung dieser Regel als Key. Die Rule kann zu mehreren Ergebnissen fuehren.
	* @param const vector<uint_fast16_t>* Values, Die Regeln.
	* @param vector<uint_fast16_t>* Value, Ergebnisse der Regel.
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    bool setRuleByValues(const vector<uint_fast16_t>* values, vector<uint_fast16_t>* value);

	/*
	* Setzten einer Regel anhand der Rule ID.
	* @param const string* Id, ID der Rule.
	* @param const uint_fast16_t Value, Ergebnis der Regel.
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    bool setRuleById(const string* id, const uint_fast16_t value);

	/*
	* Setzten einer Regel anhand der Rule ID. Die Rule kann zu mehreren Ergebnissen fuehren.
	* @param const string* Id, ID der Rule.
	* @param vector<uint_fast16_t>* Value, Ergebnisse der Regel.
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    bool setRuleById(const string* id, vector<uint_fast16_t>* value);

	/*
	* Liefert alle Regeln des jeweiligen Regelsatzes.
	* @return vector<Rule*>*, vector mit allen Regeln.
	*/
    vector<Rule*>* getAllRules() const;

};

#endif // RULES_H