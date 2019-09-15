///////////////////////////////////////////////////////////
//  Created on:      19-Apr-2016
//  Original author: Tobias Eigler
///////////////////////////////////////////////////////////

#ifndef IRULES_H
#define IRULES_H

#pragma once

#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>

using namespace std;

class Rule;

/*
Arbeitsschnittstelle/Abstrakte Klasse fuer verschiedene Regelsaetze.

Beinhaltet die Regeln der Langton Schleifen.
Regeln werden Gelesen: CTRBL -> (Center, Top, Right, Bottom, Left)
Bsp: Regel:	00001 ->Ergebniss: 2
*/
class IRules
{
public:
	/*
	* Liefert einen uint_fast16_t Integern, welcher das Ergebniss aus den Regeln beinhaltet.
	* Bsp: Regel:	00001 ->Ergebniss: 2
	* getRuleValue ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const vector<uint_fast16_t>* Values, Werte der Regeln.
	* @return uint_fast16_t, Ergebniss.
	*/
    virtual uint_fast16_t getRuleValue(const vector<uint_fast16_t>* Values) const = 0;

	/*
	* Liefert einen vektor der die Anordnung der Translation enthaelt, die bei der Eingabe definiert wurde oder die Standard Order.
	* getRuleOrder ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @return vector<uint_fast8_t>*, Vektorpointer auf die RuleOrder.
	*/
    virtual vector<uint_fast8_t>* getRuleOrder() const = 0;

	/*
	* Liefert einen Pointer auf eine Rule, ueber die Rule ID.
	* getRuleById ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const string* Id, ID der Rule aus dem XML-File.
	* @return Rule*, Pointer auf eine Rule.
	*/
    virtual Rule* getRuleById(const string* Id) const = 0;

	/*
	* Dient zur Aktivierung einer Rule, mittels der Rule ID.
	* activateRuleById ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const string* Id, ID der Rule.
	* @return bool, Bool Wert, ob die Aktivierung erfolgreich war.
	*/
    virtual bool activateRuleById(const string* Id) = 0;

	/*
	* Dient zur Deaktivierung einer Rule, mittels der Rule ID.
	* deactivateRuleById ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const string* Id, ID der Rule.
	* @return bool, Bool Wert, ob die Deaktivierung erfolgreich war.
	*/
    virtual bool deactivateRuleById(const string* Id) = 0;

	/*
	* Setzen einer Regel unter Verwendung dieser Regel als Key.
	* setRuleByValues ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const vector<uint_fast16_t>* Values, Die Regeln. 
	* @param const uint_fast16_t Value, Ergebnis der Regel. 
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    virtual bool setRuleByValues(const vector<uint_fast16_t>* Values, const uint_fast16_t Value)= 0;

	/*
	* Setzen einer Regel unter Verwendung dieser Regel als Key. Die Rule kann zu mehreren Ergebnissen fuehren.
	* setRuleByValues ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const vector<uint_fast16_t>* Values, Die Regeln. 
	* @param vector<uint_fast16_t>* Value, Ergebnisse der Regel.
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    virtual bool setRuleByValues(const vector<uint_fast16_t>* Values, vector<uint_fast16_t>* Value)= 0;

	/*
	* Setzten einer Regel anhand der Rule ID.
	* setRuleById ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const string* Id, ID der Rule.
	* @param const uint_fast16_t Value, Ergebnis der Regel.
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    virtual bool setRuleById(const string* Id, const uint_fast16_t Value) = 0;

	/*
	* Setzten einer Regel anhand der Rule ID. Die Rule kann zu mehreren Ergebnissen fuehren.
	* setRuleById ist eine Interaface Methode die in Klassen, die diese Schnittstelle nutzen, implementiert werden muss.
	* @param const string* Id, ID der Rule.
	* @param vector<uint_fast16_t>* Value, Ergebnisse der Regel.
	* @return bool, Boolwert ob setzten der Regel erfolgreich war.
	*/
    virtual bool setRuleById(const string* Id, vector<uint_fast16_t>* Value) = 0;

	/*
	* Liefert die Map, die die Translationen enthaelt, auf die man uber die Rule ID zugreifen kann.
	* @return map<string, vector<uint_fast16_t>*>*, die Map, die die Translationen enthaelt, auf die man uber die Rule ID zugreifen kann.
	*/
    map<string, vector<uint_fast16_t>*>* getVariables() const;

	/*
	* Liefert die Translation fuer die gegebenen Rule ID.
	* @param const string* Variable, Rule ID.
	* @return vector<uint_fast16_t>* , vektor mit den Translation fuer die gegebenen Rule ID.
	*/
    vector<uint_fast16_t>* getVariable(const string* Variable) const;


	/*
	* Testen der Geschwindigkeit eines getRuleValue aufrufs.
	*/
    void getRuleValueSpeedTest() const;

protected:
	//Konstruktor
    IRules();

	//Translationen mit der Rule ID als Schluessel der Map.
    map<string, vector<uint_fast16_t>*>* variables;

};

#endif // IRULES_H