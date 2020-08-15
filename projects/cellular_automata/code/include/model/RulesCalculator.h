///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt, Michell Mueller
///////////////////////////////////////////////////////////

#ifndef RULESCALCULATOR_H
#define RULESCALCULATOR_H

#include <string>
#include <cstdint>
#include <vector>
#include <map>

using namespace std;

/**
* Generiert alle moeglichen Translationen aus Variablen und Translationen
*/
class RulesCalculator
{
private:
    /**
    * Alle Variablen
    */
    const map<string, vector<uint_fast16_t>*>* variables;

    /**
    * Vector mit den Maximal Werten der Variabel Anzahlen
    */
    vector<uint_fast16_t> counterInit;

    /**
    * Auf diesem Vector wird heruntergezaehlt fuer die naechste Translation
    */
    vector<uint_fast16_t> counter;

    /**
    * Die ueber Calculate uebergebene Translation mit Variablen
    */
    vector<string> translation;

    /**
    * Die vorkalkulierten werte die in der Translation keine Variable sind
    */
    vector<uint_fast16_t> translationValues;

    /**
    * Kontroll Flag ist wahr wenn alle moeglichekeiten ueber getNext ausgegeben wurden
    */
    bool calculatedEveryPossibleTranslation;

public:
	/**
	* Konstruktor
	* @param map<string, vector<uint_fast16_t>*>* Variables alle Variablen
	*/
	RulesCalculator(const map<string, vector<uint_fast16_t>*>* Variables);

	/**
	* Destuktor
	*/
	~RulesCalculator();

	/**
	* Initialisiert neue Translations berechnungen
	* @param vector<string>* Translation die Translation mit Variablen
	*/
	void calculate(const vector<string>* Translation);

	/**
	* Gibt neue Kombination von Translation und Variablen zurueck
	* @return vector<uint_fast16_t>* neue Translation
	*/
	vector<uint_fast16_t>* getNext();

};

#endif // RULESCALCULATOR_H