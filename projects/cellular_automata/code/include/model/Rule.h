///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt, Michell Mueller
///////////////////////////////////////////////////////////

#ifndef RULE_H
#define RULE_H

#pragma once

#include <string>
#include <vector>
#include <string>
#include <cstdint>

using namespace std;

/**
* Klasse fuer eine geparste Rule aus einem Tag
*/
class Rule
{
public:
    /**
    * Rule Konstruktor
    * @param string* Die Rule ID
    * @param bool true wenn die Rule Verwendet werden soll
    */
    Rule(string* Id, bool Use = true);

    /**
    * Gibt das Translations-Ergebnis der Rule zurueck
    * @return uint_fast16_t Das Translations-Ergebnis
    */
    uint_fast16_t getValue() const;

    /**
    * Gibt das Translations-Ergebnis der Rule zurueck
    * @param vector<uint_fast16_t>* Die Translations Werte
    * @return uint_fast16_t Das Translations-Ergebnis
    */
    uint_fast16_t getValue(const vector<uint_fast16_t>* Values) const;

    /**
    * Setzt das Ergebnis der Rule
    * @param uint_fast16_t Das Ergebnis
    */
    void setSingleValue(const uint_fast16_t Value);

    /**
    * Setzt als Ergebnis der Rule einen ganzen vector
    * @param vector<uint_fast16_t> Die moeglichen Ergebnisse
    */
    void setValue(vector<uint_fast16_t>* Value);

    /**
    * Gibt alle Translationen aus
    * @return vector<vector<uint_fast16_t>*>* Alle Translationen
    */
    vector<vector<uint_fast16_t>*>* getValues() const;

    /**
    * Fuegt einen Translations Vector hinzu
    * @param vector<uint_fast16_t>* Die neue Translation
    */
    void addValues(vector<uint_fast16_t>* Values);

    /**
    * Setzt Translations Vector
    * @param vector<vector<uint_fast16_t>*>* Die neuen Translationen
    */
    void setValues(vector<vector<uint_fast16_t>*>* Values);

    /**
    * Gibt die Rules ID zurueck
    * @return string* Die Rule ID
    */
    string* getId() const;

    /**
    * Gibt zurueck ob Rule verwendet wird
    * @return bool true wenn Rule verwendet wird
    */
    bool getUsed() const;

    /**
    * Setzt den Used Wert
    * @param bool Soll Rule verwendet werden
    */
    void setUsed(bool Use);

    /**
    * Setzt den ErgebnisPosition Wert fuer (z.B sexy Loop) (-1, Deativieren)
    * @param int Die Ergebnis Position im Translations Vector
    */
    void setTranslationPosition(int VectorPosition);

private:
    /**
    * Alle Translationen die zu dieser Rule fuehren
    */
    vector<vector<uint_fast16_t>*>* Values;

    /**
    * Alle moeglichen Ergebnis Werte der Rule
    */
    vector<uint_fast16_t>* Value;

    /**
    * Die Rule ID
    */
    string* id;

    /**
    * Wird die Rule verwendet
    */
    bool use;

    /**
    * Die Translations Position als Translation Ergebnis
    */
    int ergebnisPosition;

};

#endif // RULE_H