///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <random>
#include <map>
#include "model/Rule.h"
#include "controller/Console.h"
#include "model/IRules.h"
#include "model/Rules.h"

Rule::Rule(string* Id, bool Use)
{
	this->Values = NULL;
	this->Value = NULL;
	this->id = Id;
	this->use = Use;
	this->ergebnisPosition = -1;
}

uint_fast16_t Rule::getValue() const
{
	if(Value)
	{
		int valueSize = Value->size();
		// Es handelt sich um eine Variable -> Random-Wert zurueckgeben
		if (valueSize > 1)
		{
			random_device rd;     // Engine initialisieren
			mt19937 rng(rd());    // Random-Enigine (Mersenne-Twister)
			uniform_int_distribution<int> uni(0, valueSize - 1);

			uint_fast16_t random_integer = uni(rng);
			Value->at(random_integer);
		}
		else if(valueSize == 1)// Normale Zahl
		{
			return Value->at(0);
		}
		else
		{
			Console::Stream << "Empty Rule Value Vector" << endl;
			Console::writeAsError();
			return 0;
		}
	}
	else
	{
		Console::Stream << "Rule Value Vector Null" << endl;
		Console::writeAsException();
		return 0;
	}
	return 0;
}

uint_fast16_t Rule::getValue(const vector<uint_fast16_t>* Values) const
{
	// Das Ergebnis ist abhaengig von Values
	if(ergebnisPosition >= 0)
	{
		if((int)Values->size() > ergebnisPosition)
		{
			return Values->at(ergebnisPosition);
		}
		//ungueltige ergebnisPosition
		else
		{
			Console::Stream << "Rule Bad Result Position" << endl;
			Console::writeAsError();

			// Bad Translations Position return center Value
			return Values->at(0);
		}
	}
	else
	{
		return getValue();
	}
}

void Rule::setSingleValue(const uint_fast16_t Value)
{
	if(!this->Value)
	{
		this->Value = new vector<uint_fast16_t>();
	}
	this->Value->clear();
	this->Value->push_back(Value);
}

void Rule::setValue(vector<uint_fast16_t>* Value)
{
	this->Value = Value;
}

vector<vector<uint_fast16_t>*>* Rule::getValues() const
{
	return Values;
}

void Rule::setValues(vector<vector<uint_fast16_t>*>* Values)
{
	this->Values = Values;
}

void Rule::addValues(vector<uint_fast16_t>* Values)
{
	if(!this->Values)
	{
		this->Values = new vector<vector<uint_fast16_t>*>();
	}
	this->Values->push_back(Values);
}

string* Rule::getId() const
{
	return id;
}

bool Rule::getUsed() const
{
	return use;
}

void Rule::setUsed(bool Use)
{
	this->use = Use;
}

void Rule::setTranslationPosition(int VectorPosition)
{
	this->ergebnisPosition = VectorPosition;
}