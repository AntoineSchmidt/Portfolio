///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <cstdint>
#include "model/Rule.h"
#include "model/RulePart.h"

RulePart::RulePart()
{
    ruleParts = new map<uint_fast16_t, RulePart*>();
    outCome = NULL;
}

uint_fast16_t RulePart::search(const vector<uint_fast16_t>* Values, uint_fast8_t Position)
{
    if(Position >= Values->size())
    {
        if(outCome)
        {
            //Neue Anpassung fuer (z.B sexy Loop)
            return outCome->getValue(Values);
        }
    }
    else
    {
        map<uint_fast16_t, RulePart*>::iterator iter;
        iter = ruleParts->find(Values->at(Position));
        if (iter != ruleParts->end())
        {
            return iter->second->search(Values, ++Position);
        }
    }
    // Rule not Found return center Value
    return Values->at(0);
}

bool RulePart::insert(const vector<uint_fast16_t>* Values, Rule* Result, uint_fast8_t Position)
{
    if(Position >= Values->size())
    {
        outCome = Result;
        return true;
    }
    else
    {
        map<uint_fast16_t, RulePart*>::iterator iter;
        iter = ruleParts->find(Values->at(Position));
        if (iter != ruleParts->end())
        {
            return iter->second->insert(Values, Result, ++Position);
        }
        else
        {
            RulePart* New = new RulePart();
            ruleParts->insert(pair<uint_fast16_t, RulePart*>(Values->at(Position), New));
            return New->insert(Values, Result, ++Position);
        }
    }
}

bool RulePart::setSingleValue(const vector<uint_fast16_t>* Values, const uint_fast16_t Result, uint_fast8_t Position)
{
    if(Position >= Values->size())
    {
        if(outCome)
        {
            outCome->setSingleValue(Result);
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        map<uint_fast16_t, RulePart*>::iterator iter;
        iter = ruleParts->find(Values->at(Position));
        if (iter != ruleParts->end())
        {
            return iter->second->setSingleValue(Values, Result, ++Position);
        }
        else
        {
			return false;
        }
    }
}

bool RulePart::setValue(const vector<uint_fast16_t>* Values, vector<uint_fast16_t>* Result, uint_fast8_t Position) {
	if (Position >= Values->size())
	{
		if (outCome)
		{
			outCome->setValue(Result);
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		map<uint_fast16_t, RulePart*>::iterator iter;
		iter = ruleParts->find(Values->at(Position));
		if (iter != ruleParts->end())
		{
			return iter->second->setValue(Values, Result, ++Position);
		}
		else
		{
			return false;
		}
	}
}