///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include "model/RulesCalculator.h"
#include "library/utils.h"

RulesCalculator::RulesCalculator(const map<string, vector<uint_fast16_t>*>* Variables)
{
    this->variables = Variables;
}

RulesCalculator::~RulesCalculator()
{
    //delete Variables;
}

void RulesCalculator::calculate(const vector<string>* Translation)
{
    counterInit.clear();
    translationValues.clear();
    this->translation = *Translation;
    for(unsigned int i = 0; i < Translation->size(); i++)
    {
        // Is a Variable
        if(isdigit(Translation->at(i).at(0)) == 0)
        {
            counterInit.push_back(variables->at(Translation->at(i))->size());
            translationValues.push_back(0);
        }
        // Isnt a Variable
        else
        {
            counterInit.push_back(0);
            translationValues.push_back(utils::stoi(Translation->at(i)));
        }
    }
    counter = counterInit;
    calculatedEveryPossibleTranslation = false;
}

vector<uint_fast16_t>* RulesCalculator::getNext()
{
    if(!calculatedEveryPossibleTranslation)
    {
        vector<uint_fast16_t>* ReturnVal = new vector<uint_fast16_t>(translation.size());
        for(unsigned int i = 0; i < translation.size(); i++)
        {
            // Isnt a Variable
            if(counter.at(i) == 0)
            {
                ReturnVal->at(i) = translationValues.at(i);
            }
            // Is a Variable
            else
            {
                ReturnVal->at(i) = variables->at(translation.at(i))->at(counter.at(i) - 1);
            }
        }

        //Update Counter
        bool DecreaseNext = true;
        for(unsigned int i = 0; i < counterInit.size(); i++)
        {
            if(counterInit.at(i) != 0 && DecreaseNext)
            {
                if(--counter.at(i) == 0)
                {
                    counter.at(i) = counterInit.at(i);
                    DecreaseNext = true;
                }
                else
                {
                    DecreaseNext = false;
                    break;
                }
            }
        }
        if(DecreaseNext)
        {
            calculatedEveryPossibleTranslation = true;
        }
        return ReturnVal;
    }
    else
    {
        return NULL;
    }
}