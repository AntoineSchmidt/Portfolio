///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include "model/IRules.h"
#include "controller/Console.h"
#include "benchmark/Benchmark.h"

IRules::IRules()
{
	variables = new map<string, vector<uint_fast16_t>*>();
}

map<string, vector<uint_fast16_t>*>* IRules::getVariables() const
{
	return variables;
}

vector<uint_fast16_t>* IRules::getVariable(const string* Variable) const
{
	if (variables)
	{
		map<string, vector<uint_fast16_t>*>::iterator it;
		it = variables->find(*Variable);
		if (it != variables->end())
		{
			return it->second;
		}
		else
		{
			Console::Stream << "RuleMapValue::setValues: Variable konnte nicht gefunden werden." << endl;
			Console::writeAsError();
			return NULL;
		}
	}
	else
	{
		Console::Stream << "RuleMapValue Null" << endl;
		Console::writeAsException();
		return NULL;
	}
}

void IRules::getRuleValueSpeedTest() const
{
	vector<uint_fast16_t>* Values = new vector<uint_fast16_t>();
	Values->push_back(2);
	Values->push_back(1);
	Values->push_back(2);
	Values->push_back(2);
	Values->push_back(6);

	//Benchmark::getInstance()->start("getRuleValueSpeedTest");
	for(int i = 0; i < 10; i++)
	{
		getRuleValue(Values);
	}
	//Benchmark::getInstance()->stop("getRuleValueSpeedTest", "10x 2,1,2,2,6");
}