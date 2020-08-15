///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt, Bonifaz Stuhr, Tobias Eigler
///////////////////////////////////////////////////////////

#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <sstream>
#include <algorithm>
#include "model/Rules.h"
#include "controller/Console.h"
#include "model/Rule.h"
#include "model/RulePart.h"
#include "model/RulesCalculator.h"
#include "library/tinyxml2.h"

//GCC and Stoi
#include "library/utils.h"


Rules* Rules::instance = NULL;

Rules::Rules()
{
	rulesByValues = new RulePart();
	rulesById = new map<string, Rule*>();
	allRules = new vector<Rule*>();
	ruleOrder = NULL;
}

void Rules::parse(const tinyxml2::XMLElement* node)
{
	Console::Stream << "Parsing Rules" << endl;
	Console::writeAsMessage();

	//Create Temp Values
	string Segment;
	const char* Attribute;
	uint_fast8_t ExpectedTranslationSize;

	instance = new Rules();

	//Der Variable "symmetries" wird der Wert des Attributes "symmetries" aus dem XML-Dokument uebergeben.
	Attribute = NULL;
	if((Attribute = node->Attribute("symmetries")))
	{
		instance->symmetries = new string(Attribute);
	}
	else
	{
		Console::Stream << "XML(RULE): No \"symmetries\" given, setting to Default (rotate4)" << endl;
		Console::writeAsError();
		instance->symmetries = new string("rotate4");
	}

	//Extract RuleOrder
	//Set to Default
	Attribute = NULL;
	instance->ruleOrder = new vector<uint_fast8_t>();
	if((Attribute = node->Attribute("order")))
	{
		stringstream Stream(Attribute);

		//Auftrennen des Strings am "Komma"
		while (getline(Stream, Segment, ','))
		{
			instance->ruleOrder->push_back(utils::stoi(Segment));
		}
	}
	else
	{
		Console::Stream << "XML(RULE): No \"order\" given, setting to Default (2,1,3,0)/C(T,R,B,L)" << endl;
		Console::writeAsError();

		instance->ruleOrder->push_back(2);
		instance->ruleOrder->push_back(1);
		instance->ruleOrder->push_back(3);
		instance->ruleOrder->push_back(0);
	}
	ExpectedTranslationSize = instance->ruleOrder->size() + 1;


	/*
	Hier werden die einzelnen Regeln, die aus dem XML-Dokument stammen, gelesen und in das Datenelement map(rules)
	eingefuegt.
	*/

	//Parsen der Variablen
	RulesCalculator* Calcer = NULL;

	for (const tinyxml2::XMLElement* e = node->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
	{

		if (strcmp("Var", e->Value()) == 0)
		{
			// ID = Variablenname
			Attribute = NULL;
			if((Attribute = e->Attribute("id")))
			{
				const char* Set = NULL;
				if((Set = e->GetText()))
				{
					stringstream Stream(Set);

					//Hier wird der Variable das ensprechende Ergebnis zugewiesen.
					vector<uint_fast16_t>* variableValues = new vector<uint_fast16_t>;
					while (getline(Stream, Segment, ','))
					{
						if(utils::is_digit(Segment))
						{
							variableValues->push_back(utils::stoi(Segment));
						}
						//Variable in Variable
						else
						{
							vector<uint_fast16_t>* Var = NULL;
							if((Var = instance->getVariable(&Segment)))
							{
								variableValues->insert(variableValues->end(), Var->begin(), Var->end());
							}
							else
							{
								Console::Stream << "Rules::Parse Variable not Found" << endl;
								Console::writeAsError();
							}
						}
					}

					instance->variables->insert(pair<string, vector<uint_fast16_t>*>(Attribute, variableValues));

					Console::Stream << "Rules::Parse Added Variable: " << Attribute << " = " << Set << endl;
					Console::writeAsMessage();
				}
				else
				{
					Console::Stream << "XML(RULE): Rule empty Variable Set, ignoring: " << e->ToText() << endl;
					Console::writeAsError();
				}
			}
			else
			{
				Console::Stream << "XML(RULE): No Variable \"id\" given, ignoring: " << e->ToText() << endl;
				Console::writeAsError();
			}
		}
	}

	//Initialisieren des RulesCalculators
	Calcer = new RulesCalculator(instance->variables);

	//Parsen der Translationen
	for (const tinyxml2::XMLElement* e = node->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
	{
		if (strcmp("Rule", e->Value()) == 0)
		{
			Attribute = NULL;
			if((Attribute = e->Attribute("translation")))
			{
				stringstream Stream(Attribute);
				vector<string>* Translation = new vector<string>();

				//Erstellen von Rule
				const char* Value = NULL;
				if ((Value = e->GetText()))
				{
					string* Id = new string(e->Attribute("id"));
					Rule* MyRule = new Rule(Id);
					if (utils::is_digit(Value))
					{
						//Normal Rule Result
						MyRule->setSingleValue(utils::stoi(Value));
					}
					else
					{
						//Variable as Rule Result
						vector<uint_fast16_t>* Variable = NULL;
						if(instance->getVariable(new string(Value)))
						{
							MyRule->setValue(Variable);
						}
						else
						{
							Console::Stream << "XML(RULE): Variable not found: " << Value << endl;
							Console::writeAsError();
						}
					}

					//Einfuegen der Rule
					instance->allRules->push_back(MyRule);
					instance->rulesById->insert(pair<string, Rule*>(*Id, MyRule));

					//Auftrennen des Strings
					while (getline(Stream, Segment, ','))
					{
						Translation->push_back(Segment);
					}
					//Pruefen ob Ergebnis Variable im Translations Vector ist(z.B. sexy Loop)
					if(!utils::is_digit(Value))
					{
						//Ergbnis ist Variable
						for(unsigned int i = 0; i < Translation->size(); i++)
						{
							if(strcmp(Value, Translation->at(i).c_str()) == 0)
							{
								//Ergebnis kommt in Translation vor
								MyRule->setTranslationPosition(i);
								break;
							}
						}
					}
					if(Translation->size() != ExpectedTranslationSize)
					{
						Console::Stream << "XML(RULE): Rule \"translation\" unexpected size: " << Translation->size() << endl;
						Console::writeAsError();
					}

					Calcer->calculate(Translation);
					vector<uint_fast16_t>* candidates;
					while ((candidates = Calcer->getNext()))
					{
						MyRule->addValues(new vector<uint_fast16_t>(*candidates));
						instance->rulesByValues->insert(candidates, MyRule);
						if (instance->symmetries->substr(0,6).compare("rotate") == 0)
						{
							int rotationCount = utils::stoi(instance->symmetries->substr(6, 7));
							for(int i = 1; i < rotationCount; i++)
							{
								rotate(candidates->begin() + 1, candidates->begin() + 2, candidates->end());
								MyRule->addValues(new vector<uint_fast16_t>(*candidates));
								instance->rulesByValues->insert(candidates, MyRule);
							}
						}
						else if(instance->symmetries->compare("none") != 0)
						{
							Console::Stream << "XML(RULE): Unsupported \"symmetries\" given: " << *instance->symmetries << endl;
							Console::writeAsError();
						}
					}
				}
				else
				{
					Console::Stream << "XML(RULE): No Rule Result given: " << e->ToText() << endl;
					Console::writeAsError();
				}
			}
			else
			{
				Console::Stream << "XML(RULE): No Rule \"translation\" given, ignoring: " << e->ToText() << endl;
				Console::writeAsError();
			}
		}
	}
	delete Calcer;
}

Rules* Rules::getInstance()
{
	if(!instance)
	{
		Console::Stream << "Rules Instance Null" << endl;
		Console::writeAsException();
	}
	return instance;
}

uint_fast16_t Rules::getRuleValue(const vector<uint_fast16_t>* Values) const
{
	return rulesByValues->search(Values);
}

vector<uint_fast8_t>* Rules::getRuleOrder() const
{
	if(!ruleOrder)
	{
		Console::Stream << "Rules RuleOrder NULL" << endl;
		Console::writeAsException();
	}
	return ruleOrder;
}

Rule* Rules::getRuleById(const string* Id) const
{
	map<string, Rule*>::iterator iter;
	iter = rulesById->find(*Id);
	if (iter != rulesById->end())
	{
		return iter->second;
	}
	return NULL;
}

bool Rules::activateRuleById(const string* Id)
{
	map<string, Rule*>::iterator iter;
	iter = rulesById->find(*Id);
	if (iter != rulesById->end())
	{
		iter->second->setUsed(true);
		return true;
	}
	return false;
}

bool Rules::deactivateRuleById(const string* Id)
{
	map<string, Rule*>::iterator iter;
	iter = rulesById->find(*Id);
	if (iter != rulesById->end())
	{
		iter->second->setUsed(false);
		return true;
	}
	return false;
}

bool Rules::setRuleByValues(const vector<uint_fast16_t>* Values, const uint_fast16_t Value)
{
	return rulesByValues->setSingleValue(Values, Value);
}

bool Rules::setRuleByValues(const vector<uint_fast16_t>* Values, vector<uint_fast16_t>* Value)
{
	return rulesByValues->setValue(Values, Value);
}

bool Rules::setRuleById(const string* Id, const uint_fast16_t Value)
{
	map<string, Rule*>::iterator iter;
	iter = rulesById->find(*Id);
	if (iter != rulesById->end())
	{
		iter->second->setSingleValue(Value);
		return true;
	}
	return false;
}

bool Rules::setRuleById(const string* Id, vector<uint_fast16_t>* Value)
{
	map<string, Rule*>::iterator iter;
	iter = rulesById->find(*Id);
	if (iter != rulesById->end())
	{
		iter->second->setValue(Value);
		return true;
	}
	return false;
}

vector<Rule*>* Rules::getAllRules() const
{
	return allRules;
}