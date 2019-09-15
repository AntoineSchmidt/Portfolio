///////////////////////////////////////////////////////////
//  Created on:      02-June-2016
//  Original author: Bonifaz Stuhr und Tobias Eigler
///////////////////////////////////////////////////////////

#pragma once
#include <set>
#include <vector>
#include <fstream>
#include <string>
#include "IDarwinEngine.h"

using namespace std;

/**
* Mutiert verschiedene Feldtypen der Welt
**/
class NuklearDarwin : public IDarwinEngine
{
private:
	uint_fast64_t mutationChanceCounter;
	uint_fast64_t mutationChanceDenominator;
	uint_fast64_t startAtStep;
	uint_fast64_t stepsActive;
	uint_fast32_t mutateAfterXSteps;
	vector<uint_fast16_t> mutationValues;
	set<uint_fast16_t> mutationTargets;
	uint_fast32_t intervallCounter;
	uint_fast64_t stepCounter;
	string file;
	ofstream stream;
	NuklearDarwin();

public:
	~NuklearDarwin();
	
	/**
	* Fahre über gesammte Welt und mutiere sie zufällig, verwende die Methode parametrize um Parameter einzustellen, schreibt einzelne schritte in ein fileName
	* @return void
	**/
	void doStep();

	/**
	* Parametriesiere Darwin mit Werten zur Mutation, diese werden in doStep() verwendet
	* @param int mutationChanceCounter die Chance die jede Zelle der Welt hatze zu Mutieren sobald doStep() aufgerufen wird default = 1
	* @param int mutationChanceDenominator die Chance die jede Zelle der Welt hatze zu Mutieren sobald doStep() aufgerufen wird default = 1000000
	* @param vector<int> mutationValues Vector an verschiedenen Mutationswerten die die Zelle annehmen kann, der Zufall entscheidet Welcher wert genommen wird default =0
	* @param set<int> mutationTargets Werte die anfällig für Mutationen sind default = 0
	* @param int startAtStep Wert nach dem die Mutation startet default = 0
	* @param int stepsActive Anzahl der Steps an der die Engine aktiv ist default = 0
	* @param int mutateAfterXSteps Intervall in dem mutiert wird default = 0
	* @return void
	*/
	void parametrize(uint_fast64_t mutationChanceCounter, uint_fast64_t mutationChanceDenominator, vector<uint_fast16_t> mutationValues, set<uint_fast16_t> mutationTargets, uint_fast64_t startAtStep, uint_fast64_t stepsActive, uint_fast32_t mutateAfterXSteps);
	static IDarwinEngine* parse(const tinyxml2::XMLElement* node);
	
};