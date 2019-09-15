///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <string>
#include <vector>

using namespace std;

/**
* Klasse fuer basis funktionen
*/
class utils
{
public:
    /**
    * ersatz fuer nicht vorhandenes stoi in gcc
    * @param string
    * @return int der konvertierte string
    */
    static int stoi(const string& s);

    /**
    * Stellt fest ob string aus zahlen besteht
    * fuer den gebrauch in alife nur pruefen des ersten chars
    * @param string
    * @return bool true wenn string aus zahlen besteht
    */
	static bool is_digit(const string& s);

    /**
    * Stellt fest ob char eine zahl ist
    * @param char
    * @return bool true wenn string aus zahlen besteht
    */
	static bool is_digit(const char& c);

	/**
	* Wandelt int in String um
	* @param int
	* @return string
	*/
    static string to_string(const int Value);

    /**
    * Wechselt alle Vorkommnise von from in str to to
    * @param string der zu Manipulierende String
    * @param string der zu ersetzende substring
    * @param string der einzusetzende neue substring
    * @return string der Manipuliert String
    */
    static string replaceAll(const string& str, const string& from, const string& to);

    /**
    * Erstellt RGB Vector aus string Hex Value
    * @param string Hex Color
    * @return vector<unsigned char>* Der RGB Vector
    */
    static vector<unsigned char>* to_rgb(const string& Value);

private:
    /**
    * Privater Konstruktor da rein statische klasse
    */
    utils();
};

#endif // UTILS_H