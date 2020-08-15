///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#ifndef CONSOLE_H
#define CONSOLE_H

#pragma once

#include <string>
#include <sstream>

using namespace std;

/**
* Fuer die, per bool kontrollierten consolen ausgaben
*/
class Console
{
private:
	/**
	* Konstruktor
	* Static Class
	*/
	Console();

	/**
	* Der Ausgabe Cache
	*/
	static string cache;

	/**
	* Liest vom Stream in den Cache
	*/
	inline static void prepareStream();

	/**
	* Leert den Stream
	*/
	inline static void finishStream();

public:
	/**
	* Der Stream in den die Ausgabe geschrieben wird
	*/
	static stringstream Stream;

	/**
	* Schreibt Stream als Input auf Console
	*/
	static void writeAsInformation();

	/**
	* Schreibt Messages auf Console wenn true
	*/
	static bool writeMessages;

	/**
	* Schreibt Stream als Message auf Console
	*/
	static void writeAsMessage();

	/**
	* Schreibt Errors auf Console wenn true
	*/
	static bool writeErrors;

	/**
	* Schreibt Stream als Error auf Console
	*/
	static void writeAsError();

	/**
	* Schreibt Exceptions auf Console wenn true
	*/
	static bool writeExceptions;

	/**
	* Schreibt Stream als Exceptions auf Console
	*/
	static void writeAsException();

	/**
	* Wenn true wird nichts geschrieben ausser WriteInput
	*/
	static bool mute;

};

#endif // CONSOLE_H