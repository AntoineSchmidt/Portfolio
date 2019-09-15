///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <iostream>
#include "controller/Console.h"

bool Console::writeMessages = true;
bool Console::writeErrors = true;
bool Console::writeExceptions = true;
bool Console::mute = false;

stringstream Console::Stream("");
string Console::cache = "";

Console::Console()
{

}

void Console::writeAsInformation()
{
	prepareStream();
	cout << "[I] " << cache;
	finishStream();
}

void Console::writeAsMessage()
{
	if(!mute && writeMessages)
	{
	    prepareStream();
		cout << "[M] " << cache;
	}
	finishStream();
}

void Console::writeAsError()
{
	if(!mute && writeErrors)
	{
	    prepareStream();
		cout << "[E] " << cache;
	}
	finishStream();
}

void Console::writeAsException()
{
	if(!mute && writeExceptions)
	{
	    prepareStream();
		cout << "[!] " << cache;
	}
	finishStream();
}

void Console::prepareStream()
{
    cache = Stream.str();
}

void Console::finishStream()
{
    Stream.str("");
}