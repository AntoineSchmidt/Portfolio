///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include "library/utils.h"

//GCC hat kein stoi
int utils::stoi(const string& s)
{
	istringstream str(s);
	int i;
	str >> i;
	return i;
}

bool utils::is_digit(const string& s)
{
	if (s.empty())
	{
		return false;
	}
	else
	{
		return is_digit(s[0]);
	}
}

bool utils::is_digit(const char& c)
{
    if (isdigit(c) == 0)
	{
		return false;
	}
	else
	{
		return true;
	}
}

string utils::to_string(const int Value)
{
	stringstream strs;
	strs << Value;
	return strs.str();
}

//ersetzt alle vorkommenden teile von string von from zu to
string utils::replaceAll(const string& str, const string& from, const string& to)
{
	if(from.empty())
	{
		return str;
	}
	string Answer = str;
	size_t start_pos = 0;
	while((start_pos = Answer.find(from, start_pos)) != string::npos)
	{
		Answer.replace(start_pos, from.length(), to);

		//falls to teilweise from enthaelt
		start_pos += to.length();
	}
	return Answer;
}

vector<unsigned char>* utils::to_rgb(const string& Value)
{
	if(Value.length() == 6)
	{
		vector<unsigned char>* Values = new vector<unsigned char>();
		int r,g,b;
		istringstream(Value.substr(0,2)) >> hex >> r;
		istringstream(Value.substr(2,2)) >> hex >> g;
		istringstream(Value.substr(4,2)) >> hex >> b;

		Values->push_back((unsigned char)(r));
		Values->push_back((unsigned char)(g));
		Values->push_back((unsigned char)(b));
		return Values;
	}
	else
	{
		return NULL;
	}
}