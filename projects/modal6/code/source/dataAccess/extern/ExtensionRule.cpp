///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include "dataAccess/extern/ExtensionRule.h"
#include "controller/Console.h"
#include "library/utils.h"

ExtensionRule::ExtensionRule(const char* filen) : File(filen)
{
    checkExtension(".rule");
}

ExtensionRule::~ExtensionRule()
{

}

bool ExtensionRule::buildRulesNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc)
{
	Console::Stream << "Parsing File: " << filename << endl;
	Console::writeAsMessage();

	int translationID = 0;

	root->SetAttribute("id", filename);

	if(!openFile())
	{
		return false;
	}

	try
	{
		bool segmentfound = false;
		string line;
		while(nextLine(line))
		{
			if(strcmp(line.c_str(), "@TABLE") == 0)
			{
				segmentfound = true;
				break;
			}
		}
		if(!segmentfound)
		{
			return false;
		}

		while(nextLine(line))
		{
			//cout << line << endl;
			if(line[0] == '@')
			{
				break;
			}
			if(line[0] != '#' && line != "")
			{
				//Line with symmetries
				if(line.find("symmetries") == 0)
				{
					root->SetAttribute("symmetries", line.substr(11, line.length() - 11).c_str());
				}
				else if(line.find("n_states") == 0)
				{
					root->SetAttribute("states", line.substr(9, line.length() - 9).c_str());
				}
				else if(line.find("neighborhood") == 0)
				{
					root->SetAttribute("neighborhood", line.substr(13, line.length() - 13).c_str());
				}
				//Line with variable
				else if(line.find("var") == 0)
				{
					int Eq = line.find("=");
					tinyxml2::XMLElement* Var = doc->NewElement("Var");
					Var->SetAttribute("id", line.substr(4, Eq - 4).c_str());
					Var->SetText(line.substr(Eq + 2, line.length() - Eq - 3).c_str());

					//Anhaengen
					root->LinkEndChild(Var);
				}
				//Line with translation
				else
				{
					tinyxml2::XMLElement* Rul = doc->NewElement("Rule");
					Rul->SetAttribute("id", utils::to_string(translationID++).c_str());

					//Translation ohne Kommata trennung
					if(line.find(",") == string::npos)
					{
						string translation = "";
						for(unsigned int i = 0; i < (line.length() - 1); i++)
						{
							translation += line[i];
							if((i + 1) < (line.length() - 1))
							{
								translation += ",";
							}
						}
						Rul->SetAttribute("translation", translation.c_str());
						Rul->SetText(line.substr(line.length() - 1, 1).c_str());
					}
					//Translation mit Kommata trennung
					else
					{
						//Find Position of Last ,
						int KomPos = -1;
						for(int i = (line.length() - 1); i >= 0; i--)
						{
							if(line[i] == ',')
							{
								KomPos = i;
								break;
							}
						}
						if(KomPos != -1)
						{
							Rul->SetAttribute("translation", line.substr(0, KomPos).c_str());
							Rul->SetText(line.substr(KomPos + 1, line.length() - KomPos - 1).c_str());
						}
						else
						{
							Console::Stream << "Comma not Found" << endl;
							Console::writeAsError();
						}
					}

					//Anhaengen
					root->LinkEndChild(Rul);
				}
			}
		}
		return true;
	}
	catch(...)
	{
		Console::Stream << "buildRulesNode Exception" << endl;
		Console::writeAsException();

		return false;
	}

	closeFile();
}