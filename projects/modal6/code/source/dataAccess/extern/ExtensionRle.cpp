///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <vector>
#include "dataAccess/extern/ExtensionRle.h"
#include "controller/Console.h"
#include "library/utils.h"

ExtensionRle::ExtensionRle(const char* filen) : File(filen)
{
    checkExtension(".rle");
}

ExtensionRle::~ExtensionRle()
{

}

bool ExtensionRle::buildWorldNode(tinyxml2::XMLElement* root, tinyxml2::XMLDocument* doc)
{
    Console::Stream << "Parsing File: " << filename << endl;
    Console::writeAsMessage();

    root->SetAttribute("id", filename);

    if(!openFile())
    {
        return false;
    }

    try
    {
        string line;
        unsigned int linelength;

        //World Size
        vector<int> Size = vector<int>(2, 0);
        while(nextLine(line))
        {
            if(line[0] == 'x')
            {
                linelength = line.length();

                //Parse X
                unsigned int NBegin = 0, NEnd = 0;
                for(; NBegin < linelength; NBegin++)
                {
                    if(utils::is_digit(line[NBegin]))
                    {
                        break;
                    }
                }
                for(NEnd = NBegin; NEnd < linelength; NEnd++)
                {
                    if(!utils::is_digit(line[NEnd]))
                    {
                        break;
                    }
                }
                Size.at(0) = utils::stoi(line.substr(NBegin, NEnd - NBegin));

                //Parse Y
                for(; NEnd < linelength; NEnd++)
                {
                    if(line[NEnd] == 'y')
                    {
                        break;
                    }
                }
                for(NBegin = NEnd; NBegin < linelength; NBegin++)
                {
                    if(utils::is_digit(line[NBegin]))
                    {
                        break;
                    }
                }
                for(NEnd = NBegin; NEnd < linelength; NEnd++)
                {
                    if(!utils::is_digit(line[NEnd]))
                    {
                        break;
                    }
                }
                Size.at(1) = utils::stoi(line.substr(NBegin, NEnd - NBegin));

                //Setzen des Attributes
                string sSiz = utils::to_string(Size.at(0)) + "," + utils::to_string(Size.at(1));
                root->SetAttribute("size", sSiz.c_str());
                root->SetAttribute("border", "true");
                break;
            }
        }
        //World Points
        bool PositionUpdated = false;
        unsigned int Repetition = 1;
        unsigned int JumpI = 0;
        vector<int> Position = vector<int>(2, 0);
        while(nextLine(line))
        {
            if(line[0] != '#' && line != "")
            {
                linelength = line.length();
                for(unsigned int i = 0; i < linelength; i++)
                {
                    JumpI = 0;
                    for(; Repetition > 0; Repetition--)
                    {
                        PositionUpdated = false;

                        //Optional end of Pattern
                        if(line[i] == '!')
                        {
                            return true;
                        }
                        //Optional end of Row
                        else if(line[i] == '$')
                        {
                            Position.at(0) = 0;
                            if(++Position.at(1) >= Size.at(1))
                            {
                                Console::Stream << "Bad Position" << endl;
                                Console::writeAsError();
                                return false;
                            }
                            PositionUpdated = true;
                        }
                        //Value 0
                        else if(line[i] == 'b' || line[i] == '.')
                        {
                            //Ignore
                        }
                        //Value 1
                        else if(line[i] == 'o' || line[i] == 'A')
                        {
                            string sPos = utils::to_string(Position.at(0)) + "," + utils::to_string(Position.at(1));
                            tinyxml2::XMLElement* Var = doc->NewElement("Point");
                            Var->SetAttribute("position", sPos.c_str());
                            Var->SetText("1");
                            //Anhaengen
                            root->LinkEndChild(Var);
                        }
                        //Other Point Value
                        else if(!utils::is_digit(line[i]))
                        {
                            unsigned int Val = 0;
                            bool Second = false;
                            bool BadState = false;
                            if(line[i] == 'p' )
                            {
                                Val = 24;
                                Second = true;
                            }
                            else if(line[i] == 'q' )
                            {
                                Val = 48;
                                Second = true;
                            }
                            else if(line[i] == 'y' )
                            {
                                Val = 240;
                                Second = true;
                            }
                            else if(line[i] > 'A' && line[i] <= 'X')
                            {
                                Val = line[i] - 'A' + 1;
                            }
                            else
                            {
                                BadState = true;
                                PositionUpdated = true;
                            }

                            if(Second)
                            {
                                if((i + 1) < linelength)
                                {
                                    Val += line[i + 1] - 'A' + 1;
                                    JumpI = 1;
                                }
                                else
                                {
                                    Console::Stream << "Parsing .rle Bad Line" << endl;
                                    Console::writeAsError();
                                    return false;
                                }
                            }

                            if(!BadState)
                            {
                                string sPos = utils::to_string(Position.at(0)) + "," + utils::to_string(Position.at(1));
                                tinyxml2::XMLElement* Var = doc->NewElement("Point");
                                Var->SetAttribute("position", sPos.c_str());
                                Var->SetText(utils::to_string(Val).c_str());
                                //Anhaengen
                                root->LinkEndChild(Var);
                            }
                        }
                        else
                        {
                            PositionUpdated = true;
                        }

                        //Position Update
                        if(!PositionUpdated)
                        {
                            if(++Position.at(0) >= Size.at(0))
                            {
                                Position.at(0) = 0;
                                if((i + 1 + JumpI) < linelength && line[i + 1 + JumpI] == '$')
                                {
                                    //Row Increment Coming
                                }
                                else if(++Position.at(1) >= Size.at(1))
                                {
                                    //All Positions Set
                                    return true;
                                }
                            }
                        }
                    }
                    //Number Following Times
                    if(utils::is_digit(line[i]))
                    {
                        unsigned int e = 0;
                        for(e = i + 1; e < linelength; e++)
                        {
                            if(!utils::is_digit(line[e]))
                            {
                                break;
                            }
                        }
                        Repetition = utils::stoi(line.substr(i, e - i));
                        JumpI = e - i - 1;
                    }
                    else
                    {
                        Repetition = 1;
                    }

                    i += JumpI;
                }
            }
        }
        return true;
    }
    catch(...)
    {
        Console::Stream << "buildWorldNode Exception" << endl;
        Console::writeAsException();

        return false;
    }
}