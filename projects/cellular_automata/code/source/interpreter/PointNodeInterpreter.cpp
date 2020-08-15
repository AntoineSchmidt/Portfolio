///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:26
//  Original author: Bonifaz Stuhr
///////////////////////////////////////////////////////////

#include "interpreter/PointNodeInterpreter.h"
#include "model/Rules.h"
#include "controller/Console.h"

PointNodeInterpreter::PointNodeInterpreter()
{

}

PointNodeInterpreter::~PointNodeInterpreter()
{

}

void PointNodeInterpreter::update()
{
	IWorld* W;
	IRules* R;

	//Pruefung auf Valid Pointer
	if ((W = World::getInstance()) && (R = Rules::getInstance()))
	{
		vector<PointNode*>* C;
		if((C = W->getChunk()))
		{
			vector<PointNode*>::iterator ChunkIter;
			for (ChunkIter = C->begin(); ChunkIter != C->end(); ChunkIter++)
			{
				/*
				1. getValueAndNeighbours: Hole dePunkt-Wert mit seinen Nachbar-Werten (string) z.B. 2D: "00000"
				2. getRuleValue: Der obige Punkt+Nachbar-Wert string dient als Key und wir erhalten somit den neuen Wert des Punktes z.b. 00000->0
				3. setValue: Der neue Wert wird in seinem urspruenglichen Punkt als Temp Wert gespeichet.
				Es wird erst ein Temp wert gesetzt damit ein Zeitschritt korrekt ausgefuehrt wird. Man sieht sich die Welt an veraendert dann alle Punkte
				Ohen den Tempwert wuerde man sich die Welt nach jeder Iteration erneut ansehen
				*/
                (*ChunkIter)->setTempValue(R->getRuleValue((*ChunkIter)->getValueWithNeighbours()));
			}

			/*
			Iteriere ueber den gesammten Chunk, update die Punkte und erweitere ihn falls noetig
			Hier koennte man evtl noch performance sparen
			*/

			vector<PointNode*> buffer;
			vector<PointNode*>::iterator NeighboursIter;
			//update den gesammten Chunk
			for (ChunkIter = C->begin(); ChunkIter != C->end(); ChunkIter++)
			{
				//falls sich ein Wert von 0 auf x geaendert hat
				if ((*ChunkIter)->update())
				{
					//ueberpruefe ob die ihre Nachbaren bereits im Chunk sind, Nachbaren wegen "0" Rand
					vector<PointNode*>* Neighbours;
					if((Neighbours = (*ChunkIter)->getNeighbours()))
					{

						for(NeighboursIter = Neighbours->begin(); NeighboursIter != Neighbours->end(); NeighboursIter++)
						{
							PointNode* PN;
							if((PN = (*NeighboursIter)))
							{
								if(!PN->getInChunk())
								{
									buffer.push_back(PN);
									PN->setInChunk(true);
								}
							}
							else
							{
								Console::Stream << " Rand erreicht" << endl;
								Console::writeAsMessage();
							}
						}
					}
				}
			}
			//Join Chunk and Buffer Vector
			//vector<PointNode*>::iterator ChunkIterator = C->end();
			C->insert(C->end(), buffer.begin(), buffer.end());
		}
	}
	else
	{
		Console::Stream << "World Chunk Null" << endl;
		Console::writeAsException();
	}
}