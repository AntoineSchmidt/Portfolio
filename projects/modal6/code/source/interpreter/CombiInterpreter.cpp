///////////////////////////////////////////////////////////
//  Author: Antoine Schmidt
///////////////////////////////////////////////////////////

#include "interpreter/CombiInterpreter.h"
#include "model/Rules.h"
#include "controller/Console.h"

CombiInterpreter::CombiInterpreter()
{

}

CombiInterpreter::~CombiInterpreter()
{

}

void CombiInterpreter::update()
{
    IWorld* W;
    IRules* R;

    //Pruefung auf Valid Pointer
    if ((W = World::getInstance()) && (R = Rules::getInstance()))
    {
        vector<PointNode*>* C;
        if((C = W->getChunk()))
        {
            vector<PointNode*> buffer;
            vector<PointNode*>* Neighbours;
            vector<PointNode*>::iterator NeighboursIter;
            vector<PointNode*>::iterator ChunkIter;
            for (ChunkIter = C->begin(); ChunkIter != C->end(); ChunkIter++)
            {
                if ((*ChunkIter)->getChanged())
                {
                    (*ChunkIter)->setTempValue(R->getRuleValue((*ChunkIter)->getValueWithNeighbours()));

                    //falls sich ein Wert von 0 auf x geaendert hat
                    if ((*ChunkIter)->changingOnUpdate(0))
                    {
                        //ueberpruefe ob die ihre Nachbaren bereits im Chunk sind, Nachbaren wegen "0" Rand
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
            }
            //Join Chunk and Buffer Vector
            C->insert(C->end(), buffer.begin(), buffer.end());

            //Update Chunk
            for (ChunkIter = C->begin(); ChunkIter != C->end(); ChunkIter++)
            {
                (*ChunkIter)->update();
            }
        }
        else
        {
            Console::Stream << "World Chunk Null" << endl;
            Console::writeAsException();
        }
    }
}