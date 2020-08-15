///////////////////////////////////////////////////////////
//  Author:          Antoine Schmidt, Michell Mueller, Bonifaz Stuhr, Michael Warschat, Tobias Eigler(Studium)
///////////////////////////////////////////////////////////

#include <map>
#include "model/WorldMatrix.h"
#include "controller/Console.h"

bool WorldMatrix::connect(const vector<PointNode*>* Content, const vector<uint_fast32_t>* Size, const bool Borders)
{
	try
	{
		vector<PointNode*>::const_iterator ContentIter;

		//Erstellen einer Map fuer das Schnelle finden der PointNodes nach Position
		map<vector<uint_fast32_t>, PointNode*> CoordinatesMap;
		map<vector<uint_fast32_t>, PointNode*>::iterator CoordinatesMapIter;
		for(ContentIter = Content->begin(); ContentIter != Content->end(); ContentIter++)
		{
			CoordinatesMap.insert(pair<vector<uint_fast32_t>, PointNode*>(*(*ContentIter)->getPosition(), *ContentIter));
		}

		//Gehe alle PointNodes in Content durch und Verbinde
		vector<uint_fast32_t>* CoordinatesCache;
		vector<uint_fast32_t> LocalCoordinatesCache;
		for(ContentIter = Content->begin(); ContentIter != Content->end(); ContentIter++)
		{
			if(*ContentIter)
			{
				if((CoordinatesCache = (*ContentIter)->getPosition()))
				{
					LocalCoordinatesCache = *CoordinatesCache;
					//Gehe jede Dimension der PointNode Position durch
					for(uint_fast8_t i = 0; i < LocalCoordinatesCache.size(); i++)
					{
						int_fast8_t CoordDiff;
						int_fast8_t NeighbourDiff;

						//Every Dimension adds 2 more Neighbours
						for(uint_fast8_t d = 0; d < 2; d++)
						{
							switch(d)
							{
							case 0:
								//Right Neighbour
								CoordDiff = 1;
								NeighbourDiff = 1;
								break;

							case 1:
								//Left Neighbour
								CoordDiff = -1;
								NeighbourDiff = 0;
								break;

							default:
								Console::Stream << "WorldMatrix::Connect Bad Loop" << endl;
								Console::writeAsException();
								return false;
							}

							//Set Neighbour Coordiantes
							int NeighbourCoord = (int)LocalCoordinatesCache.at(i) + CoordDiff;

							//Check if still in World
							if(NeighbourCoord >= (int)Size->at(i) || NeighbourCoord < 0)
							{
								//Normally not in World anymore
								if(!Borders)
								{
									uint_fast32_t Backup = LocalCoordinatesCache.at(i);
									//Right Border
									//Connect with Left Border
									if(NeighbourCoord >= (int)Size->at(i))
									{
										LocalCoordinatesCache.at(i) = NeighbourCoord - Size->at(i);

										CoordinatesMapIter = CoordinatesMap.find(LocalCoordinatesCache);
										if (CoordinatesMapIter != CoordinatesMap.end())
										{
											//Set as Neighbour
											(*ContentIter)->setNeighbour(2*i + NeighbourDiff, CoordinatesMapIter->second);
										}
										else
										{
											Console::Stream << "WorldMatrix::Connect Left PointNode not found " << LocalCoordinatesCache.at(i) << endl;
											Console::writeAsError();
										}
									}
									//Left Border
									//Connect with Right Border
									else if(NeighbourCoord < 0)
									{
										LocalCoordinatesCache.at(i) = NeighbourCoord + Size->at(i);

										CoordinatesMapIter = CoordinatesMap.find(LocalCoordinatesCache);
										if (CoordinatesMapIter != CoordinatesMap.end())
										{
											//Set as Neighbour
											(*ContentIter)->setNeighbour(2*i + NeighbourDiff, CoordinatesMapIter->second);
										}
										else
										{
											Console::Stream << "WorldMatrix::Connect Right PointNode not found " << LocalCoordinatesCache.at(i) << endl;
											Console::writeAsError();
										}
									}

									//Set Back
									LocalCoordinatesCache.at(i) = Backup;
								}
								else
								{
									Console::Stream << "WorldMatrix::Connect World Border Reached" << endl;
									Console::writeAsMessage();
								}
							}
							else
							{
							    //No Overflow Apply to Vector
							    LocalCoordinatesCache.at(i) += CoordDiff;

								//In World
								//Suche entsprechenden PointNode
								CoordinatesMapIter = CoordinatesMap.find(LocalCoordinatesCache);
								if (CoordinatesMapIter != CoordinatesMap.end())
								{
									//Set as Neighbour
									(*ContentIter)->setNeighbour(2*i + NeighbourDiff, CoordinatesMapIter->second);
								}
								else
								{
									Console::Stream << "WorldMatrix::Connect PointNode not found " << LocalCoordinatesCache.at(i) << endl;
									Console::writeAsError();
								}

								//Reset
								LocalCoordinatesCache.at(i) -= CoordDiff;
							}
						}
					}
				}
				else
				{
					Console::Stream << "WorldMatrix::Connect PointNode Position NULL" << endl;
					Console::writeAsError();
				}
			}
			else
			{
				Console::Stream << "WorldMatrix::Connect PointNode NULL" << endl;
				Console::writeAsError();
			}
		}
		return true;
	}
	catch(...)
	{
		Console::Stream << "WorldMatrix::Connect Exception" << endl;
		Console::writeAsException();

		return false;
	}
}

bool WorldMatrix::connectQuick(const vector<PointNode*>* Content, const vector<uint_fast32_t>* Size, const bool Borders)
{
	try
	{
		for(unsigned int i = 0; i < Content->size(); i++)
		{
			uint_fast64_t neighbourdist = 1;
			for (uint_fast8_t j = 0; j < Size->size(); j++)
			{
				//ADDED
				//Check for MapEnd
				if((i / neighbourdist) % Size->at(j) != 0)
				{
					//Check if still in Vector
					if (neighbourdist <= i)
					{
						uint_fast8_t npos = 2 * j;
						PointNode* neigbour = Content->at(i - neighbourdist);
						neigbour->setNeighbour(npos + 1, Content->at(i));
						Content->at(i)->setNeighbour(npos, neigbour);
					}
				}
				neighbourdist *= Size->at(j);
			}
		}
		return true;
	}
	catch(...)
	{
		Console::Stream << "WorldMatrix::ConnectQuick Exception" << endl;
		Console::writeAsException();

		return false;
	}
}