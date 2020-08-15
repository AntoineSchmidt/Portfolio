///////////////////////////////////////////////////////////
//  Alternative World Data Struktur
//  Created on:      14-Apr-2016 16:59:28
//  Author:         Bonifaz Stuhr, Antoine Schmidt und Tobias Eigler
///////////////////////////////////////////////////////////

#include <sstream>
#include <map>
#include <vector>
#include "model/World.h"
#include "model/WorldMatrix.h"
#include "controller/Controller.h"
#include "library/tinyxml2.h"
#include "controller/Console.h"
#include "library/utils.h"

World* World::instance = NULL;

/*
Ist der Konstruktor und parst die Daten aus dem XML-Dokument.
*/
void World::parse(const tinyxml2::XMLElement* node)
{
	Console::Stream << "Parsing World" << endl;
	Console::writeAsMessage();

	//Create Temp Values
	string Segment;
	const char* Attribute;
	uint_fast8_t ExpectedPositionSize;

	/*
	Da die Klasse ein Singleton ist, darf diese nur einmal vorkommen.
	-> Somit muss die Variable "instance" den Wert NULL beinhalten, damit die Klasse erstellt werden kann.
	*/

	instance = new World();

	//Get Border Flag from XML
	Attribute = NULL;
	if((Attribute = node->Attribute("border")))
	{
		if(strcmp("true", Attribute) == 0)
		{
			instance->bhasBorder = true;
		}
		else if(strcmp("false", Attribute) == 0)
		{
			instance->bhasBorder = false;
		}
		else
		{
			Console::Stream << "XML(World): Unrecognized \"border\" given, setting to Default (false)" << endl;
			Console::writeAsError();

			instance->bhasBorder = false;
		}
	}
	else
	{
		Console::Stream << "XML(World): No \"border\" given, setting to Default (false)" << endl;
		Console::writeAsError();

		instance->bhasBorder = false;
	}

	/*
	Zunaechst einmal wird nach der Groe�e der Welt geparst.
	Attributname der Weltgroe�e im XML Dokument ist "size".
	Attribut beinhaltet die maximalen Koordinaten der Welt.
	2-Dimensionales bsp: x=300 und y=400
	*/

	Attribute = NULL;
	if((Attribute = node->Attribute("size")))
	{
		stringstream Stream(Attribute);

		//Auftrennen des Strings am "Komma"
		while (getline(Stream, Segment, ','))
		{
			//Konvertieren des Teilstrings in einen Integer (stoi -> stringtointeger)
			//und anhaengen an den temporaeren Vektor "seglist"
			instance->size->push_back(utils::stoi(Segment));
		}
	}
	else
	{
		Console::Stream << "XML(World): No \"size\" given, setting to Default (false)" << endl;
		Console::writeAsError();
	}
	ExpectedPositionSize = instance->size->size();

	map<vector<uint_fast32_t>, uint_fast16_t> points;
	vector<uint_fast32_t> startPointCords;

	Attribute = NULL;
	if((Attribute = node->Attribute("startPoint")))
	{
		stringstream Stream(Attribute);

		//Auftrennen des Strings am "Komma"
		while (getline(Stream, Segment, ','))
		{
			startPointCords.push_back(utils::stoi(Segment));
		}

		if(startPointCords.size() != ExpectedPositionSize)
		{
			Console::Stream << "XML(World): World \"startPoint\" unexpected size " << startPointCords.size() << endl;
			Console::writeAsError();
		}
	}
	else
	{
		Console::Stream << "XML(World): No \"startPoint\" given, setting to 0" << endl;
		Console::writeAsError();
	}

	/*
	Hier werden die einzelnen Punkte, die aus dem XML-Dokument stammen, gelesen und in das Datenelement
	vector(points) eingefuegt.
	*/
	for (const tinyxml2::XMLElement* e = node->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
	{
		//Pruefen ob das XML-Element den Namen "Point" hat.
		if (strcmp(e->Value(), "Point") == 0)
		{
			Attribute = NULL;
			if((Attribute = e->Attribute("position")))
			{
				stringstream Stream(Attribute);
				vector<uint_fast32_t> seglist;

				//Auftrennen des Strings
				while (getline(Stream, Segment, ','))
				{
					//Zusammensetzten des neuen Strings ohne die Kommata
					seglist.push_back(utils::stoi(Segment));
				}

				if(seglist.size() != ExpectedPositionSize)
				{
					Console::Stream << "XML(World): Point \"position\" unexpected size " << seglist.size() << endl;
					Console::writeAsError();
				}

				//Manipulate Coordinates
				for(unsigned int i = 0; i < seglist.size() && i < startPointCords.size(); i++)
				{
					int ManipulationResult = (int)seglist.at(i) + (int)startPointCords.at(i);
					if(ManipulationResult < 0 || ManipulationResult >= (int)instance->size->at(i))
                    {
                        Console::Stream << "XML(World): Point Position after Manipulation out of World bound" << endl;
                        Console::writeAsError();
					}
					seglist.at(i) = ManipulationResult;
				}

				//Insert into Map
				Attribute = NULL;
				if((Attribute = e->GetText()))
				{
					points.insert(pair<vector<uint_fast32_t>, uint_fast16_t>(seglist, utils::stoi(Attribute)));
				}
				else
				{
					Console::Stream << "XML(World): Point empty Value, setting Value to Default: " << Point::emptyValue << endl;
					Console::writeAsError();

					points.insert(pair<vector<uint_fast32_t>, uint_fast16_t>(seglist, Point::emptyValue));
				}
			}
			else
			{
				Console::Stream << "XML(World): Point no \"position\" given, ignoring" << endl;
				Console::writeAsError();
			}
		}
		else
		{
			Console::Stream << "World Parse Not Registered" << e->Value() << endl;
			Console::writeAsError();
		}
	}

	/*
	pointNodeMatrix die die komplette World darstellt wird iniziiert
	Wird sofort maximiert um performance zu sparen (pushback() aufwendiger wie at())
	*/

	// auf 1 initialisieren wegen multiplikation
	uint_fast64_t pointNodesMatrixSize = 1;
	for (uint_fast16_t i = 0; i < instance->size->size(); i++)
	{
		/*
		bsp:
		mit einer 2D welt von 100x mal 200y:
		1. iteration: pointNodesMatrixSize = 1 * 100
		2. iteration: pointNodesMatrixSize = 100 * 200
		*/
		pointNodesMatrixSize *= instance->size->at(i);
	}
	//Initialisieren der WorldMatrix mit Berechneter Size
	instance->pointNodesMatrix = new vector<PointNode*>(pointNodesMatrixSize, NULL);
	vector<PointNode*>::iterator NodesMatrixIter;

	uint_fast16_t dimensionen = instance->size->size();
	vector<uint_fast32_t> cords(dimensionen, 0);
	map<vector<uint_fast32_t>, uint_fast16_t>::iterator iter;

	for (NodesMatrixIter = instance->pointNodesMatrix->begin(); NodesMatrixIter != instance->pointNodesMatrix->end(); NodesMatrixIter++)
	{
		//Berechne Naechste Coordinate
		for (uint_fast16_t dim = 0; cords.at(dim) >= instance->size->at(dim); dim++)
		{
			cords.at(dim) = 0;
			if (dim + 1 < dimensionen)
			{
				cords.at(dim + 1)++;
			}
			else
			{
				break;
			}
		}

		PointNode* pointNode = new PointNode(new vector<uint_fast32_t>(cords));

		//Vernetzung verschoben in WorldMatrix::ConnectQuick bzw. WorldMAtrix::Connect
		//Aufruf wenn alle PointNodes in der Matrix sind

		//Setzten des PointNode Values und Hinzufuegen zum Chunk
		//Setzen des InChunk Bool
		iter = points.find(cords);
		if (iter != points.end())
		{
			pointNode->setActiveValue(iter->second);
			pointNode->setInChunk(true);
			instance->chunk->push_back(pointNode);
			points.erase(iter);
		}

		//Hinzufuegen der PointNode in die Matrix
		*NodesMatrixIter = pointNode;

		//Increment fuer Naechsten PointNode Coordinate
		cords.at(0)++;
	}

	//Vernetzung
	//WorldMatrix::ConnectQuick(instance->pointNodesMatrix, instance->size, instance->hasBorder());
	WorldMatrix::connect(instance->pointNodesMatrix, instance->size, instance->hasBorder());

	/*
	////Initialisieren der pointNodesMatrix mit der berechneten Groe�e
	////evtl poiter fehler
	//instance->pointNodesMatrix = new vector<PointNode*>(pointNodesMatrixSize);

	//vector<PointNode*>* part;
	//part->push_back(new PointNode());

	//int dimensionen = instance->size->size();

	//Festlegen der Koordinaten.
	//Grundgedanke:
	//											  1234 Dimension
	//zunaechst wird die erste dimension aufgebaut  xxxx
	//	0xxx 1xxx 2xxx 3xxx 4xxx
	//Danach wird dieser "Part" kopiert und die Koordinaten der 2 Dimension eingefuegt bis Dimensionsbreite erreicht ist.
	//	0xxx 1xxx 2xxx 3xxx 4xxx
	//	01xx 11xx 21xx 31xx 41xx
	//	02xx 12xx 22xx 32xx 42xx
	//	03xx 13xx 23xx 33xx 43xx
	//Wenn die Dimensionsbreite erreicht ist, wird der erstellte block Kopiert(hier das obige 2D Feld).
	//Dann wird die Koordinate fuer die naechste Dimension gesetzt (hier 3. Dimension):

	//0x0x 1x0x 2x0x 3x0x 4x0x
	//010x 110x 210x 310x 410x
	//020x 120x 220x 320x 420x
	//030x 130x 230x 330x 430x

	//Fuer ersten 2D "Part" erstellt. "Part" kopieren und "naechste Ebene aufsetzten".
	//Bis Dimensionsbreite in der 3. Dimension erreicht ist.

	//0x1x 1x1x 2x1x 3x1x 4x1x
	//011x 111x 211x 311x 411x
	//021x 121x 221x 321x 421x
	//031x 131x 231x 331x 431x


	//Fuer die 4. Dimension wird dann der ganze 3D Block kopiert und wieder die Koordinate der 4. Dimension gesetzt.
	//...
	//
	//int partsize = 1;
	//for (int i = 0; i < instance->size->size(); i++)
	//{
	//	for (int j = 0; j < instance->size->at(i); j++)
	//	{
	//		//for (int x = 0; x < instance->size.at(i); x++)
	//		//{
	//		//	partsize *= instance->size.at(x);
	//		//}
	//		//part->at(j)->setNeighbour(3, part->at(j - 1));
	//		//part->at(j - 1)->setNeighbour(1, part->at(j));

	//		part->at(j)->setPosition(i, j);
	//		instance->pointNodesMatrix->insert(instance->pointNodesMatrix->end(), part->begin(), part->end());
	//	}
	//	part = instance->pointNodesMatrix;
	//}
	*/

	//Hinzufuegen der PointNodes Nachbarn
	vector<PointNode*> buffer;
	vector<PointNode*>::iterator ChunkIter;
	vector<PointNode*>::iterator NeighboursIter;
	//update den gesammten Chunk
	for (ChunkIter = instance->chunk->begin(); ChunkIter != instance->chunk->end(); ChunkIter++)
	{
		vector<PointNode*>* Neighbours;
		if((Neighbours = (*ChunkIter)->getNeighbours()))
		{
			//ueberpruefe ob die ihre Nachbaren bereits im Chunk sind, Nachbaren wegen "0" Rand
			for (NeighboursIter = Neighbours->begin(); NeighboursIter != Neighbours->end(); NeighboursIter++)
			{
				PointNode* PN;
				if ((PN = (*NeighboursIter)))
				{
					if(!PN->getInChunk())
					{
						buffer.push_back(PN);
						PN->setInChunk(true);
					}
				}
				else
				{
					//Border
				}
			}
		}
	}

	//Join Chunk and Buffer Vector
	//vector<PointNode*>::iterator ChunkIterator = C->end();
	instance->chunk->insert(instance->chunk->end(), buffer.begin(), buffer.end());

}

World* World::getInstance()
{
	if (!instance)
	{
		Console::Stream << "World Instance Null" << endl;
		Console::writeAsException();
	}
	return instance;
}

//Liefert einen Pointer auf das Chunk der World
vector<PointNode*>* World::getChunk() const
{
	return this->chunk;
}

//Liefert einen Pointer auf die PointNodesMatrix der World
vector<PointNode*>* World::getPointNodesMatrix() const
{
	return this->pointNodesMatrix;
}

//gibt Weltgroe�e zurueck
vector<uint_fast32_t>* World::getSize() const
{
	return this->size;
}

//gibt Weltgroe�e zurueck
uint_fast8_t World::getDimensions() const
{
	return this->size->size();
}

bool World::hasBorder() const
{
	return bhasBorder;
}

World::World()
{
	size = new vector<uint_fast32_t>();
	chunk = new vector<PointNode*>();
	pointNodesMatrix = new vector<PointNode*>();
}

World::~World()
{
	delete size;
	delete chunk;
	delete pointNodesMatrix;
}