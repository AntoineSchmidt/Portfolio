///////////////////////////////////////////////////////////
//  Original author: Bonifaz Stuhr
///////////////////////////////////////////////////////////

#include "debug/WorldCheck.h"
#include "controller/Console.h"
//#include <regex>


//GCC and Stoi
#include "library/utils.h"

using namespace std;

WorldCheck* WorldCheck::instance = NULL;

WorldCheck::WorldCheck()
{
	this->file = "WorldCheck_";
}

WorldCheck::~WorldCheck()
{
	fstream.close();
}

/*
Überprüft die gesammte Welt nach Vernetzungsfehler und Koordinatenfehler.
Schreibt Fehler in ein File, true wenn alles ok
Eingabe: Code für Filename Index
*/
bool WorldCheck::checkWorld(string fileIndex = "")
{
	World* world;
	bool checked = true;
	if ((world = World::getInstance()))
	{
		string fileName = instance->file + "checkWorld " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);
		fstream << "Invalid World Parameters: " << endl;
		uint_fast8_t dimension = world->getDimensions();
		vector<uint_fast32_t>* size = world->getSize();

		bool checkSize = true;
		for (uint_fast8_t i = 1; i < size->size(); i++)
		{
			if (size->at(i) <= 1)
			{
				checkSize = false;
				checked = false;
				break;
			}
		}
		if(!checkSize)
			fstream << "Invalid Size:  " << posToString(size);
		if (dimension <= 1)
		{
			fstream << "Invalid Dimensions: " << dimension << endl;
			checked = false;
		}
		fstream.close();
		checked = checkMatrix(fileIndex);
		checked = checkChunk(fileIndex);
	}
	else
	{
		Console::Stream << "WorldCheck::checkWorld: World Instance Null" << endl;
		Console::writeAsError();

		fstream << "WorldCheck::checkWorld: World Instance Null" << endl;
		fstream.close();
		checked = false;
	}
	return checked;
}

/*
Überprüft die gesammte MAtrix nach Vernetzungsfehler und Koordinatenfehler.
Schreibt Fehler in ein File, true wenn alles ok
Eingabe: Code für Filename Index
*/
bool WorldCheck::checkMatrix(string fileIndex = "")
{
	vector<PointNode*>* matrix;
	bool checked = true;
	string error;
	if ((matrix = World::getInstance()->getPointNodesMatrix()))
	{
		string fileName = instance->file + "checkMatrix " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);

		if (matrix->size() <= 0)
		{
			fstream << "Invalid MatrixSize: " << matrix->size() << endl;
			checked = false;
		}
		fstream << "Invalid MatrixPointNodes: " << endl;
		vector<PointNode*>::iterator iter;
		bool ok = true;
		for (iter = matrix->begin(); iter != matrix->end(); iter++)
		{
			ok = checkPointNode(*iter, error);
			if (!ok)
			{
				fstream << "Invalid PointNode : " << error << endl;
				printNodeToFile(*iter);
				checked = false;
			}
		}
	}
	else
	{
		fstream << "WorldCheck::checkMatrix: Matrix Instance Null" << endl;
		Console::Stream << "WorldCheck::checkMatrix: Matrix Instance Null" << endl;
		Console::writeAsError();
		checked = false;
	}
	fstream.close();
	return checked;
}

/*
Überprüft den gesammten Chunk nach Vernetzungsfehler und Koordinatenfehler
Schreibt Fehler in ein File, true wenn alles ok
Eingabe: Code für Filename Index
*/
bool WorldCheck::checkChunk(string fileIndex = "")
{
	vector<PointNode*>* chunk;
	bool checked = true;
	string error;
	if ((chunk = World::getInstance()->getChunk()))
	{
		string fileName = instance->file + "checkChunk " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);

		if (chunk->size() <= 0)
		{
			fstream << "Invalid ChunkSize: " << chunk->size() << endl;
			checked = false;
		}
		fstream << "Invalid ChunkPointNodes: " << endl;
		vector<PointNode*>::iterator iter;
		bool ok = true;
		for (iter = chunk->begin(); iter != chunk->end(); iter++)
		{
			ok = checkPointNode((*iter), error);
			if (!ok)
			{
				fstream << "Invalid PointNode : " << error << endl;;
				printNodeToFile((*iter));
				checked = false;
			}
		}
	}
	else
	{
		fstream << "WorldCheck::checkChunk: Chunk Instance Null" << endl;
		Console::Stream << "WorldCheck::checkChunk: Chunk Instance Null" << endl;
        Console::writeAsError();
		checked = false;
	}
	fstream.close();
	return checked;
}

/*
Überprüft eine PointNode ob sie im Kontext der Welt richtig liegt
Gibt true zurück wenn sie richtig liegt
*/
bool WorldCheck::checkPointNode(PointNode* node, string& error)
{
	World* W;
	if ((W = World::getInstance()))
	{
		//Prüfe ob Node Null ist
		if (!node)
		{
			error = "PointNode NUll";
			return false;
		}
		//Prüfe ob Id einen erlaubten Wert besitzt
		if (node->getId() < 0)
		{
			error = "Id kleiner 0";
			return false;
		}
		//Überprüfe ob die Nachbarschaft den dimensionen Entspricht
		unsigned int neigboursCout = 0;
		int dim = W->getDimensions();
		while (dim > 0)
		{
			//Hier könnte in Zukunft, wenn das Programm mehrere Nachbarscahften ünterstützt, je nach Nachbarschaft addiert werden
			neigboursCout += 2;
			dim--;
		}
		if (node->getNeighbours()->size() != neigboursCout)
		{
			error = "Nachbarschaft entsprich nicht der Dimension";
			return false;
		}
		//Überprüfe ob keiner der Nachbarenn NULL ist
		vector<PointNode*>::iterator iter;
		for (iter = node->getNeighbours()->begin(); iter != node->getNeighbours()->end(); iter++)
		{
			if (!(*iter)) {
				error = "Neighbour NUll";
				return false;
			}
		}
		//Überprüfe on die Position  in der Dimension gültig ist
		if (node->getPosition()->size() != W->getDimensions())
		{
			error = "Position Size invalid";
			return false;
		}
		//Überprüfe ob Positionswerte gültig sind
		for (unsigned int i = 0; i < node->getPosition()->size(); i++)
		{
			if (node->getPosition()->at(i) < 0)
			{
				error = "Position kleiner 0";
				return false;
			}
		}
		//Überprüfe ob die Kernfunktion richtige Werte liefert
		if (node->getValueWithNeighbours()->size() != neigboursCout + 1)
		{
			error = "PointValueWithNeighbours liefert falschen Wert";
			return false;
		}
		return true;
	}
	error = "Word Instance NULL";
	return false;
}

/*
Vergleicht die neue Welt mit der Welt des vorherigen Steps und gibt Unterschiede aus
Gibt die Anzahl der Unterschiede zurück und schreibt diese in ein File
Eingabe: Code für Filename Index
*/
uint_fast64_t WorldCheck::compareWorldWithOld(string fileIndex = "")
{
	World* world;
	uint_fast64_t unsimilar = 0;

	if ((world = World::getInstance()))
	{
		string fileName = instance->file + "compareWorldWithOld " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);


		uint_fast8_t newDimension = world->getDimensions();
		vector<uint_fast32_t>* newSize = world->getSize();

		if (this->oldSize != *newSize)
		{
			fstream << "Size:  " << posToString(&oldSize) << " to " << posToString(newSize);
			unsimilar++;
		}
		if (this->oldDimension != newDimension)
		{
			fstream << " Dimensions: " << oldDimension << " to " << newDimension << endl;
			unsimilar++;
		}

		fstream << " Unterschiede in World: " << unsimilar << endl;

		fstream.close();
		compareMatrixWithOld(fileIndex);
		compareChunkWithOld(fileIndex);


		this->oldDimension = newDimension;
		this->oldSize = *newSize;
	}
	else
	{
		Console::Stream << "WorldCheck::compareWorldWithOld: World Instance Null" << endl;
        Console::writeAsError();
		fstream << "WorldCheck::compareWorldWithOld: World Instance Null" << endl;
		fstream.close();
	}

	return unsimilar;
}

/*
Vergleicht die neue MAtrix mit der MAtrix des vorherigen Steps und gibt Unterschiede aus
Gibt die Anzahl der Unterschiede zurück und schreibt diese in ein File
Eingabe: Code für Filename Index
*/
uint_fast64_t  WorldCheck::compareMatrixWithOld(string fileIndex = "")
{
	vector<PointNode*>* matrix;
	uint_fast64_t unsimilar = 0;
	if ((matrix = World::getInstance()->getPointNodesMatrix()))
	{
		string fileName = instance->file + "compareMatrixWithOld " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);

		if (oldMatrix.size() != matrix->size())
		{
			fstream << "MatrixSize: " << oldMatrix.size() << " to " << matrix->size() << endl;
			unsimilar++;
		}
		fstream << "MatrixPointNodes: " << endl;
		vector<PointNode*>::iterator iter = matrix->begin();
		vector<PointNode>::iterator iter2 = oldMatrix.begin();
		for ( ; iter != (matrix->end()) && (iter2 != oldMatrix.end()); iter++, iter2++)
		{
			if (*(*iter) != (*iter2))
			{
				printNodeToFile(*iter);
				fstream << " to ";
				printNodeToFile(&(*iter2));
				unsimilar++;
			}
		}
		fstream << " Unterschiede in Matrix: " << unsimilar << endl;
	}
	else
	{
		fstream << "WorldCheck::compareMatrixWithOld: Matrix Instance Null" << endl;
		Console::Stream << "WorldCheck::compareMatrixWithOld: Matrix Instance Null" << endl;
        Console::writeAsError();
	}
	fstream.close();
	oldMatrix.clear();
	for (unsigned int i = 0; i < matrix->size(); i++)
	{
		oldMatrix.push_back(*matrix->at(i));
	}
	//oldMatrix = m;
	return unsimilar;
}

/*
Vergleicht den neuen Chunck mit dem alten Chunk des vorherigen Steps und gibt Unterschiede aus
Gibt die Anzahl der Unterschiede zurück und schreibt diese in ein File
Eingabe: Code für Filename Index
*/
uint_fast64_t WorldCheck::compareChunkWithOld(string fileIndex = "")
{
	vector<PointNode*>* chunk;
	uint_fast64_t unsimilar = 0;
	if ((chunk = World::getInstance()->getChunk()))
	{
		string fileName = instance->file +  "compareChunkWithOld " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);

		if (oldChunk.size() != chunk->size())
		{
			fstream << "ChunkSize: " << oldChunk.size() << " to " << chunk->size() << endl;
			unsimilar++;
		}
		fstream << "ChunkPointNodes: " << endl;
		vector<PointNode*>::iterator iter = chunk->begin();
		vector<PointNode>::iterator iter2 = oldChunk.begin();
		for ( ; (iter != chunk->end()) && (iter2 != oldChunk.end()); iter++, iter2++)
		{
			if (*(*iter) != (*iter2))
			{
				printNodeToFile(*iter);
				fstream << " to ";
				printNodeToFile(&(*iter2));
				unsimilar++;
			}
		}
		fstream << " Unterschiede in Chunk: " << unsimilar << endl;
	}
	else
	{
		fstream << "WorldCheck::compareChunkWithOld: Chunk Instance Null" << endl;
		Console::Stream << "WorldCheck::compareChunkWithOld: Chunk Instance Null" << endl;
        Console::writeAsError();
	}
	oldChunk.clear();
	for (unsigned int i = 0; i < chunk->size(); i++)
	{
		oldChunk.push_back(*chunk->at(i));
	}
	fstream.close();
	return unsimilar;
}

/*
Schreibt alle Informationen über die Welt in ein File
Eingabe: Code für Filename Index
*/
void WorldCheck::printWorld(string fileIndex = "")
{
	World* world;

	if ((world = World::getInstance()))
	{
		string fileName = instance->file + "printWorld " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);
		fstream << "Size:  " << posToString(world->getSize())

			<< " Dimensions: " << world->getDimensions()
			<< " Matrix: " << (world->getPointNodesMatrix() ? "Existing " : "NULL")
			<< " Chunk: " << (world->getChunk() ? "Existing" : "NULL") << endl;

		fstream.close();
		printMatrix(fileIndex);
		printChunk(fileIndex);
	}
	else
	{
		Console::Stream << "WorldCheck::printWorld: World Instance Null" << endl;
        Console::writeAsError();

		fstream << "WorldCheck::printWorld: World Instance Null" << endl;
		fstream.close();
	}
}

/*
Schreibt alle Informationen über die Matrix in ein File
Eingabe: Code für Filename Index
*/
void WorldCheck::printMatrix(string fileIndex = "")
{
	vector<PointNode*>* matrix;
	if ((matrix = World::getInstance()->getPointNodesMatrix()))
	{
		string fileName = instance->file + "printMatrix " + fileIndex + ".txt";
		// Datei erstellen
		this->fstream.open(fileName, ios::out);

		fstream << "MatrixSize: "  << matrix->size() << endl;
		fstream << "MatrixPointNodes: "<< endl;
		vector<PointNode*>::iterator iter;
		for (iter = matrix->begin(); iter != matrix->end(); iter++)
		{
			printNodeToFile(*iter);
		}

	}
	else
	{
		fstream << "WorldCheck::printMatrix: Matrix Instance Null" << endl;
		Console::Stream << "WorldCheck::printMatrix: Matrix Instance Null" << endl;
        Console::writeAsError();
	}
	fstream.close();
}

/*
Schreibt alle Informationen über das Chunck in ein File
Eingabe: Code für Filename Index
*/
void WorldCheck::printChunk(string fileIndex = "")
{
	vector<PointNode*>* chunk;
	if ((chunk = World::getInstance()->getChunk()))
	{
		string fileName = "printChunk " + fileIndex + ".txt";

		// Datei erstellen
		this->fstream.open(fileName, ios::out);

		fstream << "chunkSize: "  <<  chunk->size() << endl;
		fstream << "chunkPointNodes: " << endl;
		vector<PointNode*>::iterator iter;
		for (iter = chunk->begin(); iter != chunk->end(); iter++)
		{
			printNodeToFile((*iter));
		}
	}
	else
	{
		fstream << "WorldCheck::printChunk: Chunk Instance Null" << endl;
		Console::Stream << "WorldCheck::printChunk: Chunk Instance Null" << endl;
        Console::writeAsError();
	}
	fstream.close();
}

/*
Schreibt alle Informationen über die Node
*/
void WorldCheck::printNodeToFile(PointNode* n)
{
	if (n)
	{
		fstream << "Id: " << (n->getId() ? utils::to_string(n->getId()) : "NULL") << endl
			<< " Cord: " << (n->getPosition() ? posToString(n->getPosition()) : "NULL")
			<< " TempValue: " << (n->getTempValue() ? utils::to_string(n->getTempValue()) : "NULL")
			<< " ActiveValue: " << (n->getActiveValue() ? utils::to_string(n->getActiveValue()) : "NULL")
			<< " Neighbours: " << (n->getNeighbours()->size() ? utils::to_string(n->getNeighbours()->size()) : "NULL") << endl;
		if (n->getNeighbours())
		{
			fstream << "Neighbours: " << endl;
			vector<PointNode*>::iterator nIter;
			uint_fast8_t pos = 0;
			for (nIter = n->getNeighbours()->begin(); nIter != n->getNeighbours()->end(); nIter++)
			{
				fstream << "Pos: " << utils::to_string(pos) << " Id: "
					<< ((*nIter)->getId() ? utils::to_string((*nIter)->getId()) : "NULL") << endl
					<< " Cord: " << ((*nIter)->getPosition() ? posToString((*nIter)->getPosition()) : "NULL")
					<< " TempValue: " << ((*nIter)->getTempValue() ? utils::to_string((*nIter)->getTempValue()) : "NULL")
					<< " ActiveValue: " << ((*nIter)->getActiveValue() ? utils::to_string((*nIter)->getActiveValue()) : "NULL")
					<< " Neighbours: " << ((*nIter)->getNeighbours()->size() ? utils::to_string((*nIter)->getNeighbours()->size()) : "NULL") << endl;
				pos++;
			}
		}
		fstream << "-------------------------------------------------------------------------------------------";
	}
	else
	{
		fstream << " PointNode is Null" << endl;
	}
	fstream << endl;
}

string WorldCheck::posToString(vector<uint_fast32_t>* v)
{
	string s = "(" + utils::to_string(v->at(0));
	for (uint_fast8_t i = 1; i < v->size(); i++)
	{
		s += ("," + (utils::to_string(v->at(i))));
	}
	s += ")";
	return s;
}

WorldCheck* WorldCheck::getInstance()
{
	if (!instance)
	{
		instance = new WorldCheck();

		World* world;
		if ((world = World::getInstance()))
		{
			//Wandle Pointer um damit alter Werte gespeichert werden können
			vector<PointNode*>* matrix = world->getPointNodesMatrix();
			for (unsigned int i = 0; i < matrix->size(); i++)
			{
				instance->oldMatrix.push_back(*matrix->at(i));
			}

			vector<PointNode*>* chunk = world->getChunk();
			for (unsigned int i = 0; i < chunk->size(); i++)
			{
				instance->oldChunk.push_back(*chunk->at(i));
			}

			instance->oldDimension = world->getDimensions();
			instance->oldSize = *world->getSize();

			//// Aktuelle Zeit ermitteln
			//time_t timeStamp;
			//timeStamp = time(0);
			//tm* now = localtime(&timeStamp);

			//// Neuen Dateinamen erstellen
			//string timeSet = utils::to_string(now->tm_mday) + "_" +
			//	utils::to_string(now->tm_mon + 1) + "_" +
			//	utils::to_string(now->tm_year + 1900) + "  " +
			//	utils::to_string(now->tm_hour) + "-" +
			//	utils::to_string(now->tm_min) + "-" +
			//	utils::to_string(now->tm_sec);

			//instance->file = instance->file + "_" + timeSet + "_";

			//// Datum und Uhrzeit mit gewohnten Zeichen trennen (Als Dateiname sind diese nicht zulaessig)
			//// "_" wird zu "." und "-" wird zu ":"
			//string reg[2][2] = { { "_", "." },{ "-", ":" } };
			//for (int i = 0, j = 0; i < 2; i++)
			//{
			//	regex expr(reg[i][j]);
			//	timeSet = regex_replace(timeSet, expr, reg[i][j + 1]);
			//}
		}
		else
		{
			Console::Stream << "WorldCheck: World Instance Null" << endl;
            Console::writeAsError();
		}
	}
	return instance;
}