///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:27
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <vector>
#include "model/Point.h"
#include "controller/Console.h"
#include "model/PointNode.h"

uint_fast64_t PointNode::idCounter = 0;

PointNode::PointNode() : id(idCounter++)
{
	position = NULL;
	myNeighbours = NULL;
	changed = false;
	neighbourChanged = false;
	tempChanged = false;
	inChunk = false;
	myPoint = NULL;
}

PointNode::PointNode(vector<uint_fast32_t>* Position) : id(idCounter++)
{
	this->position = Position;
	myNeighbours = NULL;
	changed = false;
	neighbourChanged = false;
	tempChanged = false;
	inChunk = false;
	myPoint = NULL;
}

PointNode::PointNode(vector<uint_fast32_t>* Position, vector<PointNode*>* MyNeighbours) : id(idCounter++)
{
	this->position = Position;
	this->myNeighbours = MyNeighbours;
	changed = false;
	neighbourChanged = false;
	tempChanged = false;
	inChunk = false;
	myPoint = NULL;
}

PointNode::PointNode(vector<uint_fast32_t>* Position, vector<PointNode*>* MyNeighbours, const uint_fast16_t Value) : id(idCounter++)
{
	this->position = Position;
	this->myNeighbours = MyNeighbours;
	changed = true;
	neighbourChanged = false;
	tempChanged = false;
	inChunk = false;
	myPoint = new Point(Value);
}

void PointNode::setActiveValue(const uint_fast16_t Value)
{
	if(!myPoint)
	{
		Console::Stream << "setActiveValue: Creating Point" << endl;
		Console::writeAsMessage();

		myPoint = new Point(Value);
	}
	else
	{
		Console::Stream << "PointNode setActiveValue on Already Existing Point" << endl;
		Console::writeAsError();

		this->myPoint->setAllValue(Value);
	}
	setChanged(true, true, true, true);
}

uint_fast16_t PointNode::getActiveValue() const
{
	if(myPoint)
	{
		return myPoint->getActiveValue();
	}
	else
	{
		return Point::emptyValue;
	}
}

void PointNode::setPosition(vector<uint_fast32_t>* Position)
{
	this->position = Position;
}

vector<uint_fast32_t>* PointNode::getPosition() const
{
	if(!position)
	{
		Console::Stream << "PointNode Position Null" << endl;
		Console::writeAsError();
	}
	return position;
}

vector<uint_fast16_t>* PointNode::getValueWithNeighbours()
{
	vector<uint_fast8_t>* Order;
	IRules* R;
	if((R = Rules::getInstance()))
	{
		Order = R->getRuleOrder();
	}
	else
	{
		Console::Stream << "PointNode::getValueWithNeighbours returning NULL" << endl;
		Console::writeAsException();
		return NULL;
	}

	if(valueWithNeighbours.size() + 1 != Order->size())
	{
		valueWithNeighbours.resize(Order->size() + 1);
	}

	//Center Value
	valueWithNeighbours.at(0) = getActiveValue();

	if(myNeighbours)
	{
		PointNode* Neighbour;
		for (unsigned int i = 0; i < Order->size(); i++)
		{
			if((Neighbour = getNeighbour(Order->at(i))))
			{
				valueWithNeighbours.at(i + 1) = Neighbour->getActiveValue();
			}
			else
			{
				valueWithNeighbours.at(i + 1) = Point::emptyValue;
			}
		}
	}
	else
	{
		Console::Stream << "Neighbours Vector Null" << endl;
		Console::writeAsException();
	}
	return &valueWithNeighbours;
}

void PointNode::setTempValue(const uint_fast16_t Value)
{
	if(Value != Point::emptyValue || myPoint)
	{
		if(!myPoint)
		{
			Console::Stream << "setTempValue: Creating Point" << endl;
			Console::writeAsMessage();

			myPoint = new Point();
		}
		myPoint->setTempValue(Value);
		if(myPoint->changingOnUpdate())
		{
			setChanged(true, true);
		}
		else
		{
			setChanged(false);
		}
	}
	else
	{
		setChanged(false);
	}
}

uint_fast16_t PointNode::getTempValue() const
{
	if(myPoint)
	{
		return myPoint->getTempValue();
	}
	else
	{
		return Point::emptyValue;
	}
}

void PointNode::setChanged(bool Value, bool NeighboursToo, bool Instant, bool InstantNeighbour, uint_fast8_t Deep)
{
	if(Deep == 0)
	{
		if(Instant)
		{
			changed = Value;
			tempChanged = false;
		}
		else
		{
			changed = false;
			tempChanged = Value;
		}
	}
	else
	{
		if(Instant)
		{
			changed = Value;
			tempChanged = false;
		}
		else if(!neighbourChanged)
		{
			neighbourChanged = Value;
		}
	}
	if(NeighboursToo && myNeighbours)
	{
		vector <PointNode*>::iterator iter;
		for (iter = myNeighbours->begin(); iter != myNeighbours->end(); ++iter)
		{
			if(*iter)
			{
				(*iter)->setChanged(Value, false, InstantNeighbour, false, ++Deep);
			}
		}
	}
}

void PointNode::updateChanged()
{
	changed = tempChanged | neighbourChanged;
	tempChanged = false;
	neighbourChanged = false;
}

bool PointNode::getChanged() const
{
	return changed;
}

void PointNode::setInChunk(bool Value)
{
	inChunk = Value;
}

bool PointNode::getInChunk() const
{
	return inChunk;
}

bool PointNode::update()
{
	bool ReturnValue = false;
	if(tempChanged)
	{
		if(myPoint)
		{
			ReturnValue = myPoint->update();
		}
		else
		{
			Console::Stream << "Update on Non Existing Point" << endl;
			Console::writeAsError();
		}
	}
	updateChanged();
	return ReturnValue;
}

bool PointNode::changingOnUpdate(const uint_fast16_t value) const
{
	if(myPoint)
	{
		return myPoint->changingOnUpdate(0);
	}
	else
	{
		Console::Stream << "Update on Non Existing Point" << endl;
		Console::writeAsError();

		return false;
	}
}

void PointNode::setNeighbour(const uint_fast16_t Position, PointNode* Me)
{
	if(!myNeighbours)
	{
		myNeighbours = new vector<PointNode*>(Position + 1, NULL);
	}
	else if(myNeighbours->size() <= Position)
	{
		myNeighbours->resize(Position + 1, NULL);
	}
	//Insert PointNode
	myNeighbours->at(Position) = Me;

	//Update Changed
	if(changed)
	{
		Me->setChanged(true, false, true, false, 1);
	}
}

PointNode* PointNode::getNeighbour(uint_fast16_t Position) const
{
	if(myNeighbours && myNeighbours->size() > Position)
	{
		return myNeighbours->at(Position);
	}
	else
	{
		return NULL;
	}
}

vector<PointNode*>* PointNode::getNeighbours() const
{
	return myNeighbours;
}

const uint_fast64_t PointNode::getId()
{
	return this->id;
}

bool operator==(PointNode& n1, PointNode& n2)
{
	if ((n1.getId() == n2.getId()) &&
			(n1.getChanged() == n2.getChanged()) &&
			(n1.getActiveValue() == n2.getActiveValue()) &&
			(n1.getTempValue() == n2.getTempValue()) &&
			(n1.getPosition()->size() == n2.getPosition()->size()))
	{
		for (unsigned int i = 0; i < n1.getPosition()->size(); i++)
		{
			if (n1.getPosition()->at(i) != n2.getPosition()->at(i))
			{
				return false;
			}
		}
		return true;
	}
	return false;
}

bool operator!=(PointNode& n1, PointNode& n2)
{
	return !(n1 == n2);
}