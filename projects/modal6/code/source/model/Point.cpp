///////////////////////////////////////////////////////////
//  Created on:      07-Apr-2016 10:59:27
//  Original author: Bonifaz Stuhr & Tobias Eigler
///////////////////////////////////////////////////////////

#include "model/Point.h"

const uint_fast16_t Point::emptyValue = 0;

Point::Point()
{
    setAllValue(emptyValue);
}

Point::Point(const uint_fast16_t value)
{
    setAllValue(value);
}

Point::~Point()
{
}

void Point::setAllValue(const uint_fast16_t Value)
{
    this->activeValue = Value;
    this->tempValue = Value;
}

//Liefert einen String, der den aktuellen Wert des Punktes beinhaltet.
uint_fast16_t Point::getActiveValue() const
{
    return activeValue;
}

//Ermoeglicht das setzten des temporaeren Wertes.
void Point::setTempValue(const uint_fast16_t value)
{
    this->tempValue = value;
}

uint_fast16_t Point::getTempValue() const
{
    return tempValue;
}

bool Point::changingOnUpdate() const
{
    return (activeValue != tempValue);
}

bool Point::changingOnUpdate(const uint_fast16_t value) const
{
    return (changingOnUpdate() && activeValue == value);
}

//Setzt den temporaeren Wert als aktuellen Wert.
bool Point::update()
{
    bool changedEmpty = changingOnUpdate(0);
    this->activeValue = this->tempValue;
    return changedEmpty;
}