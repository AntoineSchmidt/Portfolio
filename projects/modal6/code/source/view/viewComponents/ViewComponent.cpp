///////////////////////////////////////////////////////////
//  Created on:      10-Apr-2016 10:59:26
//  Author:          Antoine Schmidt
///////////////////////////////////////////////////////////

#include <iostream>
#include "view/viewComponents/ViewComponent.h"
#include "controller/Controller.h"

ViewComponent::ViewComponent()
{
    UpdatePerFrame = 1;
}

ViewComponent::~ViewComponent()
{

}

//Prueft auf notwendigkeit der Ausfuehrung
void ViewComponent::update()
{
    executeEveryStep();
    if(Controller::getInstance()->getStepCount() % UpdatePerFrame == 0)
    {
        executeupdate();
    }
}

void ViewComponent::executeEveryStep()
{

}