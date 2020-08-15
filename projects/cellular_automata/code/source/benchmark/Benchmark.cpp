///////////////////////////////////////////////////////////
//  Created on:      21-Apr-2016 14:43:27
//  Original author: Michell Mueller
///////////////////////////////////////////////////////////

#include "benchmark/Benchmark.h"
#include "controller/Console.h"

Benchmark* Benchmark::instance = NULL;

Benchmark::Benchmark()
{
    // Standard-Dateiname beginnt mit "Benchmark"
    file = "Benchmark";
}

Benchmark::~Benchmark()
{
    stream.close();
}

Benchmark* Benchmark::getInstance()
{
    if (!instance)
    {
        instance = new Benchmark();

        // Aktuelle Zeit ermitteln
        time_t timeStamp;
        timeStamp = time(0);
        tm* now = localtime(&timeStamp);

        // Standard-Dateinamen anhand aktueller Zeit modifizieren, damit dieser eindeutig ist
        string timeSet = utils::to_string(now->tm_mday) + "_" +
                         utils::to_string(now->tm_mon + 1) + "_" +
                         utils::to_string(now->tm_year + 1900) + "  " +
                         utils::to_string(now->tm_hour) + "-" +
                         utils::to_string(now->tm_min) + "-" +
                         utils::to_string(now->tm_sec);
        instance->file += "_" + timeSet + ".txt";

        // Datei erstellen
        instance->stream.open(instance->file.c_str(), ios::out);

        Console::Stream << "Benchmark Output: " << instance->file << endl;
        Console::writeAsMessage();

        // Datum und Uhrzeit mit gewohnten Zeichen trennen (Als Dateiname sind diese nicht zulaessig)
        // "_" wird zu "." und "-" wird zu ":"
        string reg[2][2] = { { "_", "." }, { "-", ":" } };
        for (int i = 0, j = 0; i < 2; i++)
        {
            boost::regex expr(reg[i][j]);
            timeSet = regex_replace(timeSet, expr, reg[i][j+1]);
        }

        // Datei-Inhalt mit Header beschreiben
        instance->stream << "Benchmark vom " << timeSet  << " Uhr" << endl << endl;

    }
    return instance;
}

void Benchmark::start(string operationName)
{
    // Wird zur Ueberpruefung beim Einfuegen verwendet, ob der operationName bereits in der Map vorhanden ist
    pair<map<string, boost::chrono::time_point<boost::chrono::steady_clock> >::iterator, bool> insertState;

    // Einfuegen in die Map
    insertState = this->benchmarkMap.insert(pair<string, boost::chrono::time_point<boost::chrono::steady_clock> >(operationName, boost::chrono::high_resolution_clock::now()));
    if (insertState.second == false)
    {
        Console::Stream << "operationName " << operationName << " already exists." << endl;
        Console::writeAsError();
    }
}

void Benchmark::stop(string operationName, string comment)
{
    // Holt die Clock mit dem uebergebenen operationName und stoppt die Zeitmessung
    try
    {
        // Aktuelle Zeit beim Stoppen der Messung
        boost::chrono::time_point<boost::chrono::steady_clock> t_end = boost::chrono::high_resolution_clock::now();
        // Startzeit der Messung
        boost::chrono::time_point<boost::chrono::steady_clock> t_start = benchmarkMap.at(operationName);

        // Berechnung der gebrauchten Zeit
        boost::chrono::duration<double, boost::milli> fp_ms = t_end - t_start;

        // Ausgabe der gebrauchten Zeit in der Datei
        instance->stream << "OperationName: " << operationName << endl << "Zeit: " << fp_ms.count() << " ms" << endl;

        // Hinzufuegen des Kommentars (falls vorhanden)
        if (comment != "")
        {
            instance->stream << "Comment: " << comment << endl;
        }

        // Leerzeile zwischen unterschiedlichen Messungen
        instance->stream << endl;

        // Entfernen der bereits ausgegebenen Messung aus der Map
        benchmarkMap.erase(operationName);
    } // Error-Handling, falls es den Key in der Map nicht gibt
    catch (const out_of_range& oor)
    {
        Console::Stream << "Out of Range error: " << oor.what() << endl;
        Console::writeAsException();
    }
}