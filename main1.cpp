// main1.cpp MODIFICADO
#include "genetic_algorithm1.h"
#include <iostream>
#include <string>
#include <limits>
#include <stdexcept>
#include <sstream>

template<typename T>
T getInput(const std::string& prompt, T min_val, T max_val) {
  
    T value;
    std::string line;
    while (true) {
        std::cout << prompt;
        std::getline(std::cin, line);
        std::stringstream ss(line);

        if (ss >> value && (ss.eof() || ss.peek() == EOF)) {
            if (value >= min_val && value <= max_val) {
                return value;
            } else {
                std::cout << "Valor fuera de rango permitido [" << min_val << " - " << max_val << "]. Intente de nuevo." << std::endl;
            }
        } else {
            std::cout << "Entrada invalida (no es un numero valido o tiene caracteres extra). Por favor, intente de nuevo." << std::endl;
        }
    }
}

int main() {
    double pc, pm;
    int gmax, pop_s, bits_por_variable, elitism_c; 
    unsigned int seed_val;

    GeneticAlgorithm::clearScreen();
    std::cout << "========================================================" << std::endl;
    std::cout << "      CONFIGURACION DEL ALGORITMO GENETICO" << std::endl;
    std::cout << "========================================================" << std::endl;

    try {
        pm = getInput<double>("Este algoritmo usa mutación unforme por bit. Considera una tasa baja. Tasa de mutacion POR BIT (Pm) [ej. 0.001-0.05, max 1.0] -> ", 0.0, 1.0);
        pc = getInput<double>("Probabilidad de cruce (Pc) [0.0-1.0] ----------------> ", 0.0, 1.0);
        gmax = getInput<int>("Maximo numero de generaciones [1-10000] -----------> ", 1, 10000);
        pop_s = getInput<int>("Tamano de la poblacion [50-500+] -------------------> ", 50, 2000); 
        
        // --- CAMBIO IMPORTANTE AQUÍ ---
        // Pedir bits por variable en lugar de longitud total del cromosoma
        // Un rango sugerido para bits_per_variable para buena precisión es 30-40.
        // Si NUM_VARIABLES es 4, el total será 4 * bits_por_variable.
        bits_por_variable = getInput<int>("Bits POR VARIABLE [ej. 30-40, para precision decimal] -> ", 20, 50); 
        
        int max_elitism;
        if (pop_s <= 1) {
            max_elitism = 0;
        } else {
            max_elitism = pop_s - 1; 
        }
        elitism_c = getInput<int>("Cantidad de individuos elite [0-" + std::to_string(max_elitism) + "] ----------> ", 0, max_elitism);
        
        seed_val = getInput<unsigned int>("Semilla para numeros aleatorios (0 para tiempo) ----> ", 0, std::numeric_limits<unsigned int>::max());

        // Pasar bits_por_variable al constructor
        GeneticAlgorithm ga(pc, pm, gmax, pop_s, bits_por_variable, seed_val, elitism_c);
        ga.run();

    } catch (const std::invalid_argument& e) {
        std::cerr << "\nError de argumento: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nError inesperado: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nOcurrio un error desconocido." << std::endl;
        return 1;
    }
    return 0;
}