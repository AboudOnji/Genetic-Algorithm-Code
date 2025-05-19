#include "genetic_algorithm.h"
#include <iostream>
#include <string>   // Para std::string, std::stod, std::stoi
#include <limits>   // Para std::numeric_limits
#include <stdexcept> // Para std::exception

template<typename T>
T getInput(const std::string& prompt, T min_val, T max_val) {
    T value;
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.good() && value >= min_val && value <= max_val) {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Limpiar buffer
            return value;
        } else {
            std::cout << "Entrada invalida. Por favor, intente de nuevo." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
}


int main() {
    double pc, pm;
    int gmax, pop_s, code_s, elitism_c;
    unsigned int seed;

    GeneticAlgorithm::clearScreen();
    std::cout << "========================================================" << std::endl;
    std::cout << "      CONFIGURACION DEL ALGORITMO GENETICO" << std::endl;
    std::cout << "========================================================" << std::endl;

    try {
        pm = getInput<double>("Tasa de mutacion (Pm) [0.0-1.0] ---------> ", 0.0, 1.0);
        pc = getInput<double>("Probabilidad de cruce (Pc) [0.0-1.0] ----> ", 0.0, 1.0);
        gmax = getInput<int>("Maximo numero de generaciones [1-10000] --> ", 1, 10000);
        pop_s = getInput<int>("Tamano de la poblacion [10-1000] --------> ", 10, 1000);
        code_s = getInput<int>("Longitud del cromosoma (bits) [4-64] ----> ", 4, 64);
        elitism_c = getInput<int>("Cantidad de individuos elite [0-" + std::to_string(pop_s -1) + "] --> ", 0, pop_s > 1 ? pop_s -1 : 0); // Elite no puede ser >= pop_s
        seed = getInput<unsigned int>("Semilla para numeros aleatorios (0 para tiempo) -> ", 0, std::numeric_limits<unsigned int>::max());

        GeneticAlgorithm ga(pc, pm, gmax, pop_s, code_s, seed, elitism_c);
        ga.run();

    } catch (const std::invalid_argument& e) {
        std::cerr << "Error de argumento: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error inesperado: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Ocurrio un error desconocido." << std::endl;
        return 1;
    }

    return 0;
}