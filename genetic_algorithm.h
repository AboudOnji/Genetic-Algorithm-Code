#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <vector>
#include <random>
#include <string>
#include <iostream>
#include <fstream>   // Para std::ofstream
#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <functional>

class GeneticAlgorithm {
public:
    // ... (struct Individual sin cambios) ...
    struct Individual {
        std::vector<unsigned int> chrom;
        double x_value;
        double fitness;
        int parent1_idx;
        int parent2_idx;
        int crossover_site1;
        int crossover_site2;
        int mutation_bit_idx;
        bool mutated_flag;

        Individual(int chrom_size = 0) :
            chrom(chrom_size, 0), x_value(0.0), fitness(0.0),
            parent1_idx(-1), parent2_idx(-1), crossover_site1(-1), crossover_site2(-1),
            mutation_bit_idx(-1), mutated_flag(false) {}

        bool operator<(const Individual& other) const {
            return fitness < other.fitness;
        }
    };


    GeneticAlgorithm(double prob_crossover,
                     double prob_mutation,
                     int max_generations,
                     int population_size,
                     int chromosome_size,
                     unsigned int seed,
                     int elitism_count = 1);

    ~GeneticAlgorithm();

    void run();
    static void clearScreen();

private:
    // ... (miembros existentes sin cambios) ...
    double prob_crossover_;
    double prob_mutation_;
    int max_generations_;
    int population_size_;
    int chromosome_size_;
    unsigned int initial_seed_value_;
    int elitism_count_;

    std::vector<Individual> current_population_;
    std::vector<Individual> next_population_;
    Individual best_ever_individual_;
    int generation_of_best_ever_;
    int current_generation_;

    long long total_crossovers_;
    long long total_mutations_;

    double min_fitness_current_gen_;
    double max_fitness_current_gen_;     // <--- Ya tenemos este dato por generación
    double avg_fitness_current_gen_;
    double sum_fitness_current_gen_;
    Individual best_individual_current_gen_;

    std::mt19937 rng_engine_;
    std::uniform_real_distribution<double> uniform_dist_01_;
    std::vector<int> tournament_indices_;
    int tournament_pos_counter_;

    std::string results_txt_filename_;
    std::ofstream results_txt_file_stream_; // Para el reporte detallado TXT

    // NUEVOS MIEMBROS PARA EL LOG DE CONVERGENCIA
    std::string convergence_csv_filename_;
    std::ofstream convergence_csv_file_stream_;


    // --- Funciones Miembro Privadas ---
    // ... (funciones existentes sin cambios en su declaración, excepto las de reporte) ...
    void initializePopulation();
    void evaluateIndividual(Individual& individual);
    void evaluatePopulation(std::vector<Individual>& population);
    double decodeChromosome(const std::vector<unsigned int>& chrom) const;
    double objectiveFunction(double decoded_value) const;
    void calculateStatistics(const std::vector<Individual>& population); // <--- Aquí se calcula max_fitness_current_gen_
    void selectAndReproduce();
    void applyElitism();
    int tournamentSelection(std::function<int()> next_candidate_idx_provider);
    void performCrossover(const Individual& parent1, const Individual& parent2,
                          Individual& child1, Individual& child2);
    void performMutation(Individual& individual);

    void printInitialParametersToConsole() const;
    void reportCurrentGenerationToConsole(); 
    void printChromosome(const std::vector<unsigned int>& chrom, std::ostream& os = std::cout) const;
    std::string getChromosomeString(const std::vector<unsigned int>& chrom) const;

    // Funciones para el log TXT formateado
    void openResultsTxtFileAndWriteParams();
    void writeParametersToTxtFile();
    void logGenerationDataToTxt(int generation_num, const std::vector<Individual>& population);
    void writeFinalSummaryToTxtFile();

    // NUEVAS FUNCIONES PARA EL LOG DE CONVERGENCIA CSV
    void openConvergenceCsvFileAndWriteHeader();
    void logConvergencePoint(int generation_num, double max_fitness);

    // Funciones RNG (ya declaradas)
    bool getBernoulliOutcome(double probability);
    int getRandomInt(int min_val, int max_val);
};

#endif // GENETIC_ALGORITHM_H