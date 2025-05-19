#ifndef GENETIC_ALGORITHM1_H
#define GENETIC_ALGORITHM1_H

#include <vector>
#include <random>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <functional>

class GeneticAlgorithm {
public:
    static const int NUM_VARIABLES = 4;

    struct Individual {
        std::vector<unsigned int> chrom;
        std::vector<double> x_values; 
        double objective_value;      
        double fitness;
        
        int parent1_idx;
        int parent2_idx;
        int crossover_site1; 
        int crossover_site2; 
        int mutation_bit_idx; // Conteo de bits mutados
        bool mutated_flag;

        Individual(int total_chrom_size = 0, int num_vars = NUM_VARIABLES) :
            chrom(total_chrom_size, 0),
            x_values(num_vars, 0.0), 
            objective_value(0.0),
            fitness(0.0),
            parent1_idx(-1), parent2_idx(-1), crossover_site1(-1), crossover_site2(-1),
            mutation_bit_idx(0), mutated_flag(false) {}

        bool operator<(const Individual& other) const {
            return fitness < other.fitness; 
        }
    };

    GeneticAlgorithm(double prob_crossover,
                     double prob_mutation,
                     int max_generations,
                     int population_size,
                     int bits_per_variable,
                     unsigned int seed,
                     int elitism_count = 1);

    ~GeneticAlgorithm();

    void run();
    static void clearScreen();

private:
    double prob_crossover_;
    double prob_mutation_;
    int max_generations_;
    int population_size_;
    int bits_per_variable_;
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
    long long crossovers_this_generation_; 
    long long mutations_this_generation_;  

    double min_fitness_current_gen_; 
    double max_fitness_current_gen_; 
    double avg_fitness_current_gen_; 
    double sum_fitness_current_gen_; 
    Individual best_individual_current_gen_;
    double min_objective_value_overall_; 

    std::mt19937 rng_engine_;
    std::uniform_real_distribution<double> uniform_dist_01_;

    std::string results_txt_filename_;
    std::ofstream results_txt_file_stream_;

    std::string convergence_csv_filename_;
    std::ofstream convergence_csv_file_stream_;

    void initializePopulation();
    void evaluateIndividual(Individual& individual); 
    void evaluatePopulation(std::vector<Individual>& population);
    void decodeChromosome(const std::vector<unsigned int>& full_chrom, std::vector<double>& x_values) const; 
    double calculateObjectiveFunction(const std::vector<double>& x_values) const; 
    double calculateFitness(double objective_value) const; 

    void calculateStatistics(const std::vector<Individual>& population);
    void selectAndReproduce();
    void applyElitism();
    int rouletteWheelSelection(); 
    void performCrossover(const Individual& parent1, const Individual& parent2,
                          Individual& child1, Individual& child2);
    void performMutation(Individual& individual); 

    void printInitialParametersToConsole() const;
    void reportCurrentGenerationToConsole(); 
    void printChromosome(const std::vector<unsigned int>& chrom, std::ostream& os = std::cout) const;
    std::string getChromosomeString(const std::vector<unsigned int>& chrom) const;

    void openResultsTxtFileAndWriteParams();
    void writeParametersToTxtFile();
    void logGenerationDataToTxt(int generation_num, const std::vector<Individual>& population); 
    void writeFinalSummaryToTxtFile(); 

    void openConvergenceCsvFileAndWriteHeader();
    void logConvergencePoint(int generation_num, double max_fitness, double best_objective_val);

    bool getBernoulliOutcome(double probability);
    int getRandomInt(int min_val, int max_val);
};

#endif // GENETIC_ALGORITHM1_H