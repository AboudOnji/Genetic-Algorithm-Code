#include "genetic_algorithm.h" // Asegúrate que el nombre del include coincida con tu .h
#include <numeric> // Para std::iota
#include <stdexcept> 

// --- Espacio de nombres anónimo para helpers ---
namespace {
    std::string generateTimestampedFilename(const std::string& prefix, const std::string& extension) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm buf{};
#ifdef _WIN32
        localtime_s(&buf, &in_time_t);
#else
        localtime_r(&in_time_t, &buf);
#endif
        std::ostringstream ss;
        ss << prefix;
        ss << std::put_time(&buf, "%Y%m%d_%H%M%S");
        ss << extension;
        return ss.str();
    }
} // Fin del espacio de nombres anónimo

// --- Definiciones de las funciones RNG ---
bool GeneticAlgorithm::getBernoulliOutcome(double probability) {
    if (probability < 0.0) probability = 0.0;
    if (probability > 1.0) probability = 1.0;
    return std::bernoulli_distribution(probability)(rng_engine_);
}

int GeneticAlgorithm::getRandomInt(int min_val, int max_val) {
    if (min_val > max_val) {
        return min_val; 
    }
    if (min_val == max_val) return min_val;
    return std::uniform_int_distribution<int>(min_val, max_val)(rng_engine_);
}

// --- Constructor ---
GeneticAlgorithm::GeneticAlgorithm(double prob_crossover,
                                   double prob_mutation,
                                   int max_generations,
                                   int population_size,
                                   int chromosome_size,
                                   unsigned int seed,
                                   int elitism_count)
    : prob_crossover_(prob_crossover),
      prob_mutation_(prob_mutation),
      max_generations_(max_generations),
      population_size_(population_size),
      chromosome_size_(chromosome_size),
      initial_seed_value_(seed),
      elitism_count_(std::max(0, elitism_count)),
      generation_of_best_ever_(-1),
      current_generation_(0),
      total_crossovers_(0),
      total_mutations_(0),
      min_fitness_current_gen_(0.0),
      max_fitness_current_gen_(0.0),
      avg_fitness_current_gen_(0.0),
      sum_fitness_current_gen_(0.0),
      rng_engine_(seed == 0 ? static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) : seed),
      uniform_dist_01_(0.0, 1.0)
      // tournament_pos_counter_ fue eliminado porque ya no se usa torneo
      {

    if (population_size_ < 0 || chromosome_size_ < 0 || max_generations_ < 0) {
        throw std::invalid_argument("El tamano de poblacion, la longitud de cromosoma y las generaciones no pueden ser negativos.");
    }
    if (chromosome_size_ == 0 && population_size_ > 0) { 
        throw std::invalid_argument("La longitud del cromosoma no puede ser cero si el tamano de la poblacion es mayor que cero.");
    }
    if (prob_crossover_ < 0.0 || prob_crossover_ > 1.0 || prob_mutation_ < 0.0 || prob_mutation_ > 1.0) {
        throw std::invalid_argument("Las probabilidades deben estar entre 0.0 y 1.0.");
    }
    if (elitism_count_ >= population_size_ && population_size_ > 0) {
        throw std::invalid_argument("El conteo de elitismo debe ser menor que el tamano de la poblacion.");
    }

    if (population_size_ > 0) {
        current_population_.resize(population_size_, Individual(chromosome_size_));
        next_population_.resize(population_size_, Individual(chromosome_size_));
        // tournament_indices_ ya no es necesario
    }
    best_ever_individual_ = Individual(chromosome_size_);
    best_individual_current_gen_ = Individual(chromosome_size_);

    openResultsTxtFileAndWriteParams();
    openConvergenceCsvFileAndWriteHeader();
}

// --- Destructor ---
GeneticAlgorithm::~GeneticAlgorithm() {
    if (results_txt_file_stream_.is_open()) {
        writeFinalSummaryToTxtFile();
        results_txt_file_stream_.close();
        std::cout << "\nLog detallado de la corrida (tabla TXT) guardado en: " << results_txt_filename_ << std::endl;
    }
    if (convergence_csv_file_stream_.is_open()) {
        convergence_csv_file_stream_.close();
        std::cout << "Datos de convergencia (para graficar) guardados en: " << convergence_csv_filename_ << std::endl;
    }
}

// --- Manejo de Archivos ---
void GeneticAlgorithm::openResultsTxtFileAndWriteParams() {
    results_txt_filename_ = generateTimestampedFilename("GA_ReporteDetallado_", ".txt");
    results_txt_file_stream_.open(results_txt_filename_);
    if (!results_txt_file_stream_.is_open()) {
        std::cerr << "Error CRITICO: No se pudo abrir el archivo TXT de resultados: " << results_txt_filename_ << std::endl;
    } else {
        writeParametersToTxtFile();
    }
}

void GeneticAlgorithm::writeParametersToTxtFile() {
    if (!results_txt_file_stream_.is_open()) return;
    results_txt_file_stream_ << "========================================================================================================================" << std::endl;
    results_txt_file_stream_ << "                            PARAMETROS DEL ALGORITMO GENETICO" << std::endl;
    results_txt_file_stream_ << "========================================================================================================================" << std::endl;
    results_txt_file_stream_ << "Tamano de la poblacion        : " << population_size_ << std::endl;
    results_txt_file_stream_ << "Longitud del cromosoma (bits) : " << chromosome_size_ << std::endl;
    results_txt_file_stream_ << "Maximo numero de generaciones : " << max_generations_ << std::endl;
    results_txt_file_stream_ << "Probabilidad de cruce (Pc)    : " << std::fixed << std::setprecision(3) << prob_crossover_ << std::endl;
    results_txt_file_stream_ << "Probabilidad de mutacion (Pm)   : " << prob_mutation_ << " (por bit)" << std::endl;
    results_txt_file_stream_ << "Conteo de elitismo            : " << elitism_count_ << std::endl;
    results_txt_file_stream_ << "Semilla RNG (entrada)         : " << initial_seed_value_
                             << (initial_seed_value_ == 0 ? " (tiempo actual usado)" : "") << std::endl;
    results_txt_file_stream_ << "========================================================================================================================" << std::endl << std::endl;
}

void GeneticAlgorithm::logGenerationDataToTxt(int generation_num, const std::vector<Individual>& population) {
    if (!results_txt_file_stream_.is_open() || (population.empty() && !(generation_num == 0 && population_size_==0)) ) return;

    results_txt_file_stream_ << "----------------------------------------------- GENERACION # " << std::setw(3) << generation_num 
                             << " -----------------------------------------------" << std::endl;
    
    const int W_ID = 4;
    const int W_CHROM_PLACEHOLDER = std::max(10, chromosome_size_) + 2; // Ancho que ocuparía el cromosoma
    const int W_XVAL = 10;
    const int W_FIT = 9;
    const int W_PARENTS = 10; 
    const int W_XSITE = 8; 
    const int W_MUT_INFO = 12;

    // Encabezado de la tabla para esta generación
    results_txt_file_stream_ << std::left
                             << std::setw(W_ID) << "ID";
    // << std::setw(W_CHROM_PLACEHOLDER) << "Cromosoma"; // <--- LÍNEA DE ENCABEZADO DEL CROMOSOMA DESACTIVADA
    results_txt_file_stream_ << std::setw(W_CHROM_PLACEHOLDER) << " "; // Espacio para la columna del cromosoma desactivada
    results_txt_file_stream_ << std::setw(W_XVAL) << "X Val"
                             << std::setw(W_FIT) << "Fitness"
                             << std::setw(W_PARENTS) << "Padres"
                             << std::setw(W_XSITE) << "X Sitio"
                             << std::setw(W_MUT_INFO) << "Mut? (Conteo)" << std::endl;
    // La línea separadora puede seguir usando el ancho original o ajustarse
    results_txt_file_stream_ << std::string(W_ID + W_CHROM_PLACEHOLDER + W_XVAL + W_FIT + W_PARENTS + W_XSITE + W_MUT_INFO, '-') << std::endl;

    if (population.empty() && population_size_ > 0) {
         results_txt_file_stream_ << " (Poblacion vacia para esta generacion en el log)" << std::endl;
    }

    for (size_t i = 0; i < population.size(); ++i) {
        const auto& ind = population[i];
        results_txt_file_stream_ << std::left << std::setw(W_ID) << i;
        // results_txt_file_stream_ << std::setw(W_CHROM_PLACEHOLDER) << getChromosomeString(ind.chrom); // <--- LÍNEA DE DATOS DEL CROMOSOMA DESACTIVADA
        results_txt_file_stream_ << std::setw(W_CHROM_PLACEHOLDER) << " "; // Espacio para la columna del cromosoma desactivada
        results_txt_file_stream_ << std::fixed << std::setprecision(4) << std::setw(W_XVAL) << ind.x_value;
        results_txt_file_stream_ << std::fixed << std::setprecision(5) << std::setw(W_FIT) << ind.fitness;

        std::ostringstream parents_ss;
        if (ind.parent1_idx == -1) parents_ss << "(Ini)";
        else if (ind.parent1_idx == -2) parents_ss << "(Eli)";
        else if (ind.parent1_idx == -3) parents_ss << "(Rell)";
        else parents_ss << "(" << ind.parent1_idx << "," << ind.parent2_idx << ")";
        results_txt_file_stream_ << std::setw(W_PARENTS) << parents_ss.str();

        std::ostringstream xsite_ss; 
        if (ind.crossover_site1 != -1) xsite_ss << ind.crossover_site1;
        else xsite_ss << "---";
        results_txt_file_stream_ << std::setw(W_XSITE) << xsite_ss.str();

        std::ostringstream mut_ss;
        mut_ss << (ind.mutated_flag ? "Y" : "N") << " (" << ind.mutation_bit_idx << ")";
        results_txt_file_stream_ << std::setw(W_MUT_INFO) << mut_ss.str();
        results_txt_file_stream_ << std::endl;
    }

    results_txt_file_stream_ << "Estadisticas Gen #" << generation_num << ": "
                             << "MinFit=" << std::fixed << std::setprecision(5) << min_fitness_current_gen_
                             << " | MaxFit=" << max_fitness_current_gen_
                             << " | AvgFit=" << avg_fitness_current_gen_ 
                             << " | SumFit=" << sum_fitness_current_gen_ << std::endl;
    results_txt_file_stream_ << "  Mejor esta Gen: " << getChromosomeString(best_individual_current_gen_.chrom) // <--- ESTE SE MANTIENE
                             << " (Fit: " << best_individual_current_gen_.fitness << ")" << std::endl;
    results_txt_file_stream_ << std::endl; 
}

void GeneticAlgorithm::writeFinalSummaryToTxtFile() {
    if (!results_txt_file_stream_.is_open()) return;
    results_txt_file_stream_ << "========================================================================================================================" << std::endl;
    results_txt_file_stream_ << "                                RESUMEN FINAL DE LA CORRIDA" << std::endl;
    results_txt_file_stream_ << "========================================================================================================================" << std::endl;
    if (population_size_ > 0 && chromosome_size_ > 0) {
        results_txt_file_stream_ << "Total de Generaciones Procesadas: " << max_generations_ << std::endl;
        results_txt_file_stream_ << "Mejor individuo global encontrado en la generacion: " << generation_of_best_ever_ << std::endl;
        results_txt_file_stream_ << "  Fitness:   " << std::fixed << std::setprecision(5) << best_ever_individual_.fitness << std::endl;
        results_txt_file_stream_ << "  Valor X:   " << std::fixed << std::setprecision(4) << best_ever_individual_.x_value << std::endl;
        results_txt_file_stream_ << "  Cromosoma: " << getChromosomeString(best_ever_individual_.chrom) << std::endl; // <--- ESTE SE MANTIENE
        results_txt_file_stream_ << "Estadisticas Acumuladas Globales:" << std::endl;
        results_txt_file_stream_ << "  Total de Cruces: " << total_crossovers_ << std::endl;
        results_txt_file_stream_ << "  Total de Mutaciones (bits invertidos): " << total_mutations_ << std::endl;
    } else {
        results_txt_file_stream_ << "Ejecucion no realizada o con parametros invalidos (poblacion/cromosoma tamano cero)." << std::endl;
    }
    results_txt_file_stream_ << "========================================================================================================================" << std::endl;
}

void GeneticAlgorithm::openConvergenceCsvFileAndWriteHeader() {
    convergence_csv_filename_ = generateTimestampedFilename("GA_Convergencia_", ".csv");
    convergence_csv_file_stream_.open(convergence_csv_filename_);
    if (!convergence_csv_file_stream_.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo de datos de convergencia: "
                  << convergence_csv_filename_ << std::endl;
    } else {
        convergence_csv_file_stream_ << "Generation,MaxFitness" << std::endl;
    }
}

void GeneticAlgorithm::logConvergencePoint(int generation_num, double max_fitness) {
    if (convergence_csv_file_stream_.is_open()) {
        convergence_csv_file_stream_ << generation_num << ","
                                     << std::fixed << std::setprecision(5)
                                     << max_fitness << std::endl;
    }
}

void GeneticAlgorithm::run() {
    clearScreen();
    printInitialParametersToConsole(); 

    if (population_size_ == 0 || chromosome_size_ == 0) {
        std::cout << "\nAdvertencia: Tamano de poblacion o longitud de cromosoma es cero. El AG no se ejecutara." << std::endl;
        return;
    }

    initializePopulation();
    evaluatePopulation(current_population_);
    calculateStatistics(current_population_);
    
    if (!current_population_.empty()) {
        best_ever_individual_ = best_individual_current_gen_;
    }
    generation_of_best_ever_ = 0;

    reportCurrentGenerationToConsole();      
    logGenerationDataToTxt(0, current_population_); 
    logConvergencePoint(0, max_fitness_current_gen_);

    for (current_generation_ = 1; current_generation_ <= max_generations_; ++current_generation_) {
        selectAndReproduce(); 
        evaluatePopulation(next_population_);
        calculateStatistics(next_population_); 
        
        current_population_ = next_population_; 

        reportCurrentGenerationToConsole();      
        logGenerationDataToTxt(current_generation_, current_population_); 
        logConvergencePoint(current_generation_, max_fitness_current_gen_);
    }
    
    std::cout << "\n\n========================================================" << std::endl;
    std::cout << "            ALGORITMO GENETICO FINALIZADO (Consola)" << std::endl;
    std::cout << "========================================================" << std::endl;
    if (population_size_ > 0 && chromosome_size_ > 0) {
        std::cout << "Total de Generaciones Procesadas: " << max_generations_ << std::endl;
        std::cout << "Mejor individuo global encontrado en la generacion: " << generation_of_best_ever_ << std::endl;
        std::cout << "  Fitness:   " << std::fixed << std::setprecision(5) << best_ever_individual_.fitness << std::endl;
        std::cout << "  Valor X:   " << std::fixed << std::setprecision(4) << best_ever_individual_.x_value << std::endl;
        std::cout << "  Cromosoma: ";
        printChromosome(best_ever_individual_.chrom, std::cout); // <--- ESTE SE MANTIENE
        std::cout << std::endl;
    }
}

std::string GeneticAlgorithm::getChromosomeString(const std::vector<unsigned int>& chrom) const {
    if (chrom.empty()){
        int placeholder_len = chromosome_size_;
        if (chromosome_size_ == 0) placeholder_len = 0;
        return std::string(placeholder_len > 0 ? placeholder_len : 0, '-');
    }
    std::string s;
    s.reserve(chrom.size()); 
    for (int j = static_cast<int>(chrom.size()) - 1; j >= 0; --j) {
        s += std::to_string(chrom[j]);
    }
    return s;
}

void GeneticAlgorithm::initializePopulation() {
    if (population_size_ == 0 || chromosome_size_ == 0) return;
    for (int i = 0; i < population_size_; ++i) {
        current_population_[i] = Individual(chromosome_size_); 
        for (int j = 0; j < chromosome_size_; ++j) {
            current_population_[i].chrom[j] = getBernoulliOutcome(0.5) ? 1 : 0;
        }
        current_population_[i].parent1_idx = -1; 
        current_population_[i].parent2_idx = -1; 
        current_population_[i].crossover_site1 = -1;
        current_population_[i].crossover_site2 = -1;
        current_population_[i].mutated_flag = false;
        current_population_[i].mutation_bit_idx = 0;
    }
}

void GeneticAlgorithm::evaluateIndividual(Individual& individual) {
    if (individual.chrom.size() != static_cast<size_t>(chromosome_size_)) {
         if (chromosome_size_ > 0) {
            individual.chrom.assign(chromosome_size_, 0);
         } else { 
            individual.x_value = 0.0;
            individual.fitness = 0.0; 
            return;
         }
    }
    if (chromosome_size_ == 0) {
        individual.x_value = 0.0;
        individual.fitness = 0.0; 
        return;
    }
    individual.x_value = decodeChromosome(individual.chrom);
    individual.fitness = objectiveFunction(individual.x_value);
}

void GeneticAlgorithm::evaluatePopulation(std::vector<Individual>& population) {
    for (Individual& ind : population) {
        evaluateIndividual(ind);
    }
}

double GeneticAlgorithm::decodeChromosome(const std::vector<unsigned int>& chrom) const {
    if (chrom.empty()) return 0.0;
    double accumulator = 0.0;
    double power_of_2 = 1.0; 
    for (unsigned int bit : chrom) {
        if (bit == 1) {
            accumulator += power_of_2;
        }
        power_of_2 *= 2.0;
    }
    return accumulator;
}

double GeneticAlgorithm::objectiveFunction(double decoded_value) const {
    if (chromosome_size_ == 0) return 0.0;
    double max_possible_decoded_value = std::pow(2.0, static_cast<double>(chromosome_size_)) - 1.0;
    if (std::abs(max_possible_decoded_value) < 1e-9) { 
         return (std::abs(decoded_value) < 1e-9) ? 1.0 : 0.0;
    }
    double normalized_value = decoded_value / max_possible_decoded_value;
    double fitness = std::pow(normalized_value, 2.0);
    return std::floor(fitness * 10000.0 + 0.5) / 10000.0;
}

void GeneticAlgorithm::calculateStatistics(const std::vector<Individual>& population) {
    if (population.empty()) {
        min_fitness_current_gen_ = 0.0; max_fitness_current_gen_ = 0.0;
        avg_fitness_current_gen_ = 0.0; sum_fitness_current_gen_ = 0.0;
        best_individual_current_gen_ = Individual(chromosome_size_);
        return;
    }
    sum_fitness_current_gen_ = 0.0; 
    min_fitness_current_gen_ = population[0].fitness; 
    max_fitness_current_gen_ = population[0].fitness;
    best_individual_current_gen_ = population[0];
    for (const Individual& ind : population) {
        sum_fitness_current_gen_ += ind.fitness;
        if (ind.fitness < min_fitness_current_gen_) min_fitness_current_gen_ = ind.fitness;
        if (ind.fitness > max_fitness_current_gen_) {
            max_fitness_current_gen_ = ind.fitness;
            best_individual_current_gen_ = ind;
        }
    }
    if (population.size() > 0) {
        avg_fitness_current_gen_ = sum_fitness_current_gen_ / static_cast<double>(population.size());
    } else {
        avg_fitness_current_gen_ = 0.0;
    }

    bool update_best_ever = false;
    if (chromosome_size_ > 0 && !best_individual_current_gen_.chrom.empty()) {
        if (best_individual_current_gen_.fitness > best_ever_individual_.fitness || generation_of_best_ever_ == -1) {
            update_best_ever = true;
        }
    } else if (chromosome_size_ == 0) { 
         if (best_individual_current_gen_.fitness > best_ever_individual_.fitness || generation_of_best_ever_ == -1) {
             update_best_ever = true;
         }
    }
    if(update_best_ever){
        best_ever_individual_ = best_individual_current_gen_;
        generation_of_best_ever_ = current_generation_;
    }
}

int GeneticAlgorithm::rouletteWheelSelection() {
    if (population_size_ == 0) return -1; 
    if (sum_fitness_current_gen_ <= 1e-9) { 
        return getRandomInt(0, population_size_ - 1);
    }
    double random_pick = uniform_dist_01_(rng_engine_) * sum_fitness_current_gen_;
    double current_sum = 0.0;
    for (int i = 0; i < population_size_; ++i) {
        current_sum += current_population_[i].fitness;
        if (current_sum >= random_pick) {
            return i;
        }
    }
    return population_size_ - 1; 
}

void GeneticAlgorithm::performCrossover(const Individual& parent1, const Individual& parent2,
                                      Individual& child1, Individual& child2) {
    child1.chrom = parent1.chrom;
    child2.chrom = parent2.chrom;
    child1.crossover_site1 = -1; child1.crossover_site2 = -1;
    child2.crossover_site1 = -1; child2.crossover_site2 = -1;

    if (!getBernoulliOutcome(prob_crossover_)) return;
    if (chromosome_size_ < 2) return; 

    total_crossovers_++;
    int cut_point = getRandomInt(0, chromosome_size_ - 2); 

    child1.crossover_site1 = cut_point;
    child2.crossover_site1 = cut_point;

    for (int i = cut_point + 1; i < chromosome_size_; ++i) {
        unsigned int temp_bit = child1.chrom[i]; 
        child1.chrom[i] = parent2.chrom[i];      
        child2.chrom[i] = temp_bit;              
    }
}

void GeneticAlgorithm::performMutation(Individual& individual) {
    individual.mutated_flag = false;
    individual.mutation_bit_idx = 0; 

    if (individual.chrom.empty() || chromosome_size_ == 0) return;

    int mutations_in_this_ind = 0;
    for (int i = 0; i < chromosome_size_; ++i) {
        if (getBernoulliOutcome(prob_mutation_)) { 
            individual.chrom[i] = 1 - individual.chrom[i];
            individual.mutated_flag = true;
            mutations_in_this_ind++;
            total_mutations_++; 
        }
    }
    individual.mutation_bit_idx = mutations_in_this_ind;
}

void GeneticAlgorithm::selectAndReproduce() {
    if (population_size_ == 0) return;
    next_population_.assign(population_size_, Individual(chromosome_size_));
    int offspring_idx = 0; 
    applyElitism(); 
    offspring_idx += elitism_count_;

    while (offspring_idx < population_size_) {
        int parent1_idx = rouletteWheelSelection(); 
        int parent2_idx = rouletteWheelSelection(); 

        if (parent1_idx < 0 || parent2_idx < 0) { 
            if (population_size_ > 0 && offspring_idx < population_size_) {
                Individual& filler = next_population_[offspring_idx++];
                filler = current_population_[getRandomInt(0, population_size_ - 1)];
                performMutation(filler);
                filler.parent1_idx = -3; filler.parent2_idx = -3;
            } else { break; } 
            continue;
        }
        Individual child1(chromosome_size_); Individual child2(chromosome_size_); 
        performCrossover(current_population_[parent1_idx], current_population_[parent2_idx], child1, child2);
        performMutation(child1);
        child1.parent1_idx = parent1_idx; child1.parent2_idx = parent2_idx;
        if (offspring_idx < population_size_) next_population_[offspring_idx++] = child1; else break;
        if (offspring_idx < population_size_) {
            performMutation(child2);
            child2.parent1_idx = parent1_idx; child2.parent2_idx = parent2_idx;
            next_population_[offspring_idx++] = child2;
        }
    }
}

void GeneticAlgorithm::applyElitism() {
    if (elitism_count_ == 0 || current_population_.empty() || population_size_ == 0) return;
    std::vector<Individual> sorted_pop = current_population_;
    std::sort(sorted_pop.rbegin(), sorted_pop.rend()); 
    for (int i = 0; i < elitism_count_ && static_cast<size_t>(i) < sorted_pop.size(); ++i) {
        if (static_cast<size_t>(i) < next_population_.size()) { 
            next_population_[i] = sorted_pop[i];
            next_population_[i].parent1_idx = -2; next_population_[i].parent2_idx = -2;
            next_population_[i].crossover_site1 = -1; next_population_[i].crossover_site2 = -1;
            next_population_[i].mutated_flag = false; next_population_[i].mutation_bit_idx = 0; 
        }
    }
}

void GeneticAlgorithm::printInitialParametersToConsole() const {
    std::cout << "========================================================" << std::endl;
    std::cout << "        PARAMETROS DEL ALGORITMO GENETICO (Consola)" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Tamano de la poblacion        : " << population_size_ << std::endl;
    std::cout << "Longitud del cromosoma (bits) : " << chromosome_size_ << std::endl;
    std::cout << "Maximo numero de generaciones : " << max_generations_ << std::endl;
    std::cout << "Probabilidad de cruce (Pc)    : " << std::fixed << std::setprecision(3) << prob_crossover_ << std::endl;
    std::cout << "Probabilidad de mutacion (Pm)   : " << prob_mutation_ << " (por bit)" << std::endl;
    std::cout << "Conteo de elitismo            : " << elitism_count_ << std::endl;
    std::cout << "Semilla RNG (entrada)         : " << initial_seed_value_ 
              << (initial_seed_value_ == 0 ? " (tiempo actual usado)" : "") << std::endl;
    std::cout << "========================================================" << std::endl << std::endl;
}

void GeneticAlgorithm::reportCurrentGenerationToConsole() {
    if (current_population_.empty() && current_generation_ == 0 && population_size_ == 0) {
        std::cout << "\nGeneracion # 0 (Poblacion no configurada o tamano cero)" << std::endl; return;
    }
    if (current_population_.empty()) {
        std::cout << "\nGeneracion #" << current_generation_ << " (Poblacion vacia - error inesperado)" << std::endl; return;
    }
    std::cout << "\n________________________________________________________________________________________________________________________\n";
    char title_buffer[200];
    snprintf(title_buffer, sizeof(title_buffer), "                                      GENERACION # %3d (Consola)", current_generation_);
    std::cout << title_buffer << std::endl;
    std::cout << "________________________________________________________________________________________________________________________\n";
    const int W_ID_CON = 4; 
    const int W_CHROM_CON_PLACEHOLDER = std::max(10, chromosome_size_) + 1; 
    const int W_XVAL_CON = 8; const int W_FIT_CON = 9; 
    const int W_ORIGIN_CON = 30;

    std::cout << std::left << std::setw(W_ID_CON) << "ID";
    // << std::setw(W_CHROM_CON_PLACEHOLDER) << "Cromosoma"; // <--- LÍNEA DE ENCABEZADO DEL CROMOSOMA DESACTIVADA
    std::cout << std::setw(W_CHROM_CON_PLACEHOLDER) << " "; // Espacio para la columna desactivada
    std::cout << std::setw(W_XVAL_CON) << "X-Val" << std::setw(W_FIT_CON) << "Fitness"
              << "| " << std::setw(W_ORIGIN_CON) << "Origen (P1,P2 XSite Mut?(Count))" << std::endl;
    // Ajustar la línea separadora si es necesario (opcional)
    std::cout << std::string(W_ID_CON + W_CHROM_CON_PLACEHOLDER + W_XVAL_CON + W_FIT_CON + 2 + W_ORIGIN_CON, '-') << std::endl;

    for (size_t i = 0; i < current_population_.size(); ++i) {
        const Individual& ind = current_population_[i];
        std::cout << std::left << std::setw(W_ID_CON) << i;
        // std::cout << std::setw(W_CHROM_CON_PLACEHOLDER) << getChromosomeString(ind.chrom); // <--- LÍNEA DE DATOS DEL CROMOSOMA DESACTIVADA
        std::cout << std::setw(W_CHROM_CON_PLACEHOLDER) << " "; // Espacio para la columna desactivada
        std::cout << std::fixed << std::setprecision(2) << std::setw(W_XVAL_CON) << ind.x_value;
        std::cout << std::fixed << std::setprecision(4) << std::setw(W_FIT_CON) << ind.fitness << "| ";
        
        char creation_info[60];
        std::string x_site_str = (ind.crossover_site1 != -1) ? std::to_string(ind.crossover_site1) : "---";

        if (ind.parent1_idx == -1) 
            snprintf(creation_info, sizeof(creation_info), "(Inicial)         %3s   N(%d)", x_site_str.c_str(), ind.mutation_bit_idx);
        else if (ind.parent1_idx == -2) 
            snprintf(creation_info, sizeof(creation_info), "(Elite)           %3s   N(%d)", x_site_str.c_str(), ind.mutation_bit_idx);
        else if (ind.parent1_idx == -3) 
            snprintf(creation_info, sizeof(creation_info), "(Relleno)         %3s   %c(%d)", x_site_str.c_str(), ind.mutated_flag ? 'Y' : 'N', ind.mutation_bit_idx);
        else 
            snprintf(creation_info, sizeof(creation_info), "(%3d,%3d) %3s   %c(%d)",
                     ind.parent1_idx, ind.parent2_idx, x_site_str.c_str(),
                     ind.mutated_flag ? 'Y' : 'N', ind.mutation_bit_idx);
        std::cout << std::setw(W_ORIGIN_CON) << std::left << creation_info;
        std::cout << std::endl;
    }
    std::cout << "________________________________________________________________________________________________________________________\n";
    std::cout << "Estadisticas Gen #" << current_generation_ << ": MinFit=" << std::fixed << std::setprecision(5) << min_fitness_current_gen_
              << " | MaxFit=" << max_fitness_current_gen_
              << " | AvgFit=" << avg_fitness_current_gen_ 
              << " | SumFit=" << sum_fitness_current_gen_ << std::endl;
    std::cout << "  Mejor esta Gen: ";
    if (!best_individual_current_gen_.chrom.empty() || chromosome_size_ == 0) {
        std::cout << getChromosomeString(best_individual_current_gen_.chrom); // <--- ESTE SE MANTIENE
        std::cout << " (Fit: " << best_individual_current_gen_.fitness << ")" << std::endl;
    } else { std::cout << "(N/A)" << std::endl; }
    std::cout << "  Mejor GLOBAL (Gen #" << generation_of_best_ever_ << "): ";
    if (!best_ever_individual_.chrom.empty() || chromosome_size_ == 0) {
        std::cout << getChromosomeString(best_ever_individual_.chrom); // <--- ESTE SE MANTIENE
        std::cout << " (Fit: " << best_ever_individual_.fitness << ")" << std::endl;
    } else { std::cout << "(N/A)" << std::endl; }
}

void GeneticAlgorithm::printChromosome(const std::vector<unsigned int>& chrom, std::ostream& os) const {
    os << getChromosomeString(chrom);
}

void GeneticAlgorithm::clearScreen() {
#if defined(_WIN32) || defined(_WIN64)
    std::system("cls");
#else
    std::system("clear");
#endif
}