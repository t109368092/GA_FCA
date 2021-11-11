import random
import copy
import numpy as np

#Objective: Maximize Throughput
class GATest:
    def __init__(self, pop_size, chromosomes, chromosomes_fitness, selected_chromosomes, selected_chromosomes_fitness,
                 crossovered_chromosomes, crossovered_chromosomes_fitness, best_chromosome, best_chromosome_fitness,
<<<<<<< HEAD
                 number_of_genes, packet_size, packet_count, signal_weight_lte, signal_weight_nr, packet_deadline,
                 lte_resource, nr_resource, running_tti):
=======
                 number_of_genes, packet_size, packet_count, signal_weight_lte, signal_weight_nr, lte_resource,
                 nr_resource, running_tti):
>>>>>>> 604e27163fa2ae213874ad4b33992676d375feb3
        self.pop_size = pop_size
        self.chromosomes = chromosomes
        self.chromosomes_fitness = chromosomes_fitness
        self.selected_chromosomes = selected_chromosomes
        self.selected_chromosomes_fitness = selected_chromosomes_fitness
        self.crossovered_chromosomes = crossovered_chromosomes
        self.crossovered_chromosomes_fitness = crossovered_chromosomes_fitness
        self.best_chromosome = best_chromosome
        self.best_chromosome_fitness = best_chromosome_fitness
        self.number_of_genes = number_of_genes
        self.packet_size = packet_size
        self.packet_count = packet_count
        self.signal_weight_lte = signal_weight_lte
        self.signal_weight_nr = signal_weight_nr
        self.packet_deadline = packet_deadline
        self.lte_resource = lte_resource
        self.nr_resource = nr_resource
        self.running_tti = running_tti

        self.tti_count = 0

    def main(self):
        self.chromosomes_generate()
        print("All Chromosomes: {}".format(self.chromosomes))

        self.constant_chromosome = []
        self.lte_chromosome = []
        self.constant_chromosome.append([])
        self.lte_chromosome.append([])
        for _ in range(self.number_of_genes):
            self.constant_chromosome[0].append(5)
            self.lte_chromosome[0].append(0)

        self.flows_generate()

        self.chromosomes_fitness = self.fitness_calculate(self.chromosomes)
        print("Fitness Value: {}\n".format(self.chromosomes_fitness))

        self.best_chromosome = self.chromosomes[self.chromosomes_fitness.index(max(self.chromosomes_fitness))]
        self.best_chromosome_fitness = max(self.chromosomes_fitness)
        print("Best Chromosome: {}".format(self.best_chromosome))
        print("Best Chromosome Fitness Value: {}\n".format(self.best_chromosome_fitness))

        self.tti_count = self.pop_size * 10
        best_chromosomes_fitness = []
        while(self.tti_count < self.running_tti):
            self.select()
            print("Selected Chromosomes: {}".format(self.selected_chromosomes))

            self.selected_chromosomes_fitness = self.fitness_calculate(self.selected_chromosomes)
            print("Selected Chromosomes Fitness Value: {}\n".format(self.selected_chromosomes_fitness))

            self.crossover(self.selected_chromosomes)
            print("Crossovered Chromosomes: {}".format(self.crossovered_chromosomes))

            self.crossovered_chromosomes_fitness = self.fitness_calculate(self.crossovered_chromosomes)
            print("Crossovered Chromosomes Fitness Value: {}\n".format(self.crossovered_chromosomes_fitness))

            self.replace()
            print("New All Chromosomes: {}".format(self.chromosomes))
            print("New Fitness Value: {}\n".format(self.chromosomes_fitness))

            self.best_chromosome = self.chromosomes[self.chromosomes_fitness.index(max(self.chromosomes_fitness))]
            self.best_chromosome_fitness = max(self.chromosomes_fitness)
            print("Best Chromosome: {}".format(self.best_chromosome))
            print("Best Chromosome Fitness Value: {}\n".format(self.best_chromosome_fitness))

            best_chromosomes_fitness.append(self.best_chromosome_fitness)
            print("Best Chromosomes Fitness Value: {}\n".format(best_chromosomes_fitness))

            self.tti_count = self.tti_count + len(self.crossovered_chromosomes) * 10

        print("LTE signal weight:")
        print(self.signal_weight_lte)
        print("NR signal weight:")
        print(self.signal_weight_nr)

        self.constant_chromosome_fitness = self.fitness_calculate(self.constant_chromosome)
        print("Constant flow control ratio fitness:")
        print(self.constant_chromosome_fitness)
        self.lte_chromosome_fitness = self.fitness_calculate(self.lte_chromosome)
        print("Only LTE flow  ratio fitness:")
        print(self.lte_chromosome_fitness)

    def chromosomes_generate(self):
        for _ in range(self.pop_size):
            chromosome = []

            for _ in range(self.number_of_genes):
                rand_ratio = random.randint(1, 10)
                chromosome.append(rand_ratio)
            
            self.chromosomes.append(chromosome)

    def flows_generate(self):
        for _ in range(self.number_of_genes):
            rand_packet_size = random.randint(10, 20)
            self.packet_size.append(rand_packet_size)

        for _ in range(self.number_of_genes):
            rand_packet_count = random.randint(10, 20)
            self.packet_count.append(rand_packet_count)

        for _ in range(self.number_of_genes):
            rand_signal_weight_lte = random.randint(1, 5)
            self.signal_weight_lte.append(rand_signal_weight_lte)

        for _ in range(self.number_of_genes):
            rand_signal_weight_nr = random.randint(1, 5)
            self.signal_weight_nr.append(rand_signal_weight_nr)

        for _ in range(self.number_of_genes):
            rand_packet_deadline = random.randint(2, 5)
            self.packet_deadline.append(rand_packet_deadline)

    def fitness_calculate(self, chromosomes):
        chromosomes_count = len(chromosomes)
        nr_ratio = np.array(chromosomes)

        fitness = []
        for i in range(chromosomes_count):
            tti = 0
            lte_buffer = []
            nr_buffer = []
            ue_avg_lte = []
            ue_avg_nr = []
            for _ in range(self.number_of_genes):
                lte_buffer.append([])
                nr_buffer.append([])
                ue_avg_lte.append(0.0)
                ue_avg_nr.append(0.0)
            ue_max_rate_lte = copy.deepcopy(self.signal_weight_lte)
            ue_max_rate_nr = copy.deepcopy(self.signal_weight_nr)
            ue_r_R_lte = copy.deepcopy(ue_max_rate_lte)
            ue_r_R_nr = copy.deepcopy(ue_max_rate_nr)

            packets_to_nr = np.array(self.packet_count) * nr_ratio[i] / 10
            packets_to_nr = np.asarray(packets_to_nr, dtype = int)
            packets_to_lte = np.array(self.packet_count) - packets_to_nr
            packets_to_nr = packets_to_nr.tolist()
            packets_to_lte = packets_to_lte.tolist()

            for tti in range(10):
                resource_request_lte = []
                resource_request_nr = []
                ue_priority_lte = []
                ue_priority_nr = []

                for ue in range(self.number_of_genes):
                    lte_buffer[ue].append(packets_to_lte[ue])
                    nr_buffer[ue].append(packets_to_nr[ue])

                    resource_request_lte.append(sum(lte_buffer[ue]) * self.packet_size[ue] / self.signal_weight_lte[ue])
                    resource_request_nr.append(sum(nr_buffer[ue]) * self.packet_size[ue] / self.signal_weight_nr[ue])

                    if ue_avg_lte[ue] != 0.0:
                        ue_r_R_lte[ue] = float(ue_max_rate_lte[ue]) / ue_avg_lte[ue]
                    if ue_avg_nr[ue] != 0.0:
                        ue_r_R_nr[ue] = float(ue_max_rate_nr[ue]) / ue_avg_nr[ue]

<<<<<<< HEAD
=======
                    if ue_avg_nr[m] != 0.0:
                        ue_r_R_nr[m] = float(ue_max_rate_nr[m]) / ue_avg_nr[m]
>>>>>>> 604e27163fa2ae213874ad4b33992676d375feb3
                ue_r_R_lte_temp = copy.deepcopy(ue_r_R_lte)
                ue_r_R_nr_temp = copy.deepcopy(ue_r_R_nr)

                for _ in range(self.number_of_genes):
                    ue_priority_lte.append(ue_r_R_lte_temp.index(max(ue_r_R_lte_temp)))
                    ue_r_R_lte_temp[ue_r_R_lte_temp.index(max(ue_r_R_lte_temp))] = 0
                    ue_priority_nr.append(ue_r_R_nr_temp.index(max(ue_r_R_nr_temp)))
                    ue_r_R_nr_temp[ue_r_R_nr_temp.index(max(ue_r_R_nr_temp))] = 0

                remain_resource_lte = self.lte_resource
                remain_resource_nr = self.nr_resource
                allocate_lte_resource_ue_index = 0
                allocate_nr_resource_ue_index = 0
                ue_avg_lte = np.array(ue_avg_lte) * tti
                ue_avg_nr = np.array(ue_avg_nr) * tti
                while(remain_resource_lte > 0 and allocate_lte_resource_ue_index < self.number_of_genes):
                    if resource_request_lte[ue_priority_lte[allocate_lte_resource_ue_index]] <= remain_resource_lte:
                        ue_avg_lte[ue_priority_lte[allocate_lte_resource_ue_index]] = ue_avg_lte[ue_priority_lte[allocate_lte_resource_ue_index]] + sum(lte_buffer[ue_priority_lte[allocate_lte_resource_ue_index]]) * self.packet_size[ue_priority_lte[allocate_lte_resource_ue_index]]
                        remain_resource_lte = remain_resource_lte - resource_request_lte[ue_priority_lte[allocate_lte_resource_ue_index]]
                        lte_buffer[ue_priority_lte[allocate_lte_resource_ue_index]] = []
                    allocate_lte_resource_ue_index = allocate_lte_resource_ue_index + 1
                
                while(remain_resource_nr > 0 and allocate_nr_resource_ue_index < self.number_of_genes):
                    if resource_request_nr[ue_priority_nr[allocate_nr_resource_ue_index]] <= remain_resource_nr:
                        ue_avg_nr[ue_priority_nr[allocate_nr_resource_ue_index]] = ue_avg_nr[ue_priority_nr[allocate_nr_resource_ue_index]] + sum(nr_buffer[ue_priority_nr[allocate_nr_resource_ue_index]]) * self.packet_size[ue_priority_nr[allocate_nr_resource_ue_index]]
                        remain_resource_nr = remain_resource_nr - resource_request_nr[ue_priority_nr[allocate_nr_resource_ue_index]]
                        nr_buffer[ue_priority_nr[allocate_nr_resource_ue_index]] = []
                    allocate_nr_resource_ue_index = allocate_nr_resource_ue_index + 1

                for ue in range(self.number_of_genes):
                    while(len(lte_buffer[ue]) > self.packet_deadline[ue]):
                        del lte_buffer[ue][0]
                    while(len(nr_buffer[ue]) > self.packet_deadline[ue]):
                        del nr_buffer[ue][0]

                ue_avg_lte = ue_avg_lte / (tti + 1)
                ue_avg_nr = ue_avg_nr / (tti + 1)
                ue_avg_lte = ue_avg_lte.tolist()
                ue_avg_nr = ue_avg_nr.tolist()
                
            avg_throughput_lte = sum(ue_avg_lte) / self.number_of_genes
            avg_throughput_nr = sum(ue_avg_nr) / self.number_of_genes
            throughput_sum = avg_throughput_lte + avg_throughput_nr
            fitness.append(throughput_sum)

        return fitness

    def select(self):
        fitness_temp = copy.deepcopy(self.chromosomes_fitness)
        fitness_temp = np.asarray(fitness_temp, dtype = int)
        fitness_temp = fitness_temp.tolist()
        chromosomes_temp = copy.deepcopy(self.chromosomes)
        self.selected_chromosomes = []
        
        for _ in range(2):
            choice = random.choices(chromosomes_temp, weights = fitness_temp)[0]
            self.selected_chromosomes.append(choice)
            choice_index = chromosomes_temp.index(choice)
            del chromosomes_temp[choice_index]
            del fitness_temp[choice_index]

    def crossover(self, selected_chromosomes):
        self.crossovered_chromosomes = []
        crossover_index = random.randint(1, 9)

        crossovered_chromosome_A = selected_chromosomes[0][:crossover_index] + selected_chromosomes[1][crossover_index:]
        crossovered_chromosome_B = selected_chromosomes[1][:crossover_index] + selected_chromosomes[0][crossover_index:]

        self.crossovered_chromosomes = [crossovered_chromosome_A, crossovered_chromosome_B]

        for i in range(2):
            self.crossovered_chromosomes[i][random.randint(0, self.number_of_genes - 1)] = random.randint(1, 10)

    def replace(self):
        chromosomes_temp = self.chromosomes + self.crossovered_chromosomes
        chromosomes_fitness_temp = self.chromosomes_fitness + self.crossovered_chromosomes_fitness

        self.chromosomes = []
        self.chromosomes_fitness = []
        for _ in range(self.pop_size):
            max_index = chromosomes_fitness_temp.index(max(chromosomes_fitness_temp))
            self.chromosomes.append(chromosomes_temp[max_index])
            self.chromosomes_fitness.append(chromosomes_fitness_temp[max_index])

            del chromosomes_temp[max_index]
            del chromosomes_fitness_temp[max_index]
        

pop_size = 10
chromosomes = []
chromosomes_fitness = []
selected_chromosomes = []
selected_chromosomes_fitness = []
crossovered_chromosomes = []
crossovered_chromosomes_fitness = []
best_chromosome = []
best_chromosome_fitness = 0
number_of_genes = 10
packet_size = []
packet_count = []
signal_weight_lte = []
signal_weight_nr = []
packet_deadline = []
lte_resource = 100
nr_resource = 120
<<<<<<< HEAD
running_tti = 500
GATest(pop_size, chromosomes, chromosomes_fitness, selected_chromosomes, selected_chromosomes_fitness,
       crossovered_chromosomes, crossovered_chromosomes_fitness, best_chromosome, best_chromosome_fitness,
       number_of_genes, packet_size, packet_count, signal_weight_lte, signal_weight_nr, packet_deadline,
       lte_resource, nr_resource, running_tti).main()
=======
running_tti = 600
GATest(pop_size, chromosomes, chromosomes_fitness, selected_chromosomes, selected_chromosomes_fitness,
       crossovered_chromosomes, crossovered_chromosomes_fitness, best_chromosome, best_chromosome_fitness,
       number_of_genes, packet_size, packet_count, signal_weight_lte, signal_weight_nr, lte_resource,
       nr_resource, running_tti).main()
>>>>>>> 604e27163fa2ae213874ad4b33992676d375feb3
