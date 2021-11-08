from math import prod
import random
import copy
import numpy as np

#Objective: Maximize Throughput Gain Fairness
class GATest:
    def __init__(self, pop_size, chromosomes, chromosomes_fitness,
                 crossovered_chromosomes, crossovered_chromosomes_fitness,
                 best_chromosome, best_chromosome_fitness, number_of_genes,
                 packet_size, packet_count, signal_weight_lte, signal_weight_nr,
                 lte_resource, nr_resource, running_tti):
        self.pop_size = pop_size
        self.chromosomes = chromosomes
        self.chromosomes_fitness = chromosomes_fitness
        self.crossovered_chromosomes = crossovered_chromosomes
        self.crossovered_chromosomes_fitness = crossovered_chromosomes_fitness
        self.best_chromosome = best_chromosome
        self.best_chromosome_fitness = best_chromosome_fitness
        self.number_of_genes = number_of_genes
        self.packet_size = packet_size
        self.packet_count = packet_count
        self.signal_weight_lte = signal_weight_lte
        self.signal_weight_nr = signal_weight_nr
        self.lte_resource = lte_resource
        self.nr_resource = nr_resource
        self.running_tti = running_tti

        self.tti_count = 0

    def main(self):
        self.chromosomes_generate()
        print("All Chromosomes: {}".format(self.chromosomes))

        self.flows_generate()

        self.lte_chromosome = []
        self.lte_chromosome.append([])
        for _ in range(self.number_of_genes):
            self.lte_chromosome[0].append(0)
        self.lte_chromosome_rate = self.lte_rate_calculate(self.lte_chromosome)

        self.chromosomes_fitness = self.fitness_calculate(self.chromosomes)
        print("Fitness Value: {}\n".format(self.chromosomes_fitness))

        self.best_chromosome = self.chromosomes[self.chromosomes_fitness.index(max(self.chromosomes_fitness))]
        self.best_chromosome_fitness = max(self.chromosomes_fitness)
        print("Best Chromosome: {}".format(self.best_chromosome))
        print("Best Chromosome Fitness Value: {}\n".format(self.best_chromosome_fitness))

        self.tti_count = (1 + pop_size) * 10
        best_chromosomes_fitness = []
        while(self.tti_count < self.running_tti):
            self.crossover(self.chromosomes)
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

    def fitness_calculate(self, chromosomes):
        chromosomes_count = len(chromosomes)
        nr_ratio = np.array(chromosomes)

        fitness = []
        for i in range(chromosomes_count):
            tti = 0
            ue_avg_lte = []
            for _ in range(self.number_of_genes):
                ue_avg_lte.append(0.0)
            ue_avg_nr = []
            for _ in range(self.number_of_genes):
                ue_avg_nr.append(0.0)
            ue_max_rate_lte = self.signal_weight_lte.copy()
            ue_max_rate_nr = self.signal_weight_nr.copy()
            ue_r_R_lte = ue_max_rate_lte.copy()
            ue_r_R_nr = ue_max_rate_nr.copy()

            packets_to_nr = np.array(self.packet_count) * nr_ratio[i] / 10
            packets_to_nr = np.asarray(packets_to_nr, dtype = int)
            packets_to_lte = np.array(self.packet_count) - packets_to_nr

            resource_request_lte = packets_to_lte * np.array(self.packet_size) / np.array(self.signal_weight_lte)
            resource_request_nr = packets_to_nr * np.array(self.packet_size) / np.array(self.signal_weight_nr)
            resource_request_lte = np.asarray(resource_request_lte, dtype = int)
            resource_request_nr = np.asarray(resource_request_nr, dtype = int)
            resource_request_lte = resource_request_lte.tolist()
            resource_request_nr = resource_request_nr.tolist()

            for tti in range(10):
                ue_priority_lte = []
                ue_priority_nr = []

                for m in range(self.number_of_genes):
                    if ue_avg_lte[m] != 0.0:
                        ue_r_R_lte[m] = float(ue_max_rate_lte[m]) / ue_avg_lte[m]

                    if ue_avg_nr[m] != 0.0:
                        ue_r_R_nr[m] = float(ue_max_rate_nr[m]) / ue_avg_nr[m]
                ue_r_R_lte_temp = ue_r_R_lte.copy()
                ue_r_R_nr_temp = ue_r_R_nr.copy()

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
                        ue_avg_lte[ue_priority_lte[allocate_lte_resource_ue_index]] = ue_avg_lte[ue_priority_lte[allocate_lte_resource_ue_index]] + packets_to_lte[ue_priority_lte[allocate_lte_resource_ue_index]] * self.packet_size[ue_priority_lte[allocate_lte_resource_ue_index]]
                    remain_resource_lte = remain_resource_lte - resource_request_lte[ue_priority_lte[allocate_lte_resource_ue_index]]
                    allocate_lte_resource_ue_index = allocate_lte_resource_ue_index + 1
                
                while(remain_resource_nr > 0 and allocate_nr_resource_ue_index < self.number_of_genes):
                    if resource_request_nr[ue_priority_nr[allocate_nr_resource_ue_index]] <= remain_resource_nr:
                        ue_avg_nr[ue_priority_nr[allocate_nr_resource_ue_index]] = ue_avg_nr[ue_priority_nr[allocate_nr_resource_ue_index]] + packets_to_nr[ue_priority_nr[allocate_nr_resource_ue_index]] * self.packet_size[ue_priority_nr[allocate_nr_resource_ue_index]]
                    remain_resource_nr = remain_resource_nr - resource_request_nr[ue_priority_nr[allocate_nr_resource_ue_index]]
                    allocate_nr_resource_ue_index = allocate_nr_resource_ue_index + 1

                ue_avg_lte = ue_avg_lte / (tti + 1)
                ue_avg_nr = ue_avg_nr / (tti + 1)
                ue_avg_lte = ue_avg_lte.tolist()
                ue_avg_nr = ue_avg_nr.tolist()
                
            throughput_gain = prod(np.array(ue_avg_lte) + np.array(ue_avg_nr) - np.array(self.lte_chromosome_rate))
            fitness.append(throughput_gain)

        return fitness

    def lte_rate_calculate(self, chromosomes):
        chromosomes_count = len(chromosomes)
        nr_ratio = np.array(chromosomes)

        fitness = []
        for i in range(chromosomes_count):
            tti = 0
            ue_avg_lte = []
            for _ in range(self.number_of_genes):
                ue_avg_lte.append(0.0)
            ue_avg_nr = []
            for _ in range(self.number_of_genes):
                ue_avg_nr.append(0.0)
            ue_max_rate_lte = self.signal_weight_lte.copy()
            ue_max_rate_nr = self.signal_weight_nr.copy()
            ue_r_R_lte = ue_max_rate_lte.copy()
            ue_r_R_nr = ue_max_rate_nr.copy()

            packets_to_nr = np.array(self.packet_count) * nr_ratio[i] / 10
            packets_to_nr = np.asarray(packets_to_nr, dtype = int)
            packets_to_lte = np.array(self.packet_count) - packets_to_nr

            resource_request_lte = packets_to_lte * np.array(self.packet_size) / np.array(self.signal_weight_lte)
            resource_request_nr = packets_to_nr * np.array(self.packet_size) / np.array(self.signal_weight_nr)
            resource_request_lte = np.asarray(resource_request_lte, dtype = int)
            resource_request_nr = np.asarray(resource_request_nr, dtype = int)
            resource_request_lte = resource_request_lte.tolist()
            resource_request_nr = resource_request_nr.tolist()

            for tti in range(10):
                ue_priority_lte = []
                ue_priority_nr = []

                for m in range(self.number_of_genes):
                    if ue_avg_lte[m] != 0.0:
                        ue_r_R_lte[m] = float(ue_max_rate_lte[m]) / ue_avg_lte[m]

                    if ue_avg_nr[m] != 0.0:
                        ue_r_R_nr[m] = float(ue_max_rate_nr[m]) / ue_avg_nr[m]
                ue_r_R_lte_temp = ue_r_R_lte.copy()
                ue_r_R_nr_temp = ue_r_R_nr.copy()

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
                        ue_avg_lte[ue_priority_lte[allocate_lte_resource_ue_index]] = ue_avg_lte[ue_priority_lte[allocate_lte_resource_ue_index]] + packets_to_lte[ue_priority_lte[allocate_lte_resource_ue_index]] * self.packet_size[ue_priority_lte[allocate_lte_resource_ue_index]]
                    remain_resource_lte = remain_resource_lte - resource_request_lte[ue_priority_lte[allocate_lte_resource_ue_index]]
                    allocate_lte_resource_ue_index = allocate_lte_resource_ue_index + 1
                
                while(remain_resource_nr > 0 and allocate_nr_resource_ue_index < self.number_of_genes):
                    if resource_request_nr[ue_priority_nr[allocate_nr_resource_ue_index]] <= remain_resource_nr:
                        ue_avg_nr[ue_priority_nr[allocate_nr_resource_ue_index]] = ue_avg_nr[ue_priority_nr[allocate_nr_resource_ue_index]] + packets_to_nr[ue_priority_nr[allocate_nr_resource_ue_index]] * self.packet_size[ue_priority_nr[allocate_nr_resource_ue_index]]
                    remain_resource_nr = remain_resource_nr - resource_request_nr[ue_priority_nr[allocate_nr_resource_ue_index]]
                    allocate_nr_resource_ue_index = allocate_nr_resource_ue_index + 1

                ue_avg_lte = ue_avg_lte / (tti + 1)
                ue_avg_nr = ue_avg_nr / (tti + 1)
                ue_avg_lte = ue_avg_lte.tolist()
                ue_avg_nr = ue_avg_nr.tolist()

        return ue_avg_lte

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
crossovered_chromosomes = []
crossovered_chromosomes_fitness = []
best_chromosome = []
best_chromosome_fitness = 0
number_of_genes = 10
packet_size = []
packet_count = []
signal_weight_lte = []
signal_weight_nr = []
lte_resource = 100
nr_resource = 120
running_tti = 310
GATest(pop_size, chromosomes, chromosomes_fitness, crossovered_chromosomes, crossovered_chromosomes_fitness, 
       best_chromosome, best_chromosome_fitness, number_of_genes, packet_size, packet_count, signal_weight_lte,
       signal_weight_nr, lte_resource, nr_resource, running_tti).main()