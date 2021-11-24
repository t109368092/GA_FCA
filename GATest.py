import random
import copy
import csv
import numpy as np
from math import prod

class GATest:
    def __init__(self, fitness_calculate_method, number_of_genes, pop_size, lte_resource, nr_resource, tti_per_chromosome, running_tti,
                 packet_size_range, packet_count_range, signal_weight_lte_range, signal_weight_nr_range,packet_deadline_range):
        self.fitness_calculate_method = fitness_calculate_method
        if self.fitness_calculate_method == 1:
            self.fitness_calculate = self.fitness_calculate_max_throughput
        elif self.fitness_calculate_method == 2:
            self.fitness_calculate = self.fitness_calculate_fair_throughput
        elif self.fitness_calculate_method == 3:
            self.fitness_calculate = self.fitness_calculate＿delay_loss
       
        self.number_of_genes = number_of_genes
        self.pop_size = pop_size
        self.lte_resource = lte_resource
        self.nr_resource = nr_resource
        self.tti_per_chromosome = tti_per_chromosome
        self.running_tti = running_tti

        self.packet_size_range = packet_size_range
        self.packet_count_range = packet_count_range
        self.signal_weight_lte_range = signal_weight_lte_range
        self.signal_weight_nr_range = signal_weight_nr_range
        self.packet_deadline_range = packet_deadline_range

        self.chromosomes = []
        self.chromosomes_fitness = []
        self.selected_chromosomes = []
        self.selected_chromosomes_fitness = []
        self.crossovered_chromosomes = []
        self.crossovered_chromosomes_fitness = []
        self.best_chromosome = []
        self.packet_size = []
        self.packet_count = []
        self.signal_weight_lte = []
        self.signal_weight_nr = []
        self.packet_deadline = []
        self.write_row = []

        self.tti_count = 0

    def main(self):
        self.chromosomes_generate()
        print("All Chromosomes: {}".format(self.chromosomes))

        seventy_chromosome = [[]]
        fifty_chromosome = [[]]
        thirty_chromosome = [[]]
        lte_chromosome = [[]]
        for _ in range(self.number_of_genes):
            seventy_chromosome[0].append(7)
            fifty_chromosome[0].append(5)
            thirty_chromosome[0].append(3)
            lte_chromosome[0].append(0)

        self.flows_generate()

        self.chromosomes_fitness = self.fitness_calculate(self.chromosomes, False)
        print("Fitness Value: {}\n".format(self.chromosomes_fitness))

        self.best_chromosome = self.chromosomes[self.chromosomes_fitness.index(max(self.chromosomes_fitness))]
        self.best_chromosome_fitness = max(self.chromosomes_fitness)
        print("Best Chromosome: {}".format(self.best_chromosome))
        print("Best Chromosome Fitness Value: {}\n".format(self.best_chromosome_fitness))

        self.tti_count = self.pop_size * self.tti_per_chromosome
        best_chromosomes_fitness = []
        while self.tti_count < self.running_tti:
            self.select()
            print("Selected Chromosomes: {}".format(self.selected_chromosomes))

            self.selected_chromosomes_fitness = self.fitness_calculate(self.selected_chromosomes, False)
            print("Selected Chromosomes Fitness Value: {}\n".format(self.selected_chromosomes_fitness))

            self.crossover(self.selected_chromosomes)
            print("Crossovered Chromosomes: {}".format(self.crossovered_chromosomes))

            self.crossovered_chromosomes_fitness = self.fitness_calculate(self.crossovered_chromosomes, False)
            print("Crossovered Chromosomes Fitness Value: {}\n".format(self.crossovered_chromosomes_fitness))

            self.replace()
            print("New All Chromosomes: {}".format(self.chromosomes))
            print("New Fitness Value: {}\n".format(self.chromosomes_fitness))

            self.best_chromosome = [self.chromosomes[self.chromosomes_fitness.index(max(self.chromosomes_fitness))]]
            self.best_chromosome_fitness = max(self.chromosomes_fitness)
            print("Best Chromosome: {}".format(self.best_chromosome))
            print("Best Chromosome Fitness Value: {}\n".format(self.best_chromosome_fitness))

            best_chromosomes_fitness.append(self.best_chromosome_fitness)
            print("Best Chromosomes Fitness Value: {}\n".format(best_chromosomes_fitness))

            self.tti_count = self.tti_count + len(self.crossovered_chromosomes) * tti_per_chromosome

        self.best_chromosome_fitness = self.fitness_calculate(self.best_chromosome, True)
        if self.fitness_calculate_method == 1:
            seventy_chromosome_fitness = self.fitness_calculate(seventy_chromosome, True)
            print("70% NR flow control ratio fitness:")
            print(seventy_chromosome_fitness)
        fifty_chromosome_fitness = self.fitness_calculate(fifty_chromosome, True)
        print("50% NR flow control ratio fitness:")
        print(fifty_chromosome_fitness)
        if self.fitness_calculate_method == 1:
            thirty_chromosome_fitness = self.fitness_calculate(thirty_chromosome, True)
            print("30% NR flow control ratio fitness:")
            print(thirty_chromosome_fitness)
        lte_chromosome_fitness = self.fitness_calculate(lte_chromosome, True)
        print("Only LTE flow  ratio fitness:")
        print(lte_chromosome_fitness)

        w.writerow(self.write_row)

    def chromosomes_generate(self):
        for _ in range(self.pop_size):
            chromosome = []
            for _ in range(self.number_of_genes):
                rand_ratio = random.randint(1, 10)
                chromosome.append(rand_ratio)
            
            self.chromosomes.append(chromosome)

    def flows_generate(self):
        for _ in range(self.number_of_genes):
            rand_packet_size = random.uniform(self.packet_size_range[0], self.packet_size_range[1])
            self.packet_size.append(rand_packet_size)

            rand_packet_count = random.randint(self.packet_count_range[0], self.packet_count_range[1])
            self.packet_count.append(rand_packet_count)

            rand_signal_weight_lte = random.uniform(self.signal_weight_lte_range[0], self.signal_weight_lte_range[1])
            self.signal_weight_lte.append(rand_signal_weight_lte)

            rand_signal_weight_nr = random.uniform(self.signal_weight_nr_range[0], self.signal_weight_nr_range[1])
            self.signal_weight_nr.append(rand_signal_weight_nr)

            rand_packet_deadline = random.randint(self.packet_deadline_range[0], self.packet_deadline_range[1])
            self.packet_deadline.append(rand_packet_deadline)

    def fitness_calculate_max_throughput(self, chromosomes, write_data):
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

            for tti in range(tti_per_chromosome):
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
                while remain_resource_lte > 0 and allocate_lte_resource_ue_index < self.number_of_genes:
                    current_ue_lte = ue_priority_lte[allocate_lte_resource_ue_index]
                    if resource_request_lte[current_ue_lte] <= remain_resource_lte:
                        ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + sum(lte_buffer[current_ue_lte]) * self.packet_size[current_ue_lte]
                        remain_resource_lte = remain_resource_lte - resource_request_lte[current_ue_lte]
                        lte_buffer[current_ue_lte] = []

                    if resource_request_lte[current_ue_lte] > remain_resource_lte:
                        allocatable_packets = remain_resource_lte * self.signal_weight_lte[current_ue_lte] / self.packet_size[current_ue_lte]
                        while len(lte_buffer[current_ue_lte]) > 0 and allocatable_packets > lte_buffer[current_ue_lte][0]:
                            allocatable_packets = allocatable_packets - lte_buffer[current_ue_lte][0]
                            ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + lte_buffer[current_ue_lte][0] * self.packet_size[current_ue_lte]
                            del lte_buffer[current_ue_lte][0]
                        remain_resource_lte = 0

                    allocate_lte_resource_ue_index = allocate_lte_resource_ue_index + 1
                
                while remain_resource_nr > 0 and allocate_nr_resource_ue_index < self.number_of_genes:
                    current_ue_nr = ue_priority_nr[allocate_nr_resource_ue_index]
                    if resource_request_nr[current_ue_nr] <= remain_resource_nr:
                        ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + sum(nr_buffer[current_ue_nr]) * self.packet_size[current_ue_nr]
                        remain_resource_nr = remain_resource_nr - resource_request_nr[current_ue_nr]
                        nr_buffer[current_ue_nr] = []

                    if resource_request_nr[current_ue_nr] > remain_resource_nr:
                        allocatable_packets = remain_resource_nr * self.signal_weight_nr[current_ue_nr] / self.packet_size[current_ue_nr]
                        while len(nr_buffer[current_ue_nr]) > 0 and allocatable_packets > nr_buffer[current_ue_nr][0]:
                            allocatable_packets = allocatable_packets - nr_buffer[current_ue_nr][0]
                            ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + nr_buffer[current_ue_nr][0] * self.packet_size[current_ue_nr]
                            del nr_buffer[current_ue_nr][0]
                        remain_resource_nr = 0
                    
                    allocate_nr_resource_ue_index = allocate_nr_resource_ue_index + 1

                for ue in range(self.number_of_genes):
                    while len(lte_buffer[ue]) > self.packet_deadline[ue]:
                        del lte_buffer[ue][0]

                    while len(nr_buffer[ue]) > self.packet_deadline[ue]:
                        del nr_buffer[ue][0]

                ue_avg_lte = ue_avg_lte / (tti + 1)
                ue_avg_nr = ue_avg_nr / (tti + 1)
                ue_avg_lte = ue_avg_lte.tolist()
                ue_avg_nr = ue_avg_nr.tolist()
                
            avg_throughput_lte = sum(ue_avg_lte) / self.number_of_genes
            avg_throughput_nr = sum(ue_avg_nr) / self.number_of_genes
            avg_throughput = avg_throughput_lte + avg_throughput_nr
            fitness.append(avg_throughput)

            if write_data == True:
                self.write_row.append(avg_throughput)

        return fitness

    def fitness_calculate_fair_throughput(self, chromosomes, write_data):
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

            for tti in range(20):
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
                while remain_resource_lte > 0 and allocate_lte_resource_ue_index < self.number_of_genes:
                    current_ue_lte = ue_priority_lte[allocate_lte_resource_ue_index]
                    if resource_request_lte[current_ue_lte] <= remain_resource_lte:
                        ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + sum(lte_buffer[current_ue_lte]) * self.packet_size[current_ue_lte]
                        remain_resource_lte = remain_resource_lte - resource_request_lte[current_ue_lte]
                        lte_buffer[current_ue_lte] = []

                    if resource_request_lte[current_ue_lte] > remain_resource_lte:
                        allocatable_packets = remain_resource_lte * self.signal_weight_lte[current_ue_lte] / self.packet_size[current_ue_lte]
                        while len(lte_buffer[current_ue_lte]) > 0 and allocatable_packets > lte_buffer[current_ue_lte][0]:
                            allocatable_packets = allocatable_packets - lte_buffer[current_ue_lte][0]
                            ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + lte_buffer[current_ue_lte][0] * self.packet_size[current_ue_lte]
                            del lte_buffer[current_ue_lte][0]
                        remain_resource_lte = 0

                    allocate_lte_resource_ue_index = allocate_lte_resource_ue_index + 1
                
                while remain_resource_nr > 0 and allocate_nr_resource_ue_index < self.number_of_genes:
                    current_ue_nr = ue_priority_nr[allocate_nr_resource_ue_index]
                    if resource_request_nr[current_ue_nr] <= remain_resource_nr:
                        ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + sum(nr_buffer[current_ue_nr]) * self.packet_size[current_ue_nr]
                        remain_resource_nr = remain_resource_nr - resource_request_nr[current_ue_nr]
                        nr_buffer[current_ue_nr] = []

                    if resource_request_nr[current_ue_nr] > remain_resource_nr:
                        allocatable_packets = remain_resource_nr * self.signal_weight_nr[current_ue_nr] / self.packet_size[current_ue_nr]
                        while len(nr_buffer[current_ue_nr]) > 0 and allocatable_packets > nr_buffer[current_ue_nr][0]:
                            allocatable_packets = allocatable_packets - nr_buffer[current_ue_nr][0]
                            ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + nr_buffer[current_ue_nr][0] * self.packet_size[current_ue_nr]
                            del nr_buffer[current_ue_nr][0]
                        remain_resource_nr = 0

                    allocate_nr_resource_ue_index = allocate_nr_resource_ue_index + 1

                for ue in range(self.number_of_genes):
                    while len(lte_buffer[ue]) > self.packet_deadline[ue]:
                        del lte_buffer[ue][0]
                        
                    while len(nr_buffer[ue]) > self.packet_deadline[ue]:
                        del nr_buffer[ue][0]

                ue_avg_lte = ue_avg_lte / (tti + 1)
                ue_avg_nr = ue_avg_nr / (tti + 1)
                ue_avg_lte = ue_avg_lte.tolist()
                ue_avg_nr = ue_avg_nr.tolist()
            
            avg_throughput = sum(np.array(ue_avg_lte) + np.array(ue_avg_nr)) / self.number_of_genes
            throughput_fair = prod(np.array(ue_avg_nr) + np.array(ue_avg_lte)) ** (1 / self.number_of_genes)
            fitness.append(throughput_fair)

            if write_data == True:
                self.write_row.append(avg_throughput)
                self.write_row.append(throughput_fair)

        return fitness

    def fitness_calculate＿delay_loss(self, chromosomes, write_data):
        chromosomes_count = len(chromosomes)
        nr_ratio = np.array(chromosomes)

        fitness = []
        for i in range(chromosomes_count):
            tti = 0
            lte_buffer = []
            nr_buffer = []
            ue_avg_lte = []
            ue_avg_nr = []
            ue_loss_lte = []
            ue_delay_lte = []
            ue_loss_nr = []
            ue_delay_nr = []
            for _ in range(self.number_of_genes):
                lte_buffer.append([])
                nr_buffer.append([])
                ue_avg_lte.append(0.0)
                ue_avg_nr.append(0.0)
                ue_loss_lte.append([0, 0])
                ue_delay_lte.append([0, 0])
                ue_loss_nr.append([0, 0])
                ue_delay_nr.append([0, 0])
            ue_max_rate_lte = copy.deepcopy(self.signal_weight_lte)
            ue_max_rate_nr = copy.deepcopy(self.signal_weight_nr)
            ue_r_R_lte = copy.deepcopy(ue_max_rate_lte)
            ue_r_R_nr = copy.deepcopy(ue_max_rate_nr)

            packets_to_nr = np.array(self.packet_count) * nr_ratio[i] / 10
            packets_to_nr = np.asarray(packets_to_nr, dtype = int)
            packets_to_lte = np.array(self.packet_count) - packets_to_nr
            packets_to_nr = packets_to_nr.tolist()
            packets_to_lte = packets_to_lte.tolist()

            for tti in range(20):
                resource_request_lte = []
                resource_request_nr = []
                ue_priority_lte = []
                ue_priority_nr = []

                for ue in range(self.number_of_genes):
                    lte_buffer[ue].append(packets_to_lte[ue])
                    nr_buffer[ue].append(packets_to_nr[ue])

                    resource_request_lte.append(sum(lte_buffer[ue]) * self.packet_size[ue] / self.signal_weight_lte[ue])
                    resource_request_nr.append(sum(nr_buffer[ue]) * self.packet_size[ue] / self.signal_weight_nr[ue])

                    ue_loss_lte[ue][1] = ue_loss_lte[ue][1] + packets_to_lte[ue]
                    ue_loss_nr[ue][1] = ue_loss_nr[ue][1] + packets_to_nr[ue]

                    if ue_avg_lte[ue] != 0.0:
                        ue_r_R_lte[ue] = float(ue_max_rate_lte[ue]) / ue_avg_lte[ue]
                    if ue_avg_nr[ue] != 0.0:
                        ue_r_R_nr[ue] = float(ue_max_rate_nr[ue]) / ue_avg_nr[ue]

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
                while remain_resource_lte > 0 and allocate_lte_resource_ue_index < self.number_of_genes:
                    current_ue_lte = ue_priority_lte[allocate_lte_resource_ue_index]
                    if resource_request_lte[current_ue_lte] <= remain_resource_lte:
                        for index in range(len(lte_buffer[current_ue_lte])):
                            ue_delay_lte[current_ue_lte][0] = ue_delay_lte[current_ue_lte][0] + lte_buffer[current_ue_lte][index] * (len(lte_buffer[current_ue_lte]) - 1 - index)
                        ue_delay_lte[current_ue_lte][1] = ue_delay_lte[current_ue_lte][1] + sum(lte_buffer[current_ue_lte])
                        ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + sum(lte_buffer[current_ue_lte]) * self.packet_size[current_ue_lte]
                        remain_resource_lte = remain_resource_lte - resource_request_lte[current_ue_lte]
                        lte_buffer[current_ue_lte] = []

                    if resource_request_lte[current_ue_lte] > remain_resource_lte:
                        allocatable_packets = remain_resource_lte * self.signal_weight_lte[current_ue_lte] / self.packet_size[current_ue_lte]
                        while len(lte_buffer[current_ue_lte]) > 0 and allocatable_packets > lte_buffer[current_ue_lte][0]:
                            allocatable_packets = allocatable_packets - lte_buffer[current_ue_lte][0]
                            ue_delay_lte[current_ue_lte][0] = ue_delay_lte[current_ue_lte][0] + lte_buffer[current_ue_lte][0] * (len(lte_buffer[current_ue_lte]) - 1)
                            ue_delay_lte[current_ue_lte][1] = ue_delay_lte[current_ue_lte][1] + lte_buffer[current_ue_lte][0]
                            ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + lte_buffer[current_ue_lte][0] * self.packet_size[current_ue_lte]
                            del lte_buffer[current_ue_lte][0]
                        remain_resource_lte = 0

                    allocate_lte_resource_ue_index = allocate_lte_resource_ue_index + 1
                
                while remain_resource_nr > 0 and allocate_nr_resource_ue_index < self.number_of_genes:
                    current_ue_nr = ue_priority_nr[allocate_nr_resource_ue_index]
                    if resource_request_nr[current_ue_nr] <= remain_resource_nr:
                        for index in range(len(nr_buffer[current_ue_nr])):
                            ue_delay_nr[current_ue_nr][0] = ue_delay_nr[current_ue_nr][0] + nr_buffer[current_ue_nr][index] * (len(nr_buffer[current_ue_nr]) - 1 - index)
                        ue_delay_nr[current_ue_nr][1] = ue_delay_nr[current_ue_nr][1] + sum(nr_buffer[current_ue_nr])
                        ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + sum(nr_buffer[current_ue_nr]) * self.packet_size[current_ue_nr]
                        remain_resource_nr = remain_resource_nr - resource_request_nr[current_ue_nr]
                        nr_buffer[current_ue_nr] = []

                    if resource_request_nr[current_ue_nr] > remain_resource_nr:
                        allocatable_packets = remain_resource_nr * self.signal_weight_nr[current_ue_nr] / self.packet_size[current_ue_nr]
                        while len(nr_buffer[current_ue_nr]) > 0 and allocatable_packets > nr_buffer[current_ue_nr][0]:
                            allocatable_packets = allocatable_packets - nr_buffer[current_ue_nr][0]
                            ue_delay_nr[current_ue_nr][0] = ue_delay_nr[current_ue_nr][0] + nr_buffer[current_ue_nr][0] * (len(nr_buffer[current_ue_nr]) - 1)
                            ue_delay_nr[current_ue_nr][1] = ue_delay_nr[current_ue_nr][1] + nr_buffer[current_ue_nr][0]
                            ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + nr_buffer[current_ue_nr][0] * self.packet_size[current_ue_nr]
                            del nr_buffer[current_ue_nr][0]
                        remain_resource_nr = 0
                    
                    allocate_nr_resource_ue_index = allocate_nr_resource_ue_index + 1

                for ue in range(self.number_of_genes):
                    while len(lte_buffer[ue]) > self.packet_deadline[ue]:
                        ue_loss_lte[ue][0] = ue_loss_lte[ue][0] + lte_buffer[ue][0]
                        del lte_buffer[ue][0]
                    
                    while len(nr_buffer[ue]) > self.packet_deadline[ue]:
                        ue_loss_nr[ue][0] = ue_loss_nr[ue][0] + packets_to_nr[ue]
                        del nr_buffer[ue][0]
                        
                ue_avg_lte = ue_avg_lte / (tti + 1)
                ue_avg_nr = ue_avg_nr / (tti + 1)
                ue_avg_lte = ue_avg_lte.tolist()
                ue_avg_nr = ue_avg_nr.tolist()

            avg_delay_lte_x, avg_delay_lte_y, avg_delay_nr_x, avg_delay_nr_y = 0, 0, 0, 0
            for ue in range(self.number_of_genes):
                avg_delay_lte_x = avg_delay_lte_x + ue_delay_lte[ue][0]
                avg_delay_lte_y = avg_delay_lte_y + ue_delay_lte[ue][1]
                avg_delay_nr_x = avg_delay_nr_x + ue_delay_nr[ue][0]
                avg_delay_nr_y = avg_delay_nr_y + ue_delay_nr[ue][1]
            avg_delay_x = avg_delay_lte_x + avg_delay_nr_x
            avg_delay_y = avg_delay_lte_y + avg_delay_nr_y
            avg_delay = avg_delay_x / avg_delay_y

            avg_loss_lte_x, avg_loss_lte_y, avg_loss_nr_x, avg_loss_nr_y = 0, 0, 0, 0
            for ue in range(self.number_of_genes):
                avg_loss_lte_x = avg_loss_lte_x + ue_loss_lte[ue][0]
                avg_loss_lte_y = avg_loss_lte_y + ue_loss_lte[ue][1]
                avg_loss_nr_x = avg_loss_nr_x + ue_loss_nr[ue][0]
                avg_loss_nr_y = avg_loss_nr_y + ue_loss_nr[ue][1]
            avg_loss_x = avg_loss_lte_x + avg_loss_nr_x
            avg_loss_y = avg_loss_lte_y + avg_loss_nr_y
            avg_loss = avg_loss_x / avg_loss_y

            avg_throughput = sum(np.array(ue_avg_lte) + np.array(ue_avg_nr)) / self.number_of_genes
            delay_loss_indicator = 1 / ((avg_delay * avg_loss) ** 1 / 2)
            fitness.append(delay_loss_indicator)
            

            if write_data == True:
                self.write_row.append(avg_throughput)
                self.write_row.append(avg_delay)
                self.write_row.append(avg_loss)

        return fitness

    def select(self):
        fitness_temp = copy.deepcopy(self.chromosomes_fitness)
        chromosomes_temp = copy.deepcopy(self.chromosomes)
        self.selected_chromosomes = []
        
        for _ in range(2):
            choice = random.choices(chromosomes_temp, weights = fitness_temp)[0]
            self.selected_chromosomes.append(choice)
            choice_index = chromosomes_temp.index(choice)
            del chromosomes_temp[choice_index]
            del fitness_temp[choice_index]

    def crossover_double(self, selected_chromosomes):
        crossover_index = []
        crossover_index.append(random.randint(1, number_of_genes - 3))
        crossover_index.append(random.randint(crossover_index[0], number_of_genes - 2))

        crossovered_chromosome_A = selected_chromosomes[0][:crossover_index[0]] + selected_chromosomes[1][crossover_index[0]:crossover_index[1]] + selected_chromosomes[0][crossover_index[1]:]
        crossovered_chromosome_B = selected_chromosomes[1][:crossover_index[0]] + selected_chromosomes[0][crossover_index[0]:crossover_index[1]] + selected_chromosomes[1][crossover_index[1]:]

        self.crossovered_chromosomes = [crossovered_chromosome_A, crossovered_chromosome_B]

        for i in range(2):
            self.crossovered_chromosomes[i][random.randint(0, self.number_of_genes - 1)] = random.randint(1, 10)

    def crossover(self, selected_chromosomes):
        self.crossovered_chromosomes = []
        crossover_index = random.randint(1, self.number_of_genes - 1)

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
        

fitness_calculate_method = 2
simulation_times = 10

pop_size = 10
number_of_genes = 10
lte_resource = 100
nr_resource = 120
tti_per_chromosome = 20
running_tti = 1000

packet_size_range = [5,10]
packet_count_range = [5,10]
signal_weight_lte_range = [1,3]
signal_weight_nr_range = [1,3]
packet_deadline_range = [2,5]

f = open('obj{}.csv'.format(fitness_calculate_method), 'w')
w = csv.writer(f)
if fitness_calculate_method == 1:
    w.writerow(["GA Throughput", "70% NR Throughput", "50% NR Throughput", "30% NR Throughput", "Only LTE Throughput"])
elif fitness_calculate_method == 2:
    w.writerow(["GA Throughput", "GA Throughput Fairness", "50% NR Throughput", "50% NR Throughput Fairness", "Only LTE Throughput", "Only LTE Throughput Fairness"])
elif fitness_calculate_method == 3:
    w.writerow(["GA Throughput", "GA Delay", "GA Loss", "50% NR Throughput", "50% NR Delay", "50% NR Loss", "Only LTE Throughput", "Only LTE Delay", "Only LTE Loss"])

for _ in range(simulation_times):
    ga_fca = GATest(fitness_calculate_method, pop_size, number_of_genes,lte_resource, nr_resource, tti_per_chromosome, running_tti,
                    packet_size_range, packet_count_range, signal_weight_lte_range, signal_weight_nr_range, packet_deadline_range)
    ga_fca.main()
f.close()