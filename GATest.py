import random
import copy
import csv
import numpy as np
from math import prod

class GATest:
    def __init__(self, fitness_calculate_method, crossover_method,number_of_genes, pop_size, lte_resource, nr_resource, tti_per_chromosome, running_tti,
                 packet_size_range, packet_count_range, video_packet_gap_time, signal_weight_lte_range, signal_weight_nr_range,packet_deadline_range):
        if crossover_method == 1:
            self.crossover = self.crossover_single
        elif crossover_method == 2:
            self.crossover = self.crossover_double
        self.fitness_calculate_method = fitness_calculate_method      
        self.number_of_genes = number_of_genes
        self.pop_size = pop_size
        self.lte_resource = lte_resource
        self.nr_resource = nr_resource
        self.tti_per_chromosome = tti_per_chromosome
        self.running_tti = running_tti

        self.packet_size_range = packet_size_range
        self.packet_count_range = packet_count_range
        self.video_packet_gap_time = video_packet_gap_time
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

        self.tti_count = 0

        self.data = []

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

            self.tti_count = self.tti_count + len(self.crossovered_chromosomes) * self.tti_per_chromosome

        self.best_chromosome_fitness = self.fitness_calculate(self.best_chromosome, True)
        seventy_chromosome_fitness = self.fitness_calculate(seventy_chromosome, True)
        print("70% NR flow control ratio fitness:")
        print(seventy_chromosome_fitness)
        fifty_chromosome_fitness = self.fitness_calculate(fifty_chromosome, True)
        print("50% NR flow control ratio fitness:")
        print(fifty_chromosome_fitness)
        thirty_chromosome_fitness = self.fitness_calculate(thirty_chromosome, True)
        print("30% NR flow control ratio fitness:")
        print(thirty_chromosome_fitness)
        lte_chromosome_fitness = self.fitness_calculate(lte_chromosome, True)
        print("Only LTE flow ratio fitness:")
        print(lte_chromosome_fitness)

        return self.data

    def stable_sigmoid(self, x):
        sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
        return sig

    def chromosomes_generate(self):
        for _ in range(self.pop_size):
            chromosome = []
            for _ in range(self.number_of_genes):
                rand_ratio = random.randint(1, 10)
                chromosome.append(rand_ratio)
            
            self.chromosomes.append(chromosome)

    def flows_generate(self):
        for _ in range(self.number_of_genes):
            rand_packet_size_cbr = random.uniform(self.packet_size_range["cbr"][0], self.packet_size_range["cbr"][1])
            rand_packet_size_video = random.uniform(self.packet_size_range["video"][0], self.packet_size_range["video"][1])
            self.packet_size.append({"cbr": rand_packet_size_cbr, "video": rand_packet_size_video})

            rand_packet_count_cbr = random.randint(self.packet_count_range["cbr"][0], self.packet_count_range["cbr"][1])
            rand_packet_count_video = random.randint(self.packet_count_range["video"][0], self.packet_count_range["video"][1])
            self.packet_count.append({"cbr": rand_packet_count_cbr, "video": rand_packet_count_video})

            rand_signal_weight_lte = random.uniform(self.signal_weight_lte_range[0], self.signal_weight_lte_range[1])
            self.signal_weight_lte.append(rand_signal_weight_lte)

            rand_signal_weight_nr = random.uniform(self.signal_weight_nr_range[0], self.signal_weight_nr_range[1])
            self.signal_weight_nr.append(rand_signal_weight_nr)

            rand_packet_deadline_cbr = random.randint(self.packet_deadline_range["cbr"][0], self.packet_deadline_range["cbr"][1])
            rand_packet_deadline_video = random.randint(self.packet_deadline_range["video"][0], self.packet_deadline_range["video"][1])
            self.packet_deadline.append({"cbr": rand_packet_deadline_cbr, "video": rand_packet_deadline_video})

    def fitness_calculate(self, chromosomes, write_data):
        chromosomes_count = len(chromosomes)

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
                lte_buffer.append({"cbr": [], "video": []})
                nr_buffer.append({"cbr": [], "video": []})
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

            for tti in range(self.tti_per_chromosome):
                resource_request_lte = []
                resource_request_nr = []
                ue_priority_lte = []
                ue_priority_nr = []

                for ue in range(self.number_of_genes):
                    packet_count_to_nr_cbr = int(self.packet_count[ue]["cbr"] * chromosomes[i][ue] / 10)
                    packet_count_to_lte_cbr = self.packet_count[ue]["cbr"] - packet_count_to_nr_cbr
                    packet_to_lte_cbr = {"type": "cbr", "count": packet_count_to_lte_cbr, "size": self.packet_size[ue]["cbr"], "time_stamp": tti, "deadline": self.packet_deadline[ue]["cbr"]}
                    packet_to_nr_cbr = {"type": "cbr", "count": packet_count_to_nr_cbr, "size": self.packet_size[ue]["cbr"], "time_stamp": tti, "deadline": self.packet_deadline[ue]["cbr"]}
                    lte_buffer[ue]["cbr"].append(packet_to_lte_cbr)
                    nr_buffer[ue]["cbr"].append(packet_to_nr_cbr)

                    if tti % self.video_packet_gap_time == 0:
                        packet_count_to_nr_video = int(self.packet_count[ue]["video"] * chromosomes[i][ue] / 10)
                        packet_count_to_lte_video = self.packet_count[ue]["video"] - packet_count_to_nr_video
                        packet_to_lte_video = {"type": "video", "count": packet_count_to_lte_video, "size": self.packet_size[ue]["video"], "time_stamp": tti, "deadline": self.packet_deadline[ue]["video"]}
                        packet_to_nr_video = {"type": "video", "count": packet_count_to_nr_video, "size": self.packet_size[ue]["video"], "time_stamp": tti, "deadline": self.packet_deadline[ue]["video"]}
                        lte_buffer[ue]["video"].append(packet_to_lte_video)
                        nr_buffer[ue]["video"].append(packet_to_nr_video)

                    send_data_size_request_lte = 0
                    for packet in lte_buffer[ue]["cbr"]:
                        send_data_size_request_lte = send_data_size_request_lte + (packet["count"] * packet["size"])
                    for packet in lte_buffer[ue]["video"]:
                        send_data_size_request_lte = send_data_size_request_lte + (packet["count"] * packet["size"])

                    send_data_size_request_nr = 0
                    for packet in nr_buffer[ue]["cbr"]:
                        send_data_size_request_nr = send_data_size_request_nr + (packet["count"] * packet["size"])
                    for packet in nr_buffer[ue]["video"]:
                        send_data_size_request_nr = send_data_size_request_nr + (packet["count"] * packet["size"])

                    resource_request_lte.append(send_data_size_request_lte / self.signal_weight_lte[ue])
                    resource_request_nr.append(send_data_size_request_nr / self.signal_weight_nr[ue])

                    #ue_loss_lte[ue][1] = ue_loss_lte[ue][1] + packets_to_lte[ue]
                    #ue_loss_nr[ue][1] = ue_loss_nr[ue][1] + packets_to_nr[ue]

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

                print(resource_request_lte)
                remain_resource_lte = self.lte_resource
                remain_resource_nr = self.nr_resource
                allocate_lte_resource_ue_index = 0
                allocate_nr_resource_ue_index = 0
                ue_avg_lte = np.array(ue_avg_lte) * tti
                ue_avg_nr = np.array(ue_avg_nr) * tti
                while remain_resource_lte > 0 and allocate_lte_resource_ue_index < self.number_of_genes:
                    current_ue_lte = ue_priority_lte[allocate_lte_resource_ue_index]
                    if resource_request_lte[current_ue_lte] <= remain_resource_lte:
                        #for index in range(len(lte_buffer[current_ue_lte])):
                        #    ue_delay_lte[current_ue_lte][0] = ue_delay_lte[current_ue_lte][0] + lte_buffer[current_ue_lte][index] * (len(lte_buffer[current_ue_lte]) - 1 - index)
                        #ue_delay_lte[current_ue_lte][1] = ue_delay_lte[current_ue_lte][1] + sum(lte_buffer[current_ue_lte])
                        ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + (resource_request_lte[current_ue_lte] * self.signal_weight_lte[current_ue_lte])
                        remain_resource_lte = remain_resource_lte - resource_request_lte[current_ue_lte]
                        lte_buffer[current_ue_lte] = {"cbr": [], "video": []}

                    if resource_request_lte[current_ue_lte] > remain_resource_lte:
                        sent_packet = []
                        for packet in lte_buffer[current_ue_lte]["video"]:
                            packet_resource_request = (packet["count"] * packet["size"]) / self.signal_weight_lte[current_ue_lte]
                            if packet_resource_request <= remain_resource_lte:
                                ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + (packet["count"] * packet["size"])
                                remain_resource_lte = remain_resource_lte - packet_resource_request
                                sent_packet.append(packet)
                        for del_packet in sent_packet:
                            lte_buffer[current_ue_lte]["video"].remove(del_packet)

                        sent_packet = []
                        for packet in lte_buffer[current_ue_lte]["cbr"]:
                            packet_resource_request = (packet["count"] * packet["size"]) / self.signal_weight_lte[current_ue_lte]
                            if packet_resource_request <= remain_resource_lte:
                                ue_avg_lte[current_ue_lte] = ue_avg_lte[current_ue_lte] + (packet["count"] * packet["size"])
                                remain_resource_lte = remain_resource_lte - packet_resource_request
                                sent_packet.append(packet)
                        for del_packet in sent_packet:
                            lte_buffer[current_ue_lte]["cbr"].remove(del_packet)
                        #ue_delay_lte[current_ue_lte][0] = ue_delay_lte[current_ue_lte][0] + lte_buffer[current_ue_lte][0] * (len(lte_buffer[current_ue_lte]) - 1)
                        #ue_delay_lte[current_ue_lte][1] = ue_delay_lte[current_ue_lte][1] + lte_buffer[current_ue_lte][0]
                        remain_resource_lte = 0

                    allocate_lte_resource_ue_index = allocate_lte_resource_ue_index + 1
                
                while remain_resource_nr > 0 and allocate_nr_resource_ue_index < self.number_of_genes:
                    current_ue_nr = ue_priority_nr[allocate_nr_resource_ue_index]
                    if resource_request_nr[current_ue_nr] <= remain_resource_nr:
                        #for index in range(len(nr_buffer[current_ue_nr])):
                            #ue_delay_nr[current_ue_nr][0] = ue_delay_nr[current_ue_nr][0] + nr_buffer[current_ue_nr][index] * (len(nr_buffer[current_ue_nr]) - 1 - index)
                        #ue_delay_nr[current_ue_nr][1] = ue_delay_nr[current_ue_nr][1] + sum(nr_buffer[current_ue_nr])
                        ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + (resource_request_nr[current_ue_nr] * self.signal_weight_nr[current_ue_nr])
                        remain_resource_nr = remain_resource_nr - resource_request_nr[current_ue_nr]
                        nr_buffer[current_ue_nr] = {"cbr": [], "video": []}

                    if resource_request_nr[current_ue_nr] > remain_resource_nr:
                        sent_packet = []
                        for packet in nr_buffer[current_ue_nr]["video"]:
                            packet_resource_request = (packet["count"] * packet["size"]) / self.signal_weight_nr[current_ue_nr]
                            if packet_resource_request <= remain_resource_nr:
                                ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + (packet["count"] * packet["size"])
                                remain_resource_nr = remain_resource_nr - packet_resource_request
                                sent_packet.append(packet)
                        for del_packet in sent_packet:
                            nr_buffer[current_ue_nr]["video"].remove(del_packet)

                        sent_packet = []
                        for packet in nr_buffer[current_ue_nr]["cbr"]:
                            packet_resource_request = (packet["count"] * packet["size"]) / self.signal_weight_nr[current_ue_nr]
                            if packet_resource_request <= remain_resource_nr:
                                ue_avg_nr[current_ue_nr] = ue_avg_nr[current_ue_nr] + (packet["count"] * packet["size"])
                                remain_resource_nr = remain_resource_nr - packet_resource_request
                                sent_packet.append(packet)
                        for del_packet in sent_packet:
                            nr_buffer[current_ue_nr]["cbr"].remove(del_packet)
                        #ue_delay_lte[current_ue_lte][0] = ue_delay_lte[current_ue_lte][0] + lte_buffer[current_ue_lte][0] * (len(lte_buffer[current_ue_lte]) - 1)
                        #ue_delay_lte[current_ue_lte][1] = ue_delay_lte[current_ue_lte][1] + lte_buffer[current_ue_lte][0]
                        remain_resource_lte = 0
                    
                    allocate_nr_resource_ue_index = allocate_nr_resource_ue_index + 1

                ue_avg_lte = ue_avg_lte / (tti + 1)
                ue_avg_nr = ue_avg_nr / (tti + 1)
                ue_avg_lte = ue_avg_lte.tolist()
                ue_avg_nr = ue_avg_nr.tolist()

                for ue in range(self.number_of_genes):

                    over_deadline_packet = []
                    for packet in lte_buffer[ue]["cbr"]:
                        if tti - packet["time_stamp"] >= packet["deadline"]:
                            over_deadline_packet.append(packet)
                    for del_packet in over_deadline_packet:
                        lte_buffer[ue]["cbr"].remove(del_packet)
                        #ue_loss_lte[ue][0] = ue_loss_lte[ue][0] + packets_to_lte[ue]

                    over_deadline_packet = []
                    for packet in lte_buffer[ue]["video"]:
                        if tti - packet["time_stamp"] >= packet["deadline"]:
                            over_deadline_packet.append(packet)
                    for del_packet in over_deadline_packet:
                        lte_buffer[ue]["video"].remove(del_packet)
                        #ue_loss_lte[ue][0] = ue_loss_lte[ue][0] + packets_to_lte[ue]

                    over_deadline_packet = []
                    for packet in nr_buffer[ue]["cbr"]:
                        if tti - packet["time_stamp"] >= packet["deadline"]:
                            over_deadline_packet.append(packet)
                    for del_packet in over_deadline_packet:
                        nr_buffer[ue]["cbr"].remove(del_packet)
                        #ue_loss_nr[ue][0] = ue_loss_nr[ue][0] + packets_to_nr[ue]

                    over_deadline_packet = []
                    for packet in nr_buffer[ue]["video"]:
                        if tti - packet["time_stamp"] >= packet["deadline"]:
                            over_deadline_packet.append(packet)
                    for del_packet in over_deadline_packet:
                        nr_buffer[ue]["video"].remove(del_packet)
                        #ue_loss_nr[ue][0] = ue_loss_nr[ue][0] + packets_to_nr[ue]

            #avg_delay_lte_x, avg_delay_lte_y, avg_delay_nr_x, avg_delay_nr_y = 0, 0, 0, 0
            #for ue in range(self.number_of_genes):
            #    avg_delay_lte_x = avg_delay_lte_x + ue_delay_lte[ue][0]
            #    avg_delay_lte_y = avg_delay_lte_y + ue_delay_lte[ue][1]
            #    avg_delay_nr_x = avg_delay_nr_x + ue_delay_nr[ue][0]
            #    avg_delay_nr_y = avg_delay_nr_y + ue_delay_nr[ue][1]
            #avg_delay_x = avg_delay_lte_x + avg_delay_nr_x
            #avg_delay_y = avg_delay_lte_y + avg_delay_nr_y
            #avg_delay = avg_delay_x / avg_delay_y

            #avg_loss_lte_x, avg_loss_lte_y, avg_loss_nr_x, avg_loss_nr_y = 0, 0, 0, 0
            #for ue in range(self.number_of_genes):
            #    avg_loss_lte_x = avg_loss_lte_x + ue_loss_lte[ue][0]
            #    avg_loss_lte_y = avg_loss_lte_y + ue_loss_lte[ue][1]
            #    avg_loss_nr_x = avg_loss_nr_x + ue_loss_nr[ue][0]
            #    avg_loss_nr_y = avg_loss_nr_y + ue_loss_nr[ue][1]
            #avg_loss_x = avg_loss_lte_x + avg_loss_nr_x
            #avg_loss_y = avg_loss_lte_y + avg_loss_nr_y
            #avg_loss = avg_loss_x / avg_loss_y

            avg_throughput = sum(np.array(ue_avg_lte) + np.array(ue_avg_nr)) / self.number_of_genes
            throughput_fairness = prod(np.array(ue_avg_nr) + np.array(ue_avg_lte)) ** (1 / self.number_of_genes)
            #delay_loss_indicator = (self.stable_sigmoid((1.5 - avg_delay) / 1.5) * self.stable_sigmoid((0.5 - avg_loss) / 0.5)) ** 1 / 2
            if self.fitness_calculate_method == 1:
                fitness.append(avg_throughput)
            elif self.fitness_calculate_method == 2:
                fitness.append(throughput_fairness)
            #elif self.fitness_calculate_method == 3:
            #    fitness.append(delay_loss_indicator)
            
            if write_data == True:
                self.data.append([avg_throughput, throughput_fairness, 0, 0])

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

    def crossover_single(self, selected_chromosomes):
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
        

crossover_method = 1
fitness_calculate_method = [1]
simulation_times = 100

pop_size = 10
number_of_genes = 10
lte_resource = 100
nr_resource = 120
tti_per_chromosome = 20
running_tti = 1000

packet_size_range = {"cbr": [5,10], "video": [10,12]}
packet_count_range = {"cbr": [8,10], "video": [8,10]}
video_packet_gap_time = 5
signal_weight_lte_range = [2, 5]
signal_weight_nr_range = [2, 5]
packet_deadline_range = {"cbr": [3, 5], "video": [5, 8]}

f = open('ga_fca.csv', 'w')
w = csv.writer(f)

for method in fitness_calculate_method:
    data_array = []
    data_output = []
    data_type = ["GA", "70% NR", "50% NR", "30% NR", "Only LTE"]
    w.writerow(["Method{}".format(method)])
    w.writerow(["", "Avg Throughput", "Throughput Fairness", "Delay", "Loss"])
    for _ in range(simulation_times):
        ga_fca = GATest(method, crossover_method, pop_size, number_of_genes,lte_resource, nr_resource, tti_per_chromosome, running_tti,
                        packet_size_range, packet_count_range, video_packet_gap_time, signal_weight_lte_range, signal_weight_nr_range,
                        packet_deadline_range)
        data = ga_fca.main()
        data_array.append(data)

    for i in range(5):
        data_output.append([])
        for j in range(4):
            data_output[i].append(0)

    for m in range(simulation_times):
        for i in range(5):
            for j in range(4):
                data_output[i][j] = data_output[i][j] + data_array[m][i][j]
    data_output = (np.array(data_output) / simulation_times).tolist()

    for i in range(len(data_type)):
        w.writerow([data_type[i]] + data_output[i])
    
    w.writerow([])

f.close()