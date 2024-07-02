import random
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
figure(figsize=[8, 8], dpi = 100)
import math

class Graph():
	num_nodes = 50
	nodes_list = []
	graph_dict = {}
	agent_pos = -1
	pred_pos = -1
	prey_pos = -1
	dist_dict = {}
	connectivity_matrix = np.array([[0 for i in range(50)] for j in range(50)])
	possible_starts = []

	def edge_available(self, current_node, check_list = []):
		edge_sum = 0
		graph_neighbours = []
		for i in range(2,6):
			graph_neighbours.append(self.nodes_list[current_node-i])
		for i in range(2,6):
			graph_neighbours.append(self.nodes_list[(current_node+i)%len(self.nodes_list)])
		print("graph_neighbours: " + str(graph_neighbours))
		for i in graph_neighbours:
			if (i != current_node):
				print("for key " + str(i))
				print("Adding length of " + str(self.graph_dict[self.nodes_list[i]]) + " to the sum")
				edge_sum += len(self.graph_dict[self.nodes_list[i]])
				if (len(self.graph_dict[self.nodes_list[i]]) < 3):
					check_list.append(self.nodes_list[i])
		if edge_sum == 24:
			return False
		else:
			return True

	def add_edge_3(self, current_node):
		flag = False
		print("Current Node: " + str(current_node))
		print("Order of currrent node: " + str(len(self.graph_dict[self.nodes_list[current_node]])))
		print("Current Node Neighbours: " + str(self.graph_dict[current_node]))
		check_list = []
		if (self.edge_available(current_node, check_list)):
			while(flag == False and len(self.graph_dict[self.nodes_list[current_node]]) < 3):
				x = random.randint(0,len(check_list)-1)
				x = check_list[x]
				print("x: " + str(x))
				other_node = self.nodes_list[x]
				print("Node being considered: " + str(other_node))
				print("Order of that node: " + str(len(self.graph_dict[other_node])))
				print("That node Neighbours: " + str(self.graph_dict[other_node]))

				if ((other_node != self.nodes_list[current_node]) and other_node not in self.graph_dict[current_node]):
					self.graph_dict[current_node].append(other_node)
					self.graph_dict[other_node].append(current_node)
					print("Edge added")
					print(str(current_node) + ": " + str(self.graph_dict[current_node]))
					print(str(other_node) + ": " + str(self.graph_dict[other_node]))
					flag = True

	def create_graph(self):
		for i in range(50):
			self.nodes_list.append(i)

		for i in range(len(self.nodes_list)):
			print("i:" + str(i) + " Current:" + str(self.nodes_list[i]))
			print(" and Previous: " + str(self.nodes_list[i-1]) + " and Next: " + str(self.nodes_list[(i+1)%len(self.nodes_list)]))
			self.graph_dict[self.nodes_list[i]] = [self.nodes_list[i-1],self.nodes_list[(i+1)%len(self.nodes_list)]]
		
		for i in range(len(self.nodes_list)):
			self.add_edge_3(i)

		##########CONNECTIVITY MATRIX START##########
		for i in range(len(self.connectivity_matrix)):
			for val in self.graph_dict[i]:
				self.connectivity_matrix[i][val] = 1
		##########CONNECTIVITY MATRIX END##########
		
		for i in range(len(self.nodes_list)):
			print(str(i) + " " + str(len(self.graph_dict[i])))


	def initialize_positions(self):
		self.possible_starts = self.nodes_list.copy()
		self.agent_pos = random.randint(0,49)
		
		#Removing agents position from the list of possible nodes. So, the agent and the pred and the agent and the prey will never start from the same pos
		#However, the pred and the prey can spawn at the same node as told
		self.possible_starts.remove(self.agent_pos)
		self.pred_pos = random.choice(self.possible_starts)
		self.prey_pos = random.choice(self.possible_starts)


	def calc_path(self, graph_dict, from_pos, to_pos):
		closed_list = []
	
		queue = [[from_pos]]
	
		if from_pos == to_pos:
			return [from_pos]
	
		while queue:
			temp_path = queue.pop(0)
			node = temp_path[-1]
		
			if node not in closed_list:
				neighbours = graph_dict[node]
				
				for neighbour in neighbours:
					shortest_path = list(temp_path)
					shortest_path.append(neighbour)
					queue.append(shortest_path)
				
					if neighbour == to_pos:
						return shortest_path
				closed_list.append(node)


class Prey():
	def move_prey(self, prey_pos, connectivity_matrix):
		possible_next = [prey_pos]
		for i in range(len(connectivity_matrix)):
			if connectivity_matrix[prey_pos][i] == 1:
				possible_next.append(i)
		print("Current Location: " + str(prey_pos))
		prey_pos = random.choice(possible_next)
		print("Next Location: " + str(prey_pos))
		return prey_pos


class Predator():
	def move_pred(self, graph_dict, agent_pos, pred_pos, agent_no):
		print("Predator Position: " + str(pred_pos))
		y = random.randint(0,10)
		if (agent_no <= 4 or (agent_no >= 5 and y >= 4)): #random.random() gives [0,1.0). So <= 40% would be [0,3.9999...]
			graph_1 = Graph()
			path = graph_1.calc_path(graph_dict, pred_pos, agent_pos)
			pred_pos = path[1]
		else:
			pred_pos = random.choice(graph_dict[pred_pos])
		print("Predators new position: " + str(pred_pos))
		return pred_pos


class agents_common():
	step_count = 1
	pred_probabilities = {}
	prey_probabilities = {}
	last_seen_prey = -1
	last_seen_Pred = -1

	def decide_node_even(self, graph_dict, dist_to_prey, dist_to_pred, agent_pos, pred_pos, candidate_nodes):
		min_pred_dist = dist_to_pred[agent_pos]
		if min_pred_dist >= 6:
			agent_pos = self.decide_node(dist_to_prey, dist_to_pred, agent_pos, candidate_nodes)
		else:
			graph = Graph()
			#calculating possible next position of predator by taking shortest route from predator to agent. 
			#Distracted predator is not taken care of int this approach as it has lesser probability of happening
			#We observed during trail runs that the predator many times took the shortest path and not the distracted path. So not including
			path_from_Pred = graph.calc_path(graph_dict, pred_pos, agent_pos)
			if len(path_from_Pred) > 1:
				pred_next_node = path_from_Pred[1]
			else:
				pred_next_node = agent_pos
			min_pred_next_dist = len(graph.calc_path(graph_dict, pred_next_node, agent_pos))
			min_prey_dist = dist_to_prey[agent_pos]
			narrowed_candidates = []
			dist_to_pred_next = {}
			del dist_to_pred[agent_pos]
			del dist_to_prey[agent_pos]

			#Generating list of distances to the predators next possible position
			for val in graph_dict[agent_pos]:
				print("Calculating distance to pred("+str(pred_pos)+") from neighbour("+str(val)+") of current("+str(agent_pos)+")")
				dist_to_pred_next[val] = len(graph.calc_path(graph_dict, val, pred_pos))

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - Start
			for i in candidate_nodes:
				#Shortlist all neighbouring nodes that are closer to prey and farther from the predator and predator_next
				if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] > min_pred_dist and dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					#remove key, val pairs from distance to prey/pred dictionaries for nodes not selected above
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 1 check for even")

			#Checking if there are multiple nodes that are closer to prey and farther from the predator
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes that are closer to preds next possible node with the same distance
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to predator dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_pred[key]

					#If multiple, selecting node with maximum distance to predator
					narrow_pred = []
					#If multiple, selecing node with maximum distance from predator
					narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
					if (len(narrow_pred) > 1):
						for key in lcl_dist_to_pred.keys():
							if key not in narrow_pred:
								del lcl_dist_to_prey[key]

						narrow_prey = []
						narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
						if (len(narrow_prey) > 1):
							return narrow_prey[random.randint(0,len(narrow_prey)-1)]
						elif (len(narrow_prey) == 1):
							#If still multiple after selecting min(distance from prey) and max(distance from predator), max(distance from predator next), selecting randomly
							return narrow_prey[0]
					elif (len(narrow_pred) == 1):
						return narrow_pred[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes Closer to Prey and Farther from Predators next pos - Condition 1_1 - Start
			for i in candidate_nodes:
				#Shortlist all neighbouring nodes that are closer to prey and farther from predator_next
				if dist_to_prey[i] < min_prey_dist and dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					#remove key, val pairs from distance to prey, pred, pred_next dictionaries for nodes not selected above
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 1_1 check for even")

			#Checking if there are multiple nodes that are closer to prey and farther from the predator next
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes that are farthest from pred next location
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to prey dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_prey[key]

					#If multiple, selecting node with minimum distance from prey
					narrow_prey = []
					narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
					if (len(narrow_prey) > 1):
						#If more than one with same distance, select at random
						return narrow_prey[random.randint(0,len(narrow_prey)-1)]
					elif (len(narrow_prey) == 1):
						return narrow_prey[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]

			#selecting nodes Closer to Prey and Farther from Predators next pos - Condition 1_1 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes Closer to Prey and Farther from Predators current position - Condition 1_2 - Start
			for i in candidate_nodes:
				#Shortlist all neighbouring nodes that are closer to prey and farther from the predator
				if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] > min_pred_dist:
					narrowed_candidates.append(i)
				else:
					#remove key, val pairs from distance to prey,pred dictionaries for nodes not selected above
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 1_2 check for even")

			#Checking if there are multiple nodes that are closer to prey and farther from the predator
			if (len(narrowed_candidates) > 1):
				narrow_pred = []
				narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

				#Checking if there are multiple nodes that are farther from the predator with the same distance
				if (len(narrow_pred) > 1):
					#remove key, val pairs from distance to prey dictionary which are not selected above
					for key in lcl_dist_to_pred.keys():
						if key not in narrow_pred:
							del lcl_dist_to_prey[key]

					#If multiple, selecting node with minimum distance from prey
					narrow_prey = []
					narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
					if (len(narrow_prey) > 1):
						#If still multiple that have same minimum distance from prey, choose randomly
						return narrow_prey[random.randint(0,len(narrow_prey)-1)]
					elif (len(narrow_prey) == 1):
						#If only 1 with minimum distance from prey, choose that
						return narrow_prey[0]
				#If only 1 with maximum distance to predator, choose that
				elif (len(narrow_pred) == 1):
					return narrow_pred[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes Closer to Prey and Farther from Predators current position - Condition 1_2 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes farther from Predator and predator_next - Condition 2 - Start
			for i in candidate_nodes:
				if dist_to_pred[i] > min_pred_dist and dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 2 check")

			#Checking if there are multiple nodes that are farther from the predator and predator_next positions
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes witht he same distance that are farthest from predators next possible location
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to predator dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_pred[key]

					#If multiple, selecting node with maximum distance to predator
					narrow_pred = []
					narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
					if (len(narrow_pred) > 1):
						#remove key, val pairs from distance to prey dictionary which are not selected above
						for key in lcl_dist_to_pred.keys():
							if key not in narrow_pred:
								del lcl_dist_to_prey[key]

						#Select nodes that are closes to the prey from the nodes shortlisted above
						narrow_prey = []
						narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
						if (len(narrow_prey) > 1):
							#If multiple, choose randomly
							return narrow_prey[random.randint(0,len(narrow_prey)-1)]
						elif (len(narrow_prey) == 1):
							#If only one node, choose that
							return narrow_prey[0]
					elif (len(narrow_pred) == 1):
						return narrow_pred[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes farther from Predator and predator_next - Condition 2 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes farther from Predator and predator_next - Condition 2_1 - Start
			for i in candidate_nodes:
				if dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					#del lcl_dist_to_prey[i]
			
			print("Inside condition 2_1 check")

			#Select the node that is farthest from teh predators next location
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes that farthest from pred next location with same distance
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to predator dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_pred[key]

					narrow_pred = []
					#select node that is farthest from the predators current position
					narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
					if (len(narrow_pred) > 1):
						#If multiple, choose randomly
						return narrow_pred[random.randint(0,len(narrow_pred)-1)]
					elif (len(narrow_pred) == 1):
						return narrow_pred[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes farther from Predator and predator_next - Condition 2_1 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes farther from Predators current location - Condition 2_2 - Start
			for i in candidate_nodes:
				if dist_to_pred[i] > min_pred_dist:
					narrowed_candidates.append(i)
				else:
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					#del lcl_dist_to_prey[i]
			
			print("Inside condition 2_2 check")

			#Checking if there are multiple nodes that farther from the predator
			if (len(narrowed_candidates) > 1):
				narrow_pred = []
				#Choose nodes that are farthest from predators current position
				narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

				#Checking if there are multiple nodes that are farther from the predator with the same distance
				if (len(narrow_pred) > 1):
					#remove key, val pairs from distance to predator next position dictionary which are not selected above
					for key in lcl_dist_to_pred.keys():
						if key not in narrow_pred:
							del lcl_dist_to_pred_next[key]

					narrow_pred_next = []
					#Select nodes that are farthest away from the predators next position
					narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]
					if (len(narrow_pred_next) > 1):
							#If multiple, choose randomly
							return narrow_pred_next[random.randint(0,len(narrow_pred_next)-1)]
					elif (len(narrow_pred_next) == 1):
						return narrow_pred_next[0]
				elif (len(narrow_pred) == 1):
					return narrow_pred[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes farther from Predators current location - Condition 2_2 - End
		return agent_pos


	def decide_node(self, dist_to_prey, dist_to_pred, agent_pos, candidate_nodes):
		min_pred_dist = dist_to_pred[agent_pos]
		min_prey_dist = dist_to_prey[agent_pos]
		chosen_node = agent_pos
		narrowed_candidates = []
		del dist_to_pred[agent_pos]
		del dist_to_prey[agent_pos]
		lcl_dist_to_pred = dist_to_pred.copy()
		lcl_dist_to_prey = dist_to_prey.copy()

		#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - Start
		for i in candidate_nodes:
			#Shortlist all neighbouring nodes that are closer to prey and farther from the predator
			if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] > min_pred_dist:
				narrowed_candidates.append(i)
			else:
				#remove key, val pairs from distance to prey/pred dictionaries for nodes not selected above
				del lcl_dist_to_pred[i]
				del lcl_dist_to_prey[i]
		
		print("Inside condition 1 check")

		#Checking if there are multiple nodes that are closer to prey and farther from the predator
		if (len(narrowed_candidates) > 1):
			narrow_prey = []
			#If multiple, selecting node with minimum distance to prey
			narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]

			#Checking if there are multiple nodes that are closer to prey with the same distance
			if (len(narrow_prey) > 1):
				#remove key, val pairs from distance to predator dictionary which are not selected above
				for key in lcl_dist_to_pred.keys():
					if key not in narrow_prey:
						del lcl_dist_to_pred[key]

				#If multiple, selecting node with maximum distance to predator
				narrow_pred = []
				#If multiple, selecing node with maximum distance from predator
				narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
				if (len(narrow_pred) > 1):
					#If still multiple after selecting min(distance from prey) and max(distance from predator), selecting randomly
					chosen_node = narrow_pred[random.randint(0,len(narrow_pred)-1)]
					#chosen_node = random.choice(narrow_pred)#KAUTILYA - CAN USE. NEED TO CHECK BEHAVIOUR
					return chosen_node
				elif (len(narrow_pred) == 1):
					return narrow_pred[0]
			elif (len(narrow_prey) == 1):
				return narrow_prey[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - End

		#lcl_dist_to_pred = dist_to_pred.copy()
		lcl_dist_to_prey = dist_to_prey.copy()
		
		#selecting nodes Closer to Prey and not Farther from Predator - Condition 2 - Start
		for i in candidate_nodes:
			if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] == min_pred_dist:
				narrowed_candidates.append(i)
			else:
				#del lcl_dist_to_pred[i]
				del lcl_dist_to_prey[i]
		
		print("Inside condition 2 check")

		if (len(narrowed_candidates) > 1):
			narrow_prey = []
			#If multiple, selecting node with minimum distance to prey
			narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]

			if (len(narrow_prey) > 1):
				chosen_node = narrow_prey[random.randint(0,len(narrow_prey)-1)]
				return chosen_node
			elif (len(narrow_prey) == 1):
				return narrow_prey[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes Closer to Prey and not Farther from Predator - Condition 2 - End

		lcl_dist_to_pred = dist_to_pred.copy()

		#selecting nodes not farther from Prey and Farther from Predator - Condition 3 - Start
		for i in candidate_nodes:
			if dist_to_prey[i] == min_prey_dist and dist_to_pred[i] > min_pred_dist:
				narrowed_candidates.append(i)
			else:
				del lcl_dist_to_pred[i]
				#del lcl_dist_to_prey[i]
		
		print("Inside condition 3 check")

		if (len(narrowed_candidates) > 1):
			narrow_pred = []
			#If multiple, selecting node with minimum distance to prey
			narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

			if (len(narrow_pred) > 1):
				chosen_node = narrow_pred[random.randint(0,len(narrow_pred)-1)]
				return chosen_node
			elif (len(narrow_pred) == 1):
				return narrow_pred[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes not farther from Prey and Farther from Predator - Condition 3 - End

		#selecting nodes not farther from Prey and not closer to Predator - Condition 4 - Start
		for i in candidate_nodes:
			if dist_to_prey[i] == min_prey_dist and dist_to_pred[i] == min_pred_dist:
				narrowed_candidates.append(i)
		
		print("Inside condition 4 check")

		if (len(narrowed_candidates) > 1):
			chosen_node = narrowed_candidates[random.randint(0,len(narrowed_candidates)-1)]
			return chosen_node
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes not farther from Prey and not closer to Predator - Condition 4 - End

		lcl_dist_to_pred = dist_to_pred.copy()

		#selecting nodes farther from Predator - Condition 5 - Start
		for i in candidate_nodes:
			if dist_to_pred[i] > min_pred_dist:
				narrowed_candidates.append(i)
			else:
				del lcl_dist_to_pred[i]
		
		print("Inside condition 5 check")

		if (len(narrowed_candidates) > 1):
			narrow_pred = []
			#If multiple, selecting node with minimum distance to prey
			narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

			if (len(narrow_pred) > 1):
				chosen_node = narrow_pred[random.randint(0,len(narrow_pred)-1)]
				return chosen_node
			elif (len(narrow_pred) == 1):
				return narrow_pred[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes farther from Predator - Condition 5 - End

		#selecting nodes not closer to Predator - Condition 6 - Start
		for i in candidate_nodes:
			if dist_to_pred[i] == min_pred_dist:
				narrowed_candidates.append(i)
		
		print("Inside condition 6 check")

		if (len(narrowed_candidates) > 1):
			chosen_node = narrowed_candidates[random.randint(0,len(narrowed_candidates)-1)]
			return chosen_node
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes not closer to Predator - Condition 6 - End

		#Stay still and Pray - Condition 7 - Start
		print("Inside condition 7")
		return agent_pos
		#Stay still and Pray - Condition 7 - End

	def check_status(self, agent_pos, pred_pos, prey_pos):
		if agent_pos == pred_pos:
			return 2
		if agent_pos == prey_pos:
			return 1
		return 0

	def init_prob(self, graph_dict, pred_pos, agent_pos):
		print("Inside init_prob...")
		if pred_pos == -1:
			print("Initializing prey probabilities...")
			for key in graph_dict.keys():
				if key == agent_pos:
					self.prey_probabilities[key] = 0
				else:
					self.prey_probabilities[key] = 1/(len(graph_dict)-1)
		else:
			print("Initializing pred probabilities...")
			for key in graph_dict.keys():
				if key == pred_pos:
					self.pred_probabilities[key] = 1
				else:
					self.pred_probabilities[key] = 0


	def update_prey_prob_presurvey(self, graph_dict, pred_pos, agent_pos, step_count):#KAUTILYA
		print("In update_prey_prob_presurvey...")
		prey_probabilities_temp = {}

		if self.last_seen_prey < 0 and step_count == 1:
			print("Prey not yet found...ever...Calculating probability based on that")
			self.init_prob(graph_dict, -1, agent_pos)
		else:
			#P(x) = ∑(P(in x now, was in i)) for i in 0...49 = ∑(P(was in i).P(in x|was in i)) for i in 0...49
			for key in graph_dict.keys():
				temp = 0
				for k in self.prey_probabilities.keys():
					child_prob = 0

					if key in graph_dict[k] or key == k:
						child_prob = (1/(len(graph_dict[k])+1))
					else:
						child_prob = 0
					temp += (self.prey_probabilities[k] * child_prob)
				prey_probabilities_temp[key] = temp

			self.prey_probabilities = prey_probabilities_temp.copy()


	def update_prey_prob_postsurvey(self, graph_dict, prey_pos, agent_pos, step_count, survey_node, survey_result):
		print("In update_prey_prob_postsurvey...")
		prey_post_survey_prob_temp = {}

		if survey_result == True:
			print("Prey found from survey. Calculating probability based on that")
			for key in graph_dict.keys():
				if key == prey_pos:
					prey_post_survey_prob_temp[key] = 1
				else:
					prey_post_survey_prob_temp[key] = 0
		else:
			#P(survey_node) = 0
			#P(x) = P(in X, not in survey_node)/P(not in survey_node)
			#P(x) = P(x).P(not in survey_node|in x) / (∑P(i).P(not in survey_node| in i)) for i in 0...49
			for key in graph_dict.keys():
				if key == survey_node:
					P_not_in_survey_given_in_key = 0
				else:
					P_not_in_survey_given_in_key = 1
				
				P_not_in_survey = 0
				
				for k in self.prey_probabilities.keys():
					if k == survey_node:
						P_denom_second_term = 0
					else:
						P_denom_second_term = 1
					
					P_not_in_survey += (self.prey_probabilities[k]*P_denom_second_term)

				prey_post_survey_prob_temp[key] = self.prey_probabilities[key]*P_not_in_survey_given_in_key/P_not_in_survey

		self.prey_probabilities = prey_post_survey_prob_temp.copy()


	def update_prey_prob_postsurvey_fn(self, graph_dict, prey_pos, agent_pos, step_count, survey_node, survey_result):
		print("In update_prey_prob_postsurvey_fn...")
		prey_post_survey_prob_temp = {}

		#P(survey_node) = 0
		#P(x) = P(in X, not in survey_node)/P(not in survey_node)
		#P(x) = P(x).P(not in survey_node|in x) / (∑P(i).P(not in survey_node| in i)) for i in 0...49
		print("Prey not found from survey. Calculating probability based on that")
		for key in graph_dict.keys():
			if key == survey_node:
				P_not_in_survey_given_in_key = (1/10)
			else:
				P_not_in_survey_given_in_key = 1
			
			P_not_in_survey = 0
			
			for k in self.prey_probabilities.keys():
				if k == survey_node:
					P_denom_second_term = (1/10)
				else:
					P_denom_second_term = 1
				
				P_not_in_survey += (self.prey_probabilities[k]*P_denom_second_term)

			prey_post_survey_prob_temp[key] = self.prey_probabilities[key]*P_not_in_survey_given_in_key/P_not_in_survey


		self.prey_probabilities = prey_post_survey_prob_temp.copy()


	def update_pred_prob_presurvey(self, graph_dict, pred_pos, agent_pos, step_count):#KAUTILYA
		print("In update_pred_prob_presurvey...")
		pred_probabilities_temp = {}
		graph_prob = Graph()

		if step_count == 1:
			self.init_prob(graph_dict, pred_pos, agent_pos)
		else:
			#∑(P(was in i,random_way,in y) + P(was in i,shortest_way,in y))
			#P(y) = ∑(P(was in i).P(random_way|was in i).P(in y|was in i,random_way) + P(was in i).P(shortest_way|was in i).P(in y|was in i,shortest_way))
			for key in graph_dict.keys():
				temp = 0
				for k in self.pred_probabilities.keys():
					child_prob = 0
					temp_path = []

					if key in graph_dict[k]:# or key == k:
						child_prob = (1/(len(graph_dict[k])))
					else:
						child_prob = 0

					if key in graph_dict[k] and k != agent_pos:
						temp_path = graph_prob.calc_path(graph_dict, k, agent_pos)

						if key == temp_path[1]:
							prob_undictract_next = 1
						else:
							prob_undictract_next = 0
					else:
						prob_undictract_next = 0

					temp += ((self.pred_probabilities[k] * 0.4 * child_prob) + (self.pred_probabilities[k] * 0.6 * prob_undictract_next))
				pred_probabilities_temp[key] = temp

			self.pred_probabilities = pred_probabilities_temp.copy()


	def update_pred_prob_postsurvey(self, graph_dict, pred_pos, agent_pos, step_count, survey_node, survey_result):
		print("In update_pred_prob_postsurvey...")
		pred_post_survey_prob_temp = {}

		if survey_result == True:
			print("Pred found from survey. Calculating probability based on that")
			for key in graph_dict.keys():
				if key == pred_pos:
					pred_post_survey_prob_temp[key] = 1
				else:
					pred_post_survey_prob_temp[key] = 0
		else:
			#P(survey_node) = 0
			#P(x) = P(in X, not in survey_node)/P(not in survey_node)
			#P(x) = P(x).P(not in survey_node|in x) / (∑P(i).P(not in survey_node| in i)) for i in 0...49
			for key in graph_dict.keys():
				if key == survey_node:
					P_not_in_survey_given_in_key = 0
				else:
					P_not_in_survey_given_in_key = 1
				P_not_in_survey = 0
				
				for k in self.pred_probabilities.keys():
					if k == survey_node:
						P_denom_second_term = 0
					else:
						P_denom_second_term = 1

					P_not_in_survey += (self.pred_probabilities[k]*P_denom_second_term)

				if self.pred_probabilities[key]*P_not_in_survey_given_in_key == 0:
					pred_post_survey_prob_temp[key] = 0
				else:
					pred_post_survey_prob_temp[key] = self.pred_probabilities[key]*P_not_in_survey_given_in_key/P_not_in_survey

		self.pred_probabilities = pred_post_survey_prob_temp.copy()


	def update_pred_prob_postsurvey_fn(self, graph_dict, pred_pos, agent_pos, step_count, survey_node, survey_result):
		print("In update_pred_prob_postsurvey_fn...")
		pred_post_survey_prob_temp = {}

		#P(survey_node) = 0
		#P(x) = P(in X, not in survey_node)/P(not in survey_node)
		#P(x) = P(x).P(not in survey_node|in x) / (∑P(i).P(not in survey_node| in i)) for i in 0...49
		for key in graph_dict.keys():
			if key == survey_node:
				P_not_in_survey_given_in_key = (1/11)
			else:
				P_not_in_survey_given_in_key = (10/11)

			P_not_in_survey = 0
			
			for k in self.pred_probabilities.keys():
				if k == survey_node:
					P_denom_second_term = (1/11)
				else:
					P_denom_second_term = (10/11)

				P_not_in_survey += (self.pred_probabilities[k]*P_denom_second_term)

			if self.pred_probabilities[key]*P_not_in_survey_given_in_key == 0:
				pred_post_survey_prob_temp[key] = 0
			else:
				pred_post_survey_prob_temp[key] = self.pred_probabilities[key]*P_not_in_survey_given_in_key/P_not_in_survey

		self.pred_probabilities = pred_post_survey_prob_temp.copy()


class test_agent():
	def proceed(self, graph_dict, connectivity_matrix, prey_pos, pred_pos, agent_pos):
		game_over = 0
		graph_1 = Graph()
		agents_comm_1 = agents_common()
		prey_1 = Prey()
		pred_1 = Predator()

		while (agents_comm_1.step_count <= 50 and game_over == 0):
			print("agents_comm_1.step_count: " + str(agents_comm_1.step_count))
			print("agent_pos: " + str(agent_pos))
			print("pred_pos: " + str(pred_pos))
			print("prey_pos: " + str(prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_pos)+")")
			dist_to_pred[agent_pos] = graph_1.calc_path(graph_dict, agent_pos, pred_pos)
			print("Calculating distance to prey("+str(prey_pos)+") from agent("+str(agent_pos)+")")
			dist_to_prey[agent_pos] = len(graph_1.calc_path(graph_dict, agent_pos, prey_pos))
			candidate_nodes = []
			
			for val in graph_dict[agent_pos]:
				dist_to_prey[val] = len(graph_1.calc_path(graph_dict, val, prey_pos))

			prey_dist = []
			prey_dist = [key for key, value in dist_to_prey.items() if value == min(dist_to_prey.values())]
			print("Getting nodes with max prey probability: " + str(prey_dist))
			if (len(prey_dist) == 1):
				agent_pos = prey_dist[0]
			else:
				agent_pos = random.choice(prey_dist)

			print("Agent 1 position after update: "+str(agent_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_1.move_prey(prey_pos, connectivity_matrix)
			print("Prey position after update: "+str(prey_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_1.move_pred(graph_dict, agent_pos, pred_pos,1)
			print("Pred position after update: "+str(pred_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over
			agents_comm_1.step_count += 1
		return game_over


class Agent1():
	max_Step_count = 50
	def proceed(self, graph_dict, connectivity_matrix, prey_pos, pred_pos, agent_pos):
		game_over = 0
		graph_1 = Graph()
		agents_comm_1 = agents_common()
		prey_1 = Prey()
		pred_1 = Predator()

		while (agents_comm_1.step_count <= self.max_Step_count and game_over == 0):
			print("agents_comm_1.step_count: " + str(agents_comm_1.step_count))
			print("agent_pos: " + str(agent_pos))
			print("pred_pos: " + str(pred_pos))
			print("prey_pos: " + str(prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_pos)+")")
			dist_to_pred[agent_pos] = len(graph_1.calc_path(graph_dict, agent_pos, pred_pos))
			print("Calculating distance to prey("+str(prey_pos)+") from agent("+str(agent_pos)+")")
			dist_to_prey[agent_pos] = len(graph_1.calc_path(graph_dict, agent_pos, prey_pos))
			candidate_nodes = []
			
			for val in graph_dict[agent_pos]:
				dist_to_pred[val] = len(graph_1.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_1.calc_path(graph_dict, val, prey_pos))
				candidate_nodes.append(val)
			
			print("Agent 1 position before update: "+str(agent_pos))
			agent_pos = agents_comm_1.decide_node(dist_to_prey, dist_to_pred, agent_pos, candidate_nodes)
			print("Agent 1 position after update: "+str(agent_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_1.move_prey(prey_pos, connectivity_matrix)
			print("Prey position after update: "+str(prey_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_1.move_pred(graph_dict, agent_pos, pred_pos,1)
			print("Pred position after update: "+str(pred_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over
			agents_comm_1.step_count += 1
		return game_over


#AGENT2_START#
class Agent2():
	max_Step_count = 50

	def proceed(self, graph_dict, connectivity_matrix, prey_pos, pred_pos, agent_pos):
		game_over = 0
		graph_2 = Graph()
		agents_comm_2 = agents_common()
		prey_2 = Prey()
		pred_2 = Predator()

		while (agents_comm_2.step_count <= self.max_Step_count and game_over == 0):
			print("agents_comm_2.step_count: " + str(agents_comm_2.step_count))
			print("agent_pos: " + str(agent_pos))
			print("pred_pos: " + str(pred_pos))
			print("prey_pos: " + str(prey_pos))
			possible_locs = [prey_pos]
			dist_to_prey = {}
			dist_to_pred = {}

			for i in graph_dict[prey_pos]:
				possible_locs.append(i)
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_pos)+")")
			dist_to_pred[agent_pos] = len(graph_2.calc_path(graph_dict, agent_pos, pred_pos))
			chosen_prey_pos = random.choice(possible_locs)
			if len(graph_2.calc_path(graph_dict, agent_pos, prey_pos)) <= 1:
				chosen_prey_pos = prey_pos
			print("Calculating distance to prey("+str(chosen_prey_pos)+") from agent("+str(agent_pos)+")")
			dist_to_prey[agent_pos] = len(graph_2.calc_path(graph_dict, agent_pos, chosen_prey_pos))
			candidate_nodes = []
			
			for val in graph_dict[agent_pos]:
				dist_to_pred[val] = len(graph_2.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_2.calc_path(graph_dict, val, chosen_prey_pos))
				candidate_nodes.append(val)
			
			print("Agent 1 position before update: "+str(agent_pos))
			agent_pos = agents_comm_2.decide_node_even(graph_2.graph_dict, dist_to_prey, dist_to_pred, agent_pos, pred_pos, candidate_nodes)
			print("Agent 1 position after update: "+str(agent_pos))
			game_over = agents_comm_2.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_2.move_prey(prey_pos, connectivity_matrix)
			print("Prey position after update: "+str(prey_pos))
			game_over = agents_comm_2.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_2.move_pred(graph_dict, agent_pos, pred_pos,2)
			print("Pred position after update: "+str(pred_pos))
			game_over = agents_comm_2.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return game_over
			agents_comm_2.step_count += 1
		return game_over


class Agent3():
	max_Step_count = 50

	def survey_node(self, graph_dict, prey_pos, prey_possible):
		if prey_pos == prey_possible:
			return True
		return False

	def proceed(self, graph):
		game_over = 0
		agents_comm_3 = agents_common()
		prey_3 = Prey()
		pred_3 = Predator()
		prey_known_ctr = 0

		while (agents_comm_3.step_count <= self.max_Step_count and game_over == 0):
			survey_result = False
			print("agents_comm_3.step_count: " + str(agents_comm_3.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(graph.pred_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_pred[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, graph.pred_pos))

			agents_comm_3.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_3.step_count)
			print("Prey Probabilities before survey: " + str(agents_comm_3.prey_probabilities))
			print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_3.prey_probabilities.values())))

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_3.prey_probabilities.items() if value == max(agents_comm_3.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)
			
			print("Surveying Node: " + str(prey_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.prey_pos, prey_possible)

			agents_comm_3.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_3.step_count, prey_possible, survey_result)
			print("Updated Prey probabilities after survey: " + str(agents_comm_3.prey_probabilities))
			print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_3.prey_probabilities.values())))

			if survey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_3.last_seen_prey = agents_comm_3.step_count
				prey_known_ctr += 1
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_3.prey_probabilities.items() if value == max(agents_comm_3.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			print("Calculating distance to prey("+str(lcl_prey_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_prey[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos))

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				dist_to_pred[val] = len(graph.calc_path(graph.graph_dict, val, graph.pred_pos))
				dist_to_prey[val] = len(graph.calc_path(graph.graph_dict, val, lcl_prey_pos))
				candidate_nodes.append(val)

			print("Agent 3 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_3.decide_node(dist_to_prey, dist_to_pred, graph.agent_pos, candidate_nodes)
			print("Agent 3 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_3.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_3.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_3.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_3.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,3)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_3.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			agents_comm_3.step_count += 1

		if agents_comm_3.step_count > self.max_Step_count:
			prey_known_percent = (prey_known_ctr/(agents_comm_3.step_count-1))*100
		else:
			prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
			
		print("Prey Known Percentage: " + str(prey_known_percent))

		return game_over


class Agent4():
	max_Step_count = 50

	def survey_node(self, graph_dict, prey_pos, prey_possible):
		if prey_pos == prey_possible:
			return True
		return False

	def proceed(self, graph):
		game_over = 0
		agents_comm_4 = agents_common()
		prey_4 = Prey()
		pred_4 = Predator()
		pre_prob_backup = {}
		prey_next_calculated = False
		prey_known_ctr = 0

		while (agents_comm_4.step_count <= self.max_Step_count and game_over == 0):
			survey_result = False
			print("agents_comm_4.step_count: " + str(agents_comm_4.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(graph.pred_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_pred[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, graph.pred_pos))

			if (prey_next_calculated == False):
				agents_comm_4.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_4.step_count)
				print("Prey Probabilities before survey: " + str(agents_comm_4.prey_probabilities))
				print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_4.prey_probabilities.values())))
			else:
				prey_next_calculated = False

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_4.prey_probabilities.items() if value == max(agents_comm_4.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)
			
			print("Surveying Node: " + str(prey_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.prey_pos, prey_possible)

			agents_comm_4.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_4.step_count, prey_possible, survey_result)
			print("Updated Prey probabilities after survey: " + str(agents_comm_4.prey_probabilities))
			print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_4.prey_probabilities.values())))

			if survey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_4.last_seen_prey = agents_comm_4.step_count
				prey_known_ctr += 1
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_4.prey_probabilities.items() if value == max(agents_comm_4.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			if len(graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos)) <= 2:
				chosen_prey_pos = lcl_prey_pos
			else:
				pre_prob_backup = agents_comm_4.prey_probabilities.copy()
				agents_comm_4.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_4.step_count)
				prey_next_calculated = True				
				max_prey_prob_nodes = []
				max_prey_prob_nodes = [key for key, value in agents_comm_4.prey_probabilities.items() if value == max(agents_comm_4.prey_probabilities.values())]
				print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
				if (len(max_prey_prob_nodes) == 1):
					chosen_prey_pos = max_prey_prob_nodes[0]
				else:
					chosen_prey_pos = random.choice(max_prey_prob_nodes)

			print("Calculating distance to prey("+str(chosen_prey_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_prey[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, chosen_prey_pos))

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				dist_to_pred[val] = len(graph.calc_path(graph.graph_dict, val, graph.pred_pos))
				dist_to_prey[val] = len(graph.calc_path(graph.graph_dict, val, chosen_prey_pos))
				candidate_nodes.append(val)

			print("Agent 3 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_4.decide_node_even(graph.graph_dict, dist_to_prey, dist_to_pred, graph.agent_pos, graph.pred_pos, candidate_nodes)
			print("Agent 3 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_4.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_4.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_4.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_4.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,4)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_4.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			agents_comm_4.step_count += 1

		if agents_comm_4.step_count > self.max_Step_count:
			prey_known_percent = (prey_known_ctr/(agents_comm_4.step_count-1))*100
		else:
			prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
			
		print("Prey Known Percentage: " + str(prey_known_percent))

		return game_over


class Agent5():
	max_Step_count = 50

	def survey_node(self, graph_dict, pred_pos, pred_possible):
		if pred_pos == pred_possible:
			return True
		return False

	def proceed(self, graph):
		game_over = 0
		agents_comm_5 = agents_common()
		prey_5 = Prey()
		pred_5 = Predator()
		agents_comm_5.last_seen_pred = agents_comm_5.step_count
		pred_known_ctr = 0

		while (agents_comm_5.step_count <= self.max_Step_count and game_over == 0):
			survey_result = False
			print("agents_comm_5.step_count: " + str(agents_comm_5.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}

			agents_comm_5.update_pred_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_5.step_count)
			print("Pred Probabilities before survey: " + str(agents_comm_5.pred_probabilities))
			print("Sum of Pred Probabilities before survey: " + str(sum(agents_comm_5.pred_probabilities.values())))

			max_pred_prob_nodes = []
			max_pred_prob_nodes = [key for key, value in agents_comm_5.pred_probabilities.items() if value == max(agents_comm_5.pred_probabilities.values())]
			print("Getting nodes with max pred probability: " + str(max_pred_prob_nodes))
			if (len(max_pred_prob_nodes) == 1):
				pred_possible = max_pred_prob_nodes[0]
			else:
				pred_possible = random.choice(max_pred_prob_nodes)

			print("Surveying Node: " + str(pred_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.pred_pos, pred_possible)
			print("Survey Result: " + str(survey_result))

			agents_comm_5.update_pred_prob_postsurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_5.step_count, pred_possible, survey_result)
			print("Updated Pred probabilities after survey: " + str(agents_comm_5.pred_probabilities))
			print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_5.pred_probabilities.values())))

			if survey_result == True:
				print("Pred at survey Node!")
				lcl_pred_pos = pred_possible
				agents_comm_5.last_seen_Pred = agents_comm_5.step_count
				pred_known_ctr += 1
			else:
				print("Pred NOT at survey Node!")
				max_pred_prob_nodes = [key for key, value in agents_comm_5.pred_probabilities.items() if value == max(agents_comm_5.pred_probabilities.values())]
				if (len(max_pred_prob_nodes) == 1):
					lcl_pred_pos = max_pred_prob_nodes[0]
				else:
					lcl_pred_pos = random.choice(max_pred_prob_nodes)

			print("Calculating distance from agent("+str(graph.agent_pos)+") to prey("+str(graph.prey_pos)+")")
			a = graph.calc_path(graph.graph_dict, graph.agent_pos, graph.prey_pos)
			print(a)
			dist_to_prey[graph.agent_pos] = len(a)
			
			print("Calculating distance from agent("+str(graph.agent_pos)+") to pred("+str(graph.pred_pos)+")")
			b = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_pred_pos)
			print(b)
			dist_to_pred[graph.agent_pos] = len(b)

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
				a = graph.calc_path(graph.graph_dict, val, lcl_pred_pos)
				print(a)
				dist_to_pred[val] = len(a)
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to prey("+str(graph.prey_pos)+")")
				b = graph.calc_path(graph.graph_dict, val, graph.prey_pos)
				print(b)
				dist_to_prey[val] = len(b)
				candidate_nodes.append(val)

			print("Agent 5 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_5.decide_node(dist_to_prey, dist_to_pred, graph.agent_pos, candidate_nodes)
			print("Agent 5 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_5.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_5.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_5.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_5.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_5.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_5.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,5)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_5.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_5.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				return game_over

			agents_comm_5.step_count += 1

		if agents_comm_5.step_count > self.max_Step_count:
			pred_known_percent = (pred_known_ctr/(agents_comm_5.step_count-1))*100
		else:
			pred_known_percent = (pred_known_ctr/agents_comm_5.step_count)*100
			
		print("Pred Known Percentage: " + str(pred_known_percent))

		return game_over


class Agent6():
	max_Step_count = 50

	def survey_node(self, graph_dict, pred_pos, pred_possible):
		if pred_pos == pred_possible:
			return True
		return False

	def proceed(self, graph):
		game_over = 0
		agents_comm_6 = agents_common()
		prey_6 = Prey()
		pred_6 = Predator()
		agents_comm_6.last_seen_pred = agents_comm_6.step_count
		pred_known_ctr = 0

		while (agents_comm_6.step_count <= self.max_Step_count and game_over == 0):
			survey_result = False
			print("agents_comm_6.step_count: " + str(agents_comm_6.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			possible_locs = [graph.prey_pos]
			dist_to_prey = {}
			dist_to_pred = {}

			agents_comm_6.update_pred_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_6.step_count)
			print("Pred Probabilities before survey: " + str(agents_comm_6.pred_probabilities))
			print("Sum of Pred Probabilities before survey: " + str(sum(agents_comm_6.pred_probabilities.values())))

			max_pred_prob_nodes = []
			max_pred_prob_nodes = [key for key, value in agents_comm_6.pred_probabilities.items() if value == max(agents_comm_6.pred_probabilities.values())]
			print("Getting nodes with max pred probability: " + str(max_pred_prob_nodes))
			if (len(max_pred_prob_nodes) == 1):
				pred_possible = max_pred_prob_nodes[0]
			else:
				pred_possible = random.choice(max_pred_prob_nodes)

			print("Surveying Node: " + str(pred_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.pred_pos, pred_possible)
			print("Survey Result: " + str(survey_result))

			agents_comm_6.update_pred_prob_postsurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_6.step_count, pred_possible, survey_result)
			print("Updated Pred probabilities after survey: " + str(agents_comm_6.pred_probabilities))
			print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_6.pred_probabilities.values())))

			if survey_result == True:
				print("Pred at survey Node!")
				lcl_pred_pos = pred_possible
				agents_comm_6.last_seen_Pred = agents_comm_6.step_count
				pred_known_ctr += 1
			else:
				print("Pred NOT at survey Node!")
				max_pred_prob_nodes = [key for key, value in agents_comm_6.pred_probabilities.items() if value == max(agents_comm_6.pred_probabilities.values())]
				if (len(max_pred_prob_nodes) == 1):
					lcl_pred_pos = max_pred_prob_nodes[0]
				else:
					lcl_pred_pos = random.choice(max_pred_prob_nodes)

			#Calculating possible next position of prey
			for i in graph.graph_dict[graph.prey_pos]:
				possible_locs.append(i)
			chosen_prey_pos = random.choice(possible_locs)

			if len(graph.calc_path(graph.graph_dict, graph.agent_pos, graph.prey_pos)) <= 2:
				chosen_prey_pos = graph.prey_pos

			print("Calculating distance from agent("+str(graph.agent_pos)+") to prey("+str(chosen_prey_pos)+")")
			a = graph.calc_path(graph.graph_dict, graph.agent_pos, chosen_prey_pos)
			print(a)
			dist_to_prey[graph.agent_pos] = len(a)
			
			print("Calculating distance from agent("+str(graph.agent_pos)+") to pred("+str(graph.pred_pos)+")")
			b = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_pred_pos)
			print(b)
			dist_to_pred[graph.agent_pos] = len(b)

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
				a = graph.calc_path(graph.graph_dict, val, lcl_pred_pos)
				print(a)
				dist_to_pred[val] = len(a)
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to prey("+str(chosen_prey_pos)+")")
				b = graph.calc_path(graph.graph_dict, val, chosen_prey_pos)
				print(b)
				dist_to_prey[val] = len(b)
				candidate_nodes.append(val)

			print("Agent 6 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_6.decide_node_even(graph.graph_dict, dist_to_prey, dist_to_pred, graph.agent_pos, lcl_pred_pos, candidate_nodes)
			print("Agent 6 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_6.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_6.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_6.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_6.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_6.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_6.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,5)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_6.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_6.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				return game_over

			agents_comm_6.step_count += 1

		if agents_comm_6.step_count > self.max_Step_count:
			pred_known_percent = (pred_known_ctr/(agents_comm_6.step_count-1))*100
		else:
			pred_known_percent = (pred_known_ctr/agents_comm_6.step_count)*100
			
		print("Pred Known Percentage: " + str(pred_known_percent))

		return game_over


class Agent7():
	max_Step_count = 50

	def survey_node(self, graph_dict, pred_pos, prey_pos, survey_node):
		#If survey node contains none, return 0
		#If survey node contains prey, return 1
		#If survey node contains pred, return 2
		#If survey node contains both, return 3

		
		if survey_node == pred_pos and survey_node == prey_pos:
			return 3
		elif survey_node == pred_pos and survey_node != prey_pos:
			return 2
		elif survey_node != pred_pos and survey_node == prey_pos:
			return 1
		else:
			return 0

	def proceed(self, graph):
		game_over = 0
		agents_comm_7 = agents_common()
		prey_7 = Prey()
		pred_7 = Predator()
		agents_comm_7.last_seen_pred = agents_comm_7.step_count
		pred_known_ctr = 0
		prey_known_ctr = 0

		while (agents_comm_7.step_count <= self.max_Step_count and game_over == 0):
			pred_ctr_update = False
			prey_ctr_update = False
			survey_pred_result = False
			survey_prey_result = False
			survey_result = False
			print("agents_comm_7.step_count: " + str(agents_comm_7.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}

			agents_comm_7.update_pred_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_7.step_count)
			print("Pred Probabilities before survey: " + str(agents_comm_7.pred_probabilities))
			print("Sum of Pred Probabilities before survey: " + str(sum(agents_comm_7.pred_probabilities.values())))
			
			agents_comm_7.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_7.step_count)
			print("Prey Probabilities before survey: " + str(agents_comm_7.prey_probabilities))
			print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_7.prey_probabilities.values())))

			if 1 in agents_comm_7.pred_probabilities.values():
				pred_known_ctr += 1
				pred_ctr_update = True

			if 1 in agents_comm_7.prey_probabilities.values():
				prey_known_ctr += 1
				prey_ctr_update = True

			max_pred_prob_nodes = []
			max_pred_prob_nodes = [key for key, value in agents_comm_7.pred_probabilities.items() if value == max(agents_comm_7.pred_probabilities.values())]
			print("Getting nodes with max pred probability: " + str(max_pred_prob_nodes))
			if (len(max_pred_prob_nodes) == 1):
				pred_possible = max_pred_prob_nodes[0]
			else:
				pred_possible = random.choice(max_pred_prob_nodes)

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_7.prey_probabilities.items() if value == max(agents_comm_7.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)

			#To survey for prey when we are more than 50% sure where the predator is
			if agents_comm_7.pred_probabilities[pred_possible] > 0.5:
				print("Pred position pretty certain. Surveying at node " + str(prey_possible) + " for Prey...")
				survey_pos = prey_possible
			else:
				print("Pred position NOT certain. Surveying at node " + str(pred_possible) + " for Pred...")
				survey_pos = pred_possible

			#If survey node contains none, return 0
			#If survey node contains prey, return 1
			#If survey node contains pred, return 2
			#If survey node contains both, return 3
			#print("Surveying Node: " + str(pred_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.pred_pos, graph.prey_pos, survey_pos)
			print("Survey Result: " + str(survey_result))

			if survey_result == 3:
				survey_pred_result = True
				survey_prey_result = True
			elif survey_result == 2:
				survey_pred_result = True
				survey_prey_result = False
			elif survey_result == 1:
				survey_pred_result = False
				survey_prey_result = True
			else:
				survey_pred_result = False
				survey_prey_result = False

			agents_comm_7.update_pred_prob_postsurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_7.step_count, survey_pos, survey_pred_result)
			print("Updated Pred probabilities after survey: " + str(agents_comm_7.pred_probabilities))
			print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_7.pred_probabilities.values())))
				
			agents_comm_7.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_7.step_count, survey_pos, survey_prey_result)
			print("Updated Prey probabilities after survey: " + str(agents_comm_7.prey_probabilities))
			print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_7.prey_probabilities.values())))

			if survey_pred_result == True:
				print("Pred at survey Node!")
				lcl_pred_pos = pred_possible
				agents_comm_7.last_seen_pred = agents_comm_7.step_count
			else:
				print("Pred NOT at survey Node!")
				max_pred_prob_nodes = [key for key, value in agents_comm_7.pred_probabilities.items() if value == max(agents_comm_7.pred_probabilities.values())]
				if (len(max_pred_prob_nodes) == 1):
					lcl_pred_pos = max_pred_prob_nodes[0]
				else:
					lcl_pred_pos = random.choice(max_pred_prob_nodes)

			if survey_prey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_7.last_seen_Prey = agents_comm_7.step_count
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_7.prey_probabilities.items() if value == max(agents_comm_7.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			if pred_ctr_update == False:
				if 1 in agents_comm_7.pred_probabilities.values():
					pred_known_ctr += 1
					pred_ctr_update = True

			if prey_ctr_update == False:
				if 1 in agents_comm_7.prey_probabilities.values():
					prey_known_ctr += 1
					prey_ctr_update = True

			print("Calculating distance from agent("+str(graph.agent_pos)+") to prey("+str(lcl_prey_pos)+")")
			a = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos)
			print(a)
			dist_to_prey[graph.agent_pos] = len(a)
			
			print("Calculating distance from agent("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
			b = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_pred_pos)
			print(b)
			dist_to_pred[graph.agent_pos] = len(b)

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
				a = graph.calc_path(graph.graph_dict, val, lcl_pred_pos)
				print(a)
				dist_to_pred[val] = len(a)
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to prey("+str(lcl_prey_pos)+")")
				b = graph.calc_path(graph.graph_dict, val, lcl_prey_pos)
				print(b)
				dist_to_prey[val] = len(b)
				candidate_nodes.append(val)

			print("Agent 7 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_7.decide_node(dist_to_prey, dist_to_pred, graph.agent_pos, candidate_nodes)
			print("Agent 7 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_7.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_7.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_7.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_7.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,7)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_7.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			agents_comm_7.step_count += 1

		if agents_comm_7.step_count > self.max_Step_count:
			pred_known_percent = (pred_known_ctr/(agents_comm_7.step_count-1))*100
			prey_known_percent = (prey_known_ctr/(agents_comm_7.step_count-1))*100
		else:
			pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
			prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
			
		print("Pred Known Percentage: " + str(pred_known_percent))
		print("Prey Known Percentage: " + str(prey_known_percent))

		return game_over


class Agent8():
	max_Step_count = 50

	def survey_node(self, graph_dict, pred_pos, prey_pos, survey_node):
		#If survey node contains none, return 0
		#If survey node contains prey, return 1
		#If survey node contains pred, return 2
		#If survey node contains both, return 3
		if survey_node == pred_pos and survey_node == prey_pos:
			return 3
		elif survey_node == pred_pos and survey_node != prey_pos:
			return 2
		elif survey_node != pred_pos and survey_node == prey_pos:
			return 1
		else:
			return 0


	def proceed(self, graph):
		game_over = 0
		agents_comm_8 = agents_common()
		prey_8 = Prey()
		pred_8 = Predator()
		agents_comm_8.last_seen_pred = agents_comm_8.step_count
		prey_prob_backup = {}
		prey_next_calculated = False
		prey_known_ctr = 0
		pred_known_ctr = 0

		while (agents_comm_8.step_count <= self.max_Step_count and game_over == 0):
			pred_ctr_update = False
			prey_ctr_update = False
			survey_pred_result = False
			survey_prey_result = False
			survey_result = False
			print("agents_comm_8.step_count: " + str(agents_comm_8.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}

			agents_comm_8.update_pred_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count)
			print("Pred Probabilities before survey: " + str(agents_comm_8.pred_probabilities))
			print("Sum of Pred Probabilities before survey: " + str(sum(agents_comm_8.pred_probabilities.values())))
			
			if (prey_next_calculated == False):
				agents_comm_8.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count)
				print("Prey Probabilities before survey: " + str(agents_comm_8.prey_probabilities))
				print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_8.prey_probabilities.values())))
			else:
				print("Skipping calculating prey probabilities because it was computer in the earlier iteration")
				prey_next_calculated = False

			if 1 in agents_comm_8.pred_probabilities.values():
				pred_known_ctr += 1
				pred_ctr_update = True

			if 1 in agents_comm_8.prey_probabilities.values():
				prey_known_ctr += 1
				prey_ctr_update = True

			max_pred_prob_nodes = []
			max_pred_prob_nodes = [key for key, value in agents_comm_8.pred_probabilities.items() if value == max(agents_comm_8.pred_probabilities.values())]
			print("Getting nodes with max pred probability: " + str(max_pred_prob_nodes))
			if (len(max_pred_prob_nodes) == 1):
				pred_possible = max_pred_prob_nodes[0]
			else:
				pred_possible = random.choice(max_pred_prob_nodes)

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_8.prey_probabilities.items() if value == max(agents_comm_8.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)

			#To survey for prey when we are more than 50% sure where the predator is
			if agents_comm_8.pred_probabilities[pred_possible] > 0.5:
				print("Pred position pretty certain. Surveying at node " + str(prey_possible) + " for Prey...")
				survey_pos = prey_possible
			else:
				print("Pred position NOT certain. Surveying at node " + str(pred_possible) + " for Pred...")
				survey_pos = pred_possible

			print("Surveying Node: " + str(pred_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.pred_pos, graph.prey_pos, survey_pos)
			print("Survey Result: " + str(survey_result))

			if survey_result == 3:
				survey_pred_result = True
				survey_prey_result = True
			elif survey_result == 2:
				survey_pred_result = True
				survey_prey_result = False
			elif survey_result == 1:
				survey_pred_result = False
				survey_prey_result = True
			else:
				survey_pred_result = False
				survey_prey_result = False

			agents_comm_8.update_pred_prob_postsurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count, survey_pos, survey_pred_result)
			print("Updated Pred probabilities after survey: " + str(agents_comm_8.pred_probabilities))
			print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_8.pred_probabilities.values())))
				
			agents_comm_8.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_8.step_count, survey_pos, survey_prey_result)
			print("Updated Prey probabilities after survey: " + str(agents_comm_8.prey_probabilities))
			print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_8.prey_probabilities.values())))

			if survey_pred_result == True:
				print("Pred at survey Node!")
				lcl_pred_pos = pred_possible
				agents_comm_8.last_seen_pred = agents_comm_8.step_count
			else:
				print("Pred NOT at survey Node!")
				max_pred_prob_nodes = [key for key, value in agents_comm_8.pred_probabilities.items() if value == max(agents_comm_8.pred_probabilities.values())]
				if (len(max_pred_prob_nodes) == 1):
					lcl_pred_pos = max_pred_prob_nodes[0]
				else:
					lcl_pred_pos = random.choice(max_pred_prob_nodes)

			if survey_prey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_8.last_seen_Prey = agents_comm_8.step_count
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_8.prey_probabilities.items() if value == max(agents_comm_8.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			if len(graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos)) <= 2:
				chosen_prey_pos = lcl_prey_pos
			else:
				prey_prob_backup = agents_comm_8.prey_probabilities.copy()
				agents_comm_8.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count)
				prey_next_calculated = True				
				max_prey_prob_nodes = []
				max_prey_prob_nodes = [key for key, value in agents_comm_8.prey_probabilities.items() if value == max(agents_comm_8.prey_probabilities.values())]
				print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
				if (len(max_prey_prob_nodes) == 1):
					chosen_prey_pos = max_prey_prob_nodes[0]
				else:
					chosen_prey_pos = random.choice(max_prey_prob_nodes)

			if pred_ctr_update == False:
				if 1 in agents_comm_8.pred_probabilities.values():
					pred_known_ctr += 1
					pred_ctr_update = True

			if prey_ctr_update == False:
				if 1 in agents_comm_8.prey_probabilities.values():
					prey_known_ctr += 1
					prey_ctr_update = True

			print("Calculating distance from agent("+str(graph.agent_pos)+") to prey("+str(chosen_prey_pos)+")")
			a = graph.calc_path(graph.graph_dict, graph.agent_pos, chosen_prey_pos)
			print(a)
			dist_to_prey[graph.agent_pos] = len(a)
			
			print("Calculating distance from agent("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
			b = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_pred_pos)
			print(b)
			dist_to_pred[graph.agent_pos] = len(b)

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
				a = graph.calc_path(graph.graph_dict, val, lcl_pred_pos)
				print(a)
				dist_to_pred[val] = len(a)
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to prey("+str(chosen_prey_pos)+")")
				b = graph.calc_path(graph.graph_dict, val, chosen_prey_pos)
				print(b)
				dist_to_prey[val] = len(b)
				candidate_nodes.append(val)

			print("Agent 8 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_8.decide_node_even(graph.graph_dict, dist_to_prey, dist_to_pred, graph.agent_pos, lcl_pred_pos, candidate_nodes)
			print("Agent 8 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_8.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_8.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_8.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_8.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,8)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_8.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			agents_comm_8.step_count += 1

		if agents_comm_8.step_count > self.max_Step_count:
			pred_known_percent = (pred_known_ctr/(agents_comm_8.step_count-1))*100
			prey_known_percent = (prey_known_ctr/(agents_comm_8.step_count-1))*100
		else:
			pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
			prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
			
		print("Pred Known Percentage: " + str(pred_known_percent))
		print("Prey Known Percentage: " + str(prey_known_percent))

		return game_over


class Agent7_b():
	max_Step_count = 50

	def survey_node(self, graph_dict, pred_pos, prey_pos, survey_node):
		#If survey node contains none, return 0
		#If survey node contains prey, return 1 or 0
		#If survey node contains pred, return 2 or 0
		#If survey node contains both, return 3 or 0
		x = random.random()
		if survey_node == pred_pos and survey_node == prey_pos:
			if x <= 0.1:
				print("False Negative")
				return 0
			else:
				return 3
		elif survey_node == pred_pos and survey_node != prey_pos:
			if x <= 0.1:
				print("False Negative")
				return 0
			else:
				return 2
		elif survey_node != pred_pos and survey_node == prey_pos:
			if x <= 0.1:
				print("False Negative")
				return 0
			else:
				return 1
		else:
			return 0

	def proceed(self, graph):
		game_over = 0
		agents_comm_7 = agents_common()
		prey_7 = Prey()
		pred_7 = Predator()
		agents_comm_7.last_seen_pred = agents_comm_7.step_count
		prey_known_ctr = 0
		pred_known_ctr = 0

		while (agents_comm_7.step_count <= self.max_Step_count and game_over == 0):
			survey_pred_result = False
			survey_prey_result = False
			survey_result = False
			pred_ctr_update = False
			prey_ctr_update = False
			print("agents_comm_7.step_count: " + str(agents_comm_7.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}

			agents_comm_7.update_pred_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_7.step_count)
			print("Pred Probabilities before survey: " + str(agents_comm_7.pred_probabilities))
			print("Sum of Pred Probabilities before survey: " + str(sum(agents_comm_7.pred_probabilities.values())))
			
			agents_comm_7.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_7.step_count)
			print("Prey Probabilities before survey: " + str(agents_comm_7.prey_probabilities))
			print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_7.prey_probabilities.values())))

			if 1 in agents_comm_7.pred_probabilities.values():
				pred_known_ctr += 1
				pred_ctr_update = True

			if 1 in agents_comm_7.prey_probabilities.values():
				prey_known_ctr += 1
				prey_ctr_update = True

			max_pred_prob_nodes = []
			max_pred_prob_nodes = [key for key, value in agents_comm_7.pred_probabilities.items() if value == max(agents_comm_7.pred_probabilities.values())]
			print("Getting nodes with max pred probability: " + str(max_pred_prob_nodes))
			if (len(max_pred_prob_nodes) == 1):
				pred_possible = max_pred_prob_nodes[0]
			else:
				pred_possible = random.choice(max_pred_prob_nodes)

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_7.prey_probabilities.items() if value == max(agents_comm_7.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)

			#To survey for prey when we are more than 50% sure where the predator is
			if agents_comm_7.pred_probabilities[pred_possible] > 0.5:
				print("Pred position pretty certain. Surveying at node " + str(prey_possible) + " for Prey...")
				survey_pos = prey_possible
			else:
				print("Pred position NOT certain. Surveying at node " + str(pred_possible) + " for Pred...")
				survey_pos = pred_possible
			
			print("Surveying Node: " + str(pred_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.pred_pos, graph.prey_pos, survey_pos)
			print("Survey Result: " + str(survey_result))

			if survey_result == 3:
				survey_pred_result = True
				survey_prey_result = True
			elif survey_result == 2:
				survey_pred_result = True
				survey_prey_result = False
			elif survey_result == 1:
				survey_pred_result = False
				survey_prey_result = True
			else:
				survey_pred_result = False
				survey_prey_result = False

			if survey_result == 0:
				print("Inside else. Agent 7_b logic incoming...")
				agents_comm_7.update_pred_prob_postsurvey_fn(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_7.step_count, survey_pos, survey_pred_result)
				print("Updated Pred probabilities after survey: " + str(agents_comm_7.pred_probabilities))
				print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_7.pred_probabilities.values())))
				
				agents_comm_7.update_prey_prob_postsurvey_fn(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_7.step_count, survey_pos, survey_prey_result)
				print("Updated Prey probabilities after survey: " + str(agents_comm_7.prey_probabilities))
				print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_7.prey_probabilities.values())))
			else:
				print("Inside else. Going Agent 7 way...")
				agents_comm_7.update_pred_prob_postsurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_7.step_count, survey_pos, survey_pred_result)
				print("Updated Pred probabilities after survey: " + str(agents_comm_7.pred_probabilities))
				print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_7.pred_probabilities.values())))

				agents_comm_7.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_7.step_count, survey_pos, survey_prey_result)
				print("Updated Prey probabilities after survey: " + str(agents_comm_7.prey_probabilities))
				print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_7.prey_probabilities.values())))

			if survey_pred_result == True:
				print("Pred at survey Node!")
				lcl_pred_pos = pred_possible
				agents_comm_7.last_seen_pred = agents_comm_7.step_count
			else:
				print("Pred NOT at survey Node!")
				max_pred_prob_nodes = [key for key, value in agents_comm_7.pred_probabilities.items() if value == max(agents_comm_7.pred_probabilities.values())]
				if (len(max_pred_prob_nodes) == 1):
					lcl_pred_pos = max_pred_prob_nodes[0]
				else:
					lcl_pred_pos = random.choice(max_pred_prob_nodes)

			if survey_prey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_7.last_seen_Prey = agents_comm_7.step_count
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_7.prey_probabilities.items() if value == max(agents_comm_7.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			if pred_ctr_update == False:
				if 1 in agents_comm_7.pred_probabilities.values():
					pred_known_ctr += 1
					pred_ctr_update = True

			if prey_ctr_update == False:
				if 1 in agents_comm_7.prey_probabilities.values():
					prey_known_ctr += 1
					prey_ctr_update = True

			print("Calculating distance from agent("+str(graph.agent_pos)+") to prey("+str(lcl_prey_pos)+")")
			a = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos)
			print(a)
			dist_to_prey[graph.agent_pos] = len(a)
			
			print("Calculating distance from agent("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
			b = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_pred_pos)
			print(b)
			dist_to_pred[graph.agent_pos] = len(b)

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
				a = graph.calc_path(graph.graph_dict, val, lcl_pred_pos)
				print(a)
				dist_to_pred[val] = len(a)
				
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to prey("+str(lcl_prey_pos)+")")
				b = graph.calc_path(graph.graph_dict, val, lcl_prey_pos)
				print(b)
				dist_to_prey[val] = len(b)

				candidate_nodes.append(val)

			print("Agent 7_b position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_7.decide_node(dist_to_prey, dist_to_pred, graph.agent_pos, candidate_nodes)
			print("Agent 7_b position after update: "+str(graph.agent_pos))
			game_over = agents_comm_7.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_7.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_7.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_7.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,7)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_7.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			agents_comm_7.step_count += 1

		if agents_comm_7.step_count > self.max_Step_count:
			pred_known_percent = (pred_known_ctr/(agents_comm_7.step_count-1))*100
			prey_known_percent = (prey_known_ctr/(agents_comm_7.step_count-1))*100
		else:
			pred_known_percent = (pred_known_ctr/agents_comm_7.step_count)*100
			prey_known_percent = (prey_known_ctr/agents_comm_7.step_count)*100
			
		print("Pred Known Percentage: " + str(pred_known_percent))
		print("Prey Known Percentage: " + str(prey_known_percent))

		return game_over


class Agent8_b():
	max_Step_count = 50

	def survey_node(self, graph_dict, pred_pos, prey_pos, survey_node):
		#If survey node contains none, return 0
		#If survey node contains prey, return 1 or 0
		#If survey node contains pred, return 2 or 0
		#If survey node contains both, return 3 or 0

		x = random.random()
		if survey_node == pred_pos and survey_node == prey_pos:
			if x <= 0.1:
				print("False Negative")
				return 0
			else:
				return 3
		elif survey_node == pred_pos and survey_node != prey_pos:
			if x <= 0.1:
				print("False Negative")
				return 0
			else:
				return 2
		elif survey_node != pred_pos and survey_node == prey_pos:
			if x <= 0.1:
				print("False Negative")
				return 0
			else:
				return 1
		else:
			return 0

	def proceed(self, graph):
		game_over = 0
		agents_comm_8 = agents_common()
		prey_8 = Prey()
		pred_8 = Predator()
		agents_comm_8.last_seen_pred = agents_comm_8.step_count
		prey_prob_backup = {}
		prey_next_calculated = False
		prey_known_ctr = 0
		pred_known_ctr = 0

		while (agents_comm_8.step_count <= self.max_Step_count and game_over == 0):
			pred_ctr_update = False
			prey_ctr_update = False
			survey_pred_result = False
			survey_prey_result = False
			survey_result = False
			print("agents_comm_8.step_count: " + str(agents_comm_8.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}

			agents_comm_8.update_pred_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count)
			print("Pred Probabilities before survey: " + str(agents_comm_8.pred_probabilities))
			print("Sum of Pred Probabilities before survey: " + str(sum(agents_comm_8.pred_probabilities.values())))
			
			if (prey_next_calculated == False):
				agents_comm_8.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count)
				print("Prey Probabilities before survey: " + str(agents_comm_8.prey_probabilities))
				print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_8.prey_probabilities.values())))
			else:
				prey_next_calculated = False

			max_pred_prob_nodes = []
			max_pred_prob_nodes = [key for key, value in agents_comm_8.pred_probabilities.items() if value == max(agents_comm_8.pred_probabilities.values())]
			print("Getting nodes with max pred probability: " + str(max_pred_prob_nodes))
			if (len(max_pred_prob_nodes) == 1):
				pred_possible = max_pred_prob_nodes[0]
			else:
				pred_possible = random.choice(max_pred_prob_nodes)

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_8.prey_probabilities.items() if value == max(agents_comm_8.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)

			if 1 in agents_comm_8.pred_probabilities.values():
				pred_known_ctr += 1
				pred_ctr_update = True

			if 1 in agents_comm_8.prey_probabilities.values():
				prey_known_ctr += 1
				prey_ctr_update = True

			#To survey for prey when we are more than 50% sure where the predator is
			if agents_comm_8.pred_probabilities[pred_possible] > 0.5:
				print("Pred position pretty certain. Surveying at node " + str(prey_possible) + " for Prey...")
				survey_pos = prey_possible
			else:
				print("Pred position NOT certain. Surveying at node " + str(pred_possible) + " for Pred...")
				survey_pos = pred_possible
			
			print("Surveying Node: " + str(pred_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.pred_pos, graph.prey_pos, survey_pos)
			print("Survey Result: " + str(survey_result))

			if survey_result == 3:
				survey_pred_result = True
				survey_prey_result = True
			elif survey_result == 2:
				survey_pred_result = True
				survey_prey_result = False
			elif survey_result == 1:
				survey_pred_result = False
				survey_prey_result = True
			else:
				survey_pred_result = False
				survey_prey_result = False

			if survey_result == 0:
				print("Inside else. Agent 8_b logic incoming...")
				agents_comm_8.update_pred_prob_postsurvey_fn(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count, survey_pos, survey_pred_result)
				print("Updated Pred probabilities after survey: " + str(agents_comm_8.pred_probabilities))
				print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_8.pred_probabilities.values())))
				
				agents_comm_8.update_prey_prob_postsurvey_fn(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_8.step_count, survey_pos, survey_prey_result)
				print("Updated Prey probabilities after survey: " + str(agents_comm_8.prey_probabilities))
				print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_8.prey_probabilities.values())))
			else:
				print("Inside else. Going Agent 8 way...")
				agents_comm_8.update_pred_prob_postsurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count, survey_pos, survey_pred_result)
				print("Updated Pred probabilities after survey: " + str(agents_comm_8.pred_probabilities))
				print("Sum of updated Pred probabilities after survey: " + str(sum(agents_comm_8.pred_probabilities.values())))

				agents_comm_8.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_8.step_count, survey_pos, survey_prey_result)
				print("Updated Prey probabilities after survey: " + str(agents_comm_8.prey_probabilities))
				print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_8.prey_probabilities.values())))

			if survey_pred_result == True:
				print("Pred at survey Node!")
				lcl_pred_pos = pred_possible
				agents_comm_8.last_seen_pred = agents_comm_8.step_count
			else:
				print("Pred NOT at survey Node!")
				max_pred_prob_nodes = [key for key, value in agents_comm_8.pred_probabilities.items() if value == max(agents_comm_8.pred_probabilities.values())]
				if (len(max_pred_prob_nodes) == 1):
					lcl_pred_pos = max_pred_prob_nodes[0]
				else:
					lcl_pred_pos = random.choice(max_pred_prob_nodes)

			if survey_prey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_8.last_seen_Prey = agents_comm_8.step_count
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_8.prey_probabilities.items() if value == max(agents_comm_8.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			if len(graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos)) <= 2:
				chosen_prey_pos = lcl_prey_pos
			else:
				prey_prob_backup = agents_comm_8.prey_probabilities.copy()
				agents_comm_8.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_8.step_count)
				prey_next_calculated = True				
				max_prey_prob_nodes = []
				max_prey_prob_nodes = [key for key, value in agents_comm_8.prey_probabilities.items() if value == max(agents_comm_8.prey_probabilities.values())]
				print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
				if (len(max_prey_prob_nodes) == 1):
					chosen_prey_pos = max_prey_prob_nodes[0]
				else:
					chosen_prey_pos = random.choice(max_prey_prob_nodes)

			if pred_ctr_update == False:
				if 1 in agents_comm_8.pred_probabilities.values():
					pred_known_ctr += 1
					pred_ctr_update = True

			if prey_ctr_update == False:
				if 1 in agents_comm_8.prey_probabilities.values():
					prey_known_ctr += 1
					prey_ctr_update = True

			print("Calculating distance from agent("+str(graph.agent_pos)+") to prey("+str(chosen_prey_pos)+")")
			a = graph.calc_path(graph.graph_dict, graph.agent_pos, chosen_prey_pos)
			print(a)
			dist_to_prey[graph.agent_pos] = len(a)
			
			print("Calculating distance from agent("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
			b = graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_pred_pos)
			print(b)
			dist_to_pred[graph.agent_pos] = len(b)

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to pred("+str(lcl_pred_pos)+")")
				a = graph.calc_path(graph.graph_dict, val, lcl_pred_pos)
				print(a)
				dist_to_pred[val] = len(a)
				
				print("Calculating distance from neighbour("+str(val)+") of current("+str(graph.agent_pos)+") to prey("+str(chosen_prey_pos)+")")
				b = graph.calc_path(graph.graph_dict, val, chosen_prey_pos)
				print(b)
				dist_to_prey[val] = len(b)
				
				candidate_nodes.append(val)

			print("Agent 8_b position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_8.decide_node_even(graph.graph_dict, dist_to_prey, dist_to_pred, graph.agent_pos, lcl_pred_pos, candidate_nodes)
			print("Agent 8_b position after update: "+str(graph.agent_pos))
			game_over = agents_comm_8.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_8.move_prey(graph.prey_pos, graph.connectivity_matrix)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_8.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_8.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos,8)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_8.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
				prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
				print("Pred Known Percentage: " + str(pred_known_percent))
				print("Prey Known Percentage: " + str(prey_known_percent))
				return game_over

			agents_comm_8.step_count += 1

		if agents_comm_8.step_count > self.max_Step_count:
			pred_known_percent = (pred_known_ctr/(agents_comm_8.step_count-1))*100
			prey_known_percent = (prey_known_ctr/(agents_comm_8.step_count-1))*100
		else:
			pred_known_percent = (pred_known_ctr/agents_comm_8.step_count)*100
			prey_known_percent = (prey_known_ctr/agents_comm_8.step_count)*100
			
		print("Pred Known Percentage: " + str(pred_known_percent))
		print("Prey Known Percentage: " + str(prey_known_percent))

		return game_over

agnt_tst_record = []
agnt_1_record = []
agnt_2_record = []
agnt_3_record = []
agnt_4_record = []
agnt_5_record = []
agnt_6_record = []
agnt_7_record = []
agnt_8_record = []
agnt_7_b_record = []
agnt_8_b_record = []

run_test = False
run_1 = False
run_2 = False
run_3 = False
run_4 = False
run_5 = False
run_6 = False
run_7 = False
run_8 = False
run_7b = False
run_8b = True
#KJ

##########AGENT TEST START##########
if run_test == True:
	agnt_tst_survivability = []
	agnt_test_final_fail = 0
	agnt_test_final_suspend = 0
	for i in range(100):
		graph_tst = Graph()
		graph_tst.create_graph()
		for j in range(30):
			graph_tst.initialize_positions()
			agent_tst = test_agent()
			#graph_tst = graph.copy()
			result = agent_tst.proceed(graph_tst.graph_dict, graph_tst.connectivity_matrix, graph_tst.prey_pos, graph_tst.pred_pos, graph_tst.agent_pos)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_tst_record.append(0)
			elif result == 1:
				print("Agent 7 caught the Prey! Agent 7 won!")
				print("1")
				agnt_tst_record.append(1)
			else:
				print("Predator caught the agent! Agent 7 lost!")
				print("2")
				agnt_tst_record.append(2)
		agnt_tst_sum_0 = 0
		agnt_tst_sum_1 = 0
		agnt_tst_sum_2 = 0

		for a in range(30):
			if agnt_tst_record[(i*30)+a] == 0:
				agnt_tst_sum_0 += 1
			elif agnt_tst_record[(i*30)+a] == 1:
				agnt_tst_sum_1 += 1
			elif agnt_tst_record[(i*30)+a] == 2:
				agnt_tst_sum_2 += 1
		agnt_test_final_suspend += agnt_tst_sum_0
		agnt_test_final_fail += agnt_tst_sum_2
		agnt_tst_survivability.append((agnt_tst_sum_1/30)*100)

	agnt_test_final_survivability = sum(agnt_tst_survivability)/100
	print("agnt_tst_survivability: " + str(agnt_tst_survivability))
	
	print("Agent-Test Total iterations: " + str(len(agnt_tst_record)))
	print("Agent-Test Count when Game Suspended: " + str(agnt_test_final_suspend))
	print("Agent-Test Count when Agent Lost: " + str(agnt_test_final_fail))
	print("Agent-Test Count when Agent Won: " + str(len(agnt_tst_record) - (agnt_test_final_fail + agnt_test_final_suspend)))
	print("Agent-Test Survivability: " + str(agnt_test_final_survivability))
##########AGENT TEST END##########

##########AGENT 1 START##########
if run_1 == True:
	agnt_1_survivability = []
	agnt_1_final_fail = 0
	agnt_1_final_suspend = 0
	for i in range(100):
		graph1 = Graph()
		graph1.create_graph()
		for j in range(30):
			graph1.initialize_positions()
			agent_1 = Agent1()
			result = agent_1.proceed(graph1.graph_dict, graph1.connectivity_matrix, graph1.prey_pos, graph1.pred_pos, graph1.agent_pos)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_1_record.append(0)
			elif result == 1:
				print("Agent 1 caught the Prey! Agent 1 won!")
				print("1")
				agnt_1_record.append(1)
			else:
				print("Predator caught the agent! Agent 1 lost!")
				print("2")
				agnt_1_record.append(2)
		agnt_1_sum_0 = 0
		agnt_1_sum_1 = 0
		agnt_1_sum_2 = 0

		for a in range(30):
			if agnt_1_record[(i*30)+a] == 0:
				agnt_1_sum_0 += 1
			elif agnt_1_record[(i*30)+a] == 1:
				agnt_1_sum_1 += 1
			elif agnt_1_record[(i*30)+a] == 2:
				agnt_1_sum_2 += 1
		agnt_1_final_suspend += agnt_1_sum_0
		agnt_1_final_fail += agnt_1_sum_2
		agnt_1_survivability.append((agnt_1_sum_1/30)*100)

	agnt_1_final_survivability = sum(agnt_1_survivability)/100
	print("agnt_1_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 1 Total iterations: " + str(len(agnt_1_record)))
	print("Agent 1 Count when Game Suspended: " + str(agnt_1_final_suspend))
	print("Agent 1 Count when Agent Lost: " + str(agnt_1_final_fail))
	print("Agent 1 Count when Agent Won: " + str(len(agnt_1_record) - (agnt_1_final_fail + agnt_1_final_suspend)))
	print("Agent 1 Survivability: " + str(agnt_1_final_survivability))
##########AGENT 1 END##########

##########AGENT 2 START##########
if run_2 == True:
	agnt_2_survivability = []
	agnt_2_final_fail = 0
	agnt_2_final_suspend = 0
	for i in range(100):
		graph_2 = Graph()
		graph_2.create_graph()
		for j in range(30):
			graph_2.initialize_positions()
			agent_2 = Agent2()
			result = agent_2.proceed(graph_2.graph_dict, graph_2.connectivity_matrix, graph_2.prey_pos, graph_2.pred_pos, graph_2.agent_pos)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_2_record.append(0)
			elif result == 1:
				print("Agent 7 caught the Prey! Agent 7 won!")
				print("1")
				agnt_2_record.append(1)
			else:
				print("Predator caught the agent! Agent 7 lost!")
				print("2")
				agnt_2_record.append(2)
		agnt_2_sum_0 = 0
		agnt_2_sum_1 = 0
		agnt_2_sum_2 = 0

		for a in range(30):
			if agnt_2_record[(i*30)+a] == 0:
				agnt_2_sum_0 += 1
			elif agnt_2_record[(i*30)+a] == 1:
				agnt_2_sum_1 += 1
			elif agnt_2_record[(i*30)+a] == 2:
				agnt_2_sum_2 += 1
		agnt_2_final_suspend += agnt_2_sum_0
		agnt_2_final_fail += agnt_2_sum_2
		agnt_2_survivability.append((agnt_2_sum_1/30)*100)

	agnt_2_final_survivability = sum(agnt_2_survivability)/100
	print("agnt_2_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 2 Total iterations: " + str(len(agnt_2_record)))
	print("Agent 2 Count when Game Suspended: " + str(agnt_2_final_suspend))
	print("Agent 2 Count when Agent Lost: " + str(agnt_2_final_fail))
	print("Agent 2 Count when Agent Won: " + str(len(agnt_2_record) - (agnt_2_final_fail + agnt_2_final_suspend)))
	print("Agent 2 Survivability: " + str(agnt_2_final_survivability))
##########AGENT 2 END##########

##########AGENT 3 START##########
if run_3 == True:
	agnt_3_survivability = []
	agnt_3_final_fail = 0
	agnt_3_final_suspend = 0
	for i in range(100):
		graph3 = Graph()
		graph3.create_graph()
		for j in range(30):
			graph3.initialize_positions()
			agent_3 = Agent3()
			result = agent_3.proceed(graph3)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_3_record.append(0)
			elif result == 1:
				print("Agent 3 caught the Prey! Agent 3 won!")
				print("1")
				agnt_3_record.append(1)
			else:
				print("Predator caught the agent! Agent 3 lost!")
				print("2")
				agnt_3_record.append(2)
		agnt_3_sum_0 = 0
		agnt_3_sum_1 = 0
		agnt_3_sum_2 = 0

		for a in range(30):
			if agnt_3_record[(i*30)+a] == 0:
				agnt_3_sum_0 += 1
			elif agnt_3_record[(i*30)+a] == 1:
				agnt_3_sum_1 += 1
			elif agnt_3_record[(i*30)+a] == 2:
				agnt_3_sum_2 += 1
		agnt_3_final_suspend += agnt_3_sum_0
		agnt_3_final_fail += agnt_3_sum_2
		agnt_3_survivability.append((agnt_3_sum_1/30)*100)
	
	agnt_3_final_survivability = sum(agnt_3_survivability)/100
	print("agnt_3_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 3 Total iterations: " + str(len(agnt_3_record)))
	print("Agent 3 Count when Game Suspended: " + str(agnt_3_final_suspend))
	print("Agent 3 Count when Agent Lost: " + str(agnt_3_final_fail))
	print("Agent 3 Count when Agent Won: " + str(len(agnt_3_record) - (agnt_3_final_fail + agnt_3_final_suspend)))
	print("Agent 3 Survivability: " + str(agnt_3_final_survivability))
##########AGENT 3 END##########

##########AGENT 4 START##########
if run_4 == True:
	agnt_4_survivability = []
	agnt_4_final_fail = 0
	agnt_4_final_suspend = 0
	for i in range(100):
		graph_4 = Graph()
		graph_4.create_graph()
		for j in range(30):
			graph_4.initialize_positions()
			agent_4 = Agent4()
			result = agent_4.proceed(graph_4)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_4_record.append(0)
			elif result == 1:
				print("Agent 7 caught the Prey! Agent 7 won!")
				print("1")
				agnt_4_record.append(1)
			else:
				print("Predator caught the agent! Agent 7 lost!")
				print("2")
				agnt_4_record.append(2)
		agnt_4_sum_0 = 0
		agnt_4_sum_1 = 0
		agnt_4_sum_2 = 0

		for a in range(30):
			if agnt_4_record[(i*30)+a] == 0:
				agnt_4_sum_0 += 1
			elif agnt_4_record[(i*30)+a] == 1:
				agnt_4_sum_1 += 1
			elif agnt_4_record[(i*30)+a] == 2:
				agnt_4_sum_2 += 1
		agnt_4_final_suspend += agnt_4_sum_0
		agnt_4_final_fail += agnt_4_sum_2
		agnt_4_survivability.append((agnt_4_sum_1/30)*100)

	agnt_4_final_survivability = sum(agnt_4_survivability)/100
	print("agnt_4_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 4 Total iterations: " + str(len(agnt_4_record)))
	print("Agent 4 Count when Game Suspended: " + str(agnt_4_final_suspend))
	print("Agent 4 Count when Agent Lost: " + str(agnt_4_final_fail))
	print("Agent 4 Count when Agent Won: " + str(len(agnt_4_record) - (agnt_4_final_fail + agnt_4_final_suspend)))
	print("Agent 4 Survivability: " + str(agnt_4_final_survivability))
##########AGENT 4 END##########

##########AGENT 5 START##########
if run_5 == True:
	agnt_5_survivability = []
	agnt_5_final_fail = 0
	agnt_5_final_suspend = 0
	for i in range(100):
		graph5 = Graph()
		graph5.create_graph()
		for j in range(30):
			graph5.initialize_positions()
			agent_5 = Agent5()
			result = agent_5.proceed(graph5)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_5_record.append(0)
			elif result == 1:
				print("Agent 5 caught the Prey! Agent 5 won!")
				print("1")
				agnt_5_record.append(1)
			else:
				print("Predator caught the agent! Agent 5 lost!")
				print("2")
				agnt_5_record.append(2)
		agnt_5_sum_0 = 0
		agnt_5_sum_1 = 0
		agnt_5_sum_2 = 0

		for a in range(30):
			if agnt_5_record[(i*30)+a] == 0:
				agnt_5_sum_0 += 1
			elif agnt_5_record[(i*30)+a] == 1:
				agnt_5_sum_1 += 1
			elif agnt_5_record[(i*30)+a] == 2:
				agnt_5_sum_2 += 1
		agnt_5_final_suspend += agnt_5_sum_0
		agnt_5_final_fail += agnt_5_sum_2
		agnt_5_survivability.append((agnt_5_sum_1/30)*100)

	agnt_5_final_survivability = sum(agnt_5_survivability)/100
	print("agnt_5_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 5 Total iterations: " + str(len(agnt_5_record)))
	print("Agent 5 Count when Game Suspended: " + str(agnt_5_final_suspend))
	print("Agent 5 Count when Agent Lost: " + str(agnt_5_final_fail))
	print("Agent 5 Count when Agent Won: " + str(len(agnt_5_record) - (agnt_5_final_fail + agnt_5_final_suspend)))
	print("Agent 5 Survivability: " + str(agnt_5_final_survivability))
##########AGENT 5 END##########

##########AGENT 6 START##########
if run_6 == True:
	agnt_6_survivability = []
	agnt_6_final_fail = 0
	agnt_6_final_suspend = 0
	for i in range(100):
		graph_6 = Graph()
		graph_6.create_graph()
		for j in range(30):
			graph_6.initialize_positions()
			agent_6 = Agent6()
			result = agent_6.proceed(graph_6)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_6_record.append(0)
			elif result == 1:
				print("Agent 5 caught the Prey! Agent 5 won!")
				print("1")
				agnt_6_record.append(1)
			else:
				print("Predator caught the agent! Agent 5 lost!")
				print("2")
				agnt_6_record.append(2)
		agnt_6_sum_0 = 0
		agnt_6_sum_1 = 0
		agnt_6_sum_2 = 0

		for a in range(30):
			if agnt_6_record[(i*30)+a] == 0:
				agnt_6_sum_0 += 1
			elif agnt_6_record[(i*30)+a] == 1:
				agnt_6_sum_1 += 1
			elif agnt_6_record[(i*30)+a] == 2:
				agnt_6_sum_2 += 1
		agnt_6_final_suspend += agnt_6_sum_0
		agnt_6_final_fail += agnt_6_sum_2
		agnt_6_survivability.append((agnt_6_sum_1/30)*100)

	agnt_6_final_survivability = sum(agnt_6_survivability)/100
	print("agnt_6_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 6 Total iterations: " + str(len(agnt_6_record)))
	print("Agent 6 Count when Game Suspended: " + str(agnt_6_final_suspend))
	print("Agent 6 Count when Agent Lost: " + str(agnt_6_final_fail))
	print("Agent 6 Count when Agent Won: " + str(len(agnt_6_record) - (agnt_6_final_fail + agnt_6_final_suspend)))
	print("Agent 6 Survivability: " + str(agnt_6_final_survivability))
##########AGENT 6 END##########

##########AGENT 7 START##########
if run_7 == True:
	agnt_7_survivability = []
	agnt_7_final_fail = 0
	agnt_7_final_suspend = 0
	for i in range(100):
		graph7 = Graph()
		graph7.create_graph()
		for j in range(30):
			graph7.initialize_positions()
			agent_7 = Agent7()
			result = agent_7.proceed(graph7)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_7_record.append(0)
			elif result == 1:
				print("Agent 7 caught the Prey! Agent 7 won!")
				print("1")
				agnt_7_record.append(1)
			else:
				print("Predator caught the agent! Agent 7 lost!")
				print("2")
				agnt_7_record.append(2)
		agnt_7_sum_0 = 0
		agnt_7_sum_1 = 0
		agnt_7_sum_2 = 0

		for a in range(30):
			if agnt_7_record[(i*30)+a] == 0:
				agnt_7_sum_0 += 1
			elif agnt_7_record[(i*30)+a] == 1:
				agnt_7_sum_1 += 1
			elif agnt_7_record[(i*30)+a] == 2:
				agnt_7_sum_2 += 1
		agnt_7_final_suspend += agnt_7_sum_0
		agnt_7_final_fail += agnt_7_sum_2
		agnt_7_survivability.append((agnt_7_sum_1/30)*100)

	agnt_7_final_survivability = sum(agnt_7_survivability)/100
	print("agnt_7_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 7 Total iterations: " + str(len(agnt_7_record)))
	print("Agent 7 Count when Game Suspended: " + str(agnt_7_final_suspend))
	print("Agent 7 Count when Agent Lost: " + str(agnt_7_final_fail))
	print("Agent 7 Count when Agent Won: " + str(len(agnt_7_record) - (agnt_7_final_fail + agnt_7_final_suspend)))
	print("Agent 7 Survivability: " + str(agnt_7_final_survivability))
##########AGENT 7 END##########

##########AGENT 8 START##########
if run_8 == True:
	agnt_8_survivability = []
	agnt_8_final_fail = 0
	agnt_8_final_suspend = 0
	for i in range(100):
		graph_8 = Graph()
		graph_8.create_graph()
		for j in range(30):
			graph_8.initialize_positions()
			agent_8 = Agent8()
			result = agent_8.proceed(graph_8)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_8_record.append(0)
			elif result == 1:
				print("Agent 7 caught the Prey! Agent 7 won!")
				print("1")
				agnt_8_record.append(1)
			else:
				print("Predator caught the agent! Agent 7 lost!")
				print("2")
				agnt_8_record.append(2)
		agnt_8_sum_0 = 0
		agnt_8_sum_1 = 0
		agnt_8_sum_2 = 0

		for a in range(30):
			if agnt_8_record[(i*30)+a] == 0:
				agnt_8_sum_0 += 1
			elif agnt_8_record[(i*30)+a] == 1:
				agnt_8_sum_1 += 1
			elif agnt_8_record[(i*30)+a] == 2:
				agnt_8_sum_2 += 1
		agnt_8_final_suspend += agnt_8_sum_0
		agnt_8_final_fail += agnt_8_sum_2
		agnt_8_survivability.append((agnt_8_sum_1/30)*100)

	agnt_8_final_survivability = sum(agnt_8_survivability)/100
	print("agnt_8_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 8 Total iterations: " + str(len(agnt_8_record)))
	print("Agent 8 Count when Game Suspended: " + str(agnt_8_final_suspend))
	print("Agent 8 Count when Agent Lost: " + str(agnt_8_final_fail))
	print("Agent 8 Count when Agent Won: " + str(len(agnt_8_record) - (agnt_8_final_fail + agnt_8_final_suspend)))
	print("Agent 8 Survivability: " + str(agnt_8_final_survivability))
##########AGENT 8 END##########

##########AGENT 7_b START##########
if run_7b == True:
	agnt_7_b_survivability = []
	agnt_7_b_final_fail = 0
	agnt_7_b_final_suspend = 0
	for i in range(100):
		graph7_b = Graph()
		graph7_b.create_graph()
		for j in range(30):
			graph7_b.initialize_positions()
			agent_7_b = Agent7_b()
			result = agent_7_b.proceed(graph7_b)
			if result == 0:
				print("Game suspended! Agent 7_b play stopped!")
				print("0")
				agnt_7_b_record.append(0)
			elif result == 1:
				print("Agent 7_b caught the Prey! Agent 7_b won!")
				print("1")
				agnt_7_b_record.append(1)
			else:
				print("Predator caught the agent! Agent 7_b lost!")
				print("2")
				agnt_7_b_record.append(2)
		agnt_7_b_sum_0 = 0
		agnt_7_b_sum_1 = 0
		agnt_7_b_sum_2 = 0

		for a in range(30):
			if agnt_7_b_record[(i*30)+a] == 0:
				agnt_7_b_sum_0 += 1
			elif agnt_7_b_record[(i*30)+a] == 1:
				agnt_7_b_sum_1 += 1
			elif agnt_7_b_record[(i*30)+a] == 2:
				agnt_7_b_sum_2 += 1
		agnt_7_b_final_suspend += agnt_7_b_sum_0
		agnt_7_b_final_fail += agnt_7_b_sum_2
		agnt_7_b_survivability.append((agnt_7_b_sum_1/30)*100)

	agnt_7_b_final_survivability = sum(agnt_7_b_survivability)/100
	print("agnt_7_b_survivability: " + str(agnt_7_b_survivability))
	
	print("Agent 7_b Total iterations: " + str(len(agnt_7_b_record)))
	print("Agent 7_b Count when Game Suspended: " + str(agnt_7_b_final_suspend))
	print("Agent 7_b Count when Agent Lost: " + str(agnt_7_b_final_fail))
	print("Agent 7_b Count when Agent Won: " + str(len(agnt_7_b_record) - (agnt_7_b_final_fail + agnt_7_b_final_suspend)))
	print("Agent 7_b Survivability: " + str(agnt_7_b_final_survivability))
##########AGENT 7_b END##########

##########AGENT 8_b START##########
if run_8b == True:
	agnt_8_b_survivability = []
	agnt_8_b_final_fail = 0
	agnt_8_b_final_suspend = 0
	for i in range(100):
		graph_8_b = Graph()
		graph_8_b.create_graph()
		for j in range(30):
			graph_8_b.initialize_positions()
			agent_8_b = Agent8_b()
			result = agent_8_b.proceed(graph_8_b)
			if result == 0:
				print("Game suspended! Agent 8_b play stopped!")
				print("0")
				agnt_8_b_record.append(0)
			elif result == 1:
				print("Agent 8_b caught the Prey! Agent 8_b won!")
				print("1")
				agnt_8_b_record.append(1)
			else:
				print("Predator caught the agent! Agent 8_b lost!")
				print("2")
				agnt_8_b_record.append(2)
		agnt_8_b_sum_0 = 0
		agnt_8_b_sum_1 = 0
		agnt_8_b_sum_2 = 0

		for a in range(30):
			if agnt_8_b_record[(i*30)+a] == 0:
				agnt_8_b_sum_0 += 1
			elif agnt_8_b_record[(i*30)+a] == 1:
				agnt_8_b_sum_1 += 1
			elif agnt_8_b_record[(i*30)+a] == 2:
				agnt_8_b_sum_2 += 1
		agnt_8_b_final_suspend += agnt_8_b_sum_0
		agnt_8_b_final_fail += agnt_8_b_sum_2
		agnt_8_b_survivability.append((agnt_8_b_sum_1/30)*100)

	agnt_8_b_final_survivability = sum(agnt_8_b_survivability)/100
	print("agnt_8_b_survivability: " + str(agnt_8_b_survivability))
	
	print("Agent 8_b Total iterations: " + str(len(agnt_8_b_record)))
	print("Agent 8_b Count when Game Suspended: " + str(agnt_8_b_final_suspend))
	print("Agent 8_b Count when Agent Lost: " + str(agnt_8_b_final_fail))
	print("Agent 8_b Count when Agent Won: " + str(len(agnt_8_b_record) - (agnt_8_b_final_fail + agnt_8_b_final_suspend)))
	print("Agent 8_b Survivability: " + str(agnt_8_b_final_survivability))
##########AGENT 8_b END##########