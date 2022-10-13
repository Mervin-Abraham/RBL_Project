import random
from multiprocessing import Pool, Event
from datetime import datetime
from collections import namedtuple
import copy

terminating = Event()
HarmonySearchResults = namedtuple('HarmonySearchResults', ['elapsed_time', 'best_harmony', 'best_fitness', 'harmony_memories', 'harmony_histories'])


def harmony_search(objective_function, num_processes, num_iterations, initial_harmonies=None):
    pool = Pool(num_processes)
    try:
        start = datetime.now()
        pool_results = [pool.apply_async(worker, args=(objective_function, initial_harmonies,)) for i in range(num_iterations)]
        pool.close()
        pool.join()
        end = datetime.now()
        elapsed_time = end - start

        # best harmony from all iterations
        best_harmony = None
        best_fitness = float('-inf') if objective_function.maximize() else float('+inf')
        harmony_memories = list()
        harmony_histories = list()

        for result in pool_results:
            harmony, fitness, harmony_memory, harmony_history = result.get()
            if (objective_function.maximize() and fitness > best_fitness) or (not objective_function.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness
            harmony_memories.append(harmony_memory)
            harmony_histories.append(harmony_history)

        return HarmonySearchResults(elapsed_time=elapsed_time, best_harmony=best_harmony, best_fitness=best_fitness,
                                    harmony_memories=harmony_memories, harmony_histories=harmony_histories)
    except KeyboardInterrupt:
        pool.terminate()


def worker(objective_function, initial_harmonies=None):
    try:
        if not terminating.is_set():
            hs = HarmonySearch(objective_function)
            return hs.run(initial_harmonies=initial_harmonies)
    except KeyboardInterrupt:
        terminating.set()


class HarmonySearch(object):
    def __init__(self, objective_function, initial_harmonies=None):
        self._obj_fun = objective_function
        # harmony_memory stores the best hms harmonies
        self._harmony_memory = list()

        # harmony_history stores all hms harmonies every nth improvisations (i.e., one 'generation')
        self._harmony_history = list()

        # fill harmony_memory using random parameter values by default, but with initial_harmonies if provided
        self._initialize(initial_harmonies)

    def run(self, initial_harmonies=None):
        # set optional random seed
        if self._obj_fun.use_random_seed():
            random.seed(self._obj_fun.get_random_seed())

        # create max_imp improvisations
        generation = 0
        num_imp = 0
        while num_imp < self._obj_fun.get_max_imp():
            # generate new harmony
            harmony = list()
            for i in range(0, self._obj_fun.get_num_parameters()):
                if random.random() < self._obj_fun.get_hmcr():
                    self._memory_consideration(harmony, i)
                    if random.random() < self._obj_fun.get_par():
                        self._pitch_adjustment(harmony, i)
                else:
                    self._random_selection(harmony, i)

            # fitness = self._obj_fun.switch(harmony)
            fitness = self._obj_fun.get_fitness(harmony)
            self._update_harmony_memory(harmony, fitness)
            num_imp += 1

            # save harmonies every nth improvisations (i.e., one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:
                generation += 1
                harmony_list = {'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)}
                self._harmony_history.append(harmony_list)

        # return best harmony
        best_harmony = None
        best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')
        for harmony, fitness in self._harmony_memory:
            if (self._obj_fun.maximize() and fitness > best_fitness) or (
                    not self._obj_fun.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness
        return best_harmony, best_fitness, self._harmony_memory, self._harmony_history

    def _initialize(self, initial_harmonies=None):

        if initial_harmonies is not None:
            # verify that the initial harmonies are provided correctly

            if len(initial_harmonies) != self._obj_fun.get_hms():
                raise ValueError('Number of initial harmonies does not equal to the harmony memory size.')

            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_harmonies)):
                num_parameters_initial_harmonies = len(initial_harmonies[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial harmonies does not match that defined.')
        else:
            initial_harmonies = list()
            for i in range(0, self._obj_fun.get_hms()):
                harmony = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(harmony, j)
                initial_harmonies.append(harmony)

        for i in range(0, self._obj_fun.get_hms()):
            fitness = self._obj_fun.get_fitness(initial_harmonies[i])
            self._harmony_memory.append((initial_harmonies[i], fitness))

        harmony_list = {'gen': 0, 'harmonies': self._harmony_memory}
        self._harmony_history.append(harmony_list)

    def _random_selection(self, harmony, i):
        harmony.append(self._obj_fun.get_value(i))

    def _memory_consideration(self, harmony, i):
        memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
        harmony.append(self._harmony_memory[memory_index][0][i])

    def _pitch_adjustment(self, harmony, i):
        if self._obj_fun.is_variable(i):
            if self._obj_fun.is_discrete(i):
                current_index = self._obj_fun.get_index(i, harmony[i])
                if random.random() < 0.5:
                    # adjust pitch down
                    harmony[i] = self._obj_fun.get_value(i, current_index - random.randint(0,min(self._obj_fun.get_mpai(), current_index)))
                else:
                    # adjust pitch up
                    harmony[i] = self._obj_fun.get_value(i,current_index + random.randint(0, min(self._obj_fun.get_mpai(),self._obj_fun.get_num_discrete_values(
                                                                                                   i) - current_index - 1)))
            else:
                if random.random() < 0.5:
                    harmony[i] -= (harmony[i] - self._obj_fun.get_lower_bound(
                        i)) * random.random() * self._obj_fun.get_mpap()
                else:
                    # adjust pitch up
                    harmony[i] += (self._obj_fun.get_upper_bound(i) - harmony[
                        i]) * random.random() * self._obj_fun.get_mpap()

    def _update_harmony_memory(self, considered_harmony, considered_fitness):

        if (considered_harmony, considered_fitness) not in self._harmony_memory:
            worst_index = None
            worst_fitness = float('+inf') if self._obj_fun.maximize() else float('-inf')
            for i, (harmony, fitness) in enumerate(self._harmony_memory):
                if (self._obj_fun.maximize() and fitness < worst_fitness) or (
                        not self._obj_fun.maximize() and fitness > worst_fitness):
                    worst_index = i
                    worst_fitness = fitness
            if (self._obj_fun.maximize() and considered_fitness > worst_fitness) or (
                    not self._obj_fun.maximize() and considered_fitness < worst_fitness):
                self._harmony_memory[worst_index] = (considered_harmony, considered_fitness)

if __name__ == "__main__":
    pass
