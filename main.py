import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict

# Methods for heads or tails (with P of 1/2)
RAND = 0
CHOICE = 1

# Hyper parameters
ALPHA = 1
T = 10 ** (-6)
N = 10 ** 6
SAMPLES_OF_WIENER = 10 ** 4

def heads_or_tails(method: int = RAND) -> int:
    if method == RAND:
        random_number : float = np.random.rand(1)
        if random_number < 0.5:
            return 1
        elif random_number > 0.5:
            return -1
    else:
        outcomes = [-1, 1]
        return int(np.random.choice(outcomes))

def calculate_cumulative_sum(dict_to_calculate: Dict[int, int]) -> Dict[int, int]:
    cumulative_sum = 0
    new_dict = {}
    for i in range(N + SAMPLES_OF_WIENER + 1):
        if i in dict_to_calculate:
            cumulative_sum += dict_to_calculate[i]
        new_dict[i] = cumulative_sum
    return new_dict

def plot_wiener_process(
        result_dict: Dict[int, int],
        display : bool = False,
        filename: str = "wiener_plot.png",
    ) -> None:

    s_to_power_of_two = ALPHA * T
    s = np.sqrt(s_to_power_of_two) # Steps should be s. if its 1 +s and if its -1 then it should be -s

    new_dict = calculate_cumulative_sum(result_dict)
    keys = [key / N for key in new_dict.keys()] # X values (number of samples)
    values = [value * s for value in new_dict.values()] # Y values (Cumulative sum of everything before the current sample)

    plt.figure(figsize=(10, 6))
    plt.plot(keys, values, label="Cumulative Sum", color='b', linewidth=1)

    plt.xlabel("Second")
    plt.ylabel("Cumulative Sum")
    plt.title("WIENER PROCESS")
    plt.legend()

    plt.grid(True)
    plt.savefig(filename)

    if display:
        plt.show()
    else:
        plt.close()

def main():

    results_with_choice_method = []
    results_with_rand_method = []
    for i in range(100):
        results_with_choice_method.append(heads_or_tails(method=CHOICE))
        results_with_rand_method.append(heads_or_tails(method=RAND))

    print(50 * "-" + "\nCounted results from Choice method:")
    counted_results = Counter(results_with_choice_method)
    print(counted_results)

    print(50 * "-" + "\nCounted results from RAND method:")
    counted_results = Counter(results_with_rand_method)
    print(counted_results)

    s_to_power_of_two = ALPHA * T
    s = np.sqrt(s_to_power_of_two)
    print(50 * "-" + "\nHyper parameters:")
    print("ALPHA =", ALPHA)
    print("T =", T)
    print("S^2 = ALPHA * T =", s_to_power_of_two)
    print("S =", s)
    print("N =", N)
    print("Samples of wiener =", SAMPLES_OF_WIENER)

    t = N * T
    print(50 * "-" + "\nWiener process starts at:")
    print("t = N * T =",t , "Second")

    result_dict = { value: 0 for value in range(10 ** 6 + SAMPLES_OF_WIENER + 1)}
    for i in result_dict.keys():
        result_dict[i] = heads_or_tails(method=RAND)
    print(result_dict)

    summation = 0
    for i in range(1, SAMPLES_OF_WIENER + 1):
        summation += result_dict[i + N]
    print(50 * "-" + "\nsummation and mean after Wiener process starts at 1 million samples:\n"
                     f"- Summation was over {SAMPLES_OF_WIENER} samples")
    print("Summation:",summation)
    print("Mean:",summation / SAMPLES_OF_WIENER)

    plot_wiener_process(
        result_dict,
        display=True,
        filename="wiener_plot.png",
    )

if __name__ == '__main__':
    main()

