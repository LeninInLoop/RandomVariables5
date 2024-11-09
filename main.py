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

def heads_or_tails(method: int = RAND, step_size : float = 0.001) -> float:
    if method == RAND:
        random_number: float = np.random.rand()
        return step_size if random_number < 0.5 else -step_size
    else:
        outcomes = [-step_size, step_size]
        return np.random.choice(outcomes)

def calculate_cumulative_sum(dict_to_calculate: Dict[int, float]) -> Dict[int, float]:
    cumulative_sum = 0
    new_dict = {}
    for key in dict_to_calculate:  # Iterate over the keys in the dictionary
        cumulative_sum += dict_to_calculate[key]
        new_dict[key] = cumulative_sum
    return new_dict

def plot_wiener_process(
        result_dict: Dict[int, float],
        display : bool = False,
        filename: str = "wiener_plot.png",
    ) -> None:

    new_dict = calculate_cumulative_sum(result_dict)
    times = [key / N for key in new_dict.keys()] # X values (number of samples)
    values = new_dict.values() # Y values (Cumulative sum of everything before the current sample)

    plt.figure(figsize=(10, 6))
    plt.plot(times, values, label="Cumulative Sum", color='b', linewidth=1)

    # Add theoretical bounds
    std = np.sqrt(ALPHA * times)
    plt.plot(times, 2 * std, 'r--', alpha=0.5, label='±2σ bounds')
    plt.plot(times, -2 * std, 'r--', alpha=0.5)

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

    # Generate the Wiener process steps
    results = [heads_or_tails(method=RAND, step_size=0.001) for _ in range(N + SAMPLES_OF_WIENER + 1)]
    result_dict = {i: results[i] for i in range(N + SAMPLES_OF_WIENER + 1)}
    print(result_dict)

    summation = sum(result_dict[i + N] for i in range(SAMPLES_OF_WIENER + 1))
    print(50 * "-" + "\nsummation and mean after Wiener process starts at 1 million samples:\n"
                     f"- Summation was over {SAMPLES_OF_WIENER} samples")
    print("Summation:", summation)

    mean = summation / SAMPLES_OF_WIENER
    print("Mean:", mean)

    # Variance calculation
    sum_squared = sum((result_dict[i + N] - mean) ** 2 for i in range(SAMPLES_OF_WIENER + 1))
    variance_calculated = sum_squared / SAMPLES_OF_WIENER
    print("Variance:", variance_calculated)

    plot_wiener_process(
        result_dict,
        display=True,
        filename="wiener_plot.png",
    )

if __name__ == '__main__':
    main()

