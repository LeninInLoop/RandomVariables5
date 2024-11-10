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
    # 1. Initial coin flip simulations
    TRIALS = 100
    choice_results = [heads_or_tails(method=CHOICE) for _ in range(TRIALS)]
    rand_results = [heads_or_tails(method=RAND) for _ in range(TRIALS)]

    # Print results of coin flip methods
    print("-" * 50 + "\nCounted results from Choice method:")
    print(Counter(choice_results))

    print("-" * 50 + "\nCounted results from RAND method:")
    print(Counter(rand_results))

    # 2. Calculate and display hyper parameters
    variance = ALPHA * T
    std_dev = np.sqrt(variance)

    print("-" * 50 + "\nHyper parameters:")
    print("ALPHA =", ALPHA)
    print("T =", T)
    print("S^2 = ALPHA * T =", variance)
    print("S =", std_dev)
    print("N =", N)
    print("Samples of wiener =", SAMPLES_OF_WIENER)

    # 3. Wiener process initialization
    process_start_time = N * T
    print("-" * 50 + "\nWiener process starts at:")
    print("t = N * T =", process_start_time, "Second")

    # 4. Generate Wiener process
    total_samples = N + SAMPLES_OF_WIENER + 1
    wiener_results = [heads_or_tails(method=RAND, step_size=0.001) for _ in range(total_samples)]
    wiener_dict = {i: wiener_results[i] for i in range(total_samples)}

    # 5. Calculate statistics
    sample_sum = sum(wiener_dict[i + N] for i in range(SAMPLES_OF_WIENER + 1))
    sample_mean = sample_sum / SAMPLES_OF_WIENER

    print("-" * 50 + "\nsummation and mean after Wiener process starts at 1 million samples:")
    print(f"- Summation was over {SAMPLES_OF_WIENER} samples")
    print("Summation:", sample_sum)
    print("Mean:", sample_mean)

    # Calculate variance
    squared_deviations_sum = sum((wiener_dict[i + N] - sample_mean) ** 2
                                 for i in range(SAMPLES_OF_WIENER + 1))
    sample_variance = squared_deviations_sum / SAMPLES_OF_WIENER
    print("Variance:", sample_variance)

    # 6. Plot full Wiener process
    plot_wiener_process(
        wiener_dict,
        display=True,
        filename="wiener_plot_starts_at_1sec_extended.png",
    )

    # 7. Create and plot truncated process
    truncated_values = list(dict(list(wiener_dict.items())[N:]).values())
    truncated_dict = {i: truncated_values[i] for i in range(SAMPLES_OF_WIENER)}

    plot_wiener_process(
        truncated_dict,
        display=True,
        filename="wiener_plot.png",
    )

    # 8. Covariance calculation
    TIME_0 = 1000
    TIME_1 = 2000
    COVARIANCE_TRIALS = 200

    covariance_results = []
    for _ in range(COVARIANCE_TRIALS):
        # Generate new process
        trial_results = [heads_or_tails(method=RAND, step_size=0.001) for _ in range(N + TIME_1 + 1)]
        trial_dict = {i: trial_results[i] for i in range(N + TIME_1 + 1)}

        # Process results
        trial_values = list(dict(list(trial_dict.items())[N:]).values())
        trial_process = {i: trial_values[i] for i in range(TIME_1 + 1)}
        cumulative_sums = calculate_cumulative_sum(trial_process)

        # Calculate covariance
        covariance_results.append(
            list(cumulative_sums.values())[TIME_0] * list(cumulative_sums.values())[TIME_1]
        )

    print("-" * 50 + "\nRx(0.001,0.002) equals to:")
    print(np.mean(covariance_results))
    print("Wiener Rx(t0,t1) should be ALPHA * min(t0,t1) theoretically.\n"
          "ALPHA is equal to one so that the Rx(0.001,0.002) should be 0.001.")

if __name__ == '__main__':
    main()

