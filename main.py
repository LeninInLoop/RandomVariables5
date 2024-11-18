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
SAMPLES_OF_WIENER = 10 ** 6

def heads_or_tails_vectorized(method: int = RAND, step_size: float = 0.001, num_samples: int = N + SAMPLES_OF_WIENER + 1) -> np.ndarray:
  if method == RAND:
    random_numbers = np.random.rand(num_samples)
    return np.where(random_numbers < 0.5, step_size, -step_size)
  else:
    outcomes = np.array([-step_size, step_size])
    return np.random.choice(outcomes, size=num_samples)

def calculate_cumulative_sum_vectorized(values: np.ndarray) -> np.ndarray:
  return np.cumsum(values)

def plot_wiener_process(
        result: np.ndarray,
        display : bool = False,
        filename: str = "wiener_plot.png",
    ) -> None:

    plt.figure(figsize=(10, 6))
    plt.plot(result, label="Cumulative Sum", color='b', linewidth=1)

    plt.xlabel("Second")
    plt.ylabel("Cumulative Sum")
    plt.title("WIENER PROCESS")

    plt.ylim(result.min() * 1.5, result.max() * 1.5)

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
    choice_results = heads_or_tails_vectorized(method=CHOICE, step_size =0.001, num_samples=TRIALS)
    rand_results = heads_or_tails_vectorized(method=RAND, step_size =0.001, num_samples=TRIALS)

    # Print results of coin flip methods
    print("-" * 50 + "\nCounted results from Choice method:")
    print(Counter(choice_results))

    print("-" * 50 + "\nCounted results from RAND method:")
    print(Counter(rand_results))

    # 2. Calculate and display hyperparameters
    step_length_2 = ALPHA * T
    step_length = np.sqrt(step_length_2)

    print("-" * 50 + "\nHyper parameters:")
    print("ALPHA =", ALPHA)
    print("T =", T)
    print("S^2 = ALPHA * T =", step_length_2)
    print("S =", step_length)
    print("N =", N)
    print("Samples of wiener =", SAMPLES_OF_WIENER)

    # 3. Wiener process initialization
    process_start_time = N * T
    print("-" * 50 + "\nWiener process starts at:")
    print("t = N * T =", process_start_time, "Second")

    # 4. Generate Wiener process
    total_samples = N + SAMPLES_OF_WIENER + 1
    wiener_results = heads_or_tails_vectorized(method=RAND, step_size =0.001, num_samples=total_samples)

    sample_sum = calculate_cumulative_sum_vectorized(wiener_results[N:])
    sample_mean = sample_sum[-1] / SAMPLES_OF_WIENER

    print("-" * 50 + "\nsummation and mean after Wiener process starts after 1 million samples:")
    print(f"- Summation was over {SAMPLES_OF_WIENER} samples")
    print("Summation:", sample_sum[-1])
    print("Mean:", sample_mean)

    # 6. Plot full Wiener process
    plot_wiener_process(
        calculate_cumulative_sum_vectorized(wiener_results),
        display=True,
        filename="wiener_plot_starts_at_1sec_extended.png",
    )

    plot_wiener_process(
        sample_sum,
        display=True,
        filename="wiener_plot.png",
    )

    # 8. Covariance calculation
    TIME_0 = 1000
    TIME_1 = 2000
    COVARIANCE_TRIALS = 200

    covariance_results = []
    variance_results = []
    for _ in range(COVARIANCE_TRIALS):
        # Generate new process
        wiener_results = heads_or_tails_vectorized(method=RAND, step_size =0.001, num_samples= N + TIME_1 + 1)
        wiener_results_cumulative_sum = np.cumsum(wiener_results[N:])

        covariance_results.append(
            wiener_results_cumulative_sum[TIME_0] * wiener_results_cumulative_sum[TIME_1]
        )
        variance_results.append(
            wiener_results_cumulative_sum[TIME_0] * wiener_results_cumulative_sum[TIME_0]
        )

    print("-" * 50 + "\nVariance(x(0.001)) = Rx(0.001,0.001) equals to:")
    print(str(np.mean(variance_results)) + " -> t = t0 * T =",TIME_0 * T)
    print("Wiener Rx(t0,t0) should be ALPHA * t0 theoretically.\n"
          "ALPHA is equal to one so that the Rx(0.001,0.001) should be 0.001.")

    print("-" * 50 + "\nRx(0.001,0.002) equals to:")
    print(str(np.mean(covariance_results)) + " -> t = t0 * T =",TIME_0 * T)
    print("Wiener Rx(t0,t1) should be ALPHA * min(t0,t1) theoretically.\n"
          "ALPHA is equal to one so that the Rx(0.001,0.002) should be 0.001.")

if __name__ == '__main__':
    main()

