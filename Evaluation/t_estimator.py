import pandas as pd
import numpy as np

from scipy.optimize import minimize_scalar


def prob_from_logprob(logprobs, T):
    # Log-probabilities are always less than or equal to zero.
    # When the log-probability is 0, the corresponding probability is at its maximum value of 1.
    # For unclassified cases, I return 0 for the probability and np.NaN for the log-probability.
    # Therefore, I need to check here whether the value is not NaN.
    logprobs = np.asarray(logprobs, dtype=float)
    probs = np.exp(logprobs/T)

    probs = np.nan_to_num(probs, nan=0.0)

    return probs


def calibration_nll(T, logprobs, y_true_correct):
    probs = prob_from_logprob(logprobs, T)

    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)

    nll = -np.mean(
        y_true_correct * np.log(probs) + (1 - y_true_correct) * np.log(1 - probs)
    )

    return nll


def fit_temperature(logprobs, y_true_correct):
    return minimize_scalar(
        calibration_nll,
        bounds=(0.05, 20.0),
        args=(logprobs, y_true_correct),
        method="bounded"
    ).x



if __name__ == "__main__":
    input_dataset_to_calibrate = "../Datasets/09a_BalancedSampledResults.csv"
    # input_dataset_to_calibrate = "../Datasets/09b_RealDistributedSampledResults.csv"
    df = pd.read_csv(input_dataset_to_calibrate)



    df['N1 llama Correct'] = (df['N1 llama Classification'] == df['Label_N1']).astype(int).values
    T_N1_llama = fit_temperature(
        df['N1 llama Pred Log Proba'].values, 
        df['N1 llama Correct'].values
    )
    df['N2 llama Correct'] = (df['N2 llama Classification'] == df['Label_N2']).astype(int).values
    T_N2_llama = fit_temperature(
        df['N2 llama Pred Log Proba'].values, 
        df['N2 llama Correct'].values
    )
    df['N3 llama Correct'] = (df['N3 llama Classification'] == df['Label_N3']).astype(int).values
    T_N3_llama = fit_temperature(
        df['N3 llama Pred Log Proba'].values, 
        df['N3 llama Correct'].values
    )



    df['N1 fine_llama Correct'] = (df['N1 fine_llama Classification'] == df['Label_N1']).astype(int).values
    T_N1_fine_llama = fit_temperature(
        df['N1 fine_llama Pred Log Proba'].values, 
        df['N1 fine_llama Correct'].values
    )
    df['N2 fine_llama Correct'] = (df['N2 fine_llama Classification'] == df['Label_N2']).astype(int).values
    T_N2_fine_llama = fit_temperature(
        df['N2 fine_llama Pred Log Proba'].values, 
        df['N2 fine_llama Correct'].values
    )
    df['N3 fine_llama Correct'] = (df['N3 fine_llama Classification'] == df['Label_N3']).astype(int).values
    T_N3_fine_llama = fit_temperature(
        df['N3 fine_llama Pred Log Proba'].values, 
        df['N3 fine_llama Correct'].values
    )



    df['N1 ML Correct'] = (df['N1 ML Classification'] == df['Label_N1']).astype(int).values
    df['N2 ML Correct'] = (df['N2 ML Classification'] == df['Label_N2']).astype(int).values
    df['N3 ML Correct'] = (df['N3 ML Classification'] == df['Label_N3']).astype(int).values



    df['N1 llama Calibrated Proba'] = np.exp(df['N1 llama Pred Log Proba'] / T_N1_llama)
    df['N2 llama Calibrated Proba'] = np.exp(df['N2 llama Pred Log Proba'] / T_N2_llama)
    df['N3 llama Calibrated Proba'] = np.exp(df['N3 llama Pred Log Proba'] / T_N3_llama)
    df['N1 fine_llama Calibrated Proba'] = np.exp(df['N1 fine_llama Pred Log Proba'] / T_N1_fine_llama)
    df['N2 fine_llama Calibrated Proba'] = np.exp(df['N2 fine_llama Pred Log Proba'] / T_N2_fine_llama)
    df['N3 fine_llama Calibrated Proba'] = np.exp(df['N3 fine_llama Pred Log Proba'] / T_N3_fine_llama)


    calibrated_results = "../Datasets/10a_BalancedCalibratedResults.csv"
    # calibrated_results = "../Datasets/10b_RealDistributedCalibratedResults.csv"
    df.to_csv(calibrated_results, index=False)


    print("Success!")
    print(f"T_N1_llama: {T_N1_llama}")
    print(f"T_N2_llama: {T_N2_llama}")
    print(f"T_N3_llama: {T_N3_llama}")
    print(f"T_N1_fine_llama: {T_N1_fine_llama}")
    print(f"T_N2_fine_llama: {T_N2_fine_llama}")
    print(f"T_N3_fine_llama: {T_N3_fine_llama}")
