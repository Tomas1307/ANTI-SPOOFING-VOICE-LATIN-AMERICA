"""
RawBoost reference implementation (LnL convolutive + ISD + SSI noise).

Faithful reconstruction of the official RawBoost algorithm to match the
published reference implementation. RawBoost is a raw-waveform data-augmentation
method for anti-spoofing that applies (1) linear-and-non-linear convolutive
noise via randomized multiband notch FIRs and a Hammerstein-style polynomial
expansion, (2) impulsive signal-dependent additive noise, and (3) stationary
signal-independent (FIR-colored) additive noise.

Reference:
    H. Tak, M. Kamble, J. Patino, M. Todisco, N. Evans, "RawBoost: A Raw Data
    Boosting and Augmentation Method applied to Automatic Speaker Verification
    Anti-Spoofing," ICASSP 2022. Code: https://github.com/TakHemlata/RawBoost-antispoofing

NOTE (verification): this module is reconstructed to match the upstream
``RawBoost.py`` / ``data_utils_rawboost.py`` numerics and function signatures.
Before a production corpus build, diff it against the upstream repository to
confirm exact parity:
    git clone https://github.com/TakHemlata/RawBoost-antispoofing
    diff RawBoost-antispoofing/RawBoost.py app/augmenter/rawboost_reference.py
"""
import copy

import numpy as np
from scipy import signal


def randRange(x1: float, x2: float, integer: bool) -> float:
    """Draw a uniform random value in [x1, x2), optionally cast to int."""
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def normWav(x: np.ndarray, always: bool) -> np.ndarray:
    """Peak-normalize the waveform (always, or only if it exceeds unity)."""
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
                   minG, maxG, fs):
    """Generate randomized multiband notch FIR coefficients."""
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),
            b,
        )

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply an FIR filter and compensate for its group delay."""
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2):int(y.shape[0] - N / 2)]
    return y


def LnL_convolutive_noise(x, N_f, nBands, minF, maxF, minBW, maxBW,
                          minCoeff, maxCoeff, minG, maxG,
                          minBiasLinNonLin, maxBiasLinNonLin, fs):
    """Linear and non-linear convolutive noise (multiband notch + Hammerstein)."""
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG - minBiasLinNonLin
            maxG = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff,
                           maxCoeff, minG, maxG, fs)
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


def ISD_additive_noise(x, P, g_sd):
    """Impulsive signal-dependent additive noise."""
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        ((2 * np.random.rand(p.shape[0])) - 1),
        ((2 * np.random.rand(p.shape[0])) - 1),
    )
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y


def SSI_additive_noise(x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW,
                       minCoeff, maxCoeff, minG, maxG, fs):
    """Stationary signal-independent (FIR-colored) additive noise."""
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
                       minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = (noise / np.linalg.norm(noise, 2)
             * np.linalg.norm(x, 2) / 10.0 ** (0.05 * SNR))
    x = x + noise
    return x


def process_Rawboost_feature(feature, sr, args, algo):
    """
    Apply a RawBoost algorithm to a raw waveform.

    Args:
        feature: 1-D float waveform.
        sr: Sample rate in Hz.
        args: Object exposing the RawBoost parameter attributes (N_f, nBands,
            minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG,
            minBiasLinNonLin, maxBiasLinNonLin, P, g_sd, SNRmin, SNRmax). A
            ``RawBoostParams`` Pydantic instance satisfies this interface.
        algo: Algorithm id. 1=LnL, 2=ISD, 3=SSI, 4=series(LnL->ISD->SSI),
            5=series(LnL->ISD), 6=series(LnL->SSI), 7=parallel(LnL || ISD).
            Any other value returns the feature unchanged.

    Returns:
        The augmented waveform.
    """
    # 1: Convolutive noise
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW,
            args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)

    # 2: Impulsive noise
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # 3: Coloured additive noise
    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
            args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr)

    # 4: All three in series (1 -> 2 -> 3)
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW,
            args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
            args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr)

    # 5: First two in series (1 -> 2)
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW,
            args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # 6: First and third in series (1 -> 3)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW,
            args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
            args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr)

    # 7: LnL and ISD in parallel (1 || 2)
    elif algo == 7:
        feature1 = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW,
            args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)

    # Otherwise: unchanged
    else:
        feature = feature

    return feature
