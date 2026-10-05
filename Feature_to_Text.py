import pandas as pd

# INTERPRETATION FUNCTIONS

def interpret_range(val, low, high, low_label, normal_label, high_label):
    if val < low:
        return low_label
    elif val <= high:
        return normal_label
    else:
        return high_label


def interpret_binary(val, threshold, low_text, high_text):
    return low_text if val < threshold else high_text


def interpret_abnormal_percent(val):
    if val < 20:
        return "low abnormal proportion"
    elif val < 50:
        return "moderate abnormal proportion"
    else:
        return "high abnormal proportion"


def interpret_mean_stv(val):
    if val < 1.0:
        return "very low mean STV"
    elif val < 3.0:
        return "low mean STV"
    elif val <= 7.0:
        return "moderate mean STV"
    else:
        return "high mean STV"


def interpret_mean_ltv(val):
    if val < 5.0:
        return "very low mean LTV"
    elif val < 10.0:
        return "low mean LTV"
    elif val <= 20.0:
        return "moderate mean LTV"
    else:
        return "high mean LTV"


def format_num(val):
    try:
        if float(val).is_integer():
            return str(int(val))
        return f"{float(val):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(val)

# MAIN TEXT GENERATOR

def generate_clinical_text(row):

    # Core
    baseline = row['baseline value']
    acc = row['accelerations']
    movement = row['fetal_movement']
    contractions = row['uterine_contractions']

    # Variability
    var_short = row['abnormal_short_term_variability']
    var_short_mean = row['mean_value_of_short_term_variability']
    var_long = row['percentage_of_time_with_abnormal_long_term_variability']
    var_long_mean = row['mean_value_of_long_term_variability']

    # Decelerations
    light = row['light_decelerations']
    severe = row['severe_decelerations']
    prolonged = row['prolongued_decelerations']

    # Histogram
    hist_min = row['histogram_min']
    hist_max = row['histogram_max']
    hist_mean = row['histogram_mean']
    hist_median = row['histogram_median']
    hist_mode = row['histogram_mode']
    hist_width = row['histogram_width']
    hist_variance = row['histogram_variance']
    hist_tendency = row['histogram_tendency']
    hist_peaks = row['histogram_number_of_peaks']
    hist_zeroes = row['histogram_number_of_zeroes']

    # Interpretations
    baseline_s = interpret_range(baseline, 110, 160, "low", "normal", "high")
    acc_s = interpret_binary(acc, 0.005, "reduced accelerations", "normal accelerations")
    move_s = interpret_binary(movement, 0.02, "reduced fetal movements", "normal fetal movements")

    var_short_s = interpret_abnormal_percent(var_short)
    var_short_mean_s = interpret_mean_stv(var_short_mean)
    var_long_s = interpret_abnormal_percent(var_long)
    var_long_mean_s = interpret_mean_ltv(var_long_mean)

    cont_s = interpret_binary(contractions, 0.005, "mild contractions", "frequent contractions")

    dec_types = []
    if light > 0:
        dec_types.append("light")
    if prolonged > 0:
        dec_types.append("prolonged")
    if severe > 0:
        dec_types.append("severe")

    dec_s = "none" if not dec_types else ", ".join(dec_types)

    # Histogram interpretation
    if hist_width < 50:
        spread = "narrow"
    elif hist_width < 100:
        spread = "moderate"
    else:
        spread = "wide"

    skew = "left-skewed" if hist_tendency < 0 else "right-skewed" if hist_tendency > 0 else "symmetrical"


    # FINAL TEXT
    text = (
        f"Baseline heart rate {format_num(baseline)} bpm ({baseline_s}). "
        f"Accelerations: {acc_s}. Fetal movement: {move_s}. Uterine contractions: {cont_s}. "
        f"Abnormal short-term variability: {var_short_s} "
        f"({format_num(var_short)}% of time), with {var_short_mean_s} "
        f"(mean {format_num(var_short_mean)}). "
        f"Abnormal long-term variability: {var_long_s} "
        f"({format_num(var_long)}% of time), with {var_long_mean_s} "
        f"(mean {format_num(var_long_mean)}). "
        f"Decelerations present: {dec_s}. "
        f"Histogram range {format_num(hist_min)}-{format_num(hist_max)} bpm "
        f"(width {format_num(hist_width)}), mean {format_num(hist_mean)}, "
        f"median {format_num(hist_median)}, mode {format_num(hist_mode)}, "
        f"variance {format_num(hist_variance)}. "
        f"Distribution is {spread} and {skew}, with "
        f"{format_num(hist_peaks)} peaks and {format_num(hist_zeroes)} zero points."
    )

    return text.strip()


def run_pipeline(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    required_cols = {
        'baseline value',
        'accelerations',
        'fetal_movement',
        'uterine_contractions',
        'light_decelerations',
        'severe_decelerations',
        'prolongued_decelerations',
        'abnormal_short_term_variability',
        'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability',
        'histogram_width',
        'histogram_min',
        'histogram_max',
        'histogram_number_of_peaks',
        'histogram_number_of_zeroes',
        'histogram_mode',
        'histogram_mean',
        'histogram_median',
        'histogram_variance',
        'histogram_tendency',
        'fetal_health',
    }

    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    df['clinical_text'] = df.apply(generate_clinical_text, axis=1)

    df[['clinical_text', 'fetal_health']].to_csv(output_csv, index=False)

    print("✅ MAX COMPLETE dataset generated!")
    print(df[['clinical_text', 'fetal_health']].head())


if __name__ == "__main__":
    run_pipeline("fetal_health(kaggle).csv", "ctg_full_text.csv")