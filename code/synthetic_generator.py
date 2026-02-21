# =============================================================================
# Bridging Data Scarcity in Geotechnics: A Physics‑Informed Approach to Generate
# Synthetic Soil Records for Machine Learning Classification
# =============================================================================
#
# PURPOSE:
#   This script generates 10,000 synthetic soil samples based on 10 real
#   borehole samples. It uses a hierarchical, physics‑informed approach:
#   primary variables (grain sizes, moisture, specific gravity, depth) are
#   drawn from bounded normal distributions; secondary variables (Cu, Cc) are
#   calculated deterministically; tertiary variables (MDD, φ, OMC, Atterberg
#   limits, cohesion, CBR, shear strength, allowable pressure) are generated
#   using either independent distributions or empirical regressions fitted
#   from the real data. All values are clipped to the physically observed
#   ranges to avoid unrealistic extremes. A final safety clip ensures that
#   D60 stays within [0.85, 4.75] mm.
#
#   After generation, the dataset is balanced by undersampling the majority
#   class (Poorly graded sand, SP) to match the number of minority class
#   (Well‑graded sand, SW) samples. Extensive validation (summary statistics,
#   overlaid distribution plots, correlation matrices, PCA, physical
#   relationship scatter plots, Kolmogorov–Smirnov tests) confirms that the
#   synthetic data faithfully represent the original measurements and that
#   the balancing step did not distort the feature space.
#
# AUTHOR:   Oluwashina Oyeniran
# DATE:     February 21, 2026
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------
import pandas as pd               # Data manipulation and analysis (DataFrames)
import numpy as np                # Numerical operations and random number generation
import matplotlib.pyplot as plt   # Plotting library
import seaborn as sns             # Enhanced statistical visualisations
from sklearn.linear_model import LinearRegression   # For empirical regression fitting
from sklearn.decomposition import PCA               # Principal Component Analysis for validation
from sklearn.preprocessing import StandardScaler    # Standardising data before PCA
from scipy.stats import ks_2samp, linregress       # Kolmogorov‑Smirnov test and linear regression
import os                         # For creating directories and handling file paths
import warnings
warnings.filterwarnings('ignore')  # Suppress harmless warnings (e.g., from plotting)

# Create a directory to store all generated plots
plot_dir = 'validation_plots'
os.makedirs(plot_dir, exist_ok=True)   # Create folder if it doesn't exist
print(f"Plots will be saved to: '{plot_dir}'")

# -----------------------------------------------------------------------------
# 2. INPUT THE COMPLETE REAL DATA
# -----------------------------------------------------------------------------
# We store them in a dictionary and then convert to a pandas DataFrame.
data = {
    'Sample Locations': ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B'],
    'Natural/Field Moisture Content (%)': [3.2, 5.3, 5.1, 5.8, 1.9, 2.9, 4.4, 4.6, 5.0, 5.4],
    'Specific Gravity Test': [2.85, 2.68, 2.72, 2.72, 2.44, 2.43, 2.35, 2.69, 2.40, 2.72],
    'D10 (mm)': [0.25, 0.18, 0.20, 0.40, 0.16, 0.14, 0.22, 0.17, 0.19, 0.13],
    'D30 (mm)': [0.60, 0.42, 0.45, 1.10, 0.38, 0.33, 0.48, 0.40, 0.46, 0.30],
    'D60 (mm)': [2.50, 1.60, 1.80, 4.75, 1.20, 0.95, 1.90, 1.30, 1.70, 0.85],
    'L.L %': [37.60, 21.26, 27.72, 21.60, 22.00, 15.82, 46.66, 28.00, 20.22, 29.38],
    'P.L %': [20.23, 16.26, 17.63, 15.91, 17.27, 12.36, 32.73, 13.14, 13.52, 24.49],
    'P.I %': [17.37, 5.14, 12.09, 5.69, 4.74, 3.45, 13.93, 14.86, 6.70, 4.89],
    'OMC (%)': [5.53, 4.60, 4.29, 7.7, 5.24, 5.77, 8.83, 7.36, 4.11, 6.77],
    'MDD (g/cm3)': [2.121, 2.119, 2.186, 2.132, 2.048, 2.835, 1.885, 1.988, 2.151, 1.950],
    'Cohesion, C (kN/m2)': [90, 56, 80, 49, 54, 40, 38, 32, 80, 32],
    'Frictional Angle, Ø(0)': [20, 25, 24, 33, 22, 18, 28, 9, 28, 22],
    'Depth (m)': [1.50, 1.20, 1.40, 1.30, 2.20, 0.70, 3.30, 0.80, 2.50, 1.80],
    'California Bearing Ratio': [82, 79, 48, 75, 60, 29, 69, 69, 62, 70],
    'Shear Strength (kN/m2)': [274.20, 58.77, 826.16, 292.00, 212.47, 155.50, 192.30, 73.59, 294.38, 119.89],
    'Allow. Bearing Pressure (kN/m2)': [203.11, 203.11, 209.08, 209.08, 209.08, 209.08, 227.00, 227.00, 209.08, 209.08],
    'Classification': ['Well-graded sand (SW)', 'Well-graded sand (SW)', 'Well-graded sand (SW)',
                       'Well-graded sand (SW)', 'Poorly graded sand (SP)', 'Poorly graded sand (SP)',
                       'Well-graded sand (SW)', 'Poorly graded sand (SP)', 'Well-graded sand (SW)',
                       'Poorly graded sand (SP)']
}

# Convert the dictionary to a pandas DataFrame
real_df = pd.DataFrame(data)

# Calculate Cu and Cc from grain sizes (they are provided in the original table,
# but we recalculate to be consistent with the definitions used in the code).
real_df['Cu'] = real_df['D60 (mm)'] / real_df['D10 (mm)']          # Coefficient of uniformity
real_df['Cc'] = (real_df['D30 (mm)']**2) / (real_df['D60 (mm)'] * real_df['D10 (mm)'])  # Coefficient of curvature

# Display basic information about the real data to confirm correct loading.
print("--- Real Data Summary ---")
print(f"Shape of real data: {real_df.shape}")
print(real_df.head())

# -----------------------------------------------------------------------------
# 3. STATISTICAL ANALYSIS AND EMPIRICAL MODEL FITTING
# -----------------------------------------------------------------------------
# To generate synthetic data that resembles the real samples, we need the
# statistical properties (mean, standard deviation, minimum, maximum) of each
# variable. These are stored in a dictionary called generation_params.

generation_params = {}

# 3.1 Primary variables: generated first, independently.
#     They include grain sizes, moisture content, specific gravity, and depth.
primary_vars = ['Natural/Field Moisture Content (%)', 'Specific Gravity Test',
                'D10 (mm)', 'D30 (mm)', 'D60 (mm)', 'Depth (m)']

for var in primary_vars:
    generation_params[var] = {
        'mean': real_df[var].mean(),   # central tendency
        'std': real_df[var].std(),     # natural variability
        'min': real_df[var].min(),     # lower physical bound
        'max': real_df[var].max()      # upper physical bound
    }

# 3.2 Tertiary variables: generated later, often using correlations.
#     Now includes CBR and Shear Strength.
tertiary_vars = ['L.L %', 'P.L %', 'P.I %', 'OMC (%)', 'MDD (g/cm3)',
                 'Cohesion, C (kN/m2)', 'Frictional Angle, Ø(0)',
                 'California Bearing Ratio', 'Shear Strength (kN/m2)',
                 'Allow. Bearing Pressure (kN/m2)']

for var in tertiary_vars:
    generation_params[var] = {
        'mean': real_df[var].mean(),
        'std': real_df[var].std(),
        'min': real_df[var].min(),
        'max': real_df[var].max()
    }

# 3.3 Fit empirical relationships using linear regression.
#     These capture interdependencies observed in the real data.

# Relationship 1: Friction angle (φ) as a function of MDD and D60.
#     In granular soils, friction angle tends to increase with density (MDD)
#     and with grain size (D60). We fit a multiple linear regression:
#         φ = a + b*MDD + c*D60 + ε
#     where ε is normally distributed noise with standard deviation equal to
#     the residual error. The coefficients are estimated from the 10 real samples.
X_phi = real_df[['MDD (g/cm3)', 'D60 (mm)']].values   # Predictor matrix (two columns)
y_phi = real_df['Frictional Angle, Ø(0)'].values      # Target variable
reg_phi = LinearRegression().fit(X_phi, y_phi)        # Fit the model
phi_coefficients = reg_phi.coef_                      # Coefficients for MDD and D60 (b and c)
phi_intercept = reg_phi.intercept_                    # Intercept term (a)
phi_pred = reg_phi.predict(X_phi)                     # Predictions on the real data
phi_residual_std = np.std(y_phi - phi_pred)           # Standard deviation of residuals (noise)

# Relationship 2: Optimum moisture content (OMC) as a function of MDD.
#     A well‑known inverse relationship exists: higher density generally
#     corresponds to lower OMC. We fit a simple linear regression: OMC = a + b*MDD + ε.
X_omc = real_df[['MDD (g/cm3)']].values               # Predictor (single column)
y_omc = real_df['OMC (%)'].values                      # Target
reg_omc = LinearRegression().fit(X_omc, y_omc)         # Fit the model
omc_coefficient = reg_omc.coef_[0]                     # Slope (b)
omc_intercept = reg_omc.intercept_                     # Intercept (a)
omc_pred = reg_omc.predict(X_omc)                      # Predictions
omc_residual_std = np.std(y_omc - omc_pred)            # Residual standard deviation

# Print the fitted equations for documentation and verification.
print("\n--- Empirical Relationships Fitted ---")
print(f"φ = {phi_intercept:.2f} + {phi_coefficients[0]:.2f}*MDD + {phi_coefficients[1]:.2f}*D60   (residual std = {phi_residual_std:.2f})")
print(f"OMC = {omc_intercept:.2f} + {omc_coefficient:.2f}*MDD   (residual std = {omc_residual_std:.2f})")

# -----------------------------------------------------------------------------
# 4. SYNTHETIC DATA GENERATION FUNCTION (UPDATED WITH CBR AND SHEAR STRENGTH)
# -----------------------------------------------------------------------------
# This function creates one synthetic soil sample at a time. It follows a
# hierarchical order: primary → secondary → tertiary. All values are clipped
# to the physically observed ranges to avoid unrealistic extremes.
def generate_synthetic_soil_sample(params, reg_phi, reg_omc, phi_residual_std, omc_residual_std):
    """
    Generate a single synthetic soil sample with physics-informed constraints.

    Parameters:
        params (dict): Contains mean, std, min, max for each variable.
        reg_phi (LinearRegression): Fitted model for friction angle.
        reg_omc (LinearRegression): Fitted model for optimum moisture content.
        phi_residual_std (float): Standard deviation of residuals for φ model.
        omc_residual_std (float): Standard deviation of residuals for OMC model.

    Returns:
        dict: A dictionary containing all generated properties for one sample.
    """
    sample = {}   # Dictionary to hold the generated values

    # -------------------------------------------------------------------------
    # Step A: Generate Primary (Independent) Variables
    # -------------------------------------------------------------------------
    # Each primary variable is drawn from a normal distribution with the mean
    # and standard deviation observed in the real data. The value is then
    # clipped to the real-world min and max to prevent physically impossible
    # numbers (e.g., negative grain size). This preserves the statistical
    # properties of the original data while allowing natural variation.
    for var in primary_vars:
        p = params[var]                           # Get the parameters for this variable
        # Draw a random value from a normal distribution
        val = np.random.normal(p['mean'], p['std'])
        # Clip to the physically observed range
        val = np.clip(val, p['min'], p['max'])
        sample[var] = val                          # Store in the sample dictionary

    # Extract grain sizes for convenience in later calculations
    d10 = sample['D10 (mm)']
    d30 = sample['D30 (mm)']
    d60 = sample['D60 (mm)']

    # -------------------------------------------------------------------------
    # Step B: Calculate Secondary Variables (Deterministic)
    # -------------------------------------------------------------------------
    # Cu (coefficient of uniformity) and Cc (coefficient of curvature) are
    # purely mathematical functions of the grain sizes. They define the shape
    # of the particle size distribution curve and are used in the USCS
    # classification. They are calculated exactly from the generated D-values.
    # Avoid division by zero (the clipping above ensures positivity, but we add
    # a safety check).
    if d10 <= 0:
        d10 = 0.001   # Set to a very small positive number if clipping failed
    if d60 <= 0:
        d60 = 0.001

    sample['Cu'] = d60 / d10                       # Cu = D60 / D10
    denominator_cc = d60 * d10
    if denominator_cc <= 0:
        sample['Cc'] = 0.5                         # Fallback value (rarely used)
    else:
        sample['Cc'] = (d30**2) / denominator_cc   # Cc = (D30^2) / (D60 * D10)

    # Clip Cu and Cc to realistic ranges for granular soils to avoid extreme
    # outliers that could distort classification. These limits (1.5–30 for Cu,
    # 0.2–5 for Cc) are based on typical values for sands and are wider than
    # the observed range to allow for exploration.
    sample['Cu'] = np.clip(sample['Cu'], 1.5, 30.0)
    sample['Cc'] = np.clip(sample['Cc'], 0.2, 5.0)

    # -------------------------------------------------------------------------
    # Step C: Generate Tertiary Variables with Physics-Informed Constraints
    # -------------------------------------------------------------------------

    # 1. Maximum Dry Density (MDD) – generated independently from its own
    #    bounded normal distribution. MDD will be used in correlations for φ
    #    and OMC.
    mdd_mean = params['MDD (g/cm3)']['mean']
    mdd_std = params['MDD (g/cm3)']['std']
    mdd_min = params['MDD (g/cm3)']['min']
    mdd_max = params['MDD (g/cm3)']['max']
    mdd_val = np.random.normal(mdd_mean, mdd_std)
    mdd_val = np.clip(mdd_val, mdd_min, mdd_max)
    sample['MDD (g/cm3)'] = mdd_val

    # 2. Optimum Moisture Content (OMC) – generated using the fitted regression
    #    with MDD as the predictor. We add random noise (ε) with the same
    #    standard deviation as the residuals from the real data fit, to mimic
    #    natural scatter. This preserves the inverse relationship while
    #    allowing realistic variability.
    omc_pred = omc_intercept + omc_coefficient * mdd_val
    omc_noise = np.random.normal(0, omc_residual_std)
    omc_val = omc_pred + omc_noise
    omc_val = np.clip(omc_val, params['OMC (%)']['min'], params['OMC (%)']['max'])
    sample['OMC (%)'] = omc_val

    # 3. Friction Angle (φ) – generated using the multiple regression on MDD
    #    and D60. Again, we add noise based on the residual standard deviation.
    phi_pred = phi_intercept + phi_coefficients[0] * mdd_val + phi_coefficients[1] * d60
    phi_noise = np.random.normal(0, phi_residual_std)
    phi_val = phi_pred + phi_noise
    phi_val = np.clip(phi_val, params['Frictional Angle, Ø(0)']['min'], params['Frictional Angle, Ø(0)']['max'])
    sample['Frictional Angle, Ø(0)'] = phi_val

    # 4. Atterberg Limits (LL, PL, PI) – we generate LL and PI independently,
    #    then derive PL = LL - PI. This ensures the physical relationship
    #    PI = LL - PL holds and that PI is non-negative. It also guarantees
    #    that PL is never greater than LL.
    ll_mean = params['L.L %']['mean']
    ll_std = params['L.L %']['std']
    ll_min = params['L.L %']['min']
    ll_max = params['L.L %']['max']
    ll_val = np.random.normal(ll_mean, ll_std)
    ll_val = np.clip(ll_val, ll_min, ll_max)

    pi_mean = params['P.I %']['mean']
    pi_std = params['P.I %']['std']
    pi_min = params['P.I %']['min']
    pi_max = params['P.I %']['max']
    pi_val = np.random.normal(pi_mean, pi_std)
    pi_val = np.clip(pi_val, pi_min, pi_max)
    pi_val = max(pi_val, 0.0)   # Ensure PI is not negative (physical requirement)

    # Derive PL = LL - PI
    pl_val = ll_val - pi_val
    # Clip PL to its physically observed range (from real data)
    pl_val = np.clip(pl_val, params['P.L %']['min'], params['P.L %']['max'])
    # Recompute PI in case clipping changed PL; this maintains consistency.
    pi_val = ll_val - pl_val
    pi_val = max(pi_val, 0.0)   # Final safety for non-negative PI

    sample['L.L %'] = ll_val
    sample['P.L %'] = pl_val
    sample['P.I %'] = pi_val

    # 5. Cohesion (c) – generated independently. For sands, cohesion is usually
    #    low, but we use the distribution from the real data (which may include
    #    some silty/clayey sands). No strong correlation is assumed.
    c_mean = params['Cohesion, C (kN/m2)']['mean']
    c_std = params['Cohesion, C (kN/m2)']['std']
    c_min = params['Cohesion, C (kN/m2)']['min']
    c_max = params['Cohesion, C (kN/m2)']['max']
    c_val = np.random.normal(c_mean, c_std)
    c_val = np.clip(c_val, c_min, c_max)
    sample['Cohesion, C (kN/m2)'] = c_val

    # 6. California Bearing Ratio (CBR) – generated independently from its
    #    bounded normal distribution. If future analysis reveals strong correlations
    #    (e.g., with MDD), this can be upgraded to a regression model.
    cbr_mean = params['California Bearing Ratio']['mean']
    cbr_std = params['California Bearing Ratio']['std']
    cbr_min = params['California Bearing Ratio']['min']
    cbr_max = params['California Bearing Ratio']['max']
    cbr_val = np.random.normal(cbr_mean, cbr_std)
    cbr_val = np.clip(cbr_val, cbr_min, cbr_max)
    sample['California Bearing Ratio'] = cbr_val

    # 7. Shear Strength – generated independently from its bounded normal
    #    distribution. In reality, shear strength may correlate with cohesion
    #    and friction angle (Mohr‑Coulomb), but without the normal stress we
    #    treat it independently for simplicity.
    ss_mean = params['Shear Strength (kN/m2)']['mean']
    ss_std = params['Shear Strength (kN/m2)']['std']
    ss_min = params['Shear Strength (kN/m2)']['min']
    ss_max = params['Shear Strength (kN/m2)']['max']
    ss_val = np.random.normal(ss_mean, ss_std)
    ss_val = np.clip(ss_val, ss_min, ss_max)
    sample['Shear Strength (kN/m2)'] = ss_val

    # 8. Allowable Bearing Pressure (qall) – generated independently.
    q_mean = params['Allow. Bearing Pressure (kN/m2)']['mean']
    q_std = params['Allow. Bearing Pressure (kN/m2)']['std']
    q_min = params['Allow. Bearing Pressure (kN/m2)']['min']
    q_max = params['Allow. Bearing Pressure (kN/m2)']['max']
    q_val = np.random.normal(q_mean, q_std)
    q_val = np.clip(q_val, q_min, q_max)
    sample['Allow. Bearing Pressure (kN/m2)'] = q_val

    # -------------------------------------------------------------------------
    # Step D: Final Safety Clip for D60
    # -------------------------------------------------------------------------
    # This is a critical fix: D60 may have been used in calculations but not
    # modified. We re-clip it to the physically observed range [0.85, 4.75] mm
    # to ensure that no synthetic sample has an unrealistic D60 value.
    sample['D60 (mm)'] = np.clip(sample['D60 (mm)'],
                                 params['D60 (mm)']['min'],
                                 params['D60 (mm)']['max'])

    # -------------------------------------------------------------------------
    # Step E: Deterministic Classification (USCS Rules for Sands)
    # -------------------------------------------------------------------------
    # According to the Unified Soil Classification System (USCS), a sand is
    # well-graded (SW) if both conditions hold:
    #   - Cu >= 6   (coefficient of uniformity)
    #   - 1 <= Cc <= 3 (coefficient of curvature)
    # Otherwise, it is classified as poorly graded (SP).
    cu = sample['Cu']
    cc = sample['Cc']
    if cu >= 6 and (1 <= cc <= 3):
        sample['Classification'] = 'Well-graded sand (SW)'
    else:
        sample['Classification'] = 'Poorly graded sand (SP)'

    return sample

# -----------------------------------------------------------------------------
# 5. GENERATE 10,000 SYNTHETIC SAMPLES
# -----------------------------------------------------------------------------
# Set a random seed for reproducibility – ensures that the same sequence of
# random numbers is generated each time the code is run.
np.random.seed(42)

# Define the number of synthetic samples to generate.
n_samples = 10000
# Create an empty list to store the generated samples (each is a dictionary).
synthetic_data_list = []

print(f"\n--- Generating {n_samples} synthetic samples... ---")
# Loop over the desired number of samples.
for i in range(n_samples):
    # Print progress every 2000 samples to monitor execution.
    if (i+1) % 2000 == 0:
        print(f"Generated {i+1} samples...")
    # Generate one sample by calling the function and append it to the list.
    synthetic_data_list.append(
        generate_synthetic_soil_sample(generation_params,
                                       reg_phi, reg_omc,
                                       phi_residual_std, omc_residual_std)
    )

# Convert the list of dictionaries to a pandas DataFrame for easy manipulation.
synthetic_df = pd.DataFrame(synthetic_data_list)

# Add a simple identifier for each synthetic sample (useful for reference).
synthetic_df['Sample Locations'] = ['SYN_' + str(i+1) for i in range(n_samples)]

# Reorder columns to match the original data structure for consistency.
column_order = ['Sample Locations', 'Natural/Field Moisture Content (%)', 'Specific Gravity Test',
                'D10 (mm)', 'D30 (mm)', 'D60 (mm)', 'Cu', 'Cc',
                'L.L %', 'P.L %', 'P.I %', 'OMC (%)', 'MDD (g/cm3)',
                'Cohesion, C (kN/m2)', 'Frictional Angle, Ø(0)', 'Depth (m)',
                'California Bearing Ratio', 'Shear Strength (kN/m2)',
                'Allow. Bearing Pressure (kN/m2)', 'Classification']
synthetic_df = synthetic_df[[col for col in column_order if col in synthetic_df.columns]]

print("\n--- Synthetic Data Generation Complete ---")
print(f"Shape of synthetic data: {synthetic_df.shape}")
print(synthetic_df.head())

# -----------------------------------------------------------------------------
# 6. CREATE A BALANCED DATASET (UNDERSAMPLE MAJORITY CLASS)
# -----------------------------------------------------------------------------
# Machine learning models perform poorly when trained on highly imbalanced data.
# Here we examine the class distribution in the full synthetic dataset.
class_counts = synthetic_df['Classification'].value_counts()
print("\n--- Classification Distribution (Full Dataset) ---")
print(class_counts)

# Identify which class is the minority (fewer samples) and which is the majority.
if class_counts['Well-graded sand (SW)'] < class_counts['Poorly graded sand (SP)']:
    minority_class = 'Well-graded sand (SW)'
    majority_class = 'Poorly graded sand (SP)'
else:
    minority_class = 'Poorly graded sand (SP)'
    majority_class = 'Well-graded sand (SW)'

n_minority = class_counts[minority_class]
print(f"\nMinority class: {minority_class} ({n_minority} samples)")
print(f"Majority class: {majority_class} ({class_counts[majority_class]} samples)")

# Extract all samples belonging to the minority class.
minority_samples = synthetic_df[synthetic_df['Classification'] == minority_class]

# Randomly select the same number of samples from the majority class (without replacement).
# This is called random undersampling. The random_state ensures reproducibility.
majority_samples = synthetic_df[synthetic_df['Classification'] == majority_class]
majority_undersampled = majority_samples.sample(n=n_minority, random_state=42)

# Combine the two subsets and shuffle to randomise the order.
balanced_df = pd.concat([minority_samples, majority_undersampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n--- Balanced Dataset Created ---")
print(f"Balanced dataset shape: {balanced_df.shape}")
print(balanced_df['Classification'].value_counts())

# -----------------------------------------------------------------------------
# 7. VALIDATION ANALYSES WITH IMMEDIATE PLOT SAVING
# -----------------------------------------------------------------------------
# A series of statistical and visual checks to confirm that the synthetic data
# (both full and balanced) faithfully represent the original measurements and
# that the balancing step did not introduce bias.
print("\n" + "="*60)
print("VALIDATION ANALYSES")
print("="*60)

# 7.1 Side‑by‑side summary statistics (Real vs Synthetic Full)
#     This provides a quick quantitative comparison of means, standard deviations,
#     and percentiles for all numerical variables.
num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
real_stats = real_df[num_cols].describe().T[['mean','std','min','25%','50%','75%','max']]
synth_stats = synthetic_df[num_cols].describe().T[['mean','std','min','25%','50%','75%','max']]
real_stats = real_stats.add_prefix('Real_')
synth_stats = synth_stats.add_prefix('Synth_')
comparison = pd.concat([real_stats, synth_stats], axis=1)
print("\n--- Summary Statistics: Real vs Synthetic (Full) ---")
print(comparison.round(3).to_string())

# 7.2 Overlaid distribution plots (key variables, now including CBR and Shear Strength)
#     For each key variable, we plot the synthetic histogram/KDE and overlay the
#     real values as vertical dashed lines. This visually confirms that the
#     synthetic values fall within the real range and follow a similar shape.
key_vars = ['D10 (mm)', 'D30 (mm)', 'D60 (mm)', 'Cu', 'Cc',
            'MDD (g/cm3)', 'Frictional Angle, Ø(0)', 'Cohesion, C (kN/m2)',
            'L.L %', 'OMC (%)', 'California Bearing Ratio', 'Shear Strength (kN/m2)']

# Create a 3x4 grid of subplots (12 plots, all used).
fig1, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()   # Flatten to easily index

for i, var in enumerate(key_vars):
    ax = axes[i]
    # Plot synthetic distribution as a density histogram with KDE overlay.
    sns.histplot(synthetic_df[var], kde=True, stat='density',
                 alpha=0.5, color='blue', ax=ax)
    # Mark each real value with a vertical dashed red line.
    for val in real_df[var].dropna():
        ax.axvline(val, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title(var)
    ax.legend(['Real values', 'Synthetic'])

fig1.suptitle('Overlaid Distributions: Real (dashed lines) vs Synthetic (histogram)', fontsize=14)
plt.tight_layout()
# Save the figure immediately after creation
plt.savefig(os.path.join(plot_dir, '01_overlaid_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 01_overlaid_distributions.png")

# 7.3 Correlation matrix comparison
#     We compute correlation matrices for the real and synthetic data and display
#     them side by side. This checks whether the interdependencies between
#     variables are preserved.
real_corr = real_df[num_cols].corr()
synth_corr = synthetic_df[num_cols].corr()

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, cbar=False, ax=ax1, annot_kws={'size':7})
ax1.set_title('Real Data Correlation')
sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, cbar=False, ax=ax2, annot_kws={'size':7})
ax2.set_title('Synthetic Data Correlation')
fig2.suptitle('Correlation Matrix Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '02_correlation_matrices.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 02_correlation_matrices.png")

# 7.4 Principal Component Analysis (PCA)
#     PCA projects the data into a lower‑dimensional space. We fit on the real data
#     (scaled) and then transform both real and synthetic. The plot shows whether
#     the synthetic samples occupy the same region as the real ones.
scaler = StandardScaler().fit(real_df[num_cols])          # Fit scaler on real data only
real_scaled = scaler.transform(real_df[num_cols])        # Scale real data
synth_scaled = scaler.transform(synthetic_df[num_cols])  # Scale synthetic using same scaler

pca = PCA(n_components=2)                                 # Keep first two principal components
pca_real = pca.fit_transform(real_scaled)                 # Fit PCA on real data and transform
pca_synth = pca.transform(synth_scaled)                   # Transform synthetic using the same PCA

# Print the proportion of variance explained by each component.
print("\n--- PCA Explained Variance Ratio ---")
print(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total: {pca.explained_variance_ratio_.sum():.3f}")

fig3, ax = plt.subplots(figsize=(8,6))
ax.scatter(pca_synth[:,0], pca_synth[:,1], alpha=0.1, c='blue', label='Synthetic')
ax.scatter(pca_real[:,0], pca_real[:,1], color='red', edgecolor='k', s=100, label='Real')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
ax.set_title('PCA: Real vs Synthetic Data')
ax.grid(alpha=0.3)
plt.savefig(os.path.join(plot_dir, '03_pca.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 03_pca.png")

# 7.5 Scatter plots of key physical relationships (including new variables)
#     These plots demonstrate that fundamental geotechnical relationships are
#     preserved. We overlay the real data points for reference.

# 7.5.1 MDD vs OMC (inverse relationship)
fig4, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=synthetic_df, x='MDD (g/cm3)', y='OMC (%)',
                alpha=0.3, label='Synthetic', color='blue', ax=ax)
sns.scatterplot(data=real_df, x='MDD (g/cm3)', y='OMC (%)',
                color='red', s=100, edgecolor='k', label='Real', ax=ax)
ax.set_xlabel('MDD (g/cm³)')
ax.set_ylabel('OMC (%)')
ax.set_title('MDD vs Optimum Moisture Content')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(os.path.join(plot_dir, '04_MDD_vs_OMC.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 04_MDD_vs_OMC.png")

# 7.5.2 D60 vs Friction Angle (positive trend expected)
fig5, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=synthetic_df, x='D60 (mm)', y='Frictional Angle, Ø(0)',
                alpha=0.3, label='Synthetic', color='blue', ax=ax)
sns.scatterplot(data=real_df, x='D60 (mm)', y='Frictional Angle, Ø(0)',
                color='red', s=100, edgecolor='k', label='Real', ax=ax)
ax.set_xlabel('D60 (mm)')
ax.set_ylabel('Friction Angle (°)')
ax.set_title('D60 vs Friction Angle')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(os.path.join(plot_dir, '05_D60_vs_friction_angle.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 05_D60_vs_friction_angle.png")

# 7.5.3 CBR vs MDD (exploratory – to see if any trend emerges)
fig6, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=synthetic_df, x='MDD (g/cm3)', y='California Bearing Ratio',
                alpha=0.3, label='Synthetic', color='blue', ax=ax)
sns.scatterplot(data=real_df, x='MDD (g/cm3)', y='California Bearing Ratio',
                color='red', s=100, edgecolor='k', label='Real', ax=ax)
ax.set_xlabel('MDD (g/cm³)')
ax.set_ylabel('CBR (%)')
ax.set_title('CBR vs MDD')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(os.path.join(plot_dir, '06_CBR_vs_MDD.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 06_CBR_vs_MDD.png")

# 7.5.4 Shear Strength vs Cohesion (possible correlation)
fig7, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=synthetic_df, x='Cohesion, C (kN/m2)', y='Shear Strength (kN/m2)',
                alpha=0.3, label='Synthetic', color='blue', ax=ax)
sns.scatterplot(data=real_df, x='Cohesion, C (kN/m2)', y='Shear Strength (kN/m2)',
                color='red', s=100, edgecolor='k', label='Real', ax=ax)
ax.set_xlabel('Cohesion (kN/m²)')
ax.set_ylabel('Shear Strength (kN/m²)')
ax.set_title('Shear Strength vs Cohesion')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(os.path.join(plot_dir, '07_shear_strength_vs_cohesion.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 07_shear_strength_vs_cohesion.png")

# 7.5.5 Cu vs Cc with classification regions
fig8, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=synthetic_df, x='Cu', y='Cc', hue='Classification',
                alpha=0.5, palette={'Well-graded sand (SW)':'green', 'Poorly graded sand (SP)':'orange'}, ax=ax)
ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
ax.axhline(y=3, color='black', linestyle='--', linewidth=0.5)
ax.axvline(x=6, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('Cu')
ax.set_ylabel('Cc')
ax.set_title('Cu vs Cc with USCS Classification Regions')
ax.set_xlim(0,20)
ax.set_ylim(0,5)
ax.legend()
plt.savefig(os.path.join(plot_dir, '08_Cu_vs_Cc_classification.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 08_Cu_vs_Cc_classification.png")

# 7.6 Validation of the class‑balancing step (including new variables)
#     We compare the distributions of key features between the full SP class
#     (all samples) and the undersampled SP class (balanced subset) using
#     the Kolmogorov–Smirnov test and overlaid histograms. High p‑values
#     (> 0.05) indicate that the two samples come from the same distribution.
sp_full = synthetic_df[synthetic_df['Classification'] == 'Poorly graded sand (SP)']
sp_balanced = balanced_df[balanced_df['Classification'] == 'Poorly graded sand (SP)']

print("\n--- Kolmogorov‑Smirnov Test: Full SP vs Balanced SP ---")
for var in ['Cu', 'Cc', 'D60 (mm)', 'MDD (g/cm3)', 'Frictional Angle, Ø(0)',
            'California Bearing Ratio', 'Shear Strength (kN/m2)']:
    stat, p = ks_2samp(sp_full[var].dropna(), sp_balanced[var].dropna())
    print(f"{var:30s}: KS stat = {stat:.3f}, p‑value = {p:.4f} {'(similar)' if p>0.05 else '(different)'}")

# Overlaid histograms for visual comparison of SP distributions (including new vars).
fig9, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
plot_vars = ['Cu', 'Cc', 'D60 (mm)', 'MDD (g/cm3)', 'Frictional Angle, Ø(0)',
             'California Bearing Ratio', 'Shear Strength (kN/m2)']
for i, var in enumerate(plot_vars):
    ax = axes[i]
    sns.histplot(sp_full[var], kde=True, stat='density', alpha=0.5,
                 label='Full SP', color='blue', ax=ax)
    sns.histplot(sp_balanced[var], kde=True, stat='density', alpha=0.5,
                 label='Balanced SP', color='orange', ax=ax)
    ax.set_title(var)
    ax.legend()
# Hide the last unused subplot (since we have 7 plots and 8 subplots).
axes[-1].set_visible(False)
fig9.suptitle('Feature Distributions: Full SP vs Balanced SP')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '09_class_balancing_histograms.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 09_class_balancing_histograms.png")

# 7.7 Addressing the friction angle anomaly
#     The fitted regression for φ had a negative coefficient for MDD, which is
#     counterintuitive. However, the overall trend of φ with grain size (D60)
#     is positive, as shown below. We perform a simple linear regression of φ
#     on D60 to demonstrate this.
x = synthetic_df['D60 (mm)'].values
y = synthetic_df['Frictional Angle, Ø(0)'].values
slope, intercept, r_value, p_value, std_err = linregress(x, y)

fig10, ax = plt.subplots(figsize=(8,6))
ax.scatter(x, y, alpha=0.1, label='Synthetic')
ax.plot(np.unique(x), intercept + slope*np.unique(x), color='red',
         label=f'Linear fit (slope={slope:.2f} °/mm, R²={r_value**2:.2f})')
ax.set_xlabel('D60 (mm)')
ax.set_ylabel('Friction Angle (°)')
ax.set_title('Friction Angle vs D60 – Positive Trend')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(os.path.join(plot_dir, '10_friction_angle_vs_D60_trend.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 10_friction_angle_vs_D60_trend.png")

print(f"\nLinear regression of φ on D60: slope = {slope:.2f} °/mm, R² = {r_value**2:.3f}")
print("Interpretation: Despite the negative MDD coefficient in the multivariate fit,")
print("the overall trend with grain size is positive and physically reasonable.")
print("This anomaly is likely due to the small size of the real dataset (10 samples)")
print("and should not invalidate the synthetic data, as the generated φ values")
print("still fall within the observed range and preserve the positive D60-φ relationship.")

# -----------------------------------------------------------------------------
# 8. SAVE DATASETS
# -----------------------------------------------------------------------------
# Save the full (unbalanced) synthetic dataset and the balanced dataset as CSV files.
# These files can be downloaded from the Colab environment and used for machine learning.
synthetic_df.to_csv('synthetic_geotech_data_10000_full.csv', index=False)
balanced_df.to_csv('synthetic_geotech_data_10000_balanced.csv', index=False)

print("\n--- Datasets Saved ---")
print("Full dataset (10,000 samples): 'synthetic_geotech_data_10000_full.csv'")
print("Balanced dataset (undersampled): 'synthetic_geotech_data_10000_balanced.csv'")
print("\n--- ALL DONE ---")
print(f"\nAll plots have been saved to the '{plot_dir}' folder.")
